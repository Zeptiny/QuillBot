"""The Qwen-Audio realtime client, against a local fake of the server."""

import base64
import json

from aiohttp import web
import pytest

from core.realtime_voice import RealtimeError, RealtimeVoiceClient, error_message, realtime_url

from helpers import check


class FakeServer:
    """Speaks just enough of the realtime protocol: session handshake, audio in, one response."""

    def __init__(self, *, reject=None, bad_update=False, silent=False, hang_up=False):
        self.reject = reject
        self.bad_update = bad_update
        self.silent = silent
        self.hang_up = hang_up
        self.headers = {}
        self.query = {}
        self.events = []
        self.audio = bytearray()

    async def handler(self, request):
        self.headers = dict(request.headers)
        self.query = dict(request.query)
        if self.reject:
            return web.Response(status=self.reject)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if self.hang_up:
            await ws.close()
            return ws
        if not self.silent:
            await ws.send_json({'type': 'session.created', 'session': {'id': 'sess_1'}})
        async for msg in ws:
            event = json.loads(msg.data)
            self.events.append(event)
            kind = event['type']
            if kind == 'session.update':
                if self.bad_update:
                    await ws.send_json({'type': 'error', 'error': {'message': 'Voice not supported: nope', 'code': 'invalid_value', 'param': 'voice'}})
                else:
                    await ws.send_json({'type': 'session.updated', 'session': event['session']})
            elif kind == 'input_audio_buffer.append':
                self.audio += base64.b64decode(event['audio'])
            elif kind == 'response.create':
                await ws.send_json({'type': 'response.created', 'response': {'id': 'r1'}})
                await ws.send_str('not json')
                await ws.send_json({'type': 'response.audio.delta', 'delta': base64.b64encode(b'\x01\x00' * 4).decode()})
                await ws.send_json({'type': 'response.done', 'response': {'status': 'completed'}})
                await ws.close()
        return ws


async def serve(fake):
    app = web.Application()
    app.router.add_get('/api-ws/v1/realtime', fake.handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', 0).start()
    return runner, f'ws://127.0.0.1:{runner.addresses[0][1]}/api-ws/v1/realtime'


def test_url_gets_the_model():
    check('appended', realtime_url('wss://h/api-ws/v1/realtime', 'm1') == 'wss://h/api-ws/v1/realtime?model=m1')
    check('kept when given', realtime_url('wss://h/x?model=m0', 'm1') == 'wss://h/x?model=m0')
    check('other params kept', realtime_url('wss://h/x?a=1', 'm1') == 'wss://h/x?a=1&model=m1')


def test_error_messages():
    check('message', error_message({'type': 'error', 'error': {'message': 'bad', 'code': 'x'}}) == 'bad')
    check('code when no message', error_message({'error': {'code': 'quota'}}) == 'quota')
    check('plain string', error_message({'error': 'oops'}) == 'oops')


async def test_full_exchange():
    fake = FakeServer()
    runner, url = await serve(fake)
    try:
        client = RealtimeVoiceClient(url, 'sk-test', 'qwen-audio-3.0-realtime-plus')
        session = await client.connect({'voice': 'longanqian', 'instructions': 'oi'})
        check('bearer key', fake.headers.get('Authorization') == 'Bearer sk-test', fake.headers)
        check('model in the URL', fake.query.get('model') == 'qwen-audio-3.0-realtime-plus', fake.query)
        check('confirmed session returned', session == {'voice': 'longanqian', 'instructions': 'oi'}, session)
        check('open', not client.closed)

        await client.append_audio(b'\x02\x00' * 320)
        await client.add_message('user', '[cumprimente]')
        await client.delete_item('item_9')
        await client.create_response()
        events = [e['type'] async for e in client.events()]
        check('server events, bad frames skipped', events == ['response.created', 'response.audio.delta', 'response.done'], events)
        check('audio arrived', bytes(fake.audio) == b'\x02\x00' * 320 and client.audio_bytes_sent == 640)
        item = next(e for e in fake.events if e['type'] == 'conversation.item.create')['item']
        check('text item', item == {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': '[cumprimente]'}]}, item)
        check('item deleted', {'type': 'conversation.item.delete', 'item_id': 'item_9'} in fake.events)

        await client.close()
        check('closed', client.closed)
        with pytest.raises(ConnectionError):
            await client.cancel_response()
    finally:
        await runner.cleanup()


async def test_refused_key():
    runner, url = await serve(FakeServer(reject=401))
    try:
        client = RealtimeVoiceClient(url, 'sk-bad', 'm')
        with pytest.raises(RealtimeError, match='chave'):
            await client.connect({})
        check('cleaned up', client.closed)
    finally:
        await runner.cleanup()


async def test_rejected_session_settings():
    runner, url = await serve(FakeServer(bad_update=True))
    try:
        client = RealtimeVoiceClient(url, 'k', 'm')
        with pytest.raises(RealtimeError, match=r'\(invalid_value\)') as refused:
            await client.connect({'voice': 'nope'})
        check('people see the code, not what the server echoed', 'nope' not in str(refused.value), str(refused.value))
        check('socket closed after the failure', client.closed)
    finally:
        await runner.cleanup()


async def test_server_that_never_answers_or_hangs_up():
    runner, url = await serve(FakeServer(silent=True))
    try:
        with pytest.raises(RealtimeError, match='não respondeu'):
            await RealtimeVoiceClient(url, 'k', 'm', timeout=0.3).connect({})
    finally:
        await runner.cleanup()
    runner, url = await serve(FakeServer(hang_up=True))
    try:
        with pytest.raises(RealtimeError, match='fechou'):
            await RealtimeVoiceClient(url, 'k', 'm').connect({})
    finally:
        await runner.cleanup()


async def test_nothing_listening():
    with pytest.raises(RealtimeError, match='conectar'):
        await RealtimeVoiceClient('ws://127.0.0.1:1/x', 'k', 'm', timeout=2).connect({})
