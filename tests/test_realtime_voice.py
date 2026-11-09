"""Protocol ordering, barge-in, silence and teardown with an in-memory WebSocket."""

import asyncio
import base64
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core import realtime_voice
from core.realtime_voice import QwenRealtime, VoiceConversation, VoiceError
from core.voice_audio import INPUT_FRAME_BYTES, OUTPUT_FRAME_BYTES


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def send(self, text):
        self.sent.append(json.loads(text))

    async def recv(self):
        event = await self.incoming.get()
        if isinstance(event, Exception):
            raise event
        return json.dumps(event)

    async def close(self):
        self.closed = True


def transport():
    return QwenRealtime(
        api_key='test-secret', url='wss://example.test/realtime?model=old',
        model='qwen-audio-3.0-realtime-plus', voice='longanqian', instructions='Be brief.',
    )


def voice_client():
    return SimpleNamespace(
        channel=SimpleNamespace(id=42), play=Mock(), listen=Mock(), stop_playing=Mock(),
        stop_listening=Mock(), disconnect=AsyncMock(), is_connected=lambda: True,
    )


async def test_handshake_waits_for_configuration_ack_and_uses_tls(monkeypatch):
    socket = Socket()
    factory = AsyncMock(return_value=socket)
    monkeypatch.setattr(realtime_voice, 'connect', factory)
    qwen = transport()
    opening = asyncio.create_task(qwen.open())
    await asyncio.sleep(0)
    assert not opening.done()
    socket.incoming.put_nowait({'type': 'session.created'})
    await asyncio.sleep(0)
    assert not opening.done()
    assert [e['type'] for e in socket.sent] == ['session.update']
    session = socket.sent[0]['session']
    assert session['input_audio_format'] == session['output_audio_format'] == 'pcm'
    assert session['voice'] == 'longanqian'
    assert session['turn_detection'] == {'type': 'server_vad', 'threshold': 0.5, 'silence_duration_ms': 600}
    assert session['modalities'] == ['audio', 'text']
    socket.incoming.put_nowait({'type': 'session.updated'})
    await opening
    args, kwargs = factory.call_args
    assert args == ('wss://example.test/realtime?model=qwen-audio-3.0-realtime-plus',)
    assert kwargs['additional_headers'] == {'Authorization': 'Bearer test-secret'}
    assert kwargs['ssl'].check_hostname
    assert not kwargs['logger'].isEnabledFor(10)
    await qwen.close()
    assert socket.closed


async def test_error_during_handshake_closes_socket_without_exposing_payload(monkeypatch):
    socket = Socket()
    socket.incoming.put_nowait({'type': 'session.created'})
    socket.incoming.put_nowait({'type': 'error', 'error': {'message': 'test-secret raw-transcript'}})
    monkeypatch.setattr(realtime_voice, 'connect', AsyncMock(return_value=socket))
    with pytest.raises(VoiceError) as error:
        await transport().open()
    assert 'test-secret' not in str(error.value)
    assert 'raw-transcript' not in str(error.value)
    assert socket.closed


async def test_cancelled_handshake_closes_socket(monkeypatch):
    socket = Socket()
    monkeypatch.setattr(realtime_voice, 'connect', AsyncMock(return_value=socket))
    task = asyncio.create_task(transport().open())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert socket.closed


@pytest.mark.parametrize('url,key', [('ws://example.test', 'key'), ('wss://example.test', '')])
async def test_invalid_configuration_never_connects(monkeypatch, url, key):
    factory = AsyncMock()
    monkeypatch.setattr(realtime_voice, 'connect', factory)
    qwen = QwenRealtime(api_key=key, url=url, model='model', voice='voice', instructions='')
    with pytest.raises(VoiceError):
        await qwen.open()
    factory.assert_not_called()


def test_barge_in_drops_buffered_and_late_audio_but_accepts_next_response():
    conversation = VoiceConversation(voice_client(), transport(), idle_timeout=300, max_duration=1800)
    audio = base64.b64encode(b'\xff\x0f' * 1000).decode()

    def delta(response_id):
        return {'type': 'response.audio.delta', 'response_id': response_id, 'delta': audio}

    conversation.handle_event({'type': 'response.created', 'response': {'id': 'old'}})
    conversation.handle_event(delta('old'))
    assert conversation.output.read() != bytes(OUTPUT_FRAME_BYTES)
    conversation.handle_event({'type': 'input_audio_buffer.speech_started'})
    conversation.handle_event(delta('old'))
    assert conversation.output.read() == bytes(OUTPUT_FRAME_BYTES)
    conversation.handle_event({'type': 'response.created', 'response': {'id': 'new'}})
    conversation.handle_event(delta('old'))
    conversation.handle_event({'type': 'response.done', 'response': {'id': 'old', 'status': 'cancelled'}})
    assert conversation.output.read() == bytes(OUTPUT_FRAME_BYTES)
    conversation.handle_event(delta('new'))
    assert conversation.output.read() != bytes(OUTPUT_FRAME_BYTES)
    conversation.handle_event({'type': 'response.done', 'response': {'id': 'new', 'status': 'cancelled'}})
    assert conversation.output.read() == bytes(OUTPUT_FRAME_BYTES)


async def test_silence_is_sent_and_disconnect_stops_both_audio_directions():
    socket = Socket()
    qwen = transport()
    qwen._ws = socket
    client = voice_client()
    conversation = VoiceConversation(client, qwen, idle_timeout=300, max_duration=1800)
    task = asyncio.create_task(conversation.run())
    for _ in range(10):
        await asyncio.sleep(0)
        if socket.sent:
            break
    assert socket.sent[0]['type'] == 'input_audio_buffer.append'
    assert base64.b64decode(socket.sent[0]['audio']) == bytes(INPUT_FRAME_BYTES)
    socket.incoming.put_nowait(ConnectionError('disconnected'))
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(task, 1)
    assert socket.closed
    client.stop_playing.assert_called_once()
    client.stop_listening.assert_called_once()
    client.disconnect.assert_awaited_once_with(force=True)
    assert conversation.output.read() == b''
    await conversation.close()
    client.disconnect.assert_awaited_once()


async def test_cancellation_cleans_up_without_leaving_sender_running():
    socket = Socket()
    qwen = transport()
    qwen._ws = socket
    client = voice_client()
    conversation = VoiceConversation(client, qwen, idle_timeout=300, max_duration=1800)
    task = asyncio.create_task(conversation.run())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert socket.closed
    client.disconnect.assert_awaited_once()
    sent = len(socket.sent)
    await asyncio.sleep(0.03)
    assert len(socket.sent) == sent


@pytest.mark.parametrize('limit', ['idle', 'duration'])
async def test_watchdog_closes_idle_and_expired_sessions(limit):
    socket = Socket()
    qwen = transport()
    qwen._ws = socket
    client = voice_client()
    conversation = VoiceConversation(
        client, qwen, idle_timeout=0 if limit == 'idle' else 300, max_duration=0 if limit == 'duration' else 1800,
    )
    reason = await asyncio.wait_for(conversation.run(), 2)
    assert ('inatividade' if limit == 'idle' else 'duração') in reason
    client.disconnect.assert_awaited_once()
    assert socket.closed


async def test_continuous_incoming_audio_is_not_idle(monkeypatch):
    conversation = VoiceConversation(voice_client(), transport(), idle_timeout=2, max_duration=1800)
    conversation._last_activity = time.monotonic() - 100
    # A long utterance only gets one server speech_started event; new input
    # packets still count as activity until the speaker finishes.
    monkeypatch.setattr(type(conversation.input), 'last_activity', property(lambda self: time.monotonic()))
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(conversation._watchdog(), 1.1)
    await conversation.close()


async def test_cleanup_attempts_every_resource_even_if_reader_stop_fails():
    socket = Socket()
    qwen = transport()
    qwen._ws = socket
    client = voice_client()
    client.stop_listening.side_effect = RuntimeError('reader failed')
    conversation = VoiceConversation(client, qwen, idle_timeout=300, max_duration=1800)
    with pytest.raises(RuntimeError):
        await conversation.close()
    assert socket.closed
    client.stop_playing.assert_called_once()
    client.disconnect.assert_awaited_once()
