"""Qwen Token Plan WebSocket transport and a single Discord voice conversation."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import ssl
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import certifi
from discord.ext import voice_recv
from websockets.asyncio.client import connect

from core.voice_audio import DiscordInput, DiscordOutput

# websockets DEBUG logs include Authorization headers and audio payloads. Keep
# this dedicated transport logger quiet even when the bot uses LOG_LEVEL=DEBUG.
_ws_logger = logging.getLogger('quillbot.voice.websocket')
_ws_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Known provider error codes and honest user-facing translations. Anything
# else keeps the generic hint, but the code is always logged for diagnosis.
_PROVIDER_ERRORS = {
    'response_idle_timeout': (
        'Ficou tempo demais sem nenhuma fala gerar resposta e o Qwen encerrou a sessão '
        'por inatividade. Use /voice entrar para iniciar outra.'
    ),
}


class VoiceError(Exception):
    """A safe, user-facing error; never carries a provider payload or credential."""


class QwenRealtime:
    def __init__(self, *, api_key: str, url: str, model: str, voice: str, instructions: str) -> None:
        self._api_key = api_key
        self._url = url
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._ws = None

    async def open(self) -> None:
        parts = urlsplit(self._url)
        if parts.scheme != 'wss' or not parts.hostname or parts.username or parts.password or parts.fragment:
            raise VoiceError('QWEN_REALTIME_URL deve ser um endereço wss:// válido.')
        if not self._api_key:
            raise VoiceError('Configure TOKEN_PLAN_API_KEY (ou ANTHROPIC_AUTH_TOKEN) e reinicie o bot.')
        query = dict(parse_qsl(parts.query))
        query['model'] = self._model
        url = urlunsplit(parts._replace(query=urlencode(query)))
        try:
            async with asyncio.timeout(20):
                self._ws = await connect(
                    url,
                    additional_headers={'Authorization': f'Bearer {self._api_key}'},
                    ssl=ssl.create_default_context(cafile=certifi.where()),
                    open_timeout=10,
                    close_timeout=5,
                    max_size=4 * 1024 * 1024,
                    max_queue=8,
                    logger=_ws_logger,
                )
                await self._wait_for('session.created')
                await self.send({
                    'type': 'session.update',
                    'session': {
                        'modalities': ['audio', 'text'],
                        'voice': self._voice,
                        'instructions': self._instructions,
                        'input_audio_format': 'pcm',
                        'output_audio_format': 'pcm',
                        'turn_detection': {
                            'type': 'server_vad', 'threshold': 0.5, 'silence_duration_ms': 600,
                        },
                    },
                })
                await self._wait_for('session.updated')
        except BaseException:
            await self.close()
            raise

    async def _wait_for(self, event_type: str) -> None:
        while (await self.receive())['type'] != event_type:
            pass

    async def send(self, event: dict) -> None:
        await self._ws.send(json.dumps(event))

    async def receive(self) -> dict:
        event = json.loads(await self._ws.recv())
        if not isinstance(event, dict) or not isinstance(event.get('type'), str):
            raise VoiceError('O serviço de voz enviou uma resposta inválida.')
        if event['type'] == 'error':
            # Provider messages can echo request data. Log and forward only the
            # short machine code, never the message payload.
            error = event.get('error')
            code = str(error.get('code', ''))[:64] if isinstance(error, dict) else ''
            logger.warning('Voice provider error (code=%s)', code or 'unknown')
            raise VoiceError(_PROVIDER_ERRORS.get(code) or (
                'O Qwen recusou a sessão de voz. Confira a chave, a região, o modelo e a voz configurados.'
            ))
        return event

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None


class VoiceConversation:
    """One Qwen session shared by the humans in one Discord voice channel."""

    def __init__(
        self,
        voice_client: voice_recv.VoiceRecvClient,
        qwen: QwenRealtime,
        *,
        idle_timeout: float,
        max_duration: float,
    ) -> None:
        self.voice_client = voice_client
        self.channel = voice_client.channel
        self.qwen = qwen
        self.input = DiscordInput()
        self.output = DiscordOutput()
        self.idle_timeout = idle_timeout
        self.max_duration = max_duration
        self._started_at = self._last_activity = time.monotonic()
        self._response_id: str | None = None
        self._accept_audio = False
        self._closing = False
        self._close_task: asyncio.Task | None = None

    @property
    def closing(self) -> bool:
        return self._closing

    async def run(self) -> str:
        loop = asyncio.get_running_loop()
        audio_stopped = loop.create_future()

        def stopped(_error: Exception | None) -> None:
            def notify() -> None:
                if not self._closing and not audio_stopped.done():
                    audio_stopped.set_result('A conexão de áudio foi encerrada. Use /voice entrar para reconectar.')
            if not loop.is_closed():
                loop.call_soon_threadsafe(notify)

        tasks = []
        try:
            self.voice_client.play(self.output, after=stopped)
            self.voice_client.listen(self.input, after=stopped)
            tasks = [
                asyncio.create_task(self._send_audio()),
                asyncio.create_task(self._receive_audio()),
                asyncio.create_task(self._watchdog()),
            ]
            _done, pending = await asyncio.wait([*tasks, audio_stopped], return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            # When the provider closes the socket, sender and receiver fail
            # together. Collect every outcome so the receiver's VoiceError (the
            # server's own words) always beats the sender's ConnectionClosed.
            outcomes = await asyncio.gather(*tasks, audio_stopped, return_exceptions=True)
            errors = [o for o in outcomes if isinstance(o, BaseException) and not isinstance(o, asyncio.CancelledError)]
            voice_errors = [o for o in errors if isinstance(o, VoiceError)]
            if voice_errors:
                raise voice_errors[0]
            if errors:
                raise errors[0]
            for outcome in outcomes:
                if isinstance(outcome, str):
                    return outcome
            return None
        finally:
            for task in tasks:
                task.cancel()
            audio_stopped.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.close()

    async def _send_audio(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while True:
            await self.qwen.send({
                'type': 'input_audio_buffer.append',
                'audio': base64.b64encode(self.input.read()).decode('ascii'),
            })
            # Do not accumulate a burst of stale frames if networking stalls.
            deadline = max(deadline + 0.02, loop.time())
            await asyncio.sleep(max(0, deadline - loop.time()))

    async def _receive_audio(self) -> None:
        while True:
            self.handle_event(await self.qwen.receive())

    def handle_event(self, event: dict) -> None:
        kind = event['type']
        if kind == 'input_audio_buffer.speech_started':
            self._last_activity = time.monotonic()
            self._accept_audio = False
            self.output.clear()
            # server_vad cancels generation itself. Ignore late deltas until a
            # new response.created; an explicit cancel can race the new turn.
        elif kind == 'response.created':
            self._response_id = event['response']['id']
            self._accept_audio = True
        elif kind == 'response.audio.delta':
            if self._accept_audio and event.get('response_id') == self._response_id:
                self._last_activity = time.monotonic()
                self.output.append(base64.b64decode(event['delta'], validate=True))
        elif kind == 'response.done' and event['response'].get('id') == self._response_id:
            status = event['response'].get('status')
            self._accept_audio = False
            if status == 'cancelled':
                self.output.clear()
            elif status == 'failed':
                raise VoiceError('O Qwen não conseguiu responder. Use /voice entrar para reconectar.')

    async def _watchdog(self) -> str:
        while True:
            await asyncio.sleep(1)
            self.input.log_stats_if_due()
            now = time.monotonic()
            if not self.voice_client.is_connected():
                return 'A conexão de voz foi encerrada.'
            if now - self._started_at >= self.max_duration:
                return 'A sessão de voz atingiu o limite de duração. Use /voice entrar para iniciar outra.'
            if now - max(self._last_activity, self.input.last_activity) >= self.idle_timeout:
                return 'Saí do canal de voz por inatividade.'

    async def close(self) -> None:
        # Multiple callers (disconnect, /sair, unload) join the same cleanup.
        # Shield it so cancellation during startup or shutdown cannot leak a VC.
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        try:
            self.input.cleanup()
            self.output.cleanup()
            try:
                self.voice_client.stop_listening()
            finally:
                self.voice_client.stop_playing()
        finally:
            try:
                await self.qwen.close()
            finally:
                await self.voice_client.disconnect(force=True)
