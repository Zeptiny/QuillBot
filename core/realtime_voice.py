"""Client for Qwen-Audio realtime voice conversation (Alibaba Cloud Model Studio).

One WebSocket per call. The client streams 16 kHz mono PCM in
(``input_audio_buffer.append``); the server finds the turns itself (VAD or
semantic turn detection), answers with 24 kHz mono PCM
(``response.audio.delta``) plus transcripts of both sides, and everything
else is JSON events named after OpenAI's Realtime API. Reference:
https://www.alibabacloud.com/help/en/model-studio/fun-audiochat-client-events

The endpoint and key are configuration: a Model Studio pay-as-you-go key
(``sk-…``) uses the workspace's realtime URL, a Token Plan key (``sk-sp-…``)
the Token Plan one.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp

logger = logging.getLogger(__name__)


class RealtimeError(RuntimeError):
    """The realtime session couldn't be opened or configured (message is user-facing)."""


def realtime_url(base: str, model: str) -> str:
    """*base* with ``?model=`` set, unless it already names one."""
    parts = urlsplit(base)
    query = parse_qsl(parts.query, keep_blank_values=True)
    if not any(k == 'model' for k, _ in query):
        query.append(('model', model))
    return urlunsplit(parts._replace(query=urlencode(query)))


def error_code(event: dict) -> str:
    err = event.get('error')
    return str(err.get('code') or '')[:64] if isinstance(err, dict) else ''


def error_message(event: dict) -> str:
    err = event.get('error')
    if isinstance(err, dict):
        text = err.get('message') or err.get('code') or err
    else:
        text = err or event
    return str(text)[:300]


class RealtimeVoiceClient:
    """One realtime conversation: :meth:`connect`, stream audio, iterate :meth:`events`."""

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        *,
        http: aiohttp.ClientSession | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.url = realtime_url(url, model)
        self.model = model
        self._api_key = api_key
        self._http = http
        self._owns_http = http is None
        self._timeout = timeout
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._send_lock = asyncio.Lock()
        self.session: dict = {}
        self.audio_bytes_sent = 0

    @property
    def closed(self) -> bool:
        return self._ws is None or self._ws.closed

    @property
    def close_code(self) -> int | None:
        return None if self._ws is None else self._ws.close_code

    async def connect(self, session_config: dict) -> dict:
        """Open the socket and apply *session_config*; returns the session the server confirmed."""
        if self._http is None:
            self._http = aiohttp.ClientSession()
        try:
            self._ws = await asyncio.wait_for(
                self._http.ws_connect(
                    self.url,
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    max_msg_size=0,
                ),
                self._timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            await self.close()
            if e.status in (401, 403):
                raise RealtimeError(f'a chave da API de voz foi recusada ({e.status})') from None
            raise RealtimeError(f'o servidor de voz recusou a conexão ({e.status})') from None
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            await self.close()
            raise RealtimeError(f'não consegui conectar ao servidor de voz ({type(e).__name__})') from None
        try:
            await self._expect('session.created')
            await self.send({'type': 'session.update', 'session': session_config})
            updated = await self._expect('session.updated')
        except BaseException:
            await self.close()
            raise
        self.session = updated.get('session') or {}
        return self.session

    async def _expect(self, event_type: str) -> dict:
        """The next *event_type* event, failing on an ``error`` event, a close or the timeout."""
        assert self._ws is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RealtimeError(f'o servidor de voz não respondeu ({event_type})')
            try:
                msg = await self._ws.receive(timeout=remaining)
            except asyncio.TimeoutError:
                raise RealtimeError(f'o servidor de voz não respondeu ({event_type})') from None
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise RealtimeError(f'o servidor de voz fechou a conexão ({self._ws.close_code})')
            try:
                event = json.loads(msg.data)
            except ValueError:
                continue
            if event.get('type') == 'error':
                # The message can echo what was sent (the instructions, say): it goes
                # to the log; people see the code.
                logger.warning('[voice] realtime session refused: %s', error_message(event))
                raise RealtimeError(f'o servidor de voz recusou a sessão ({error_code(event) or "sem código"})')
            if event.get('type') == event_type:
                return event

    async def events(self) -> AsyncIterator[dict]:
        """Server events until the connection closes."""
        if self._ws is None:
            return
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(msg.data)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    yield event
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.warning('[voice] realtime socket error: %s', self._ws.exception())
                break

    async def send(self, event: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            raise ConnectionError('realtime session is closed')
        async with self._send_lock:
            await ws.send_str(json.dumps(event, ensure_ascii=False))

    async def append_audio(self, pcm16k: bytes) -> None:
        await self.send({'type': 'input_audio_buffer.append', 'audio': base64.b64encode(pcm16k).decode()})
        self.audio_bytes_sent += len(pcm16k)

    async def add_message(self, role: str, text: str) -> None:
        """Add a text item to the conversation (``system`` or ``user``)."""
        await self.send({
            'type': 'conversation.item.create',
            'item': {'type': 'message', 'role': role, 'content': [{'type': 'input_text', 'text': text}]},
        })

    async def delete_item(self, item_id: str) -> None:
        """Take an item out of the conversation, so the model doesn't remember it."""
        await self.send({'type': 'conversation.item.delete', 'item_id': item_id})

    async def create_response(self) -> None:
        await self.send({'type': 'response.create'})

    async def cancel_response(self) -> None:
        await self.send({'type': 'response.cancel'})

    async def close(self) -> None:
        ws = self._ws  # kept, so close_code stays readable
        try:
            if ws is not None and not ws.closed:
                await ws.close()
        except Exception:
            logger.debug('[voice] closing the realtime socket failed', exc_info=True)
        if self._owns_http and self._http is not None and not self._http.closed:
            await self._http.close()
