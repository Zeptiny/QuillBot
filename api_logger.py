"""Transport-level API request logging.

Patches ``aiohttp.ClientSession._request`` and ``httpx.AsyncClient.send`` so
every outbound HTTP request made by the bot — including discord.py's REST
client, the OpenAI SDK (LLM/embeddings/rerank) and tavily-python — is logged
as one JSON line to a rotating file. An inbound listener logs every Discord
interaction (slash commands, buttons, modals, autocomplete).

Request/response bodies are captured only for services listed in
``API_REQUEST_LOG_BODY`` (httpx-based services: openai, tavily). Headers and
authorization credentials are never logged; sensitive query-string values
are redacted.
"""

import datetime
import json
import logging
import logging.handlers
import os
import time
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp
import discord
import httpx
from discord.ext import commands

from config import (
    API_REQUEST_LOG_BACKUPS,
    API_REQUEST_LOG_BODY,
    API_REQUEST_LOG_BODY_MAX_CHARS,
    API_REQUEST_LOG_CONSOLE,
    API_REQUEST_LOG_DISCORD,
    API_REQUEST_LOG_ENABLED,
    API_REQUEST_LOG_MAX_BYTES,
    API_REQUEST_LOG_PATH,
    OPENAI_BASE_URL,
)

_logger = logging.getLogger('quillbot.api')
_logger.propagate = False

_ORIG_AIOHTTP_REQUEST = aiohttp.ClientSession._request
_ORIG_HTTPX_SEND = httpx.AsyncClient.send

_SENSITIVE_PARAMS: Final[tuple[str, ...]] = (
    'key', 'token', 'secret', 'password', 'signature', 'session', 'auth', 'code',
)


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or '').lower()
    except Exception:
        return ''


_LLM_HOST: Final[str] = _host(OPENAI_BASE_URL)


def _service(url: str) -> str:
    h = _host(url)
    if not h:
        return 'other'
    if h == 'discord.com' or h.endswith('.discord.com') or h == 'discordapp.com' or h.endswith('.discordapp.com'):
        return 'discord'
    if _LLM_HOST and (h == _LLM_HOST or h.endswith('.' + _LLM_HOST)):
        return 'openai'
    if 'tavily' in h:
        return 'tavily'
    if h == 'github.com' or h.endswith('.github.com') or h.endswith('.githubusercontent.com'):
        return 'github'
    return 'other'


def _redact_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if not parts.query:
            return url
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        if not any(any(s in k.lower() for s in _SENSITIVE_PARAMS) for k, _ in pairs):
            return url
        redacted = [
            (k, 'REDACTED' if any(s in k.lower() for s in _SENSITIVE_PARAMS) else v)
            for k, v in pairs
        ]
        return urlunsplit(parts._replace(query=urlencode(redacted)))
    except Exception:
        return url


def _body_value(raw: bytes | str | None) -> Any:
    """Normalize a wire body for the log: JSON object when it fits, else a truncated string.

    Short-circuits on oversized bodies *before* parsing so a ~1 MB vision
    request (base64 data URIs) never pays a full ``json.loads``/``dumps``
    on the event loop.
    """
    if raw is None:
        return None
    text = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
    text = text.strip()
    if not text:
        return None
    if len(text) > API_REQUEST_LOG_BODY_MAX_CHARS:
        return text[:API_REQUEST_LOG_BODY_MAX_CHARS] + f'…[truncated {len(text) - API_REQUEST_LOG_BODY_MAX_CHARS} chars]'
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _should_capture(service: str) -> bool:
    return '*' in API_REQUEST_LOG_BODY or service in API_REQUEST_LOG_BODY


def _emit(record: dict[str, Any]) -> None:
    try:
        if not API_REQUEST_LOG_ENABLED:
            return
        if (
            record.get('dir') == 'outbound'
            and record.get('service') == 'discord'
            and not API_REQUEST_LOG_DISCORD
        ):
            return
        record.setdefault('ts', datetime.datetime.now(datetime.timezone.utc).isoformat())
        _logger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        pass  # logging must never break the request path


def _err_str(exc: BaseException) -> str:
    # ClientResponseError/InvalidURL embed the full URL (query string included)
    # in str(exc), so the redactor must run here too.
    return _redact_url(f'{type(exc).__name__}: {exc}')[:300]


async def _patched_aiohttp_request(
    self: aiohttp.ClientSession, method: str, str_or_url: Any, **kwargs: Any
) -> aiohttp.ClientResponse:
    url = str_or_url if isinstance(str_or_url, str) else str(str_or_url)
    start = time.monotonic()
    try:
        response = await _ORIG_AIOHTTP_REQUEST(self, method, str_or_url, **kwargs)
    except Exception as exc:
        _emit({
            'dir': 'outbound',
            'service': _service(url),
            'method': str(method),
            'url': _redact_url(url),
            'duration_ms': round((time.monotonic() - start) * 1000, 1),
            'error': _err_str(exc),
        })
        raise
    _emit({
        'dir': 'outbound',
        'service': _service(str(response.url)),
        'method': str(method),
        'url': _redact_url(str(response.url)),
        'status': response.status,
        'duration_ms': round((time.monotonic() - start) * 1000, 1),
    })
    return response


async def _patched_httpx_send(
    self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
) -> httpx.Response:
    url = str(request.url)
    service = _service(url)
    capture = _should_capture(service) and not kwargs.get('stream')
    start = time.monotonic()
    raw_request_body: bytes | None = None
    if capture:
        try:
            raw_request_body = request.read()
        except Exception:
            raw_request_body = None
    try:
        response = await _ORIG_HTTPX_SEND(self, request, **kwargs)
    except Exception as exc:
        _emit({
            'dir': 'outbound',
            'service': service,
            'method': request.method,
            'url': _redact_url(url),
            'duration_ms': round((time.monotonic() - start) * 1000, 1),
            'error': _err_str(exc),
        })
        raise
    record: dict[str, Any] = {
        'dir': 'outbound',
        'service': service,
        'method': request.method,
        'url': _redact_url(url),
        'status': response.status_code,
        'duration_ms': round((time.monotonic() - start) * 1000, 1),
    }
    if capture:
        if raw_request_body is not None:
            request_body = _body_value(raw_request_body)
            record['request_body'] = request_body
            if isinstance(request_body, dict) and isinstance(request_body.get('model'), str):
                record['model'] = request_body['model']
        try:
            record['response_body'] = _body_value(await response.aread())
        except Exception:
            record['response_body'] = None
    _emit(record)
    return response


_installed = False
_inbound_installed = False


def install() -> None:
    """Attach the JSONL handler and patch both HTTP transports. Idempotent.

    Must run before any HTTP client is constructed (module import in
    ``main.py``) so library-internal sessions are captured too.
    """
    global _installed
    if _installed or not API_REQUEST_LOG_ENABLED:
        return
    _installed = True

    parent = os.path.dirname(API_REQUEST_LOG_PATH)
    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            API_REQUEST_LOG_PATH,
            maxBytes=API_REQUEST_LOG_MAX_BYTES,
            backupCount=API_REQUEST_LOG_BACKUPS,
            encoding='utf-8',
        )
    except OSError as exc:
        # Logging must never break the bot — not even at startup.
        logging.getLogger('quillbot').warning(
            'API request logging disabled: cannot open %s (%s)', API_REQUEST_LOG_PATH, exc
        )
        return
    handler.setFormatter(logging.Formatter('%(message)s'))
    _logger.addHandler(handler)
    if API_REQUEST_LOG_CONSOLE:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(message)s'))
        _logger.addHandler(console)
    _logger.setLevel(logging.INFO)

    aiohttp.ClientSession._request = _patched_aiohttp_request
    httpx.AsyncClient.send = _patched_httpx_send

    _emit({
        'dir': 'meta',
        'event': 'api_request_logging_installed',
        'path': API_REQUEST_LOG_PATH,
        'body_scope': sorted(API_REQUEST_LOG_BODY),
        'discord': API_REQUEST_LOG_DISCORD,
    })


def install_inbound_hooks(bot: commands.Bot) -> None:
    """Register the inbound Discord interaction listener on *bot* (idempotent)."""
    global _inbound_installed
    if _inbound_installed or not API_REQUEST_LOG_ENABLED:
        return
    _inbound_installed = True

    async def _on_interaction(interaction: discord.Interaction) -> None:
        data = interaction.data if isinstance(interaction.data, dict) else {}
        _emit({
            'dir': 'inbound',
            'service': 'discord',
            'interaction_type': interaction.type.name if interaction.type else 'unknown',
            'command': data.get('name'),
            'custom_id': data.get('custom_id'),
            'user': str(interaction.user) if interaction.user else None,
            'user_id': interaction.user.id if interaction.user else None,
            'guild_id': interaction.guild_id,
            'channel_id': interaction.channel_id,
        })

    bot.add_listener(_on_interaction, 'on_interaction')
