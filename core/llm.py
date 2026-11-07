"""The bot's one LLM client: chat completions (streamed, retried) and embeddings.

Every feature used to build its own ``AsyncOpenAI`` client and its own local
embedding model; now they all go through :class:`LLMService`
(``bot.services.llm``). The module-level :func:`create_chat_completion` stays
usable with any OpenAI-compatible client, which is what the tests fake.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from typing import Any

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LLM_REASONING_EFFORT,
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_BASE_DELAY,
    LOCAL_EMBEDDING_DEVICE,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    chat_priority_extra_body,
)
from core.local_inference import run_local_model

logger = logging.getLogger(__name__)

_MAX_RETRY_AFTER = 10.0


def _get(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def cached_tokens(usage: Any) -> int | None:
    """Cached prompt tokens from OpenAI/OpenRouter or Anthropic-style usage."""
    if usage is None:
        return None
    details = _get(usage, 'prompt_tokens_details', None)
    cached = _get(details, 'cached_tokens', None) if details is not None else None
    if cached is None:
        cached = _get(usage, 'cache_read_input_tokens', None)
    return cached


def usage_summary(response: Any) -> str:
    """Format token usage (incl. cached prefix tokens) for log lines."""
    usage = _get(response, 'usage', None)
    if usage is None:
        return 'n/a'
    parts = [
        f'prompt={_get(usage, "prompt_tokens", None)}',
        f'completion={_get(usage, "completion_tokens", None)}',
    ]
    cached = cached_tokens(usage)
    if cached is not None:
        prompt = _get(usage, 'prompt_tokens', None) or 0
        suffix = f' (hit={round(100 * cached / prompt)}%)' if prompt else ''
        parts.append(f'cached={cached}{suffix}')
    details = _get(usage, 'completion_tokens_details', None)
    reasoning = _get(details, 'reasoning_tokens', None) if details is not None else None
    if reasoning is not None:
        parts.append(f'reasoning={reasoning}')
    return ', '.join(parts)


def _is_transient(error: BaseException) -> bool:
    """Whether *error* is worth retrying: throttling, timeouts, dropped connections, 5xx."""
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency
        return False
    if isinstance(error, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(error, openai.InternalServerError):
        return True
    status = getattr(error, 'status_code', None)
    return isinstance(error, openai.APIStatusError) and isinstance(status, int) and status >= 500


def _retry_delay(error: BaseException, attempt: int) -> float:
    response = getattr(error, 'response', None)
    headers = getattr(response, 'headers', None) or {}
    try:
        after = float(headers.get('retry-after'))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        after = None
    if after is not None and after >= 0:
        return min(after, _MAX_RETRY_AFTER)
    base = LLM_RETRY_BASE_DELAY * (2 ** attempt)
    return base + random.uniform(0, base / 4)


def _content_text(value: Any) -> str:
    """The answer text in a delta's ``content``, whichever shape the provider sends.

    Most providers stream ``content`` as a string, but Mistral's reasoning
    models send a list of typed chunks (``{"type": "text", "text": ...}`` and
    ``{"type": "thinking", "thinking": [...]}``). Only the ``text`` chunks are
    the answer; thinking chunks are dropped, like a reasoning trace the bot
    never shows or replays.
    """
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ''.join(_content_text(part) for part in value)
    if _get(value, 'type', None) == 'text':
        text = _get(value, 'text', None)
        return text if isinstance(text, str) else ''
    return ''


async def _stream_completion(client: Any, kwargs: dict) -> Any:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    stream = await client.chat.completions.create(
        **kwargs, stream=True, stream_options={'include_usage': True},
    )
    content: list[str] = []
    extras: dict[str, list[str]] = {}
    calls: dict[int, dict] = {}
    finish_reason = None
    usage = None
    meta: dict[str, Any] = {}
    try:
        async for chunk in stream:
            for key in ('id', 'model', 'created'):
                if not meta.get(key) and _get(chunk, key, None):
                    meta[key] = _get(chunk, key)
            if _get(chunk, 'usage', None) is not None:
                usage = chunk.usage
            for choice in _get(chunk, 'choices', None) or []:
                if _get(choice, 'index', 0):
                    continue  # only n=1 is ever requested
                if _get(choice, 'finish_reason', None):
                    finish_reason = choice.finish_reason
                delta = _get(choice, 'delta', None)
                if delta is None:
                    continue
                text = _content_text(_get(delta, 'content', None))
                if text:
                    content.append(text)
                for key, value in (getattr(delta, 'model_extra', None) or {}).items():
                    if isinstance(value, str) and value:
                        extras.setdefault(key, []).append(value)
                for tc in _get(delta, 'tool_calls', None) or []:
                    index = _get(tc, 'index', None)
                    if index is None:  # some providers send whole calls without an index
                        index = len(calls)
                    call = calls.setdefault(index, {'id': None, 'name': '', 'arguments': ''})
                    if _get(tc, 'id', None):
                        call['id'] = tc.id
                    fn = _get(tc, 'function', None)
                    if fn is not None:
                        if _get(fn, 'name', None):
                            call['name'] += fn.name
                        if _get(fn, 'arguments', None):
                            call['arguments'] += fn.arguments
    finally:
        close = getattr(stream, 'close', None) or getattr(stream, 'aclose', None)
        if close is not None:
            await close()

    message_fields: dict[str, Any] = {
        'role': 'assistant',
        'content': ''.join(content) if content else None,
    }
    if calls:
        message_fields['tool_calls'] = [
            {
                'id': call['id'] or f'call_{uuid.uuid4().hex[:24]}',
                'type': 'function',
                'function': {'name': call['name'], 'arguments': call['arguments']},
            }
            for _, call in sorted(calls.items())
        ]
    for key, parts in extras.items():
        message_fields.setdefault(key, ''.join(parts))
    message = ChatCompletionMessage(**message_fields)
    return ChatCompletion.model_construct(
        id=meta.get('id') or '',
        object='chat.completion',
        created=meta.get('created') or 0,
        model=meta.get('model') or kwargs.get('model') or '',
        choices=[Choice.model_construct(index=0, finish_reason=finish_reason, message=message)],
        usage=usage,
    )


async def create_chat_completion(client: Any, **kwargs: Any) -> Any:
    """``client.chat.completions.create`` over a stream, returned as one completion.

    Some providers behind the gateway (e.g. zai) reject non-streaming chat
    completions with ``streaming_only``, so every call streams and the chunks
    are folded back into a ``ChatCompletion``-shaped object: callers keep
    reading ``response.choices[0].message`` / ``finish_reason`` / ``usage``.
    The request parameters (``messages`` included) are passed through as given,
    so the prompt prefix the provider caches is unchanged.

    The assembled message carries only what the stream delivered (role,
    content, tool calls, plus any provider string extras such as
    ``reasoning_content``), and it is a real ``ChatCompletionMessage`` so it
    can be appended to ``messages`` and replayed like before.

    Transient failures (429, timeouts, connection errors, 5xx) are retried up
    to ``LLM_RETRY_ATTEMPTS`` times with exponential backoff; the last error
    is raised. Unless the caller passed its own ``extra_body``, the Merge
    Gateway vendor priority is attached when the base URL points at merge.dev,
    and the configured ``reasoning_effort`` is sent unless the caller passed its own.
    """
    priority_extra = chat_priority_extra_body(str(kwargs.get('model') or ''))
    if priority_extra is not None:
        kwargs.setdefault('extra_body', priority_extra)
    if LLM_REASONING_EFFORT:
        kwargs.setdefault('reasoning_effort', LLM_REASONING_EFFORT)
    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            return await _stream_completion(client, kwargs)
        except Exception as e:
            if attempt + 1 >= LLM_RETRY_ATTEMPTS or not _is_transient(e):
                raise
            delay = _retry_delay(e, attempt)
            logger.warning(
                'LLM call failed (%s: %s); retrying in %.1fs (attempt %d/%d)',
                type(e).__name__, str(e)[:160], delay, attempt + 2, LLM_RETRY_ATTEMPTS,
            )
            await asyncio.sleep(delay)
    raise RuntimeError('unreachable')  # pragma: no cover


class LLMService:
    """One OpenAI-compatible client and one embedding backend for the whole bot."""

    def __init__(self, client: Any = None, *, base_url: str = OPENAI_BASE_URL, api_key: str | None = OPENAI_API_KEY):
        self.client = client
        if self.client is None:
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or 'not-needed')
                if not api_key and 'openrouter.ai' in base_url:
                    logger.warning('OPENAI_API_KEY not set -- AI features will fail until configured')
            except Exception:
                logger.exception('Failed to initialize the LLM client')
                self.client = None
        self._local_model = None
        self.local_dim: int | None = None

    @property
    def available(self) -> bool:
        return self.client is not None

    async def complete(self, **kwargs: Any) -> Any:
        """One chat completion (streamed and retried, see :func:`create_chat_completion`)."""
        if self.client is None:
            raise RuntimeError('LLM client unavailable -- check OPENAI_API_KEY / OPENAI_BASE_URL')
        return await create_chat_completion(self.client, **kwargs)

    async def text(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        """Plain system+user completion returning the answer text ('' when empty)."""
        response = await self.complete(
            model=model,
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            max_tokens=max_tokens,
        )
        logger.debug('LLM text completion usage=[%s]', usage_summary(response))
        return (response.choices[0].message.content or '').strip()

    # --- Embeddings ---

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info('Loading local embedding model %s (device=%s)', LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DEVICE)
            self._local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device=LOCAL_EMBEDDING_DEVICE)
            test = self._local_model.encode(['test'], normalize_embeddings=True)
            self.local_dim = test.shape[1] if hasattr(test, 'shape') else len(test[0])
            logger.info('Local embedding model loaded (dim=%s)', self.local_dim)
        return self._local_model

    async def warm_up(self) -> None:
        """Load the local embedding model ahead of the first request (local provider only)."""
        if EMBEDDING_PROVIDER != 'local':
            return
        try:
            await run_local_model(self._get_local_model)
        except Exception:
            logger.exception('Failed to load local embedding model %s', LOCAL_EMBEDDING_MODEL)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* with the configured provider (one vector per text)."""
        if not texts:
            return []
        if EMBEDDING_PROVIDER == 'local':
            vectors = await run_local_model(
                lambda: self._get_local_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
            )
            return vectors.tolist() if hasattr(vectors, 'tolist') else [list(v) for v in vectors]
        if self.client is None:
            raise RuntimeError('LLM client unavailable for embeddings')
        response = await self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in response.data]
