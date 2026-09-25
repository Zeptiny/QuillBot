"""Persistent image store — download once, re-encode small, inline as base64.

Discord CDN URLs expire after ~24h, and some OpenAI-compatible providers cannot
fetch remote images at all (glm-5.3-flash on surplusintelligence.ai hangs or
returns HTTP 400 on any ``image_url`` that is not a data URI). Conversation
turns therefore store local refs (``img:<sha256>.jpg``) instead of URLs, and
the message builders inline the stored bytes as data URIs for recent turns,
falling back to a text marker for older or missing ones.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import os
import time
from typing import Any, Iterable

import aiohttp
from cachetools import LRUCache
from PIL import Image, ImageOps

from config import (
    IMAGE_JPEG_QUALITY,
    IMAGE_MAX_SIDE,
    IMAGE_RETENTION_SECONDS,
    IMAGES_DIR,
    MAX_CONTENT_SIZE,
)

logger = logging.getLogger(__name__)

REF_PREFIX = 'img:'
MAX_IMAGES_PER_TURN = 4
_SWEEP_INTERVAL = 600.0
_last_sweep = 0.0
_sweep_lock = asyncio.Lock()
# attachment id -> stored ref (validated against the file on every hit)
_attachment_refs: LRUCache = LRUCache(maxsize=1024)


def is_image_ref(value: str) -> bool:
    """True when a stored turn image is a local ref (not a raw URL)."""
    return bool(value) and value.startswith(REF_PREFIX)


def image_path(ref: str) -> str:
    """Filesystem path for a ref (basename() guards against path traversal)."""
    return os.path.join(IMAGES_DIR, os.path.basename(ref[len(REF_PREFIX):]))


def _encode(data: bytes) -> bytes:
    """Re-encode raw image bytes as a downscaled JPEG (~150-250 KB per image)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format='JPEG', quality=IMAGE_JPEG_QUALITY, optimize=True)
    return out.getvalue()


async def persist(data: bytes) -> str | None:
    """Re-encode and store raw image bytes; returns a ref or None on failure."""
    if not data or len(data) > MAX_CONTENT_SIZE:
        return None
    try:
        encoded = await asyncio.to_thread(_encode, data)
    except Exception:
        logger.exception('Failed to re-encode image (%d bytes)', len(data))
        return None
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    ref = f'{REF_PREFIX}{digest}.jpg'
    path = image_path(ref)

    def _write() -> None:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        if os.path.exists(path):
            # Re-referenced: refresh mtime so the retention sweep measures age
            # from the latest use, not from the first download.
            os.utime(path, None)
            return
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'wb') as f:
            f.write(encoded)
        os.replace(tmp, path)

    try:
        await asyncio.to_thread(_write)
    except OSError:
        logger.exception('Failed to store image %s', path)
        return None
    await _maybe_sweep()
    return ref


def _touch(ref: str) -> bool:
    """Refresh a stored image's mtime; False when the file is gone (swept)."""
    try:
        os.utime(image_path(ref), None)
        return True
    except OSError:
        return False


async def persist_attachment(attachment: Any) -> str | None:
    """Download a discord.Attachment once and store it re-encoded.

    Attachments are immutable, so a known attachment id reuses its stored ref
    instead of re-downloading — channel context re-reads the same recent
    messages on every mention.
    """
    att_id = getattr(attachment, 'id', None)
    if att_id is not None:
        ref = _attachment_refs.get(att_id)
        if ref and _touch(ref):
            return ref
    try:
        data = await attachment.read()
    except Exception:
        logger.exception('Failed to read attachment %s', getattr(attachment, 'url', '?'))
        return None
    ref = await persist(data)
    if ref and att_id is not None:
        _attachment_refs[att_id] = ref
    return ref


async def persist_url(url: str) -> str | None:
    """Download an image URL once and store it re-encoded."""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning('Image download failed: HTTP %d for %s', resp.status, url[:120])
                    return None
                data = await resp.read()
    except Exception:
        logger.exception('Failed to download image %s', url[:120])
        return None
    return await persist(data)


async def persist_images(attachments: Iterable[Any]) -> list[str]:
    """Persist an iterable of attachments/urls, capped per turn, dropping failures."""
    refs: list[str] = []
    for att in list(attachments)[:MAX_IMAGES_PER_TURN]:
        url = getattr(att, 'url', None)
        ref = await persist_attachment(att) if url else (await persist_url(att) if att else None)
        if ref:
            refs.append(ref)
    return refs


def image_part(ref: str) -> dict | None:
    """OpenAI vision content part with the stored image inlined as a data URI."""
    path = image_path(ref)
    try:
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
    except OSError:
        logger.warning('Stored image missing, falling back to marker: %s', path)
        return None
    return {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}


def image_marker(count: int = 1) -> str:
    """Text placeholder for images not being re-sent in the current request."""
    noun = 'uma imagem' if count == 1 else f'{count} imagens'
    return f'[o usuário compartilhou {noun} neste turno]'


def context_image_marker(count: int = 1) -> str:
    """Placeholder for channel-context images of a replayed turn not re-sent now."""
    if count == 1:
        return '[uma imagem do canal deste turno não foi reenviada]'
    return f'[{count} imagens do canal deste turno não foram reenviadas]'


def context_image_note(count: int, *, after_user_images: bool = False) -> str:
    """Text note tying inlined channel-context images to their ``[anexo:…]`` lines.

    Without it the image parts arrive unlabeled after a long context block, and
    the model tends to read ``[anexo:x.png]`` as a bare filename it cannot see.
    """
    if count == 1:
        note = 'a imagem mais recente anexada nas mensagens do canal acima está incluída nesta mensagem'
    else:
        note = (
            f'as {count} imagens mais recentes anexadas nas mensagens do canal acima '
            'estão incluídas nesta mensagem, em ordem cronológica'
        )
    if after_user_images:
        note += ', depois das enviadas pelo usuário'
    return f'[{note}]'


async def _maybe_sweep() -> None:
    global _last_sweep
    if time.monotonic() - _last_sweep < _SWEEP_INTERVAL:
        return
    async with _sweep_lock:
        if time.monotonic() - _last_sweep < _SWEEP_INTERVAL:
            return
        _last_sweep = time.monotonic()
        await asyncio.to_thread(_sweep)


def _sweep() -> None:
    cutoff = time.time() - IMAGE_RETENTION_SECONDS
    try:
        names = os.listdir(IMAGES_DIR)
    except OSError:
        return
    removed = 0
    for name in names:
        path = os.path.join(IMAGES_DIR, name)
        try:
            if os.path.isfile(path) and os.stat(path).st_mtime < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info('Image store sweep removed %d file(s)', removed)
