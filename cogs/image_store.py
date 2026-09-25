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
from PIL import Image, ImageDraw, ImageFont, ImageOps

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
# Animated images (GIFs, animated stickers) become one sheet of this many frames
ANIMATION_FRAMES = 4
_BACKGROUND = (255, 255, 255)
_LABEL_COLOR = (32, 34, 37)
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


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def _flatten_alpha(frame: Image.Image) -> Image.Image:
    """RGB copy with transparency composited on white.

    A plain ``convert('RGB')`` turns transparent pixels black, which hides the
    dark outlines of most stickers and emojis.
    """
    if frame.mode in ('RGBA', 'LA', 'PA') or 'transparency' in frame.info:
        rgba = frame.convert('RGBA')
        flat = Image.new('RGB', rgba.size, _BACKGROUND)
        flat.paste(rgba, mask=rgba.getchannel('A'))
        return flat
    return frame.convert('RGB') if frame.mode != 'RGB' else frame


_CAPTION_H = 22


def _fit(draw: ImageDraw.ImageDraw, text: str, font: Any, width: int) -> str:
    """*text* cut with a trailing '..' until it fits *width* pixels."""
    if draw.textlength(text, font=font) <= width:
        return text
    while len(text) > 1 and draw.textlength(text + '..', font=font) > width:
        text = text[:-1]
    return text + '..'


def _grid(
    tiles: list[tuple[Image.Image, str]], columns: int, cell: int,
    *, caption: str = '', numbered: bool = False,
) -> Image.Image:
    """Lay tiles out on a white sheet.

    Labels go under their tile; ``numbered`` instead stamps 1, 2, … in each
    tile's corner (no extra height). ``caption`` adds a line at the bottom.
    """
    label_h = max(14, cell // 7) if not numbered and any(label for _, label in tiles) else 0
    caption_h = _CAPTION_H if caption else 0
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * cell, rows * (cell + label_h) + caption_h), _BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = _font(max(11, label_h - 3)) if label_h else None
    number_font = _font(max(12, cell // 12)) if numbered else None
    for i, (tile, label) in enumerate(tiles):
        x = (i % columns) * cell
        y = (i // columns) * (cell + label_h)
        tile = ImageOps.contain(tile, (cell - 4, cell - 4), Image.LANCZOS)  # up or down
        sheet.paste(tile, (x + (cell - tile.width) // 2, y + (cell - tile.height) // 2))
        if numbered:
            box = draw.textbbox((x + 6, y + 4), str(i + 1), font=number_font)
            draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=_BACKGROUND)
            draw.text((x + 6, y + 4), str(i + 1), fill=_LABEL_COLOR, font=number_font)
        elif label:
            draw.text((x + cell // 2, y + cell + label_h // 2), _fit(draw, label, font, cell - 4),
                      fill=_LABEL_COLOR, font=font, anchor='mm')
    if caption:
        draw.text(
            (sheet.width // 2, sheet.height - caption_h // 2), caption,
            fill=_LABEL_COLOR, font=_font(14), anchor='mm',
        )
    return sheet


def _animation_sheet(img: Image.Image) -> Image.Image | None:
    """Evenly spaced frames of an animated image as one numbered sheet.

    Without it the model only sees the first frame of a GIF, which is often a
    blank or transitional one. ``None`` for still images.
    """
    total = getattr(img, 'n_frames', 1)
    if not getattr(img, 'is_animated', False) or total < 2:
        return None
    count = min(ANIMATION_FRAMES, total)
    indexes = sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})
    frames: list[tuple[Image.Image, str]] = []
    for index in indexes:
        img.seek(index)
        frames.append((_flatten_alpha(img.copy()), ''))
    columns = 2 if len(frames) > 2 else len(frames)
    rows = (len(frames) + columns - 1) // columns
    # Size cells so the whole sheet (caption included) fits IMAGE_MAX_SIDE unscaled.
    cell = min(IMAGE_MAX_SIDE // columns, (IMAGE_MAX_SIDE - _CAPTION_H) // rows)
    return _grid(
        frames, columns, cell, numbered=True,
        caption=f'imagem animada: {len(frames)} quadros em ordem',  # ASCII: default font lacks ç/ã
    )


def labeled_grid(items: list[tuple[bytes, str]], cell: int = 96, columns: int = 4) -> bytes:
    """PNG sheet of small labelled images (e.g. custom emojis with their names)."""
    tiles: list[tuple[Image.Image, str]] = []
    for data, label in items:
        try:
            img = Image.open(io.BytesIO(data))
            img.seek(0)
            tiles.append((_flatten_alpha(img.copy()), label))
        except Exception:
            logger.warning('Skipping undecodable grid tile %r', label)
    if not tiles:
        raise ValueError('no decodable images for grid')
    out = io.BytesIO()
    _grid(tiles, min(columns, len(tiles)), cell).save(out, format='PNG')
    return out.getvalue()


def _encode(data: bytes) -> bytes:
    """Re-encode raw image bytes as a downscaled JPEG (~150-250 KB per image).

    Animated images become a sheet of sampled frames; transparency is
    composited on white.
    """
    img = Image.open(io.BytesIO(data))
    sheet = _animation_sheet(img)
    if sheet is not None:
        img = sheet
    else:
        img = _flatten_alpha(ImageOps.exif_transpose(img))
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
    messages on every mention. Any object with ``id`` and ``async read()``
    works the same way (stickers, link embeds and emoji sheets from
    :mod:`cogs.message_media`).
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


async def download(url: str) -> bytes | None:
    """Fetch an image URL (at most MAX_CONTENT_SIZE bytes); ``None`` on failure."""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning('Image download failed: HTTP %d for %s', resp.status, url[:120])
                    return None
                if (resp.content_length or 0) > MAX_CONTENT_SIZE:
                    logger.warning('Image too large (%d bytes): %s', resp.content_length, url[:120])
                    return None
                return await resp.read()
    except Exception:
        logger.exception('Failed to download image %s', url[:120])
        return None


async def persist_url(url: str) -> str | None:
    """Download an image URL once and store it re-encoded."""
    data = await download(url)
    return await persist(data) if data else None


async def persist_images(attachments: Iterable[Any]) -> list[str]:
    """Persist an iterable of attachments/urls, capped per turn, dropping failures."""
    refs: list[str] = []
    for att in list(attachments)[:MAX_IMAGES_PER_TURN]:
        if hasattr(att, 'read'):
            ref = await persist_attachment(att)
        else:
            ref = await persist_url(att) if isinstance(att, str) and att else None
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
