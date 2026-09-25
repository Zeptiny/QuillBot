"""Stickers, GIF embeds, custom emojis and reactions of Discord messages.

Attachments are not the only visual part of a message: stickers (figurinhas),
GIF and image link embeds (Tenor, Giphy, direct image links) and custom server
emojis inline in the text all reach the model as nothing but ids or URLs
unless they are rendered. This module turns them into

- text markers for the canonical message line (``[figurinha:Nome]``,
  ``[gif: cat dance]``, ``:kek:`` instead of ``<:kek:123>``);
- image sources :func:`cogs.image_store.persist_images` can store, each with
  an ``id`` (download cache key) and ``async read()``;
- a reactions suffix listing each reaction with its count and who reacted.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

from cachetools import LRUCache

from cogs import image_store
from config import CHANNEL_CONTEXT_REACTIONS_ENABLED, REACTION_USERS_LIMIT

logger = logging.getLogger(__name__)

CUSTOM_EMOJI_RE = re.compile(r'<(a?):([A-Za-z0-9_~]{1,32}):(\d{15,21})>')
# Custom emojis drawn on one sheet per message
EMOJI_SHEET_MAX = 12
VISUAL_EMBED_TYPES = frozenset({'gifv', 'image'})
# Tenor media paths end in a 5-char rendition code (AAAAC gif, AAAAM tinygif,
# AAAAe still preview, AAAPo mp4) after the media id.
_TENOR_PATH_RE = re.compile(r'^/(?P<id>[A-Za-z0-9_-]+)(?P<code>AAA[A-Za-z0-9_-]{2})/(?P<slug>[^/]+?)\.\w+$')
_GIPHY_MEDIA_RE = re.compile(r'/media/(?:v1\.[^/]+/)?(?P<id>[A-Za-z0-9]+)/')
_GIPHY_PAGE_RE = re.compile(r'/gifs/(?:[^/]*-)?(?P<id>[A-Za-z0-9]+)/?$')

# Reactor names per reaction are fetched for at most this many reactions per
# render, all together, within this many seconds; the rest show counts only.
REACTION_FETCH_BUDGET = 20
REACTION_FETCH_TIMEOUT = 5.0
# (message id, emoji key, count) -> first reactor names
_reactors: LRUCache = LRUCache(maxsize=2048)


def _flatten(s: str) -> str:
    return ' '.join(s.split())


# ---------------------------------------------------------------------------
# Custom emojis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CustomEmoji:
    name: str
    id: int
    animated: bool = False

    @property
    def url(self) -> str:
        # .png serves a still frame for animated emojis too: one sheet, one frame each.
        return f'https://cdn.discordapp.com/emojis/{self.id}.png?size=96'


def emoji_text(text: str) -> str:
    """Replace ``<:name:id>`` / ``<a:name:id>`` markup with ``:name:``."""
    return CUSTOM_EMOJI_RE.sub(lambda m: f':{m.group(2)}:', text or '')


def custom_emojis(text: str) -> list[CustomEmoji]:
    """Distinct custom emojis in *text*, in order of first appearance."""
    seen: dict[int, CustomEmoji] = {}
    for animated, name, emoji_id in CUSTOM_EMOJI_RE.findall(text or ''):
        seen.setdefault(int(emoji_id), CustomEmoji(name, int(emoji_id), bool(animated)))
    return list(seen.values())


class EmojiSheet:
    """Image source: a message's custom emojis on one sheet, labelled ``:name:``.

    One sheet per message keeps emojis to a single image slot, and the labels
    tie each picture to the ``:name:`` the model reads in the text.
    """

    def __init__(self, emojis: Iterable[CustomEmoji]):
        self.emojis = tuple(emojis)[:EMOJI_SHEET_MAX]
        self.id = 'emojis:' + ','.join(str(e.id) for e in self.emojis)

    async def read(self) -> bytes:
        datas = await asyncio.gather(*(image_store.download(e.url) for e in self.emojis))
        items = [(data, f':{e.name}:') for data, e in zip(datas, self.emojis) if data]
        if not items:
            raise ValueError('no custom emoji could be downloaded')
        return await asyncio.to_thread(image_store.labeled_grid, items)


def emoji_sheet(msg: Any) -> EmojiSheet | None:
    emojis = custom_emojis(getattr(msg, 'content', '') or '')
    return EmojiSheet(emojis) if emojis else None


# ---------------------------------------------------------------------------
# Stickers
# ---------------------------------------------------------------------------

def stickers(msg: Any) -> list[Any]:
    return list(getattr(msg, 'stickers', None) or [])


def _is_lottie(sticker: Any) -> bool:
    fmt = getattr(sticker, 'format', None)
    return getattr(fmt, 'name', str(fmt)) == 'lottie'


def readable_stickers(msg: Any) -> list[Any]:
    """Stickers that can be downloaded as images (lottie ones are vector JSON)."""
    return [s for s in stickers(msg) if not _is_lottie(s)]


def sticker_markers(msg: Any) -> list[str]:
    return [f"[figurinha:{_flatten(getattr(s, 'name', '') or '?')}]" for s in stickers(msg)]


# ---------------------------------------------------------------------------
# GIF / image link embeds
# ---------------------------------------------------------------------------

def visual_embeds(msg: Any) -> list[Any]:
    return [e for e in (getattr(msg, 'embeds', None) or []) if getattr(e, 'type', None) in VISUAL_EMBED_TYPES]


def _media_urls(proxy: Any) -> list[str]:
    return [u for u in (getattr(proxy, 'proxy_url', None), getattr(proxy, 'url', None)) if u]


def _animated_url(embed: Any) -> str | None:
    """Small animated GIF rendition for Tenor/Giphy embeds.

    Discord only exposes an mp4 and a still preview for these, so the GIF is
    derived from the provider's URL scheme; callers fall back to the preview.
    """
    candidates = [getattr(embed, 'url', None)]
    for proxy in (getattr(embed, 'thumbnail', None), getattr(embed, 'video', None)):
        candidates.append(getattr(proxy, 'url', None))
    for url in filter(None, candidates):
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.endswith('tenor.com'):
            m = _TENOR_PATH_RE.match(parsed.path)
            if m:
                return f"https://media.tenor.com/{m['id']}AAAAM/{m['slug']}.gif"
        elif host.endswith('giphy.com'):
            m = _GIPHY_MEDIA_RE.search(parsed.path) or _GIPHY_PAGE_RE.search(parsed.path)
            if m:
                return f"https://media.giphy.com/media/{m['id']}/200w.gif"
    return None


class LinkedMedia:
    """Image source for a GIF/image embed: first URL that downloads wins."""

    def __init__(self, urls: Iterable[str]):
        self.urls = tuple(dict.fromkeys(u for u in urls if u))
        self.id = self.urls[0] if self.urls else ''

    async def read(self) -> bytes:
        for url in self.urls:
            data = await image_store.download(url)
            if data:
                return data
        raise ValueError('no embed media URL could be downloaded')


def embed_media(embed: Any) -> LinkedMedia | None:
    urls = [_animated_url(embed)]
    urls += _media_urls(getattr(embed, 'thumbnail', None))
    urls += _media_urls(getattr(embed, 'image', None))
    media = LinkedMedia(urls)
    return media if media.urls else None


def _gif_label(embed: Any) -> str:
    title = _flatten(getattr(embed, 'title', None) or '')
    if title:
        return title[:60]
    parsed = urlparse(getattr(embed, 'url', None) or '')
    segment = parsed.path.rstrip('/').rsplit('/', 1)[-1]
    words = [w for w in segment.split('-') if w and w.lower() != 'gif' and not w.isdigit()]
    if parsed.netloc.lower().endswith('giphy.com') and len(words) > 1:
        words = words[:-1]  # trailing Giphy media id
    return ' '.join(words)[:60]


def embed_markers(msg: Any) -> list[str]:
    markers = []
    for embed in visual_embeds(msg):
        if embed.type == 'gifv':
            label = _gif_label(embed)
            markers.append(f'[gif: {label}]' if label else '[gif]')
        else:
            markers.append('[imagem do link]')
    return markers


# ---------------------------------------------------------------------------
# Everything visual in one message
# ---------------------------------------------------------------------------

def image_attachments(msg: Any) -> list[Any]:
    return [
        a for a in (getattr(msg, 'attachments', None) or [])
        if (getattr(a, 'content_type', None) or '').startswith('image/')
    ]


def media_sources(msg: Any) -> list[Any]:
    """Image attachments, stickers and GIF/image embeds, in that order."""
    sources: list[Any] = [*image_attachments(msg), *readable_stickers(msg)]
    sources += [m for m in (embed_media(e) for e in visual_embeds(msg)) if m]
    return sources


def visual_sources(msg: Any) -> list[Any]:
    """:func:`media_sources` plus the message's custom-emoji sheet, last."""
    sources = media_sources(msg)
    sheet = emoji_sheet(msg)
    if sheet is not None:
        sources.append(sheet)
    return sources


def visual_markers(msg: Any) -> str:
    """Sticker and GIF markers for a question built from raw message text."""
    return ' '.join(sticker_markers(msg) + embed_markers(msg))


def question_with_media(question: str, msg: Any) -> str:
    """A question typed in *msg*, with ``:name:`` emojis and sticker/GIF markers."""
    text = emoji_text(question).strip()
    markers = visual_markers(msg)
    return f'{text} {markers}'.strip() if markers else text


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------

def _emoji_key(emoji: Any) -> str:
    return emoji if isinstance(emoji, str) else str(getattr(emoji, 'id', None) or getattr(emoji, 'name', '?'))


def reaction_label(emoji: Any) -> str:
    if isinstance(emoji, str):
        return emoji
    return f":{getattr(emoji, 'name', None) or '?'}:"


async def _reactor_names(reaction: Any, limit: int) -> list[str]:
    names: list[str] = []
    async for user in reaction.users(limit=limit):
        names.append(_flatten(getattr(user, 'display_name', None) or str(user)))
    return names


def _format_reactions(msg: Any, names: dict[tuple, list[str]]) -> str:
    parts = []
    for reaction in getattr(msg, 'reactions', None) or []:
        count = getattr(reaction, 'count', 0) or 0
        part = f'{reaction_label(reaction.emoji)} {count}'
        who = names.get((msg.id, _emoji_key(reaction.emoji), count))
        if who:
            extra = count - len(who)
            part += f" ({', '.join(who)}{f' +{extra}' if extra > 0 else ''})"
        parts.append(part)
    return f"[reações: {'; '.join(parts)}]" if parts else ''


async def reaction_summaries(messages: Iterable[Any]) -> dict[Any, str]:
    """``{message id: '[reações: 👍 3 (Ana, Bruno +1); :kek: 1 (Caio)]'}``.

    Pass messages newest first: reactor names are fetched for the first
    ``REACTION_FETCH_BUDGET`` uncached reactions, concurrently and bounded by
    ``REACTION_FETCH_TIMEOUT``; anything not fetched in time shows its count
    only. Empty when ``CHANNEL_CONTEXT_REACTIONS_ENABLED`` is off.
    """
    if not CHANNEL_CONTEXT_REACTIONS_ENABLED:
        return {}
    messages = [m for m in messages if getattr(m, 'reactions', None)]
    if not messages:
        return {}
    names: dict[tuple, list[str]] = {}
    pending: dict[tuple, asyncio.Task] = {}
    for msg in messages:
        for reaction in msg.reactions:
            key = (msg.id, _emoji_key(reaction.emoji), getattr(reaction, 'count', 0) or 0)
            cached = _reactors.get(key)
            if cached is not None:
                names[key] = cached
            elif REACTION_USERS_LIMIT > 0 and len(pending) < REACTION_FETCH_BUDGET and key not in pending:
                pending[key] = asyncio.ensure_future(_reactor_names(reaction, REACTION_USERS_LIMIT))
    if pending:
        done, not_done = await asyncio.wait(pending.values(), timeout=REACTION_FETCH_TIMEOUT)
        for task in not_done:
            task.cancel()
        for key, task in pending.items():
            if task not in done:
                continue
            if task.exception() is not None:
                logger.warning('Failed to fetch reactors for %s: %s', key, task.exception())
                continue
            names[key] = _reactors[key] = task.result()
    return {msg.id: _format_reactions(msg, names) for msg in messages}
