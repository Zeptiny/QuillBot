"""Action GIFs: an anime GIF of one person hugging, kissing, slapping… another.

Used by ``/hug``, ``/kiss``, ``/slap`` and the rest (``cogs/actions.py``), by
their Retribuir/Revidar buttons and by the ``send_action_gif`` tool, where the
bot does the action itself.

GIFs come from free anime GIF APIs, tried in each action's order: nekos.best
(which also names the anime) and otakugifs.xyz. Every GIF fetched is
remembered per action (``RECENT_GIFS`` of them), so when the APIs are down an
earlier one is reused; with none at all, the action goes out as text. How many
times each person did each action to each other person is counted in the
same SQLite file.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import random
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp
import discord

from core.turn import NO_REPLY_TOKEN

logger = logging.getLogger(__name__)

NEKOS = 'nekos.best'
OTAKU = 'otakugifs'

# GIFs kept per action for when the APIs are down.
RECENT_GIFS = 50
# Only GIFs hosted by the APIs themselves are posted: an embed shows whatever URL it gets.
_ALLOWED_HOSTS = ('nekos.best', 'otakugifs.xyz')
_NEKOS_URL = 'https://nekos.best/api/v2/{category}'
_OTAKU_URL = 'https://api.otakugifs.xyz/gif'

AFFECTION = 'carinho'
PLAYFUL = 'brincadeira'
ROUGH = 'agressao'
_COLORS = {AFFECTION: 0xF47FFF, PLAYFUL: 0xF1C40F, ROUGH: 0xE74C3C}
_RETURN_LABELS = {AFFECTION: 'Retribuir', PLAYFUL: 'Devolver', ROUGH: 'Revidar'}


@dataclass(frozen=True)
class Action:
    """One action: its command, how it reads and where its GIFs come from."""

    key: str  # command name (/hug) and the tool's value
    emoji: str
    past: str  # "**Ana** abraçou @Bruno", "Ana já abraçou Bruno 3 vezes"
    description: str  # the slash command's description
    kind: str  # AFFECTION, PLAYFUL or ROUGH
    sources: tuple[tuple[str, str], ...]  # (provider, category), tried in order

    @property
    def return_label(self) -> str:
        """The button that does it back: Retribuir a hug, Revidar a slap."""
        return _RETURN_LABELS[self.kind]

    @property
    def color(self) -> int:
        return _COLORS[self.kind]


def _both(category: str) -> tuple[tuple[str, str], ...]:
    return ((NEKOS, category), (OTAKU, category))


ACTIONS: dict[str, Action] = {a.key: a for a in (
    Action('hug', '🤗', 'abraçou', 'Dê um abraço em alguém', AFFECTION, _both('hug')),
    Action('kiss', '😘', 'beijou', 'Dê um beijo em alguém', AFFECTION, _both('kiss')),
    Action('slap', '🖐️', 'deu um tapa em', 'Dê um tapa em alguém', ROUGH, _both('slap')),
    Action('lick', '👅', 'lambeu', 'Lamba alguém', PLAYFUL, ((OTAKU, 'lick'),)),
    Action('pat', '🫶', 'fez cafuné em', 'Faça cafuné em alguém', AFFECTION, _both('pat')),
    Action('cuddle', '🥰', 'se aconchegou em', 'Aconchegue-se em alguém', AFFECTION, _both('cuddle')),
    Action('poke', '👉', 'cutucou', 'Cutuque alguém', PLAYFUL, _both('poke')),
    Action('bite', '😬', 'mordeu', 'Morda alguém', PLAYFUL, _both('bite')),
    Action('bonk', '🔨', 'deu um bonk em', 'Dê um bonk em alguém', PLAYFUL, ((NEKOS, 'bonk'),)),
    Action('punch', '👊', 'deu um soco em', 'Dê um soco em alguém', ROUGH, _both('punch')),
    Action('kick', '🦵', 'chutou', 'Chute alguém', ROUGH, ((NEKOS, 'kick'),)),
    Action('tickle', '🤭', 'fez cócegas em', 'Faça cócegas em alguém', PLAYFUL, _both('tickle')),
    Action('highfive', '🙌', 'bateu um toca aqui com', 'Bata um toca aqui com alguém', AFFECTION, ((NEKOS, 'highfive'),)),
    Action('handhold', '🤝', 'segurou a mão de', 'Segure a mão de alguém', AFFECTION, _both('handhold')),
    Action('feed', '🍙', 'deu comida para', 'Dê comida para alguém', AFFECTION, ((NEKOS, 'feed'),)),
)}


# ---------------------------------------------------------------------------
# GIF providers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gif:
    url: str
    anime: str = ''


def allowed_url(url: Any) -> bool:
    """Whether *url* is an https link on one of the APIs' own hosts."""
    if not isinstance(url, str):
        return False
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    return parsed.scheme == 'https' and any(host == h or host.endswith('.' + h) for h in _ALLOWED_HOSTS)


class ProviderError(Exception):
    """A GIF API answered with an error status."""


async def _nekos(session: aiohttp.ClientSession, category: str) -> Gif | None:
    async with session.get(_NEKOS_URL.format(category=category)) as resp:
        if resp.status != 200:
            raise ProviderError(f'HTTP {resp.status}')
        data = await resp.json(content_type=None)
    result = ((data or {}).get('results') or [{}])[0] or {}
    url = result.get('url')
    return Gif(url, str(result.get('anime_name') or '')) if allowed_url(url) else None


async def _otaku(session: aiohttp.ClientSession, category: str) -> Gif | None:
    async with session.get(_OTAKU_URL, params={'reaction': category, 'format': 'gif'}) as resp:
        if resp.status != 200:
            raise ProviderError(f'HTTP {resp.status}')
        data = await resp.json(content_type=None)
    url = (data or {}).get('url')
    return Gif(url) if allowed_url(url) else None


PROVIDERS = {NEKOS: _nekos, OTAKU: _otaku}


class GifPicker:
    """A GIF for an action: each of its sources in order, then a recent one."""

    def __init__(self, session: aiohttp.ClientSession, store: ActionStore):
        self.session = session
        self.store = store

    async def pick(self, action: Action) -> Gif | None:
        for provider, category in action.sources:
            try:
                gif = await PROVIDERS[provider](self.session, category)
            except Exception as e:  # network, timeout, error status, bad JSON: try the next source
                logger.warning('[actions] %s failed for %s: %r', provider, category, e)
                continue
            if gif is not None:
                await asyncio.to_thread(self.store.remember_gif, action.key, gif)
                return gif
            logger.info('[actions] %s had no GIF for %s', provider, category)
        gif = await asyncio.to_thread(self.store.recent_gif, action.key)
        if gif is None:
            logger.warning('[actions] no GIF for %s: sending it as text', action.key)
        return gif


# ---------------------------------------------------------------------------
# Counts and recent GIFs
# ---------------------------------------------------------------------------

class ActionStore:
    """SQLite (``ACTIONS_DB_PATH``): action counts per pair and recent GIF URLs."""

    def __init__(self, path: str, *, recent: int = RECENT_GIFS):
        self.path = path
        self.recent = recent
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def load(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        con = self._connect()
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS action_counts (
                    guild_id INTEGER NOT NULL,
                    actor_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    last_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, actor_id, target_id, action)
                )""")
            con.execute("""
                CREATE TABLE IF NOT EXISTS action_gifs (
                    action TEXT NOT NULL,
                    url TEXT NOT NULL,
                    anime TEXT NOT NULL DEFAULT '',
                    seen_at REAL NOT NULL,
                    PRIMARY KEY (action, url)
                )""")
            con.commit()
        finally:
            con.close()

    def bump(self, guild_id: int, actor_id: int, target_id: int, action: str) -> int:
        """Count one more *action* from *actor_id* to *target_id*; returns the new count."""
        key = (int(guild_id), int(actor_id), int(target_id), action)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            con = self._connect()
            try:
                con.execute(
                    'INSERT INTO action_counts (guild_id, actor_id, target_id, action, count, last_at) '
                    'VALUES (?,?,?,?,1,?) ON CONFLICT (guild_id, actor_id, target_id, action) '
                    'DO UPDATE SET count = count + 1, last_at = excluded.last_at',
                    (*key, now),
                )
                con.commit()
                row = con.execute(
                    'SELECT count FROM action_counts WHERE guild_id=? AND actor_id=? AND target_id=? AND action=?', key,
                ).fetchone()
            finally:
                con.close()
        return int(row[0]) if row else 1

    def remember_gif(self, action: str, gif: Gif) -> None:
        """Keep *gif* among *action*'s newest ``recent`` GIFs."""
        with self._lock:
            con = self._connect()
            try:
                con.execute(
                    'INSERT OR REPLACE INTO action_gifs (action, url, anime, seen_at) VALUES (?,?,?,?)',
                    (action, gif.url, gif.anime, time.time()),
                )
                con.execute(
                    'DELETE FROM action_gifs WHERE action=? AND url NOT IN '
                    '(SELECT url FROM action_gifs WHERE action=? ORDER BY seen_at DESC LIMIT ?)',
                    (action, action, self.recent),
                )
                con.commit()
            finally:
                con.close()

    def recent_gif(self, action: str) -> Gif | None:
        """One of *action*'s remembered GIFs, at random."""
        con = self._connect()
        try:
            rows = con.execute('SELECT url, anime FROM action_gifs WHERE action=?', (action,)).fetchall()
        finally:
            con.close()
        if not rows:
            return None
        url, anime = random.choice(rows)
        return Gif(url, anime or '')


# ---------------------------------------------------------------------------
# How an action reads
# ---------------------------------------------------------------------------

def _name(person: Any) -> str:
    return str(getattr(person, 'display_name', None) or getattr(person, 'name', None) or person)


def count_text(action: Action, actor: Any, target: Any, count: int) -> str:
    """"Ana já abraçou Bruno 3 vezes" (plain text: an embed footer or a subtext line)."""
    if count <= 1:
        return f'Primeira vez que {_name(actor)} {action.past} {_name(target)}'
    return f'{_name(actor)} já {action.past} {_name(target)} {count} vezes'


def render(
    action: Action, actor: Any, target: Any, gif: Gif | None, count: int, *,
    note: str = '', returned: bool = False,
) -> tuple[str, discord.Embed | None]:
    """The message: ``(content, embed)``; without a GIF, text only (``embed`` is ``None``).

    The content names the actor in bold and mentions the target, so the chat
    (and the bot, reading the channel) sees who did what to whom.
    """
    actor_name = discord.utils.escape_markdown(_name(actor))
    line = f'**{actor_name}** {action.past} {getattr(target, "mention", _name(target))}'
    if returned:
        line += ' de volta'
    content = f'{action.emoji} {note}: {line}!' if note else f'{action.emoji} {line}!'
    counted = count_text(action, actor, target, count)
    if gif is None:
        return f'{content}\n-# {counted}', None
    embed = discord.Embed(color=action.color)
    embed.set_image(url=gif.url)
    embed.set_footer(text=f'{counted} · Anime: {gif.anime}' if gif.anime else counted)
    return content, embed


def action_question(action: Action, actor: Any, *, returned: bool = False, can_send: bool = True) -> str:
    """The turn the bot answers when someone does *action* to it (``/hug @bot``)."""
    did = 'devolveu o que você fez com' if returned else 'usou'
    ways = 'uma resposta curta, uma reação com `add_reaction`'
    if can_send:
        ways += ', devolver a ação com `send_action_gif`' + (' (ela já está devolvendo: não precisa)' if returned else '')
    header = (
        f'[{_name(actor)} {did} /{action.key} em você — o GIF já está no canal. Reaja do seu jeito, '
        f'como alguém do chat reagiria: {ways}, ou nada ({NO_REPLY_TOKEN}).]'
    )
    return f'{header}\n---\n*{action.past} você* {action.emoji}'


# ---------------------------------------------------------------------------
# Who the tool means
# ---------------------------------------------------------------------------

_USER_REF_RE = re.compile(r'<@!?(\d+)>|(\d+)')


def _norm(text: str) -> str:
    text = unicodedata.normalize('NFKD', text or '')
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return ' '.join(text.lower().split())


def find_member(guild: Any, raw: Any) -> Any:
    """The member *raw* names: an id, ``<@id>``, a username or a display name."""
    text = str(raw or '').strip()
    if not text or guild is None:
        return None
    m = _USER_REF_RE.fullmatch(text)
    if m:
        member = guild.get_member(int(m.group(1) or m.group(2)))
        if member is not None or m.group(1):
            return member
    name = text.lstrip('@')
    member = guild.get_member_named(name)
    if member is not None:
        return member
    wanted = _norm(name)
    return next(
        (m for m in getattr(guild, 'members', None) or [] if wanted in (_norm(m.display_name), _norm(m.name))),
        None,
    )
