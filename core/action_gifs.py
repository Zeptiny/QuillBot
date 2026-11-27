"""Action GIFs: an anime GIF of one person hugging, kissing, slapping… another.

Used by ``/hug``, ``/kiss``, ``/slap`` and the rest (``cogs/actions.py``), by
their Retribuir/Revidar buttons and by the ``send_action_gif`` tool, where the
bot does the action itself.

GIFs come from free anime GIF APIs, tried in each action's order: nekos.best
(which also names the anime) and otakugifs.xyz. Every distinct GIF they return
is recorded, and its file is downloaded once in the background (identical
files are kept once). When no API answers, a stored file is uploaded as an
attachment; without files, a stored URL is reused; with neither, the action
goes out as text. How many times each person did each action to each other
person is counted in the same SQLite file.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
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

# Only GIFs hosted by the APIs themselves are posted: an embed shows whatever URL it gets.
_ALLOWED_HOSTS = ('nekos.best', 'otakugifs.xyz')
_NEKOS_URL = 'https://nekos.best/api/v2/{category}'
_OTAKU_URL = 'https://api.otakugifs.xyz/gif'
# Stored files must fit Discord's upload limit on servers without boosts (10 MiB).
MAX_FILE_BYTES = 10 * 1024 * 1024
_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=60)
_GIF_MAGIC = (b'GIF87a', b'GIF89a')
# Downloads at once: a CDN refused some of six parallel ones in testing (HTTP 428).
_PARALLEL_DOWNLOADS = 2

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

    @property
    def attachment_name(self) -> str:
        """The file name a stored GIF is uploaded under."""
        return f'{self.key}.gif'


def _both(category: str) -> tuple[tuple[str, str], ...]:
    return ((NEKOS, category), (OTAKU, category))


def _nekos_only(category: str) -> tuple[tuple[str, str], ...]:
    return ((NEKOS, category),)


def _otaku_only(category: str) -> tuple[tuple[str, str], ...]:
    return ((OTAKU, category),)


ACTIONS: dict[str, Action] = {a.key: a for a in (
    Action('hug', '🤗', 'abraçou', 'Dê um abraço em alguém', AFFECTION, _both('hug')),
    Action('kiss', '😘', 'beijou', 'Dê um beijo em alguém', AFFECTION, _both('kiss')),
    Action('peck', '😚', 'deu um selinho em', 'Dê um selinho em alguém', AFFECTION, _nekos_only('peck')),
    Action(
        'blowkiss', '💋', 'mandou um beijo para', 'Mande um beijo para alguém', AFFECTION,
        ((NEKOS, 'blowkiss'), (OTAKU, 'airkiss')),
    ),
    Action('pat', '🫶', 'fez cafuné em', 'Faça cafuné em alguém', AFFECTION, _both('pat')),
    Action('cuddle', '🥰', 'se aconchegou em', 'Aconchegue-se em alguém', AFFECTION, _both('cuddle')),
    Action('nuzzle', '😊', 'deu um cheirinho em', 'Dê um cheirinho em alguém', AFFECTION, _otaku_only('nuzzle')),
    Action('lappillow', '😴', 'deitou no colo de', 'Deite no colo de alguém', AFFECTION, _nekos_only('lappillow')),
    Action('handhold', '💞', 'segurou a mão de', 'Segure a mão de alguém', AFFECTION, _both('handhold')),
    Action('handshake', '🤝', 'apertou a mão de', 'Aperte a mão de alguém', AFFECTION, _nekos_only('handshake')),
    Action('highfive', '🙌', 'bateu um toca aqui com', 'Bata um toca aqui com alguém', AFFECTION, _nekos_only('highfive')),
    Action('brofist', '🤜', 'bateu um soquinho com', 'Bata um soquinho com alguém', AFFECTION, _otaku_only('brofist')),
    Action('wave', '👋', 'acenou para', 'Acene para alguém', AFFECTION, _both('wave')),
    Action('cheers', '🍻', 'brindou com', 'Brinde com alguém', AFFECTION, _otaku_only('cheers')),
    Action('feed', '🍙', 'deu comida para', 'Dê comida para alguém', AFFECTION, _nekos_only('feed')),
    Action('wink', '😉', 'piscou para', 'Pisque para alguém', PLAYFUL, _both('wink')),
    Action('stare', '👀', 'encarou', 'Encare alguém', PLAYFUL, _both('stare')),
    Action('lick', '👅', 'lambeu', 'Lamba alguém', PLAYFUL, _otaku_only('lick')),
    Action('poke', '👉', 'cutucou', 'Cutuque alguém', PLAYFUL, _both('poke')),
    Action('pinch', '🤏', 'beliscou', 'Belisque alguém', PLAYFUL, _otaku_only('pinch')),
    Action('tickle', '🤭', 'fez cócegas em', 'Faça cócegas em alguém', PLAYFUL, _both('tickle')),
    Action('bite', '😬', 'mordeu', 'Morda alguém', PLAYFUL, _both('bite')),
    Action('bonk', '🔨', 'deu um bonk em', 'Dê um bonk em alguém', PLAYFUL, _nekos_only('bonk')),
    Action('slap', '🖐️', 'deu um tapa em', 'Dê um tapa em alguém', ROUGH, _both('slap')),
    Action('punch', '👊', 'deu um soco em', 'Dê um soco em alguém', ROUGH, _both('punch')),
    Action('kick', '🦵', 'chutou', 'Chute alguém', ROUGH, _nekos_only('kick')),
    Action('yeet', '🚀', 'arremessou', 'Arremesse alguém', ROUGH, _nekos_only('yeet')),
)}


# ---------------------------------------------------------------------------
# GIF providers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gif:
    url: str
    anime: str = ''
    # A stored copy to upload as an attachment (the APIs didn't answer).
    path: str = ''


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


def file_path(files_dir: str, sha: str) -> str:
    return os.path.join(files_dir, f'{sha}.gif')


def _write_file(files_dir: str, sha: str, data: bytes) -> None:
    os.makedirs(files_dir, exist_ok=True)
    path = file_path(files_dir, sha)
    if os.path.exists(path):
        return
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


class GifPicker:
    """A GIF for an action: its sources in order, then a stored file, then a stored URL.

    Every GIF a source returns is recorded; with *files_dir*, its file is
    downloaded in the background the first time it shows up (and again the
    next time it's served, if that failed), so that when no source answers
    one can be uploaded as an attachment.
    """

    def __init__(self, session: aiohttp.ClientSession, store: ActionStore, *, files_dir: str | None = None):
        self.session = session
        self.store = store
        self.files_dir = files_dir
        self._saving: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._downloads = asyncio.Semaphore(_PARALLEL_DOWNLOADS)

    async def pick(self, action: Action) -> Gif | None:
        for provider, category in action.sources:
            try:
                gif = await PROVIDERS[provider](self.session, category)
            except Exception as e:  # network, timeout, error status, bad JSON: try the next source
                logger.warning('[actions] %s failed for %s: %r', provider, category, e)
                continue
            if gif is not None:
                file_missing = await asyncio.to_thread(self.store.remember_gif, action.key, gif)
                if file_missing:
                    self._save_later(action.key, gif.url)
                return gif
            logger.info('[actions] %s had no GIF for %s', provider, category)
        gif = await asyncio.to_thread(self._stored_file, action.key)
        if gif is not None:
            logger.info('[actions] no API answered for %s: uploading a stored GIF', action.key)
            return gif
        gif = await asyncio.to_thread(self.store.recent_gif, action.key)
        if gif is None:
            logger.warning('[actions] no GIF for %s: sending it as text', action.key)
        return gif

    def _stored_file(self, action_key: str) -> Gif | None:
        """A stored GIF of *action_key* whose file is on disk, at random."""
        if not self.files_dir:
            return None
        rows = self.store.stored_files(action_key)
        random.shuffle(rows)
        for url, anime, sha in rows:
            path = file_path(self.files_dir, sha)
            if os.path.isfile(path):
                return Gif(url, anime, path)
            self.store.forget_file(sha)  # deleted from disk: downloaded again next time it's served
        return None

    # --- saving files ---

    def _save_later(self, action_key: str, url: str) -> None:
        if not self.files_dir or url in self._saving:
            return
        self._saving.add(url)
        task = asyncio.get_running_loop().create_task(self._save(action_key, url))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _save(self, action_key: str, url: str) -> None:
        assert self.files_dir is not None
        try:
            async with self._downloads:
                data = await self._download(url)
            if data is None:
                return
            sha = hashlib.sha256(data).hexdigest()
            await asyncio.to_thread(_write_file, self.files_dir, sha, data)
            await asyncio.to_thread(self.store.set_file, action_key, url, sha)
        except Exception as e:
            logger.warning('[actions] could not save %s: %r', url, e)
        finally:
            self._saving.discard(url)

    async def _download(self, url: str) -> bytes | None:
        """The GIF at *url*, or ``None`` when it's too big to upload or not a GIF."""
        async with self.session.get(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
            if resp.status != 200:
                raise ProviderError(f'HTTP {resp.status}')
            if (resp.content_length or 0) > MAX_FILE_BYTES:
                logger.info('[actions] %s is over %d bytes; not stored', url, MAX_FILE_BYTES)
                return None
            data = bytearray()
            async for chunk in resp.content.iter_chunked(1 << 16):
                data += chunk
                if len(data) > MAX_FILE_BYTES:
                    logger.info('[actions] %s is over %d bytes; not stored', url, MAX_FILE_BYTES)
                    return None
        if bytes(data[:6]) not in _GIF_MAGIC:
            logger.info('[actions] %s is not a GIF; not stored', url)
            return None
        return bytes(data)

    async def drain(self) -> None:
        """Wait for the downloads in progress."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()


# ---------------------------------------------------------------------------
# Counts and stored GIFs
# ---------------------------------------------------------------------------

class ActionStore:
    """SQLite (``ACTIONS_DB_PATH``): action counts per pair, and every GIF seen per action.

    A GIF row keeps its URL, the anime it's from and, once downloaded, the
    SHA-256 of its file (files are named by it, so identical GIFs share one).
    """

    def __init__(self, path: str):
        self.path = path
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
                    file TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (action, url)
                )""")
            columns = {row[1] for row in con.execute('PRAGMA table_info(action_gifs)')}
            if 'file' not in columns:  # tables made before files were stored
                con.execute("ALTER TABLE action_gifs ADD COLUMN file TEXT NOT NULL DEFAULT ''")
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

    def remember_gif(self, action: str, gif: Gif) -> bool:
        """Record *gif* for *action* (every distinct URL is kept); returns whether it has no file yet."""
        with self._lock:
            con = self._connect()
            try:
                con.execute(
                    'INSERT INTO action_gifs (action, url, anime, seen_at) VALUES (?,?,?,?) '
                    'ON CONFLICT (action, url) DO UPDATE SET seen_at = excluded.seen_at, '
                    "anime = CASE WHEN excluded.anime != '' THEN excluded.anime ELSE anime END",
                    (action, gif.url, gif.anime, time.time()),
                )
                con.commit()
                row = con.execute('SELECT file FROM action_gifs WHERE action=? AND url=?', (action, gif.url)).fetchone()
            finally:
                con.close()
        return not (row and row[0])

    def set_file(self, action: str, url: str, sha: str) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.execute('UPDATE action_gifs SET file=? WHERE action=? AND url=?', (sha, action, url))
                con.commit()
            finally:
                con.close()

    def forget_file(self, sha: str) -> None:
        """The file *sha* is gone: its GIFs count as not downloaded."""
        with self._lock:
            con = self._connect()
            try:
                con.execute("UPDATE action_gifs SET file='' WHERE file=?", (sha,))
                con.commit()
            finally:
                con.close()

    def stored_files(self, action: str) -> list[tuple[str, str, str]]:
        """``(url, anime, sha)`` of *action*'s downloaded GIFs."""
        con = self._connect()
        try:
            return con.execute(
                "SELECT url, anime, file FROM action_gifs WHERE action=? AND file != ''", (action,),
            ).fetchall()
        finally:
            con.close()

    def recent_gif(self, action: str) -> Gif | None:
        """One of *action*'s recorded GIF URLs, at random."""
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
    # A stored copy goes up with the message (see ``attachment``); the embed shows it.
    embed.set_image(url=f'attachment://{action.attachment_name}' if gif.path else gif.url)
    embed.set_footer(text=f'{counted} · Anime: {gif.anime}' if gif.anime else counted)
    return content, embed


def attachment(action: Action, gif: Gif | None) -> discord.File | None:
    """The file to upload with the message when *gif* is a stored copy."""
    if gif is None or not gif.path:
        return None
    return discord.File(gif.path, filename=action.attachment_name)


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
