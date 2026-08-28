"""Persistent, attributed conversation store for multi-turn chats.

Backs the reply-to-bot follow-up flows (``/chat`` + @mention in Commands, and
``/ask``/Spark in DocsRAG) with SQLite so conversations survive restarts.

Design
------
- A conversation is keyed by the id of the **first bot reply message**.
- Every subsequent bot reply registers a *handle* (msg_id -> conv_id), so
  replying to ANY message of a thread continues the same linear conversation
  (no branching).
- Each turn stores full attribution: author, timestamp, user message id,
  channel, images and sources used — so the LLM always knows who asked what.
- Activity-based TTL: conversations expire ``ttl_seconds`` after their **last
  activity** (read or write), not after creation, so active threads never die
  mid-conversation.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sqlite3
import time
from typing import Any

from cogs import image_store
from cogs.utils import BR_TZ
from config import CONVERSATIONS_GAP_MESSAGES, CONVERSATIONS_IMAGE_TURNS

logger = logging.getLogger(__name__)

PRIOR_CONTEXT_HEADER = 'Mensagens no canal antes desta pergunta'


# ---------------------------------------------------------------------------
# Turn / metadata helpers
# ---------------------------------------------------------------------------

def author_info(user: Any) -> dict:
    """Extract stable author info (id / handle / display name) from a user."""
    if user is None:
        return {'id': '', 'name': '', 'display': ''}
    name = getattr(user, 'name', '') or ''
    display = getattr(user, 'display_name', None) or name
    return {'id': str(getattr(user, 'id', '')), 'name': name, 'display': display}


def make_turn(
    question: str,
    answer: str,
    *,
    author: dict,
    ts: float,
    message_id: str | int | None = None,
    channel_id: str | int | None = None,
    channel_name: str | None = None,
    images: list[str] | None = None,
    sources: list[dict] | None = None,
    reply_to: str | int | None = None,
    prior_context: list[str] | None = None,
) -> dict:
    """Build a single conversation turn record."""
    return {
        'question': question,
        'answer': answer,
        'author': author or {'id': '', 'name': '', 'display': ''},
        'ts': float(ts),
        'message_id': str(message_id) if message_id else None,
        'channel_id': str(channel_id) if channel_id else None,
        'channel_name': channel_name,
        'images': [u for u in (images or []) if u][:4],
        'sources': [
            {'title': (s.get('title') or s.get('url') or '')[:100], 'url': s.get('url', '')}
            for s in (sources or [])
            if s.get('url')
        ][:6],
        'reply_to': str(reply_to) if reply_to else None,
        'prior_context': [l for l in (prior_context or []) if l][:max(30, CONVERSATIONS_GAP_MESSAGES)],
    }


def cap_turns(turns: list[dict], max_turns: int) -> list[dict]:
    """Keep only the most recent ``max_turns`` turns."""
    return turns[-max_turns:] if len(turns) > max_turns else turns


def add_participant(participants: list[dict], info: dict) -> None:
    """Add or refresh a participant entry (matched by user id)."""
    if not info.get('id'):
        return
    for p in participants:
        if p.get('id') == info['id']:
            if info.get('name'):
                p['name'] = info['name']
            if info.get('display'):
                p['display'] = info['display']
            return
    participants.append({
        'id': info['id'],
        'name': info.get('name', ''),
        'display': info.get('display', ''),
    })


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return 'data desconhecida'
    try:
        return datetime.datetime.fromtimestamp(float(ts), tz=BR_TZ).strftime('%d/%m/%Y %H:%M')
    except (ValueError, OSError, OverflowError):
        return 'data desconhecida'


def _author_label(author: dict | None) -> str:
    a = author or {}
    display = a.get('display') or a.get('name') or 'usuário'
    name = a.get('name')
    return f'{display} (@{name})' if name else display


# ---------------------------------------------------------------------------
# LLM message builders (author-attributed history)
# ---------------------------------------------------------------------------

def _prior_context_head(lines: list[str], channel_id: str | int | None) -> str:
    """Render the prior-context header (with channel when known) and its lines."""
    if not lines:
        return ''
    header = f'[{PRIOR_CONTEXT_HEADER}]'
    if channel_id:
        header = f'[{PRIOR_CONTEXT_HEADER} (channel_id={channel_id})]'
    return header + '\n' + '\n'.join(lines) + '\n\n'


def _prior_context_block(turn: dict) -> str:
    """Render the channel chatter captured before a turn's question, if any."""
    prior = [l for l in (turn.get('prior_context') or []) if l]
    return _prior_context_head(prior, turn.get('channel_id'))


def _split_images(images: list[str]) -> tuple[list[dict], int]:
    """(inline data-URI parts, count not inlined) for stored image refs/URLs."""
    parts: list[dict] = []
    dropped = 0
    for img in images:
        part = image_store.image_part(img) if image_store.is_image_ref(img) else None
        if part is not None:
            parts.append(part)
        else:
            dropped += 1
    return parts, dropped


def _append_image_marker(text: str, count: int) -> str:
    marker = image_store.image_marker(count)
    return f'{text}\n\n{marker}' if text else marker


def build_history_messages(
    history: list[dict], *, max_turns: int = 16, max_images: int = 4,
    image_turns: int = CONVERSATIONS_IMAGE_TURNS,
) -> list[dict]:
    """Render stored turns as attributed user/assistant message pairs.

    Each user message is prefixed with ``[Por Autor (@handle) • author_id=… •
    data hora]`` plus extra metadata: speaker change, ``↩ reply_to=`` and
    ``msg_id=``. Channel chatter captured between turns (``prior_context``) is
    injected above the question as a ``[Mensagens no canal antes desta
    pergunta]`` block, tagged with the channel when known. Sources used in a
    turn are appended to its assistant message.

    Images are inlined as base64 data URIs only for the last ``image_turns``
    turns (within ``max_images``); older turns — and turns whose images are
    legacy URLs or missing files — get a text marker instead. Raw URLs are
    never sent: some providers hang or reject URL-based ``image_url`` parts.
    """
    turns = history[-max_turns:]
    inline_from = max(0, len(turns) - image_turns)
    messages: list[dict] = []
    prev_author_id: str | None = None
    image_budget = max_images
    for idx, turn in enumerate(turns):
        author = turn.get('author') or {}
        meta = [f"Por {_author_label(author)}"]
        if author.get('id'):
            meta.append(f"author_id={author['id']}")
        meta.append(_fmt_ts(turn.get('ts')))
        author_id = author.get('id')
        if prev_author_id is not None and author_id and author_id != prev_author_id:
            meta.append('nova pessoa na conversa')
        prev_author_id = author_id
        if turn.get('reply_to'):
            meta.append(f"↩ reply_to={turn['reply_to']}")
        if turn.get('message_id'):
            meta.append(f"msg_id={turn['message_id']}")
        prefix = f"[{' • '.join(meta)}]\n"

        question = turn.get('question') or ''
        head = _prior_context_block(turn)
        images = [u for u in turn.get('images', []) if u]
        parts: list[dict] | None = None
        if images:
            inline: list[dict] = []
            if idx >= inline_from and image_budget > 0:
                inline, _ = _split_images(images[:image_budget])
                image_budget -= len(inline)
            dropped = len(images) - len(inline)
            if dropped:
                question = _append_image_marker(question, dropped)
            if inline:
                parts = [{'type': 'text', 'text': head + prefix + question}, *inline]
        if parts is not None:
            messages.append({'role': 'user', 'content': parts})
        else:
            messages.append({'role': 'user', 'content': head + prefix + question})

        answer = turn.get('answer') or ''
        sources = turn.get('sources') or []
        if sources:
            cited = '; '.join(f"{s['title']} ({s['url']})" for s in sources[:4])
            answer = f'{answer}\n\n[Fontes consultadas neste turno: {cited}]'
        messages.append({'role': 'assistant', 'content': answer})
    return messages


def build_current_message(
    question: str,
    *,
    author: dict | None,
    ts: float | None,
    image_urls: list[str] | None = None,
    reply_to: str | int | None = None,
    in_conversation: bool = False,
    prior_context: list[str] | None = None,
    channel_id: str | int | None = None,
    context_blocks: str | None = None,
) -> dict:
    """Build the current user message, attributed when part of a conversation.

    ``context_blocks`` carries per-request context (current time/place, memory
    selection, recent channel window). Keeping it in the *tail* message —
    instead of the system prompt — leaves the history prefix byte-identical
    across follow-ups, which makes provider prefix caching effective.
    """
    images = [u for u in (image_urls or []) if u][:4]
    parts: list[dict] | None = None
    head = (f"{context_blocks}\n\n" if context_blocks else '') + _prior_context_head(
        [l for l in (prior_context or []) if l], channel_id
    )
    text = question
    if in_conversation:
        meta = [f'Agora — {_author_label(author)}']
        if (author or {}).get('id'):
            meta.append(f"author_id={author['id']}")
        meta.append(_fmt_ts(ts))
        if reply_to:
            meta.append(f'↩ reply_to={reply_to}')
        text = f"{head}[{' • '.join(meta)}]\n{question}"
    elif head:
        text = f"{head}{question}"
    if images:
        inline, dropped = _split_images(images)
        if dropped:
            text = _append_image_marker(text, dropped)
        if inline:
            parts = [{'type': 'text', 'text': text}, *inline]
    if parts is not None:
        return {'role': 'user', 'content': parts}
    return {'role': 'user', 'content': text}


def build_conversation_block(history: list[dict]) -> str:
    """System-prompt block describing the ongoing conversation and participants.

    Deliberately **static per conversation**: it depends only on stored history,
    never on the current request's clock or current sender.  This lets the
    rendered history that follows it stay byte-identical across follow-ups so
    providers can prefix-cache it.  The current sender is instead signalled by
    the ``[Agora — …]`` prefix of the current message.
    """
    participants: list[dict] = []
    for turn in history:
        add_participant(participants, turn.get('author') or {})
    start_ts = history[0].get('ts') if history else None
    last_channel_id = history[-1].get('channel_id') if history else None
    lines = [
        '<conversa_em_andamento>',
        f'Conversa iniciada em {_fmt_ts(start_ts)}.',
        f'Participantes: {", ".join(_author_label(p) for p in participants)}.',
    ]
    if last_channel_id:
        lines.append(f'Canal atual da conversa: channel_id={last_channel_id} (use com get_message_context).')
    lines.extend([
        'Cada pergunta do histórico aparece prefixada com [Por Autor (@usuário) • author_id=… • data hora] —',
        'use esses prefixos para saber quem perguntou o quê, e de qual resposta do bot.',
        'Mensagens do canal (ferramentas de histórico) usam o mesmo author_id/msg_id —',
        'correlacione-as para saber quem disse o quê na conversa do canal.',
        f'Blocos "[{PRIOR_CONTEXT_HEADER}]" trazem a conversa do canal',
        'que aconteceu entre as perguntas direcionadas ao bot.',
        'Nos blocos de mensagens do canal, respostas do bot e perguntas já registradas como '
        'turnos são omitidas; use get_channel_history para vê-las.',
        '</conversa_em_andamento>',
    ])
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# SQLite-backed store
# ---------------------------------------------------------------------------

class ConversationStore:
    """SQLite-backed conversation store with activity-based TTL.

    Each instance serves one ``kind`` (e.g. ``'chat'`` or ``'ask'``) so multiple
    cogs can share a database file without their message-id handles colliding.
    """

    def __init__(
        self,
        db_path: str,
        *,
        kind: str,
        ttl_seconds: float = 1800.0,
        max_stored: int = 200,
    ):
        self.db_path = db_path
        self.kind = kind
        self.ttl_seconds = ttl_seconds
        self.max_stored = max_stored
        self._locks: dict[str, asyncio.Lock] = {}
        self._ensure_db()

    # --- sync internals (invoked via asyncio.to_thread) ---

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _ensure_db(self) -> None:
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        con = self._connect()
        try:
            try:
                con.execute('PRAGMA journal_mode=WAL;')
                con.execute('PRAGMA synchronous=NORMAL;')
            except sqlite3.Error:
                pass
            con.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                guild_id TEXT,
                channel_id TEXT,
                started_ts REAL NOT NULL,
                last_active_ts REAL NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (kind, conv_id)
            )""")
            con.execute(
                'CREATE INDEX IF NOT EXISTS idx_conversations_active '
                'ON conversations(kind, last_active_ts)'
            )
            con.execute("""
            CREATE TABLE IF NOT EXISTS conv_handles (
                kind TEXT NOT NULL,
                msg_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                PRIMARY KEY (kind, msg_id)
            )""")
            con.execute(
                'CREATE INDEX IF NOT EXISTS idx_conv_handles_conv '
                'ON conv_handles(kind, conv_id)'
            )
            con.commit()
        finally:
            con.close()

    def _prune(self, con: sqlite3.Connection) -> None:
        cutoff = time.time() - self.ttl_seconds
        con.execute(
            'DELETE FROM conversations WHERE kind=? AND last_active_ts < ?',
            (self.kind, cutoff),
        )
        con.execute(
            'DELETE FROM conversations WHERE kind=? AND conv_id NOT IN ('
            'SELECT conv_id FROM conversations WHERE kind=? '
            'ORDER BY last_active_ts DESC LIMIT ?)',
            (self.kind, self.kind, self.max_stored),
        )
        con.execute(
            'DELETE FROM conv_handles WHERE kind=? AND conv_id NOT IN '
            '(SELECT conv_id FROM conversations WHERE kind=?)',
            (self.kind, self.kind),
        )

    def _get_by_handle(self, msg_id: str) -> dict | None:
        now = time.time()
        con = self._connect()
        try:
            cur = con.execute(
                'SELECT c.conv_id, c.data, c.last_active_ts FROM conv_handles h '
                'JOIN conversations c ON c.kind = h.kind AND c.conv_id = h.conv_id '
                'WHERE h.kind = ? AND h.msg_id = ?',
                (self.kind, str(msg_id)),
            )
            row = cur.fetchone()
            if row is None:
                return None
            conv_id, data_json, last_active = row
            if now - last_active > self.ttl_seconds:
                con.execute(
                    'DELETE FROM conversations WHERE kind=? AND conv_id=?',
                    (self.kind, conv_id),
                )
                con.execute(
                    'DELETE FROM conv_handles WHERE kind=? AND conv_id=?',
                    (self.kind, conv_id),
                )
                con.commit()
                return None
            con.execute(
                'UPDATE conversations SET last_active_ts=? WHERE kind=? AND conv_id=?',
                (now, self.kind, conv_id),
            )
            con.commit()
            try:
                data = json.loads(data_json)
            except (json.JSONDecodeError, TypeError):
                logger.exception('Corrupt conversation data for conv %s', conv_id)
                return None
            return {'conv_id': conv_id, 'data': data}
        finally:
            con.close()

    def _create(
        self,
        conv_id: str,
        *,
        guild_id: str | None,
        channel_id: str | None,
        data: dict,
    ) -> None:
        now = time.time()
        started = float(data.get('started_ts') or now)
        con = self._connect()
        try:
            con.execute(
                'INSERT OR REPLACE INTO conversations '
                '(conv_id, kind, guild_id, channel_id, started_ts, last_active_ts, data) '
                'VALUES (?,?,?,?,?,?,?)',
                (
                    str(conv_id), self.kind, guild_id, channel_id,
                    started, now, json.dumps(data, ensure_ascii=False),
                ),
            )
            con.execute(
                'INSERT OR REPLACE INTO conv_handles (kind, msg_id, conv_id) VALUES (?,?,?)',
                (self.kind, str(conv_id), str(conv_id)),
            )
            self._prune(con)
            con.commit()
        finally:
            con.close()

    def _update(
        self,
        conv_id: str,
        data: dict,
        *,
        new_handle_msg_id: str | None,
    ) -> None:
        now = time.time()
        conv_id = str(conv_id)
        con = self._connect()
        try:
            cur = con.execute(
                'SELECT 1 FROM conversations WHERE kind=? AND conv_id=?',
                (self.kind, conv_id),
            )
            if cur.fetchone() is None:
                origin = data.get('origin', {})
                started = float(data.get('started_ts') or now)
                con.execute(
                    'INSERT INTO conversations '
                    '(conv_id, kind, guild_id, channel_id, started_ts, last_active_ts, data) '
                    'VALUES (?,?,?,?,?,?,?)',
                    (
                        conv_id, self.kind,
                        origin.get('guild_id'), origin.get('channel_id'),
                        started, now, json.dumps(data, ensure_ascii=False),
                    ),
                )
            else:
                con.execute(
                    'UPDATE conversations SET data=?, last_active_ts=? '
                    'WHERE kind=? AND conv_id=?',
                    (json.dumps(data, ensure_ascii=False), now, self.kind, conv_id),
                )
            if new_handle_msg_id:
                con.execute(
                    'INSERT OR REPLACE INTO conv_handles (kind, msg_id, conv_id) VALUES (?,?,?)',
                    (self.kind, str(new_handle_msg_id), conv_id),
                )
            self._prune(con)
            con.commit()
        finally:
            con.close()

    def _count_active(self) -> int:
        cutoff = time.time() - self.ttl_seconds
        con = self._connect()
        try:
            cur = con.execute(
                'SELECT COUNT(*) FROM conversations WHERE kind=? AND last_active_ts >= ?',
                (self.kind, cutoff),
            )
            return int(cur.fetchone()[0])
        finally:
            con.close()

    # --- async public API ---

    def conversation_lock(self, conv_id: str | int) -> asyncio.Lock:
        """Serialize follow-up processing per conversation so concurrent replies can't lose turns (last-write-wins)."""
        key = str(conv_id)
        lock = self._locks.get(key)
        if lock is not None:
            return lock
        if len(self._locks) > 1024:
            for stale_key, stale_lock in list(self._locks.items()):
                if not stale_lock.locked():
                    del self._locks[stale_key]
        lock = asyncio.Lock()
        self._locks[key] = lock
        return lock

    async def get_by_handle(self, msg_id: str | int) -> dict | None:
        """Look up a conversation by bot-reply message id (touches TTL)."""
        return await asyncio.to_thread(self._get_by_handle, str(msg_id))

    async def create(
        self,
        conv_id: str | int,
        *,
        guild_id: str | None = None,
        channel_id: str | None = None,
        data: dict | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._create, str(conv_id),
            guild_id=guild_id, channel_id=channel_id, data=data or {},
        )

    async def update(
        self,
        conv_id: str | int,
        data: dict,
        *,
        new_handle_msg_id: str | int | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._update, str(conv_id), data,
            new_handle_msg_id=str(new_handle_msg_id) if new_handle_msg_id else None,
        )

    async def count_active(self) -> int:
        return await asyncio.to_thread(self._count_active)
