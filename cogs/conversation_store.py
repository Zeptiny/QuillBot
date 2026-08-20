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

from cogs.utils import BR_TZ

logger = logging.getLogger(__name__)


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

def build_history_messages(
    history: list[dict], *, max_turns: int = 16, max_images: int = 4,
) -> list[dict]:
    """Render stored turns as attributed user/assistant message pairs.

    Each user message is prefixed with ``[Por Autor (@handle) • data hora]``
    plus extra metadata (speaker change, replied-to message, message id).
    Sources used in a turn are appended to its assistant message.
    """
    messages: list[dict] = []
    prev_author_id: str | None = None
    image_budget = max_images
    for turn in history[-max_turns:]:
        meta = [f"Por {_author_label(turn.get('author'))}", _fmt_ts(turn.get('ts'))]
        author_id = (turn.get('author') or {}).get('id')
        if prev_author_id is not None and author_id and author_id != prev_author_id:
            meta.append('nova pessoa na conversa')
        prev_author_id = author_id
        if turn.get('reply_to'):
            meta.append(f"respondendo à msg {turn['reply_to']}")
        if turn.get('message_id'):
            meta.append(f"msg {turn['message_id']}")
        prefix = f"[{' • '.join(meta)}]\n"

        question = turn.get('question') or ''
        images = [u for u in turn.get('images', []) if u][:image_budget]
        if images:
            image_budget -= len(images)
            parts: list[dict] = [{'type': 'text', 'text': prefix + question}]
            for url in images:
                parts.append({'type': 'image_url', 'image_url': {'url': url}})
            messages.append({'role': 'user', 'content': parts})
        else:
            messages.append({'role': 'user', 'content': prefix + question})

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
) -> dict:
    """Build the current user message, attributed when part of a conversation."""
    urls = [u for u in (image_urls or []) if u][:4]
    parts: list[dict] | None = None
    text = question
    if in_conversation:
        meta = [f'Agora — {_author_label(author)}', _fmt_ts(ts)]
        if reply_to:
            meta.append(f'respondendo à msg {reply_to}')
        text = f"[{' • '.join(meta)}]\n{question}"
    if urls:
        parts = [{'type': 'text', 'text': text}]
        for url in urls:
            parts.append({'type': 'image_url', 'image_url': {'url': url}})
    if parts is not None:
        return {'role': 'user', 'content': parts}
    return {'role': 'user', 'content': text}


def build_conversation_block(history: list[dict], *, current_author: dict | None) -> str:
    """System-prompt block describing the ongoing conversation and participants."""
    participants: list[dict] = []
    for turn in history:
        add_participant(participants, turn.get('author') or {})
    start_ts = history[0].get('ts') if history else None
    current = _author_label(current_author)
    lines = [
        '<conversa_em_andamento>',
        f'Conversa iniciada em {_fmt_ts(start_ts)}.',
        f'Participantes: {", ".join(_author_label(p) for p in participants)}.',
        'Cada pergunta do histórico aparece prefixada com [Por Autor (@usuário) • data hora] —',
        'use esses prefixos para saber quem perguntou o quê, e de qual resposta do bot.',
        f'O turno atual foi enviado por {current}.',
        '</conversa_em_andamento>',
    ]
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
