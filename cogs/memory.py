"""Persistent bot memory — atomic, self-maintaining recall across conversations.

Memories are atomic one-sentence facts scoped to the guild (``subject=''``) or
to a single user (``subject=<user id>``). The LLM writes them inline via the
``memory_write`` tool (create/update/forget, dedupe-guarded) and reads them via
``memory_search`` / ``memory_about``. Beyond tool pulls, every LLM turn gets an
automatically injected ``<memory>`` block: pinned memories always ride along
and the current question semantically selects the most relevant ones.

Every mutation funnels through :class:`MemoryStore`, which applies the change
and records a before/after snapshot in ``memory_history`` inside a single
transaction, then mirrors it to a configurable admin log channel with a Revert
button. Reads update ``recall_count`` / ``last_recalled_at`` passively.

Privacy: user-scoped memories are only injected (or returned by search) when
their subject is part of the current conversation — speaker, mentioned users or
reply targets. Users can inspect and purge their own memories at any time.
"""

from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import os
import re
import sqlite3
import time
import unicodedata
from collections import deque

import discord
import numpy as np
from discord import app_commands
from discord.ext import commands

from cogs.utils import PaginatedEmbedView
from config import (
    LORE_DB_PATH,
    MEMORY_BOT_WRITE_LIMIT,
    MEMORY_DB_PATH,
    MEMORY_DEDUPE_THRESHOLD,
    MEMORY_ENABLED,
    MEMORY_INJECT_LIMIT,
    MEMORY_LOG_CHANNEL_ID,
    MEMORY_MAX_PER_SUBJECT,
    MEMORY_PIN_LIMIT,
    MEMORY_SEMANTIC_MIN_SCORE,
)

logger = logging.getLogger(__name__)

_KIND_LABELS = {
    'fact': 'Fato',
    'preference': 'Preferência',
    'person': 'Pessoa',
    'event': 'Evento',
    'skill': 'Habilidade',
}
MEMORY_KINDS = tuple(_KIND_LABELS)
_KIND_CHOICES = [
    app_commands.Choice(name=label, value=key) for key, label in _KIND_LABELS.items()
]
_DEFAULT_IMPORTANCE = {'person': 4, 'preference': 4, 'fact': 3, 'event': 3, 'skill': 3}
_ACTION_VERBS = {
    'create': 'criada',
    'update': 'atualizada',
    'forget': 'arquivada',
    'revert': 'revertida',
    'restore': 'restaurada',
    'pin': 'fixada',
    'unpin': 'desafixada',
    'purge': 'apagada',
}
_SNAPSHOT_FIELDS = ('subject', 'kind', 'content', 'importance', 'pinned', 'origin', 'status')
_MAX_CONTENT = 500
_MIN_CONTENT = 10
_MAX_REASON = 200
# Everything pinned up to the per-guild cap must actually inject — otherwise an
# accepted pin would silently never appear in conversations.
_MAX_PINNED_INJECT = MEMORY_PIN_LIMIT

MEMORY_SEARCH_TOOL = {
    'type': 'function',
    'function': {
        'name': 'memory_search',
        'description': (
            'Busca nas memórias persistentes do bot sobre este servidor e seus '
            'membros (preferências, fatos, pessoas, eventos, habilidades). '
            'Use quando o bloco <memory> injetado não cobrir o que você precisa — '
            'ex: "o que o bot sabe sobre X?", "quando aconteceu Y?". '
            'Memórias sobre um usuário só aparecem se ele participou da conversa '
            '(autor da pergunta, mencionado ou alvo de resposta).'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Tema, nome ou pergunta sobre o que recordar.',
                },
                'kind': {
                    'type': 'string',
                    'enum': list(MEMORY_KINDS),
                    'description': 'Filtrar por tipo de memória (opcional).',
                },
                'about_user': {
                    'type': 'string',
                    'description': 'ID ou nome do usuário para buscar memórias sobre ele (opcional).',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Número máximo de memórias (padrão 4, máximo 8).',
                },
            },
            'required': ['query'],
        },
    },
}

MEMORY_WRITE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'memory_write',
        'description': (
            'Escreve nas memórias persistentes do bot. Cada memória é UMA frase '
            'autocontida em terceira pessoa (ex: "nyuu prefere Paper a Fabric"). '
            'Salve preferências estáveis, fatos sobre pessoas, eventos marcantes e '
            'habilidades ensinadas — quando descobrir algo explicitamente dito ou '
            'comprovado no histórico. Use action=update quando um fato mudar, e '
            'action=forget quando deixar de valer. NÃO salve: humor do momento, '
            'tópicos técnicos já cobertos pela documentação, dados sensíveis ou '
            'coisas que a pessoa pediu para não lembrar. Use pinned=true APENAS '
            'para fatos centrais e permanentes (identidade do dono, regras do '
            'servidor, preferências duradouras) — memórias fixadas entram em toda '
            'conversa.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'action': {
                    'type': 'string',
                    'enum': ['create', 'update', 'forget'],
                    'description': 'create: nova memória; update: editar existente; forget: arquivar (reversível).',
                },
                'memory_id': {
                    'type': 'integer',
                    'description': 'ID da memória (para update/forget). Alternativa a content_match.',
                },
                'content': {
                    'type': 'string',
                    'description': 'A frase da memória — obrigatória em create; em update, o novo texto.',
                },
                'content_match': {
                    'type': 'string',
                    'description': 'Para update/forget sem memory_id: trecho que identifica a memória existente.',
                },
                'kind': {
                    'type': 'string',
                    'enum': list(MEMORY_KINDS),
                    'description': 'Tipo da memória (padrão: fact).',
                },
                'about_user': {
                    'type': 'string',
                    'description': 'ID ou nome do usuário sobre quem é a memória. Omita para fatos do servidor.',
                },
                'importance': {
                    'type': 'integer',
                    'description': 'Importância 1-5 (padrão por tipo; preferências e pessoas = 4).',
                },
                'pinned': {
                    'type': 'boolean',
                    'description': 'Fixar a memória (injeta em toda conversa). Limite rígido — use com parcimônia.',
                },
                'reason': {
                    'type': 'string',
                    'description': 'Justificativa curta, uma linha. Sempre obrigatória.',
                },
            },
            'required': ['action', 'reason'],
        },
    },
}

MEMORY_ABOUT_TOOL = {
    'type': 'function',
    'function': {
        'name': 'memory_about',
        'description': (
            'Perfil compacto do que o bot lembra sobre um usuário: memórias ativas '
            'agrupadas por tipo, ordenadas por importância. Use para contextualizar '
            'quem é alguém antes de responder — mais barato que memory_search para '
            'isso. Só funciona para o autor da pergunta, mencionados ou alvos de '
            'resposta na conversa atual.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'user': {
                    'type': 'string',
                    'description': 'ID ou nome do usuário (padrão: autor da pergunta atual).',
                },
            },
            'required': [],
        },
    },
}


class MemoryError(Exception):
    """User-facing memory operation error."""


def _norm(text: str) -> str:
    text = unicodedata.normalize('NFKD', text or '')
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', text).strip().lower()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _snippet(text: str, n: int = 350) -> str:
    text = (text or '').strip()
    return text if len(text) <= n else text[:n] + '…'


def _fmt_brt(ts_iso: str | None) -> str:
    if not ts_iso:
        return '—'
    try:
        dt = datetime.datetime.fromisoformat(ts_iso.replace('Z', '+00:00'))
        return dt.astimezone(datetime.timezone(datetime.timedelta(hours=-3))).strftime('%d/%m/%Y %H:%M')
    except Exception:
        return ts_iso[:19]


def _parse_importance(raw, default: int) -> int:
    try:
        return max(1, min(5, int(raw)))
    except (TypeError, ValueError):
        return default


def _parse_bool(raw, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ('true', '1', 'yes')


def _entry_from_row(r: sqlite3.Row) -> dict:
    embedding = None
    if r['embedding']:
        try:
            embedding = np.frombuffer(r['embedding'], dtype=np.float32)
        except ValueError:
            embedding = None
    return {
        'id': r['id'],
        'guild_id': r['guild_id'],
        'subject': r['subject'] or '',
        'kind': r['kind'],
        'content': r['content'],
        'importance': r['importance'],
        'pinned': bool(r['pinned']),
        'pinned_by': r['pinned_by'],
        'origin': r['origin'],
        'recall_count': r['recall_count'],
        'last_recalled_at': r['last_recalled_at'],
        'status': r['status'],
        'created_by': r['created_by'],
        'updated_by': r['updated_by'],
        'created_at': r['created_at'],
        'updated_at': r['updated_at'],
        'embedding': embedding,
    }


def _snapshot(entry: dict) -> dict:
    return {k: entry[k] for k in _SNAPSHOT_FIELDS}


def _subject_key(raw, guild: discord.Guild | None, participants: set[str] | None = None) -> str:
    """Resolve an about_user argument to a subject key ('' = guild-wide).

    Falls back to matching participant names (via the guild member cache) so a
    display name said in conversation still resolves to the right user id.
    Returns 'name:<x>' only when the user cannot be identified at all.
    """
    raw = (raw or '').strip()
    if not raw:
        return ''
    if re.fullmatch(r'\d{15,25}', raw):
        return raw
    if participants and raw in participants:
        return raw
    target = _norm(raw)
    if guild is not None:
        member = guild.get_member_named(raw)
        if member is not None:
            return str(member.id)
        for m in guild.members:
            if target in (_norm(m.display_name), _norm(m.name)):
                return str(m.id)
        for pid in participants or ():
            m = guild.get_member(int(pid)) if pid.isdigit() else None
            if m is not None and target in (_norm(m.display_name), _norm(m.name)):
                return str(pid)
    return f'name:{target}'


def _cosine(a: np.ndarray, b: np.ndarray, b_norm: float | None = None) -> float:
    if a.shape[0] != b.shape[0]:
        return 0.0
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b)) if b_norm is None else b_norm
    denom = an * bn
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class MemoryStore:
    """SQLite store — the single mutation path, memory write + history in one transaction."""

    def __init__(self, path: str):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def ensure(self) -> None:
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        con = self._connect()
        try:
            try:
                con.execute('PRAGMA journal_mode=WAL;')
            except Exception:
                pass
            con.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    subject TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL DEFAULT 'fact',
                    content TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 3,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    pinned_by TEXT NOT NULL DEFAULT '',
                    origin TEXT NOT NULL DEFAULT '',
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    last_recalled_at TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_by TEXT NOT NULL DEFAULT '',
                    updated_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    embedding BLOB
                )
            """)
            con.execute('CREATE INDEX IF NOT EXISTS idx_mem_guild ON memories(guild_id, status)')
            con.execute('CREATE INDEX IF NOT EXISTS idx_mem_subject ON memories(guild_id, subject)')
            con.execute("""
                CREATE TABLE IF NOT EXISTS memory_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    actor_name TEXT NOT NULL DEFAULT '',
                    before TEXT,
                    after TEXT,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            con.execute('CREATE INDEX IF NOT EXISTS idx_mem_hist_memory ON memory_history(memory_id)')
            con.execute('CREATE INDEX IF NOT EXISTS idx_mem_hist_guild ON memory_history(guild_id)')
            con.execute("""
                CREATE TABLE IF NOT EXISTS memory_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            con.commit()
        finally:
            con.close()

    def list_memories(self, guild_id: int, *, subject: str | None = None) -> list[dict]:
        con = self._connect()
        try:
            if subject is None:
                rows = con.execute(
                    'SELECT * FROM memories WHERE guild_id=? ORDER BY pinned DESC, importance DESC, id DESC',
                    (guild_id,),
                ).fetchall()
            else:
                rows = con.execute(
                    'SELECT * FROM memories WHERE guild_id=? AND subject=? '
                    'ORDER BY pinned DESC, importance DESC, id DESC',
                    (guild_id, subject),
                ).fetchall()
        finally:
            con.close()
        return [_entry_from_row(r) for r in rows]

    def get_memory(self, guild_id: int, memory_id: int) -> dict | None:
        con = self._connect()
        try:
            row = con.execute(
                'SELECT * FROM memories WHERE id=? AND guild_id=?',
                (memory_id, guild_id),
            ).fetchone()
        finally:
            con.close()
        return _entry_from_row(row) if row else None

    def active_for_subjects(self, guild_id: int, subjects: list[str]) -> list[dict]:
        """Active memories where subject is '' (guild) or in *subjects*."""
        if not subjects:
            subjects = ['']
        placeholders = ', '.join('?' for _ in subjects)
        con = self._connect()
        try:
            rows = con.execute(
                f'SELECT * FROM memories WHERE guild_id=? AND status=\'active\' '
                f'AND (subject=\'\' OR subject IN ({placeholders}))',
                (guild_id, *subjects),
            ).fetchall()
        finally:
            con.close()
        return [_entry_from_row(r) for r in rows]

    def count_pins(self, guild_id: int) -> int:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE guild_id=? AND status='active' AND pinned=1",
                (guild_id,),
            ).fetchone()
            return int(row['n'])
        finally:
            con.close()

    def count_subject(self, guild_id: int, subject: str) -> int:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE guild_id=? AND subject=? AND status='active'",
                (guild_id, subject),
            ).fetchone()
            return int(row['n'])
        finally:
            con.close()

    def create_with_history(
        self, guild_id: int, *, subject: str, kind: str, content: str,
        importance: int, pinned: bool, pinned_by: str, origin: str,
        actor: str, actor_name: str, reason: str, embedding=None,
        status: str = 'active', created_at: str | None = None,
    ) -> tuple[dict, int]:
        now = created_at or _now_iso()
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        con = self._connect()
        try:
            cur = con.execute(
                'INSERT INTO memories '
                '(guild_id, subject, kind, content, importance, pinned, pinned_by, origin, '
                'recall_count, last_recalled_at, status, created_by, updated_by, created_at, updated_at, embedding) '
                'VALUES (?,?,?,?,?,?,?,?,0,NULL,?,?,?,?,?,?)',
                (
                    guild_id, subject, kind, content, importance,
                    int(pinned), pinned_by, origin, status, actor, actor, now, now, blob,
                ),
            )
            memory_id = cur.lastrowid
            row = con.execute('SELECT * FROM memories WHERE id=?', (memory_id,)).fetchone()
            entry = _entry_from_row(row)
            after = _snapshot(entry)
            hist = con.execute(
                'INSERT INTO memory_history '
                '(memory_id, guild_id, action, actor, actor_name, before, after, reason, created_at) '
                'VALUES (?,?,?,?,?,?,?,?,?)',
                (memory_id, guild_id, 'create', actor, actor_name, None,
                 json.dumps(after, ensure_ascii=False), reason, now),
            )
            con.commit()
            return entry, hist.lastrowid or 0
        except Exception:
            try:
                con.rollback()
            except Exception:
                pass
            raise
        finally:
            con.close()

    def update_with_history(
        self, guild_id: int, memory_id: int, patch: dict, *,
        action: str, actor: str, actor_name: str, reason: str, embedding=None,
    ) -> tuple[dict, dict, int]:
        """Apply *patch* and record the history row in one transaction.

        The before-snapshot is derived from a fresh read inside the transaction,
        so concurrent writers cannot be silently overwritten by stale state.
        Returns (updated, before, history_id).
        """
        cols = {'subject', 'kind', 'content', 'importance', 'pinned', 'pinned_by', 'origin', 'status'}
        con = self._connect()
        try:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute(
                'SELECT * FROM memories WHERE id=? AND guild_id=?',
                (memory_id, guild_id),
            ).fetchone()
            if row is None:
                con.rollback()
                raise MemoryError('Memória não encontrada.')
            current = _entry_from_row(row)
            before = _snapshot(current)
            sets: list[str] = ['updated_by=?', 'updated_at=?']
            vals: list = [actor, _now_iso()]
            for k, v in patch.items():
                if k not in cols:
                    continue
                if k in ('pinned',):
                    v = int(bool(v))
                sets.append(f'{k}=?')
                vals.append(v)
            if embedding is not None:
                sets.append('embedding=?')
                vals.append(embedding.astype(np.float32).tobytes())
            vals.extend([memory_id, guild_id])
            con.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id=? AND guild_id=?",
                vals,
            )
            row2 = con.execute('SELECT * FROM memories WHERE id=?', (memory_id,)).fetchone()
            updated = _entry_from_row(row2)
            hist = con.execute(
                'INSERT INTO memory_history '
                '(memory_id, guild_id, action, actor, actor_name, before, after, reason, created_at) '
                'VALUES (?,?,?,?,?,?,?,?,?)',
                (
                    memory_id, guild_id, action, actor, actor_name,
                    json.dumps(before, ensure_ascii=False),
                    json.dumps(_snapshot(updated), ensure_ascii=False),
                    reason, _now_iso(),
                ),
            )
            con.commit()
            return updated, before, hist.lastrowid or 0
        except sqlite3.IntegrityError as e:
            try:
                con.rollback()
            except Exception:
                pass
            raise MemoryError('Conflito ao gravar a memória.') from e
        except Exception:
            try:
                con.rollback()
            except Exception:
                pass
            raise
        finally:
            con.close()

    def mark_recalled(self, guild_id: int, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        now = _now_iso()
        con = self._connect()
        try:
            con.execute(
                'UPDATE memories SET recall_count=recall_count+1, last_recalled_at=? '
                f'WHERE guild_id=? AND id IN ({", ".join("?" for _ in memory_ids)})',
                (now, guild_id, *memory_ids),
            )
            con.commit()
        except Exception:
            logger.exception('[memory] mark_recalled failed')
        finally:
            con.close()

    def count_subjects_all(self, guild_id: int, subjects: list[str]) -> int:
        if not subjects:
            return 0
        placeholders = ', '.join('?' for _ in subjects)
        con = self._connect()
        try:
            row = con.execute(
                f'SELECT COUNT(*) AS n FROM memories WHERE guild_id=? AND subject IN ({placeholders})',
                (guild_id, *subjects),
            ).fetchone()
            return int(row['n'])
        finally:
            con.close()

    def purge_subjects(self, guild_id: int, subjects: list[str]) -> int:
        """Hard-delete every memory (and history) for the given subjects. Privacy path."""
        if not subjects:
            return 0
        placeholders = ', '.join('?' for _ in subjects)
        con = self._connect()
        try:
            con.execute('BEGIN IMMEDIATE')
            ids = [r['id'] for r in con.execute(
                f'SELECT id FROM memories WHERE guild_id=? AND subject IN ({placeholders})',
                (guild_id, *subjects),
            ).fetchall()]
            for mid in ids:
                con.execute('DELETE FROM memory_history WHERE memory_id=?', (mid,))
                con.execute('DELETE FROM memories WHERE id=?', (mid,))
            con.commit()
            return len(ids)
        except Exception:
            try:
                con.rollback()
            except Exception:
                pass
            raise
        finally:
            con.close()

    def get_history(self, guild_id: int, history_id: int) -> sqlite3.Row | None:
        con = self._connect()
        try:
            return con.execute(
                'SELECT * FROM memory_history WHERE id=? AND guild_id=?',
                (history_id, guild_id),
            ).fetchone()
        finally:
            con.close()

    def history_for_memory(self, guild_id: int, memory_id: int, limit: int = 8) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                'SELECT id, action, actor_name, reason, created_at FROM memory_history '
                'WHERE memory_id=? AND guild_id=? ORDER BY id DESC LIMIT ?',
                (memory_id, guild_id, limit),
            ).fetchall()
        finally:
            con.close()
        return [dict(r) for r in rows]

    def history_for_guild(self, guild_id: int) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                'SELECT * FROM memory_history WHERE guild_id=? ORDER BY id ASC',
                (guild_id,),
            ).fetchall()
        finally:
            con.close()
        return [dict(r) for r in rows]


class MemoryLogView(discord.ui.View):
    """Revert button attached to log-channel mutation embeds."""

    def __init__(self, cog: 'Memory', guild_id: int, memory_id: int, history_id: int):
        super().__init__(timeout=1800)
        self.cog = cog
        self.guild_id = guild_id
        self.memory_id = memory_id
        self.history_id = history_id

    def _allowed(self, interaction: discord.Interaction) -> bool:
        if not isinstance(interaction.user, discord.Member):
            return False
        if interaction.guild_id != self.guild_id:
            return False
        return (
            interaction.user.guild_permissions.administrator
            or interaction.user.guild_permissions.manage_guild
        )

    @discord.ui.button(label='↩️ Reverter', style=discord.ButtonStyle.secondary)
    async def revert_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._allowed(interaction):
            return await interaction.response.send_message(
                'Apenas administradores deste servidor podem usar este botão.', ephemeral=True
            )
        try:
            entry = await self.cog.revert(
                self.guild_id, self.history_id,
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except MemoryError as e:
            return await self._done(interaction, f'⚠️ {e}')
        except Exception:
            logger.exception('[memory] revert button failed')
            return await self._done(interaction, '⚠️ Erro interno ao reverter.')
        await self._done(
            interaction,
            f'↩️ Revertido por {interaction.user.display_name} — memória #{entry["id"]} restaurada.',
        )

    async def _done(self, interaction: discord.Interaction, message: str) -> None:
        for child in self.children:
            child.disabled = True
        try:
            await interaction.response.edit_message(content=message, view=self, embeds=[])
        except discord.HTTPException:
            try:
                await interaction.followup.send(message)
            except Exception:
                pass


class ConfirmView(discord.ui.View):
    def __init__(self, confirm_label: str = 'Confirmar'):
        super().__init__(timeout=60)
        self.children[0].label = confirm_label
        self.confirmed = False

    @discord.ui.button(label='Confirmar', style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = True
        self.stop()
        await interaction.response.defer()
        for child in self.children:
            child.disabled = True
        await interaction.edit_original_response(view=self)

    @discord.ui.button(label='Cancelar', style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = False
        self.stop()
        await interaction.response.defer()
        for child in self.children:
            child.disabled = True
        await interaction.edit_original_response(view=self)


def _is_admin(interaction: discord.Interaction) -> bool:
    return isinstance(interaction.user, discord.Member) and (
        interaction.user.guild_permissions.administrator
        or interaction.user.guild_permissions.manage_guild
    )


def _subject_keys_for_member(member) -> list[str]:
    """Every subject key a member's memories could live under (id + name fallbacks)."""
    keys = {str(member.id)}
    for n in (getattr(member, 'name', ''), getattr(member, 'display_name', '')):
        if n:
            keys.add(f'name:{_norm(n)}')
    return sorted(keys)


class Memory(commands.Cog, name='Memory'):
    """Persistent bot memory: auto-injection, inline LLM writes, admin control."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.store = MemoryStore(MEMORY_DB_PATH)
        self._bot_writes: dict[int, deque[float]] = {}
        self._log_tasks: set[asyncio.Task] = set()

    async def cog_load(self):
        try:
            await asyncio.to_thread(self.store.ensure)
            await asyncio.to_thread(self._migrate_lore)
        except Exception:
            logger.exception("Memory DB init failed — memory tools will degrade to error messages")

    # --- One-time lore migration ---

    def _migrate_lore(self) -> None:
        con = self.store._connect()
        try:
            done = con.execute(
                "SELECT value FROM memory_meta WHERE key='lore_migrated'"
            ).fetchone()
            if done is not None:
                return
            if not os.path.exists(LORE_DB_PATH):
                con.execute(
                    "INSERT OR REPLACE INTO memory_meta (key, value) VALUES ('lore_migrated', ?)",
                    (_now_iso(),),
                )
                con.commit()
                return
            kind_map = {'person': 'person', 'event': 'event', 'milestone': 'event',
                        'joke': 'fact', 'glossary': 'fact'}
            imported = 0
            pinned = 0
            try:
                lore = sqlite3.connect(LORE_DB_PATH, timeout=30)
                lore.row_factory = sqlite3.Row
                try:
                    rows = lore.execute('SELECT * FROM lore_entries').fetchall()
                finally:
                    lore.close()
            except sqlite3.Error:
                # Transient read failure (e.g. locked DB): leave the migration
                # flag unset so the next startup retries the import.
                logger.exception('[memory] failed to read legacy lore.db — will retry on next start')
                return
            for r in rows:
                if r['status'] == 'pending':
                    continue
                is_pinned = r['status'] == 'curated'
                status = 'archived' if r['archived'] else 'active'
                sources = json.loads(r['sources'] or '[]')
                content = _snippet(
                    f"{r['term']}: {r['content']}".replace('\n', ' ').strip(),
                    _MAX_CONTENT,
                )
                now = r['updated_at'] or _now_iso()
                created = r['created_at'] or now
                cur = con.execute(
                    'INSERT INTO memories '
                    '(guild_id, subject, kind, content, importance, pinned, pinned_by, origin, '
                    'recall_count, last_recalled_at, status, created_by, updated_by, created_at, updated_at, embedding) '
                    "VALUES (?, '', ?, ?, ?, ?, 'lore', ?, 0, NULL, ?, 'lore', 'lore', ?, ?, NULL)",
                    (
                        r['guild_id'], kind_map.get(r['type'], 'fact'), content,
                        4 if is_pinned else 3, int(is_pinned),
                        sources[0] if sources else '', status, created, now,
                    ),
                )
                mid = cur.lastrowid
                con.execute(
                    'INSERT INTO memory_history '
                    '(memory_id, guild_id, action, actor, actor_name, before, after, reason, created_at) '
                    "VALUES (?,?, 'migrate', 'lore', 'lore', NULL, ?, 'Migração da enciclopédia de lore', ?)",
                    (mid, r['guild_id'],
                     json.dumps({'subject': '', 'kind': kind_map.get(r['type'], 'fact'),
                                 'content': content, 'importance': 4 if is_pinned else 3,
                                 'pinned': is_pinned, 'origin': sources[0] if sources else '',
                                 'status': status}, ensure_ascii=False), now),
                )
                imported += 1
                if is_pinned and status == 'active':
                    pinned += 1
            con.execute(
                "INSERT OR REPLACE INTO memory_meta (key, value) VALUES ('lore_migrated', ?)",
                (_now_iso(),),
            )
            con.commit()
            if imported:
                logger.info('[memory] migrated %d lore entries (%d pinned)', imported, pinned)
        finally:
            con.close()

    # --- Embeddings (reuse DocsRAG / HistoryRAG clients) ---

    async def _embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        for cog_name in ('DocsRAG', 'HistoryRAG'):
            cog = self.bot.get_cog(cog_name)
            if cog is not None and hasattr(cog, '_embed_batch'):
                try:
                    return await cog._embed_batch(texts)
                except Exception:
                    logger.exception('[memory] embedding via %s failed', cog_name)
        return None

    async def _embed_content(self, content: str):
        result = await self._embed_texts([content])
        if not result:
            return None
        return np.array(result[0], dtype=np.float32)

    # --- Scoring helpers ---

    @staticmethod
    def _lexical(entry: dict, qn: str) -> float:
        if len(qn) < 3:
            return 0.0
        cn = _norm(entry['content'])
        if qn == cn:
            return 3.0
        if qn in cn or cn in qn:
            return 2.0
        for token in qn.split():
            if len(token) >= 4 and token in cn:
                return 1.5
        return 0.0

    @staticmethod
    def _recency_bonus(entry: dict) -> float:
        ts = entry.get('last_recalled_at') or entry.get('updated_at')
        if not ts:
            return 0.0
        try:
            dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
            days = (datetime.datetime.now(datetime.timezone.utc) - dt).days
        except Exception:
            return 0.0
        if days <= 7:
            return 0.15
        if days <= 30:
            return 0.08
        return 0.0

    # --- Search ---

    async def search(
        self, guild_id: int, query: str, *, subjects: list[str] | None = None,
        kind: str | None = None, limit: int = 4,
    ) -> list[tuple[dict, float, str]]:
        subjects = subjects if subjects is not None else ['']
        entries = [
            e for e in await asyncio.to_thread(self.store.active_for_subjects, guild_id, subjects)
            if kind is None or e['kind'] == kind
        ]
        if not entries:
            return []
        qn = _norm(query)
        scored: list[tuple[dict, float, str]] = []
        for e in entries:
            lex = self._lexical(e, qn)
            if lex > 0:
                scored.append((e, lex + e['importance'] * 0.2 + self._recency_bonus(e), 'lexical'))
        scored.sort(key=lambda t: t[1], reverse=True)
        if len(scored) >= limit:
            return scored[:limit]
        already = {e['id'] for e, _s, _h in scored}
        candidates = [e for e in entries if e['embedding'] is not None and e['id'] not in already]
        if candidates:
            q_emb = await self._embed_texts([query])
            if q_emb:
                q_arr = np.array(q_emb[0], dtype=np.float32)
                for e in candidates:
                    cos = _cosine(e['embedding'], q_arr)
                    if cos >= MEMORY_SEMANTIC_MIN_SCORE:
                        score = cos + e['importance'] * 0.2 + self._recency_bonus(e)
                        scored.append((e, score, 'semantic'))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    # --- Automatic injection ---

    async def build_memory_block(
        self, guild_id: int, query: str, *,
        speaker_id: str | None = None,
        participant_ids: set[str] | None = None,
    ) -> str | None:
        """Build the <memory> block injected into the system prompt.

        Pinned memories always ride along (importance-ranked, hard cap). The
        rest are selected semantically from the current query. User-scoped
        memories only participate when their subject is a conversation
        participant (speaker, mention, reply target).
        """
        try:
            participants = {p for p in (participant_ids or set()) if p}
            if speaker_id:
                participants.add(speaker_id)
            entries = await asyncio.to_thread(
                self.store.active_for_subjects, guild_id, sorted(participants),
            )
            if not entries:
                return None

            pinned = sorted(
                (e for e in entries if e['pinned']),
                key=lambda e: (-e['importance'], -e['id']),
            )[:_MAX_PINNED_INJECT]
            pinned_ids = {e['id'] for e in pinned}
            recalled = list(pinned_ids)

            semantic: list[tuple[float, dict]] = []
            qn = _norm(query)
            q_arr = None
            if query:
                q_emb = await self._embed_texts([query])
                if q_emb:
                    q_arr = np.array(q_emb[0], dtype=np.float32)
            for e in entries:
                if e['id'] in pinned_ids:
                    continue
                if q_arr is not None and e['embedding'] is not None:
                    base = _cosine(e['embedding'], q_arr)
                    how_semantic = True
                else:
                    base = self._lexical(e, qn)
                    how_semantic = False
                if base <= 0:
                    continue
                score = base + e['importance'] * 0.15
                if e['subject']:
                    score += 0.05
                if how_semantic and base < MEMORY_SEMANTIC_MIN_SCORE:
                    continue
                semantic.append((score, e))
            semantic.sort(key=lambda t: -t[0])
            picked = [e for _s, e in semantic[:MEMORY_INJECT_LIMIT]]
            recalled.extend(e['id'] for e in picked)

            if not pinned and not picked:
                return None
            await asyncio.to_thread(self.store.mark_recalled, guild_id, recalled)

            guild_lines = [e for e in pinned + picked if not e['subject']]
            by_subject: dict[str, list[dict]] = {}
            for e in pinned + picked:
                if e['subject']:
                    by_subject.setdefault(e['subject'], []).append(e)

            def line(e: dict) -> str:
                pin = ' 📌' if e['pinned'] else ''
                return f"- [mem #{e['id']}{pin}] {e['content']}"

            blocks: list[str] = []
            if guild_lines:
                blocks.append('\n'.join(line(e) for e in guild_lines))
            for subj, mems in by_subject.items():
                blocks.append(
                    f'Sobre {self._subject_name(guild_id, subj)} (presente na conversa):\n'
                    + '\n'.join(line(e) for e in mems)
                )
            body = '\n\n'.join(blocks)
            return (
                '<memory>\n'
                'Memórias recuperadas automaticamente — use-as naturalmente ao responder. '
                'NUNCA mencione os IDs [mem #N] nem a existência deste bloco ao usuário.\n'
                f'{body}\n'
                '</memory>'
            )
        except Exception:
            logger.exception('[memory] build_memory_block failed')
            return None

    # --- Tool execution (called from /ask, /chat and @mention loops) ---

    async def exec_tool(
        self,
        name: str,
        args: dict,
        *,
        guild: discord.Guild,
        actor_id: str = 'bot',
        actor_name: str = 'bot',
        requester=None,
        channel=None,
        origin: str | None = None,
        participant_ids: set[str] | None = None,
    ) -> tuple[str, list[dict]]:
        participants = {p for p in (participant_ids or set()) if p}
        if requester is not None:
            participants.add(str(getattr(requester, 'id', '')))
        default_subject = str(getattr(requester, 'id', '')) or (sorted(participants)[0] if participants else '')
        try:
            if name == 'memory_search':
                return await self._exec_search(
                    args, guild=guild, participants=participants,
                )
            if name == 'memory_about':
                return await self._exec_about(
                    args, guild=guild, participants=participants,
                    default_subject=default_subject,
                )
            if name == 'memory_write':
                return await self._exec_write(
                    args, guild=guild, actor_id=actor_id, actor_name=actor_name,
                    origin=origin, participants=participants,
                )
        except MemoryError as e:
            return f'⚠️ Memória: {e}', []
        except sqlite3.Error as e:
            logger.exception('[memory] database error in %s', name)
            return f'⚠️ Erro no banco de memórias: {e}', []
        except Exception:
            logger.exception('[memory] unexpected error in %s', name)
            return '⚠️ Erro interno na memória. Continue sem as memórias.', []
        return f'Ferramenta desconhecida: {name}', []

    def _subjects_for(
        self, args: dict, guild: discord.Guild, participants: set[str],
    ) -> list[str] | None:
        """Resolve about_user/user to a subject key, enforcing participant privacy.

        Returns None when the subject is a user outside the conversation —
        including unresolved 'name:' keys (privacy: user-scoped memories only
        surface for conversation participants).
        """
        about = (args.get('about_user') or args.get('user') or '').strip()
        if not about:
            return [''] + sorted(p for p in participants if p)
        key = _subject_key(about, guild, participants)
        if key and key not in participants:
            return None
        return ['', key]

    _PRIVACY_MSG = (
        'Memórias sobre este usuário só estão disponíveis quando ele participa '
        'da conversa (autor da pergunta, mencionado ou alvo de resposta).'
    )

    async def _exec_search(
        self, args: dict, *, guild: discord.Guild, participants: set[str],
    ) -> tuple[str, list[dict]]:
        query = (args.get('query') or '').strip()
        try:
            limit = max(1, min(8, int(args.get('limit', 4))))
        except (TypeError, ValueError):
            limit = 4
        if not query:
            return 'Consulta vazia.', []
        kind = args.get('kind') if args.get('kind') in MEMORY_KINDS else None
        subjects = self._subjects_for(args, guild, participants)
        if subjects is None:
            return self._PRIVACY_MSG, []
        hits = await self.search(
            guild.id, query, subjects=subjects, kind=kind, limit=limit,
        )
        logger.info('[memory] search %r -> %d hit(s)', query, len(hits))
        for entry, score, how in hits:
            logger.info(
                '[memory]   hit: %r (id=%d, kind=%s, imp=%d, score=%.2f, match=%s)',
                entry['content'][:60], entry['id'], entry['kind'],
                entry['importance'], score, how,
            )
        if not hits:
            return (
                'Nenhuma memória encontrada para esta consulta. '
                'Se a conversa ou o histórico revelar algo digno de memória, '
                'você pode registrar com memory_write.'
            ), []
        await asyncio.to_thread(
            self.store.mark_recalled, guild.id, [e['id'] for e, _s, _h in hits],
        )
        parts = []
        for entry, score, _how in hits:
            scope = self._subject_name(guild.id, entry['subject'])
            origin = f' | Origem: {entry["origin"]}' if entry['origin'] else ''
            parts.append(
                f"[mem #{entry['id']}] ({scope}, {_KIND_LABELS.get(entry['kind'], entry['kind'])}, "
                f"importância {entry['importance']}, score {score:.2f})\n"
                f'{entry["content"]}{origin}'
            )
        return '\n\n---\n\n'.join(parts), []

    def _subject_name(self, guild_id: int, subject: str) -> str:
        """Human-readable scope label for LLM-facing text."""
        if not subject:
            return 'servidor'
        if subject.startswith('name:'):
            return subject[5:]
        g = self.bot.get_guild(guild_id)
        member = g.get_member(int(subject)) if g and subject.isdigit() else None
        return member.display_name if member else f'usuário {subject}'

    async def _exec_about(
        self, args: dict, *, guild: discord.Guild, participants: set[str],
        default_subject: str = '',
    ) -> tuple[str, list[dict]]:
        subjects = self._subjects_for(args, guild, participants)
        if subjects is None:
            return self._PRIVACY_MSG, []
        if args.get('user') or args.get('about_user'):
            subject = subjects[-1] if len(subjects) > 1 else ''
        else:
            subject = default_subject
        entries = [
            e for e in await asyncio.to_thread(
                self.store.list_memories, guild.id, subject=subject,
            )
            if e['status'] == 'active'
        ]
        logger.info('[memory] about %s in guild %d -> %d active', subject or '(guild)', guild.id, len(entries))
        if not entries:
            return f'Nenhuma memória ativa sobre {self._subject_name(guild.id, subject)}.', []
        await asyncio.to_thread(self.store.mark_recalled, guild.id, [e['id'] for e in entries])
        entries.sort(key=lambda e: (-e['importance'], -e['pinned'], -e['recall_count']))
        by_kind: dict[str, list[dict]] = {}
        for e in entries:
            by_kind.setdefault(e['kind'], []).append(e)
        parts = [f'Memórias ativas sobre {self._subject_name(guild.id, subject)} ({len(entries)}):']
        for k, mems in by_kind.items():
            lines = [
                f"- [mem #{e['id']}]{' 📌' if e['pinned'] else ''} {e['content']}"
                for e in mems[:6]
            ]
            parts.append(f'{_KIND_LABELS.get(k, k)}:\n' + '\n'.join(lines))
        return '\n\n'.join(parts), []

    # --- Bot write rate limiting (peek first, debit only on actual writes) ---

    def _bot_write_slots(self, guild_id: int) -> bool:
        now = time.monotonic()
        dq = self._bot_writes.setdefault(guild_id, deque())
        while dq and now - dq[0] > 3600:
            dq.popleft()
        return len(dq) < MEMORY_BOT_WRITE_LIMIT

    def _bot_write_debit(self, guild_id: int) -> None:
        self._bot_writes.setdefault(guild_id, deque()).append(time.monotonic())

    async def _find_target(
        self, args: dict, guild: discord.Guild, subjects: list[str],
    ) -> tuple[dict | None, str]:
        """Locate an existing memory by id or content match within scope."""
        mid = args.get('memory_id')
        if mid is not None:
            try:
                entry = await asyncio.to_thread(
                    self.store.get_memory, guild.id, int(mid),
                )
            except (TypeError, ValueError):
                return None, '`memory_id` inválido.'
            if entry is None or entry['status'] != 'active':
                return None, f'Nenhuma memória ativa com id={mid}.'
            return entry, ''
        match = (args.get('content_match') or args.get('content') or '').strip()
        if not match:
            return None, 'Informe `memory_id` ou `content_match` para localizar a memória.'
        mn = _norm(match)
        entries = [
            e for e in await asyncio.to_thread(
                self.store.active_for_subjects, guild.id, [s for s in subjects if s],
            )
            if e['subject'] in subjects
        ]
        emb = None
        needs_semantic = [e for e in entries if e['embedding'] is not None]
        if needs_semantic:
            emb = await self._embed_content(match)
        best: tuple[float, dict] | None = None
        for e in entries:
            cn = _norm(e['content'])
            if mn == cn or mn in cn or cn in mn:
                return e, ''
            if emb is not None and e['embedding'] is not None:
                cos = _cosine(e['embedding'], emb)
                if best is None or cos > best[0]:
                    best = (cos, e)
        if best is not None and best[0] >= 0.6:
            return best[1], ''
        return None, 'Nenhuma memória ativa corresponde ao conteúdo informado.'

    async def _dedupe_check(
        self, guild: discord.Guild, subject: str, content: str, embedding,
    ) -> tuple[dict, float] | None:
        entries = await asyncio.to_thread(
            self.store.list_memories, guild.id, subject=subject,
        )
        active = [e for e in entries if e['status'] == 'active']
        cn = _norm(content)
        for e in active:
            if _norm(e['content']) == cn:
                return e, 1.0
        if embedding is None:
            return None
        best: tuple[float, dict] | None = None
        for e in active:
            if e['embedding'] is None:
                continue
            cos = _cosine(e['embedding'], embedding)
            if best is None or cos > best[0]:
                best = (cos, e)
        if best is not None and best[0] >= MEMORY_DEDUPE_THRESHOLD:
            return best[1], best[0]
        return None

    async def _exec_write(
        self, args: dict, *, guild: discord.Guild, actor_id: str, actor_name: str,
        origin: str | None, participants: set[str],
    ) -> tuple[str, list[dict]]:
        if actor_id == 'bot' and not self._bot_write_slots(guild.id):
            return 'Limite de escritas do bot por hora atingido. Tente novamente mais tarde.', []
        action = args.get('action')
        reason = (args.get('reason') or '').strip()[:_MAX_REASON]
        if action not in ('create', 'update', 'forget'):
            return 'action inválida — use create, update ou forget.', []
        if not reason:
            return '`reason` é obrigatório (uma linha).', []
        subject = _subject_key(args.get('about_user'), guild, participants)
        if subject.startswith('name:'):
            return (
                'Não consegui identificar este usuário — memórias sobre pessoas exigem que '
                'ele participe da conversa. Use o ID dele (disponível no contexto da '
                'conversa), ou salve como fato do servidor omitindo `about_user`.'
            ), []
        origin_url = (origin or '').strip() or ''
        if origin_url and not origin_url.startswith('http'):
            origin_url = ''

        if action == 'create':
            content = (args.get('content') or '').strip()
            if len(content) < _MIN_CONTENT:
                return f'`content` muito curto — escreva uma frase completa (mínimo {_MIN_CONTENT} caracteres).', []
            content = content[:_MAX_CONTENT]
            kind = args.get('kind') if args.get('kind') in MEMORY_KINDS else 'fact'
            importance = _parse_importance(
                args.get('importance'), _DEFAULT_IMPORTANCE.get(kind, 3),
            )
            pinned = _parse_bool(args.get('pinned'))
            if pinned and actor_id == 'bot':
                pins = await asyncio.to_thread(self.store.count_pins, guild.id)
                if pins >= MEMORY_PIN_LIMIT:
                    return (
                        f'Limite de memórias fixadas atingido ({pins}/{MEMORY_PIN_LIMIT}). '
                        'Fixe apenas o essencial — desafixe outra com action=update, pinned=false '
                        'ou deixe esta sem fixação.'
                    ), []
            count = await asyncio.to_thread(self.store.count_subject, guild.id, subject)
            if count >= MEMORY_MAX_PER_SUBJECT:
                return (
                    f'Limite de {MEMORY_MAX_PER_SUBJECT} memórias ativas atingido para este '
                    'escopo. Use action=forget para remover memórias desatualizadas antes.'
                ), []
            embedding = await self._embed_content(content)
            dup = await self._dedupe_check(guild, subject, content, embedding)
            if dup is not None:
                dup_entry, sim = dup
                return (
                    f'Memória semelhante já existe: [mem #{dup_entry["id"]}] '
                    f'"{dup_entry["content"]}" (semelhança {sim:.2f}). '
                    'Use action=update para editá-la em vez de duplicar.'
                ), []
            if actor_id == 'bot':
                self._bot_write_debit(guild.id)
            entry = await self._create(
                guild.id, subject=subject, kind=kind, content=content,
                importance=importance, pinned=pinned, pinned_by=actor_id,
                origin=origin_url, actor_id=actor_id, actor_name=actor_name,
                reason=reason, embedding=embedding,
            )
            note = ' — fixada: entra em toda conversa.' if pinned else ''
            return (
                f"Memória criada: [mem #{entry['id']}] \"{content}\" "
                f'(tipo={kind}, importância={importance}{note})'
            ), []

        subjects = ['', subject] if subject else ['']
        entry, err = await self._find_target(args, guild, subjects)
        if err:
            return err, []

        if action == 'forget':
            if actor_id == 'bot':
                self._bot_write_debit(guild.id)
            entry = await self._apply(
                guild.id, entry, {'status': 'archived'}, action='forget',
                actor_id=actor_id, actor_name=actor_name, reason=reason,
            )
            return (
                f"Memória arquivada: [mem #{entry['id']}] \"{_snippet(entry['content'], 80)}\" "
                '— reversível por admins via histórico.'
            ), []

        # update
        patch: dict = {}
        content = (args.get('content') or '').strip()
        if content:
            if len(content) < _MIN_CONTENT:
                return f'`content` muito curto (mínimo {_MIN_CONTENT} caracteres).', []
            patch['content'] = content[:_MAX_CONTENT]
        if args.get('kind') in MEMORY_KINDS:
            patch['kind'] = args['kind']
        if args.get('importance') is not None:
            patch['importance'] = _parse_importance(args.get('importance'), entry['importance'])
        if args.get('pinned') is not None:
            want_pin = _parse_bool(args['pinned'])
            if want_pin and not entry['pinned'] and actor_id == 'bot':
                pins = await asyncio.to_thread(self.store.count_pins, guild.id)
                if pins >= MEMORY_PIN_LIMIT:
                    return (
                        f'Limite de memórias fixadas atingido ({pins}/{MEMORY_PIN_LIMIT}).'
                    ), []
            if want_pin != entry['pinned']:
                patch['pinned'] = want_pin
                patch['pinned_by'] = actor_id if want_pin else ''
        if not patch:
            return 'Nada para atualizar — forneça content, kind, importance e/ou pinned.', []
        if actor_id == 'bot':
            self._bot_write_debit(guild.id)
        entry = await self._apply(
            guild.id, entry, patch, action='update',
            actor_id=actor_id, actor_name=actor_name, reason=reason,
        )
        return (
            f"Memória atualizada: [mem #{entry['id']}] \"{_snippet(entry['content'], 80)}\"."
        ), []

    # --- Mutation helpers (single transaction + log channel) ---

    async def _create(
        self, guild_id: int, *, subject: str, kind: str, content: str,
        importance: int, pinned: bool, pinned_by: str, origin: str,
        actor_id: str, actor_name: str, reason: str, embedding=None,
    ) -> dict:
        entry, history_id = await asyncio.to_thread(
            self.store.create_with_history, guild_id,
            subject=subject, kind=kind, content=content, importance=importance,
            pinned=pinned, pinned_by=pinned_by, origin=origin,
            actor=actor_id, actor_name=actor_name, reason=reason,
            embedding=embedding,
        )
        logger.info(
            "[memory] create #%d '%s' (kind=%s, imp=%d, pinned=%s) by %s",
            entry['id'], content[:60], kind, importance, pinned, actor_name,
        )
        self._schedule_log(
            guild_id, action='create', entry=entry, before=None,
            actor_name=actor_name, reason=reason, history_id=history_id,
        )
        return entry

    async def _apply(
        self, guild_id: int, entry: dict, patch: dict, *, action: str,
        actor_id: str, actor_name: str, reason: str,
    ) -> dict:
        embedding = None
        if 'content' in patch:
            embedding = await self._embed_content(patch['content'])
        updated, before, history_id = await asyncio.to_thread(
            self.store.update_with_history, guild_id, entry['id'], patch,
            action=action, actor=actor_id, actor_name=actor_name,
            reason=reason, embedding=embedding,
        )
        logger.info(
            '[memory] %s #%d by %s — %s',
            action, updated['id'], actor_name, reason[:80],
        )
        self._schedule_log(
            guild_id, action=action, entry=updated, before=before,
            actor_name=actor_name, reason=reason, history_id=history_id,
        )
        return updated

    async def revert(
        self, guild_id: int, history_id: int, *, actor_id: str, actor_name: str,
    ) -> dict:
        row = await asyncio.to_thread(self.store.get_history, guild_id, history_id)
        if row is None:
            raise MemoryError('Registro de histórico não encontrado.')
        entry = await asyncio.to_thread(self.store.get_memory, guild_id, row['memory_id'])
        if entry is None:
            raise MemoryError('Memória não encontrada (pode ter sido apagada).')
        if row['before'] is None:
            patch = {'status': 'archived'}
        else:
            before = json.loads(row['before'])
            patch = {k: before[k] for k in _SNAPSHOT_FIELDS if k in before}
            if 'pinned' in patch:
                patch['pinned_by'] = ''
        return await self._apply(
            guild_id, entry, patch, action='revert',
            actor_id=actor_id, actor_name=actor_name,
            reason=f'Reversão do registro de histórico #{history_id}',
        )

    async def restore_version(
        self, guild_id: int, history_id: int, *, actor_id: str, actor_name: str,
    ) -> dict:
        row = await asyncio.to_thread(self.store.get_history, guild_id, history_id)
        if row is None:
            raise MemoryError('Registro de histórico não encontrado.')
        entry = await asyncio.to_thread(self.store.get_memory, guild_id, row['memory_id'])
        if entry is None:
            raise MemoryError('Memória não encontrada (pode ter sido apagada).')
        if row['after'] is None:
            raise MemoryError('Este registro não possui um estado para restaurar.')
        after = json.loads(row['after'])
        patch = {k: after[k] for k in _SNAPSHOT_FIELDS if k in after}
        return await self._apply(
            guild_id, entry, patch, action='restore',
            actor_id=actor_id, actor_name=actor_name,
            reason=f'Restauração da versão do histórico #{history_id}',
        )

    async def set_pinned(
        self, guild_id: int, memory_id: int, *, pinned: bool,
        actor_id: str, actor_name: str,
    ) -> dict:
        entry = await asyncio.to_thread(self.store.get_memory, guild_id, memory_id)
        if entry is None or entry['status'] != 'active':
            raise MemoryError('Memória ativa não encontrada.')
        if pinned and not entry['pinned']:
            pins = await asyncio.to_thread(self.store.count_pins, guild_id)
            if pins >= MEMORY_PIN_LIMIT:
                raise MemoryError(
                    f'Limite de memórias fixadas atingido ({pins}/{MEMORY_PIN_LIMIT}). '
                    'Desafixe outra primeiro.'
                )
        patch = {'pinned': pinned, 'pinned_by': actor_id if pinned else ''}
        return await self._apply(
            guild_id, entry, patch,
            action='pin' if pinned else 'unpin',
            actor_id=actor_id, actor_name=actor_name,
            reason='Fixação manual' if pinned else 'Desfixação manual',
        )

    # --- Admin log channel ---

    def _schedule_log(
        self, guild_id: int, *, action: str, entry: dict, before: dict | None,
        actor_name: str, reason: str, history_id: int,
    ) -> None:
        task = asyncio.get_running_loop().create_task(
            self._post_log(
                guild_id, action=action, entry=entry, before=before,
                actor_name=actor_name, reason=reason, history_id=history_id,
            )
        )
        self._log_tasks.add(task)
        task.add_done_callback(self._log_tasks.discard)

    async def _post_log(
        self, guild_id: int, *, action: str, entry: dict, before: dict | None,
        actor_name: str, reason: str, history_id: int,
    ) -> None:
        if not MEMORY_LOG_CHANNEL_ID:
            return
        try:
            channel = self.bot.get_channel(MEMORY_LOG_CHANNEL_ID)
            if channel is None:
                channel = await self.bot.fetch_channel(MEMORY_LOG_CHANNEL_ID)
        except Exception:
            logger.exception('[memory] log channel unavailable')
            return
        colors = {
            'create': discord.Color.green(),
            'update': discord.Color.gold(),
            'forget': discord.Color.red(),
            'revert': discord.Color.dark_grey(),
            'restore': discord.Color.dark_grey(),
            'pin': discord.Color.blurple(),
            'unpin': discord.Color.blurple(),
        }
        verb = _ACTION_VERBS.get(action, action)
        embed = discord.Embed(
            title=f"🧠 Memória #{entry['id']} — {verb}",
            description=f"Por **{actor_name}** — {reason or 'sem motivo'}",
            color=colors.get(action, discord.Color.greyple()),
        )
        embed.add_field(
            name='Escopo',
            value=self._scope_text(guild_id, entry['subject']),
            inline=True,
        )
        embed.add_field(name='Tipo', value=_KIND_LABELS.get(entry['kind'], entry['kind']), inline=True)
        embed.add_field(
            name='Importância',
            value=f"{'📌 ' if entry['pinned'] else ''}{entry['importance']}/5",
            inline=True,
        )
        embed.add_field(name='Conteúdo', value=_snippet(entry['content'], 800), inline=False)
        if action == 'update' and before:
            embed.add_field(name='Antes', value=_snippet(before.get('content', ''), 400), inline=False)
        if entry['origin']:
            embed.add_field(name='Origem', value=entry['origin'], inline=False)
        embed.set_footer(text=f'hist #{history_id} • {_fmt_brt(entry["updated_at"])}')
        view = MemoryLogView(self, guild_id=guild_id, memory_id=entry['id'], history_id=history_id)
        try:
            await channel.send(embed=embed, view=view)
        except Exception:
            logger.exception('[memory] failed to post log embed')

    def _scope_text(self, guild_id: int, subject: str) -> str:
        if not subject:
            return 'Servidor'
        g = self.bot.get_guild(guild_id)
        member = g.get_member(int(subject)) if g and subject.isdigit() else None
        return member.mention if member else self._subject_name(guild_id, subject)

    # --- Slash commands ---

    memory = app_commands.Group(name='memory', description='Memórias persistentes do bot')

    @memory.command(name='list', description='Listar memórias')
    @app_commands.describe(kind='Filtrar por tipo')
    @app_commands.choices(kind=_KIND_CHOICES)
    async def memory_list(self, interaction: discord.Interaction, kind: str | None = None):
        guild = interaction.guild
        if guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        admin = _is_admin(interaction)
        me = str(interaction.user.id)
        entries = await asyncio.to_thread(self.store.list_memories, guild.id)
        entries = [
            e for e in entries
            if (admin or not e['subject'] or e['subject'] == me)
            and (kind is None or e['kind'] == kind)
        ]
        if not entries:
            return await interaction.response.send_message(
                'Nenhuma memória encontrada. O bot cria memórias automaticamente a partir '
                'das conversas — ou peça a ele para lembrar de algo.',
                ephemeral=True,
            )
        lines = []
        for e in entries:
            flags = ''
            if e['status'] == 'archived':
                flags += ' • arquivada'
            if e['pinned']:
                flags += ' • 📌'
            scope = self._scope_text(guild.id, e['subject'])
            lines.append(
                f"**[#{e['id']}]** {scope} — {_snippet(e['content'], 90)}{flags}"
            )
        pages: list[discord.Embed] = []
        for i in range(0, len(lines), 12):
            pages.append(discord.Embed(
                title='🧠 Memórias',
                description='\n'.join(lines[i:i + 12]),
                color=discord.Color.dark_teal(),
            ))
        for i, page in enumerate(pages):
            page.set_footer(text=f'{len(entries)} memórias • Página {i + 1}/{len(pages)}')
        if len(pages) == 1:
            await interaction.response.send_message(embed=pages[0], ephemeral=True)
        else:
            await interaction.response.send_message(embed=pages[0], view=PaginatedEmbedView(pages), ephemeral=True)

    @memory.command(name='show', description='Ver uma memória pelo ID')
    @app_commands.describe(id='ID da memória')
    async def memory_show(self, interaction: discord.Interaction, id: int):
        guild = interaction.guild
        if guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        entry = await asyncio.to_thread(self.store.get_memory, guild.id, id)
        if entry is None:
            return await interaction.response.send_message(f'Memória #{id} não encontrada.', ephemeral=True)
        if not _is_admin(interaction) and entry['subject'] not in ('', str(interaction.user.id)):
            return await interaction.response.send_message('Você só pode ver memórias do servidor ou suas.', ephemeral=True)
        embed = discord.Embed(
            title=f"🧠 Memória #{entry['id']}",
            description=_snippet(entry['content'], 3900),
            color=discord.Color.dark_teal(),
        )
        embed.add_field(name='Escopo', value=self._scope_text(guild.id, entry['subject']), inline=True)
        embed.add_field(name='Tipo', value=_KIND_LABELS.get(entry['kind'], entry['kind']), inline=True)
        embed.add_field(name='Importância', value=f"{'📌 ' if entry['pinned'] else ''}{entry['importance']}/5", inline=True)
        embed.add_field(name='Status', value='arquivada' if entry['status'] == 'archived' else 'ativa', inline=True)
        embed.add_field(name='Recalls', value=str(entry['recall_count']), inline=True)
        embed.add_field(name='Atualizada', value=_fmt_brt(entry['updated_at']), inline=True)
        if entry['origin']:
            embed.add_field(name='Origem', value=entry['origin'], inline=False)
        if _is_admin(interaction):
            history = await asyncio.to_thread(self.store.history_for_memory, guild.id, entry['id'], 5)
            if history:
                hist_lines = [
                    f'`#{h["id"]}` {h["action"]} — {h["actor_name"]} — {_fmt_brt(h["created_at"])}'
                    for h in history
                ]
                embed.add_field(
                    name='Histórico (use /memory restore <id>)',
                    value='\n'.join(hist_lines),
                    inline=False,
                )
        await interaction.response.send_message(embed=embed)

    @memory.command(name='edit', description='Editar uma memória (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(id='ID da memória', content='Novo texto', importance='Nova importância (1-5)')
    async def memory_edit(
        self, interaction: discord.Interaction, id: int,
        content: str | None = None, importance: app_commands.Range[int, 1, 5] | None = None,
    ):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        entry = await asyncio.to_thread(self.store.get_memory, interaction.guild.id, id)
        if entry is None or entry['status'] != 'active':
            return await interaction.response.send_message(f'Memória ativa #{id} não encontrada.', ephemeral=True)
        patch: dict = {}
        if content is not None and len(content.strip()) >= _MIN_CONTENT:
            patch['content'] = content.strip()[:_MAX_CONTENT]
        if importance is not None:
            patch['importance'] = importance
        if not patch:
            return await interaction.response.send_message('Informe `content` e/ou `importance`.', ephemeral=True)
        await interaction.response.defer(ephemeral=True)
        try:
            entry = await self._apply(
                interaction.guild.id, entry, patch, action='update',
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
                reason='Edição manual via /memory edit',
            )
        except MemoryError as e:
            return await interaction.followup.send(f'⚠️ {e}', ephemeral=True)
        await interaction.followup.send(f"✅ Memória #{entry['id']} atualizada.", ephemeral=True)

    @memory.command(name='delete', description='Arquivar uma memória (reversível) (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(id='ID da memória')
    async def memory_delete(self, interaction: discord.Interaction, id: int):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        entry = await asyncio.to_thread(self.store.get_memory, interaction.guild.id, id)
        if entry is None or entry['status'] != 'active':
            return await interaction.response.send_message(f'Memória ativa #{id} não encontrada.', ephemeral=True)
        view = ConfirmView(confirm_label='🗑️ Arquivar')
        await interaction.response.send_message(
            f"Arquivar memória #{id}: \"{_snippet(entry['content'], 120)}\"? Reversível via histórico.",
            view=view,
            ephemeral=True,
        )
        await view.wait()
        if not view.confirmed:
            return
        try:
            await self._apply(
                interaction.guild.id, entry, {'status': 'archived'}, action='forget',
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
                reason='Arquivamento manual via /memory delete',
            )
        except MemoryError as e:
            return await interaction.followup.send(f'⚠️ {e}', ephemeral=True)
        await interaction.followup.send(f'🗑️ Memória #{id} arquivada.', ephemeral=True)

    @memory.command(name='restore', description='Restaurar uma versão pelo ID do histórico (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(id='ID do registro de histórico (veja /memory show)')
    async def memory_restore(self, interaction: discord.Interaction, id: int):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        await interaction.response.defer(ephemeral=True)
        try:
            entry = await self.restore_version(
                interaction.guild.id, id,
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except MemoryError as e:
            return await interaction.followup.send(f'⚠️ {e}', ephemeral=True)
        await interaction.followup.send(f'✅ Memória #{entry["id"]} restaurada à versão do histórico #{id}.')

    @memory.command(name='pin', description='Fixar uma memória — injeta em toda conversa (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(id='ID da memória')
    async def memory_pin(self, interaction: discord.Interaction, id: int):
        await self._pin_toggle(interaction, id, pinned=True)

    @memory.command(name='unpin', description='Desafixar uma memória (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(id='ID da memória')
    async def memory_unpin(self, interaction: discord.Interaction, id: int):
        await self._pin_toggle(interaction, id, pinned=False)

    async def _pin_toggle(self, interaction: discord.Interaction, id: int, *, pinned: bool):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        try:
            entry = await self.set_pinned(
                interaction.guild.id, id, pinned=pinned,
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except MemoryError as e:
            return await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
        verb = 'fixada — entra em toda conversa' if pinned else 'desafixada'
        await interaction.response.send_message(f"{'📌' if pinned else '📍'} Memória #{entry['id']} {verb}.")

    @memory.command(name='whoami', description='Ver o que o bot lembra sobre você')
    async def memory_whoami(self, interaction: discord.Interaction):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        me = str(interaction.user.id)
        entries = [
            e for e in await asyncio.to_thread(
                self.store.list_memories, interaction.guild.id, subject=me,
            )
            if e['status'] == 'active'
        ]
        if not entries:
            return await interaction.response.send_message(
                '🧠 Nenhuma memória ativa sobre você. Converse com o bot ou peça para ele '
                'lembrar de algo.', ephemeral=True,
            )
        await asyncio.to_thread(self.store.mark_recalled, interaction.guild.id, [e['id'] for e in entries])
        lines = [
            f"- [#{e['id']}]{' 📌' if e['pinned'] else ''} {_snippet(e['content'], 120)}"
            for e in entries[:20]
        ]
        embed = discord.Embed(
            title=f"🧠 O que lembro sobre {interaction.user.display_name}",
            description='\n'.join(lines),
            color=discord.Color.dark_teal(),
        )
        embed.set_footer(text=f'{len(entries)} memórias ativas • /memory forget-me para apagar')
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @memory.command(name='forget-me', description='Apagar permanentemente todas as memórias sobre você')
    async def memory_forget_me(self, interaction: discord.Interaction):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        me = _subject_keys_for_member(interaction.user)
        count = await asyncio.to_thread(self.store.count_subjects_all, interaction.guild.id, me)
        if count == 0:
            return await interaction.response.send_message('Não há memórias sobre você neste servidor.', ephemeral=True)
        view = ConfirmView(confirm_label='🧹 Apagar tudo')
        await interaction.response.send_message(
            f'Apagar **permanentemente** {count} memória(s) sobre você neste servidor? '
            'Não há como desfazer.',
            view=view,
            ephemeral=True,
        )
        await view.wait()
        if not view.confirmed:
            return
        await asyncio.to_thread(self.store.purge_subjects, interaction.guild.id, me)
        logger.info('[memory] purged %d memories for user %s in guild %d (self-service)', count, me, interaction.guild.id)
        await interaction.followup.send(f'🧹 {count} memória(s) apagada(s). Não lembro mais de nada sobre você.', ephemeral=True)

    @memory.command(name='forget', description='Apagar permanentemente as memórias sobre um usuário (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(user='Usuário alvo')
    async def memory_forget(self, interaction: discord.Interaction, user: discord.Member):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        key = _subject_keys_for_member(user)
        count = await asyncio.to_thread(self.store.count_subjects_all, interaction.guild.id, key)
        if count == 0:
            return await interaction.response.send_message(f'Não há memórias sobre {user.display_name}.', ephemeral=True)
        view = ConfirmView(confirm_label='🧹 Apagar tudo')
        await interaction.response.send_message(
            f'Apagar **permanentemente** {count} memória(s) sobre {user.display_name}? '
            'Não há como desfazer.',
            view=view,
            ephemeral=True,
        )
        await view.wait()
        if not view.confirmed:
            return
        await asyncio.to_thread(self.store.purge_subjects, interaction.guild.id, key)
        logger.info(
            '[memory] purged %d memories for user %s in guild %d by %s',
            count, key, interaction.guild.id, interaction.user.display_name,
        )
        await interaction.followup.send(f'🧹 {count} memória(s) sobre {user.display_name} apagada(s).', ephemeral=True)

    @memory.command(name='export', description='Exportar as memórias em JSON (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    async def memory_export(self, interaction: discord.Interaction):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        await interaction.response.defer(ephemeral=True)
        entries = await asyncio.to_thread(self.store.list_memories, interaction.guild.id)
        history = await asyncio.to_thread(self.store.history_for_guild, interaction.guild.id)
        data = {
            'exported_at': _now_iso(),
            'guild_id': interaction.guild.id,
            'memories': [
                {k: v for k, v in e.items() if k != 'embedding'} for e in entries
            ],
            'history': history,
        }
        content = json.dumps(data, ensure_ascii=False, indent=2)
        file = discord.File(
            io.BytesIO(content.encode('utf-8')),
            filename=f'memory-{interaction.guild.id}.json',
        )
        await interaction.followup.send(
            f'📦 Exportadas {len(entries)} memórias e {len(history)} registros de histórico.',
            file=file,
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    if not MEMORY_ENABLED:
        logger.info('Memory disabled via MEMORY_ENABLED=false')
        return
    await bot.add_cog(Memory(bot))
