"""Server lore encyclopedia — structured, bot-managed community memory.

Entries (events, people, glossary terms, milestones, inside jokes) are written
by the LLM via the ``save_lore`` tool (provenance required) and read via
``search_lore``. Every mutation funnels through :class:`LoreStore`, which
applies the change and records a before/after snapshot in ``lore_history``
inside a single transaction, then mirrors it to a configurable admin log
channel with Revert / Lock / Approve buttons.

Reads are silent: ``search_lore`` returns no sources so user-facing responses
stay clean — hits are console-logged instead.
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
    LORE_BOT_WRITE_LIMIT,
    LORE_DB_PATH,
    LORE_ENABLED,
    LORE_LOG_CHANNEL_ID,
)

logger = logging.getLogger(__name__)

_TYPE_LABELS = {
    'event': 'Evento',
    'person': 'Pessoa',
    'glossary': 'Glossário',
    'milestone': 'Marco',
    'joke': 'Piada interna',
}
LORE_TYPES = tuple(_TYPE_LABELS)
_CATEGORY_CHOICES = [
    app_commands.Choice(name=label, value=key) for key, label in _TYPE_LABELS.items()
]
_STATUS_LABELS = {
    'pending': 'pendente',
    'auto': 'auto',
    'curated': 'protegida',
}
_ACTION_VERBS = {
    'create': 'criada',
    'update': 'atualizada',
    'archive': 'arquivada',
    'revert': 'revertida',
    'restore': 'restaurada',
    'lock': 'protegida',
    'unlock': 'desprotegida',
    'approve': 'aprovada',
}
_SEMANTIC_MIN_SCORE = 0.35
_MIN_SUBSTRING_LEN = 3
_MAX_TERM = 128
_MAX_CONTENT = 4000
_MAX_ALIAS = 64
_MAX_REASON = 200
_SNAPSHOT_FIELDS = ('term', 'type', 'content', 'aliases', 'status', 'archived')

SEARCH_LORE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'search_lore',
        'description': (
            'Busca na enciclopédia de lore do servidor: história, eventos marcantes, '
            'piadas internas, glossário da comunidade, pessoas marcantes e marcos. '
            'Use ANTES de search_history para perguntas sobre o passado/tradições do '
            'servidor (ex: "quem causou o Grande Incidente?", "o que significa X aqui?"). '
            'Retorna entradas curadas com apelidos e fontes (links das mensagens de origem).'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Termo, apelido ou pergunta sobre o lore do servidor.',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Número máximo de entradas (padrão 3, máximo 6).',
                },
            },
            'required': ['query'],
        },
    },
}

SAVE_LORE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'save_lore',
        'description': (
            'Salva na enciclopédia de lore do servidor. Use quando descobrir, com base '
            'no histórico do servidor, um fato digno de memória: eventos, piadas internas, '
            'termos do glossário, marcos ou pessoas marcantes. '
            'Em create e update, `sources` é OBRIGATÓRIO com links (discord.com/channels/...) '
            'das mensagens que comprovam o fato — use os jump_urls retornados por search_history '
            'ou get_message_context. Escritas sem proveniência são recusadas. '
            'Entradas tipo "person" criadas pelo bot ficam pendentes até aprovação de um admin. '
            'Não use para opiniões, dúvidas técnicas de Minecraft ou conteúdo que já existe '
            'na documentação.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'action': {
                    'type': 'string',
                    'enum': ['create', 'update', 'archive'],
                    'description': 'create: nova entrada; update: editar existente; archive: remover (reversível, exige apenas reason).',
                },
                'term': {
                    'type': 'string',
                    'description': 'Termo canônico (ex: "Grande Incidente de 2025").',
                },
                'type': {
                    'type': 'string',
                    'enum': list(LORE_TYPES),
                    'description': 'Categoria — obrigatória em create.',
                },
                'content': {
                    'type': 'string',
                    'description': 'Descrição factual e concisa — obrigatória em create.',
                },
                'aliases': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'Apelidos/variantes pelo qual o termo é conhecido na comunidade (fortemente recomendado).',
                },
                'sources': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'Links (discord.com/channels/...) das mensagens que comprovam o fato. Obrigatório em create e update.',
                },
                'reason': {
                    'type': 'string',
                    'description': 'Justificativa curta, uma linha. Sempre obrigatória.',
                },
            },
            'required': ['action', 'term', 'reason'],
        },
    },
}


class LoreError(Exception):
    """User-facing lore operation error."""


def _norm(text: str) -> str:
    text = unicodedata.normalize('NFKD', text or '')
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', text).strip().lower()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse_aliases(raw) -> list[str]:
    if isinstance(raw, str):
        raw = raw.split(',')
    result: list[str] = []
    for a in raw or []:
        a = str(a).strip()[:_MAX_ALIAS]
        if len(a) >= 2 and a not in result:
            result.append(a)
    return result[:10]


def _parse_sources(raw) -> list[str]:
    if isinstance(raw, str):
        raw = raw.split(',')
    result: list[str] = []
    for s in raw or []:
        s = str(s).strip()
        if s.startswith('http') and s not in result:
            result.append(s)
    return result[:6]


def _is_discord_jump_url(url: str, guild_id: int | None = None) -> bool:
    if not url.startswith('https://discord.com/channels/'):
        return False
    if guild_id is None:
        return True
    return url.startswith(f'https://discord.com/channels/{guild_id}/')


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


def _entry_from_row(r: sqlite3.Row) -> dict:
    embedding = None
    if r['embedding']:
        embedding = np.frombuffer(r['embedding'], dtype=np.float32)
    return {
        'id': r['id'],
        'guild_id': r['guild_id'],
        'term': r['term'],
        'type': r['type'],
        'content': r['content'],
        'aliases': json.loads(r['aliases'] or '[]'),
        'sources': json.loads(r['sources'] or '[]'),
        'status': r['status'],
        'archived': bool(r['archived']),
        'created_by': r['created_by'],
        'updated_by': r['updated_by'],
        'created_at': r['created_at'],
        'updated_at': r['updated_at'],
        'embedding': embedding,
    }


def _snapshot(entry: dict) -> dict:
    return {k: entry[k] for k in _SNAPSHOT_FIELDS}


class LoreStore:
    """SQLite store — the single mutation path, entry write + history in one transaction."""

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
                CREATE TABLE IF NOT EXISTS lore_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    term TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'glossary',
                    content TEXT NOT NULL,
                    aliases TEXT NOT NULL DEFAULT '[]',
                    sources TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'auto',
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_by TEXT NOT NULL DEFAULT '',
                    updated_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    embedding BLOB
                )
            """)
            con.execute(
                'CREATE UNIQUE INDEX IF NOT EXISTS idx_lore_term '
                'ON lore_entries(guild_id, term) WHERE archived = 0'
            )
            con.execute('CREATE INDEX IF NOT EXISTS idx_lore_guild ON lore_entries(guild_id)')
            con.execute("""
                CREATE TABLE IF NOT EXISTS lore_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    actor_name TEXT NOT NULL DEFAULT '',
                    before TEXT,
                    after TEXT,
                    reason TEXT NOT NULL DEFAULT '',
                    sources TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
            """)
            con.execute('CREATE INDEX IF NOT EXISTS idx_lore_hist_entry ON lore_history(entry_id)')
            con.execute('CREATE INDEX IF NOT EXISTS idx_lore_hist_guild ON lore_history(guild_id)')
            con.commit()
        finally:
            con.close()

    def list_entries(self, guild_id: int) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                'SELECT * FROM lore_entries WHERE guild_id=? ORDER BY term COLLATE NOCASE, id DESC',
                (guild_id,),
            ).fetchall()
        finally:
            con.close()
        return [_entry_from_row(r) for r in rows]

    def get_entry(self, guild_id: int, entry_id: int) -> dict | None:
        con = self._connect()
        try:
            row = con.execute(
                'SELECT * FROM lore_entries WHERE id=? AND guild_id=?',
                (entry_id, guild_id),
            ).fetchone()
        finally:
            con.close()
        return _entry_from_row(row) if row else None

    def find_by_term(self, guild_id: int, term: str, *, include_archived: bool = False) -> dict | None:
        """Find an entry matching *term* — exact term match wins over alias match,
        active matches win over archived ones."""
        target = _norm(term)
        if not target:
            return None
        entries = self.list_entries(guild_id)
        active: dict | None = None
        archived: dict | None = None

        def consider(entry: dict) -> None:
            nonlocal active, archived
            if entry['archived']:
                archived = archived or entry
            elif active is None:
                active = entry

        for entry in entries:
            if _norm(entry['term']) == target:
                consider(entry)
        if active is None and archived is None:
            for entry in entries:
                if any(_norm(a) == target for a in entry['aliases']):
                    consider(entry)
        if active is not None:
            return active
        return archived if include_archived else None

    def create_with_history(
        self, guild_id: int, *, term: str, etype: str, content: str,
        aliases: list[str], sources: list[str], status: str,
        actor: str, actor_name: str, reason: str, embedding=None,
    ) -> tuple[dict, int]:
        """Insert the entry and its 'create' history row in one transaction."""
        now = _now_iso()
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        con = self._connect()
        try:
            cur = con.execute(
                'INSERT INTO lore_entries '
                '(guild_id, term, type, content, aliases, sources, status, archived, '
                'created_by, updated_by, created_at, updated_at, embedding) '
                'VALUES (?,?,?,?,?,?,?,0,?,?,?,?,?)',
                (
                    guild_id, term, etype, content,
                    json.dumps(aliases, ensure_ascii=False),
                    json.dumps(sources, ensure_ascii=False),
                    status, actor, actor, now, now, blob,
                ),
            )
            entry_id = cur.lastrowid
            row = con.execute('SELECT * FROM lore_entries WHERE id=?', (entry_id,)).fetchone()
            entry = _entry_from_row(row)
            after = _snapshot(entry)
            hist = con.execute(
                'INSERT INTO lore_history '
                '(entry_id, guild_id, action, actor, actor_name, before, after, reason, sources, created_at) '
                'VALUES (?,?,?,?,?,?,?,?,?,?)',
                (
                    entry_id, guild_id, 'create', actor, actor_name, None,
                    json.dumps(after, ensure_ascii=False), reason,
                    json.dumps(sources, ensure_ascii=False), now,
                ),
            )
            con.commit()
            return entry, hist.lastrowid or 0
        except sqlite3.IntegrityError as e:
            try:
                con.rollback()
            except Exception:
                pass
            raise LoreError(f"O termo '{term}' já existe.") from e
        except Exception:
            try:
                con.rollback()
            except Exception:
                pass
            raise
        finally:
            con.close()

    def update_with_history(
        self, guild_id: int, entry_id: int, patch: dict, *,
        action: str, actor: str, actor_name: str, reason: str,
        sources: list[str] | None = None, embedding=None,
    ) -> tuple[dict, dict, int, list[str]]:
        """Apply *patch* and record the history row in one transaction.

        The before-snapshot and the sources merge are derived from a fresh read
        inside the transaction, so concurrent writers cannot be silently
        overwritten by stale state. Returns (updated, before, history_id, sources).
        """
        cols = {'term', 'type', 'content', 'aliases', 'status', 'archived'}
        con = self._connect()
        try:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute(
                'SELECT * FROM lore_entries WHERE id=? AND guild_id=?',
                (entry_id, guild_id),
            ).fetchone()
            if row is None:
                con.rollback()
                raise LoreError('Entrada não encontrada.')
            current = _entry_from_row(row)
            before = _snapshot(current)
            merged_sources = current['sources']
            sets: list[str] = ['updated_by=?', 'updated_at=?']
            vals: list = [actor, _now_iso()]
            for k, v in patch.items():
                if k not in cols:
                    continue
                if k == 'aliases':
                    v = json.dumps(v, ensure_ascii=False)
                sets.append(f'{k}=?')
                vals.append(v)
            if sources:
                merged = list(current['sources'])
                for s in sources:
                    if s not in merged:
                        merged.append(s)
                merged_sources = merged[:8]
                sets.append('sources=?')
                vals.append(json.dumps(merged_sources, ensure_ascii=False))
            if embedding is not None:
                sets.append('embedding=?')
                vals.append(embedding.astype(np.float32).tobytes())
            vals.extend([entry_id, guild_id])
            con.execute(
                f"UPDATE lore_entries SET {', '.join(sets)} WHERE id=? AND guild_id=?",
                vals,
            )
            row2 = con.execute('SELECT * FROM lore_entries WHERE id=?', (entry_id,)).fetchone()
            updated = _entry_from_row(row2)
            after = _snapshot(updated)
            hist = con.execute(
                'INSERT INTO lore_history '
                '(entry_id, guild_id, action, actor, actor_name, before, after, reason, sources, created_at) '
                'VALUES (?,?,?,?,?,?,?,?,?,?)',
                (
                    entry_id, guild_id, action, actor, actor_name,
                    json.dumps(before, ensure_ascii=False),
                    json.dumps(after, ensure_ascii=False),
                    reason,
                    json.dumps(merged_sources, ensure_ascii=False),
                    _now_iso(),
                ),
            )
            con.commit()
            return updated, before, hist.lastrowid or 0, merged_sources
        except sqlite3.IntegrityError as e:
            try:
                con.rollback()
            except Exception:
                pass
            raise LoreError('Conflito: já existe outra entrada com este termo.') from e
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
                'SELECT * FROM lore_history WHERE id=? AND guild_id=?',
                (history_id, guild_id),
            ).fetchone()
        finally:
            con.close()

    def history_for_entry(self, guild_id: int, entry_id: int, limit: int = 8) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                'SELECT id, action, actor, actor_name, reason, created_at FROM lore_history '
                'WHERE entry_id=? AND guild_id=? ORDER BY id DESC LIMIT ?',
                (entry_id, guild_id, limit),
            ).fetchall()
        finally:
            con.close()
        return [dict(r) for r in rows]

    def history_for_guild(self, guild_id: int) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                'SELECT * FROM lore_history WHERE guild_id=? ORDER BY id ASC',
                (guild_id,),
            ).fetchall()
        finally:
            con.close()
        return [dict(r) for r in rows]


class LoreLogView(discord.ui.View):
    """Revert / Lock / Approve buttons attached to log-channel mutation embeds."""

    def __init__(self, cog: 'Lore', guild_id: int, entry_id: int, history_id: int, pending: bool):
        super().__init__(timeout=1800)
        self.cog = cog
        self.guild_id = guild_id
        self.entry_id = entry_id
        self.history_id = history_id
        if not pending:
            self.remove_item(self.approve_btn)

    def _allowed(self, interaction: discord.Interaction) -> bool:
        if not isinstance(interaction.user, discord.Member):
            return False
        if interaction.guild_id != self.guild_id:
            return False
        return (
            interaction.user.guild_permissions.administrator
            or interaction.user.guild_permissions.manage_guild
        )

    async def _deny(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            'Apenas administradores deste servidor podem usar estes botões.', ephemeral=True
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

    @discord.ui.button(label='↩️ Reverter', style=discord.ButtonStyle.secondary)
    async def revert_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._allowed(interaction):
            return await self._deny(interaction)
        try:
            entry = await self.cog.revert(
                self.guild_id, self.history_id,
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except LoreError as e:
            return await self._done(interaction, f'⚠️ {e}')
        except Exception:
            logger.exception("[lore] revert button failed")
            return await self._done(interaction, '⚠️ Erro interno ao reverter.')
        await self._done(
            interaction,
            f'↩️ Revertido por {interaction.user.display_name} — "{entry["term"]}" restaurado ao estado anterior.',
        )

    @discord.ui.button(label='🔒 Proteger', style=discord.ButtonStyle.secondary)
    async def lock_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._allowed(interaction):
            return await self._deny(interaction)
        try:
            entry = await self.cog.set_locked(
                self.guild_id, self.entry_id, locked=True,
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except LoreError as e:
            return await self._done(interaction, f'⚠️ {e}')
        except Exception:
            logger.exception("[lore] lock button failed")
            return await self._done(interaction, '⚠️ Erro interno ao proteger.')
        await self._done(
            interaction,
            f'🔒 "{entry["term"]}" protegida por {interaction.user.display_name} — o bot não pode mais editá-la.',
        )

    @discord.ui.button(label='✅ Aprovar', style=discord.ButtonStyle.success)
    async def approve_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._allowed(interaction):
            return await self._deny(interaction)
        try:
            entry = await self.cog.approve(
                self.guild_id, self.entry_id,
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except LoreError as e:
            return await self._done(interaction, f'⚠️ {e}')
        except Exception:
            logger.exception("[lore] approve button failed")
            return await self._done(interaction, '⚠️ Erro interno ao aprovar.')
        await self._done(
            interaction,
            f'✅ "{entry["term"]}" aprovada por {interaction.user.display_name} — agora aparece nas buscas.',
        )


class ConfirmArchiveView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=60)
        self.confirmed = False

    @discord.ui.button(label='🗑️ Arquivar', style=discord.ButtonStyle.danger)
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


class Lore(commands.Cog, name='Lore'):
    """Server lore encyclopedia: bot-managed entries with full admin control."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.store = LoreStore(LORE_DB_PATH)
        self._bot_writes: dict[int, deque[float]] = {}
        self._log_tasks: set[asyncio.Task] = set()

    async def cog_load(self):
        try:
            await asyncio.to_thread(self.store.ensure)
        except Exception:
            logger.exception("Lore DB init failed — lore commands/tools will degrade to error messages")

    # --- Embeddings (reuse DocsRAG / HistoryRAG clients) ---

    async def _embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        for cog_name in ('DocsRAG', 'HistoryRAG'):
            cog = self.bot.get_cog(cog_name)
            if cog is not None and hasattr(cog, '_embed_batch'):
                try:
                    return await cog._embed_batch(texts)
                except Exception:
                    logger.exception("[lore] embedding via %s failed", cog_name)
        return None

    async def _embed_entry(self, term: str, aliases: list[str], content: str):
        text = f"{term}\nApelidos: {', '.join(aliases)}\n{content[:1000]}"
        result = await self._embed_texts([text])
        if not result:
            return None
        return np.array(result[0], dtype=np.float32)

    # --- Search (silent: console-logged, no sources returned) ---

    async def search(self, guild_id: int, query: str, limit: int = 3) -> list[tuple[dict, float, str]]:
        entries = [
            e for e in await asyncio.to_thread(self.store.list_entries, guild_id)
            if not e['archived'] and e['status'] != 'pending'
        ]
        if not entries:
            return []
        qn = _norm(query)
        scored: list[tuple[dict, float, str]] = []
        for e in entries:
            keys = [_norm(e['term'])] + [_norm(a) for a in e['aliases']]
            if any(k == qn for k in keys):
                scored.append((e, 3.0, 'exact'))
            elif len(qn) >= _MIN_SUBSTRING_LEN and any(
                len(k) >= _MIN_SUBSTRING_LEN and (k in qn or qn in k) for k in keys
            ):
                scored.append((e, 2.0, 'alias'))
            elif len(qn) >= _MIN_SUBSTRING_LEN and qn in _norm(e['content']):
                scored.append((e, 1.5, 'content'))
        scored.sort(key=lambda t: t[1], reverse=True)
        if len(scored) >= limit:
            return scored[:limit]
        # Semantic fallback only fills empty slots (score <= 1.0 never beats lexical hits)
        already = {e['id'] for e, _s, _h in scored}
        candidates = [e for e in entries if e['embedding'] is not None and e['id'] not in already]
        if candidates:
            q_emb = await self._embed_texts([query])
            if q_emb:
                q_arr = np.array(q_emb[0], dtype=np.float32)
                q_norm = float(np.linalg.norm(q_arr))
                for e in candidates:
                    if e['embedding'].shape[0] != q_arr.shape[0]:
                        continue
                    denom = float(np.linalg.norm(e['embedding'])) * q_norm
                    if denom > 0:
                        score = float(np.dot(e['embedding'], q_arr) / denom)
                        if score >= _SEMANTIC_MIN_SCORE:
                            scored.append((e, score, 'semantic'))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

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
    ) -> tuple[str, list[dict]]:
        try:
            if name == 'search_lore':
                return await self._exec_search(args, guild=guild, requester=requester, channel=channel)
            if name == 'save_lore':
                return await self._exec_save(
                    args, guild=guild, actor_id=actor_id, actor_name=actor_name,
                )
        except LoreError as e:
            return f'⚠️ Lore: {e}', []
        except sqlite3.Error as e:
            logger.exception("[lore] database error in %s", name)
            return f'⚠️ Erro no banco de lore: {e}', []
        except Exception:
            logger.exception("[lore] unexpected error in %s", name)
            return '⚠️ Erro interno na enciclopédia de lore. Continue sem o lore.', []
        return f'Ferramenta desconhecida: {name}', []

    async def _exec_search(
        self, args: dict, *, guild: discord.Guild, requester=None, channel=None,
    ) -> tuple[str, list[dict]]:
        query = (args.get('query') or '').strip()
        try:
            limit = max(1, min(6, int(args.get('limit', 3))))
        except (TypeError, ValueError):
            limit = 3
        if not query:
            return 'Consulta vazia.', []
        hits = await self.search(guild.id, query, limit)
        ctx = ''
        if requester is not None:
            ctx = f' by @{getattr(requester, "name", "?")}'
        if channel is not None:
            ctx += f' in #{getattr(channel, "name", "?")}'
        logger.info("[lore] search %r%s -> %d hit(s)", query, ctx, len(hits))
        for entry, score, how in hits:
            logger.info(
                "[lore]   hit: %r (id=%d, aliases=+%d, type=%s, score=%.2f, match=%s)",
                entry['term'], entry['id'], len(entry['aliases']),
                entry['type'], score, how,
            )
        if not hits:
            return (
                'Nenhuma entrada encontrada para esta consulta. '
                'Se o histórico do servidor tiver algo digno de memória, '
                'você pode registrar com save_lore (exija fontes).'
            ), []
        parts = []
        for entry, _score, _how in hits:
            aliases = (
                f" | Apelidos: {', '.join(entry['aliases'])}"
                if entry['aliases'] else ''
            )
            sources = '\n'.join(f'- {s}' for s in entry['sources'][:4])
            updated = entry['updated_at'][:10]
            parts.append(
                f"**{entry['term']}** ({_TYPE_LABELS.get(entry['type'], entry['type'])}){aliases}\n"
                f"{entry['content']}\n"
                f"Atualizada em {updated}. Fontes:\n{sources or '- proveniência não registrada'}"
            )
        return '\n\n---\n\n'.join(parts), []

    # --- Bot write rate limiting (peek first, debit only on actual writes) ---

    def _bot_write_slots(self, guild_id: int) -> bool:
        now = time.monotonic()
        dq = self._bot_writes.setdefault(guild_id, deque())
        while dq and now - dq[0] > 3600:
            dq.popleft()
        return len(dq) < LORE_BOT_WRITE_LIMIT

    def _bot_write_debit(self, guild_id: int) -> None:
        self._bot_writes.setdefault(guild_id, deque()).append(time.monotonic())

    async def _exec_save(
        self, args: dict, *, guild: discord.Guild, actor_id: str, actor_name: str,
    ) -> tuple[str, list[dict]]:
        if actor_id == 'bot' and not self._bot_write_slots(guild.id):
            return 'Limite de escritas do bot por hora atingido. Tente novamente mais tarde.', []
        action = args.get('action')
        term = (args.get('term') or '').strip()[:_MAX_TERM]
        reason = (args.get('reason') or '').strip()[:_MAX_REASON]
        if action not in ('create', 'update', 'archive'):
            return 'action inválida — use create, update ou archive.', []
        if not term:
            return '`term` é obrigatório.', []
        if not reason:
            return '`reason` é obrigatório (uma linha).', []
        needs_sources = action in ('create', 'update')
        sources = _parse_sources(args.get('sources')) if needs_sources else []
        if needs_sources:
            bad_sources = [s for s in sources if not _is_discord_jump_url(s, guild.id)]
            sources = [s for s in sources if _is_discord_jump_url(s, guild.id)]
            if bad_sources or not sources:
                return (
                    'Escrita recusada: `sources` deve conter links no formato '
                    f'discord.com/channels/{guild.id}/<canal>/<mensagem> das mensagens que comprovam '
                    'o fato. Use os jump_urls retornados por search_history ou get_message_context.'
                ), []

        entry = await asyncio.to_thread(
            self.store.find_by_term, guild.id, term, include_archived=True,
        )

        if action == 'create':
            if entry and not entry['archived']:
                return (
                    f"O termo '{entry['term']}' já existe (id={entry['id']}). "
                    'Use action=update para editá-lo.'
                ), []
            etype = args.get('type') or 'glossary'
            if etype not in LORE_TYPES:
                return f'type inválido. Use um destes: {", ".join(LORE_TYPES)}.', []
            content = (args.get('content') or '').strip()[:_MAX_CONTENT]
            if not content:
                return '`content` é obrigatório em create.', []
            aliases = _parse_aliases(args.get('aliases'))
            status = 'pending' if actor_id == 'bot' and etype == 'person' else 'auto'
            if actor_id == 'bot':
                self._bot_write_debit(guild.id)
            entry = await self._create(
                guild.id, term=term, etype=etype, content=content,
                aliases=aliases, sources=sources, status=status,
                actor_id=actor_id, actor_name=actor_name, reason=reason,
            )
            note = (
                ' — pendente: entradas sobre pessoas só aparecem nas buscas após aprovação de um admin.'
                if status == 'pending' else ''
            )
            return f"Entrada criada: '{entry['term']}' (id={entry['id']}, type={etype}, status={status}){note}", []

        if action == 'update':
            if not entry or entry['archived']:
                return f"Nenhuma entrada ativa encontrada para '{term}'. Use action=create.", []
            if entry['status'] == 'curated' and actor_id == 'bot':
                return (
                    f"A entrada '{entry['term']}' está protegida (curated) — "
                    'apenas administradores podem editá-la.'
                ), []
            patch: dict = {}
            content = (args.get('content') or '').strip()
            if content:
                patch['content'] = content[:_MAX_CONTENT]
            if args.get('aliases') is not None:
                patch['aliases'] = _parse_aliases(args.get('aliases'))
            if args.get('type') in LORE_TYPES:
                patch['type'] = args['type']
            if not patch:
                return 'Nada para atualizar — forneça content, aliases e/ou type.', []
            if actor_id == 'bot':
                self._bot_write_debit(guild.id)
            entry = await self._apply(
                guild.id, entry, patch, action='update',
                actor_id=actor_id, actor_name=actor_name,
                reason=reason, sources=sources,
            )
            return f"Entrada atualizada: '{entry['term']}' (id={entry['id']}).", []

        if action == 'archive':
            if not entry or entry['archived']:
                return f"Nenhuma entrada ativa encontrada para '{term}'.", []
            if entry['status'] == 'curated' and actor_id == 'bot':
                return (
                    f"A entrada '{entry['term']}' está protegida (curated) — "
                    'apenas administradores podem removê-la.'
                ), []
            if actor_id == 'bot':
                self._bot_write_debit(guild.id)
            entry = await self._apply(
                guild.id, entry, {'archived': True}, action='archive',
                actor_id=actor_id, actor_name=actor_name, reason=reason, sources=[],
            )
            return f"Entrada arquivada: '{entry['term']}' (id={entry['id']}) — reversível por admins.", []

        return 'action inválida.', []

    # --- Mutation helpers (single transaction + log channel) ---

    async def _create(
        self, guild_id: int, *, term: str, etype: str, content: str,
        aliases: list[str], sources: list[str], status: str,
        actor_id: str, actor_name: str, reason: str,
    ) -> dict:
        embedding = await self._embed_entry(term, aliases, content)
        entry, history_id = await asyncio.to_thread(
            self.store.create_with_history, guild_id,
            term=term, etype=etype, content=content, aliases=aliases,
            sources=sources, status=status, actor=actor_id,
            actor_name=actor_name, reason=reason, embedding=embedding,
        )
        after = _snapshot(entry)
        logger.info("[lore] create '%s' (id=%d, status=%s) by %s", term, entry['id'], status, actor_name)
        self._schedule_log(
            guild_id, action='create', entry=entry, before=None, after=after,
            actor_name=actor_name, reason=reason, sources=sources, history_id=history_id,
        )
        return entry

    async def _apply(
        self, guild_id: int, entry: dict, patch: dict, *, action: str,
        actor_id: str, actor_name: str, reason: str, sources: list[str],
    ) -> dict:
        text_changed = any(k in patch for k in ('term', 'content', 'aliases'))
        embedding = None
        if text_changed:
            embedding = await self._embed_entry(
                patch.get('term', entry['term']),
                patch.get('aliases', entry['aliases']),
                patch.get('content', entry['content']),
            )
        updated, before, history_id, merged_sources = await asyncio.to_thread(
            self.store.update_with_history, guild_id, entry['id'], patch,
            action=action, actor=actor_id, actor_name=actor_name,
            reason=reason, sources=sources or None, embedding=embedding,
        )
        logger.info(
            "[lore] %s '%s' (id=%d) by %s — %s",
            action, updated['term'], updated['id'], actor_name, reason[:80],
        )
        self._schedule_log(
            guild_id, action=action, entry=updated, before=before,
            after=_snapshot(updated), actor_name=actor_name, reason=reason,
            sources=merged_sources, history_id=history_id,
        )
        return updated

    async def revert(
        self, guild_id: int, history_id: int, *, actor_id: str, actor_name: str,
    ) -> dict:
        row = await asyncio.to_thread(self.store.get_history, guild_id, history_id)
        if row is None:
            raise LoreError('Registro de histórico não encontrado.')
        entry = await asyncio.to_thread(self.store.get_entry, guild_id, row['entry_id'])
        if entry is None:
            raise LoreError('Entrada não encontrada.')
        if row['before'] is None:
            patch = {'archived': True}
        else:
            before = json.loads(row['before'])
            patch = {k: before[k] for k in _SNAPSHOT_FIELDS if k in before}
        return await self._apply(
            guild_id, entry, patch, action='revert',
            actor_id=actor_id, actor_name=actor_name,
            reason=f'Reversão do registro de histórico #{history_id}', sources=[],
        )

    async def restore_version(
        self, guild_id: int, history_id: int, *, actor_id: str, actor_name: str,
    ) -> dict:
        row = await asyncio.to_thread(self.store.get_history, guild_id, history_id)
        if row is None:
            raise LoreError('Registro de histórico não encontrado.')
        entry = await asyncio.to_thread(self.store.get_entry, guild_id, row['entry_id'])
        if entry is None:
            raise LoreError('Entrada não encontrada.')
        if row['after'] is None:
            raise LoreError('Este registro não possui um estado para restaurar.')
        after = json.loads(row['after'])
        patch = {k: after[k] for k in _SNAPSHOT_FIELDS if k in after}
        return await self._apply(
            guild_id, entry, patch, action='restore',
            actor_id=actor_id, actor_name=actor_name,
            reason=f'Restauração da versão do histórico #{history_id}', sources=[],
        )

    async def set_locked(
        self, guild_id: int, entry_id: int, *, locked: bool,
        actor_id: str, actor_name: str,
    ) -> dict:
        entry = await asyncio.to_thread(self.store.get_entry, guild_id, entry_id)
        if entry is None:
            raise LoreError('Entrada não encontrada.')
        if entry['status'] == 'pending':
            raise LoreError('Entrada pendente — aprove com ✅ Aprovar ou /lore approve antes de proteger.')
        return await self._apply(
            guild_id, entry, {'status': 'curated' if locked else 'auto'},
            action='lock' if locked else 'unlock',
            actor_id=actor_id, actor_name=actor_name,
            reason='Proteção manual' if locked else 'Desproteção manual', sources=[],
        )

    async def approve(
        self, guild_id: int, entry_id: int, *, actor_id: str, actor_name: str,
    ) -> dict:
        entry = await asyncio.to_thread(self.store.get_entry, guild_id, entry_id)
        if entry is None:
            raise LoreError('Entrada não encontrada.')
        if entry['status'] != 'pending':
            raise LoreError('A entrada não está pendente.')
        return await self._apply(
            guild_id, entry, {'status': 'auto'}, action='approve',
            actor_id=actor_id, actor_name=actor_name, reason='Aprovação manual', sources=[],
        )

    # --- Admin log channel ---

    def _schedule_log(
        self, guild_id: int, *, action: str, entry: dict, before: dict | None,
        after: dict, actor_name: str, reason: str,
        sources: list[str], history_id: int,
    ) -> None:
        task = asyncio.get_running_loop().create_task(
            self._post_log(
                guild_id, action=action, entry=entry, before=before, after=after,
                actor_name=actor_name, reason=reason, sources=sources, history_id=history_id,
            )
        )
        self._log_tasks.add(task)
        task.add_done_callback(self._log_tasks.discard)

    async def _post_log(
        self, guild_id: int, *, action: str, entry: dict, before: dict | None,
        after: dict, actor_name: str, reason: str,
        sources: list[str], history_id: int,
    ) -> None:
        if not LORE_LOG_CHANNEL_ID:
            return
        try:
            channel = self.bot.get_channel(LORE_LOG_CHANNEL_ID)
            if channel is None:
                channel = await self.bot.fetch_channel(LORE_LOG_CHANNEL_ID)
        except Exception:
            logger.exception("[lore] log channel unavailable")
            return
        colors = {
            'create': discord.Color.green(),
            'update': discord.Color.gold(),
            'archive': discord.Color.red(),
            'revert': discord.Color.dark_grey(),
            'restore': discord.Color.dark_grey(),
            'lock': discord.Color.blurple(),
            'unlock': discord.Color.blurple(),
            'approve': discord.Color.green(),
        }
        verb = _ACTION_VERBS.get(action, action)
        embed = discord.Embed(
            title=f"📖 Lore — {_snippet(entry['term'], 250)}",
            description=f"Entrada **{verb}** por **{actor_name}**",
            color=colors.get(action, discord.Color.greyple()),
        )
        embed.add_field(name='ID', value=f"{entry['id']} (hist #{history_id})", inline=True)
        embed.add_field(name='Tipo', value=_TYPE_LABELS.get(entry['type'], entry['type']), inline=True)
        embed.add_field(name='Status', value=_STATUS_LABELS.get(after.get('status', ''), '?'), inline=True)
        if reason:
            embed.add_field(name='Motivo', value=_snippet(reason, 200), inline=False)
        if action == 'update' and before:
            embed.add_field(name='Antes', value=_snippet(before.get('content', ''), 300), inline=False)
            embed.add_field(name='Depois', value=_snippet(after.get('content', ''), 300), inline=False)
        elif action in ('archive', 'revert', 'restore', 'lock', 'unlock', 'approve'):
            embed.add_field(name='Conteúdo', value=_snippet(entry['content'], 300), inline=False)
        if sources:
            links = '\n'.join(f'- {s}' for s in sources[:5])
            embed.add_field(name='Fontes', value=links[:1000], inline=False)
        embed.set_footer(text=f'Atualizada em {_fmt_brt(entry["updated_at"])}')
        view = LoreLogView(
            self, guild_id=guild_id, entry_id=entry['id'],
            history_id=history_id, pending=after.get('status') == 'pending',
        )
        try:
            await channel.send(embed=embed, view=view)
        except Exception:
            logger.exception("[lore] failed to post log embed")

    # --- Slash commands ---

    lore = app_commands.Group(name='lore', description='Enciclopédia do servidor (lore)')

    @lore.command(name='list', description='Listar entradas da enciclopédia de lore')
    @app_commands.describe(category='Filtrar por categoria')
    @app_commands.choices(category=_CATEGORY_CHOICES)
    async def lore_list(self, interaction: discord.Interaction, category: str | None = None):
        guild = interaction.guild
        if guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        admin = _is_admin(interaction)
        entries = await asyncio.to_thread(self.store.list_entries, guild.id)
        entries = [
            e for e in entries
            if not e['archived']
            and (admin or e['status'] != 'pending')
            and (category is None or e['type'] == category)
        ]
        if not entries:
            return await interaction.response.send_message(
                'Nenhuma entrada encontrada. O bot cria entradas automaticamente '
                'a partir de fatos comprovados no histórico — ou use `/lore add`.',
                ephemeral=True,
            )
        lines = []
        for e in entries:
            flags = ''
            if e['status'] == 'pending':
                flags += ' • pendente'
            if e['status'] == 'curated':
                flags += ' • protegida'
            lines.append(f"**{_snippet(e['term'], 100)}** — {_TYPE_LABELS.get(e['type'], e['type'])}{flags}")
        pages: list[discord.Embed] = []
        for i in range(0, len(lines), 12):
            pages.append(discord.Embed(
                title='📖 Enciclopédia de Lore',
                description='\n'.join(lines[i:i + 12]),
                color=discord.Color.dark_gold(),
            ))
        for i, page in enumerate(pages):
            page.set_footer(text=f'{len(entries)} entradas • Página {i + 1}/{len(pages)}')
        if len(pages) == 1:
            await interaction.response.send_message(embed=pages[0])
        else:
            await interaction.response.send_message(embed=pages[0], view=PaginatedEmbedView(pages))

    @lore.command(name='show', description='Ver uma entrada da enciclopédia')
    @app_commands.describe(term='Termo ou apelido da entrada')
    async def lore_show(self, interaction: discord.Interaction, term: str):
        guild = interaction.guild
        if guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        admin = _is_admin(interaction)
        entry = await asyncio.to_thread(self.store.find_by_term, guild.id, term, include_archived=True)
        if entry is None or (not admin and (entry['archived'] or entry['status'] == 'pending')):
            return await interaction.response.send_message(f'Nenhuma entrada encontrada para "{term}".', ephemeral=True)
        embed = discord.Embed(
            title=f"📖 {_snippet(entry['term'], 250)}",
            description=_snippet(entry['content'], 3900),
            color=discord.Color.dark_gold(),
        )
        embed.add_field(name='Tipo', value=_TYPE_LABELS.get(entry['type'], entry['type']), inline=True)
        embed.add_field(name='Status', value=_STATUS_LABELS.get(entry['status'], entry['status']), inline=True)
        embed.add_field(name='Arquivada', value='sim' if entry['archived'] else 'não', inline=True)
        if entry['aliases']:
            embed.add_field(name='Apelidos', value=_snippet(', '.join(entry['aliases']), 1000), inline=False)
        if entry['sources']:
            embed.add_field(
                name='Fontes',
                value='\n'.join(f'- {s}' for s in entry['sources'][:5])[:1000],
                inline=False,
            )
        embed.set_footer(text=f'Atualizada em {_fmt_brt(entry["updated_at"])}')
        if admin:
            history = await asyncio.to_thread(
                self.store.history_for_entry, guild.id, entry['id'], 5,
            )
            if history:
                hist_lines = [
                    f'`#{h["id"]}` {h["action"]} — {h["actor_name"]} — {_fmt_brt(h["created_at"])}'
                    for h in history
                ]
                embed.add_field(
                    name='Histórico (use /lore restore <id>)',
                    value='\n'.join(hist_lines),
                    inline=False,
                )
        await interaction.response.send_message(embed=embed)

    @lore.command(name='add', description='Adicionar uma entrada à enciclopédia (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(
        term='Termo canônico',
        category='Categoria da entrada',
        content='Descrição factual',
        aliases='Apelidos separados por vírgula (opcional)',
        sources='Links das mensagens de origem, separados por vírgula (opcional)',
    )
    @app_commands.choices(category=_CATEGORY_CHOICES)
    async def lore_add(
        self,
        interaction: discord.Interaction,
        term: str,
        category: str,
        content: str,
        aliases: str | None = None,
        sources: str | None = None,
    ):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        term = term.strip()[:_MAX_TERM]
        if not term:
            return await interaction.response.send_message('Termo inválido.', ephemeral=True)
        if category not in LORE_TYPES:
            return await interaction.response.send_message('Categoria inválida.', ephemeral=True)
        try:
            entry = await self._create(
                interaction.guild.id,
                term=term,
                etype=category,
                content=content.strip()[:_MAX_CONTENT],
                aliases=_parse_aliases(aliases),
                sources=_parse_sources(sources),
                status='auto',
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
                reason='Criação manual via /lore add',
            )
        except LoreError as e:
            return await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
        await interaction.response.send_message(f"✅ Entrada criada: **{entry['term']}** (id={entry['id']}).")

    @lore.command(name='edit', description='Editar uma entrada da enciclopédia (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(
        term='Termo ou apelido da entrada',
        content='Nova descrição (opcional)',
        aliases='Novos apelidos separados por vírgula (opcional)',
        category='Nova categoria (opcional)',
    )
    @app_commands.choices(category=_CATEGORY_CHOICES)
    async def lore_edit(
        self,
        interaction: discord.Interaction,
        term: str,
        content: str | None = None,
        aliases: str | None = None,
        category: str | None = None,
    ):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        if content is None and aliases is None and category is None:
            return await interaction.response.send_message(
                'Informe ao menos um campo para editar: `content`, `aliases` ou `category`.',
                ephemeral=True,
            )
        entry = await asyncio.to_thread(self.store.find_by_term, interaction.guild.id, term)
        if entry is None:
            return await interaction.response.send_message(f'Nenhuma entrada ativa encontrada para "{term}".', ephemeral=True)
        patch: dict = {}
        if content is not None and content.strip():
            patch['content'] = content.strip()[:_MAX_CONTENT]
        if aliases is not None:
            patch['aliases'] = _parse_aliases(aliases)
        if category is not None and category in LORE_TYPES:
            patch['type'] = category
        if not patch:
            return await interaction.response.send_message('Nada para atualizar.', ephemeral=True)
        try:
            entry = await self._apply(
                interaction.guild.id, entry, patch, action='update',
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
                reason='Edição manual via /lore edit', sources=[],
            )
        except LoreError as e:
            return await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
        await interaction.response.send_message(f"✅ Entrada atualizada: **{entry['term']}** (id={entry['id']}).")

    @lore.command(name='delete', description='Arquivar uma entrada (reversível) (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(term='Termo ou apelido da entrada')
    async def lore_delete(self, interaction: discord.Interaction, term: str):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        entry = await asyncio.to_thread(self.store.find_by_term, interaction.guild.id, term)
        if entry is None:
            return await interaction.response.send_message(f'Nenhuma entrada ativa encontrada para "{term}".', ephemeral=True)
        view = ConfirmArchiveView()
        await interaction.response.send_message(
            f"Arquivar **{entry['term']}** (id={entry['id']})? A ação é reversível via histórico.",
            view=view,
            ephemeral=True,
        )
        await view.wait()
        if not view.confirmed:
            return
        try:
            await self._apply(
                interaction.guild.id, entry, {'archived': True}, action='archive',
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
                reason='Arquivamento manual via /lore delete', sources=[],
            )
        except LoreError as e:
            return await interaction.followup.send(f'⚠️ {e}', ephemeral=True)
        await interaction.followup.send(f"🗑️ **{entry['term']}** arquivada.", ephemeral=True)

    @lore.command(name='restore', description='Restaurar uma versão anterior pelo ID do histórico (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(id='ID do registro de histórico (veja /lore show)')
    async def lore_restore(self, interaction: discord.Interaction, id: int):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        await interaction.response.defer(ephemeral=True)
        try:
            entry = await self.restore_version(
                interaction.guild.id, id,
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except LoreError as e:
            return await interaction.followup.send(f'⚠️ {e}', ephemeral=True)
        await interaction.followup.send(f"✅ **{entry['term']}** restaurada à versão do histórico #{id}.")

    @lore.command(name='approve', description='Aprovar uma entrada pendente (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(term='Termo ou apelido da entrada pendente')
    async def lore_approve(self, interaction: discord.Interaction, term: str):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        entry = await asyncio.to_thread(self.store.find_by_term, interaction.guild.id, term)
        if entry is None:
            return await interaction.response.send_message(f'Nenhuma entrada ativa encontrada para "{term}".', ephemeral=True)
        try:
            entry = await self.approve(
                interaction.guild.id, entry['id'],
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except LoreError as e:
            return await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
        await interaction.response.send_message(f"✅ **{entry['term']}** aprovada — agora aparece nas buscas.")

    @lore.command(name='lock', description='Proteger uma entrada contra edições do bot (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(term='Termo ou apelido da entrada')
    async def lore_lock(self, interaction: discord.Interaction, term: str):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        entry = await asyncio.to_thread(self.store.find_by_term, interaction.guild.id, term)
        if entry is None:
            return await interaction.response.send_message(f'Nenhuma entrada ativa encontrada para "{term}".', ephemeral=True)
        try:
            entry = await self.set_locked(
                interaction.guild.id, entry['id'], locked=True,
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except LoreError as e:
            return await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
        await interaction.response.send_message(f"🔒 **{entry['term']}** protegida — o bot não pode mais editá-la.")

    @lore.command(name='unlock', description='Permitir novamente edições do bot (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(term='Termo ou apelido da entrada')
    async def lore_unlock(self, interaction: discord.Interaction, term: str):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        entry = await asyncio.to_thread(self.store.find_by_term, interaction.guild.id, term)
        if entry is None:
            return await interaction.response.send_message(f'Nenhuma entrada ativa encontrada para "{term}".', ephemeral=True)
        try:
            entry = await self.set_locked(
                interaction.guild.id, entry['id'], locked=False,
                actor_id=str(interaction.user.id),
                actor_name=interaction.user.display_name,
            )
        except LoreError as e:
            return await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
        await interaction.response.send_message(f"🔓 **{entry['term']}** desprotegida.")

    @lore.command(name='export', description='Exportar a enciclopédia em JSON (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_export(self, interaction: discord.Interaction):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        await interaction.response.defer(ephemeral=True)
        entries = await asyncio.to_thread(self.store.list_entries, interaction.guild.id)
        history = await asyncio.to_thread(self.store.history_for_guild, interaction.guild.id)
        data = {
            'exported_at': _now_iso(),
            'guild_id': interaction.guild.id,
            'entries': [
                {k: v for k, v in e.items() if k != 'embedding'} for e in entries
            ],
            'history': history,
        }
        content = json.dumps(data, ensure_ascii=False, indent=2)
        file = discord.File(
            io.BytesIO(content.encode('utf-8')),
            filename=f'lore-{interaction.guild.id}.json',
        )
        await interaction.followup.send(
            f'📦 Exportadas {len(entries)} entradas e {len(history)} registros de histórico.',
            file=file,
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    if not LORE_ENABLED:
        logger.info('Lore disabled via LORE_ENABLED=false')
        return
    await bot.add_cog(Lore(bot))
