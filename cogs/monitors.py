"""Monitors — standing triggers that call the bot on matching chat messages.

A monitor watches the guild's messages and, when one matches its filters, runs
the bot on that message the same way an @mention does: the same ``_run_chat``
pipeline and toolset, the answer goes out as a reply, and replying to it
continues the conversation.  The model is told the message reached it through
a monitor (nobody called it) and may stay silent by answering
``NO_REPLY_TOKEN`` — nothing is posted then, though it can still act through
its tools (react, write a memory, schedule something…).

Filters (all optional, combined with AND; at least one is required)
-------------------------------------------------------------------
- **channels**: the message is in one of these channels, or a thread under one
- **authors**: the message was sent by one of these users
- **mentions**: the message mentions one of these users
- **regex**: the message content matches (``re.search``, case-insensitive)

Messages from bots never trigger monitors, and neither do messages the bot
already answers (an @mention of the bot, a reply to one of its conversations).
When several monitors match one message they share a single run.  Each
monitor has a cooldown so a busy channel can't turn into one LLM call per
message.

Monitors are managed with ``/monitor`` and by the bot itself through the
``monitor_*`` tools; creating, editing and removing them needs Manage Server.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import sqlite3
import time
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from cogs.utils import BR_TZ, EMPTY_ANSWER_FALLBACK, PaginatedEmbedView
from config import (
    CHAT_MENTION_ENABLED,
    MONITORS_DB_PATH,
    MONITORS_DEFAULT_COOLDOWN,
    MONITORS_ENABLED,
    MONITORS_MAX_PER_GUILD,
    MONITORS_MAX_PROMPT,
    MONITORS_MIN_COOLDOWN,
)

logger = logging.getLogger(__name__)

# What the model answers to stay silent on a monitor trigger.
NO_REPLY_TOKEN = '[NO_REPLY]'

_MAX_NAME = 80
_MAX_REGEX = 200
_MAX_COOLDOWN = 24 * 3600
_MAX_IDS = 25
_FILTER_FIELDS = ('channels', 'authors', 'mentions', 'regex')
_LIST_FIELDS = ('channels', 'authors', 'mentions')
# A quantified group that itself contains a quantifier — "(a+)+", "(\w*x)*" —
# is the classic catastrophic-backtracking shape; Python's re has no timeout.
_NESTED_QUANTIFIER_RE = re.compile(r'\((?:[^()\\]|\\.)*[*+}](?:[^()\\]|\\.)*\)(?:[*+]|\{\d*,)')


class MonitorError(Exception):
    """User-facing monitor error."""


def _now_iso() -> str:
    return datetime.datetime.now(BR_TZ).isoformat()


def _fmt_brt(iso: str | None) -> str:
    if not iso:
        return '—'
    try:
        return datetime.datetime.fromisoformat(iso).astimezone(BR_TZ).strftime('%d/%m/%Y %H:%M')
    except ValueError:
        return iso


def _snippet(text: str, n: int = 350) -> str:
    text = (text or '').strip()
    return text if len(text) <= n else text[:n] + '…'


def can_manage(member) -> bool:
    """Whether *member* may create, edit or remove monitors (Manage Server or admin)."""
    perms = getattr(member, 'guild_permissions', None)
    return bool(perms and (perms.administrator or perms.manage_guild))


def is_silent(answer: str | None) -> bool:
    """Whether a monitor run's *answer* means "don't reply"."""
    text = (answer or '').strip()
    return not text or text == EMPTY_ANSWER_FALLBACK or 'NO_REPLY' in text.upper()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compile_regex(pattern: str) -> re.Pattern:
    """Compile a monitor regex, rejecting overlong or backtracking-prone patterns."""
    if len(pattern) > _MAX_REGEX:
        raise MonitorError(f'regex excede {_MAX_REGEX} caracteres.')
    if _NESTED_QUANTIFIER_RE.search(pattern):
        raise MonitorError(
            'regex com quantificador aninhado (ex.: "(a+)+") não é permitida — '
            'ela pode travar o bot. Simplifique o padrão.'
        )
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise MonitorError(f'regex inválida: {e}')


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [v for v in re.split(r'[\s,]+', value) if v]
    return [value]


def _resolve_channel(value, guild) -> int:
    text = str(value).strip()
    m = re.fullmatch(r'<#(\d+)>|(\d+)', text)
    if m:
        cid = int(m.group(1) or m.group(2))
    else:
        name = text.lstrip('#').lower()
        found = None
        if guild is not None:
            for ch in list(getattr(guild, 'channels', [])) + list(getattr(guild, 'threads', [])):
                if (getattr(ch, 'name', '') or '').lower() == name:
                    found = ch
                    break
        if found is None:
            raise MonitorError(f'canal "{text}" não encontrado — use o ID ou <#id>.')
        cid = found.id
    lookup = getattr(guild, 'get_channel_or_thread', None) or getattr(guild, 'get_channel', None)
    if lookup is not None and lookup(cid) is None:
        raise MonitorError(f'canal {cid} não existe neste servidor.')
    return cid


def _resolve_user(value, guild) -> int:
    text = str(value).strip()
    m = re.fullmatch(r'<@!?(\d+)>|(\d+)', text)
    if m:
        return int(m.group(1) or m.group(2))
    member = None
    if guild is not None and hasattr(guild, 'get_member_named'):
        member = guild.get_member_named(text.lstrip('@'))
    if member is None:
        raise MonitorError(f'usuário "{text}" não encontrado — use o ID ou <@id> (veja find_user).')
    return member.id


def normalize_fields(args: dict, guild, *, partial: bool) -> dict:
    """Validate monitor fields from a tool call or slash command.

    Only keys present in *args* are returned (``partial``) — an empty list or
    string clears that filter.  Raises ``MonitorError`` on invalid input.
    """
    out: dict[str, Any] = {}
    if not partial or args.get('prompt') is not None:
        prompt = str(args.get('prompt') or '').strip()
        if not prompt:
            raise MonitorError('prompt é obrigatório.')
        if len(prompt) > MONITORS_MAX_PROMPT:
            raise MonitorError(f'prompt excede {MONITORS_MAX_PROMPT} caracteres.')
        out['prompt'] = prompt
    if args.get('name') is not None:
        out['name'] = str(args['name']).strip()[:_MAX_NAME]
    for key, resolve in (('channels', _resolve_channel), ('authors', _resolve_user), ('mentions', _resolve_user)):
        if key in args and args[key] is not None:
            values = _as_list(args[key])
            if len(values) > _MAX_IDS:
                raise MonitorError(f'{key}: no máximo {_MAX_IDS} itens.')
            ids: list[int] = []
            for v in values:
                i = resolve(v, guild)
                if i not in ids:
                    ids.append(i)
            out[key] = ids
    if 'regex' in args and args['regex'] is not None:
        pattern = str(args['regex']).strip()
        if pattern:
            compile_regex(pattern)
        out['regex'] = pattern
    if args.get('cooldown_seconds') is not None:
        try:
            cooldown = int(args['cooldown_seconds'])
        except (TypeError, ValueError):
            raise MonitorError('cooldown_seconds deve ser um número inteiro.')
        out['cooldown_seconds'] = max(MONITORS_MIN_COOLDOWN, min(_MAX_COOLDOWN, cooldown))
    return out


def has_filter(mon: dict) -> bool:
    return any(mon.get(k) for k in _FILTER_FIELDS)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def monitor_matches(mon: dict, message) -> bool:
    """Whether *message* passes every filter *mon* sets."""
    if not has_filter(mon):
        return False
    channels = mon.get('channels')
    if channels:
        channel = message.channel
        if channel.id not in channels and getattr(channel, 'parent_id', None) not in channels:
            return False
    authors = mon.get('authors')
    if authors and message.author.id not in authors:
        return False
    mentions = mon.get('mentions')
    if mentions and not any(getattr(u, 'id', None) in mentions for u in (message.mentions or [])):
        return False
    regex = mon.get('regex')
    if regex:
        compiled = mon.get('_compiled') or compile_regex(regex)
        if not compiled.search(message.content or ''):
            return False
    return True


def describe_filters(mon: dict) -> str:
    parts = []
    if mon.get('channels'):
        parts.append('canais ' + ', '.join(f'<#{c}>' for c in mon['channels']))
    if mon.get('authors'):
        parts.append('autores ' + ', '.join(f'<@{u}>' for u in mon['authors']))
    if mon.get('mentions'):
        parts.append('menções a ' + ', '.join(f'<@{u}>' for u in mon['mentions']))
    if mon.get('regex'):
        parts.append(f'regex `{mon["regex"]}`')
    return '; '.join(parts) or '(sem filtros)'


def build_trigger_header(monitors: list[dict]) -> str:
    """Instruction block prefixed to a monitored message before the chat run."""
    lines = [
        '[Monitor disparado — ninguém chamou você]',
        'Esta mensagem NÃO foi dirigida a você: ela bateu com monitor(es) configurado(s) '
        'no servidor, que chamaram você automaticamente. Siga a instrução de cada monitor:',
    ]
    for mon in monitors:
        label = f'Monitor #{mon["id"]}' + (f' "{mon["name"]}"' if mon.get('name') else '')
        lines.append(
            f'- {label} (criado por {mon.get("created_by_name") or "?"}; '
            f'filtros: {describe_filters(mon)}): {mon["prompt"]}'
        )
    lines.append(
        f'Você não é obrigado a responder. Se a instrução não pedir uma resposta para esta '
        f'mensagem, responda exatamente {NO_REPLY_TOKEN} e nada mais — nada será enviado. '
        f'Você pode usar ferramentas (ex.: add_reaction) e ainda assim responder {NO_REPLY_TOKEN}. '
        f'Se responder, a resposta vai como reply à mensagem abaixo; não diga que foi mencionado.'
    )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------

def _entry_from_row(r: sqlite3.Row) -> dict:
    def ids(v):
        try:
            return [int(x) for x in json.loads(v or '[]')]
        except (ValueError, TypeError):
            return []
    return {
        'id': r['id'],
        'guild_id': r['guild_id'],
        'name': r['name'] or '',
        'prompt': r['prompt'],
        'channels': ids(r['channel_ids']),
        'authors': ids(r['author_ids']),
        'mentions': ids(r['mention_ids']),
        'regex': r['regex'] or '',
        'cooldown_seconds': r['cooldown_seconds'],
        'status': r['status'],
        'created_by': r['created_by'],
        'created_by_name': r['created_by_name'],
        'created_at': r['created_at'],
        'updated_at': r['updated_at'],
        'last_fired_at': r['last_fired_at'],
        'fire_count': r['fire_count'],
    }


_COLUMNS = {
    'name': 'name',
    'prompt': 'prompt',
    'channels': 'channel_ids',
    'authors': 'author_ids',
    'mentions': 'mention_ids',
    'regex': 'regex',
    'cooldown_seconds': 'cooldown_seconds',
    'status': 'status',
}


def _column_value(key: str, value):
    return json.dumps(value) if key in _LIST_FIELDS else value


class MonitorStore:
    """SQLite store for monitors."""

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
                CREATE TABLE IF NOT EXISTS monitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    prompt TEXT NOT NULL,
                    channel_ids TEXT NOT NULL DEFAULT '[]',
                    author_ids TEXT NOT NULL DEFAULT '[]',
                    mention_ids TEXT NOT NULL DEFAULT '[]',
                    regex TEXT NOT NULL DEFAULT '',
                    cooldown_seconds INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_by INTEGER NOT NULL,
                    created_by_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_fired_at TEXT,
                    fire_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            con.execute('CREATE INDEX IF NOT EXISTS idx_monitors_guild ON monitors(guild_id, status)')
            con.commit()
        finally:
            con.close()

    def count(self, guild_id: int) -> int:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM monitors WHERE guild_id=? AND status!='deleted'",
                (guild_id,),
            ).fetchone()
            return int(row['n'])
        finally:
            con.close()

    def create(self, *, guild_id: int, fields: dict, created_by: int, created_by_name: str) -> dict:
        now = _now_iso()
        con = self._connect()
        try:
            cur = con.execute(
                'INSERT INTO monitors (guild_id, name, prompt, channel_ids, author_ids, mention_ids, '
                'regex, cooldown_seconds, status, created_by, created_by_name, created_at, updated_at) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (
                    guild_id, fields.get('name', ''), fields['prompt'],
                    json.dumps(fields.get('channels', [])),
                    json.dumps(fields.get('authors', [])),
                    json.dumps(fields.get('mentions', [])),
                    fields.get('regex', ''),
                    fields.get('cooldown_seconds', MONITORS_DEFAULT_COOLDOWN),
                    'active', created_by, created_by_name, now, now,
                ),
            )
            row = con.execute('SELECT * FROM monitors WHERE id=?', (cur.lastrowid,)).fetchone()
            con.commit()
            return _entry_from_row(row)
        finally:
            con.close()

    def get(self, guild_id: int, monitor_id: int) -> dict | None:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT * FROM monitors WHERE id=? AND guild_id=? AND status!='deleted'",
                (monitor_id, guild_id),
            ).fetchone()
            return _entry_from_row(row) if row else None
        finally:
            con.close()

    def list(self, guild_id: int, *, active_only: bool = False) -> list[dict]:
        status_sql = "status='active'" if active_only else "status!='deleted'"
        con = self._connect()
        try:
            rows = con.execute(
                f'SELECT * FROM monitors WHERE guild_id=? AND {status_sql} ORDER BY id',
                (guild_id,),
            ).fetchall()
        finally:
            con.close()
        return [_entry_from_row(r) for r in rows]

    def update(self, guild_id: int, monitor_id: int, fields: dict) -> dict | None:
        sets = [f'{_COLUMNS[k]}=?' for k in fields if k in _COLUMNS]
        vals: list[Any] = [_column_value(k, v) for k, v in fields.items() if k in _COLUMNS]
        con = self._connect()
        try:
            if sets:
                con.execute(
                    f"UPDATE monitors SET {', '.join(sets)}, updated_at=? WHERE id=? AND guild_id=?",
                    (*vals, _now_iso(), monitor_id, guild_id),
                )
                con.commit()
            row = con.execute(
                'SELECT * FROM monitors WHERE id=? AND guild_id=?', (monitor_id, guild_id),
            ).fetchone()
            return _entry_from_row(row) if row else None
        finally:
            con.close()

    def mark_fired(self, guild_id: int, monitor_ids: list[int]) -> None:
        now = _now_iso()
        con = self._connect()
        try:
            con.executemany(
                'UPDATE monitors SET last_fired_at=?, fire_count=fire_count+1 WHERE id=? AND guild_id=?',
                [(now, mid, guild_id) for mid in monitor_ids],
            )
            con.commit()
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Monitors(commands.Cog, name='Monitors'):
    """Standing triggers that run the bot on matching messages."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.store = MonitorStore(MONITORS_DB_PATH)
        # guild_id -> active monitors (with compiled regex), loaded lazily and
        # dropped on every write so on_message never touches SQLite per message.
        self._cache: dict[int, list[dict]] = {}
        # monitor_id -> monotonic time of the last firing (cooldowns)
        self._last_fired: dict[int, float] = {}

    async def cog_load(self):
        await asyncio.to_thread(self.store.ensure)

    # --- Matching / firing ---

    async def _active(self, guild_id: int) -> list[dict]:
        cached = self._cache.get(guild_id)
        if cached is not None:
            return cached
        monitors = await asyncio.to_thread(self.store.list, guild_id, active_only=True)
        for mon in monitors:
            if mon['regex']:
                try:
                    mon['_compiled'] = compile_regex(mon['regex'])
                except MonitorError:
                    logger.warning('[monitors] monitor %s has an invalid regex; skipping it', mon['id'])
                    mon['_skip'] = True
        monitors = [m for m in monitors if not m.get('_skip')]
        self._cache[guild_id] = monitors
        return monitors

    def _invalidate(self, guild_id: int) -> None:
        self._cache.pop(guild_id, None)

    async def _answered_elsewhere(self, message: discord.Message) -> bool:
        """Whether the mention/reply handlers already answer *message*."""
        if CHAT_MENTION_ENABLED and self.bot.user and self.bot.user.mentioned_in(message):
            return True
        ref_id = message.reference.message_id if message.reference else None
        if ref_id:
            for cog_name in ('Commands', 'DocsRAG'):
                cog = self.bot.get_cog(cog_name)
                store = getattr(cog, 'store', None)
                if store is None:
                    continue
                try:
                    if await store.get_by_handle(ref_id):
                        return True
                except Exception:
                    logger.exception('[monitors] conversation lookup failed in %s', cog_name)
        return False

    async def matching(self, message: discord.Message) -> list[dict]:
        """Active monitors that fire on *message*, cooldowns applied and started."""
        if message.author.bot or message.guild is None:
            return []
        monitors = await self._active(message.guild.id)
        if not monitors:
            return []
        matched = [m for m in monitors if monitor_matches(m, message)]
        if not matched or await self._answered_elsewhere(message):
            return []
        now = time.monotonic()
        due = [
            m for m in matched
            if now - self._last_fired.get(m['id'], float('-inf')) >= m['cooldown_seconds']
        ]
        for m in due:
            self._last_fired[m['id']] = now
        return due

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        try:
            due = await self.matching(message)
        except Exception:
            logger.exception('[monitors] matching failed for message %s', message.id)
            return
        if not due:
            return
        await asyncio.to_thread(self.store.mark_fired, message.guild.id, [m['id'] for m in due])
        commands_cog = self.bot.get_cog('Commands')
        if commands_cog is None or not commands_cog.client:
            logger.warning('[monitors] chat unavailable; monitors %s not run', [m['id'] for m in due])
            return
        logger.info(
            '[monitors] firing %s on message=%s channel=%s author=%s',
            [m['id'] for m in due], message.id, message.channel.id, message.author.id,
        )
        try:
            replied = await commands_cog.answer_monitor(message, build_trigger_header(due))
            logger.info('[monitors] monitors %s %s', [m['id'] for m in due], 'replied' if replied else 'stayed silent')
        except Exception:
            # Nobody called the bot, so a failure stays in the logs.
            logger.exception('[monitors] run failed for monitors %s', [m['id'] for m in due])

    # --- Core operations (shared by slash commands and tools) ---

    async def create(self, guild: discord.Guild, requester, args: dict) -> dict:
        if not can_manage(requester):
            raise MonitorError('criar monitores exige a permissão Gerenciar Servidor.')
        fields = normalize_fields(args, guild, partial=False)
        if not has_filter(fields):
            raise MonitorError(
                'defina ao menos um filtro: channels, authors, mentions ou regex.'
            )
        count = await asyncio.to_thread(self.store.count, guild.id)
        if count >= MONITORS_MAX_PER_GUILD:
            raise MonitorError(f'limite de {MONITORS_MAX_PER_GUILD} monitores por servidor.')
        mon = await asyncio.to_thread(
            self.store.create,
            guild_id=guild.id, fields=fields,
            created_by=getattr(requester, 'id', 0) or 0,
            created_by_name=getattr(requester, 'display_name', None) or 'bot',
        )
        self._invalidate(guild.id)
        return mon

    async def update(self, guild: discord.Guild, requester, monitor_id, args: dict) -> dict:
        if not can_manage(requester):
            raise MonitorError('editar monitores exige a permissão Gerenciar Servidor.')
        mon = await self._get(guild, monitor_id)
        fields = normalize_fields(args, guild, partial=True)
        status = args.get('status')
        if status is not None:
            if status not in ('active', 'paused'):
                raise MonitorError('status deve ser "active" ou "paused".')
            fields['status'] = status
        if not fields:
            raise MonitorError('nada para alterar.')
        if not has_filter({**mon, **fields}):
            raise MonitorError('o monitor precisa manter ao menos um filtro.')
        updated = await asyncio.to_thread(self.store.update, guild.id, mon['id'], fields)
        self._invalidate(guild.id)
        return updated or mon

    async def delete(self, guild: discord.Guild, requester, monitor_id) -> dict:
        if not can_manage(requester):
            raise MonitorError('remover monitores exige a permissão Gerenciar Servidor.')
        mon = await self._get(guild, monitor_id)
        await asyncio.to_thread(self.store.update, guild.id, mon['id'], {'status': 'deleted'})
        self._invalidate(guild.id)
        self._last_fired.pop(mon['id'], None)
        return mon

    async def _get(self, guild: discord.Guild, monitor_id) -> dict:
        try:
            mid = int(monitor_id)
        except (TypeError, ValueError):
            raise MonitorError('id deve ser um número.')
        mon = await asyncio.to_thread(self.store.get, guild.id, mid)
        if mon is None:
            raise MonitorError(f'monitor #{mid} não encontrado.')
        return mon

    # --- Slash commands ---

    monitor = app_commands.Group(name='monitor', description='Monitores: gatilhos que chamam o bot em mensagens')

    async def _respond(self, interaction: discord.Interaction, op, ok_text) -> None:
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        try:
            mon = await op()
        except MonitorError as e:
            return await interaction.response.send_message(f'❌ {e}', ephemeral=True)
        await interaction.response.send_message(
            ok_text(mon) or None, embed=build_monitor_embed(mon), ephemeral=True,
        )

    @monitor.command(name='create', description='Criar um monitor (Gerenciar Servidor)')
    @app_commands.describe(
        prompt='O que o bot deve fazer quando uma mensagem bater (ele pode não responder)',
        name='Nome curto do monitor',
        channel='Só mensagens neste canal (e threads dele)',
        author='Só mensagens enviadas por este usuário',
        mention='Só mensagens que mencionam este usuário',
        regex='Só mensagens cujo texto bate com esta regex (sem diferenciar maiúsculas)',
        cooldown='Segundos mínimos entre disparos (padrão: %d)' % MONITORS_DEFAULT_COOLDOWN,
    )
    async def monitor_create(
        self,
        interaction: discord.Interaction,
        prompt: str,
        name: str | None = None,
        channel: discord.TextChannel | None = None,
        author: discord.Member | None = None,
        mention: discord.Member | None = None,
        regex: str | None = None,
        cooldown: int | None = None,
    ):
        args = {'prompt': prompt, 'name': name, 'regex': regex, 'cooldown_seconds': cooldown}
        if channel:
            args['channels'] = [channel.id]
        if author:
            args['authors'] = [author.id]
        if mention:
            args['mentions'] = [mention.id]
        await self._respond(
            interaction,
            lambda: self.create(interaction.guild, interaction.user, args),
            lambda m: f'✅ Monitor **#{m["id"]}** criado.',
        )

    @monitor.command(name='list', description='Listar monitores')
    async def monitor_list(self, interaction: discord.Interaction):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        entries = await asyncio.to_thread(self.store.list, interaction.guild.id)
        if not entries:
            return await interaction.response.send_message(
                'Nenhum monitor. Use `/monitor create` ou peça ao bot.', ephemeral=True,
            )
        lines = [_list_line(e) for e in entries]
        pages = [
            discord.Embed(
                title='👁️ Monitores',
                description='\n'.join(lines[i:i + 10]),
                color=discord.Color.dark_teal(),
            )
            for i in range(0, len(lines), 10)
        ]
        for i, page in enumerate(pages):
            page.set_footer(text=f'{len(entries)} monitores • Página {i + 1}/{len(pages)}')
        kwargs: dict = {'embed': pages[0], 'ephemeral': True}
        if len(pages) > 1:
            kwargs['view'] = PaginatedEmbedView(pages)
        await interaction.response.send_message(**kwargs)

    @monitor.command(name='show', description='Ver detalhes de um monitor')
    @app_commands.describe(id='ID do monitor')
    async def monitor_show(self, interaction: discord.Interaction, id: int):
        await self._respond(interaction, lambda: self._get(interaction.guild, id), lambda m: '')

    @monitor.command(name='edit', description='Editar um monitor (Gerenciar Servidor)')
    @app_commands.describe(
        id='ID do monitor',
        prompt='Nova instrução',
        name='Novo nome',
        channel='Trocar o filtro de canal por este canal',
        author='Trocar o filtro de autor por este usuário',
        mention='Trocar o filtro de menção por este usuário',
        regex='Nova regex',
        cooldown='Novo cooldown em segundos',
        clear='Remover um filtro',
    )
    @app_commands.choices(clear=[
        app_commands.Choice(name='canais', value='channels'),
        app_commands.Choice(name='autores', value='authors'),
        app_commands.Choice(name='menções', value='mentions'),
        app_commands.Choice(name='regex', value='regex'),
    ])
    async def monitor_edit(
        self,
        interaction: discord.Interaction,
        id: int,
        prompt: str | None = None,
        name: str | None = None,
        channel: discord.TextChannel | None = None,
        author: discord.Member | None = None,
        mention: discord.Member | None = None,
        regex: str | None = None,
        cooldown: int | None = None,
        clear: app_commands.Choice[str] | None = None,
    ):
        args: dict[str, Any] = {'prompt': prompt, 'name': name, 'regex': regex, 'cooldown_seconds': cooldown}
        if clear is not None:
            args[clear.value] = '' if clear.value == 'regex' else []
        if channel:
            args['channels'] = [channel.id]
        if author:
            args['authors'] = [author.id]
        if mention:
            args['mentions'] = [mention.id]
        await self._respond(
            interaction,
            lambda: self.update(interaction.guild, interaction.user, id, args),
            lambda m: f'✅ Monitor **#{m["id"]}** atualizado.',
        )

    @monitor.command(name='pause', description='Pausar um monitor (Gerenciar Servidor)')
    @app_commands.describe(id='ID do monitor')
    async def monitor_pause(self, interaction: discord.Interaction, id: int):
        await self._respond(
            interaction,
            lambda: self.update(interaction.guild, interaction.user, id, {'status': 'paused'}),
            lambda m: f'⏸️ Monitor **#{m["id"]}** pausado.',
        )

    @monitor.command(name='resume', description='Retomar um monitor pausado (Gerenciar Servidor)')
    @app_commands.describe(id='ID do monitor')
    async def monitor_resume(self, interaction: discord.Interaction, id: int):
        await self._respond(
            interaction,
            lambda: self.update(interaction.guild, interaction.user, id, {'status': 'active'}),
            lambda m: f'▶️ Monitor **#{m["id"]}** retomado.',
        )

    @monitor.command(name='delete', description='Remover um monitor (Gerenciar Servidor)')
    @app_commands.describe(id='ID do monitor')
    async def monitor_delete(self, interaction: discord.Interaction, id: int):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)
        try:
            mon = await self.delete(interaction.guild, interaction.user, id)
        except MonitorError as e:
            return await interaction.response.send_message(f'❌ {e}', ephemeral=True)
        await interaction.response.send_message(f'🗑️ Monitor **#{mon["id"]}** removido.', ephemeral=True)

    # --- Agent tool execution (LLM tool-calling) ---

    async def exec_tool(
        self, name: str, args: dict, *, guild: discord.Guild, requester=None,
    ) -> tuple[str, list[dict]]:
        """Execute a ``monitor_*`` tool call; mirrors ``Scheduler.exec_tool``."""
        try:
            if name == 'monitor_create':
                mon = await self.create(guild, requester, args)
                return f'✅ Monitor #{mon["id"]} criado. {_list_line(mon)}', []
            if name == 'monitor_list':
                entries = await asyncio.to_thread(self.store.list, guild.id)
                if not entries:
                    return 'Nenhum monitor configurado.', []
                return (
                    f'**Monitores ({len(entries)}):**\n'
                    + '\n'.join(_list_line(e, prompt_len=200) for e in entries),
                    [],
                )
            if name == 'monitor_update':
                fields = {k: v for k, v in args.items() if k != 'id'}
                mon = await self.update(guild, requester, args.get('id'), fields)
                return f'✅ Monitor #{mon["id"]} atualizado. {_list_line(mon)}', []
            if name == 'monitor_delete':
                mon = await self.delete(guild, requester, args.get('id'))
                return f'✅ Monitor #{mon["id"]} removido.', []
        except MonitorError as e:
            return f'⚠️ Monitores: {e}', []
        except Exception:
            logger.exception('[monitors] unexpected error in %s', name)
            return '⚠️ Erro interno nos monitores.', []
        return f'Ferramenta desconhecida: {name}', []


def _list_line(mon: dict, prompt_len: int = 80) -> str:
    icon = '▶️' if mon['status'] == 'active' else '⏸️'
    name = f' "{mon["name"]}"' if mon.get('name') else ''
    return (
        f'{icon} **#{mon["id"]}**{name} — {describe_filters(mon)} — '
        f'cooldown {mon["cooldown_seconds"]}s — {mon["fire_count"]} disparos\n'
        f'   ↳ {_snippet(mon["prompt"], prompt_len)}'
    )


def build_monitor_embed(mon: dict) -> discord.Embed:
    status_map = {'active': '▶️ Ativo', 'paused': '⏸️ Pausado', 'deleted': '🗑️ Removido'}
    embed = discord.Embed(
        title=f'👁️ Monitor #{mon["id"]}' + (f' — {mon["name"]}' if mon.get('name') else ''),
        description=_snippet(mon['prompt'], 3800),
        color=discord.Color.dark_teal(),
    )
    embed.add_field(name='Filtros', value=_snippet(describe_filters(mon), 1000), inline=False)
    embed.add_field(name='Status', value=status_map.get(mon['status'], mon['status']), inline=True)
    embed.add_field(name='Cooldown', value=f'{mon["cooldown_seconds"]}s', inline=True)
    embed.add_field(name='Disparos', value=str(mon['fire_count']), inline=True)
    embed.add_field(name='Último disparo', value=_fmt_brt(mon['last_fired_at']), inline=True)
    embed.add_field(name='Criado por', value=mon['created_by_name'] or '?', inline=True)
    embed.add_field(name='Criado em', value=_fmt_brt(mon['created_at']), inline=True)
    return embed


# ---------------------------------------------------------------------------
# Agent tool schemas (for LLM tool-calling)
# ---------------------------------------------------------------------------

_ID_LIST = {'type': 'array', 'items': {'type': 'string'}}

_FILTER_PROPERTIES = {
    'channels': {
        **_ID_LIST,
        'description': 'Canais vigiados (IDs ou <#id>); threads desses canais também contam. Vazio = todos os canais.',
    },
    'authors': {
        **_ID_LIST,
        'description': 'Só mensagens enviadas por estes usuários (author_id ou <@id>).',
    },
    'mentions': {
        **_ID_LIST,
        'description': 'Só mensagens que mencionam algum destes usuários (author_id ou <@id>).',
    },
    'regex': {
        'type': 'string',
        'description': 'Regex (Python, sem diferenciar maiúsculas) que o texto da mensagem deve conter. Ex: "\\b(lag|travando)\\b".',
    },
    'cooldown_seconds': {
        'type': 'integer',
        'description': f'Segundos mínimos entre disparos deste monitor (padrão {MONITORS_DEFAULT_COOLDOWN}, mínimo {MONITORS_MIN_COOLDOWN}).',
    },
    'name': {'type': 'string', 'description': 'Nome curto para identificar o monitor.'},
    'prompt': {
        'type': 'string',
        'description': (
            'Instrução do que você deve fazer quando uma mensagem bater, incluindo quando '
            'NÃO responder. Ex: "Se a pessoa estiver com dúvida sobre lag, ajude; senão fique em silêncio".'
        ),
    },
}

MONITOR_CREATE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'monitor_create',
        'description': (
            'Cria um monitor: um gatilho permanente que chama você automaticamente quando uma '
            'mensagem do servidor bate com TODOS os filtros definidos (canais, autores, menções a '
            'usuários, regex). Ao disparar, você recebe a mensagem como se tivesse sido mencionado, '
            'mas sabendo que veio do monitor, e pode escolher não responder. Exige ao menos um filtro '
            'e a permissão Gerenciar Servidor de quem pede. Use para pedidos "sempre que…", '
            '"quando alguém…", "fique de olho em…".'
        ),
        'parameters': {
            'type': 'object',
            'properties': _FILTER_PROPERTIES,
            'required': ['prompt'],
        },
    },
}

MONITOR_LIST_TOOL = {
    'type': 'function',
    'function': {
        'name': 'monitor_list',
        'description': 'Lista os monitores do servidor com ID, filtros, cooldown, status e instrução.',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    },
}

MONITOR_UPDATE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'monitor_update',
        'description': (
            'Edita um monitor pelo ID. Envie só os campos a mudar: listas substituem as atuais, '
            'lista vazia ou regex "" remove aquele filtro. status="paused" pausa e "active" retoma. '
            'Exige a permissão Gerenciar Servidor de quem pede.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'id': {'type': 'integer', 'description': 'ID do monitor.'},
                **_FILTER_PROPERTIES,
                'status': {'type': 'string', 'enum': ['active', 'paused']},
            },
            'required': ['id'],
        },
    },
}

MONITOR_DELETE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'monitor_delete',
        'description': 'Remove um monitor pelo ID. Exige a permissão Gerenciar Servidor de quem pede.',
        'parameters': {
            'type': 'object',
            'properties': {'id': {'type': 'integer', 'description': 'ID do monitor.'}},
            'required': ['id'],
        },
    },
}

MONITOR_TOOLS = [MONITOR_CREATE_TOOL, MONITOR_LIST_TOOL, MONITOR_UPDATE_TOOL, MONITOR_DELETE_TOOL]
MONITOR_TOOL_NAMES = {t['function']['name'] for t in MONITOR_TOOLS}


def monitor_tool_status(name: str, args: dict) -> str:
    if name == 'monitor_create':
        return f"👁️ Criando monitor: *{str(args.get('name') or args.get('prompt') or '')[:40]}*"
    if name == 'monitor_list':
        return '👁️ Listando monitores…'
    if name == 'monitor_update':
        return f'👁️ Editando monitor #{args.get("id", "?")}'
    return f'👁️ Removendo monitor #{args.get("id", "?")}'


async def setup(bot: commands.Bot):
    if not MONITORS_ENABLED:
        logger.info('Monitors disabled via MONITORS_ENABLED=false')
        return
    await bot.add_cog(Monitors(bot))
