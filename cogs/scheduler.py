"""Persistent scheduler — one-shot and recurring (cron) tasks for the bot.

Each scheduled job stores a free-text prompt that the LLM acts on when the job
fires, posting its response to a target channel.  This reuses the same
``_run_chat`` pipeline as ``/chat``, so scheduled tasks have the full toolset
(web search, memory, channel history, etc.).

Schedule types
--------------
- **once**: fires at a specific datetime, then auto-deleted.
- **cron**: 5-field cron expression (min hour day month weekday), evaluated
  against ``BR_TZ`` (America/Sao_Paulo).  Recurring.

Both types survive restarts — on boot the loop picks up overdue ``once`` jobs
immediately and recomputes ``next_fire`` for ``cron`` jobs.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import re
import sqlite3
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from cogs.utils import BR_TZ, PaginatedEmbedView, extract_pingable_mentions
from config import (
    SCHEDULER_DB_PATH,
    SCHEDULER_ENABLED,
    SCHEDULER_LOOP_INTERVAL,
    SCHEDULER_MAX_JOBS_PER_GUILD,
    SCHEDULER_MAX_PROMPT,
)

logger = logging.getLogger(__name__)

_CRON_FIELDS = ('minute', 'hour', 'day', 'month', 'weekday')
_CRON_RANGES = {
    'minute': (0, 59),
    'hour': (0, 23),
    'day': (1, 31),
    'month': (1, 12),
    'weekday': (0, 6),
}
_WEEKDAY_NAMES = {
    'sun': 0, 'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6,
}
_MAX_DELAY_HOURS = 24 * 365  # cap one-shot delays at ~1 year
_CRON_SEARCH_CAP_MIN = 24 * 60 * 366  # max minutes to search forward for next fire


def _now() -> datetime.datetime:
    return datetime.datetime.now(BR_TZ)


def _now_iso() -> str:
    return _now().isoformat()


def _fmt_brt(dt: datetime.datetime | None) -> str:
    if dt is None:
        return '—'
    try:
        return dt.astimezone(BR_TZ).strftime('%d/%m/%Y %H:%M')
    except Exception:
        return str(dt)


def _parse_delay(delay: str) -> datetime.datetime:
    """Parse a delay string like '5m', '2h', '30s', or an absolute datetime.

    Returns the fire datetime in BR_TZ.  Raises ValueError on invalid input.
    """
    delay = delay.strip().lower()
    now = _now()

    rel = re.fullmatch(r'(\d+)\s*([smhd])', delay)
    if rel:
        n = int(rel.group(1))
        unit = rel.group(2)
        if unit == 's':
            delta = datetime.timedelta(seconds=n)
        elif unit == 'm':
            delta = datetime.timedelta(minutes=n)
        elif unit == 'h':
            delta = datetime.timedelta(hours=n)
        else:
            delta = datetime.timedelta(days=n)
        if delta.total_seconds() > _MAX_DELAY_HOURS * 3600:
            raise ValueError(f'Delay máximo é {_MAX_DELAY_HOURS} horas.')
        return now + delta

    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M'):
        try:
            dt_naive = datetime.datetime.strptime(delay, fmt)
            return dt_naive.replace(tzinfo=BR_TZ)
        except ValueError:
            continue

    raise ValueError(
        'Formato inválido. Use "5m", "2h", "2026-09-04 14:30" ou "04/09/2026 14:30".'
    )


# ---------------------------------------------------------------------------
# Cron expression parsing
# ---------------------------------------------------------------------------

def _parse_cron_field(expr: str, field: str) -> set[int]:
    """Parse a single cron field into a set of valid values."""
    lo, hi = _CRON_RANGES[field]
    expr = expr.strip().lower()

    for name, val in _WEEKDAY_NAMES.items():
        expr = expr.replace(name, str(val))

    values: set[int] = set()
    for part in expr.split(','):
        part = part.strip()
        if not part:
            continue
        if part == '*':
            values.update(range(lo, hi + 1))
            continue
        step_match = re.fullmatch(r'(.+)/(\d+)', part)
        if step_match:
            base = step_match.group(1)
            step = int(step_match.group(2))
            if step <= 0:
                raise ValueError(f'Passo inválido em "{field}": {part}')
            if base == '*':
                values.update(range(lo, hi + 1, step))
                continue
            range_match = re.fullmatch(r'(\d+)-(\d+)', base)
            if range_match:
                a, b = int(range_match.group(1)), int(range_match.group(2))
                a, b = max(lo, a), min(hi, b)
                if a > b:
                    a, b = b, a
                values.update(range(a, b + 1, step))
                continue
            single = int(base)
            if not (lo <= single <= hi):
                raise ValueError(f'Valor {single} fora do range [{lo}-{hi}] em "{field}"')
            values.update(range(single, hi + 1, step))
            continue
        range_match = re.fullmatch(r'(\d+)-(\d+)', part)
        if range_match:
            a, b = int(range_match.group(1)), int(range_match.group(2))
            a, b = max(lo, a), min(hi, b)
            if a > b:
                a, b = b, a
            values.update(range(a, b + 1))
            continue
        single = int(part)
        if not (lo <= single <= hi):
            raise ValueError(f'Valor {single} fora do range [{lo}-{hi}] em "{field}"')
        values.add(single)
    if not values:
        raise ValueError(f'Campo "{field}" não contém valores válidos')
    return values


def _parse_cron(expr: str) -> dict[str, set[int]]:
    """Parse a 5-field cron expression into a dict of sets."""
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(
            'Expressão cron deve ter 5 campos: minuto hora dia mês dia-da-semana.\n'
            'Ex: "0 2 * * *" = todo dia às 2 AM.\n'
            'Ex: "*/30 * * * *" = a cada 30 minutos.'
        )
    return {
        name: _parse_cron_field(part, name)
        for name, part in zip(_CRON_FIELDS, parts)
    }


def _next_fire(cron: dict[str, set[int]], after: datetime.datetime) -> datetime.datetime:
    """Compute the next fire time for a parsed cron expression after *after*."""
    base = after.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
    checked = 0
    while checked < _CRON_SEARCH_CAP_MIN:
        if (base.minute in cron['minute']
                and base.hour in cron['hour']
                and base.day in cron['day']
                and base.month in cron['month']
                and (base.weekday() + 1) % 7 in cron['weekday']):
            return base
        base += datetime.timedelta(minutes=1)
        checked += 1
    raise ValueError('Não foi possível encontrar o próximo disparo em um ano.')


def _cron_human(cron: dict[str, set[int]]) -> str:
    """Human-readable description of a parsed cron expression."""
    def _field(name: str) -> str:
        vals = sorted(cron[name])
        lo, hi = _CRON_RANGES[name]
        if vals == list(range(lo, hi + 1)):
            return '*'
        return ','.join(str(v) for v in vals)
    return ' '.join(_field(n) for n in _CRON_FIELDS)


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------

def _entry_from_row(r: sqlite3.Row) -> dict:
    return {
        'id': r['id'],
        'guild_id': r['guild_id'],
        'channel_id': r['channel_id'],
        'type': r['type'],
        'cron_expr': r['cron_expr'] or '',
        'next_fire': r['next_fire'],
        'prompt': r['prompt'],
        'created_by': r['created_by'],
        'created_by_name': r['created_by_name'],
        'status': r['status'],
        'created_at': r['created_at'],
        'updated_at': r['updated_at'],
        'last_fired_at': r['last_fired_at'],
        'fire_count': r['fire_count'],
    }


def _parse_iso(s: str | None) -> datetime.datetime | None:
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s)
    except ValueError:
        return None


class SchedulerStore:
    """SQLite store for scheduled jobs."""

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
                CREATE TABLE IF NOT EXISTS schedules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    cron_expr TEXT,
                    next_fire TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    created_by INTEGER NOT NULL,
                    created_by_name TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_fired_at TEXT,
                    fire_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            con.execute('CREATE INDEX IF NOT EXISTS idx_sched_guild ON schedules(guild_id, status)')
            con.execute('CREATE INDEX IF NOT EXISTS idx_sched_fire ON schedules(status, next_fire)')
            con.execute('CREATE INDEX IF NOT EXISTS idx_sched_channel ON schedules(channel_id)')
            con.commit()
        finally:
            con.close()

    def count_active(self, guild_id: int) -> int:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM schedules WHERE guild_id=? AND status='active'",
                (guild_id,),
            ).fetchone()
            return int(row['n'])
        finally:
            con.close()

    def create(
        self, *, guild_id: int, channel_id: int, job_type: str,
        cron_expr: str | None, next_fire: str, prompt: str,
        created_by: int, created_by_name: str,
    ) -> dict:
        now = _now_iso()
        con = self._connect()
        try:
            cur = con.execute(
                'INSERT INTO schedules '
                '(guild_id, channel_id, type, cron_expr, next_fire, prompt, '
                'created_by, created_by_name, status, created_at, updated_at, fire_count) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,0)',
                (
                    guild_id, channel_id, job_type, cron_expr, next_fire, prompt,
                    created_by, created_by_name, 'active', now, now,
                ),
            )
            row = con.execute(
                'SELECT * FROM schedules WHERE id=?', (cur.lastrowid,)
            ).fetchone()
            con.commit()
            return _entry_from_row(row)
        finally:
            con.close()

    def get(self, guild_id: int, job_id: int) -> dict | None:
        con = self._connect()
        try:
            row = con.execute(
                'SELECT * FROM schedules WHERE id=? AND guild_id=?',
                (job_id, guild_id),
            ).fetchone()
            return _entry_from_row(row) if row else None
        finally:
            con.close()

    def list_active(self, guild_id: int) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT * FROM schedules WHERE guild_id=? AND status!='deleted' ORDER BY next_fire",
                (guild_id,),
            ).fetchall()
        finally:
            con.close()
        return [_entry_from_row(r) for r in rows]

    def list_due(self, now_iso: str) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT * FROM schedules WHERE status='active' AND next_fire <= ?",
                (now_iso,),
            ).fetchall()
        finally:
            con.close()
        return [_entry_from_row(r) for r in rows]

    def update(
        self, guild_id: int, job_id: int, *,
        prompt: str | None = None,
        next_fire: str | None = None,
        cron_expr: str | None = None,
        channel_id: int | None = None,
        status: str | None = None,
    ) -> dict | None:
        con = self._connect()
        try:
            sets: list[str] = []
            vals: list[Any] = []
            if prompt is not None:
                sets.append('prompt=?')
                vals.append(prompt)
            if next_fire is not None:
                sets.append('next_fire=?')
                vals.append(next_fire)
            if cron_expr is not None:
                sets.append('cron_expr=?')
                vals.append(cron_expr)
            if channel_id is not None:
                sets.append('channel_id=?')
                vals.append(channel_id)
            if status is not None:
                sets.append('status=?')
                vals.append(status)
            if not sets:
                return self.get(guild_id, job_id)
            sets.append('updated_at=?')
            vals.append(_now_iso())
            vals.extend((job_id, guild_id))
            con.execute(
                f"UPDATE schedules SET {', '.join(sets)} WHERE id=? AND guild_id=?",
                vals,
            )
            con.commit()
            row = con.execute(
                'SELECT * FROM schedules WHERE id=? AND guild_id=?',
                (job_id, guild_id),
            ).fetchone()
            return _entry_from_row(row) if row else None
        finally:
            con.close()

    def mark_fired(self, guild_id: int, job_id: int, *, next_fire: str | None, delete: bool) -> None:
        now = _now_iso()
        con = self._connect()
        try:
            if delete:
                con.execute(
                    "UPDATE schedules SET status='deleted', last_fired_at=?, "
                    "fire_count=fire_count+1, updated_at=? WHERE id=? AND guild_id=?",
                    (now, now, job_id, guild_id),
                )
            else:
                con.execute(
                    "UPDATE schedules SET last_fired_at=?, fire_count=fire_count+1, "
                    "next_fire=?, updated_at=? WHERE id=? AND guild_id=?",
                    (now, next_fire, now, job_id, guild_id),
                )
            con.commit()
        finally:
            con.close()

    def recompute_cron_next_fires(self) -> int:
        """Recompute next_fire for all active cron jobs (used on restart)."""
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT * FROM schedules WHERE status='active' AND type='cron'"
            ).fetchall()
            now = _now()
            updated = 0
            for row in rows:
                try:
                    cron = _parse_cron(row['cron_expr'])
                    nxt = _next_fire(cron, now)
                    con.execute(
                        "UPDATE schedules SET next_fire=? WHERE id=?",
                        (nxt.isoformat(), row['id']),
                    )
                    updated += 1
                except Exception:
                    logger.exception("[scheduler] failed to recompute cron for job %s", row['id'])
            con.commit()
            return updated
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Confirm view (matches Memory cog's ConfirmView pattern)
# ---------------------------------------------------------------------------

class _ConfirmView(discord.ui.View):
    def __init__(self, owner_id: int):
        super().__init__(timeout=60)
        self.owner_id = owner_id
        self.confirmed = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.owner_id

    @discord.ui.button(label='Confirmar', style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = True
        self.stop()
        await interaction.response.edit_message(view=None)

    @discord.ui.button(label='Cancelar', style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.stop()
        await interaction.response.edit_message(view=None)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class SchedulerError(Exception):
    """User-facing scheduler error."""


def _is_admin(interaction: discord.Interaction) -> bool:
    return interaction.user.guild_permissions.administrator if interaction.guild else False


class Scheduler(commands.Cog, name='Scheduler'):
    """Persistent scheduler: one-shot and cron-recurring LLM tasks."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.store = SchedulerStore(SCHEDULER_DB_PATH)
        self._loop_task: asyncio.Task | None = None
        self._fire_tasks: set[asyncio.Task] = set()

    async def cog_load(self):
        await asyncio.to_thread(self.store.ensure)
        n = await asyncio.to_thread(self.store.recompute_cron_next_fires)
        if n:
            logger.info("[scheduler] recomputed next_fire for %d cron job(s)", n)
        self._loop_task = asyncio.get_running_loop().create_task(self._run_loop())

    async def cog_unload(self):
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        for task in list(self._fire_tasks):
            task.cancel()
        self._fire_tasks.clear()

    # --- Background loop ---

    async def _run_loop(self):
        while True:
            try:
                await asyncio.sleep(SCHEDULER_LOOP_INTERVAL)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[scheduler] loop error")

    async def _tick(self):
        now = _now_iso()
        due = await asyncio.to_thread(self.store.list_due, now)
        for job in due:
            task = asyncio.get_running_loop().create_task(self._fire(job))
            self._fire_tasks.add(task)
            task.add_done_callback(self._fire_tasks.discard)

    async def _fire(self, job: dict):
        """Fire a scheduled job: mark fired, run LLM, post response to channel."""
        guild = self.bot.get_guild(job['guild_id'])
        if guild is None:
            logger.warning("[scheduler] guild %s not found for job %s", job['guild_id'], job['id'])
            return

        channel = guild.get_channel(job['channel_id'])
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(job['channel_id'])
            except Exception:
                logger.exception("[scheduler] channel %s not found for job %s", job['channel_id'], job['id'])
                return

        is_once = job['type'] == 'once'
        await asyncio.to_thread(
            self.store.mark_fired, job['guild_id'], job['id'],
            next_fire=None, delete=is_once,
        )

        try:
            await self._run_scheduled_prompt(job, guild, channel)
        except Exception:
            logger.exception("[scheduler] failed to fire job %s", job['id'])

        if not is_once:
            try:
                cron = _parse_cron(job['cron_expr'])
                nxt = _next_fire(cron, _now())
                await asyncio.to_thread(
                    self.store.update, job['guild_id'], job['id'],
                    next_fire=nxt.isoformat(),
                )
            except Exception:
                logger.exception("[scheduler] failed to recompute next_fire for job %s", job['id'])

    async def _run_scheduled_prompt(self, job: dict, guild: discord.Guild, channel):
        """Run the LLM on the job's prompt and post the result to the channel."""
        commands_cog = self.bot.get_cog('Commands')
        if commands_cog is None or not commands_cog.client:
            try:
                await channel.send('⚠️ Tarefa agendada não pode executar: cliente de IA indisponível.')
            except discord.HTTPException:
                pass
            return

        owner = guild.get_member(job['created_by'])
        prompt = (
            f"[Tarefa agendada por {job['created_by_name']}]\n"
            f"Esta é uma tarefa automática disparada pelo agendador. "
            f"Aja naturalmente conforme a instrução abaixo:\n\n{job['prompt']}"
        )

        try:
            async with channel.typing():
                answer, embeds, sources, _capture = await commands_cog._run_chat(
                    prompt,
                    user=owner,
                    guild=guild,
                    channel=channel,
                    created_at=datetime.datetime.now(BR_TZ),
                )
            if embeds:
                # Mentions in embeds never notify — extracted user pings ride
                # as message content.  everyone/roles stay unparsable so a
                # cron job can never ping @everyone.
                mentions = extract_pingable_mentions(answer, guild)
                send_kwargs: dict = {'embed': embeds[0]}
                if len(embeds) > 1:
                    send_kwargs['view'] = PaginatedEmbedView(embeds)
                if mentions:
                    send_kwargs['content'] = mentions
                send_kwargs['allowed_mentions'] = discord.AllowedMentions(
                    users=True, roles=False, everyone=False,
                )
                await channel.send(**send_kwargs)
            elif answer:
                await channel.send(
                    answer[:2000],
                    allowed_mentions=discord.AllowedMentions(
                        users=True, roles=False, everyone=False,
                    ),
                )
        except Exception:
            logger.exception("[scheduler] LLM execution failed for job %s", job['id'])
            try:
                await channel.send('⚠️ Tarefa agendada falhou ao executar. Verifique os logs.')
            except discord.HTTPException:
                pass

    # --- Slash commands ---

    schedule = app_commands.Group(name='schedule', description='Agendamento de tarefas')

    @schedule.command(name='create', description='Criar uma tarefa agendada')
    @app_commands.describe(
        type='Tipo: once (uma vez) ou cron (recorrente)',
        prompt='O que o bot deve fazer quando a tarefa disparar (ex: avisar usuários)',
        delay='Para once: "5m", "2h", "2026-09-04 14:30" ou "04/09/2026 14:30"',
        cron='Para cron: 5 campos "min hora dia mes diasemana" — ex: "0 2 * * *" = 2 AM todo dia',
        channel='Canal onde o bot deve responder (padrão: canal atual)',
    )
    @app_commands.choices(type=[
        app_commands.Choice(name='Uma vez (once)', value='once'),
        app_commands.Choice(name='Recorrente (cron)', value='cron'),
    ])
    async def schedule_create(
        self,
        interaction: discord.Interaction,
        type: app_commands.Choice[str],
        prompt: str,
        delay: str | None = None,
        cron: str | None = None,
        channel: discord.TextChannel | None = None,
    ):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)

        job_type = type.value
        if len(prompt) > SCHEDULER_MAX_PROMPT:
            return await interaction.response.send_message(
                f'Prompt muito longo. Máximo {SCHEDULER_MAX_PROMPT} caracteres.',
                ephemeral=True,
            )
        if channel is None:
            channel = interaction.channel

        count = await asyncio.to_thread(self.store.count_active, interaction.guild.id)
        if count >= SCHEDULER_MAX_JOBS_PER_GUILD:
            return await interaction.response.send_message(
                f'Limite de {SCHEDULER_MAX_JOBS_PER_GUILD} tarefas ativas por servidor.',
                ephemeral=True,
            )

        try:
            if job_type == 'once':
                if not delay:
                    return await interaction.response.send_message(
                        'Tarefas "once" precisam do parâmetro **delay**. '
                        'Ex: "5m", "2h", "2026-09-04 14:30".',
                        ephemeral=True,
                    )
                fire_dt = _parse_delay(delay)
                if fire_dt <= _now():
                    return await interaction.response.send_message(
                        'A data/hora do disparo deve estar no futuro.',
                        ephemeral=True,
                    )
                cron_expr = None
                next_fire = fire_dt.isoformat()
            else:
                if not cron:
                    return await interaction.response.send_message(
                        'Tarefas "cron" precisam do parâmetro **cron**. '
                        'Ex: "0 2 * * *" = todo dia às 2 AM.\n'
                        'Campos: minuto hora dia mês dia-da-semana.',
                        ephemeral=True,
                    )
                parsed = _parse_cron(cron)
                fire_dt = _next_fire(parsed, _now())
                cron_expr = cron
                next_fire = fire_dt.isoformat()
        except ValueError as exc:
            return await interaction.response.send_message(f'❌ {exc}', ephemeral=True)

        await interaction.response.defer(ephemeral=True)

        job = await asyncio.to_thread(
            self.store.create,
            guild_id=interaction.guild.id,
            channel_id=channel.id,
            job_type=job_type,
            cron_expr=cron_expr,
            next_fire=next_fire,
            prompt=prompt,
            created_by=interaction.user.id,
            created_by_name=interaction.user.display_name,
        )

        embed = self._build_job_embed(job, interaction.guild)
        await interaction.followup.send(
            f'✅ Tarefa **#{job["id"]}** criada! Dispara em {_fmt_brt(fire_dt)}.',
            embed=embed,
            ephemeral=True,
        )

    @schedule.command(name='list', description='Listar tarefas agendadas')
    async def schedule_list(self, interaction: discord.Interaction):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)

        entries = await asyncio.to_thread(self.store.list_active, interaction.guild.id)
        if not entries:
            return await interaction.response.send_message(
                'Nenhuma tarefa agendada. Use `/schedule create` para criar uma.',
                ephemeral=True,
            )

        lines = []
        for e in entries:
            status_icon = '▶️' if e['status'] == 'active' else '⏸️' if e['status'] == 'paused' else '🗑️'
            fire_dt = _parse_iso(e['next_fire'])
            lines.append(
                f"{status_icon} **[#{e['id']}]** {_fmt_brt(fire_dt)} • "
                f"{'cron' if e['type'] == 'cron' else 'uma vez'} • "
                f"<#{e['channel_id']}>\n"
                f"   ↳ {_snippet(e['prompt'], 80)}"
            )

        pages: list[discord.Embed] = []
        for i in range(0, len(lines), 10):
            pages.append(discord.Embed(
                title='⏰ Tarefas Agendadas',
                description='\n'.join(lines[i:i + 10]),
                color=discord.Color.dark_blue(),
            ))
        for i, page in enumerate(pages):
            page.set_footer(text=f'{len(entries)} tarefas • Página {i + 1}/{len(pages)}')

        if len(pages) == 1:
            await interaction.response.send_message(embed=pages[0], ephemeral=True)
        else:
            await interaction.response.send_message(
                embed=pages[0], view=PaginatedEmbedView(pages), ephemeral=True,
            )

    @schedule.command(name='show', description='Ver detalhes de uma tarefa')
    @app_commands.describe(id='ID da tarefa')
    async def schedule_show(self, interaction: discord.Interaction, id: int):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)

        job = await asyncio.to_thread(self.store.get, interaction.guild.id, id)
        if job is None:
            return await interaction.response.send_message(f'Tarefa #{id} não encontrada.', ephemeral=True)

        if not _is_admin(interaction) and job['created_by'] != interaction.user.id:
            return await interaction.response.send_message(
                'Você só pode ver suas próprias tarefas (ou seja admin).',
                ephemeral=True,
            )

        embed = self._build_job_embed(job, interaction.guild)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @schedule.command(name='edit', description='Editar uma tarefa (Admin ou criador)')
    @app_commands.describe(
        id='ID da tarefa',
        prompt='Novo prompt (deixe vazio para manter)',
        delay='Novo delay para once (deixe vazio para manter)',
        cron='Nova expressão cron para cron (deixe vazio para manter)',
        channel='Novo canal (deixe vazio para manter)',
    )
    async def schedule_edit(
        self,
        interaction: discord.Interaction,
        id: int,
        prompt: str | None = None,
        delay: str | None = None,
        cron: str | None = None,
        channel: discord.TextChannel | None = None,
    ):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)

        job = await asyncio.to_thread(self.store.get, interaction.guild.id, id)
        if job is None:
            return await interaction.response.send_message(f'Tarefa #{id} não encontrada.', ephemeral=True)

        if not _is_admin(interaction) and job['created_by'] != interaction.user.id:
            return await interaction.response.send_message(
                'Você só pode editar suas próprias tarefas (ou seja admin).',
                ephemeral=True,
            )

        if job['status'] == 'deleted':
            return await interaction.response.send_message('Tarefa deletada não pode ser editada.', ephemeral=True)

        new_prompt = None
        new_next_fire = None
        new_cron_expr = None
        new_channel_id = None

        if prompt is not None:
            if len(prompt) > SCHEDULER_MAX_PROMPT:
                return await interaction.response.send_message(
                    f'Prompt muito longo. Máximo {SCHEDULER_MAX_PROMPT} caracteres.',
                    ephemeral=True,
                )
            new_prompt = prompt

        if delay is not None:
            if job['type'] != 'once':
                return await interaction.response.send_message(
                    'O parâmetro **delay** só se aplica a tarefas "once".',
                    ephemeral=True,
                )
            try:
                fire_dt = _parse_delay(delay)
                if fire_dt <= _now():
                    return await interaction.response.send_message(
                        'A data/hora deve estar no futuro.', ephemeral=True,
                    )
                new_next_fire = fire_dt.isoformat()
            except ValueError as exc:
                return await interaction.response.send_message(f'❌ {exc}', ephemeral=True)

        if cron is not None:
            if job['type'] != 'cron':
                return await interaction.response.send_message(
                    'O parâmetro **cron** só se aplica a tarefas "cron".',
                    ephemeral=True,
                )
            try:
                parsed = _parse_cron(cron)
                fire_dt = _next_fire(parsed, _now())
                new_cron_expr = cron
                new_next_fire = fire_dt.isoformat()
            except ValueError as exc:
                return await interaction.response.send_message(f'❌ {exc}', ephemeral=True)

        if channel is not None:
            new_channel_id = channel.id

        await interaction.response.defer(ephemeral=True)

        updated = await asyncio.to_thread(
            self.store.update,
            interaction.guild.id, id,
            prompt=new_prompt,
            next_fire=new_next_fire,
            cron_expr=new_cron_expr,
            channel_id=new_channel_id,
        )

        await interaction.followup.send(
            f'✅ Tarefa **#{id}** atualizada!',
            embed=self._build_job_embed(updated or job, interaction.guild),
            ephemeral=True,
        )

    @schedule.command(name='pause', description='Pausar uma tarefa')
    @app_commands.describe(id='ID da tarefa')
    async def schedule_pause(self, interaction: discord.Interaction, id: int):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)

        job = await asyncio.to_thread(self.store.get, interaction.guild.id, id)
        if job is None:
            return await interaction.response.send_message(f'Tarefa #{id} não encontrada.', ephemeral=True)

        if not _is_admin(interaction) and job['created_by'] != interaction.user.id:
            return await interaction.response.send_message(
                'Você só pode pausar suas próprias tarefas (ou seja admin).',
                ephemeral=True,
            )

        if job['status'] == 'paused':
            return await interaction.response.send_message('Tarefa já está pausada.', ephemeral=True)

        await asyncio.to_thread(self.store.update, interaction.guild.id, id, status='paused')
        await interaction.response.send_message(f'⏸️ Tarefa **#{id}** pausada.', ephemeral=True)

    @schedule.command(name='resume', description='Retomar uma tarefa pausada')
    @app_commands.describe(id='ID da tarefa')
    async def schedule_resume(self, interaction: discord.Interaction, id: int):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)

        job = await asyncio.to_thread(self.store.get, interaction.guild.id, id)
        if job is None:
            return await interaction.response.send_message(f'Tarefa #{id} não encontrada.', ephemeral=True)

        if not _is_admin(interaction) and job['created_by'] != interaction.user.id:
            return await interaction.response.send_message(
                'Você só pode retomar suas próprias tarefas (ou seja admin).',
                ephemeral=True,
            )

        if job['status'] == 'active':
            return await interaction.response.send_message('Tarefa já está ativa.', ephemeral=True)

        if job['status'] == 'deleted':
            return await interaction.response.send_message('Tarefa deletada não pode ser retomada.', ephemeral=True)

        next_fire = job['next_fire']
        if job['type'] == 'cron' and job['cron_expr']:
            try:
                parsed = _parse_cron(job['cron_expr'])
                next_fire = _next_fire(parsed, _now()).isoformat()
            except Exception:
                logger.exception("[scheduler] failed to recompute next_fire on resume for job %s", id)
        elif job['type'] == 'once':
            fire_dt = _parse_iso(next_fire)
            if fire_dt and fire_dt <= _now():
                next_fire = (_now() + datetime.timedelta(minutes=1)).isoformat()

        await asyncio.to_thread(
            self.store.update, interaction.guild.id, id,
            status='active', next_fire=next_fire,
        )
        await interaction.response.send_message(f'▶️ Tarefa **#{id}** retomada.', ephemeral=True)

    @schedule.command(name='delete', description='Deletar uma tarefa')
    @app_commands.describe(id='ID da tarefa')
    async def schedule_delete(self, interaction: discord.Interaction, id: int):
        if interaction.guild is None:
            return await interaction.response.send_message('Requer estar em um servidor.', ephemeral=True)

        job = await asyncio.to_thread(self.store.get, interaction.guild.id, id)
        if job is None:
            return await interaction.response.send_message(f'Tarefa #{id} não encontrada.', ephemeral=True)

        if not _is_admin(interaction) and job['created_by'] != interaction.user.id:
            return await interaction.response.send_message(
                'Você só pode deletar suas próprias tarefas (ou seja admin).',
                ephemeral=True,
            )

        view = _ConfirmView(interaction.user.id)
        await interaction.response.send_message(
            f'⚠️ Confirmar exclusão da tarefa **#{id}**?\n'
            f'Prompt: {_snippet(job["prompt"], 100)}',
            view=view, ephemeral=True,
        )
        await view.wait()
        if not view.confirmed:
            return

        await asyncio.to_thread(self.store.update, interaction.guild.id, id, status='deleted')
        await interaction.followup.send(f'🗑️ Tarefa **#{id}** deletada.', ephemeral=True)

    # --- Agent tool execution (LLM tool-calling) ---

    async def exec_tool(
        self,
        name: str,
        args: dict,
        *,
        guild: discord.Guild,
        actor_name: str = 'bot',
        requester=None,
        channel=None,
    ) -> tuple[str, list[dict]]:
        """Execute a scheduler tool call from the LLM tool loop.

        Returns ``(result_text, sources)`` — sources is always empty for
        scheduler operations.  Mirrors ``Memory.exec_tool``'s signature.
        """
        try:
            if name == 'schedule_create':
                return await self._exec_create(
                    args, guild=guild, actor_name=actor_name,
                    requester=requester, channel=channel,
                )
            if name == 'schedule_list':
                return await self._exec_list(args, guild=guild)
            if name == 'schedule_delete':
                return await self._exec_delete(args, guild=guild, requester=requester)
        except SchedulerError as e:
            return f'⚠️ Agendador: {e}', []
        except Exception:
            logger.exception('[scheduler] unexpected error in %s', name)
            return '⚠️ Erro interno no agendador.', []
        return f'Ferramenta desconhecida: {name}', []

    async def _exec_create(
        self, args: dict, *, guild: discord.Guild, actor_name: str,
        requester, channel,
    ) -> tuple[str, list[dict]]:
        prompt = (args.get('prompt') or '').strip()
        if not prompt:
            raise SchedulerError('prompt é obrigatório.')
        if len(prompt) > SCHEDULER_MAX_PROMPT:
            raise SchedulerError(f'prompt excede {SCHEDULER_MAX_PROMPT} caracteres.')

        job_type = (args.get('type') or 'once').strip().lower()
        if job_type not in ('once', 'cron'):
            raise SchedulerError('type deve ser "once" ou "cron".')

        delay = args.get('delay')
        cron_expr = args.get('cron')
        channel_id = getattr(channel, 'id', None) if channel else None
        if not channel_id:
            raise SchedulerError('requer estar em um canal de texto.')

        count = await asyncio.to_thread(self.store.count_active, guild.id)
        if count >= SCHEDULER_MAX_JOBS_PER_GUILD:
            raise SchedulerError(f'limite de {SCHEDULER_MAX_JOBS_PER_GUILD} tarefas ativas por servidor.')

        try:
            if job_type == 'once':
                if not delay:
                    raise SchedulerError(
                        'tarefas "once" precisam de delay. Ex: "5m", "2h", "2026-09-04 14:30".'
                    )
                fire_dt = _parse_delay(str(delay))
                if fire_dt <= _now():
                    raise SchedulerError('a data/hora deve estar no futuro.')
                cron_expr_store = None
                next_fire = fire_dt.isoformat()
            else:
                if not cron_expr:
                    raise SchedulerError(
                        'tarefas "cron" precisam de uma expressão cron. Ex: "0 2 * * *".'
                    )
                parsed = _parse_cron(str(cron_expr))
                fire_dt = _next_fire(parsed, _now())
                cron_expr_store = str(cron_expr)
                next_fire = fire_dt.isoformat()
        except ValueError as e:
            raise SchedulerError(str(e))

        actor_id = str(getattr(requester, 'id', '')) or '0'
        actor_display = getattr(requester, 'display_name', None) or actor_name

        job = await asyncio.to_thread(
            self.store.create,
            guild_id=guild.id,
            channel_id=channel_id,
            job_type=job_type,
            cron_expr=cron_expr_store,
            next_fire=next_fire,
            prompt=prompt,
            created_by=int(actor_id) if actor_id.isdigit() else 0,
            created_by_name=actor_display,
        )

        return (
            f'✅ Tarefa #{job["id"]} criada! Dispara em {_fmt_brt(fire_dt)} '
            f'({job_type}). Prompt: {_snippet(prompt, 80)}.',
            [],
        )

    async def _exec_list(self, args: dict, *, guild: discord.Guild) -> tuple[str, list[dict]]:
        entries = await asyncio.to_thread(self.store.list_active, guild.id)
        if not entries:
            return 'Nenhuma tarefa agendada ativa.', []
        lines = []
        for e in entries:
            fire_dt = _parse_iso(e['next_fire'])
            icon = '▶️' if e['status'] == 'active' else '⏸️'
            lines.append(
                f'{icon} #{e["id"]} — {e["type"]} — {_fmt_brt(fire_dt)} '
                f'— <#{e["channel_id"]}> — {_snippet(e["prompt"], 60)}'
            )
        return f'**Tarefas agendadas ({len(entries)}):**\n' + '\n'.join(lines), []

    async def _exec_delete(
        self, args: dict, *, guild: discord.Guild, requester,
    ) -> tuple[str, list[dict]]:
        job_id = args.get('id')
        if not job_id:
            raise SchedulerError('id é obrigatório.')
        try:
            job_id_int = int(job_id)
        except (TypeError, ValueError):
            raise SchedulerError('id deve ser um número.')

        job = await asyncio.to_thread(self.store.get, guild.id, job_id_int)
        if job is None:
            raise SchedulerError(f'tarefa #{job_id_int} não encontrada.')

        is_admin = (
            getattr(getattr(requester, 'guild_permissions', None), 'administrator', False)
            if requester else False
        )
        requester_id = str(getattr(requester, 'id', ''))
        if not is_admin and str(job['created_by']) != requester_id:
            raise SchedulerError('você só pode deletar suas próprias tarefas.')

        await asyncio.to_thread(self.store.update, guild.id, job_id_int, status='deleted')
        return f'✅ Tarefa #{job_id_int} deletada.', []

    # --- Helpers ---

    def _build_job_embed(self, job: dict, guild: discord.Guild) -> discord.Embed:
        fire_dt = _parse_iso(job['next_fire'])
        last_dt = _parse_iso(job['last_fired_at'])
        status_map = {'active': '▶️ Ativa', 'paused': '⏸️ Pausada', 'deleted': '🗑️ Deletada'}
        embed = discord.Embed(
            title=f"⏰ Tarefa #{job['id']}",
            description=_snippet(job['prompt'], 3800),
            color=discord.Color.dark_blue(),
        )
        embed.add_field(name='Tipo', value='cron' if job['type'] == 'cron' else 'uma vez', inline=True)
        embed.add_field(name='Status', value=status_map.get(job['status'], job['status']), inline=True)
        embed.add_field(name='Próximo disparo', value=_fmt_brt(fire_dt), inline=True)
        if job['type'] == 'cron' and job['cron_expr']:
            try:
                parsed = _parse_cron(job['cron_expr'])
                embed.add_field(name='Cron', value=f'`{_cron_human(parsed)}`', inline=False)
            except Exception:
                embed.add_field(name='Cron', value=f'`{job["cron_expr"]}`', inline=False)
        embed.add_field(name='Canal', value=f'<#{job["channel_id"]}>', inline=True)
        embed.add_field(name='Disparos', value=str(job['fire_count']), inline=True)
        embed.add_field(name='Último disparo', value=_fmt_brt(last_dt), inline=True)
        embed.add_field(name='Criado por', value=job['created_by_name'] or '?', inline=True)
        embed.add_field(name='Criado em', value=_fmt_brt(_parse_iso(job['created_at'])), inline=True)
        embed.set_footer(text=f'ID {job["id"]} • guild {guild.id}')
        return embed


def _snippet(text: str, n: int = 350) -> str:
    text = (text or '').strip()
    return text if len(text) <= n else text[:n] + '…'


# ---------------------------------------------------------------------------
# Agent tool schemas (for LLM tool-calling)
# ---------------------------------------------------------------------------

SCHEDULE_CREATE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'schedule_create',
        'description': (
            'Cria uma tarefa agendada que dispara automaticamente no futuro. '
            'O bot executa o prompt via LLM no horário marcado e posta a resposta no canal. '
            'Use para lembretes, verificações periódicas, ou tarefas que devem correr sem intervenção humana. '
            'Tipos: "once" (dispara uma vez e é removido) ou "cron" (recorrente). '
            'Para once, forneça delay ("5m", "2h", "1d") ou datetime ("2026-09-04 14:30"). '
            'Para cron, forneça uma expressão cron de 5 campos: minuto hora dia mês dia-da-semana. '
            'Ex: "0 2 * * *" = todo dia às 2 AM; "*/30 * * * *" = a cada 30 min; "0 9 * * 1-5" = 9 AM de seg a sex.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'prompt': {
                    'type': 'string',
                    'description': 'Instrução do que o bot deve fazer quando a tarefa disparar. Ex: "Avise os usuários que o servidor reinicia em 5 minutos".',
                },
                'type': {
                    'type': 'string',
                    'enum': ['once', 'cron'],
                    'description': 'once: dispara uma vez e é removido; cron: recorrente.',
                },
                'delay': {
                    'type': 'string',
                    'description': 'Para once: "5m", "2h", "1d", ou datetime "2026-09-04 14:30".',
                },
                'cron': {
                    'type': 'string',
                    'description': 'Para cron: 5 campos "min hora dia mês dia-semana". Ex: "0 2 * * *".',
                },
            },
            'required': ['prompt', 'type'],
        },
    },
}

SCHEDULE_LIST_TOOL = {
    'type': 'function',
    'function': {
        'name': 'schedule_list',
        'description': (
            'Lista as tarefas agendadas ativas do servidor. Retorna ID, tipo, próximo disparo, '
            'canal e um resumo do prompt. Use quando o usuário perguntar sobre tarefas existentes.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {},
            'required': [],
        },
    },
}

SCHEDULE_DELETE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'schedule_delete',
        'description': (
            'Remove (deleta) uma tarefa agendada pelo ID. Use quando o usuário pedir para '
            'cancelar ou remover um agendamento. Não pode ser desfeito.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'id': {
                    'type': 'integer',
                    'description': 'ID da tarefa a deletar.',
                },
            },
            'required': ['id'],
        },
    },
}

SCHEDULER_TOOLS = [SCHEDULE_CREATE_TOOL, SCHEDULE_LIST_TOOL, SCHEDULE_DELETE_TOOL]


async def setup(bot: commands.Bot):
    if not SCHEDULER_ENABLED:
        logger.info('Scheduler disabled via SCHEDULER_ENABLED=false')
        return
    await bot.add_cog(Scheduler(bot))
