"""Context the model gets with each turn: who, where, when, and what was said.

Builds the ``<contexto>`` block (author, server, channel, clock) and fetches
channel messages: the recent window injected into a turn, the chatter between
two turns of a conversation, the history tool and the ±N window around one
message. Images in those messages are persisted for vision.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Final

import discord

from config import (
    BR_TZ,
    CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES,
    CHANNEL_CONTEXT_IMAGES_ENABLED,
    CHANNEL_CONTEXT_MAX_CHARS,
    CHANNEL_CONTEXT_MESSAGES,
    CONVERSATIONS_GAP_MESSAGES,
)
from core import image_store, message_media
from core.conversations import CONTEXT_ONLY_NOTE
from core.discord_text import _fmt_dt, format_message_lines

logger = logging.getLogger(__name__)

CHANNEL_CONTEXT_MAX_IMAGES: Final[int] = 4


@dataclass
class ChannelContext:
    """Textual channel context plus persisted images available to the model."""

    text: str = ''
    lines: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text or self.lines or self.images)



def build_user_context(member: discord.abc.User | discord.Member | None) -> str:
    if not member:
        return "- Autor desta mensagem: desconhecido"
    # "Autor desta mensagem", not "Usuário": this block is stored with its turn
    # and replayed verbatim, so it must stay true once newer messages follow.
    lines = [f"- Autor desta mensagem: {getattr(member, 'display_name', str(member))} (@{member.name}) id={member.id}"]
    created = getattr(member, 'created_at', None)
    if created:
        lines.append(f"  Conta criada: {_fmt_dt(created)}")
    if isinstance(member, discord.Member):
        joined = getattr(member, 'joined_at', None)
        if joined:
            lines.append(f"  Entrou no servidor: {_fmt_dt(joined)}")
        try:
            roles = [r.name for r in member.roles if r.name != "@everyone"]
            if roles:
                lines.append(f"  Cargos: {', '.join(roles[:10])}")
            if member.guild_permissions.administrator:
                lines.append("  Permissão: Administrador")
            elif member.guild_permissions.manage_guild:
                lines.append("  Permissão: Gerenciador do servidor")
        except Exception:
            pass
        if member.nick and member.nick != member.display_name:
            lines.append(f"  Apelido: {member.nick}")
    return "\n".join(lines)


def build_guild_context(guild: discord.Guild | None) -> str:
    if not guild:
        return "Servidor: DM / mensagem direta (sem guild)"
    try:
        owner = getattr(guild, 'owner', None)
        owner_str = f"{owner} ({guild.owner_id})" if owner else str(guild.owner_id)
    except Exception:
        owner_str = str(getattr(guild, 'owner_id', '—'))
    text_channels = len([c for c in guild.channels if isinstance(c, discord.TextChannel)])
    voice_channels = len([c for c in guild.channels if isinstance(c, discord.VoiceChannel)])
    role_names = [r.name for r in guild.roles if r.name != "@everyone"][:15]
    lines = [
        f"- Servidor: {guild.name} id={guild.id}",
        f"  Membros: {guild.member_count or len(guild.members) if hasattr(guild, 'members') else '—'} | Canais: #{text_channels} texto / {voice_channels} voz",
        f"  Dono: {owner_str}",
        f"  Criado em: {_fmt_dt(getattr(guild, 'created_at', None))}",
    ]
    if role_names:
        lines.append(f"  Cargos ({len(guild.roles)-1}): {', '.join(role_names)}")
    return "\n".join(lines)


def build_channel_context(channel: discord.abc.GuildChannel | discord.Thread | discord.DMChannel | None) -> str:
    if not channel:
        return "Canal: desconhecido"
    ch_type = type(channel).__name__
    name = getattr(channel, 'name', getattr(channel, 'recipient', 'DM'))
    topic = getattr(channel, 'topic', None)
    lines = [f"- Canal: #{name} id={channel.id} tipo={ch_type}"]
    if topic:
        lines.append(f"  Tópico: {topic[:200]}")
    parent = getattr(channel, 'parent', None)
    if parent:
        lines.append(f"  Thread de: #{parent.name}")
    return "\n".join(lines)


def build_temporal_context(now: datetime.datetime | None = None, created_at: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    br_now = now.astimezone(BR_TZ)
    lines = [
        f"- Data/hora desta mensagem: {br_now.strftime('%d/%m/%Y %H:%M:%S BRT')} ({now.strftime('%Y-%m-%d %H:%M UTC')})",
    ]
    if created_at:
        lines.append(f"  Mensagem enviada em: {_fmt_dt(created_at)} UTC={created_at.strftime('%Y-%m-%d %H:%M UTC') if hasattr(created_at, 'strftime') else created_at}")
    return "\n".join(lines)


def build_full_context_block(
    user: discord.abc.User | discord.Member | None,
    guild: discord.Guild | None,
    channel: discord.abc.GuildChannel | discord.Thread | discord.DMChannel | None,
    created_at: datetime.datetime | None = None,
) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    parts = [
        "<contexto>",
        build_user_context(user),
        build_guild_context(guild),
        build_channel_context(channel),
        build_temporal_context(now, created_at),
        "</contexto>",
    ]
    return "\n".join(parts)


def _context_images_eligible(msg: discord.Message, now: datetime.datetime) -> bool:
    """False when channel-context images are disabled or *msg* is too old.

    Stale images are rarely what a question is about (see
    ``CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES``), and every one sent costs tokens.
    """
    if not CHANNEL_CONTEXT_IMAGES_ENABLED:
        return False
    created = getattr(msg, 'created_at', None)
    return not (
        CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES > 0
        and created is not None
        and now - created > datetime.timedelta(minutes=CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES)
    )


def _context_image_attachments(msg: discord.Message, now: datetime.datetime) -> list[Any]:
    """Visual media of a context message eligible to be shown to the model.

    Image attachments (GIFs included), stickers and GIF/image link embeds;
    empty when :func:`_context_images_eligible` says no.
    """
    if not _context_images_eligible(msg, now):
        return []
    return message_media.media_sources(msg)


def _context_image_groups(messages: list[discord.Message], now: datetime.datetime) -> list[list[Any]]:
    """Per-message image sources for *messages* (newest first), within budget.

    Media (attachments, stickers, GIFs) take the ``CHANNEL_CONTEXT_MAX_IMAGES``
    slots first, newest messages first; custom-emoji sheets (one per message)
    only fill the slots left over, since their names are already in the text.
    Groups keep the newest-first order :func:`_persist_context_images` expects.
    """
    slots = CHANNEL_CONTEXT_MAX_IMAGES
    by_index: dict[int, list[Any]] = {}
    for index, msg in enumerate(messages):
        if slots <= 0:
            break
        media = _context_image_attachments(msg, now)[:slots]
        if media:
            by_index[index] = media
            slots -= len(media)
    for index, msg in enumerate(messages):
        if slots <= 0:
            break
        sheet = message_media.emoji_sheet(msg) if _context_images_eligible(msg, now) else None
        if sheet is not None:
            by_index.setdefault(index, []).append(sheet)
            slots -= 1
    return [by_index[i] for i in sorted(by_index)]


async def _persist_context_images(groups: list[list[Any]]) -> list[str]:
    """Persist per-message attachment groups concurrently.

    *groups* come newest-first (history walk order); the refs are returned in
    chronological order, matching the rendered context lines.
    """
    if not groups:
        return []
    results = await asyncio.gather(
        *(image_store.persist_images(group) for group in groups),
        return_exceptions=True,
    )
    refs: list[str] = []
    for result in reversed(results):
        if isinstance(result, BaseException):
            logger.warning('Failed to persist channel-context images: %s', result)
            continue
        refs.extend(result)
    return refs


async def fetch_channel_history(
    bot: discord.Client,
    channel: discord.abc.Messageable | None,
    limit: int = 20,
    channel_id: str | None = None,
    before: discord.abc.Snowflake | None = None,
    include_images: bool = False,
) -> str | ChannelContext:
    target = channel
    if channel_id:
        try:
            cid = int(channel_id)
            target = bot.get_channel(cid)
            if target is None:
                try:
                    target = await bot.fetch_channel(cid)
                except Exception:
                    return f"Canal {channel_id} não encontrado ou sem acesso."
        except ValueError:
            return f"channel_id inválido: {channel_id}"
    if target is None or not hasattr(target, "history"):
        return "Histórico não disponível para este canal."
    limit = max(1, min(50, int(limit)))
    messages: list[discord.Message] = []
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        async for msg in target.history(limit=limit, before=before):
            messages.append(msg)
        lines = await format_message_lines(messages)
    except discord.Forbidden:
        return "Sem permissão para ler histórico deste canal."
    except Exception as e:
        logger.exception("fetch_channel_history failed")
        return f"Erro ao buscar histórico: {e}"
    if not lines:
        empty = "Nenhuma mensagem encontrada no histórico."
        return ChannelContext(text=empty) if include_images else empty
    lines.reverse()
    header = (
        f"Histórico de #{getattr(target, 'name', target.id)} "
        f"(channel_id={target.id}) — últimas {len(lines)} mensagens, cronológica:\n"
    )
    text = header + "\n".join(lines)
    if not include_images:
        return text
    images = await _persist_context_images(_context_image_groups(messages, now))
    return ChannelContext(text=text, lines=lines, images=images)


async def fetch_recent_channel_context(
    bot: discord.Client,
    channel: discord.abc.Messageable | None,
    *,
    before: discord.abc.Snowflake | None = None,
    limit: int | None = None,
) -> ChannelContext | None:
    """Latest channel messages as auto-injected LLM context.

    Controlled by ``CHANNEL_CONTEXT_MESSAGES`` (0 disables). Returns ``None``
    when disabled or without a readable channel. Image attachments are
    persisted and returned separately from the formatted text. ``before``
    excludes the triggering message from the window (mention/follow-up flows).
    The block opens with a note that it is background: the window rides in
    the same user message as the question, and without it the model answers
    the channel's other topics too.
    """
    n = CHANNEL_CONTEXT_MESSAGES if limit is None else limit
    if n <= 0 or channel is None:
        return None
    context = await fetch_channel_history(
        bot, channel, limit=n, before=before, include_images=True,
    )
    if not isinstance(context, ChannelContext):
        context = ChannelContext(text=context)
    context = _trim_oldest(context, CHANNEL_CONTEXT_MAX_CHARS)
    logger.info(
        "Recent channel context channel=%s messages=%d images=%d",
        getattr(channel, 'id', None), len(context.lines), len(context.images),
    )
    return ChannelContext(
        text=(
            "<mensagens_recentes_do_canal>\n"
            f"(O que o canal falava antes da mensagem que você vai responder — {CONTEXT_ONLY_NOTE}.)\n"
            f"{context.text}\n</mensagens_recentes_do_canal>"
        ),
        lines=context.lines,
        images=context.images,
    )


def _trim_oldest(context: ChannelContext, max_chars: int) -> ChannelContext:
    """Drop the oldest lines until the window fits *max_chars* (newest lines always kept)."""
    lines = context.lines
    if max_chars <= 0 or not lines or sum(len(line) + 1 for line in lines) <= max_chars:
        return context
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        if kept and used + len(line) + 1 > max_chars:
            break
        kept.append(line)
        used += len(line) + 1
    kept.reverse()
    header = context.text.split('\n', 1)[0].replace(f'últimas {len(lines)} mensagens', f'últimas {len(kept)} mensagens')
    return ChannelContext(text=header + '\n' + '\n'.join(kept), lines=kept, images=context.images)


async def fetch_channel_gap(
    channel: discord.abc.Messageable | None,
    *,
    after_id: str | int,
    before: discord.abc.Snowflake,
    skip_ids: set[str] | None = None,
    limit: int | None = None,
) -> ChannelContext:
    """Human channel messages between a stored turn and a new follow-up.

    Anchors on the previous bot-directed turn's message id and returns the
    chatter in between as canonical lines (bot messages excluded — bot answers
    are already replayed as conversation turns). The walk goes backwards from
    ``before`` and stops at the anchor id, so nothing older than the previous
    turn leaks in; only human messages consume the budget, so truncation keeps
    the NEWEST chatter lines. Controlled by ``CONVERSATIONS_GAP_MESSAGES``
    (0 disables). Returns an empty context on failure — the recent-channel
    window is then the fallback.
    """
    n = CONVERSATIONS_GAP_MESSAGES if limit is None else limit
    if n <= 0 or channel is None or not hasattr(channel, "history"):
        return ChannelContext()
    try:
        anchor = discord.Object(id=int(after_id))
    except (TypeError, ValueError):
        return ChannelContext()
    skip = {str(s) for s in (skip_ids or set()) if s}
    skip.add(str(after_id))
    messages: list[discord.Message] = []
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        async for msg in channel.history(limit=max(n * 3, 100), before=before):
            if msg.id <= anchor.id:
                break
            if msg.author.bot or str(msg.id) in skip:
                continue
            if not msg.content and not msg.attachments and not msg.embeds and not message_media.stickers(msg):
                continue
            messages.append(msg)
            if len(messages) >= n:
                break
        lines = await format_message_lines(messages)
        images = await _persist_context_images(_context_image_groups(messages, now))
    except Exception:
        logger.exception("fetch_channel_gap failed")
        return ChannelContext()
    lines.reverse()
    return ChannelContext(text="\n".join(lines), lines=lines, images=images)


async def fetch_turn_gap(message: discord.Message, history: list[dict]) -> ChannelContext:
    """Channel chatter since the previous bot-directed turn (same channel only)."""
    last_turn = history[-1] if history else None
    anchor_id = (last_turn or {}).get('message_id')
    if not anchor_id:
        return ChannelContext()
    if last_turn.get('channel_id') and str(last_turn['channel_id']) != str(message.channel.id):
        return ChannelContext()
    return await fetch_channel_gap(
        message.channel,
        after_id=anchor_id,
        before=message,
        skip_ids={t.get('message_id') for t in history if t.get('message_id')},
    )


async def fetch_message_context(
    bot: discord.Client,
    channel_id: str,
    message_id: str,
    window: int = 5,
    guild_id: int | None = None,
) -> str | None:
    """Fetch ±window messages around one message via the Discord API.

    Returns None on any failure (missing message, missing channel, no
    permission, invalid ids, channel outside the requesting guild) so callers
    can fall back to the history index.
    """
    try:
        window = max(1, min(10, int(window)))
    except (TypeError, ValueError):
        window = 5
    try:
        cid = int(channel_id)
        mid = int(message_id)
    except (TypeError, ValueError):
        return None
    channel = bot.get_channel(cid)
    if channel is None:
        try:
            channel = await bot.fetch_channel(cid)
        except Exception:
            return None
    if guild_id is not None and getattr(channel, "guild", None) is not None and channel.guild.id != guild_id:
        return None
    if not hasattr(channel, "fetch_message") or not hasattr(channel, "history"):
        return None
    try:
        target = await channel.fetch_message(mid)
    except discord.NotFound:
        return None
    except discord.Forbidden:
        return None
    except Exception:
        return None
    before: list[discord.Message] = []
    after: list[discord.Message] = []
    try:
        async for m in channel.history(limit=window, before=target):
            before.append(m)
        before.reverse()
    except Exception:
        pass
    try:
        async for m in channel.history(limit=window, after=target):
            after.append(m)
    except Exception:
        pass
    window_messages = [*before, target, *after]
    lines = [
        ("▶ " if m is target else "  ") + line
        for m, line in zip(window_messages, await format_message_lines(window_messages))
    ]
    header = (
        f"Contexto ao redor de {message_id} em #{getattr(channel, 'name', channel_id)} "
        f"(channel_id={channel.id}, ±{window}):\n"
    )
    return header + "\n".join(lines)
