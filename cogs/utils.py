"""Shared text utilities used across multiple cogs."""

import datetime
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from zoneinfo import ZoneInfo

import discord
from openai import AsyncOpenAI

from config import (
    CHANNEL_CONTEXT_MESSAGES,
    CONVERSATION_GAP_MESSAGES,
    HISTORY_MAX_MSG_LENGTH,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

BR_TZ = ZoneInfo("America/Sao_Paulo")

CHANNEL_HISTORY_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_channel_history',
        'description': (
            'Busca mensagens recentes do canal atual ou de outro canal do servidor. '
            'Use quando o usuário perguntar sobre conversas anteriores, contexto recente, '
            'ou quando precisar entender o que foi discutido antes. Cada linha segue o formato '
            '[data] Nome (@usuário) <author_id=…>: conteúdo ↩ reply_to=… [msg_id=…]. '
            'Use author_id/msg_id/reply_to com search_history, get_user_stats ou get_message_context.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'limit': {
                    'type': 'integer',
                    'description': 'Número de mensagens a buscar (padrão 20, máximo 50).',
                },
                'channel_id': {
                    'type': 'string',
                    'description': 'ID do canal para buscar. Omita para usar o canal atual.',
                },
            },
            'required': [],
        },
    },
}

GUILD_INFO_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_guild_info',
        'description': (
            'Retorna informações detalhadas sobre o servidor Discord atual: nome, '
            'quantidade de membros, canais, cargos e dono.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {},
            'required': [],
        },
    },
}

SEARCH_HISTORY_TOOL = {
    'type': 'function',
    'function': {
        'name': 'search_history',
        'description': (
            'Busca semanticamente no histórico completo do servidor (RAG). '
            'Use quando o usuário perguntar sobre conversas anteriores, decisões, '
            'problemas já discutidos, ou contexto que pode estar no chat. Retorna '
            'mensagens relevantes com autor, canal, horário e link, incluindo 5 mensagens de contexto local.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Consulta em linguagem natural sobre o histórico.',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Número de resultados (padrão 5, máximo 12).',
                },
                'channel_id': {
                    'type': 'string',
                    'description': 'Opcional: restringir busca a um canal específico (ID).',
                },
                'author_id': {
                    'type': 'string',
                    'description': 'Opcional: filtrar por autor (ID do usuário Discord).',
                },
                'author_name': {
                    'type': 'string',
                    'description': 'Opcional: filtrar por nome/apelido do autor (busca parcial, ex: "joao").',
                },
                'after': {
                    'type': 'string',
                    'description': 'Opcional: data inicial ISO (YYYY-MM-DD ou YYYY-MM-DDTHH:MM:SS). Filtra mensagens após esta data.',
                },
                'before': {
                    'type': 'string',
                    'description': 'Opcional: data final ISO. Filtra mensagens antes desta data.',
                },
                'search_mode': {
                    'type': 'string',
                    'enum': ['semantic', 'keyword', 'hybrid'],
                    'description': 'Modo de busca: semantic (embedding), keyword (texto exato), hybrid (0.7 semantic + 0.3 keyword, padrão).',
                },
                'sort_by': {
                    'type': 'string',
                    'enum': ['relevance', 'recent'],
                    'description': 'Ordenação: relevance (similaridade) ou recent (mais recentes primeiro).',
                },
            },
            'required': ['query'],
        },
    },
}

GET_USER_STATS_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_user_stats',
        'description': (
            'Retorna estatísticas de um usuário no histórico: total de mensagens, canais mais ativos, '
            'horários mais ativos, primeira/última mensagem, média de tamanho. Use para "o que X mais fala?" ou perfil do usuário.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'author_id': {'type': 'string', 'description': 'ID do usuário Discord.'},
                'author_name': {'type': 'string', 'description': 'Nome/apelido parcial se ID desconhecido.'},
            },
            'required': [],
        },
    },
}

COUNT_MENTIONS_TOOL = {
    'type': 'function',
    'function': {
        'name': 'count_mentions',
        'description': (
            'Conta menções de um tópico/query no histórico e agrupa por autor ou canal. '
            'Use para "quem fala mais sobre X?" ou "quantas vezes falamos sobre X?".'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Tópico/query a contar.'},
                'group_by': {'type': 'string', 'enum': ['author', 'channel', 'day'], 'description': 'Agrupamento (padrão author).'},
                'limit': {'type': 'integer', 'description': 'Número de grupos (padrão 10).'},
                'after': {'type': 'string', 'description': 'Data inicial ISO.'},
                'before': {'type': 'string', 'description': 'Data final ISO.'},
            },
            'required': ['query'],
        },
    },
}

GET_MESSAGE_CONTEXT_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_message_context',
        'description': (
            'Retorna o contexto local ao redor de uma mensagem específica (5 antes e 5 depois). '
            'Use após search_history para expandir o contexto de um resultado relevante. '
            'Aceita o msg_id ou reply_to visto em qualquer linha de mensagem.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'message_id': {
                    'type': 'string',
                    'description': 'ID da mensagem central.',
                },
                'channel_id': {
                    'type': 'string',
                    'description': 'ID do canal da mensagem.',
                },
                'window': {
                    'type': 'integer',
                    'description': 'Janela de contexto (padrão 5, máximo 10).',
                },
            },
            'required': ['message_id', 'channel_id'],
        },
    },
}


def _fmt_dt(dt: datetime.datetime | None) -> str:
    if not dt:
        return "—"
    try:
        return dt.astimezone(BR_TZ).strftime("%d/%m/%Y %H:%M BRT")
    except Exception:
        return dt.isoformat()


def _fmt_dt_line(dt: datetime.datetime | None) -> str:
    if not dt:
        return "—"
    try:
        return dt.astimezone(BR_TZ).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return dt.isoformat()


# ---------------------------------------------------------------------------
# Canonical message-line format
#
# Every renderer of Discord messages (channel history, message context,
# search results, history chunks, conversation turns) emits lines in this
# exact shape so the LLM can cross-reference messages between tools:
#
#   [dd/mm/aaaa HH:MM] Display (@handle) <author_id=123>: content ↩ reply_to=456 (Alvo) [msg_id=789]
#
# - ``author_id`` feeds search_history/get_user_stats filters;
# - ``reply_to`` (only for replies) feeds get_message_context;
# - ``msg_id`` always comes LAST so content truncation never eats it.
# ---------------------------------------------------------------------------

def message_content_text(msg: discord.Message, *, max_length: int = HISTORY_MAX_MSG_LENGTH) -> str:
    """Flattened message content with attachment/embed markers, never empty."""
    c = msg.content or ""
    if msg.attachments:
        c += " " + " ".join(f"[anexo:{a.filename}]" for a in msg.attachments)
    if not c.strip() and msg.embeds:
        try:
            c = f"[embed: {msg.embeds[0].title or msg.embeds[0].description[:100]}]"
        except Exception:
            c = "[embed]"
    c = c.replace("\n", " ").strip()
    if not c:
        c = "[sem texto]"
    if len(c) > max_length:
        c = c[:max_length] + "…"
    return c


def format_message_line(msg: discord.Message, *, is_target: bool = False) -> str:
    """Render a Discord message in the canonical line format."""
    display = getattr(msg.author, 'display_name', str(msg.author))
    handle = getattr(msg.author, 'name', None)
    author = f"{display} (@{handle})" if handle else display
    author_id = getattr(msg.author, 'id', '')
    line = f"[{_fmt_dt_line(msg.created_at)}] {author} <author_id={author_id}>: {message_content_text(msg)}"
    ref = getattr(msg, 'reference', None)
    ref_id = getattr(ref, 'message_id', None)
    if ref_id:
        target = ""
        resolved = getattr(ref, 'resolved', None)
        if resolved is not None and not isinstance(resolved, discord.DeletedReferencedMessage):
            rdisplay = getattr(resolved.author, 'display_name', '')
            if rdisplay:
                target = f" ({rdisplay})"
        line += f" ↩ reply_to={ref_id}{target}"
    line += f" [msg_id={msg.id}]"
    if is_target:
        line += "  ← alvo"
    return line


def format_chunk_line(ch: dict) -> str:
    """Render a stored history chunk (cogs/history_rag.py) in the canonical format."""
    ts_raw = str(ch.get('ts', ''))
    try:
        dt = datetime.datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        ts = dt.astimezone(BR_TZ).strftime("%d/%m/%Y %H:%M")
    except Exception:
        ts = ts_raw[:19]
    author = ch.get('author_full') or ch.get('author_name') or '?'
    content = str(ch.get('content', ''))[:HISTORY_MAX_MSG_LENGTH]
    if not content:
        content = '[sem texto]'
    line = f"[{ts}] {author} <author_id={ch.get('author_id', '?')}>: {content}"
    if ch.get('reply_to'):
        line += f" ↩ reply_to={ch['reply_to']}"
    return f"{line} [msg_id={ch.get('msg_id', '')}]"


def render_search_results(results: list[dict], *, window_chars: int = 1200) -> str:
    """Render search_history results (shared by commands.py and docs_rag.py)."""
    parts: list[str] = []
    for r in results:
        jump = r.get('jump_url', '')
        link = f"[ver]({jump})" if jump else ''
        header = (
            f"**{r.get('author_full', '?')}** em #{r.get('channel_name', '?')} "
            f"(channel_id={r.get('channel_id', '?')}) — {str(r.get('ts', ''))[:19]} "
            f"{link} (score {r.get('_score', 0):.2f})"
        )
        window = str(r.get('chunk_text', r.get('content', '')))[:window_chars]
        window = window.replace('```', 'ˋˋˋ')
        parts.append(f"{header}\n```\n{window}\n```\n`msg_id={r.get('msg_id')}`")
    return "\n\n---\n\n".join(parts)


def build_user_context(member: discord.abc.User | discord.Member | None) -> str:
    if not member:
        return "Usuário: desconhecido"
    lines = [f"- Usuário: {getattr(member, 'display_name', str(member))} (@{member.name}) id={member.id}"]
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
        f"- Agora: {br_now.strftime('%d/%m/%Y %H:%M:%S BRT')} ({now.strftime('%Y-%m-%d %H:%M UTC')})",
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


async def fetch_channel_history(
    bot: discord.Client,
    channel: discord.abc.Messageable | None,
    limit: int = 20,
    channel_id: str | None = None,
    before: discord.abc.Snowflake | None = None,
) -> str:
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
    lines: list[str] = []
    try:
        async for msg in target.history(limit=limit, before=before):
            lines.append(format_message_line(msg))
    except discord.Forbidden:
        return "Sem permissão para ler histórico deste canal."
    except Exception as e:
        logger.exception("fetch_channel_history failed")
        return f"Erro ao buscar histórico: {e}"
    if not lines:
        return "Nenhuma mensagem encontrada no histórico."
    lines.reverse()
    header = (
        f"Histórico de #{getattr(target, 'name', target.id)} "
        f"(channel_id={target.id}) — últimas {len(lines)} mensagens, cronológica:\n"
    )
    return header + "\n".join(lines)


async def fetch_recent_channel_context(
    bot: discord.Client,
    channel: discord.abc.Messageable | None,
    *,
    before: discord.abc.Snowflake | None = None,
    limit: int | None = None,
) -> str | None:
    """Latest channel messages as auto-injected LLM context.

    Controlled by ``CHANNEL_CONTEXT_MESSAGES`` (0 disables). Returns ``None``
    when disabled or without a readable channel. ``before`` excludes the
    triggering message from the window (mention/follow-up flows).
    """
    n = CHANNEL_CONTEXT_MESSAGES if limit is None else limit
    if n <= 0 or channel is None:
        return None
    text = await fetch_channel_history(bot, channel, limit=n, before=before)
    return f"<mensagens_recentes_do_canal>\n{text}\n</mensagens_recentes_do_canal>"


async def fetch_channel_gap(
    channel: discord.abc.Messageable | None,
    *,
    after_id: str | int,
    before: discord.abc.Snowflake,
    skip_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Human channel messages between a stored turn and a new follow-up.

    Anchors on the previous bot-directed turn's message id and returns the
    chatter in between as canonical lines (bot messages excluded — bot answers
    are already replayed as conversation turns). Controlled by
    ``CONVERSATION_GAP_MESSAGES`` (0 disables). Returns [] on failure — the
    recent-channel window is then the fallback.
    """
    n = CONVERSATION_GAP_MESSAGES if limit is None else limit
    if n <= 0 or channel is None or not hasattr(channel, "history"):
        return []
    try:
        anchor = discord.Object(id=int(after_id))
    except (TypeError, ValueError):
        return []
    skip = {str(s) for s in (skip_ids or set()) if s}
    skip.add(str(after_id))
    lines: list[str] = []
    try:
        async for msg in channel.history(limit=n, after=anchor, before=before):
            if msg.author.bot or str(msg.id) in skip:
                continue
            if not msg.content and not msg.attachments and not msg.embeds:
                continue
            lines.append(format_message_line(msg))
    except Exception:
        logger.exception("fetch_channel_gap failed")
        return []
    return lines


async def fetch_message_context(
    bot: discord.Client,
    channel_id: str,
    message_id: str,
    window: int = 5,
) -> str:
    window = max(1, min(10, int(window)))
    try:
        cid = int(channel_id)
        mid = int(message_id)
    except ValueError:
        return "IDs inválidos."
    channel = bot.get_channel(cid)
    if channel is None:
        try:
            channel = await bot.fetch_channel(cid)
        except Exception:
            return f"Canal {channel_id} não encontrado."
    if not hasattr(channel, "fetch_message") or not hasattr(channel, "history"):
        return "Canal não suporta histórico."
    try:
        target = await channel.fetch_message(mid)
    except discord.NotFound:
        return f"Mensagem {message_id} não encontrada."
    except discord.Forbidden:
        return "Sem permissão para ler esta mensagem."
    except Exception as e:
        return f"Erro ao buscar mensagem: {e}"
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
    def fmt(m: discord.Message, highlight: bool = False) -> str:
        prefix = "▶ " if highlight else "  "
        return prefix + format_message_line(m, is_target=highlight)
    lines: list[str] = []
    for m in before:
        lines.append(fmt(m))
    lines.append(fmt(target, True))
    for m in after:
        lines.append(fmt(m))
    header = (
        f"Contexto ao redor de {message_id} em #{getattr(channel, 'name', channel_id)} "
        f"(channel_id={channel.id}, ±{window}):\n"
    )
    return header + "\n".join(lines)


def _usage_summary(response: Any) -> str:
    """Format token usage info from a chat completion for log lines."""
    usage = getattr(response, 'usage', None)
    if usage is None:
        return 'n/a'
    parts = [f'prompt={usage.prompt_tokens}', f'completion={usage.completion_tokens}']
    details = getattr(usage, 'completion_tokens_details', None)
    reasoning = getattr(details, 'reasoning_tokens', None) if details is not None else None
    if reasoning is not None:
        parts.append(f'reasoning={reasoning}')
    return ', '.join(parts)


async def run_tool_loop(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    exec_tool: Callable[[str, dict[str, Any]], Awaitable[tuple[str, list[dict]]]],
    status_label: Callable[[str, dict[str, Any]], str] | None = None,
    interaction: discord.Interaction | None = None,
    max_rounds: int = 6,
    dedup_key: Callable[[dict], str] | None = None,
) -> tuple[str, list[dict]]:
    """Run an agentic tool-calling loop until the LLM produces a final answer.

    Parameters
    ----------
    client:
        The OpenAI-compatible async client.
    model:
        The model ID to use for all chat completions.
    messages:
        The message list (system prompt, history, current question).  Mutated
        in place with tool calls and results.
    tools:
        The tool definitions to offer, or ``None`` to skip the loop entirely.
    exec_tool:
        Async callable ``(name, args) -> (result_text, sources)``.
    status_label:
        Optional callable ``(name, args) -> status_text`` shown on the
        deferred interaction while a tool runs.  When ``None``, the status is
        not updated.
    interaction:
        Optional Discord interaction to update with live status messages.
    max_rounds:
        Maximum iterations of the tool-calling loop.
    dedup_key:
        Optional callable ``(source_dict) -> str`` that returns a unique
        identifier for a source.  When provided, duplicate sources across
        rounds are filtered out.  When ``None``, no deduplication is applied.

    Returns
    -------
    ``(answer_text, all_sources)`` where *all_sources* is the deduplicated list
    of source dicts aggregated across all tool rounds.
    """
    all_sources: list[dict] = []
    seen_keys: set[str] = set()
    rounds_used = 0
    finish_reasons: list[str] = []

    for round_num in range(1, max_rounds + 1):
        rounds_used = round_num
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            tools=tools,
        )

        choice = response.choices[0]
        finish_reason = getattr(choice, 'finish_reason', None) or 'unknown'
        finish_reasons.append(finish_reason)
        logger.debug(
            "Tool loop round %d/%d: model=%s finish_reason=%s tool_calls=%d content_chars=%d usage=[%s]",
            round_num,
            max_rounds,
            model,
            finish_reason,
            len(choice.message.tool_calls or []),
            len(choice.message.content or ''),
            _usage_summary(response),
        )

        if not choice.message.tool_calls:
            break

        messages.append(choice.message)

        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Malformed tool call arguments (model=%s finish_reason=%s): %r",
                    model,
                    finish_reason,
                    (tc.function.arguments or '')[:200],
                )
                args = {}

            if interaction is not None and status_label is not None:
                try:
                    await interaction.edit_original_response(
                        content=status_label(tc.function.name, args)
                    )
                except discord.HTTPException:
                    pass

            result_text, sources = await exec_tool(tc.function.name, args)

            if sources:
                if dedup_key is not None:
                    new_sources = [
                        s for s in sources if dedup_key(s) not in seen_keys
                    ]
                    if len(new_sources) < len(sources):
                        logger.info(
                            "Filtered %d duplicate source(s) from %s tool result",
                            len(sources) - len(new_sources),
                            tc.function.name,
                        )
                    for s in sources:
                        k = dedup_key(s)
                        if k:
                            seen_keys.add(k)
                    all_sources.extend(new_sources)

                    if not new_sources and sources:
                        result_text = (
                            'Os resultados desta busca já foram retornados '
                            'em uma rodada anterior. Use as informações já '
                            'fornecidas para formular sua resposta.'
                        )
                else:
                    all_sources.extend(sources)

            messages.append({
                'role': 'tool',
                'tool_call_id': tc.id,
                'content': truncate_safe(result_text, limit=6000),
            })
    else:
        logger.info(
            "Tool loop exhausted max_rounds=%d without a final answer (finish_reasons=%s); "
            "forcing one completion without tools",
            max_rounds,
            finish_reasons,
        )
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
        )

    final_choice = response.choices[0]
    answer = final_choice.message.content or ''
    if not answer.strip():
        logger.warning(
            "LLM returned an empty answer (finish_reason=%s usage=[%s]); "
            "retrying once without tools and with a conciseness nudge",
            getattr(final_choice, 'finish_reason', None),
            _usage_summary(response),
        )
        retry_messages = [*messages, {
            'role': 'user',
            'content': (
                'Responda agora, em português, de forma direta e concisa, sem '
                'usar nenhuma ferramenta. Use apenas as informações já coletadas.'
            ),
        }]
        response = await client.chat.completions.create(
            model=model,
            messages=retry_messages,
            max_tokens=LLM_MAX_TOKENS,
        )
        final_choice = response.choices[0]
        answer = final_choice.message.content or ''
    if not answer.strip():
        logger.warning(
            "LLM returned an empty answer (user got the fallback message): "
            "model=%s rounds=%d final_finish_reason=%s finish_reasons=%s usage=[%s]",
            model,
            rounds_used,
            getattr(final_choice, 'finish_reason', None),
            finish_reasons,
            _usage_summary(response),
        )
        answer = 'Não foi possível gerar uma resposta.'
    return answer, all_sources


def truncate_safe(text: str, limit: int = 3800, suffix: str = '\n\n...') -> str:
    """Truncate *text* at the last newline before *limit* to avoid breaking markdown."""
    if len(text) <= limit:
        return text
    idx = text.rfind('\n', 0, limit)
    if idx == -1:
        idx = limit
    return text[:idx] + suffix


def split_response(text: str, chunk_size: int = 3500) -> list[str]:
    """Split *text* into chunks of at most *chunk_size* chars, breaking at newlines."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break
        idx = text.rfind('\n', 0, chunk_size)
        if idx == -1:
            idx = chunk_size
        chunks.append(text[:idx])
        text = text[idx:].lstrip('\n')
    return chunks


def build_source_pages(
    source_lines: list[str],
    title: str,
    color: discord.Color,
    footer_base: str = '',
    page_size: int = 4000,
) -> list[discord.Embed]:
    """Paginate *source_lines* into embeds using the description field (4096 limit).

    Each embed's description stays under *page_size* chars.  Sources are
    always placed on their own page(s) so they never hit the 1024-char
    field-value limit.
    """
    if not source_lines:
        return []

    pages: list[discord.Embed] = []
    current_lines: list[str] = []
    current_len = 0

    for line in source_lines:
        added_len = len(line) + (1 if current_lines else 0)
        if current_lines and current_len + added_len > page_size:
            e = discord.Embed(
                title=title,
                description='\n'.join(current_lines),
                color=color,
            )
            pages.append(e)
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += added_len

    if current_lines:
        e = discord.Embed(
            title=title,
            description='\n'.join(current_lines),
            color=color,
        )
        pages.append(e)

    for i, e in enumerate(pages):
        src_idx = i + 1
        footer = f"Fontes {src_idx}/{len(pages)}"
        if footer_base:
            footer = f"{footer} • {footer_base}"
        e.set_footer(text=footer)

    return pages


class PaginatedEmbedView(discord.ui.View):
    """Navigation view for multi-page embed responses."""

    def __init__(self, embeds: list[discord.Embed]):
        super().__init__(timeout=120)
        self.embeds = embeds
        self.index = 0
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        self.prev_btn.disabled = self.index == 0
        self.next_btn.disabled = self.index >= len(self.embeds) - 1
        self.counter_btn.label = f'{self.index + 1}/{len(self.embeds)}'

    @discord.ui.button(label='◄', style=discord.ButtonStyle.secondary)
    async def prev_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index -= 1
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.embeds[self.index], view=self)

    @discord.ui.button(label='1/1', style=discord.ButtonStyle.secondary, disabled=True)
    async def counter_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()

    @discord.ui.button(label='►', style=discord.ButtonStyle.secondary)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index += 1
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.embeds[self.index], view=self)
