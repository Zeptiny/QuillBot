"""Shared text utilities used across multiple cogs."""

import datetime
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from zoneinfo import ZoneInfo

import discord
from openai import AsyncOpenAI

from config import LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

BR_TZ = ZoneInfo("America/Sao_Paulo")

CHANNEL_HISTORY_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_channel_history',
        'description': (
            'Busca mensagens recentes do canal atual ou de outro canal do servidor. '
            'Use quando o usuário perguntar sobre conversas anteriores, contexto recente, '
            'ou quando precisar entender o que foi discutido antes. Retorna autor, horário e conteúdo.'
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

AGGREGATE_USER_TOPICS_TOOL = {
    'type': 'function',
    'function': {
        'name': 'aggregate_user_topics',
        'description': (
            'Agrupa e conta os principais tópicos/assuntos de um usuário. Retorna lista de tópicos com contagem e exemplo. '
            'Use para "sobre o que X fala mais?" ou "quais os assuntos favoritos de X?".'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'author_id': {'type': 'string', 'description': 'ID do usuário.'},
                'author_name': {'type': 'string', 'description': 'Nome parcial se ID desconhecido.'},
                'top_k': {'type': 'integer', 'description': 'Número de tópicos (padrão 5, máx 10).'},
            },
            'required': [],
        },
    },
}

GET_USER_TIMELINE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_user_timeline',
        'description': (
            'Retorna timeline cronológica de mensagens de um usuário, opcionalmente filtrada por tópico/query. '
            'Use para "quando X mencionou Y?" ou "mostre histórico de X sobre Y".'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'author_id': {'type': 'string', 'description': 'ID do usuário.'},
                'author_name': {'type': 'string', 'description': 'Nome parcial.'},
                'query': {'type': 'string', 'description': 'Opcional: filtrar timeline por tópico/query semântica.'},
                'limit': {'type': 'integer', 'description': 'Número de resultados (padrão 10, máx 20).'},
                'after': {'type': 'string', 'description': 'Data inicial ISO.'},
                'before': {'type': 'string', 'description': 'Data final ISO.'},
                'channel_id': {'type': 'string', 'description': 'Opcional: restringir a canal.'},
                'sort_by': {'type': 'string', 'enum': ['recent', 'oldest'], 'description': 'Ordenação (padrão recent).'},
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

GET_TEMPORAL_HEATMAP_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_temporal_heatmap',
        'description': (
            'Gera heatmap temporal de um tópico: contagem de mensagens por dia/semana. '
            'Use para "quando falamos sobre X?" ou evolução temporal de um assunto.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Tópico/query.'},
                'bucket': {'type': 'string', 'enum': ['day', 'week'], 'description': 'Granularidade (padrão day).'},
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
            'Use após search_history para expandir o contexto de um resultado relevante.'
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
        async for msg in target.history(limit=limit):
            ts = _fmt_dt(msg.created_at)
            author = getattr(msg.author, 'display_name', str(msg.author))
            content = msg.content or ""
            if msg.attachments:
                content += " " + " ".join(f"[anexo:{a.filename}]" for a in msg.attachments)
            if msg.embeds and not content:
                content = f"[embed: {msg.embeds[0].title or 'sem título'}]"
            content = content.replace("\n", " ").strip()
            if len(content) > 350:
                content = content[:350] + "…"
            if not content:
                content = "[sem texto]"
            lines.append(f"[{ts}] {author}: {content}")
    except discord.Forbidden:
        return "Sem permissão para ler histórico deste canal."
    except Exception as e:
        logger.exception("fetch_channel_history failed")
        return f"Erro ao buscar histórico: {e}"
    if not lines:
        return "Nenhuma mensagem encontrada no histórico."
    lines.reverse()
    header = f"Histórico de #{getattr(target, 'name', target.id)} (últimas {len(lines)} mensagens, cronológica):\n"
    return header + "\n".join(lines)


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
        ts = _fmt_dt(m.created_at)
        author = getattr(m.author, 'display_name', str(m.author))
        content = (m.content or "").replace("\n", " ").strip()
        if m.attachments:
            content += " " + " ".join(f"[anexo:{a.filename}]" for a in m.attachments)
        if not content:
            content = "[sem texto]"
        if len(content) > 350:
            content = content[:350] + "…"
        prefix = "▶ " if highlight else "  "
        return f"{prefix}[{ts}] {author}: {content} (id={m.id})"
    lines: list[str] = []
    for m in before:
        lines.append(fmt(m))
    lines.append(fmt(target, True) + "  ← alvo")
    for m in after:
        lines.append(fmt(m))
    header = f"Contexto ao redor de {message_id} em #{getattr(channel, 'name', channel_id)} (±{window}):\n"
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
