"""Shared text utilities used across multiple cogs."""

import datetime
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Final
from zoneinfo import ZoneInfo

import discord
from openai import AsyncOpenAI

from config import (
    CHANNEL_CONTEXT_MESSAGES,
    CONVERSATIONS_GAP_MESSAGES,
    HISTORY_SQL_TOOL_ENABLED,
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
            'mensagens relevantes com autor, canal, horário e link, incluindo 5 mensagens de contexto local. '
            'Cada resultado traz channel_id e msg_id — use com get_message_context para expandir o contexto. '
            'Mensagens do mesmo trecho de conversa são agrupadas em um único resultado. '
            'Para filtrar por autor conhecendo apenas o nome, resolva o author_id com find_user primeiro.'
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
            'Aceita o msg_id ou reply_to visto em qualquer linha de mensagem, desde que combinados '
            'com o channel_id exibido nos cabeçalhos de resultado/blocos ou com o canal atual. '
            'Resultados próximos no tempo são agrupados pelo search_history — use esta ferramenta '
            'para ver as mensagens vizinhas de um cluster.'
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

FIND_USER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'find_user',
        'description': (
            'Resolve um usuário do servidor pelo nome (atual, antigo ou apelido), @handle ou ID. '
            'Retorna a identidade canônica (author_id), apelidos conhecidos, total de mensagens, '
            'período de atividade e canais mais ativos. '
            'Use ANTES de search_history ou get_user_stats quando souber apenas o '
            'nome da pessoa — passe o author_id retornado como filtro dessas ferramentas. '
            'Sem query, lista os usuários mais ativos do servidor.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Nome (parcial ok, sem diferenciação de maiúsculas), @handle ou ID do Discord. Omita para listar os mais ativos.',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Máximo de usuários (padrão 5, máximo 12).',
                },
            },
            'required': [],
        },
    },
}

SQL_HISTORY_TOOL = {
    'type': 'function',
    'function': {
        'name': 'sql_history',
        'description': (
            'Executa UM SELECT SQL somente-leitura no banco do histórico do servidor '
            'para perguntas analíticas que o search_history não cobre: contagens exatas, '
            'agrupamentos, cruzamentos, regex, menções e respostas. '
            'OBRIGATÓRIO: a consulta deve citar o guild_id do servidor atual e ler apenas '
            'as tabelas abaixo (leituras da coluna chunks.embedding e qualquer escrita são bloqueadas). '
            'Sem busca semântica aqui — para "sobre o que falamos" use search_history. '
            'Esquema: chunks(msg_id, guild_id, channel_id, channel_name, author_id, author_name, '
            'author_full, content, chunk_text, window_line, window_lines, reply_to, ts ISO-8601 UTC, '
            'jump_url, embedding BLOB proibido) — 1 linha por mensagem indexada (bots excluídos); '
            'authors(guild_id, author_id, display_name, handle, full_label, aliases JSON, first_seen, '
            'last_seen, msg_count); chunks_fts(msg_id, guild_id, chunk_text, content) — índice FTS5. '
            'Exemplos: top falantes de um tópico: '
            "\"SELECT c.author_full, COUNT(*) n FROM chunks c JOIN chunks_fts f ON f.msg_id=c.msg_id "
            "WHERE f.guild_id=123 AND chunks_fts MATCH '\"lag\" AND server' GROUP BY c.author_full ORDER BY n DESC\"; "
            'menções de usuário: c.content LIKE \'%<@ID>%\' OR c.content LIKE \'%<@!ID>%\'; '
            'respostas a uma mensagem: c.reply_to=\'MSG_ID\'; respostas em geral: c.reply_to IS NOT NULL; '
            'por mês: GROUP BY substr(c.ts,1,7); por hora: GROUP BY substr(c.ts,12,2); '
            "FTS5: MATCH '\"frase exata\" AND (a OR b) NOT c', prefixo 'palavra*'; "
            'REGEXP(padrão, texto) e REGEXP sempre com re.IGNORECASE; '
            'data: compare c.ts como texto ISO ou use datetime(c.ts). Sempre termine com LIMIT.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'sql': {
                    'type': 'string',
                    'description': 'Um único SELECT (ou WITH ... SELECT) SQLite, já com o filtro guild_id do servidor atual.',
                },
            },
            'required': ['sql'],
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

MESSAGE_LINE_MAX_CONTENT: Final[int] = 800


def _flatten(s: str) -> str:
    return ' '.join(s.split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + '…'


def message_content_text(msg: discord.Message, *, max_length: int = MESSAGE_LINE_MAX_CONTENT) -> str:
    """Flattened message content with attachment/embed markers, never empty."""
    c = msg.content or ""
    if msg.attachments:
        c += " " + " ".join(f"[anexo:{a.filename}]" for a in msg.attachments)
    if not c.strip() and msg.embeds:
        try:
            c = f"[embed: {msg.embeds[0].title or msg.embeds[0].description[:100]}]"
        except Exception:
            c = "[embed]"
    c = _flatten(c)
    if not c:
        c = "[sem texto]"
    return _clip(c, max_length)


def format_message_line(msg: discord.Message) -> str:
    """Render a Discord message in the canonical line format."""
    display = _flatten(getattr(msg.author, 'display_name', None) or str(msg.author))
    handle = _flatten(getattr(msg.author, 'name', None) or '')
    author = f"{display} (@{handle})" if handle else display
    author_id = getattr(msg.author, 'id', '')
    line = f"[{_fmt_dt_line(msg.created_at)}] {author} <author_id={author_id}>: {message_content_text(msg)}"
    ref = getattr(msg, 'reference', None)
    ref_id = getattr(ref, 'message_id', None)
    if ref_id:
        target = ""
        resolved = getattr(ref, 'resolved', None)
        if resolved is not None and not isinstance(resolved, discord.DeletedReferencedMessage):
            rdisplay = _flatten(getattr(resolved.author, 'display_name', '') or '')
            if rdisplay:
                target = f" ({rdisplay})"
        line += f" ↩ reply_to={ref_id}{target}"
    line += f" [msg_id={msg.id}]"
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
    author = _flatten(str(ch.get('author_full') or ch.get('author_name') or '?'))
    content = _clip(_flatten(str(ch.get('content', ''))), MESSAGE_LINE_MAX_CONTENT)
    if not content:
        content = '[sem texto]'
    line = f"[{ts}] {author} <author_id={ch.get('author_id', '?')}>: {content}"
    if ch.get('reply_to'):
        line += f" ↩ reply_to={ch['reply_to']}"
    return f"{line} [msg_id={ch.get('msg_id', '')}]"


def render_search_results(results: list[dict], *, window_chars: int = 1200, include_msg_id: bool = True) -> str:
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
        raw_lines = str(r.get('chunk_text', r.get('content', ''))).replace('```', 'ˋˋˋ').split('\n')
        kept: list[str] = []
        used = 0
        dropped = 0
        for line in raw_lines:
            cost = len(line) + (1 if kept else 0)
            if used + cost > window_chars:
                dropped = len(raw_lines) - len(kept)
                break
            kept.append(line)
            used += cost
        if dropped:
            kept.append(f"… (+{dropped} linhas)")
        body = f"{header}\n```\n" + '\n'.join(kept) + "\n```"
        if include_msg_id:
            body += f"\n`msg_id={r.get('msg_id')}`"
        parts.append(body)
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
    are already replayed as conversation turns). The walk goes backwards from
    ``before`` and stops at the anchor id, so nothing older than the previous
    turn leaks in; only human messages consume the budget, so truncation keeps
    the NEWEST chatter lines. Controlled by ``CONVERSATIONS_GAP_MESSAGES``
    (0 disables). Returns [] on failure — the recent-channel window is then
    the fallback.
    """
    n = CONVERSATIONS_GAP_MESSAGES if limit is None else limit
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
        async for msg in channel.history(limit=max(n * 3, 100), before=before):
            if msg.id <= anchor.id:
                break
            if msg.author.bot or str(msg.id) in skip:
                continue
            if not msg.content and not msg.attachments and not msg.embeds:
                continue
            lines.append(format_message_line(msg))
            if len(lines) >= n:
                break
    except Exception:
        logger.exception("fetch_channel_gap failed")
        return []
    lines.reverse()
    return lines


async def fetch_turn_gap(message: discord.Message, history: list[dict]) -> list[str]:
    """Channel chatter since the previous bot-directed turn (same channel only)."""
    last_turn = history[-1] if history else None
    anchor_id = (last_turn or {}).get('message_id')
    if not anchor_id:
        return []
    if last_turn.get('channel_id') and str(last_turn['channel_id']) != str(message.channel.id):
        return []
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
    def fmt(m: discord.Message, highlight: bool = False) -> str:
        prefix = "▶ " if highlight else "  "
        return prefix + format_message_line(m)
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


async def exec_history_tool(
    name: str,
    args: dict,
    *,
    bot: discord.Client,
    guild: discord.Guild | None,
    channel: discord.abc.Messageable | None,
) -> tuple[str, list[dict]] | None:
    """Run one of the shared history/context tools.

    Returns ``None`` for any other tool name so callers fall through to their
    own handlers. Guild/channel resolution stays at the caller.
    """
    if name == 'get_channel_history':
        text = await fetch_channel_history(bot, channel, limit=args.get('limit', 20), channel_id=args.get('channel_id'))
        return text, []
    if name == 'get_guild_info':
        if not guild:
            return "Fora de um servidor (DM).", []
        text = build_guild_context(guild)
        try:
            chs = [f"#{c.name} ({c.id})" for c in guild.channels if isinstance(c, discord.TextChannel)][:30]
            if chs:
                text += "\nCanais de texto: " + ", ".join(chs)
        except Exception:
            pass
        return text, []
    if name == 'search_history':
        hist = bot.get_cog('HistoryRAG')
        if not hist:
            return "Histórico não disponível.", []
        if not guild:
            return "Busca no histórico requer estar em um servidor.", []
        query = args.get('query', '')
        limit = max(1, min(12, int(args.get('limit', 5))))
        try:
            results = await hist.search(query, guild.id, limit=limit, channel_id=args.get('channel_id'), author_id=args.get('author_id'), author_name=args.get('author_name'), after=args.get('after'), before=args.get('before'), search_mode=args.get('search_mode','hybrid'), sort_by=args.get('sort_by','relevance'))  # type: ignore
        except Exception:
            logger.exception("search_history failed")
            return "Erro ao buscar no histórico.", []
        if not results:
            return "Nenhuma mensagem relevante encontrada no histórico.", []
        return render_search_results(results), []
    if name == 'get_user_stats':
        hist = bot.get_cog('HistoryRAG')
        if not hist:
            return "Histórico não disponível.", []
        if not guild:
            return "Requer servidor.", []
        try:
            stats = await hist.get_user_stats(guild.id, author_id=args.get('author_id'), author_name=args.get('author_name'))  # type: ignore
        except Exception:
            logger.exception("get_user_stats failed")
            return "Erro ao buscar estatísticas.", []
        if "error" in stats:
            return stats["error"], []
        lines = [f"Usuário: {stats['author_full']} ({', '.join(stats['author_ids'])})", f"Total: {stats['total_messages']} msgs | Média: {stats['avg_length']} chars", f"Canais: {', '.join(f'{k}={v}' for k,v in stats['top_channels'])}", f"Horários: {', '.join(f'{h}h={v}' for h,v in stats['top_hours'])}", f"Período: {stats['first_seen']} → {stats['last_seen']}", f"Exemplo: {stats['example_content']} {stats['example_jump']}"]
        return "\n".join(lines), []
    if name == 'count_mentions':
        hist = bot.get_cog('HistoryRAG')
        if not hist:
            return "Histórico não disponível.", []
        if not guild:
            return "Requer servidor.", []
        try:
            groups = await hist.count_mentions(guild.id, query=args.get('query',''), group_by=args.get('group_by','author'), limit=int(args.get('limit',10)), after=args.get('after'), before=args.get('before'))  # type: ignore
        except Exception:
            logger.exception("count_mentions failed")
            return "Erro ao contar menções.", []
        if not groups:
            return "Nenhuma menção encontrada.", []
        lines = [f"{gr['key']}: {gr['count']}× — ex: {gr['example'].get('content','')[:120]}" for gr in groups]
        return "\n".join(lines), []
    if name == 'find_user':
        hist = bot.get_cog('HistoryRAG')
        if not hist:
            return "Histórico não disponível.", []
        if not guild:
            return "Busca de usuários requer estar em um servidor.", []
        query = args.get('query', '')
        try:
            limit = max(1, min(12, int(args.get('limit', 5))))
        except (TypeError, ValueError):
            limit = 5
        try:
            users = await hist.find_users(guild.id, query, limit=limit)  # type: ignore
        except Exception:
            logger.exception("find_user failed")
            return "Erro ao buscar usuários.", []
        if not users:
            return (
                f"Nenhum usuário encontrado para: {query or '(vazio)'}. "
                "Tente search_history com author_name se a pessoa já falou no servidor."
            ), []
        lines = []
        for u in users:
            label = u.get('full_label') or u.get('display_name') or u.get('author_id', '?')
            lines.append(
                f"**{label}** <author_id={u.get('author_id','?')}> — "
                f"{u.get('msg_count', 0)} msgs, {str(u.get('first_seen',''))[:10]} → {str(u.get('last_seen',''))[:10]}"
            )
            aliases = [a for a in u.get('aliases', []) if a and a != u.get('display_name')]
            if aliases:
                lines.append(f"  Também conhecido como: {', '.join(aliases[:5])}")
            if u.get('top_channels'):
                chans = ', '.join(f"#{c} ({n})" for c, n in u['top_channels'][:3])
                lines.append(f"  Canais mais ativos: {chans}")
        return "\n".join(lines), []
    if name == 'sql_history':
        if not HISTORY_SQL_TOOL_ENABLED:
            return 'Ferramenta sql_history desativada por configuração (HISTORY_SQL_TOOL_ENABLED).', []
        hist = bot.get_cog('HistoryRAG')
        if not hist:
            return 'Histórico não disponível.', []
        if not guild:
            return 'Consulta SQL requer estar em um servidor.', []
        sql = args.get('sql', '')
        try:
            text = await hist.exec_sql(guild.id, sql)  # type: ignore
        except ValueError as e:
            return f'Consulta rejeitada: {e} Reescreva e tente de novo.', []
        except Exception as e:
            logger.exception('sql_history failed')
            return f'Erro SQL: {e} Corrija a consulta (veja esquema na descrição da ferramenta).', []
        return text, []
    if name == 'get_message_context':
        try:
            window = max(1, min(10, int(args.get('window', 5))))
        except (TypeError, ValueError):
            window = 5
        text = await fetch_message_context(
            bot,
            channel_id=args.get('channel_id',''),
            message_id=args.get('message_id',''),
            window=window,
            guild_id=guild.id if guild else None,
        )
        if text is None:
            hist = bot.get_cog('HistoryRAG')
            if hist and guild:
                try:
                    text = await hist.get_message_context_from_index(  # type: ignore
                        guild.id, args.get('channel_id',''), args.get('message_id',''), window,
                    )
                except Exception:
                    logger.exception("indexed get_message_context fallback failed")
        if text is None:
            text = f"Mensagem {args.get('message_id','')} não encontrada (nem via API, nem no índice)."
        return text, []
    return None


def history_tool_status(name: str, args: dict) -> str | None:
    """Status label for the shared history/context tools, ``None`` otherwise."""
    if name == 'get_channel_history':
        lim = args.get('limit', 20)
        cid = args.get('channel_id')
        return f'📜 Lendo histórico ({lim} msgs)' + (f' canal {cid}' if cid else '')
    if name == 'get_guild_info':
        return '🏰 Coletando informações do servidor'
    if name == 'search_history':
        q = args.get('query','')[:40]
        return f'🔎 Buscando no histórico: *{q}*'
    if name == 'get_user_stats':
        return f'📊 Estatísticas de {args.get("author_id") or args.get("author_name","usuário")}…'
    if name == 'count_mentions':
        return f'🔢 Contando menções: *{args.get("query","")[:30]}*'
    if name == 'get_message_context':
        return f'🧩 Contexto da mensagem {args.get("message_id","")}…'
    if name == 'find_user':
        return f'👤 Localizando usuário: *{str(args.get("query") or "")[:40] or "mais ativos"}*'
    if name == 'sql_history':
        snippet = ' '.join(str(args.get('sql') or '').split())[:60]
        return f'🗄️ Consultando histórico (SQL): `{snippet}`'
    return None


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


def _get(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def serialize_trajectory(messages: list[Any]) -> list[dict]:
    """Convert appended loop messages (SDK objects or dicts) to JSON-safe dicts.

    Only a whitelist of fields is kept (role/content/tool_calls/tool_call_id):
    provider extras such as ``reasoning``/thinking payloads are dropped because
    most APIs reject them on input, and the result is what gets persisted in the
    conversation store and replayed verbatim on follow-ups.
    """
    out: list[dict] = []
    for msg in messages:
        role = _get(msg, 'role')
        if role == 'assistant':
            content = _get(msg, 'content')
            entry: dict = {
                'role': 'assistant',
                'content': content if isinstance(content, str) else None,
            }
            calls: list[dict] = []
            for tc in _get(msg, 'tool_calls') or []:
                fn = _get(tc, 'function') or {}
                name = _get(fn, 'name')
                tc_id = _get(tc, 'id')
                if not name or not tc_id:
                    continue
                calls.append({
                    'id': tc_id,
                    'type': 'function',
                    'function': {
                        'name': name,
                        'arguments': _get(fn, 'arguments') or '{}',
                    },
                })
            if calls:
                entry['tool_calls'] = calls
            out.append(entry)
        elif role == 'tool':
            tc_id = _get(msg, 'tool_call_id')
            content = _get(msg, 'content')
            if not tc_id or not isinstance(content, str):
                continue
            out.append({'role': 'tool', 'tool_call_id': tc_id, 'content': content})
        elif role == 'user':
            # A leading user entry can only be the empty-answer retry nudge
            # (emitted when the model returned no tool calls AND no content in
            # round 1).  Keeping it would replay as two consecutive user
            # messages — providers enforcing strict user/assistant alternation
            # (Anthropic) reject that — so it is dropped from the capture.
            if not out:
                continue
            content = _get(msg, 'content')
            out.append({'role': 'user', 'content': content if isinstance(content, str) else ''})
    return out


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
) -> tuple[str, list[dict], list[dict]]:
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
    ``(answer_text, all_sources, trajectory)`` where *all_sources* is the
    deduplicated list of source dicts aggregated across all tool rounds and
    *trajectory* is the JSON-safe list of messages appended during the loop
    (assistant tool calls, tool results, the final answer — plus the
    empty-answer retry nudge when it fired).  It is persisted with the
    conversation turn and replayed verbatim on follow-ups, so later turns see
    the full internal steps of earlier ones and the replayed prefix stays
    byte-identical to what was actually sent (keeping provider prefix caches
    effective).
    """
    all_sources: list[dict] = []
    seen_keys: set[str] = set()
    rounds_used = 0
    finish_reasons: list[str] = []
    capture_start = len(messages)

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
        # The nudge is appended to `messages` itself (payload-identical to the
        # previous copy) so the captured trajectory includes exactly what was
        # sent and the follow-up prefix keeps matching the provider cache.
        messages.append({
            'role': 'user',
            'content': (
                'Responda agora, em português, de forma direta e concisa, sem '
                'usar nenhuma ferramenta. Use apenas as informações já coletadas.'
            ),
        })
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
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
    messages.append({'role': 'assistant', 'content': answer})
    trajectory = serialize_trajectory(messages[capture_start:])
    return answer, all_sources, trajectory


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
