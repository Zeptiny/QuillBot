"""The one turn runner every entry point uses.

``/chat``, ``/ask``, @mentions, replies, threads, name calls, monitors,
scheduled tasks and Spark analyses used to be two copies of the same code
(``Commands._run_chat`` and ``DocsRAG._run_agent``). :func:`run_turn` builds
the request once, from a :class:`~core.turn.TurnContext` and its profile:

    [system: profile prompt + <personalidade> + identity + conversation block]
    [Spark report prelude, when one is loaded]
    [replayed history]
    [current message: <contexto>, <canal>, <memory>, recent channel window, question]

Everything that changes per request rides on the final user message, so the
system prompt and the replayed history stay byte-identical across follow-ups
and providers can prefix-cache them.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from config import CHANNEL_CONTEXT_MESSAGES, CONVERSATIONS_HISTORY_TURNS
from core.context import build_full_context_block, fetch_recent_channel_context
from core.conversations import (
    apply_cache_control,
    author_info,
    author_stamp,
    build_conversation_block,
    build_current_message,
    build_history_messages,
    select_image_refs,
)
from core.discord_text import build_identity_block
from core.prompts import lag_spike_warning
from core.tool_loop import run_tool_loop
from core.turn import TurnContext, TurnResult

logger = logging.getLogger(__name__)


def _source_entry(source: dict) -> dict | None:
    url = source.get('url') or source.get('doc_url') or ''
    if not url:
        return None
    title = str(source.get('title') or url)[:100]
    entry = {'title': title, 'url': url}
    if source.get('source'):
        entry['source'] = str(source['source'])
    return entry


def normalize_sources(raw: list[dict]) -> list[dict]:
    """Unique ``{title, url[, source]}`` entries, in first-seen order."""
    seen: set[str] = set()
    out: list[dict] = []
    for s in raw or []:
        entry = _source_entry(s) if isinstance(s, dict) else None
        if entry is None or entry['url'] in seen:
            continue
        seen.add(entry['url'])
        out.append(entry)
    return out


def _system_prompt(ctx: TurnContext) -> str:
    system = ctx.profile.system_prompt
    persona = getattr(ctx.services, 'persona', None)
    if persona is not None and ctx.guild is not None:
        # Per server and rarely changed: stays inside the cached prefix.
        block = persona.block(ctx.guild.id)
        if block:
            system += '\n\n' + block
    report = ctx.spark_report
    if report is not None and getattr(report, 'tick_length_threshold_ms', 0) > 0:
        system += lag_spike_warning(report.tick_length_threshold_ms)
    identity = build_identity_block(getattr(ctx.bot, 'user', None), ctx.guild)
    if identity:
        system += '\n\n' + identity
    if ctx.history:
        # Built from the FULL history (not the replay window) so it only
        # changes when participants/channel actually change.
        system += '\n\n' + build_conversation_block(ctx.history)
    return system


# "pesquise", "procura aí", "busca isso", "dá um google": the person asked for a search.
_SEARCH_REQUEST_RE = re.compile(r'\b(pesquis\w*|procur[ae]\w*|busc[ae]\w*|busque\w*|googl\w*)\b', re.I)
SEARCH_HINT = (
    '[Nesta mensagem a pessoa pediu uma pesquisa: use web_search (ou search_docs, para Minecraft) '
    'antes de responder.]'
)


def search_requested(question: str) -> bool:
    """Whether the person's own words (not a quoted message or routing header) ask for a search."""
    own = (question or '').rsplit('\n---\n', 1)[-1]
    return bool(_SEARCH_REQUEST_RE.search(own))


def _channel_block(ctx: TurnContext) -> str | None:
    instructions = str(ctx.channel_settings.get('instructions') or '').strip()
    if not instructions:
        return None
    return f'<canal>\nInstruções deste canal, definidas pelo servidor: {instructions}\n</canal>'


async def _context_blocks(ctx: TurnContext) -> tuple[list[str], list[str]]:
    """Per-request blocks for the final user message, and channel images."""
    blocks: list[str] = []
    images = [u for u in ctx.context_image_urls if u]
    if ctx.user or ctx.guild or ctx.channel:
        try:
            block = build_full_context_block(ctx.user, ctx.guild, ctx.channel, ctx.created_at)
            if block:
                blocks.append(block)
        except Exception:
            logger.exception('Failed to build context block')
    channel_block = _channel_block(ctx)
    if channel_block:
        blocks.append(channel_block)
    memory = getattr(ctx.services, 'memory', None)
    if memory is not None and ctx.guild is not None:
        try:
            mem_block = await memory.build_memory_block(
                ctx.guild.id, ctx.question,
                speaker_id=str(ctx.user.id) if ctx.user is not None else None,
                participant_ids=ctx.participant_ids,
                for_label=author_stamp(
                    author_info(ctx.user), ctx.created_at.timestamp() if ctx.created_at else None,
                ) if ctx.user is not None else None,
            )
            if mem_block:
                blocks.append(mem_block)
        except Exception:
            logger.exception('Failed to build memory block')
    # The chatter captured since the previous turn replaces the recent window.
    has_gap = bool(ctx.prior_context) or any(t.get('prior_context') for t in ctx.history)
    if CHANNEL_CONTEXT_MESSAGES > 0 and not has_gap and ctx.channel is not None:
        try:
            recent = await fetch_recent_channel_context(
                ctx.bot, ctx.channel, before=ctx.window_before or ctx.message,
            )
            if recent:
                blocks.append(recent.text)
                images.extend(recent.images)
        except Exception:
            logger.exception('Failed to build recent channel context')
    return blocks, images


def _spark_prelude(ctx: TurnContext) -> list[dict]:
    """The report summary as a synthetic prior exchange (data before the question)."""
    from cogs.spark_parser import build_summary
    summary = '[Relatório Spark carregado]\n\n' + build_summary(ctx.spark_report)
    return [
        apply_cache_control({'role': 'user', 'content': summary}),
        {
            'role': 'assistant',
            'content': 'Entendido. Analisei o resumo do relatório Spark e estou pronto para diagnosticar.',
        },
    ]


async def run_turn(ctx: TurnContext) -> TurnResult:
    """Run one turn through the tool loop and return what it produced."""
    services = ctx.services
    profile = ctx.profile
    blocks, context_images = await _context_blocks(ctx)
    offered = set(profile.tools) & set(services.tools.names())
    if offered & {'web_search', 'search_docs'} and search_requested(ctx.question):
        blocks.append(SEARCH_HINT)

    messages: list[dict] = [apply_cache_control({'role': 'system', 'content': _system_prompt(ctx)})]
    if ctx.spark_report is not None:
        messages.extend(_spark_prelude(ctx))
    if ctx.history:
        messages.extend(build_history_messages(ctx.history, max_turns=CONVERSATIONS_HISTORY_TURNS))
        # Second breakpoint at the end of the replayed conversation.
        messages[-1] = apply_cache_control(messages[-1])

    image_urls = [u for u in ctx.image_urls if u]
    current = build_current_message(
        ctx.question,
        author=author_info(ctx.user),
        ts=ctx.created_at.timestamp() if ctx.created_at else None,
        image_urls=image_urls,
        reply_to=ctx.reply_to,
        prior_context=ctx.prior_context,
        channel_id=getattr(ctx.channel, 'id', None),
        context_blocks='\n\n'.join(blocks) or None,
        context_image_urls=context_images,
        previous_author=ctx.history[-1].get('author') if ctx.history else None,
    )
    messages.append(current)

    registry = services.tools
    tools = registry.select(profile.tools)
    by_name = {t.name: t for t in tools}

    async def exec_tool(name: str, args: dict[str, Any]) -> tuple[str, list[dict]]:
        ctx.used_tools.append(name)
        if name not in by_name:
            return f'Ferramenta desconhecida: {name}', []
        return await registry.execute(name, args, ctx)

    def status_label(name: str, args: dict[str, Any]) -> str:
        tool = by_name.get(name)
        return tool.status_text(args, ctx) if tool else f'🔧 Executando: {name}'

    answer, raw_sources, trajectory = await run_tool_loop(
        services.llm.client,
        profile.model,
        messages,
        [t.schema for t in tools] or None,
        exec_tool,
        status_label=status_label,
        on_status=ctx.on_status,
        max_rounds=profile.max_rounds,
        dedup_key=lambda s: s.get('url') or s.get('doc_url') or s.get('path') or '',
    )

    image_refs, context_refs, _ = select_image_refs(image_urls, context_images)
    structured = profile.structured or any(
        by_name[n].structured for n in ctx.used_tools if n in by_name
    )
    return TurnResult(
        answer=answer,
        sources=normalize_sources(raw_sources),
        user_message=current,
        trajectory=trajectory,
        image_urls=image_refs,
        context_image_urls=context_refs,
        thread_title=ctx.thread_title,
        used_tools=list(ctx.used_tools),
        structured=structured,
    )
