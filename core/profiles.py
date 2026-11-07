"""Profiles: the kinds of turn the one runner can play.

A profile picks the system prompt, the tools offered (by name, in a fixed
order so the prompt prefix stays cacheable), the model and how the answer
looks. ``/chat``, @mentions, replies, threads, name calls, monitors and
scheduled tasks all use ``chat``; ``/ask`` (and replies to it) use the
docs-first ``ask``; Spark analyses use ``spark``.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import CHAT_MODEL, DOCS_BASE_URL, SPARK_MODEL
from core.prompts import ASK_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, SPARK_SYSTEM_PROMPT_SUFFIX
from core.radar_tools import RADAR_TOOLS

CONTEXT_TOOLS = (
    'get_channel_history',
    'get_guild_info',
    'search_history',
    'get_message_context',
    'get_user_stats',
    'count_mentions',
    'find_user',
    'sql_history',
)
MEMORY_TOOLS = ('memory_search', 'memory_write', 'memory_about')
DOCS_TOOLS = ('search_docs', 'search_plugins')
WEB_TOOLS = ('web_search', 'web_extract')
SCHEDULER_TOOLS = ('schedule_create', 'schedule_list', 'schedule_delete')
MONITOR_TOOLS = ('monitor_create', 'monitor_list', 'monitor_update', 'monitor_delete')
SPARK_TOOLS = ('get_spark_detail', 'get_config_key')

TEAL = 0x1ABC9C
BLUE = 0x3498DB
ORANGE = 0xE67E22

DOCS_FOOTER = f'Documentação • {DOCS_BASE_URL} • 💬 Responda a esta mensagem para continuar a conversa'


@dataclass(frozen=True)
class Profile:
    name: str
    system_prompt: str
    tools: tuple[str, ...]
    model: str
    max_rounds: int = 6
    # Answers always go out as embeds (Spark reports).
    structured: bool = False
    color: int = TEAL
    # Embed title is the question itself (docs Q&A), and the footer under it.
    question_title: bool = False
    footer: str = ''
    sources_title: str = '🌐 Fontes'


CHAT = Profile(
    name='chat',
    system_prompt=CHAT_SYSTEM_PROMPT,
    tools=(
        *WEB_TOOLS,
        *RADAR_TOOLS,
        *DOCS_TOOLS,
        *CONTEXT_TOOLS,
        *MEMORY_TOOLS,
        *SCHEDULER_TOOLS,
        *MONITOR_TOOLS,
        'summarize_channel',
        'server_wrapped',
        'persona_update',
        'add_reaction',
        'open_thread',
    ),
    model=CHAT_MODEL,
)

ASK = Profile(
    name='ask',
    system_prompt=ASK_SYSTEM_PROMPT,
    tools=(*DOCS_TOOLS, *MEMORY_TOOLS, *CONTEXT_TOOLS, 'add_reaction'),
    model=CHAT_MODEL,
    color=BLUE,
    question_title=True,
    footer=DOCS_FOOTER,
    sources_title='📄 Fontes da Documentação',
)

SPARK = Profile(
    name='spark',
    system_prompt=ASK_SYSTEM_PROMPT + SPARK_SYSTEM_PROMPT_SUFFIX,
    tools=(*ASK.tools, *SPARK_TOOLS),
    model=SPARK_MODEL,
    structured=True,
    color=ORANGE,
    footer=DOCS_FOOTER,
    sources_title='📄 Fontes da Documentação',
)

PROFILES: dict[str, Profile] = {p.name: p for p in (CHAT, ASK, SPARK)}


def profile_for(name: str | None, *, has_spark_report: bool = False) -> Profile:
    """The profile a stored conversation continues with.

    A Spark conversation whose report is gone (restart) continues as ``ask``.
    """
    if name == 'spark' and not has_spark_report:
        return ASK
    return PROFILES.get(name or 'chat', CHAT)
