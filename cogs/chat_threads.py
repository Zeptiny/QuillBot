"""Chat threads — keep long or off-topic bot conversations out of busy channels.

Every chat answer is a reply in the channel, so a long back-and-forth, or a
question unrelated to what the channel is discussing, buries everyone else's
conversation. A conversation moves into a thread when

- it reaches ``CHAT_THREAD_AFTER_TURNS`` turns,
- it happens in one of ``CHAT_THREAD_CHANNELS``, or
- the model calls ``open_thread`` (off-topic question, casual chat).

The thread is opened on the message that asked, so the channel keeps only that
message with the thread under it. A conversation that moved keeps answering in
its thread, even when someone continues it from the channel. Inside a thread
the bot opened, every message reaches the bot without a mention (see
``Commands.on_message``); those carry ``UNPROMPTED_HEADER`` and the model may
stay silent with ``[NO_REPLY]`` when people there talk among themselves.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import discord

from config import (
    CHAT_THREAD_AFTER_TURNS,
    CHAT_THREAD_ARCHIVE_MINUTES,
    CHAT_THREAD_CHANNELS,
    CHAT_THREADS_ENABLED,
)

logger = logging.getLogger(__name__)

UNPROMPTED_HEADER = '[Mensagem na sua thread, sem menção a você]'

# Discord only accepts these; anything else rounds up to the next one.
ARCHIVE_MINUTES = next((m for m in (60, 1440, 4320, 10080) if m >= CHAT_THREAD_ARCHIVE_MINUTES), 10080)

_TITLE_MAX = 90  # Discord allows 100
_MARKUP_RE = re.compile(r'<a?(:\w+:)\d+>|<(?:@[!&]?|#)\d+>')
_FORMAT_RE = re.compile(r'[*`]|~~|\|\|')

OPEN_THREAD_TOOL = {
    'type': 'function',
    'function': {
        'name': 'open_thread',
        'description': (
            'Abre uma thread na mensagem de quem perguntou e envia sua resposta (e o resto da '
            'conversa) para ela, deixando o canal limpo. Use quando a pergunta não tem relação '
            'com o assunto atual do canal, for papo casual ou tende a virar um vai-e-volta longo; '
            'não use para perguntas rápidas sobre o assunto do canal. Depois de chamar, responda '
            'normalmente — a resposta vai para a thread.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'title': {
                    'type': 'string',
                    'description': 'Título curto da thread (até 80 caracteres), sobre o assunto da conversa.',
                },
            },
            'required': ['title'],
        },
    },
}


def can_open_thread(channel: Any) -> bool:
    """Whether the bot can open a thread for an answer in *channel*.

    Only text (and announcement) channels have message threads; threads, forum
    posts, voice-channel chats and DMs answer in place.
    """
    if not CHAT_THREADS_ENABLED or not isinstance(channel, discord.TextChannel):
        return False
    perms = channel.permissions_for(channel.guild.me)
    return perms.create_public_threads and perms.send_messages_in_threads


def thread_due(channel: Any, turn: int) -> bool:
    """Whether turn number *turn* of a conversation in *channel* moves it to a thread."""
    if not CHAT_THREADS_ENABLED:
        return False
    if getattr(channel, 'id', None) in CHAT_THREAD_CHANNELS:
        return True
    return CHAT_THREAD_AFTER_TURNS > 0 and turn >= CHAT_THREAD_AFTER_TURNS


def answers_unprompted(message: Any, bot_user_id: int | str) -> bool:
    """Whether *message* reaches the bot without a mention: it was posted in a thread the bot opened."""
    channel = message.channel
    return (
        CHAT_THREADS_ENABLED
        and isinstance(channel, discord.Thread)
        and str(channel.owner_id) == str(bot_user_id)
        and message.type in (discord.MessageType.default, discord.MessageType.reply)
    )


def thread_title(text: str, fallback: str) -> str:
    """Thread name for a question: its own text (no quoted message or markup), shortened."""
    text = (text or '').rsplit('\n---\n', 1)[-1]
    text = _MARKUP_RE.sub(lambda m: m.group(1) or '', text)
    text = ' '.join(_FORMAT_RE.sub('', text).split())
    if not text:
        return fallback[:_TITLE_MAX]
    if len(text) > _TITLE_MAX:
        text = text[:_TITLE_MAX - 1].rstrip() + '…'
    return text


def thread_tool_status(args: dict) -> str:
    return f"🧵 Abrindo thread: *{str(args.get('title') or '')[:40]}*"


class ThreadRequest:
    """Per-answer executor for ``open_thread``: records the thread the model asked for.

    The thread itself is opened once the answer is ready, so a failed or
    silent run never leaves an empty thread behind. *available* is whether
    this answer can still go to a new thread.
    """

    def __init__(self, available: bool):
        self.available = available
        self.title: str | None = None

    def __call__(self, args: dict) -> str:
        if not self.available:
            return (
                'Não dá para abrir uma thread aqui (já é uma thread, o canal não permite '
                'ou a conversa já tem uma). Responda normalmente.'
            )
        title = thread_title(str(args.get('title') or ''), '')
        if not title:
            return 'Informe um título curto para a thread.'
        self.title = title
        return f'Thread "{title}" será aberta e sua resposta irá para ela. Responda normalmente.'


async def open_thread(anchor: discord.Message, title: str) -> discord.Thread | None:
    """Open a public thread on *anchor*; ``None`` (answer in place) when Discord refuses."""
    try:
        return await anchor.create_thread(name=title, auto_archive_duration=ARCHIVE_MINUTES)
    except discord.HTTPException:
        logger.warning('[threads] could not open a thread on message %s', anchor.id, exc_info=True)
        return None


async def conversation_thread(guild: Any, thread_id: Any) -> discord.Thread | None:
    """The thread a conversation moved to, unarchived; ``None`` when it's gone or locked."""
    if guild is None or not thread_id:
        return None
    try:
        tid = int(thread_id)
    except (TypeError, ValueError):
        return None
    thread = guild.get_thread(tid)
    try:
        if thread is None:
            thread = await guild.fetch_channel(tid)  # archived threads aren't cached
        if not isinstance(thread, discord.Thread) or thread.locked:
            return None
        if thread.archived:
            thread = await thread.edit(archived=False)
    except discord.HTTPException:
        logger.info('[threads] conversation thread %s is unavailable', tid, exc_info=True)
        return None
    return thread
