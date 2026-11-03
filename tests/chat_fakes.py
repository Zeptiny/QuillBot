"""Fake Discord objects and a chat engine wired to them, for the conversation tests.

``make_engine`` builds a real :class:`core.chat_engine.ChatEngine` on real
services (SQLite stores in a temp dir) with ``run_turn`` replaced by a fake
that records each :class:`core.turn.TurnContext` and answers from a script.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace as NS

import discord

from core import chat_engine, chat_threads
from core.budget import Budget
from core.context import ChannelContext
from core.conversations import ConversationStore
from core.llm import LLMService
from core.services import Services
from core.settings import SettingsStore
from core.turn import TurnResult

BOT_ID = 900
ASKER = NS(id=222, name='john', display_name='John', bot=False, mention='<@222>')
OTHER = NS(id=333, name='ana', display_name='Ana', bot=False, mention='<@333>')
NOW = datetime.datetime(2026, 10, 28, 12, 0, tzinfo=datetime.timezone.utc)


class NoTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def history_of(messages):
    def history(*, limit, before=None, after=None):
        async def gen():
            for m in messages[:limit]:
                yield m
        return gen()
    return history


class Guild:
    def __init__(self, members=()):
        self.id = 1
        self.name = 'Miners'
        self.me = NS(id=BOT_ID, display_name='QuillBot', name='quillbot')
        self.threads = {}
        self.members = list(members) or [ASKER, OTHER]
        self.emojis = []
        self.channels = []
        self.description = None
        self.owner_id = 1

    def get_thread(self, tid):
        return self.threads.get(tid)

    def get_channel(self, cid):
        return None

    def get_channel_or_thread(self, cid):
        return self.threads.get(cid)

    def get_member(self, uid):
        return next((m for m in self.members if m.id == uid), None)

    def get_member_named(self, name):
        return next((m for m in self.members if name in (m.name, m.display_name)), None)

    async def fetch_channel(self, cid):
        raise discord.NotFound(NS(status=404, reason='Not Found'), 'Unknown Channel')


class Sent:
    """A message the bot sent; editable and deletable like a PartialMessage."""

    def __init__(self, mid, channel, kwargs):
        self.id = mid
        self.channel = channel
        self.kwargs = kwargs
        self.deleted = False

    @property
    def content(self):
        return self.kwargs.get('content')

    async def edit(self, **kwargs):
        self.kwargs = {**self.kwargs, **kwargs}
        return self

    async def delete(self):
        self.deleted = True


class TextChannel(discord.TextChannel):
    """A text channel that passes isinstance checks without a gateway."""

    def __init__(self, guild, *, cid=10, can_thread=True, missing=()):
        self.id = cid
        self.name = 'geral'
        self.guild = guild
        self.topic = None
        self.missing = set(missing) if can_thread else {'create_public_threads', 'send_messages_in_threads'}
        self.messages = []
        self.sent = []
        self.next_id = 5000

    @property
    def parent_id(self):
        return None

    def permissions_for(self, member):
        return discord.Permissions(**{
            flag: flag not in self.missing for flag in ('create_public_threads', 'send_messages_in_threads')
        })

    def history(self, *, limit, before=None, after=None):
        return history_of(self.messages)(limit=limit, before=before)

    def typing(self):
        return NoTyping()

    async def fetch_message(self, mid):
        for m in self.messages:
            if m.id == mid:
                return m
        raise discord.NotFound(NS(status=404, reason='Not Found'), 'Unknown Message')

    async def send(self, content=None, **kwargs):
        self.next_id += 1
        if content is not None:
            kwargs['content'] = content
        msg = Sent(self.next_id, self, kwargs)
        self.sent.append(msg)
        return msg


class Thread(discord.Thread):
    def __init__(self, guild, *, tid=700, owner_id=BOT_ID, name='thread', archived=False, parent_id=10):
        self.id = tid
        self.name = name
        self.guild = guild
        self.owner_id = owner_id
        self.parent_id = parent_id
        self.archived = archived
        self.locked = False
        self.sent = []
        self.messages = []
        guild.threads[tid] = self

    async def send(self, content=None, **kwargs):
        if content is not None:
            kwargs['content'] = content
        self.sent.append(kwargs)
        return NS(id=self.id + len(self.sent))

    async def edit(self, **kwargs):
        self.archived = kwargs.get('archived', self.archived)
        return self

    def history(self, *, limit, before=None, after=None):
        return history_of(self.messages)(limit=limit, before=before)

    def typing(self):
        return NoTyping()


class Message:
    def __init__(self, mid, channel, content='', *, author=ASKER, mtype=discord.MessageType.default, reference=None):
        self.id = mid
        self.channel = channel
        self.guild = channel.guild
        self.author = author
        self.content = content
        self.type = mtype
        self.reference = reference
        self.attachments = []
        self.embeds = []
        self.stickers = []
        self.mentions = []
        self.created_at = NOW
        self.jump_url = f'https://discord.com/channels/1/{channel.id}/{mid}'
        self.replies = []
        self.reactions_added = []
        self.opened = []

    async def reply(self, content=None, **kwargs):
        if content is not None:
            kwargs['content'] = content
        self.replies.append(kwargs)
        return NS(id=self.id + 1)

    async def add_reaction(self, emoji):
        self.reactions_added.append(emoji)

    async def create_thread(self, *, name, auto_archive_duration):
        self.opened.append((name, auto_archive_duration))
        return Thread(self.guild, tid=self.id + 100, name=name)


def reply_to(mid):
    return NS(message_id=mid, resolved=None)


class PartialChannel:
    """``bot.get_partial_messageable`` stand-in that records edits and deletes."""

    def __init__(self, log):
        self.log = log

    def get_partial_message(self, mid):
        log = self.log

        class _Partial:
            id = mid

            async def edit(self, **kwargs):
                log.append(('edit', mid, kwargs))
                return NS(id=mid)

            async def delete(self):
                log.append(('delete', mid))
        return _Partial()


def make_services(tmp_path):
    settings = SettingsStore(str(tmp_path / 'settings.db'))
    settings.load()
    return Services(
        llm=LLMService(client=object()),
        settings=settings,
        conversations=ConversationStore(str(tmp_path / 'conv.db'), kind='chat', keep_expired=True),
    )


def make_engine(tmp_path, monkeypatch, *, answers=('Reduz o view-distance.',), thread_title=None,
                structured=False, budget=None, mentioned=None, thread_name=''):
    """A ChatEngine on real stores whose turns answer from *answers* (the last one repeats).

    Threads the bot opens on its own are named *thread_name* ('' falls back to
    the first question; an exception is raised); the naming calls are kept in
    ``engine.title_calls``.
    """
    services = make_services(tmp_path)
    title_calls: list[dict] = []

    async def fake_text(**kwargs):
        title_calls.append(kwargs)
        if isinstance(thread_name, Exception):
            raise thread_name
        return thread_name

    services.llm.text = fake_text
    partial_log: list = []
    bot_user = NS(
        id=BOT_ID, name='quillbot', display_name='QuillBot',
        mentioned_in=mentioned or (lambda m: f'<@{BOT_ID}>' in (m.content or '')),
    )
    bot = NS(
        user=bot_user, services=services,
        get_partial_messageable=lambda cid, guild_id=None: PartialChannel(partial_log),
        get_channel=lambda cid: None, get_guild=lambda gid: None,
    )
    engine = chat_engine.ChatEngine(
        bot, services, burst_delay=0,
        budget=budget or Budget(user_limit=100, user_window=60, guild_limit=100, guild_window=60),
    )
    services.chat = engine
    engine.turns = []
    engine.title_calls = title_calls
    engine.partial_log = partial_log
    script = list(answers)

    async def fake_run_turn(ctx):
        engine.turns.append(ctx)
        answer = script.pop(0) if len(script) > 1 else script[0]
        return TurnResult(
            answer=answer, sources=[], user_message={'role': 'user', 'content': ctx.question},
            thread_title=thread_title, structured=structured,
        )

    async def no_participants(msg):
        return []

    async def no_gap(message, history):
        return ChannelContext()

    async def no_images(urls):
        return []

    monkeypatch.setattr(chat_engine, 'run_turn', fake_run_turn)
    monkeypatch.setattr(chat_engine, 'message_participant_infos', no_participants)
    monkeypatch.setattr(chat_engine, 'fetch_turn_gap', no_gap)
    monkeypatch.setattr(chat_engine.image_store, 'persist_images', no_images)
    monkeypatch.setattr(chat_engine.image_store, 'persist_shown', no_images)
    monkeypatch.setattr(chat_engine, 'CHAT_MENTION_ENABLED', True)
    monkeypatch.setattr(chat_engine, 'CHAT_MENTION_CONTINUE_ENABLED', True)
    monkeypatch.setattr(chat_engine, 'CHAT_MENTION_CONTINUE_WINDOW', 10)
    monkeypatch.setattr(chat_engine, 'CHAT_BURST_SHORT_SECONDS', 0)  # tests opt in to the longer wait
    monkeypatch.setattr(chat_threads, 'CHAT_THREADS_ENABLED', True)
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_AFTER_TURNS', 3)
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_CHANNELS', frozenset())
    return engine


async def seed(engine, turns, *, handle=60, thread_id=None, channel_id='10', first_question='como reduzo o lag do servidor?',
               author_id='222'):
    """A stored conversation of *turns* turns by *author_id*, continued through bot message *handle*."""
    data = {
        'turns': [
            {'question': first_question if i == 0 else f'q{i}', 'answer': f'a{i}',
             'message_id': str(50 + i), 'channel_id': channel_id, 'author': {'id': author_id}}
            for i in range(turns)
        ],
        'participants': [],
    }
    if thread_id:
        data['thread_id'] = str(thread_id)
    await engine.store.create(handle, channel_id=channel_id, data=data)
    return await engine.store.get_by_handle(handle)
