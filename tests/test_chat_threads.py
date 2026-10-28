"""Chat threads: long or off-topic conversations move out of the channel."""

import datetime
from types import SimpleNamespace as NS

import discord

from cogs import chat_threads
from cogs import commands as commands_mod
from cogs import monitors as mon_mod
from cogs.conversation_store import ConversationStore
from cogs.utils import ChannelContext

from helpers import as_stream, check

BOT_ID = 900
ASKER = NS(id=222, name='john', display_name='John', bot=False, mention='<@222>')
FIRST_QUESTION = 'como reduzo o lag do servidor?'


class NoTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def history_of(messages):
    def history(*, limit, before=None):
        async def gen():
            for m in messages[:limit]:
                yield m
        return gen()
    return history


class Guild:
    def __init__(self):
        self.id = 1
        self.name = 'Miners'
        self.me = NS(id=BOT_ID, display_name='QuillBot')
        self.threads = {}

    def get_thread(self, tid):
        return self.threads.get(tid)

    def get_channel(self, cid):
        return None

    async def fetch_channel(self, cid):
        raise discord.NotFound(NS(status=404, reason='Not Found'), 'Unknown Channel')


class TextChannel(discord.TextChannel):
    """A text channel that passes isinstance checks without a gateway."""

    def __init__(self, guild, *, cid=10, can_thread=True, missing=()):
        self.id = cid
        self.name = 'geral'
        self.guild = guild
        self.missing = set(missing) if can_thread else {'create_public_threads', 'send_messages_in_threads'}
        self.messages = []

    def permissions_for(self, member):
        return discord.Permissions(**{
            flag: flag not in self.missing for flag in ('create_public_threads', 'send_messages_in_threads')
        })

    def history(self, *, limit, before=None):
        return history_of(self.messages)(limit=limit, before=before)

    def typing(self):
        return NoTyping()


class Thread(discord.Thread):
    def __init__(self, guild, *, tid=700, owner_id=BOT_ID, name='thread', archived=False):
        self.id = tid
        self.name = name
        self.guild = guild
        self.owner_id = owner_id
        self.parent_id = 10
        self.archived = archived
        self.locked = False
        self.sent = []
        self.messages = []
        guild.threads[tid] = self

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        return NS(id=self.id + len(self.sent))

    async def edit(self, **kwargs):
        self.archived = kwargs.get('archived', self.archived)
        return self

    def history(self, *, limit, before=None):
        return history_of(self.messages)(limit=limit, before=before)

    def typing(self):
        return NoTyping()


class Message:
    def __init__(self, mid, channel, content='', *, mtype=discord.MessageType.default):
        self.id = mid
        self.channel = channel
        self.guild = channel.guild
        self.author = ASKER
        self.content = content
        self.type = mtype
        self.reference = None
        self.attachments = []
        self.embeds = []
        self.stickers = []
        self.mentions = []
        self.created_at = datetime.datetime(2026, 10, 28, 12, 0, tzinfo=datetime.timezone.utc)
        self.jump_url = f'https://discord.com/channels/1/{channel.id}/{mid}'
        self.replies = []
        self.opened = []

    async def reply(self, *args, **kwargs):
        self.replies.append(kwargs)
        return NS(id=self.id + 1)

    async def create_thread(self, *, name, auto_archive_duration):
        self.opened.append((name, auto_archive_duration))
        return Thread(self.guild, tid=self.id + 100, name=name)


def make_cog(tmp_path, monkeypatch, *, answer='Reduz o view-distance.', thread_title=None):
    cog = commands_mod.Commands.__new__(commands_mod.Commands)
    bot_user = NS(id=BOT_ID, display_name='QuillBot', mentioned_in=lambda m: f'<@{BOT_ID}>' in m.content)
    cog.bot = NS(user=bot_user, get_cog=lambda name: None)
    cog.store = ConversationStore(str(tmp_path / 'conv.db'), kind='chat')
    cog.client = object()
    cog._followup_cd = {}
    cog.runs = []

    async def fake_run_chat(question, **kw):
        cog.runs.append((question, kw))
        return answer, [discord.Embed(description=answer)], [], {'thread_title': thread_title}

    async def no_participants(msg):
        return []

    async def no_gap(message, history):
        return ChannelContext()

    async def no_images(urls):
        return []

    cog._run_chat = fake_run_chat
    monkeypatch.setattr(commands_mod, 'message_participant_infos', no_participants)
    monkeypatch.setattr(commands_mod, 'fetch_turn_gap', no_gap)
    monkeypatch.setattr(commands_mod.image_store, 'persist_images', no_images)
    monkeypatch.setattr(commands_mod, 'CHAT_MENTION_ENABLED', True)
    monkeypatch.setattr(commands_mod, 'CHAT_MENTION_CONTINUE_ENABLED', True)
    monkeypatch.setattr(commands_mod, 'CHANNEL_CONTEXT_MESSAGES', 10)
    monkeypatch.setattr(chat_threads, 'CHAT_THREADS_ENABLED', True)
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_AFTER_TURNS', 3)
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_CHANNELS', frozenset())
    return cog


async def seed(cog, turns, *, handle=60, thread_id=None):
    """A stored conversation of *turns* turns, continued through bot message *handle*."""
    data = {
        'turns': [
            {'question': FIRST_QUESTION if i == 0 else f'q{i}', 'answer': f'a{i}',
             'message_id': str(50 + i), 'channel_id': '10', 'author': {'id': '222'}}
            for i in range(turns)
        ],
        'participants': [],
    }
    if thread_id:
        data['thread_id'] = str(thread_id)
    await cog.store.create(handle, channel_id='10', data=data)
    return await cog.store.get_by_handle(handle)


# --- Helpers ------------------------------------------------------------------

def test_thread_title_keeps_the_question_only():
    quoted = '[Mensagem respondida — Ana (@ana)]\nConteúdo: oi\n---\n**Como** usar <@123> no `paper.yml`? <:pog:42>'
    check('quoted message and markup dropped', chat_threads.thread_title(quoted, 'x') == 'Como usar no paper.yml? :pog:',
          chat_threads.thread_title(quoted, 'x'))
    long = chat_threads.thread_title('a' * 200, 'x')
    check('long questions are shortened', len(long) == 90 and long.endswith('…'), long)
    check('empty text uses the fallback', chat_threads.thread_title(' <@1> ', 'Conversa com John') == 'Conversa com John')


def test_thread_due_on_long_conversations_and_listed_channels(monkeypatch):
    monkeypatch.setattr(chat_threads, 'CHAT_THREADS_ENABLED', True)
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_AFTER_TURNS', 3)
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_CHANNELS', frozenset({20}))
    channel = NS(id=10)
    check('first two turns stay in the channel', not chat_threads.thread_due(channel, 1) and not chat_threads.thread_due(channel, 2))
    check('the third turn moves', chat_threads.thread_due(channel, 3))
    check('listed channels move on the first turn', chat_threads.thread_due(NS(id=20), 1))
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_AFTER_TURNS', 0)
    check('0 turns never moves by length', not chat_threads.thread_due(channel, 10))
    monkeypatch.setattr(chat_threads, 'CHAT_THREADS_ENABLED', False)
    check('the master switch turns everything off', not chat_threads.thread_due(NS(id=20), 1))


def test_thread_request_only_records_when_available():
    req = chat_threads.ThreadRequest(None)
    result = req({'title': 'Receita de **bolo**'})
    check('title recorded without markup', req.title == 'Receita de bolo', req.title)
    check('model told its answer goes there', 'Receita de bolo' in result, result)
    req = chat_threads.ThreadRequest(None)
    req({'title': '  '})
    check('an empty title is refused', req.title is None)
    req = chat_threads.ThreadRequest('a conversa já está numa thread')
    result = req({'title': 'x'})
    check('unavailable here, with the reason', req.title is None and 'numa thread' in result, result)


def test_can_open_thread_needs_a_text_channel_and_permissions(monkeypatch):
    monkeypatch.setattr(chat_threads, 'CHAT_THREADS_ENABLED', True)
    guild = Guild()
    check('text channel with thread permissions', chat_threads.can_open_thread(TextChannel(guild)))
    check('no thread permissions', not chat_threads.can_open_thread(TextChannel(guild, can_thread=False)))
    check('threads answer in place', not chat_threads.can_open_thread(Thread(guild)))
    check('DMs and the like answer in place', not chat_threads.can_open_thread(NS(id=5)))


def test_thread_blocker_names_the_missing_permission(monkeypatch, caplog):
    monkeypatch.setattr(chat_threads, 'CHAT_THREADS_ENABLED', True)
    monkeypatch.setattr(chat_threads, '_warned_channels', set())
    guild = Guild()
    check('nothing blocks a channel with both permissions', chat_threads.thread_blocker(TextChannel(guild)) is None)
    reason = chat_threads.thread_blocker(TextChannel(guild, missing={'send_messages_in_threads'}))
    check('names the missing permission', 'Enviar mensagens em threads' in reason, reason)
    check('only the missing one', 'Criar threads públicas' not in reason, reason)
    check('a thread says so', 'numa thread' in chat_threads.thread_blocker(Thread(guild)))
    check('other channels say so', 'tipo de canal' in chat_threads.thread_blocker(NS(id=5)))
    chat_threads.thread_blocker(TextChannel(guild, missing={'send_messages_in_threads'}))
    warnings = [r for r in caplog.records if 'cannot open threads' in r.getMessage()]
    check('logged once per channel', len(warnings) == 1, [r.getMessage() for r in warnings])
    check('the log names the flag', 'send_messages_in_threads' in warnings[0].getMessage(), warnings[0].getMessage())


def test_answers_unprompted_only_in_the_bot_threads(monkeypatch):
    monkeypatch.setattr(chat_threads, 'CHAT_THREADS_ENABLED', True)
    guild = Guild()
    own = Thread(guild, tid=700)
    check('message in the bot thread', chat_threads.answers_unprompted(Message(1, own), BOT_ID))
    check('someone else thread', not chat_threads.answers_unprompted(Message(1, Thread(guild, tid=701, owner_id=5)), BOT_ID))
    check('regular channel', not chat_threads.answers_unprompted(Message(1, TextChannel(guild)), BOT_ID))
    check('system messages', not chat_threads.answers_unprompted(Message(1, own, mtype=discord.MessageType.pins_add), BOT_ID))
    monkeypatch.setattr(chat_threads, 'CHAT_THREADS_ENABLED', False)
    check('disabled', not chat_threads.answers_unprompted(Message(1, own), BOT_ID))


# --- Moving a conversation ----------------------------------------------------

async def test_second_follow_up_moves_the_conversation_to_a_thread(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    guild = Guild()
    channel = TextChannel(guild)
    conv = await seed(cog, 1)

    first = Message(80, channel, 'e o simulation-distance?')
    await cog._continue_conversation(first, 60, conv, reply_to=60)
    check('turn 2 replies in the channel', len(first.replies) == 1 and first.opened == [], first.opened)
    check('the model may still open one itself', cog.runs[-1][1]['thread_blocker'] is None)

    cog._followup_cd.clear()
    conv = await cog.store.get_by_handle(81)
    second = Message(90, channel, 'e o view-distance?')
    await cog._continue_conversation(second, 81, conv, reply_to=81)
    check('turn 3 opens a thread on the asker message, named after the first question',
          second.opened == [(FIRST_QUESTION, chat_threads.ARCHIVE_MINUTES)], second.opened)
    thread = guild.threads[190]
    check('answer posted in the thread, pinging the asker',
          len(thread.sent) == 1 and thread.sent[0]['content'] == '<@222>', thread.sent)
    check('nothing left in the channel', second.replies == [], second.replies)
    moved = await cog.store.get_by_handle(191)
    check('the thread message continues the conversation', moved and moved['conv_id'] == '60', moved)
    check('the conversation remembers its thread', moved['data']['thread_id'] == '190', moved['data'])
    check('all turns kept', len(moved['data']['turns']) == 3, moved['data']['turns'])


async def test_moved_conversation_answers_in_its_thread_from_the_channel(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    guild = Guild()
    thread = Thread(guild, tid=700, archived=True)
    conv = await seed(cog, 3, thread_id=700)
    msg = Message(80, TextChannel(guild), 'mais uma coisa')
    await cog._continue_conversation(msg, 60, conv, reply_to=60)
    check('no second thread', msg.opened == [], msg.opened)
    check('the model is told the conversation has one', cog.runs[0][1]['thread_blocker'] == chat_threads.HAS_THREAD)
    check('archived thread revived and used', not thread.archived and len(thread.sent) == 1, thread.sent)
    check('nothing in the channel', msg.replies == [])


async def test_vanished_thread_opens_a_new_one(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    guild = Guild()
    conv = await seed(cog, 3, thread_id=999)
    msg = Message(80, TextChannel(guild), 'voltei')
    await cog._continue_conversation(msg, 60, conv, reply_to=60)
    check('a long conversation whose thread is gone gets a new one', len(msg.opened) == 1, msg.opened)
    stored = await cog.store.get_by_handle(181)
    check('and points to it', stored['data']['thread_id'] == '180', stored['data'])


async def test_quick_mention_answers_in_place(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    channel = TextChannel(Guild())
    msg = Message(80, channel, f'<@{BOT_ID}> qual a versão do paper?')
    await cog.on_message(msg)
    check('the tool is offered', cog.runs[0][1]['thread_blocker'] is None)
    check('first answer replies in the channel', len(msg.replies) == 1 and msg.opened == [], msg.opened)
    conv = await cog.store.get_by_handle(81)
    check('no thread stored', conv and 'thread_id' not in conv['data'], conv)


async def test_model_moves_an_off_topic_mention_to_a_thread(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch, thread_title='Receita de bolo')
    guild = Guild()
    msg = Message(80, TextChannel(guild), f'<@{BOT_ID}> me passa uma receita de bolo')
    await cog.on_message(msg)
    check('thread named by the model', msg.opened == [('Receita de bolo', chat_threads.ARCHIVE_MINUTES)], msg.opened)
    check('answer in the thread', len(guild.threads[180].sent) == 1 and msg.replies == [])
    conv = await cog.store.get_by_handle(181)
    check('the conversation starts in the thread', conv and conv['data']['thread_id'] == '180', conv)


async def test_listed_channels_always_use_threads(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_CHANNELS', frozenset({10}))
    msg = Message(80, TextChannel(Guild()), f'<@{BOT_ID}> me passa uma receita de bolo')
    await cog.on_message(msg)
    check('thread named after the question, without the mention',
          msg.opened == [('me passa uma receita de bolo', chat_threads.ARCHIVE_MINUTES)], msg.opened)


async def test_channel_without_thread_permission_answers_in_place(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_CHANNELS', frozenset({10}))
    msg = Message(80, TextChannel(Guild(), can_thread=False), f'<@{BOT_ID}> oi')
    await cog.on_message(msg)
    check('the model is told the permission is missing', 'Criar threads públicas' in cog.runs[0][1]['thread_blocker'])
    check('reply in place', len(msg.replies) == 1 and msg.opened == [], msg.opened)


# --- Inside the bot's thread --------------------------------------------------

async def test_message_in_bot_thread_continues_without_a_mention(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    guild = Guild()
    thread = Thread(guild, tid=700)
    thread.messages = [NS(id=60, author=NS(id=BOT_ID))]
    await seed(cog, 3, thread_id=700)
    msg = Message(80, thread, 'e o view-distance?')
    await cog.on_message(msg)
    check('the model is told nobody mentioned it',
          cog.runs and cog.runs[0][0].startswith(chat_threads.UNPROMPTED_HEADER), cog.runs)
    check('answered right there', len(msg.replies) == 1 and msg.opened == [] and thread.sent == [])
    conv = await cog.store.get_by_handle(81)
    check('same conversation, one more turn', conv and len(conv['data']['turns']) == 4, conv)


async def test_chatter_in_bot_thread_can_stay_silent(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch, answer='[NO_REPLY]')
    guild = Guild()
    thread = Thread(guild, tid=700)
    thread.messages = [NS(id=60, author=NS(id=BOT_ID))]
    await seed(cog, 3, thread_id=700)
    msg = Message(80, thread, 'kkkk verdade, Ana')
    await cog.on_message(msg)
    check('nothing sent', msg.replies == [] and thread.sent == [], msg.replies)
    conv = await cog.store.get_by_handle(60)
    check('no turn stored', len(conv['data']['turns']) == 3, conv['data']['turns'])


async def test_new_conversation_in_bot_thread(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    thread = Thread(Guild(), tid=700)
    msg = Message(80, thread, 'outra pergunta')
    await cog.on_message(msg)
    check('expired conversation: a new one starts unprompted',
          cog.runs and cog.runs[0][0] == f'{chat_threads.UNPROMPTED_HEADER}\n---\noutra pergunta', cog.runs)
    check('answered in place', len(msg.replies) == 1 and msg.opened == [])

    silent = make_cog(tmp_path / 'b', monkeypatch, answer='[NO_REPLY]')
    msg = Message(90, thread, 'conversa entre outras pessoas')
    await silent.on_message(msg)
    check('silent run posts and stores nothing',
          msg.replies == [] and await silent.store.get_by_handle(91) is None, msg.replies)


async def test_other_threads_still_need_a_mention(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    guild = Guild()
    await cog.on_message(Message(80, Thread(guild, tid=701, owner_id=5), 'oi gente'))
    await cog.on_message(Message(81, TextChannel(guild), 'oi gente'))
    await cog.on_message(Message(82, Thread(guild, tid=700), '', mtype=discord.MessageType.pins_add))
    check('nothing answered', cog.runs == [], cog.runs)


async def test_monitors_skip_messages_in_bot_threads(monkeypatch):
    monkeypatch.setattr(chat_threads, 'CHAT_THREADS_ENABLED', True)
    monkeypatch.setattr(mon_mod, 'CHAT_MENTION_ENABLED', True)
    cog = mon_mod.Monitors.__new__(mon_mod.Monitors)
    cog.bot = NS(user=NS(id=BOT_ID, mentioned_in=lambda m: False), get_cog=lambda name: None)
    guild = Guild()
    check('the thread handler answers it', await cog._answered_elsewhere(Message(1, Thread(guild, tid=700))))
    check('other threads still fire', not await cog._answered_elsewhere(Message(2, Thread(guild, tid=701, owner_id=5))))


# --- The open_thread tool -----------------------------------------------------

class Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.tools = []
        self.calls = []

    async def create(self, **kwargs):
        self.tools.append([t['function']['name'] for t in kwargs.get('tools') or []])
        self.calls.append(kwargs['messages'])
        return as_stream(self.responses.pop(0))


def completion(content=None, tool_calls=None):
    tcs = [
        NS(id=f'call_{i}', type='function', function=NS(name=name, arguments=args))
        for i, (name, args) in enumerate(tool_calls or [])
    ] or None
    message = NS(role='assistant', content=content, tool_calls=tcs)
    return NS(choices=[NS(message=message, finish_reason='stop')], usage=None)


async def _run_chat_with_tool(monkeypatch, available):
    completions = Completions([
        completion(tool_calls=[('open_thread', '{"title": "Receita de bolo"}')]),
        completion('Aqui vai a receita.'),
    ])
    cog = commands_mod.Commands.__new__(commands_mod.Commands)
    cog.bot = NS(user=NS(id=BOT_ID, name='QuillBot', display_name='QuillBot'), get_cog=lambda name: None)
    cog.client = NS(chat=NS(completions=completions))
    monkeypatch.setattr(commands_mod, 'CHANNEL_CONTEXT_MESSAGES', 0)
    result = await cog._run_chat('me passa uma receita de bolo', user=ASKER, thread_blocker=None if available else chat_threads.ANSWERS_IN_PLACE)
    return result, completions


async def test_open_thread_tool_reports_the_title(monkeypatch):
    (answer, _, _, capture), completions = await _run_chat_with_tool(monkeypatch, True)
    check('tool offered', 'open_thread' in completions.tools[0], completions.tools)
    check('answer unchanged', answer == 'Aqui vai a receita.', answer)
    check('title handed to the delivery code', capture['thread_title'] == 'Receita de bolo', capture)


async def test_open_thread_tool_refuses_when_unavailable(monkeypatch):
    (_, _, _, capture), completions = await _run_chat_with_tool(monkeypatch, False)
    check('tool list stays the same for prompt caching', 'open_thread' in completions.tools[0], completions.tools)
    check('no thread requested', capture['thread_title'] is None, capture)
    tool_msgs = [m for m in completions.calls[1] if isinstance(m, dict) and m.get('role') == 'tool']
    check('the model is told why', tool_msgs and chat_threads.ANSWERS_IN_PLACE in tool_msgs[0]['content'], tool_msgs)


async def test_cooldown_skips_quietly_in_bot_thread(tmp_path, monkeypatch):
    cog = make_cog(tmp_path, monkeypatch)
    guild = Guild()
    thread = Thread(guild, tid=700)
    thread.messages = [NS(id=60, author=NS(id=BOT_ID))]
    await seed(cog, 3, thread_id=700)
    cog._followup_cd[ASKER.id] = True
    msg = Message(80, thread, 'mais uma')
    await cog.on_message(msg)
    check('no cooldown notice in the thread', msg.replies == [] and cog.runs == [], msg.replies)
    mention = Message(81, TextChannel(guild), f'<@{BOT_ID}> oi')
    await cog.on_message(mention)
    check('a mention still gets the notice', len(mention.replies) == 1 and cog.runs == [], mention.replies)
