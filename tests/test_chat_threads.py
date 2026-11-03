"""Chat threads: long or off-topic conversations move out of the channel."""

from types import SimpleNamespace as NS

import discord

from core import chat_threads
from core.context_tools import register as register_core_tools
from core.profiles import CHAT
from core.runner import run_turn
from core.turn import TurnContext
import core.runner as runner_mod

from chat_fakes import (
    ASKER, BOT_ID, OTHER, Guild, Message, TextChannel, Thread, make_engine, make_services, reply_to, seed,
)
from helpers import as_stream, check

FIRST_QUESTION = 'como reduzo o lag do servidor?'


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
    engine = make_engine(tmp_path, monkeypatch)
    guild = Guild()
    channel = TextChannel(guild)
    await seed(engine, 1)

    first = Message(80, channel, 'e o simulation-distance?', reference=reply_to(60))
    await engine.handle_message(first)
    check('turn 2 replies in the channel', len(first.replies) == 1 and first.opened == [], first.opened)
    check('the model may still open one itself', engine.turns[-1].thread_blocker is None)
    check('plain message, not an embed', 'content' in first.replies[0] and 'embed' not in first.replies[0], first.replies)

    second = Message(90, channel, 'e o view-distance?', reference=reply_to(81))
    await engine.handle_message(second)
    check('turn 3 opens a thread on the asker message, named after the first question',
          second.opened == [(FIRST_QUESTION, chat_threads.ARCHIVE_MINUTES)], second.opened)
    thread = guild.threads[190]
    check('answer posted in the thread, pinging the asker',
          len(thread.sent) == 1 and thread.sent[0]['content'].startswith('<@222> Reduz'), thread.sent)
    check('nothing left in the channel', second.replies == [], second.replies)
    moved = await engine.store.get_by_handle(191)
    check('the thread message continues the conversation', moved and moved['conv_id'] == '60', moved)
    check('the conversation remembers its thread', moved['data']['thread_id'] == '190', moved['data'])
    check('all turns kept', len(moved['data']['turns']) == 3, moved['data']['turns'])


async def test_moved_conversation_answers_in_its_thread_from_the_channel(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    guild = Guild()
    thread = Thread(guild, tid=700, archived=True)
    await seed(engine, 3, thread_id=700)
    msg = Message(80, TextChannel(guild), 'mais uma coisa', reference=reply_to(60))
    await engine.handle_message(msg)
    check('no second thread', msg.opened == [], msg.opened)
    check('the model is told the conversation has one', engine.turns[0].thread_blocker == chat_threads.HAS_THREAD)
    check('archived thread revived and used', not thread.archived and len(thread.sent) == 1, thread.sent)
    check('nothing in the channel', msg.replies == [])


async def test_being_named_in_passing_does_not_move_the_conversation(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    channel = TextChannel(Guild())
    channel.messages = [NS(id=60, author=NS(id=BOT_ID))]
    await seed(engine, 2)
    msg = Message(80, channel, 'só de citar o nome quill ele pode responder')
    await engine.handle_message(msg)
    check('third turn, but only named: answered in the channel', msg.opened == [] and len(msg.replies) == 1, msg.opened)
    conv = await engine.store.get_by_trigger(80)
    check('still the same conversation, not moved', conv['conv_id'] == '60' and 'thread_id' not in conv['data'], conv)


async def test_a_group_chatting_with_the_bot_stays_in_the_channel(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    channel = TextChannel(Guild())
    channel.messages = [NS(id=60, author=NS(id=BOT_ID))]
    await seed(engine, 2, author_id=str(OTHER.id))
    first = Message(80, channel, f'<@{BOT_ID}> as urnas são fraudadas?')
    await engine.handle_message(first)
    check("someone else's third turn stays in the channel", first.opened == [] and len(first.replies) == 1, first.opened)
    second = Message(90, channel, 'e quem auditou?', reference=reply_to(81))
    await engine.handle_message(second)
    check('their second turn too', second.opened == [] and len(second.replies) == 1, second.opened)
    third = Message(100, channel, 'e em 2022?', reference=reply_to(91))
    await engine.handle_message(third)
    check('their own third turn in a row moves it, named after their first question',
          third.opened == [('as urnas são fraudadas?', chat_threads.ARCHIVE_MINUTES)], third.opened)


async def test_moved_conversation_does_not_pull_new_calls_into_its_thread(tmp_path, monkeypatch):
    """Ana and Nowzin calling the bot in the channel got answered in John's thread."""
    engine = make_engine(tmp_path, monkeypatch)
    guild = Guild()
    thread = Thread(guild, tid=700)
    channel = TextChannel(guild)
    channel.messages = [NS(id=60, author=NS(id=BOT_ID))]
    await seed(engine, 3, thread_id=700)
    named = Message(80, channel, 'quill as urnas sao fraudadas?', author=OTHER)
    await engine.handle_message(named)
    mention = Message(90, channel, f'<@{BOT_ID}> me ignora não', author=OTHER)
    await engine.handle_message(mention)
    check('both answered in the channel', len(named.replies) == 1 and len(mention.replies) == 1, (named.replies, mention.replies))
    check('nothing posted in the thread', thread.sent == [], thread.sent)
    conv = await engine.store.get_by_trigger(80)
    check('a new conversation, here', conv['conv_id'] == '81' and 'thread_id' not in conv['data'], conv)
    check('the moved one is untouched', len((await engine.store.get_by_handle(60))['data']['turns']) == 3)


async def test_follow_ups_in_the_channel_stay_there_after_a_move(tmp_path, monkeypatch):
    from core.burst import Burst

    engine = make_engine(tmp_path, monkeypatch)
    guild = Guild()
    thread = Thread(guild, tid=700)
    channel = TextChannel(guild)
    await seed(engine, 3, thread_id=700)
    engine._last_answer[(10, ASKER.id)] = 60  # their last answer went to the thread
    msg = Message(80, channel, 'hm')
    await engine.answer_burst(Burst((10, ASKER.id), [msg], ['join']))
    check('answered in the channel', len(msg.replies) == 1 and thread.sent == [], thread.sent)


async def test_the_bot_names_the_threads_it_opens_on_its_own(tmp_path, monkeypatch):
    from config import CHAT_THREAD_TITLE_MODEL
    from core.prompts import THREAD_TITLE_PROMPT

    engine = make_engine(tmp_path, monkeypatch, thread_name='Título: "Lag e view-distance no Paper".')
    guild = Guild()
    channel = TextChannel(guild)
    await seed(engine, 1)
    await engine.handle_message(Message(80, channel, 'e o simulation-distance?', reference=reply_to(60)))
    check('no naming while the answer stays in the channel', engine.title_calls == [], engine.title_calls)
    msg = Message(90, channel, 'e o view-distance?', reference=reply_to(81))
    await engine.handle_message(msg)
    check('named by the model, label and quotes dropped',
          msg.opened == [('Lag e view-distance no Paper', chat_threads.ARCHIVE_MINUTES)], msg.opened)
    call = engine.title_calls[0]
    check('one cheap call with the title prompt', len(engine.title_calls) == 1 and call['system'] == THREAD_TITLE_PROMPT
          and call['model'] == CHAT_THREAD_TITLE_MODEL, engine.title_calls)
    check('it sees the whole back-and-forth, this answer included',
          all(t in call['user'] for t in (FIRST_QUESTION, 'a0', 'e o simulation-distance?', 'e o view-distance?', 'Reduz o view-distance.')), call['user'])


async def test_thread_name_falls_back_to_the_first_question(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, thread_name=RuntimeError('model down'))
    await seed(engine, 2)
    msg = Message(90, TextChannel(Guild()), 'e o view-distance?', reference=reply_to(60))
    await engine.handle_message(msg)
    check('still opened, named after the first question', msg.opened == [(FIRST_QUESTION, chat_threads.ARCHIVE_MINUTES)], msg.opened)


def test_open_thread_is_for_long_or_off_topic_subjects_only():
    from core.prompts import CHAT_SYSTEM_PROMPT

    description = chat_threads.OPEN_THREAD_TOOL['function']['description']
    check('the tool keeps chat in the channel', 'papo casual' not in description and 'Não use para papo' in description, description)
    check('and so does the prompt', 'papo casual ou' not in CHAT_SYSTEM_PROMPT and 'ficam no canal' in CHAT_SYSTEM_PROMPT)


async def test_vanished_thread_opens_a_new_one(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    guild = Guild()
    await seed(engine, 3, thread_id=999)
    msg = Message(80, TextChannel(guild), 'voltei', reference=reply_to(60))
    await engine.handle_message(msg)
    check('a long conversation whose thread is gone gets a new one', len(msg.opened) == 1, msg.opened)
    stored = await engine.store.get_by_handle(181)
    check('and points to it', stored['data']['thread_id'] == '180', stored['data'])


async def test_quick_mention_answers_in_place(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    channel = TextChannel(Guild())
    msg = Message(80, channel, f'<@{BOT_ID}> qual a versão do paper?')
    await engine.handle_message(msg)
    check('the tool is offered', engine.turns[0].thread_blocker is None)
    check('first answer replies in the channel', len(msg.replies) == 1 and msg.opened == [], msg.opened)
    check('a mention reply does not ping the asker', msg.replies[0].get('mention_author') is False, msg.replies)
    conv = await engine.store.get_by_handle(81)
    check('no thread stored', conv and 'thread_id' not in conv['data'], conv)
    check('the trigger maps to the conversation', (await engine.store.get_by_trigger(80))['conv_id'] == '81')


async def test_model_moves_an_off_topic_mention_to_a_thread(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, thread_title='Receita de bolo')
    guild = Guild()
    msg = Message(80, TextChannel(guild), f'<@{BOT_ID}> me passa uma receita de bolo')
    await engine.handle_message(msg)
    check('thread named by the model', msg.opened == [('Receita de bolo', chat_threads.ARCHIVE_MINUTES)], msg.opened)
    check('answer in the thread', len(guild.threads[180].sent) == 1 and msg.replies == [])
    check('no second call to name it', engine.title_calls == [], engine.title_calls)
    conv = await engine.store.get_by_handle(181)
    check('the conversation starts in the thread', conv and conv['data']['thread_id'] == '180', conv)


async def test_listed_channels_always_use_threads(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(chat_threads, 'CHAT_THREAD_CHANNELS', frozenset({10}))
    import core.settings as settings_mod
    monkeypatch.setattr(settings_mod, 'CHAT_THREAD_CHANNELS', frozenset({10}))
    msg = Message(80, TextChannel(Guild()), f'<@{BOT_ID}> me passa uma receita de bolo')
    await engine.handle_message(msg)
    check('thread named after the question, without the mention',
          msg.opened == [('me passa uma receita de bolo', chat_threads.ARCHIVE_MINUTES)], msg.opened)


async def test_threads_setting_per_channel(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    engine.services.settings.write(1, 10, 'threads', 'always')
    msg = Message(80, TextChannel(Guild()), f'<@{BOT_ID}> oi')
    await engine.handle_message(msg)
    check('"always" opens a thread on the first answer', len(msg.opened) == 1, msg.opened)

    engine.services.settings.write(1, 10, 'threads', 'never')
    await seed(engine, 5, handle=600)
    msg = Message(90, TextChannel(Guild()), 'mais', reference=reply_to(600))
    await engine.handle_message(msg)
    check('"never" keeps even long conversations in the channel', msg.opened == [] and len(msg.replies) == 1, msg.opened)
    check('and the model is told why', 'configuração do servidor' in engine.turns[-1].thread_blocker, engine.turns[-1].thread_blocker)


async def test_channel_without_thread_permission_answers_in_place(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    engine.services.settings.write(1, 10, 'threads', 'always')
    msg = Message(80, TextChannel(Guild(), can_thread=False), f'<@{BOT_ID}> oi')
    await engine.handle_message(msg)
    check('the model is told the permission is missing', 'Criar threads públicas' in engine.turns[0].thread_blocker)
    check('reply in place', len(msg.replies) == 1 and msg.opened == [], msg.opened)


# --- Inside the bot's thread --------------------------------------------------

async def test_message_in_bot_thread_continues_without_a_mention(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    guild = Guild()
    thread = Thread(guild, tid=700)
    thread.messages = [NS(id=60, author=NS(id=BOT_ID))]
    await seed(engine, 3, thread_id=700)
    msg = Message(80, thread, 'e o view-distance?')
    await engine.handle_message(msg)
    check('the model is told nobody mentioned it',
          engine.turns and engine.turns[0].question.startswith(chat_threads.UNPROMPTED_HEADER), engine.turns)
    check('answered right there', len(msg.replies) == 1 and msg.opened == [] and thread.sent == [])
    conv = await engine.store.get_by_handle(81)
    check('same conversation, one more turn', conv and len(conv['data']['turns']) == 4, conv)


async def test_chatter_in_bot_thread_can_stay_silent(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, answers=('[NO_REPLY]',))
    guild = Guild()
    thread = Thread(guild, tid=700)
    thread.messages = [NS(id=60, author=NS(id=BOT_ID))]
    await seed(engine, 3, thread_id=700)
    msg = Message(80, thread, 'kkkk verdade, Ana')
    await engine.handle_message(msg)
    check('nothing sent', msg.replies == [] and thread.sent == [], msg.replies)
    conv = await engine.store.get_by_handle(60)
    check('no turn stored', len(conv['data']['turns']) == 3, conv['data']['turns'])


async def test_new_conversation_in_bot_thread(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    thread = Thread(Guild(), tid=700)
    msg = Message(80, thread, 'outra pergunta')
    await engine.handle_message(msg)
    check('expired conversation: a new one starts unprompted',
          engine.turns and engine.turns[0].question == f'{chat_threads.UNPROMPTED_HEADER}\n---\noutra pergunta', engine.turns)
    check('answered in place', len(msg.replies) == 1 and msg.opened == [])

    silent = make_engine(tmp_path / 'b', monkeypatch, answers=('[NO_REPLY]',))
    msg = Message(90, thread, 'conversa entre outras pessoas')
    await silent.handle_message(msg)
    check('silent run posts and stores nothing',
          msg.replies == [] and await silent.store.get_by_handle(91) is None, msg.replies)


async def test_other_threads_still_need_a_mention(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    guild = Guild()
    claimed = [
        await engine.handle_message(Message(80, Thread(guild, tid=701, owner_id=5), 'oi gente')),
        await engine.handle_message(Message(81, TextChannel(guild), 'oi gente')),
        await engine.handle_message(Message(82, Thread(guild, tid=700), '', mtype=discord.MessageType.pins_add)),
    ]
    check('nothing answered', engine.turns == [] and claimed == [False, False, False], (engine.turns, claimed))


# --- The open_thread tool (through the real runner) ---------------------------

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


async def _run_with_tool(tmp_path, monkeypatch, available):
    completions = Completions([
        completion(tool_calls=[('open_thread', '{"title": "Receita de bolo"}')]),
        completion('Aqui vai a receita.'),
    ])
    services = make_services(tmp_path)
    services.llm.client = NS(chat=NS(completions=completions))
    register_core_tools(services.tools)
    monkeypatch.setattr(runner_mod, 'CHANNEL_CONTEXT_MESSAGES', 0)
    ctx = TurnContext(
        bot=NS(user=NS(id=BOT_ID, name='QuillBot', display_name='QuillBot'), services=services),
        profile=CHAT, question='me passa uma receita de bolo', user=ASKER,
        thread_blocker=None if available else chat_threads.ANSWERS_IN_PLACE,
    )
    return await run_turn(ctx), completions


async def test_open_thread_tool_reports_the_title(tmp_path, monkeypatch):
    result, completions = await _run_with_tool(tmp_path, monkeypatch, True)
    check('tool offered', 'open_thread' in completions.tools[0], completions.tools)
    check('answer unchanged', result.answer == 'Aqui vai a receita.', result.answer)
    check('title handed to the delivery code', result.thread_title == 'Receita de bolo', result)


async def test_open_thread_tool_refuses_when_unavailable(tmp_path, monkeypatch):
    result, completions = await _run_with_tool(tmp_path, monkeypatch, False)
    check('tool list stays the same for prompt caching', 'open_thread' in completions.tools[0], completions.tools)
    check('no thread requested', result.thread_title is None, result)
    tool_msgs = [m for m in completions.calls[1] if isinstance(m, dict) and m.get('role') == 'tool']
    check('the model is told why', tool_msgs and chat_threads.ANSWERS_IN_PLACE in tool_msgs[0]['content'], tool_msgs)
