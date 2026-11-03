"""How the bot takes part in conversations: bursts, budget, name calls,
staying quiet, per-channel settings, and edits/deletes of answered messages."""

import asyncio

from core import chat_engine
from core.budget import Budget
from core.tool_loop import EMPTY_ANSWER_FALLBACK
from core.turn import FOLLOW_HEADER, NAME_HEADER, TurnResult

from chat_fakes import BOT_ID, NOW, Guild, Message, TextChannel, make_engine, reply_to, seed
from helpers import check


# --- Bursts -------------------------------------------------------------------

async def test_quick_messages_are_answered_together(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    engine.bursts.delay = 0.05
    channel = TextChannel(Guild())
    first = Message(80, channel, f'<@{BOT_ID}> ei')
    second = Message(81, channel, 'sabe aquele erro')
    third = Message(82, channel, 'de ontem?')
    check('a mention opens a burst', await engine.handle_message(first))
    check('a plain follow-up joins it', await engine.handle_message(second))
    check('and another', await engine.handle_message(third))
    stranger = Message(83, channel, 'oi gente', author=chat_engine_other())
    check("someone else's message doesn't join", not await engine.handle_message(stranger))
    await asyncio.sleep(0.2)
    check('one turn for the whole burst', len(engine.turns) == 1, engine.turns)
    check('all lines, mention stripped', engine.turns[0].question == 'ei\nsabe aquele erro\nde ontem?', engine.turns[0].question)
    check('the answer replies to the last message', third.replies and not first.replies and not second.replies)
    stored = await engine.store.get_by_trigger(80)
    turn = stored['data']['turns'][0]
    check('the turn remembers every message', turn['message_ids'] == ['80', '81', '82'] and turn['message_id'] == '82', turn)
    check('the window is read before the first message', engine.turns[0].window_before is first)


def chat_engine_other():
    from chat_fakes import OTHER
    return OTHER


async def test_messages_sent_while_answering_become_the_next_turn(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    engine.bursts.delay = 0.02
    release = asyncio.Event()
    turns = []

    async def slow_turn(ctx):
        turns.append(ctx.question)
        if len(turns) == 1:
            await release.wait()
        return TurnResult(answer='ok', sources=[], user_message={'role': 'user', 'content': ctx.question})

    monkeypatch.setattr(chat_engine, 'run_turn', slow_turn)
    channel = TextChannel(Guild())
    await engine.handle_message(Message(80, channel, f'<@{BOT_ID}> pergunta'))
    await asyncio.sleep(0.06)
    check('first turn running', turns == ['pergunta'], turns)
    late = Message(90, channel, 'ah, e outra coisa')
    check('a message while answering is taken', await engine.handle_message(late))
    release.set()
    await asyncio.sleep(0.15)
    check('it is answered next, flagged as not addressed', len(turns) == 2 and turns[1].startswith(FOLLOW_HEADER), turns)
    stored = await engine.store.get_by_trigger(90)
    check('in the same conversation, though the answer came after it',
          stored and stored['conv_id'] == '81' and len(stored['data']['turns']) == 2, stored)


# --- Budget -------------------------------------------------------------------

async def test_budget_reacts_instead_of_answering(tmp_path, monkeypatch):
    engine = make_engine(
        tmp_path, monkeypatch,
        budget=Budget(user_limit=1, user_window=300, guild_limit=100, guild_window=3600),
    )
    channel = TextChannel(Guild())
    first = Message(80, channel, f'<@{BOT_ID}> oi')
    await engine.handle_message(first)
    second = Message(90, channel, f'<@{BOT_ID}> e agora?')
    await engine.handle_message(second)
    check('first answered', len(first.replies) == 1)
    check('second gets ⏳, not an answer or a dropped-message notice', second.replies == [] and second.reactions_added == ['⏳'], second.replies)
    check('only one turn ran', len(engine.turns) == 1)


def test_budget_windows():
    now = [0.0]
    budget = Budget(user_limit=2, user_window=60, guild_limit=3, guild_window=600, clock=lambda: now[0])
    check('two turns for a person', budget.try_spend(1, 9) and budget.try_spend(1, 9))
    check('third refused', not budget.try_spend(1, 9))
    check('someone else still has budget', budget.try_spend(2, 9))
    check('server budget used up', not budget.try_spend(3, 9))
    check('monitors spend only the server budget', not budget.try_spend(None, 9, per_user=False))
    now[0] = 61
    check('person budget refills', budget.users.full(1, now[0]) is False)
    check('retry hint while the server is full', budget.retry_after(1, 9) > 0)


# --- Being called by name -----------------------------------------------------

async def test_name_without_mention_reaches_the_bot(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    channel = TextChannel(Guild())
    msg = Message(80, channel, 'quill, o que acha disso?')
    check('claimed', await engine.handle_message(msg))
    check('the model is told it was only named', engine.turns[0].question.startswith(NAME_HEADER), engine.turns[0].question)
    check('answered', len(msg.replies) == 1)
    for text in ('olha https://quill.example/x', 'usa `quill` aqui', 'quillbots são legais', 'tranquilo'):
        check(f'not a call: {text!r}', not await engine.handle_message(Message(90, channel, text)))


async def test_aliases_and_respond_settings(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    settings = engine.services.settings
    channel = TextChannel(Guild())
    check('unknown nickname ignored', not await engine.handle_message(Message(80, channel, 'pena, tudo bem?')))
    settings.write(1, 0, 'aliases', 'Pena')
    check('server alias answers', await engine.handle_message(Message(81, channel, 'pena, tudo bem?')))

    settings.write(1, 10, 'respond', 'mentions')
    check('"mentions" ignores name calls', not await engine.handle_message(Message(90, channel, 'quill, oi')))
    check('but answers mentions', await engine.handle_message(Message(91, channel, f'<@{BOT_ID}> oi')))

    settings.write(1, 10, 'respond', 'off')
    before = len(engine.turns)
    check('"off" ignores even mentions', not await engine.handle_message(Message(100, channel, f'<@{BOT_ID}> oi')))
    check('no turn ran', len(engine.turns) == before)

    settings.write(1, 10, 'respond', 'on')
    settings.write(1, 10, 'name_trigger', 'false')
    check('name trigger can be switched off per channel', not await engine.handle_message(Message(110, channel, 'quill, oi')))


async def test_channel_settings_reach_the_turn(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    engine.services.settings.write(1, 10, 'instructions', 'aqui é papo técnico, seja direto')
    await engine.handle_message(Message(80, TextChannel(Guild()), f'<@{BOT_ID}> oi'))
    check('instructions in the turn', engine.turns[0].channel_settings['instructions'] == 'aqui é papo técnico, seja direto')


async def test_reply_style_embed_setting(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    engine.services.settings.write(1, 10, 'reply_style', 'embed')
    msg = Message(80, TextChannel(Guild()), f'<@{BOT_ID}> oi')
    await engine.handle_message(msg)
    check('the channel gets embeds', 'embed' in msg.replies[0], msg.replies)


# --- Staying quiet ------------------------------------------------------------

async def test_thanks_can_be_answered_with_silence(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, answers=('[NO_REPLY]',))
    await seed(engine, 1)
    msg = Message(80, TextChannel(Guild()), 'valeu!', reference=reply_to(60))
    await engine.handle_message(msg)
    check('a direct reply may get no text answer', msg.replies == [], msg.replies)
    conv = await engine.store.get_by_handle(60)
    check('nothing stored', len(conv['data']['turns']) == 1)


async def test_failed_answer_is_still_reported_to_who_asked(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, answers=(EMPTY_ANSWER_FALLBACK,))
    msg = Message(80, TextChannel(Guild()), f'<@{BOT_ID}> oi')
    await engine.handle_message(msg)
    check('the fallback is sent on a direct mention', msg.replies and EMPTY_ANSWER_FALLBACK in msg.replies[0]['content'])


async def test_bare_mention_gets_the_hint(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    msg = Message(80, TextChannel(Guild()), f'<@{BOT_ID}>')
    await engine.handle_message(msg)
    check('hint instead of a turn', engine.turns == [] and 'Me mencione com uma pergunta' in msg.replies[0]['content'], msg.replies)


# --- Edits and deletes --------------------------------------------------------

async def test_deleting_the_question_deletes_the_answer(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    msg = Message(80, TextChannel(Guild()), f'<@{BOT_ID}> oi')
    await engine.handle_message(msg)
    await engine.message_deleted(80, 10, 1)
    check('answer deleted', ('delete', 81) in engine.partial_log, engine.partial_log)
    check('single-turn conversation gone', await engine.store.get_by_trigger(80) is None)
    check('and unreachable by reply', await engine.store.get_by_handle(81) is None)


async def test_deleting_a_later_question_keeps_the_conversation(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    channel = TextChannel(Guild())
    await engine.handle_message(Message(80, channel, f'<@{BOT_ID}> oi'))
    await engine.handle_message(Message(90, channel, 'e depois?', reference=reply_to(81)))
    await engine.message_deleted(90, 10, 1)
    conv = await engine.store.get_by_trigger(80)
    check('only that turn removed', conv and len(conv['data']['turns']) == 1, conv)
    check('its answer deleted', ('delete', 91) in engine.partial_log, engine.partial_log)


async def test_deleting_part_of_a_burst_keeps_the_answer(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    engine.bursts.delay = 0.03
    channel = TextChannel(Guild())
    await engine.handle_message(Message(80, channel, f'<@{BOT_ID}> ei'))
    await engine.handle_message(Message(81, channel, 'tudo bem?'))
    await asyncio.sleep(0.1)
    await engine.message_deleted(80, 10, 1)
    conv = await engine.store.get_by_trigger(81)
    check('answer kept', engine.partial_log == [], engine.partial_log)
    check('turn keeps the remaining message', conv['data']['turns'][0]['message_ids'] == ['81'], conv['data']['turns'][0])


async def test_editing_the_question_re_answers_in_place(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, answers=('primeira', 'segunda'))
    channel = TextChannel(Guild())
    msg = Message(80, channel, f'<@{BOT_ID}> qual a versão?')
    channel.messages = [msg]
    engine.bot.get_channel = lambda cid: channel
    await engine.handle_message(msg)
    msg.content = f'<@{BOT_ID}> qual a versão do paper?'
    await engine.message_edited(80, 10, 1, msg.content)
    check('re-ran with the new text and no history', len(engine.turns) == 2 and engine.turns[1].question == 'qual a versão do paper?' and engine.turns[1].history == [], engine.turns[-1].question)
    edits = [e for e in engine.partial_log if e[0] == 'edit']
    check('the old answer was edited, not a new message', edits and edits[0][1] == 81 and edits[0][2]['content'] == 'segunda', engine.partial_log)
    conv = await engine.store.get_by_handle(81)
    check('still one turn, now the new one', len(conv['data']['turns']) == 1 and conv['data']['turns'][0]['answer'] == 'segunda', conv)


async def test_unchanged_or_late_edits_are_ignored(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    channel = TextChannel(Guild())
    msg = Message(80, channel, f'<@{BOT_ID}> oi')
    channel.messages = [msg]
    engine.bot.get_channel = lambda cid: channel
    await engine.handle_message(msg)
    await engine.message_edited(80, 10, 1, msg.content)
    check('embed unfurl (same text) ignored', len(engine.turns) == 1)
    monkeypatch.setattr(chat_engine, 'CHAT_EDIT_WINDOW_SECONDS', 0)
    msg.content = f'<@{BOT_ID}> outra coisa'
    await engine.message_edited(80, 10, 1, msg.content)
    check('edits outside the window ignored', len(engine.turns) == 1)


# --- Slash commands -----------------------------------------------------------

class FakeInteraction:
    def __init__(self, channel, user):
        from types import SimpleNamespace as NS
        self.channel = channel
        self.guild = channel.guild
        self.user = user
        self.created_at = NOW
        self.events = []
        self.sent_ids = []
        self.response = NS(defer=self._defer, send_message=self._send_message)
        self.followup = NS(send=self._followup)

    async def _defer(self, **kw):
        self.events.append(('defer', kw))

    async def _send_message(self, content=None, **kw):
        self.events.append(('send_message', content, kw))

    async def edit_original_response(self, **kw):
        self.events.append(('edit', kw.get('content')))

    async def original_response(self):
        return Message(700, self.channel, '')

    async def _followup(self, content=None, **kw):
        from types import SimpleNamespace as NS
        self.events.append(('followup', kw.get('content', content), kw))
        self.sent_ids.append(800 + len(self.events))
        return NS(id=self.sent_ids[-1])


async def test_slash_chat_answers_and_starts_a_conversation(tmp_path, monkeypatch):
    from chat_fakes import ASKER
    from core.profiles import ASK
    engine = make_engine(tmp_path, monkeypatch)
    itx = FakeInteraction(TextChannel(Guild()), ASKER)
    await engine.answer_interaction(itx, 'qual a melhor versão?')
    kinds = [e[0] for e in itx.events]
    check('deferred, status shown, answered', kinds[0] == 'defer' and ('edit', '💭 Pensando…') in itx.events and kinds[-1] == 'followup', itx.events)
    check('plain answer', itx.events[-1][1] == 'Reduz o view-distance.', itx.events[-1])
    check('no silence on slash commands', engine.turns[0].can_stay_silent is False)
    conv = await engine.store.get_by_handle(itx.sent_ids[-1])
    check('replying to the answer continues it', conv and conv['data']['profile'] == 'chat', conv)

    itx = FakeInteraction(TextChannel(Guild()), ASKER)
    await engine.answer_interaction(itx, 'como configuro o paper?', profile=ASK)
    check('/ask uses the docs-first profile', engine.turns[-1].profile is ASK)


async def test_slash_chat_in_a_thread_channel_names_the_thread(tmp_path, monkeypatch):
    from chat_fakes import ASKER
    engine = make_engine(tmp_path, monkeypatch, thread_name='Melhor versão do Paper')
    engine.services.settings.write(1, 10, 'threads', 'always')
    itx = FakeInteraction(TextChannel(Guild()), ASKER)
    await engine.answer_interaction(itx, 'qual a melhor versão?')
    check('status names the thread', ('edit', '🧵 Resposta na thread **Melhor versão do Paper**') in itx.events, itx.events)
    check('from the question and the answer', 'qual a melhor versão?' in engine.title_calls[0]['user']
          and 'Reduz o view-distance.' in engine.title_calls[0]['user'], engine.title_calls)


async def test_slash_chat_over_budget_is_told_ephemerally(tmp_path, monkeypatch):
    from chat_fakes import ASKER
    engine = make_engine(tmp_path, monkeypatch, budget=Budget(user_limit=1, user_window=300, guild_limit=9, guild_window=300))
    await engine.answer_interaction(FakeInteraction(TextChannel(Guild()), ASKER), 'oi')
    itx = FakeInteraction(TextChannel(Guild()), ASKER)
    await engine.answer_interaction(itx, 'oi de novo')
    kind, content, kw = itx.events[0]
    check('ephemeral notice instead of a turn', kind == 'send_message' and '⏳' in content and kw['ephemeral'], itx.events)
    check('one turn only', len(engine.turns) == 1)
