"""The question stands apart from its context.

The channel chatter rides in the same user message as the question, so it is
labelled as background, the prompt says to answer only what the person under
the ``[Por …]`` tag wrote, and a change of speaker is marked on the turn that
brings it.
"""

import datetime
from types import SimpleNamespace as NS

import core.runner as runner_mod
from core import context as context_mod
from core.conversations import CONTEXT_ONLY_NOTE, PRIOR_CONTEXT_HEADER, build_current_message, make_turn
from core.profiles import ASK, CHAT, SPARK
from core.prompts import CONTEXT_INSTRUCTIONS
from core.runner import run_turn
from core.turn import TurnContext

from chat_fakes import make_services
from helpers import as_stream, check

ALICE = {'id': '111', 'name': 'alice_h', 'display': 'Alice'}
BOB_USER = NS(id=222, name='bob_h', display_name='Bob', bot=False)
NOW = datetime.datetime(2026, 11, 13, 18, 0, tzinfo=datetime.timezone.utc)


class FakeChannel:
    id = 10
    name = 'geral'

    def __init__(self, messages):
        self._messages = messages  # newest first, like Discord's history()

    def history(self, *, limit, before=None):
        async def _iterate():
            for message in self._messages[:limit]:
                if before is None or message.id < before.id:
                    yield message
        return _iterate()


def _line(mid, content):
    author = NS(display_name='Ana', name='ana', id=333, bot=False)
    return NS(id=mid, content=content, attachments=[], embeds=[], author=author, created_at=NOW, reference=None)


def test_prompts_say_to_answer_only_the_tagged_message():
    for profile in (CHAT, ASK, SPARK):
        check(f'{profile.name} prompt explains how to read a turn', CONTEXT_INSTRUCTIONS in profile.system_prompt)
    check('bursts are the lines under the tag, not the channel messages',
          'As mensagens do canal que vêm antes da marca não contam' in CHAT.system_prompt)


def test_gap_chatter_is_labelled_as_context():
    text = build_current_message(
        'pesquisa isso', author=ALICE, ts=1.7e9, channel_id=10,
        prior_context=['[13/11/2026 15:06] Ana (@ana) <author_id=333>: saiu snapshot nova [msg_id=300]'],
    )['content']
    header = f'[{PRIOR_CONTEXT_HEADER} (channel_id=10) — {CONTEXT_ONLY_NOTE}]'
    check('gap header says it is only context', text.startswith(header), text)
    check('the question still comes last, under its tag', text.endswith('\npesquisa isso')
          and text.split('\n')[-2].startswith('[Por Alice'), text)


async def test_recent_window_is_labelled_as_context():
    channel = FakeChannel([_line(2, 'de novo?? kkkk'), _line(1, 'mexeram no combate')])
    recent = await context_mod.fetch_recent_channel_context(NS(), channel, before=NS(id=3), limit=5)
    lines = recent.text.split('\n')
    check('block opens with the background note', lines[0] == '<mensagens_recentes_do_canal>'
          and CONTEXT_ONLY_NOTE in lines[1], recent.text)
    check('then the messages, oldest first', lines[2].startswith('Histórico de #geral')
          and 'mexeram no combate' in lines[3] and 'de novo' in lines[4], recent.text)


def _current(messages):
    """Text of the last user message sent (the loop appends its answer to the same list)."""
    return [m for m in messages if isinstance(m, dict) and m.get('role') == 'user'][-1]['content']


async def test_request_marks_the_new_speaker(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, 'CHANNEL_CONTEXT_MESSAGES', 0)
    services = make_services(tmp_path)
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            msg = NS(role='assistant', content='ok', tool_calls=None)
            return as_stream(NS(choices=[NS(message=msg, finish_reason='stop')], usage=None))

    services.llm.client = NS(chat=NS(completions=Completions()))
    bot = NS(user=NS(id=900, name='quill', display_name='Quill'), services=services)
    history = [make_turn('qual plugin de proteção?', 'GriefPrevention.', author=ALICE, ts=1.7e9)]

    def last_request(user):
        calls.clear()
        return run_turn(TurnContext(
            bot=bot, profile=CHAT, question='e o combate?', user=user, created_at=NOW, history=list(history),
        ))

    await last_request(BOB_USER)
    messages = calls[0]['messages']
    current = _current(messages)
    tag = current.split('\n')[-2]
    check('a new speaker is marked on their turn', tag.startswith('[Por Bob (@bob_h)')
          and 'outra pessoa — antes falava Alice (@alice_h)' in tag, current)
    system = messages[0]['content']
    system = system if isinstance(system, str) else system[0]['text']
    check('system block lists participants so far', 'Participantes até agora: Alice (@alice_h).' in system)

    await last_request(NS(id=111, name='alice_h', display_name='Alice', bot=False))
    current = _current(calls[0]['messages'])
    check('the same speaker is not marked', 'outra pessoa' not in current, current)
