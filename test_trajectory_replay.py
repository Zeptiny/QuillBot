"""Standalone smoke test for trajectory capture + verbatim replay.

Exercises: serialize_trajectory round-trip (tool pairing invariants, reasoning
stripping), validate_trajectory fallbacks, make_turn capture (+ disable/oversize
guards), stepped window hysteresis, verbatim replay with append-only growth
(prefix-cache stability), SQLite persistence of captured turns, and
apply_cache_control. No Discord or LLM calls.

Run: python3 test_trajectory_replay.py
"""

import asyncio
import shutil
import tempfile
from types import SimpleNamespace

from cogs import conversation_store as cs
from cogs.conversation_store import (
    ConversationStore,
    apply_cache_control,
    build_history_messages,
    make_turn,
    message_text,
    stepped_floor,
    validate_trajectory,
)
from cogs.utils import run_tool_loop, serialize_trajectory

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  ok  {name}')
    else:
        FAIL += 1
        print(f'FAIL  {name}  {detail}')


def fake_assistant(content=None, tool_calls=None, reasoning='chain-of-thought'):
    """Mimic an OpenAI SDK ChatCompletionMessage, with provider extras."""
    tcs = None
    if tool_calls:
        tcs = [
            SimpleNamespace(
                id=f'call_{i}',
                type='function',
                function=SimpleNamespace(name=name, arguments=args),
            )
            for i, (name, args) in enumerate(tool_calls)
        ]
    return SimpleNamespace(role='assistant', content=content, tool_calls=tcs, reasoning=reasoning)


def fake_response(message, finish_reason='stop'):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=None,
    )


def tool_msg(call_id, content):
    return {'role': 'tool', 'tool_call_id': call_id, 'content': content}


AUTHOR = {'id': '42', 'name': 'nyuu', 'display': 'Nyuu'}


def captured_turn(i, *, answer=None, trajectory=None, big=False):
    um = {'role': 'user', 'content': f'[Agora — Nyuu]\nQuestion {i}'}
    traj = trajectory if trajectory is not None else [
        {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': f'c{i}', 'type': 'function',
            'function': {'name': 'web_search', 'arguments': '{"q": "t%d"}' % i},
        }]},
        tool_msg(f'c{i}', 'x' * 200 if big else f'result {i}'),
        {'role': 'assistant', 'content': f'answer {i}'},
    ]
    return make_turn(
        f'Question {i}', answer if answer is not None else f'answer {i}',
        author=AUTHOR, ts=1000.0 + i,
        user_message=um, trajectory=traj,
    )


def legacy_turn(i):
    return make_turn(f'Question {i}', f'answer {i}', author=AUTHOR, ts=1000.0 + i)


def test_serialize_and_validate():
    print('serialize_trajectory / validate_trajectory')
    asst = fake_assistant(tool_calls=[('web_search', '{"q":"lag"}')])
    ser = serialize_trajectory([asst, tool_msg('call_0', 'hits'), fake_assistant('done!')])
    check('keeps tool_calls + ids', ser[0]['tool_calls'][0]['id'] == 'call_0', str(ser[0]))
    check('strips reasoning extras', 'reasoning' not in ser[0] and 'reasoning' not in ser[2], str(ser))
    check('final answer kept', ser[-1] == {'role': 'assistant', 'content': 'done!'}, str(ser[-1]))

    ok = validate_trajectory(ser)
    check('round-trip validates', ok is not None and len(ok) == 3, str(ok))
    check('dangling tool_call rejected', validate_trajectory([ser[0]]) is None)
    check('orphan tool result rejected', validate_trajectory([tool_msg('nope', 'x')]) is None)
    check('unknown role rejected', validate_trajectory([{'role': 'system', 'content': 'x'}]) is None)
    check('empty trajectory is None', validate_trajectory([]) is None)
    check('leading user nudge dropped from capture', serialize_trajectory([
        {'role': 'user', 'content': 'Responda agora…'},
        fake_assistant('ok'),
    ]) == [{'role': 'assistant', 'content': 'ok'}])
    parallel = serialize_trajectory([fake_assistant(tool_calls=[
        ('a', '{}'), ('b', '{}'),
    ])])
    check('parallel tool_calls preserved', len(parallel[0]['tool_calls']) == 2)


def test_make_turn_capture():
    print('make_turn capture')
    turn = captured_turn(1)
    check('user_message stored as plain text', turn['user_message'] == '[Agora — Nyuu]\nQuestion 1')
    check('trajectory stored', len(turn['trajectory']) == 3)

    parts_msg = {'role': 'user', 'content': [
        {'type': 'text', 'text': 'analyze'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}},
    ]}
    check(
        'message_text extracts text part only',
        message_text(parts_msg) == 'analyze' and message_text({'role': 'user', 'content': 'plain'}) == 'plain',
    )
    check('parts message stores no base64', 'base64' not in repr(make_turn(
        'q', 'a', author=AUTHOR, ts=1.0, user_message=parts_msg,
    )))

    old_max = cs.CONVERSATIONS_TRAJECTORY_MAX_CHARS
    try:
        cs.CONVERSATIONS_TRAJECTORY_MAX_CHARS = 10
        check('oversized trajectory dropped', 'trajectory' not in captured_turn(1, big=True))
    finally:
        cs.CONVERSATIONS_TRAJECTORY_MAX_CHARS = old_max

    old_enabled = cs.CONVERSATIONS_TRAJECTORY_ENABLED
    try:
        cs.CONVERSATIONS_TRAJECTORY_ENABLED = False
        check('disabled drops trajectory', 'trajectory' not in captured_turn(1))
    finally:
        cs.CONVERSATIONS_TRAJECTORY_ENABLED = old_enabled


def test_stepped_floor():
    print('stepped_floor hysteresis')
    cases = {6: 0, 7: 3, 9: 3, 10: 6, 12: 6, 13: 9}
    for count, want in cases.items():
        got = stepped_floor(count, window=6, step=3)
        check(f'floor({count})={want}', got == want, f'got {got}')
    check('slice never exceeds window', all(
        n - stepped_floor(n, 6, 3) <= 6 for n in range(0, 40)
    ))
    check('step > window never empties nor exceeds window', all(
        1 <= n - stepped_floor(n, 6, 20) <= max(1, min(6, n)) for n in range(1, 40)
    ))


def test_replay_and_cache_stability():
    print('build_history_messages replay + cache stability')
    kwargs = dict(max_turns=16, trajectory_turns=6, trajectory_step=3)

    turns = [captured_turn(i) for i in range(4)]
    msgs = build_history_messages(turns, **kwargs)
    roles = [m['role'] for m in msgs]
    check('verbatim turn replayed with trajectory', roles == ['user', 'assistant', 'tool', 'assistant'] * 4, str(roles))
    check('user text is byte-identical', msgs[0]['content'] == '[Agora — Nyuu]\nQuestion 0')

    # Append-only growth while the stepped floor does not move: the previous
    # request's prefix is a strict prefix of the next one (cache hit).
    history = [captured_turn(i) for i in range(6)]
    m5 = build_history_messages(history[:5], **kwargs)
    m6 = build_history_messages(history, **kwargs)
    check('len5 -> len6 append-only', m5 == m6[:len(m5)])
    m8 = build_history_messages(history + [captured_turn(6), captured_turn(7)], **kwargs)
    m9 = build_history_messages(history + [captured_turn(6), captured_turn(7), captured_turn(8)], **kwargs)
    check('len8 -> len9 append-only (same step bucket)', m8 == m9[:len(m8)])

    # Mixed legacy + captured: legacy render stays compact and prefixed.
    mixed = [legacy_turn(0), legacy_turn(1), captured_turn(2)]
    msgs = build_history_messages(mixed, **kwargs)
    check('legacy turns keep attribution prefix', msgs[0]['content'].startswith('[Por Nyuu'), msgs[0]['content'][:40])
    check('captured turn replays verbatim after legacy', msgs[4]['content'] == '[Agora — Nyuu]\nQuestion 2',
          str([m.get('content') for m in msgs[:5]]))

    # Broken trajectory falls back to user text + plain answer.
    broken = captured_turn(3)
    broken['trajectory'] = [broken['trajectory'][0]]  # dangling tool_call
    msgs = build_history_messages([broken], **kwargs)
    check(
        'dangling trajectory falls back to Q/A',
        [m['role'] for m in msgs] == ['user', 'assistant'] and msgs[1]['content'] == 'answer 3',
        str(msgs),
    )

    # Char budget collapses verbatim turns in step-sized chunks.
    big = [captured_turn(i, big=True) for i in range(3)]
    msgs = build_history_messages(big, trajectory_turns=6, trajectory_step=3, trajectory_max_chars=500)
    check('budget over window -> all compact', [m['role'] for m in msgs] == ['user', 'assistant'] * 3, str(len(msgs)))
    msgs = build_history_messages(big[:1], trajectory_turns=6, trajectory_step=3, trajectory_max_chars=10)
    check('single turn over budget -> compact', [m['role'] for m in msgs] == ['user', 'assistant'], str(len(msgs)))

    # Replay window beyond max_turns drops turns in step-aligned batches.
    many = [captured_turn(i) for i in range(20)]
    msgs = build_history_messages(many, **kwargs)
    verbatim_qs = [m['content'] for m in msgs if m['role'] == 'user' and isinstance(m['content'], str)]
    check('long history replayed within window', len(verbatim_qs) == 14, f'{len(verbatim_qs)} user msgs')
    check('oldest turn dropped (not compact-rendered)', not any('Question 0' in c for c in verbatim_qs))

    # Defensive branch: trajectory not ending with an assistant message.
    mid = captured_turn(4)
    mid['trajectory'] = mid['trajectory'][:2]  # ends with tool result
    msgs = build_history_messages([mid], **kwargs)
    check('trajectory ending mid-loop gets stored answer appended',
          [m['role'] for m in msgs] == ['user', 'assistant', 'tool', 'assistant'], str([m['role'] for m in msgs]))

    # Corrupt stored shapes degrade to the compact rendering, never crash.
    corrupt = captured_turn(5)
    corrupt['user_message'] = {'text': 'not a string'}
    corrupt2 = captured_turn(6)
    corrupt2['images'] = 'img:not-a-list'
    corrupt3 = captured_turn(7)
    corrupt3['trajectory'] = [{'role': 'assistant', 'content': None, 'tool_calls': [5]}]
    msgs = build_history_messages([corrupt, corrupt2, corrupt3], **kwargs)
    check('corrupt turns degrade without crashing',
          [m['role'] for m in msgs] == ['user', 'assistant', 'user', 'assistant', 'tool', 'assistant', 'user', 'assistant'],
          str([m['role'] for m in msgs]))


def test_verbatim_image_budget():
    print('verbatim replay image budget + markers')
    real_is_ref, real_part = cs.image_store.is_image_ref, cs.image_store.image_part
    real_marker = cs.image_store.image_marker
    try:
        cs.image_store.is_image_ref = lambda v: isinstance(v, str) and v.startswith('img:')
        cs.image_store.image_part = lambda ref: {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{ref}'}}
        cs.image_store.image_marker = lambda n: f'[marker:{n}]'
        turns = []
        for i in range(3):
            t = captured_turn(i)
            t['images'] = [f'img:{i}.jpg', f'img:{i}b.jpg']
            turns.append(t)
        msgs = build_history_messages(
            turns, max_turns=16, trajectory_turns=6, trajectory_step=3, max_images=2,
        )
        parts = [m['content'] for m in msgs if isinstance(m.get('content'), list)]
        inlined = sum(1 for c in parts for p in c if p.get('type') == 'image_url')
        check('image budget enforced across verbatim window', inlined == 2, f'{inlined} inlined')
        user_texts: list[str] = []
        for m in msgs:
            if m['role'] != 'user':
                continue
            c = m['content']
            if isinstance(c, str):
                user_texts.append(c)
            elif isinstance(c, list):
                user_texts.extend(str(p.get('text', '')) for p in c if isinstance(p, dict))
        check('overflow images get text marker', any('[marker:' in t for t in user_texts), str(user_texts)[:160])
        again = build_history_messages(
            turns, max_turns=16, trajectory_turns=6, trajectory_step=3, max_images=2,
        )
        check('replay deterministic', again == msgs)
    finally:
        cs.image_store.is_image_ref, cs.image_store.image_part = real_is_ref, real_part
        cs.image_store.image_marker = real_marker


def test_apply_cache_control():
    print('apply_cache_control')
    msg = {'role': 'user', 'content': 'hello'}
    marked = apply_cache_control(msg)
    check('str wrapped with breakpoint', marked['content'][-1]['cache_control'] == {'type': 'ephemeral'})
    check('original untouched', msg['content'] == 'hello')
    check('idempotent', apply_cache_control(marked) == marked)
    tool_like = {'role': 'assistant', 'content': None, 'tool_calls': []}
    check('non-text content passthrough', apply_cache_control(tool_like) is tool_like)


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _fake_client(responses):
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(responses)))


async def test_run_tool_loop_capture():
    print('run_tool_loop trajectory capture')
    tc_msg = fake_assistant(tool_calls=[('web_search', '{"q":"lag"}')], reasoning='thinking...')
    client = _fake_client([
        fake_response(tc_msg, 'tool_calls'),
        fake_response(fake_assistant('Resposta final!'), 'stop'),
    ])
    messages = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'pergunta'}]

    async def exec_tool(name, args):
        return f'resultado de {name}', []

    answer, sources, trajectory = await run_tool_loop(
        client, 'model-x', messages, [{'type': 'function', 'function': {'name': 'web_search'}}],
        exec_tool,
    )
    check('answer returned', answer == 'Resposta final!' and sources == [])
    check('trajectory = tc + tool + answer', [
        (m['role'], m.get('tool_calls') is not None) for m in trajectory
    ] == [('assistant', True), ('tool', False), ('assistant', False)], str(trajectory))
    check('reasoning stripped from capture', all('reasoning' not in m for m in trajectory))
    check('capture is JSON-safe', all(
        isinstance(m.get('content'), (str, type(None))) for m in trajectory
    ))


async def test_run_tool_loop_retry_capture():
    print('run_tool_loop empty-answer retry capture')
    tc_msg = fake_assistant(tool_calls=[('web_search', '{}')])
    client = _fake_client([
        fake_response(tc_msg, 'tool_calls'),
        fake_response(fake_assistant(''), 'stop'),
        fake_response(fake_assistant('Ok, respondendo.'), 'stop'),
    ])
    messages = [{'role': 'user', 'content': 'q'}]

    async def exec_tool(name, args):
        return 'r', []

    answer, _, trajectory = await run_tool_loop(client, 'model-x', messages, [], exec_tool)
    check('retry answer used', answer == 'Ok, respondendo.')
    check('nudge + answer captured', trajectory[-2]['role'] == 'user' and trajectory[-1] == {
        'role': 'assistant', 'content': 'Ok, respondendo.'
    }, str(trajectory[-2:]))


async def test_store_roundtrip():
    print('ConversationStore persistence round-trip')
    tmp = tempfile.mkdtemp()
    try:
        store = ConversationStore(f'{tmp}/convs.db', kind='chat')
        turn = captured_turn(1)
        await store.create(
            '100', guild_id='1', channel_id='2',
            data={'turns': [turn], 'participants': [AUTHOR], 'origin': {}, 'started_ts': 1.0},
        )
        conv = await store.get_by_handle('100')
        stored = conv['data']['turns'][0]
        check('trajectory survives JSON storage', stored.get('trajectory') == turn['trajectory'])
        check('user_message survives JSON storage', stored.get('user_message') == turn['user_message'])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def main():
    test_serialize_and_validate()
    test_make_turn_capture()
    test_stepped_floor()
    test_replay_and_cache_stability()
    test_apply_cache_control()
    await test_run_tool_loop_capture()
    await test_run_tool_loop_retry_capture()
    await test_store_roundtrip()
    test_verbatim_image_budget()
    print(f'\n{PASS} passed, {FAIL} failed')
    raise SystemExit(1 if FAIL else 0)


if __name__ == '__main__':
    asyncio.run(main())
