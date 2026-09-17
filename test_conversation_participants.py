"""Smoke tests for participant scope in shared conversations.

Run: python test_conversation_participants.py
"""

import asyncio
from types import SimpleNamespace

from cogs.conversation_store import (
    conversation_participant_ids,
    message_participant_infos,
)
from cogs.memory import Memory


PASS = 0
FAIL = 0


def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  ok  {name}')
    else:
        FAIL += 1
        print(f'FAIL  {name} {detail}')


def user(uid, name, *, bot=False):
    return SimpleNamespace(id=uid, name=name, display_name=name.title(), bot=bot)


async def main():
    alice = user(100, 'alice')
    bob = user(200, 'bob')
    carol = user(300, 'carol')

    class Channel:
        async def fetch_message(self, msg_id):
            check('unresolved reply target fetched', msg_id == 999)
            return SimpleNamespace(author=carol)

    message = SimpleNamespace(
        author=alice,
        mentions=[bob, user(400, 'quillbot', bot=True)],
        reference=SimpleNamespace(message_id=999, resolved=None),
        channel=Channel(),
    )
    incoming = await message_participant_infos(message)
    check(
        'message participants include author, mention and fetched reply target',
        {p['id'] for p in incoming} == {'100', '200', '300'}, incoming,
    )

    data = {
        'participants': [{'id': '100', 'name': 'alice', 'display': 'Alice'}],
        'turns': [{'author': {'id': '200', 'name': 'bob', 'display': 'Bob'}}],
    }
    check(
        'canonical scope combines persisted, legacy-turn and incoming participants',
        conversation_participant_ids(data, incoming) == {'100', '200', '300'},
    )

    guild = SimpleNamespace(
        get_member_named=lambda raw: None,
        members=[user(200, 'bruno')],
        id=1,
    )
    memory = Memory.__new__(Memory)
    result, _sources = await memory._exec_write(
        {'action': 'create', 'reason': 'teste', 'about_user': 'bruno'},
        guild=guild,
        actor_id='tester',
        actor_name='tester',
        origin=None,
        participants={'100'},
    )
    check('writes reject a guild member outside the conversation', result == memory._PRIVACY_MSG, result)

    class Store:
        def get_memory(self, guild_id, memory_id):
            return {'id': memory_id, 'subject': '200', 'status': 'active'}

    memory.store = Store()
    real_to_thread = asyncio.to_thread

    async def inline_to_thread(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    asyncio.to_thread = inline_to_thread
    try:
        entry, error = await memory._find_target(
            {'memory_id': 7}, guild, [''], allowed_subjects={'', '100'},
        )
    finally:
        asyncio.to_thread = real_to_thread
    check('memory-id mutations respect participant scope', entry is None and error == memory._PRIVACY_MSG, error)

    print(f'\n{PASS} passed, {FAIL} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
