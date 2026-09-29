"""Who said what: turn tags, replayed per-turn blocks and memory attribution."""

from types import SimpleNamespace

from cogs.conversation_store import (
    author_stamp,
    build_conversation_block,
    build_current_message,
    build_history_messages,
    make_turn,
)
from cogs.memory import Memory, MemoryStore

from helpers import check

ALICE = {'id': '111', 'name': 'alice_h', 'display': 'Alice'}
BOB = {'id': '222', 'name': 'bob_h', 'display': 'Bob'}


def test_question_is_tagged_with_its_author():
    msg = build_current_message(
        'qual loader eu uso?', author=ALICE, ts=1.7e9,
        context_blocks='<mensagens_recentes_do_canal>\n[..] Carol (@carol) <author_id=333>: fabric\n</mensagens_recentes_do_canal>',
    )
    text = msg['content']
    tag, question = text.split('\n')[-2:]
    check('first turn: author tag right above the question',
          tag.startswith('[Por Alice (@alice_h) • author_id=111') and question == 'qual loader eu uso?', text)
    check('tag never says "now"', 'Agora' not in text, text)

    reply = build_current_message('e eu?', author=BOB, ts=1.7e9, reply_to=99)
    check('reply target stays in the tag', '↩ reply_to=99]' in reply['content'], reply['content'])
    check('no author, no tag', build_current_message('p', author=None, ts=None)['content'] == 'p')


def test_replayed_turns_stay_byte_identical_and_timeless():
    sent = []
    turns = []
    for i, (author, blocks) in enumerate([
        (ALICE, '<contexto>\n- Autor desta mensagem: Alice (@alice_h) id=111\n</contexto>'),
        (BOB, '<contexto>\n- Autor desta mensagem: Bob (@bob_h) id=222\n</contexto>'),
    ]):
        um = build_current_message(f'pergunta {i}', author=author, ts=1.7e9 + i, context_blocks=blocks)
        sent.append(um['content'])
        turns.append(make_turn(f'pergunta {i}', f'resposta {i}', author=author, ts=1.7e9 + i,
                               message_id=10 + i, user_message=um))
    replay = build_history_messages(turns)
    users = [m['content'] for m in replay if m['role'] == 'user']
    check('replayed user messages are exactly what was sent (prefix cache)', users == sent, users)
    check('each replayed turn keeps its own context block', 'Alice (@alice_h) id=111' in users[0]
          and 'Bob (@bob_h) id=222' in users[1])

    block = build_conversation_block(turns)
    check('conversation block names the last user message as the current one', 'ÚLTIMA mensagem de usuário' in block)
    check('conversation block says newer per-turn blocks win', 'vale o da mensagem mais recente' in block)


def _member(uid, name, display):
    return SimpleNamespace(id=uid, name=name, display_name=display, nick=None, global_name=None)


def _memory_cog(tmp_path, members):
    guild = SimpleNamespace(id=1, get_member=lambda uid: members.get(uid))
    cog = Memory.__new__(Memory)
    cog.bot = SimpleNamespace(get_guild=lambda gid: guild)
    cog.store = MemoryStore(str(tmp_path / 'memory.db'))
    cog.store.ensure()

    async def no_embeddings(texts):
        return None

    cog._embed_texts = no_embeddings
    return cog, guild


def test_attribution_rules(tmp_path):
    nyuu = _member(111, 'nyuu_dev', '✨Artur Silva✨')
    cog, guild = _memory_cog(tmp_path, {111: nyuu})
    err = cog._attribution_error
    check('person memory without about_user is rejected', err(guild, '', 'person', 'Artur gosta de Paper'))
    check('preference without about_user is rejected', err(guild, '', 'preference', 'Artur prefere Paper'))
    check('server fact without about_user is fine', err(guild, '', 'fact', 'O servidor usa Paper') == '')
    check('pronoun-only memory is rejected', err(guild, '111', 'fact', 'Ele prefere Paper a Fabric'))
    check('display name counts', err(guild, '111', 'fact', 'Artur Silva prefere Paper') == '')
    check('one word of the display name counts', err(guild, '111', 'fact', 'artur prefere Paper') == '')
    check('handle counts', err(guild, '111', 'fact', 'nyuu_dev prefere Paper') == '')
    check('part of another word does not count', err(guild, '111', 'fact', 'Arturo prefere Paper'))
    check('unknown member is not checked', err(guild, '999', 'fact', 'ele prefere Paper') == '')


async def test_memory_block_names_subjects_and_turn(tmp_path):
    alice = _member(111, 'alice_h', 'Alice')
    bob = _member(222, 'bob_h', 'Bob')
    cog, _guild = _memory_cog(tmp_path, {111: alice, 222: bob})
    for subject, content in [('111', 'Alice joga survival'), ('222', 'Bob joga survival'), ('', 'O servidor tem survival')]:
        cog.store.create_with_history(
            1, subject=subject, kind='fact', content=content, importance=3, pinned=False,
            pinned_by='', origin='', actor='t', actor_name='t', reason='t',
        )
    block = await cog.build_memory_block(
        1, 'survival', speaker_id='222', participant_ids={'111'},
        for_label=author_stamp(BOB, 1.7e9),
    )
    check('block says whose message it was built for', 'para a mensagem de Bob (@bob_h) •' in block, block)
    check('subjects carry handle and id', 'Sobre Alice (@alice_h) id=111' in block
          and 'Sobre Bob (@bob_h) id=222' in block, block)
    check('server memories are marked as not about a person', 'Do servidor (não se referem' in block, block)
