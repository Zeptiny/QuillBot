"""An @mention continues the bot's recent conversation instead of starting a new one."""

from types import SimpleNamespace

from cogs import commands as commands_mod
from cogs.conversation_store import ConversationStore, find_recent_conversation

from helpers import check

BOT_ID = 900


def msg(mid, author_id, content='', *, reference=None):
    return SimpleNamespace(
        id=mid,
        author=SimpleNamespace(id=author_id, bot=author_id == BOT_ID, name=f'u{author_id}', display_name=f'U{author_id}'),
        content=content,
        reference=reference,
        attachments=[],
        mentions=[],
        guild=None,
    )


class Channel:
    """Channel whose history is given newest first, like discord.py's."""

    def __init__(self, messages):
        self.messages = messages
        self.calls = []
        self.id = 1

    def history(self, *, limit, before):
        self.calls.append((limit, before))
        items = self.messages[:limit]

        async def gen():
            for m in items:
                yield m
        return gen()


async def make_store(tmp_path):
    store = ConversationStore(str(tmp_path / 'conv.db'), kind='chat')
    await store.create(50, channel_id='1', data={'turns': [{'answer': 'old'}]})
    await store.update(50, {'turns': [{'answer': 'old'}, {'answer': 'new'}]}, new_handle_msg_id=60)
    return store


async def test_finds_newest_conversation_in_window(tmp_path):
    store = await make_store(tmp_path)
    channel = Channel([
        msg(70, 1, 'oi'),
        msg(65, BOT_ID, 'not a conversation'),
        msg(60, BOT_ID, 'latest answer'),
        msg(55, 2, 'chatter'),
        msg(50, BOT_ID, 'first answer'),
    ])
    trigger = msg(80, 1)
    found = await find_recent_conversation(store, channel, BOT_ID, before=trigger, limit=10)
    check('picks the newest bot message that is a conversation handle', found and found[0] == 60, found)
    check('returns that conversation', found[1]['conv_id'] == '50', found)
    check('window is the channel-context size, before the mention', channel.calls == [(10, trigger)], channel.calls)


async def test_no_conversation_outside_window(tmp_path):
    store = await make_store(tmp_path)
    channel = Channel([msg(i, 1, 'chatter') for i in range(79, 70, -1)] + [msg(60, BOT_ID, 'answer')])
    found = await find_recent_conversation(store, channel, BOT_ID, before=msg(80, 1), limit=5)
    check('bot message older than the window is ignored', found is None, found)
    found = await find_recent_conversation(store, channel, BOT_ID, before=msg(80, 1), limit=0)
    check('a zero window disables the lookup', found is None, found)


async def test_ignores_other_users_messages_with_handle_ids(tmp_path):
    store = await make_store(tmp_path)
    channel = Channel([msg(60, 1, 'same id but not from the bot')])
    found = await find_recent_conversation(store, channel, BOT_ID, before=msg(80, 1), limit=10)
    check('only the bot own messages count', found is None, found)


def make_cog(store, monkeypatch):
    cog = commands_mod.Commands.__new__(commands_mod.Commands)
    bot_user = SimpleNamespace(id=BOT_ID, display_name='QuillBot', mentioned_in=lambda m: True)
    cog.bot = SimpleNamespace(user=bot_user, get_cog=lambda name: None)
    cog.store = store
    cog.client = object()
    cog._followup_cd = {}
    cog.continued = []
    cog.ref_contexts = []

    async def fake_continue(message, handle_id, conv, *, reply_to, ref_context='', ref_image_urls=None):
        cog.continued.append((message.id, handle_id, conv['conv_id'], reply_to))
        cog.ref_contexts.append(ref_context)
    cog._continue_conversation = fake_continue
    monkeypatch.setattr(commands_mod, 'CHAT_MENTION_ENABLED', True)
    monkeypatch.setattr(commands_mod, 'CHANNEL_CONTEXT_MESSAGES', 10)
    return cog


async def test_mention_continues_recent_conversation(tmp_path, monkeypatch):
    store = await make_store(tmp_path)
    cog = make_cog(store, monkeypatch)
    monkeypatch.setattr(commands_mod, 'CHAT_MENTION_CONTINUE_ENABLED', True)
    mention = msg(80, 1, f'<@{BOT_ID}> e agora?')
    mention.channel = Channel([msg(70, 2, 'chatter'), msg(60, BOT_ID, 'answer')])
    await cog.on_message(mention)
    check('mention merged into the conversation, with no reply target', cog.continued == [(80, 60, '50', None)], cog.continued)


async def test_mention_without_recent_conversation_starts_new(tmp_path, monkeypatch):
    store = await make_store(tmp_path)
    cog = make_cog(store, monkeypatch)
    monkeypatch.setattr(commands_mod, 'CHAT_MENTION_CONTINUE_ENABLED', True)
    started = []

    async def stop_before_new_chat(*a, **kw):
        started.append(True)
        raise RuntimeError('new conversation path reached')
    monkeypatch.setattr(commands_mod.image_store, 'persist_images', stop_before_new_chat)
    mention = msg(80, 1, f'<@{BOT_ID}> oi')
    mention.channel = Channel([msg(70, 2, 'chatter')])
    try:
        await cog.on_message(mention)
    except RuntimeError:
        pass
    check('no conversation in the window leaves the new-conversation path', started and not cog.continued, cog.continued)


async def test_continuation_can_be_disabled(tmp_path, monkeypatch):
    store = await make_store(tmp_path)
    cog = make_cog(store, monkeypatch)
    monkeypatch.setattr(commands_mod, 'CHAT_MENTION_CONTINUE_ENABLED', False)

    async def stop(*a, **kw):
        raise RuntimeError('new conversation path reached')
    monkeypatch.setattr(commands_mod.image_store, 'persist_images', stop)
    mention = msg(80, 1, f'<@{BOT_ID}> e agora?')
    mention.channel = Channel([msg(60, BOT_ID, 'answer')])
    try:
        await cog.on_message(mention)
    except RuntimeError:
        pass
    check('disabled flag always starts a new conversation', cog.continued == [], cog.continued)


async def test_mention_replying_to_someone_continues_with_context(tmp_path, monkeypatch):
    store = await make_store(tmp_path)
    cog = make_cog(store, monkeypatch)
    monkeypatch.setattr(commands_mod, 'CHAT_MENTION_CONTINUE_ENABLED', True)
    quoted = msg(70, 2, 'o servidor caiu de novo')
    quoted.embeds = []
    mention = msg(
        80, 1, f'<@{BOT_ID}> e isso?',
        reference=SimpleNamespace(message_id=70, resolved=quoted),
    )
    mention.channel = Channel([quoted, msg(60, BOT_ID, 'answer')])
    await cog.on_message(mention)
    check('reply to another user still merges, replying to that message', cog.continued == [(80, 60, '50', 70)], cog.continued)
    check('quoted message rides along as context', 'o servidor caiu de novo' in cog.ref_contexts[0], cog.ref_contexts)
