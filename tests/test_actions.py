"""Action GIFs: /hug, /slap…, their providers and fallbacks, counts, the return button,
the bot answering actions done to it, and the send_action_gif tool."""

from types import SimpleNamespace as NS

import discord

from chat_fakes import ASKER, BOT_ID, OTHER, Guild, Message, TextChannel, make_engine, make_services
from cogs import actions as actions_mod
from core import action_gifs
from core.action_gifs import ACTIONS, ActionStore, Gif, GifPicker
from core.budget import Budget
from core.profiles import CHAT
from core.settings import SETTINGS, parse_value
from core.turn import TurnContext

from helpers import check

NEKOS_GIF = 'https://nekos.best/api/v2/hug/abc.gif'
OTAKU_GIF = 'https://cdn.otakugifs.xyz/gifs/lick/123.gif'
BOT_MEMBER = NS(id=BOT_ID, name='quillbot', display_name='QuillBot', bot=True, mention=f'<@{BOT_ID}>')
OTHER_BOT = NS(id=444, name='musica', display_name='Música', bot=True, mention='<@444>')


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status, data):
        self.status = status
        self.data = data

    async def __aenter__(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, content_type=None):
        return self.data


class FakeSession:
    """``aiohttp.ClientSession.get`` answering per host: (status, json) or an exception."""

    def __init__(self, **by_host):
        self.by_host = by_host
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        host = 'otaku' if 'otakugifs' in url else 'nekos'
        status, data = self.by_host.get(host, (500, None))
        return _Resp(status, data)


def nekos_ok(url=NEKOS_GIF, anime='Sword Art Online'):
    return 200, {'results': [{'anime_name': anime, 'url': url}]}


def otaku_ok(url=OTAKU_GIF):
    return 200, {'url': url}


class Sent:
    def __init__(self, mid, kwargs):
        self.id = mid
        self.kwargs = kwargs
        self.edits = []

    async def edit(self, **kwargs):
        self.edits.append(kwargs)
        return self


class Sender:
    def __init__(self):
        self.sent = []

    async def __call__(self, **kwargs):
        msg = Sent(7000 + len(self.sent), kwargs)
        self.sent.append(msg)
        return msg


class FakeResponse:
    def __init__(self):
        self.messages = []
        self.deferred = False

    async def send_message(self, content=None, **kwargs):
        self.messages.append((content, kwargs))

    async def defer(self, **kwargs):
        self.deferred = True


class FakeInteraction:
    def __init__(self, guild, channel, user):
        self.guild = guild
        self.channel = channel
        self.user = user
        self.response = FakeResponse()
        self.sender = Sender()
        self.followup = NS(send=self._followup)
        self.edited = []

    async def _followup(self, content=None, *, wait=False, **kwargs):
        if content is not None:
            kwargs['content'] = content
        return await self.sender(**kwargs)

    async def edit_original_response(self, **kwargs):
        self.edited.append(kwargs)


class FakeChat:
    def __init__(self):
        self.calls = []

    async def answer_action(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return True


class ActionChannel(TextChannel):
    def get_partial_message(self, mid):
        return NS(id=mid, channel=self, guild=self.guild)


def make_cog(tmp_path, *, session=None, chat=None):
    services = make_services(tmp_path)
    services.chat = chat
    bot = NS(services=services, user=BOT_MEMBER, tree=None)
    cog = actions_mod.Actions(bot)
    cog.store = ActionStore(str(tmp_path / 'actions.db'))
    cog.store.load()
    cog.gifs = GifPicker(session or FakeSession(nekos=nekos_ok(), otaku=otaku_ok()), cog.store)
    guild = Guild(members=[ASKER, OTHER, BOT_MEMBER, OTHER_BOT])
    guild.me = BOT_MEMBER
    channel = ActionChannel(guild)
    return cog, guild, channel


# ---------------------------------------------------------------------------
# Providers, fallbacks and the store
# ---------------------------------------------------------------------------

async def test_gif_comes_from_the_first_source_that_answers(tmp_path):
    store = ActionStore(str(tmp_path / 'a.db'))
    store.load()
    session = FakeSession(nekos=nekos_ok(), otaku=otaku_ok())
    gif = await GifPicker(session, store).pick(ACTIONS['hug'])
    check('nekos.best first, with the anime', gif == Gif(NEKOS_GIF, 'Sword Art Online'), gif)
    check('one request', len(session.calls) == 1, session.calls)

    session = FakeSession(nekos=(503, None), otaku=otaku_ok())
    gif = await GifPicker(session, store).pick(ACTIONS['hug'])
    check('falls back to otakugifs', gif == Gif(OTAKU_GIF), gif)
    check('asks otakugifs for that reaction as a GIF', session.calls[-1][1] == {'reaction': 'hug', 'format': 'gif'}, session.calls)


async def test_lick_only_asks_otakugifs(tmp_path):
    store = ActionStore(str(tmp_path / 'a.db'))
    store.load()
    session = FakeSession(nekos=nekos_ok(), otaku=otaku_ok())
    gif = await GifPicker(session, store).pick(ACTIONS['lick'])
    check('nekos.best has no lick', gif.url == OTAKU_GIF and all('otakugifs' in u for u, _ in session.calls), session.calls)


async def test_apis_down_reuse_a_recent_gif_then_text(tmp_path):
    store = ActionStore(str(tmp_path / 'a.db'))
    store.load()
    down = FakeSession(nekos=(0, OSError('connection refused')), otaku=(0, TimeoutError()))
    check('nothing seen yet: no GIF', await GifPicker(down, store).pick(ACTIONS['hug']) is None)
    await GifPicker(FakeSession(nekos=nekos_ok()), store).pick(ACTIONS['hug'])
    gif = await GifPicker(down, store).pick(ACTIONS['hug'])
    check('the GIF seen before is reused', gif == Gif(NEKOS_GIF, 'Sword Art Online'), gif)
    check('per action', await GifPicker(down, store).pick(ACTIONS['kiss']) is None)


async def test_gifs_from_other_hosts_are_refused(tmp_path):
    store = ActionStore(str(tmp_path / 'a.db'))
    store.load()
    session = FakeSession(nekos=nekos_ok(url='https://evil.example/hug.gif'), otaku=otaku_ok(url='http://cdn.otakugifs.xyz/x.gif'))
    check('no foreign host, no plain http', await GifPicker(session, store).pick(ACTIONS['hug']) is None)
    check('allowed hosts', action_gifs.allowed_url(NEKOS_GIF) and action_gifs.allowed_url(OTAKU_GIF))
    check('look-alike host refused', not action_gifs.allowed_url('https://notnekos.best/x.gif'))


def test_store_counts_per_pair_and_keeps_recent_gifs(tmp_path):
    store = ActionStore(str(tmp_path / 'a.db'), recent=3)
    store.load()
    check('first', store.bump(1, 222, 333, 'hug') == 1)
    check('second', store.bump(1, 222, 333, 'hug') == 2)
    check('other direction apart', store.bump(1, 333, 222, 'hug') == 1)
    check('other action apart', store.bump(1, 222, 333, 'slap') == 1)
    check('other server apart', store.bump(2, 222, 333, 'hug') == 1)
    for i in range(5):
        store.remember_gif('hug', Gif(f'https://nekos.best/{i}.gif'))
    urls = {store.recent_gif('hug').url for _ in range(40)}
    check('only the newest are kept', urls <= {f'https://nekos.best/{i}.gif' for i in (2, 3, 4)} and urls, urls)


# ---------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------

def test_render_with_and_without_a_gif():
    hug = ACTIONS['hug']
    content, embed = action_gifs.render(hug, ASKER, OTHER, Gif(NEKOS_GIF, 'K-On!'), 3)
    check('actor in bold, target mentioned', content == '🤗 **John** abraçou <@333>!', content)
    check('GIF in the embed', embed.image.url == NEKOS_GIF)
    check('count and anime in the footer', embed.footer.text == 'John já abraçou Ana 3 vezes · Anime: K-On!', embed.footer.text)
    content, embed = action_gifs.render(ACTIONS['slap'], ASKER, OTHER, None, 1, returned=True)
    check('text only when there is no GIF', embed is None and content == '🖐️ **John** deu um tapa em <@333> de volta!\n-# Primeira vez que John deu um tapa em Ana', content)
    content, _ = action_gifs.render(hug, BOT_MEMBER, ASKER, None, 1, note='Ninguém merece fazer isso sozinho')
    check('note first', content.startswith('🤗 Ninguém merece fazer isso sozinho: **QuillBot** abraçou <@222>!'), content)
    sneaky = NS(id=5, display_name='*__x__*', name='x')
    content, _ = action_gifs.render(hug, sneaky, OTHER, None, 1)
    check('names cannot break the markdown', '\\*\\_\\_x\\_\\_\\*' in content, content)


def test_every_action_has_a_source_and_a_return_label():
    check('fifteen actions', len(ACTIONS) == 15, list(ACTIONS))
    for action in ACTIONS.values():
        check(f'{action.key} sources', action.sources and all(p in action_gifs.PROVIDERS for p, _ in action.sources))
    check('labels by kind', (ACTIONS['hug'].return_label, ACTIONS['slap'].return_label, ACTIONS['poke'].return_label)
          == ('Retribuir', 'Revidar', 'Devolver'))


def test_find_member():
    guild = Guild(members=[ASKER, OTHER, NS(id=123456789012345678, name='joao', display_name='João Silva', bot=False)])
    check('by id', action_gifs.find_member(guild, '123456789012345678').name == 'joao')
    check('by mention', action_gifs.find_member(guild, '<@!123456789012345678>').name == 'joao')
    check('by @name', action_gifs.find_member(guild, '@ana') is OTHER)
    check('accent and case insensitive', action_gifs.find_member(guild, 'joao silva').name == 'joao')
    check('unknown', action_gifs.find_member(guild, 'ninguém') is None)


def test_question_for_the_bot():
    q = action_gifs.action_question(ACTIONS['slap'], ASKER)
    check('says who did what', q.startswith('[John usou /slap em você') and 'send_action_gif' in q and '[NO_REPLY]' in q, q)
    q = action_gifs.action_question(ACTIONS['hug'], ASKER, returned=True, can_send=False)
    check('a return is said to be one', 'devolveu o que você fez com /hug' in q and 'send_action_gif' not in q, q)


def test_setting_and_profile():
    check('on by default', SETTINGS['actions'].default is True and SETTINGS['actions'].scope == 'channel')
    check('sim/não', parse_value(SETTINGS['actions'], 'não') == 'false')
    check('chat offers the tool', 'send_action_gif' in CHAT.tools)


# ---------------------------------------------------------------------------
# Commands, the return button and the bot answering
# ---------------------------------------------------------------------------

async def test_user_agent(tmp_path, monkeypatch):
    cog, _, _ = make_cog(tmp_path)
    cog.bot.application_id = 1234
    check('bot name and invite link', cog.user_agent() == 'quillbot (https://discord.com/oauth2/authorize?client_id=1234)', cog.user_agent())
    monkeypatch.setattr(actions_mod, 'ACTIONS_USER_AGENT', 'Quill (dev@example.com)')
    check('override', cog.user_agent() == 'Quill (dev@example.com)')


async def test_commands_are_built_from_the_table(tmp_path):
    cog, _, _ = make_cog(tmp_path)
    command = cog.make_command(ACTIONS['hug'])
    check('name and description', command.name == 'hug' and command.description == '🤗 Dê um abraço em alguém')
    check('one member option', [(p.name, p.type) for p in command.parameters] == [('usuario', discord.AppCommandOptionType.user)])
    check('servers only, marked as an action', command.guild_only and command.extras == {'action': 'hug'})


async def test_slash_command_posts_the_gif_with_a_return_button(tmp_path, monkeypatch):
    monkeypatch.setattr(actions_mod, 'ACTIONS_PING_TARGET', True)
    cog, guild, channel = make_cog(tmp_path)
    inter = FakeInteraction(guild, channel, ASKER)
    await cog.use(inter, ACTIONS['hug'], OTHER)
    check('deferred, then sent', inter.response.deferred and len(inter.sender.sent) == 1)
    kwargs = inter.sender.sent[0].kwargs
    check('content', kwargs['content'] == '🤗 **John** abraçou <@333>!', kwargs['content'])
    check('gif', kwargs['embed'].image.url == NEKOS_GIF)
    check('pings only the target', [u.id for u in kwargs['allowed_mentions'].users] == [333] and not kwargs['allowed_mentions'].everyone)
    view = kwargs['view']
    check('return button', isinstance(view, actions_mod.ReturnView) and view.children[0].label == 'Retribuir')
    check('the view knows its message', view.message is inter.sender.sent[0])

    # Another person can't press it; the target can, once, and it goes back the other way.
    stranger = FakeInteraction(guild, channel, NS(id=999, display_name='Zé'))
    await view.pressed(stranger)
    check('only the target', stranger.response.messages[0][0] == 'Só Ana pode retribuir.' and stranger.response.messages[0][1]['ephemeral'])
    press = FakeInteraction(guild, channel, OTHER)
    await view.pressed(press)
    back = press.sender.sent[0].kwargs
    check('done back', back['content'] == '🤗 **Ana** abraçou <@222> de volta!', back['content'])
    check('a return has no button', 'view' not in back)
    check('button removed', press.edited == [{'view': None}])
    again = FakeInteraction(guild, channel, OTHER)
    await view.pressed(again)
    check('only once', again.response.messages[0][0] == 'Já foi.' and not again.sender.sent)
    check('counted both ways', cog.store.bump(1, 222, 333, 'hug') == 2 and cog.store.bump(1, 333, 222, 'hug') == 2)


async def test_no_ping_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(actions_mod, 'ACTIONS_PING_TARGET', False)
    cog, guild, channel = make_cog(tmp_path)
    send = Sender()
    await cog.perform(ACTIONS['pat'], actor=ASKER, target=OTHER, guild=guild, channel=channel, send=send)
    check('nobody pinged', send.sent[0].kwargs['allowed_mentions'].users is False, send.sent[0].kwargs['allowed_mentions'])


async def test_refusals_cooldown_and_self_actions(tmp_path, monkeypatch):
    monkeypatch.setattr(actions_mod, 'ACTIONS_COOLDOWN_SECONDS', 60)
    cog, guild, channel = make_cog(tmp_path)

    inter = FakeInteraction(guild, channel, ASKER)
    await cog.use(inter, ACTIONS['slap'], ASKER)
    check('no slapping yourself', inter.response.messages[0][1].get('ephemeral') and not inter.sender.sent, inter.response.messages)

    inter = FakeInteraction(guild, channel, ASKER)
    await cog.use(inter, ACTIONS['hug'], ASKER)
    content = inter.sender.sent[0].kwargs['content']
    check('a hug to yourself comes from the bot', content == '🤗 Ninguém merece fazer isso sozinho: **QuillBot** abraçou <@222>!', content)

    inter = FakeInteraction(guild, channel, ASKER)
    await cog.use(inter, ACTIONS['kiss'], OTHER)
    check('cooldown shared by every action', inter.response.messages and inter.response.messages[0][0].startswith('⏳'), inter.response.messages)

    cog.services.settings.write(1, 0, 'actions', 'false')
    inter = FakeInteraction(guild, channel, OTHER)
    await cog.use(inter, ACTIONS['kiss'], ASKER)
    check('off in the channel', 'desligados' in inter.response.messages[0][0] and not inter.sender.sent)


async def test_text_only_when_no_gif_at_all(tmp_path):
    cog, guild, channel = make_cog(tmp_path, session=FakeSession(nekos=(0, OSError()), otaku=(0, OSError())))
    send = Sender()
    await cog.perform(ACTIONS['bonk'], actor=ASKER, target=OTHER, guild=guild, channel=channel, send=send)
    kwargs = send.sent[0].kwargs
    check('still sent, without an embed', 'embed' not in kwargs and kwargs['content'].startswith('🔨 **John** deu um bonk em <@333>!'), kwargs)


async def test_doing_it_to_the_bot_gets_an_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(actions_mod, 'ACTIONS_BOT_REPLY', True)
    chat = FakeChat()
    cog, guild, channel = make_cog(tmp_path, chat=chat)
    send = Sender()
    await cog.perform(ACTIONS['slap'], actor=ASKER, target=BOT_MEMBER, guild=guild, channel=channel, send=send)
    check('no button for a bot', 'view' not in send.sent[0].kwargs)
    check('no ping for a bot', send.sent[0].kwargs['allowed_mentions'].users is False)
    check('the chat engine answers it', len(chat.calls) == 1, chat.calls)
    anchor, kwargs = chat.calls[0]
    check('anchored on the GIF, in the channel', anchor.id == send.sent[0].id and kwargs['channel'] is channel and kwargs['actor'] is ASKER)
    check('question says what happened', kwargs['question'].startswith('[John usou /slap em você'), kwargs['question'])

    send = Sender()
    await cog.perform(ACTIONS['slap'], actor=ASKER, target=OTHER_BOT, guild=guild, channel=channel, send=send)
    check('other bots: GIF only', len(chat.calls) == 1 and 'view' not in send.sent[0].kwargs)

    cog.services.settings.write(1, 0, 'respond', 'off')
    await cog.perform(ACTIONS['hug'], actor=ASKER, target=BOT_MEMBER, guild=guild, channel=channel, send=Sender())
    check('not where the bot does not converse', len(chat.calls) == 1)

    monkeypatch.setattr(actions_mod, 'ACTIONS_BOT_REPLY', False)
    cog.services.settings.write(1, 0, 'respond', None)
    await cog.perform(ACTIONS['hug'], actor=ASKER, target=BOT_MEMBER, guild=guild, channel=channel, send=Sender())
    check('ACTIONS_BOT_REPLY=false', len(chat.calls) == 1)


async def test_returning_the_bots_action_tells_it_so(tmp_path, monkeypatch):
    monkeypatch.setattr(actions_mod, 'ACTIONS_BOT_REPLY', True)
    chat = FakeChat()
    cog, guild, channel = make_cog(tmp_path, chat=chat)
    send = Sender()
    await cog.perform(ACTIONS['hug'], actor=BOT_MEMBER, target=ASKER, guild=guild, channel=channel, send=send)
    press = FakeInteraction(guild, channel, ASKER)
    await send.sent[0].kwargs['view'].pressed(press)
    check('hugged back', press.sender.sent[0].kwargs['content'] == '🤗 **John** abraçou <@900> de volta!', press.sender.sent[0].kwargs['content'])
    check('the bot hears it as a return', 'devolveu o que você fez com /hug' in chat.calls[0][1]['question'])


# ---------------------------------------------------------------------------
# The send_action_gif tool
# ---------------------------------------------------------------------------

def tool_ctx(guild, channel):
    return TurnContext(bot=None, profile=CHAT, question='dá um abraço na Ana', user=ASKER, guild=guild, channel=channel)


async def test_tool_sends_once_per_answer(tmp_path):
    cog, guild, channel = make_cog(tmp_path)
    ctx = tool_ctx(guild, channel)
    result = await cog.run_tool(ctx, {'action': 'hug', 'user': '333'})
    check('posted in the channel by the bot', channel.sent and channel.sent[0].kwargs['content'] == '🤗 **QuillBot** abraçou <@333>!', channel.sent)
    check('the target can hug back', isinstance(channel.sent[0].kwargs['view'], actions_mod.ReturnView))
    check('tells the model not to describe it', result.startswith('GIF enviado no canal: QuillBot abraçou Ana') and '[NO_REPLY]' in result, result)
    again = await cog.run_tool(ctx, {'action': 'slap', 'user': '333'})
    check('once per answer', 'já mandou' in again and len(channel.sent) == 1, again)


async def test_tool_refusals(tmp_path):
    cog, guild, channel = make_cog(tmp_path)
    check('unknown action', 'Ação desconhecida' in await cog.run_tool(tool_ctx(guild, channel), {'action': 'dance', 'user': '333'}))
    check('unknown person', 'Não encontrei' in await cog.run_tool(tool_ctx(guild, channel), {'action': 'hug', 'user': 'ninguém'}))
    check('not itself', 'si mesmo' in await cog.run_tool(tool_ctx(guild, channel), {'action': 'hug', 'user': str(BOT_ID)}))
    cog.services.settings.write(1, 0, 'actions', 'false')
    check('off in the channel', 'desligados' in await cog.run_tool(tool_ctx(guild, channel), {'action': 'hug', 'user': '333'}))
    check('nothing sent', not channel.sent)


# ---------------------------------------------------------------------------
# The chat engine's answer to an action
# ---------------------------------------------------------------------------

async def test_engine_answers_the_action_and_starts_a_conversation(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, answers=('Ai! Por que isso? 😭',))
    guild = Guild()
    channel = TextChannel(guild)
    gif_message = Message(800, channel, '🖐️ **John** deu um tapa em <@900>!', author=BOT_MEMBER)
    question = action_gifs.action_question(ACTIONS['slap'], ASKER)
    answered = await engine.answer_action(gif_message, actor=ASKER, guild=guild, channel=channel, question=question)
    check('answered', answered and gif_message.replies and gif_message.replies[0]['content'] == 'Ai! Por que isso? 😭', gif_message.replies)
    ctx = engine.turns[0]
    check('the turn is the actor\'s', ctx.user is ASKER and ctx.question == question and ctx.message is gif_message)
    check('reactions default to the GIF', ctx.message.id == 800)
    conv = await engine.store.get_by_handle(801)
    check('replying continues it', conv and conv['data']['turns'][0]['author']['id'] == '222', conv)


async def test_engine_action_silent_or_over_budget(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, answers=('[NO_REPLY]',))
    guild = Guild()
    channel = TextChannel(guild)
    gif_message = Message(800, channel, '🤗', author=BOT_MEMBER)
    check('may stay silent', not await engine.answer_action(gif_message, actor=ASKER, guild=guild, channel=channel, question='q'))
    check('nothing sent', not gif_message.replies)

    (tmp_path / 'b').mkdir()
    engine = make_engine(tmp_path / 'b', monkeypatch, budget=Budget(user_limit=1, user_window=60, guild_limit=10, guild_window=60))
    engine.budget.try_spend(ASKER.id, guild.id)
    check('over budget: no turn', not await engine.answer_action(gif_message, actor=ASKER, guild=guild, channel=channel, question='q'))
    check('no turn run', not engine.turns)
