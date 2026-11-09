"""A live voice call (VoiceCall) with Discord and the realtime model faked."""

import asyncio
import base64
from collections import Counter
from types import SimpleNamespace as NS

import discord
import numpy as np

from core.prompts import VOICE_SILENCE_TOOL_NAME, voice_greeting, voice_instructions
from core.settings import SETTINGS
from core.voice_call import VoiceCall, mentions, user_line

from helpers import check

ANA, BETO, MUSIC_BOT = 1, 2, 99
NAMES = {ANA: 'Ana', BETO: 'Beto', MUSIC_BOT: 'Rádio'}


class FakeRealtime:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.audio = bytearray()
        self.frames = 0
        self.sent = []
        self.cancels = 0
        self.deleted = []
        self.closed = False

    async def events(self):
        while True:
            event = await self.queue.get()
            if event is None:
                return
            yield event

    def emit(self, kind, **fields):
        self.queue.put_nowait({'type': kind, **fields})

    async def append_audio(self, pcm):
        if self.closed:
            raise ConnectionError
        self.audio += pcm
        self.frames += 1

    async def add_message(self, role, text):
        self.sent.append((role, text))

    async def create_response(self):
        self.sent.append('response.create')

    async def cancel_response(self):
        self.cancels += 1

    async def delete_item(self, item_id):
        self.deleted.append(item_id)

    async def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put_nowait(None)


class FakeMessage:
    def __init__(self, content):
        self.content = content
        self.edits = []

    async def edit(self, content, **kw):
        self.content = content
        self.edits.append(content)


class FakeChannel:
    name = 'Geral'
    mention = '<#5>'

    def __init__(self):
        self.members = [NS(bot=False), NS(bot=False), NS(bot=True)]
        self.messages = []
        self.forbidden = False
        self.attempts = 0

    async def send(self, text, **kw):
        self.attempts += 1
        if self.forbidden:
            raise discord.Forbidden(NS(status=403, reason='Forbidden'), 'Missing Permissions')
        msg = FakeMessage(text)
        self.messages.append(msg)
        return msg

    @property
    def posts(self):
        return [m.content for m in self.messages]


class FakeVoiceClient:
    def __init__(self, channel):
        self.channel = channel
        self.on_audio = None
        self.playing = False
        self.plays = []
        self.stops = 0
        self.connected = True
        self.disconnected = False
        self.receiver = NS(stats=Counter())

    def listen(self, callback):
        self.on_audio = callback

    def stop_listening(self):
        self.on_audio = None

    def is_connected(self):
        return self.connected

    def is_playing(self):
        return self.playing

    def play(self, source, *, after=None, **kw):
        if self.playing:
            raise discord.ClientException('Already playing audio.')
        self.playing = True
        self.plays.append((source, after, kw))

    def stop(self):
        self.stops += 1
        self.playing = False

    async def disconnect(self, force=False):
        self.connected = False
        self.disconnected = True


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_call(**kw):
    channel = FakeChannel()
    vc = FakeVoiceClient(channel)
    rt = FakeRealtime()
    ended = []
    options = dict(
        voice_client=vc, realtime=rt, is_bot=lambda uid: uid == MUSIC_BOT, name_of=NAMES.__getitem__,
        on_end=ended.append, watchdog_interval=3600,
    )
    options.update(kw)
    call = VoiceCall(**options)
    return call, vc, rt, channel, ended


def speech(ms=20, freq=300):
    """*ms* of 48 kHz stereo audio, as the receive thread delivers it."""
    n = 48 * ms
    mono = (6000 * np.sin(2 * np.pi * freq * np.arange(n) / 48000)).astype('<i2')
    return np.repeat(mono, 2).tobytes()


def talk(vc, who, ms):
    """*who* talking for *ms*, as Discord delivers it (one packet per 20 ms)."""
    for _ in range(ms // 20):
        vc.on_audio(who, speech())


def answer_audio(seconds=0.5):
    """A ``response.audio.delta`` payload: *seconds* of 24 kHz mono, base64."""
    pcm = (3000 * np.sin(np.arange(int(24000 * seconds)) / 5)).astype('<i2')
    return base64.b64encode(pcm.tobytes()).decode()


async def settle(n=10):
    for _ in range(n):
        await asyncio.sleep(0)


async def wait_for(cond, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            raise AssertionError('condition not reached')
        await asyncio.sleep(0.01)


async def test_people_reach_the_model_and_bots_dont():
    call, vc, rt, _, _ = make_call()
    await call.start()
    for _ in range(10):
        vc.on_audio(ANA, speech())
        vc.on_audio(MUSIC_BOT, speech(freq=1000))
    await wait_for(lambda: rt.frames >= 8)
    heard = set().union(*(s for _, s in call._timeline))
    check('Ana mixed in, the music bot never', heard == {ANA}, heard)
    samples = np.frombuffer(bytes(rt.audio), '<i2')
    check('16 kHz frames of 40 ms', len(rt.audio) % 1280 == 0 and np.abs(samples).max() > 1000, len(rt.audio))
    frames = rt.frames
    await wait_for(lambda: rt.frames >= frames + 3)  # nobody talking: silence keeps the server's clock running
    check('silence after the speech', not np.frombuffer(bytes(rt.audio[-1280:]), '<i2').any())
    await call.end('fim')


async def test_greeting_asks_the_model_to_speak_first():
    call, _, rt, _, _ = make_call()
    await call.start(greeting=voice_greeting('Ana'))
    check('a user turn then a response', rt.sent[0][0] == 'user' and 'Ana' in rt.sent[0][1] and rt.sent[1] == 'response.create', rt.sent)
    await call.end('fim')


async def test_an_exchange_is_played_and_transcribed_in_order():
    call, vc, rt, channel, _ = make_call()
    await call.start()
    for _ in range(25):
        vc.on_audio(BETO, speech())
    await wait_for(lambda: call._timeline)
    await asyncio.sleep(0.2)
    rt.emit('input_audio_buffer.speech_started', item_id='i1', audio_start_ms=0)
    rt.emit('input_audio_buffer.speech_stopped', item_id='i1', audio_end_ms=call._sent_ms)
    rt.emit('input_audio_buffer.committed', item_id='i1')
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio())
    rt.emit('response.audio_transcript.done', transcript='Fala, Beto!')
    rt.emit('response.done', response={'status': 'completed', 'usage': {'input_tokens': 30, 'output_tokens': 12}})
    await settle()
    check('playing through discord.py, tuned for voice', len(vc.plays) == 1 and vc.plays[0][2].get('application') == 'voip', vc.plays)
    check('the bot line waits for the user line before it', channel.posts == [], channel.posts)
    # The transcript of what Beto said arrives after the bot already answered.
    rt.emit('conversation.item.input_audio_transcription.completed', item_id='i1', transcript='e aí bot')
    await wait_for(lambda: len(channel.posts) == 2)
    check('in conversation order, with who spoke', channel.posts == ['🎙️ **Beto:** e aí bot', '🔊 Fala, Beto!'], channel.posts)
    check('usage and turns counted', (call.turns, call.tokens_in, call.tokens_out) == (1, 30, 12))
    await call.end('fim')


async def test_talking_over_the_bot_cuts_it_off():
    call, vc, rt, channel, _ = make_call(barge_in_ms=80)
    await call.start()
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(2.0))
    rt.emit('response.audio_transcript.done', transcript='Uma resposta bem comprida')
    await settle()
    check('the bot is talking', vc.playing and call.playback.has_audio)
    talk(vc, ANA, 300)
    await wait_for(lambda: call.interruptions == 1)
    check('playback stopped and dropped', vc.stops == 1 and not call.playback.has_audio)
    check('the response is cancelled', rt.cancels == 1)
    rt.emit('response.audio.delta', delta=answer_audio())  # still in flight from the server
    await settle()
    check('late audio of the cancelled answer is dropped', not call.playback.has_audio)
    rt.emit('response.done', response={'status': 'cancelled'})
    await wait_for(lambda: channel.posts)
    check('the line shows it was cut', channel.posts == ['🔊 Uma resposta bem comprida …'], channel.posts)
    await call.end('fim')


async def test_a_short_interjection_does_not_interrupt():
    call, vc, rt, _, _ = make_call(barge_in_ms=400)
    await call.start()
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(2.0))
    await settle()
    for _ in range(5):  # a 100 ms "uhum"
        vc.on_audio(ANA, speech())
    await asyncio.sleep(0.4)
    check('still talking', call.interruptions == 0 and vc.stops == 0 and call.playback.has_audio)
    await call.end('fim')


async def test_barge_in_can_be_turned_off():
    call, vc, rt, _, _ = make_call(barge_in_ms=0)
    await call.start()
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(2.0))
    await settle()
    talk(vc, ANA, 1000)
    await asyncio.sleep(0.3)
    check('talked over for a second, still talking', call.interruptions == 0 and vc.stops == 0 and rt.cancels == 0)
    await call.end('fim')


async def test_the_servers_speech_detection_alone_does_not_interrupt():
    # smart_turn reports speech it later finds wasn't a turn for the bot (people
    # talking among themselves), so only the barge-in time or a new answer cuts.
    call, vc, rt, _, _ = make_call()
    await call.start()
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(2.0))
    rt.emit('response.done', response={'status': 'completed'})
    await settle()
    rt.emit('input_audio_buffer.speech_started', item_id='i2', audio_start_ms=100)
    rt.emit('input_audio_buffer.speech_stopped', item_id='i2', audio_end_ms=900, reason='turn_invalid')
    await settle()
    check('still talking', vc.stops == 0 and call.playback.has_audio and call.interruptions == 0)
    await call.end('fim')


async def test_a_new_answer_cuts_what_is_left_of_the_last_one():
    call, vc, rt, channel, _ = make_call()
    await call.start()
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(3.0))
    rt.emit('response.audio_transcript.done', transcript='Primeira resposta')
    rt.emit('response.done', response={'status': 'completed'})
    await wait_for(lambda: channel.posts == ['🔊 Primeira resposta'])
    # Someone took a turn (a short "para!" the barge-in time doesn't catch).
    rt.emit('input_audio_buffer.committed', item_id='i2')
    rt.emit('conversation.item.input_audio_transcription.completed', item_id='i2', transcript='para')
    rt.emit('response.created', response={'id': 'r2'})
    await settle()
    check('nothing cut before the new answer speaks', vc.stops == 0)
    new_audio = answer_audio(0.3)
    rt.emit('response.audio.delta', delta=new_audio)
    await wait_for(lambda: vc.stops == 1)
    queued = call.playback.queued_seconds
    check('only the new answer is queued', 0.2 < queued < 0.4, queued)
    await wait_for(lambda: channel.messages[0].edits)
    check('the first one shows where it stopped', channel.posts[0] == '🔊 Primeira resposta …', channel.posts)
    check('counted as an interruption', call.interruptions == 1)
    await call.end('fim')


async def test_a_line_posted_whole_is_edited_when_the_bot_is_cut_off():
    call, vc, rt, channel, _ = make_call(barge_in_ms=80)
    await call.start()
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(3.0))
    rt.emit('response.audio_transcript.done', transcript='Explicação longa')
    rt.emit('response.done', response={'status': 'completed'})
    await wait_for(lambda: channel.posts == ['🔊 Explicação longa'])
    talk(vc, BETO, 300)
    await wait_for(lambda: channel.messages[0].edits)
    check('edited to show where it stopped', channel.posts == ['🔊 Explicação longa …'], channel.posts)
    await call.end('fim')


async def test_playback_restarts_when_more_audio_queued_as_a_player_finished():
    call, vc, rt, _, _ = make_call()
    await call.start()
    call.playback.begin()
    call.playback.feed(base64.b64decode(answer_audio(0.2)))
    vc.playing = True  # a player that's finishing: play() would refuse
    call._ensure_playing()
    check('not started over a running player', vc.plays == [])
    vc.playing = False
    call._after_play(None)  # discord.py's player thread, done with the previous stretch
    await settle()
    check('started by the after-callback', len(vc.plays) == 1)
    await call.end('fim')


async def test_errors_are_counted_but_cancel_races_are_not():
    call, _, rt, _, _ = make_call()
    await call.start()
    rt.emit('error', error={'message': 'Conversation has no active response.', 'param': 'response.cancel'})
    rt.emit('error', error={'message': 'Something broke'})
    await settle()
    check('only the real one', call.errors == 1, call.errors)
    await call.end('fim')


async def test_watchdog_reasons():
    clock = Clock()
    call, vc, _, channel, _ = make_call(clock=clock, idle_seconds=300, max_seconds=1800)
    check('healthy', call.check() is None)
    clock.t += 301
    check('idle', 'ninguém falou' in (call.check() or ''))
    call.last_activity = clock.t
    clock.t += 1500
    call.last_activity = clock.t
    check('too long', 'limite de 30 min' in (call.check() or ''))
    clock.t, call.started_at, call.last_activity = 0.0, 0.0, 0.0
    channel.members = [NS(bot=True)]
    check('everyone left', call.check() == 'todo mundo saiu da call')
    channel.members = [NS(bot=False)]
    vc.connected = False
    check('a voice reconnect gets some time', call.check() is None)
    clock.t += 31
    check('then the call is over', call.check() == 'perdi a conexão com a call')


async def test_the_watchdog_ends_the_call_and_cleans_up():
    clock = Clock()
    call, vc, rt, channel, ended = make_call(clock=clock, watchdog_interval=0.02)
    await call.start()
    channel.members = []
    await wait_for(lambda: ended)
    check('ended once, with the reason', ended == [call] and call.end_reason == 'todo mundo saiu da call')
    check('model session closed, voice left', rt.closed and vc.disconnected and vc.on_audio is None)
    check('goodbye posted', channel.posts[-1] == '📴 Saí da call: todo mundo saiu da call.', channel.posts)
    check('no task left running', all(t.done() for t in call._tasks))
    await call.end('again')
    check('end is idempotent', ended == [call] and len(channel.posts) == 1)


async def test_ending_mid_answer_flushes_the_transcript():
    call, vc, rt, channel, ended = make_call()
    await call.start()
    rt.emit('input_audio_buffer.committed', item_id='i1')  # its transcript never comes
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio_transcript.delta', delta='Então, o melhor')
    await settle()
    await call.end('pedido de Ana')
    check('partial answer marked, unknown line skipped, goodbye',
          channel.posts == ['🔊 Então, o melhor …', '📴 Saí da call: pedido de Ana.'], channel.posts)


async def test_losing_the_model_ends_the_call():
    call, vc, rt, channel, ended = make_call()
    await call.start()
    await rt.close()  # the socket closed under us
    await wait_for(lambda: ended)
    check('reason given', call.end_reason == 'a conexão com a IA de voz caiu' and vc.disconnected)


async def test_no_chat_permission_turns_transcripts_off_quietly():
    call, _, rt, channel, _ = make_call()
    channel.forbidden = True
    await call.start()
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio_transcript.done', transcript='oi')
    rt.emit('response.done', response={'status': 'completed'})
    await wait_for(lambda: channel.attempts == 1)
    await call.end('fim')
    check('tried once, then stopped trying', channel.attempts == 1, channel.attempts)


async def test_transcripts_off():
    call, _, rt, channel, _ = make_call(transcripts=False)
    await call.start()
    rt.emit('input_audio_buffer.committed', item_id='i1')
    rt.emit('conversation.item.input_audio_transcription.completed', item_id='i1', transcript='oi')
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.done', response={'status': 'completed'})
    await settle()
    await call.end('fim')
    check('only the goodbye', channel.posts == ['📴 Saí da call: fim.'], channel.posts)


def test_user_line_format():
    check('names', user_line(['Ana', 'B_e'], 'oi') == '🎙️ **Ana, B\\_e:** oi')
    check('nobody known', user_line([], 'oi') == '🎙️ oi')


def test_voice_prompt():
    text = voice_instructions(server='S', channel='Geral', caller='Ana', participants=['Ana', 'Beto'],
                              now='09/11/2026 15:00', persona='<personalidade>\nbase\n</personalidade>')
    check('speech, Brazilian Portuguese', 'português brasileiro' in text and 'vira fala' in text)
    check('context', 'Call: Geral' in text and 'Ana, Beto' in text and '09/11/2026 15:00' in text)
    check('server personality included', text.rstrip().endswith('</personalidade>'))
    check('no personality, no block', '<personalidade>\n' not in voice_instructions(
        server='S', channel='c', caller='a', participants=[], now='n'))


def test_servers_opt_in():
    setting = SETTINGS['voice']
    check('per server, off by default', (setting.scope, setting.kind, setting.default) == ('guild', 'bool', False))


async def test_voice_cog_stays_out_unless_enabled():
    import cogs.voice as voice_cog

    added = []
    await voice_cog.setup(NS(add_cog=lambda cog: added.append(cog)))
    check('VOICE_ENABLED is off in tests', added == [])
    config = voice_cog.session_config('instr')
    check('session asks for speech in and out',
          config['modalities'] == ['audio', 'text'] and config['instructions'] == 'instr'
          and config['turn_detection']['type'] == voice_cog.VOICE_TURN_DETECTION
          and config['input_audio_format'] == config['output_audio_format'] == 'pcm', str(config))
    check('the silence tool when it can be used', ('tools' in config) == voice_cog.can_stay_silent())
    check('tools and web search never together', not ('tools' in config and config.get('enable_search')))


async def test_a_failing_cleanup_still_releases_the_call():
    call, vc, rt, _, ended = make_call()
    await call.start()

    async def broken_close():
        raise RuntimeError('socket exploded')

    rt.close = broken_close
    await call.end('fim')
    check('forgotten anyway, voice left', ended == [call] and vc.disconnected)


def voice_cog(*, enabled=True, calls=None):
    import cogs.voice as voice_cog_mod

    cog = voice_cog_mod.Voice(NS(services=NS(settings=NS(get=lambda gid, ch, key: enabled))))
    cog.calls.update(calls or {})
    return cog


def guild_and_member(channel, *, admin=False):
    me = NS(id=1)
    guild = NS(id=10, me=me)
    perms = NS(manage_guild=admin, administrator=False, move_members=False)
    member = NS(voice=NS(channel=channel) if channel is not None else None, guild_permissions=perms)
    return guild, member


class VoiceChannelStub:
    mention = '<#20>'

    def __init__(self, *, connect=True, speak=True, user_limit=0, members=2):
        self._perms = NS(connect=connect, speak=speak, move_members=False)
        self.user_limit = user_limit
        self.members = [NS(bot=False)] * members

    def permissions_for(self, who):
        return self._perms


def test_voice_cog_join_checks():
    channel = VoiceChannelStub()
    guild, member = guild_and_member(channel)
    check('all good', voice_cog()._join_problem(guild, member) is None)

    off = voice_cog(enabled=False)
    check('server opted out (member)', 'Um admin' in off._join_problem(guild, member))
    check('server opted out (admin gets the command)',
          '/config servidor opcao:Conversa por voz' in off._join_problem(*guild_and_member(channel, admin=True)))
    check('not in a call', 'Entre numa call' in voice_cog()._join_problem(*guild_and_member(None)))

    busy = NS(ended=False, channel=channel)
    check('already here', voice_cog(calls={10: busy})._join_problem(guild, member) == 'Já estou nesta call.')
    elsewhere = NS(ended=False, channel=NS(mention='<#30>'))
    check('in another call here', '<#30>' in voice_cog(calls={10: elsewhere})._join_problem(guild, member))
    leaving = NS(ended=True, channel=channel)
    check('still leaving the last one', 'saindo' in voice_cog(calls={10: leaving})._join_problem(guild, member))
    check('global cap', 'máximo' in voice_cog(calls={99: busy})._join_problem(guild, member))

    mute = VoiceChannelStub(speak=False)
    check('needs Speak', 'Falar' in voice_cog()._join_problem(*guild_and_member(mute)))
    full = VoiceChannelStub(user_limit=2, members=2)
    check('full call', 'cheia' in voice_cog()._join_problem(*guild_and_member(full)))


async def test_cut_while_its_line_is_being_sent_still_shows():
    call, vc, rt, channel, _ = make_call(barge_in_ms=80)
    sending = asyncio.Event()
    release = asyncio.Event()
    plain_send = channel.send

    async def slow_send(text, **kw):  # Discord takes a moment to answer
        sending.set()
        await release.wait()
        return await plain_send(text, **kw)

    channel.send = slow_send
    await call.start()
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(3.0))
    rt.emit('response.audio_transcript.done', transcript='Resposta')
    rt.emit('response.done', response={'status': 'completed'})
    await asyncio.wait_for(sending.wait(), 2)
    talk(vc, ANA, 300)
    await wait_for(lambda: call.interruptions == 1)
    release.set()
    await wait_for(lambda: channel.messages and channel.messages[0].edits)
    check('fixed up once it was sent', channel.posts == ['🔊 Resposta …'], channel.posts)
    await call.end('fim')


async def test_the_model_hanging_up_on_a_quiet_call_says_so():
    call, _, rt, channel, ended = make_call()
    await call.start()
    rt.emit('error', error={'code': 'response_idle_timeout', 'message': 'Response idle timeout'})
    await rt.close()
    await wait_for(lambda: ended)
    check('honest reason', 'sem conversa' in call.end_reason, call.end_reason)


# --- choosing when to answer ---


async def test_the_model_can_choose_to_stay_silent():
    call, vc, rt, channel, _ = make_call()
    await call.start()
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(2.0))
    rt.emit('response.audio_transcript.done', transcript='Resposta')
    rt.emit('response.done', response={'status': 'completed'})
    await wait_for(lambda: channel.posts == ['🔊 Resposta'])
    rt.emit('input_audio_buffer.committed', item_id='i2')
    rt.emit('conversation.item.input_audio_transcription.completed', item_id='i2', transcript='Beto, vem cá')
    rt.emit('response.created', response={'id': 'r2'})
    rt.emit('response.done', response={'status': 'completed', 'output': [
        {'id': 'f1', 'type': 'function_call', 'name': VOICE_SILENCE_TOOL_NAME, 'call_id': 'c1', 'arguments': '{}'},
    ]})
    await wait_for(lambda: len(channel.posts) == 2)
    await settle()
    check('nothing said or posted for it', channel.posts == ['🔊 Resposta', '🎙️ Beto, vem cá'], channel.posts)
    check('what it was saying goes on', vc.stops == 0 and call.playback.has_audio)
    check('counted apart', (call.silences, call.turns) == (1, 1), (call.silences, call.turns))
    await call.end('fim')


async def test_with_a_wake_word_an_answer_waits_for_the_name():
    call, vc, rt, channel, _ = make_call(wake_words=['Quill', 'Kill'])
    await call.start()
    rt.emit('input_audio_buffer.committed', item_id='i1')
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(0.5))
    await settle()
    check('held until the transcript comes', vc.plays == [] and not call.playback.has_audio)
    rt.emit('conversation.item.input_audio_transcription.completed', item_id='i1', transcript='Ô quíll, que horas são?')
    await settle()
    check('called by name: it plays', len(vc.plays) == 1 and call.playback.has_audio)
    rt.emit('response.audio.delta', delta=answer_audio(0.5))
    rt.emit('response.audio_transcript.done', transcript='Três da tarde.')
    rt.emit('response.done', response={'status': 'completed', 'output': [{'id': 'a1', 'type': 'message'}]})
    await wait_for(lambda: len(channel.posts) == 2)
    check('posted', channel.posts == ['🎙️ Ô quíll, que horas são?', '🔊 Três da tarde.'], channel.posts)
    check('the rest of it queued', 0.9 < call.playback.queued_seconds < 1.1, call.playback.queued_seconds)
    check('kept', rt.deleted == [] and call.turns == 1)
    await call.end('fim')


async def test_with_a_wake_word_other_talk_gets_no_answer():
    call, vc, rt, channel, _ = make_call(wake_words=['Quill'])
    await call.start()
    rt.emit('input_audio_buffer.committed', item_id='i1')
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(0.5))
    rt.emit('conversation.item.input_audio_transcription.completed', item_id='i1', transcript='Beto, bora pro Nether?')
    await settle()
    check('cancelled, never played', rt.cancels == 1 and vc.plays == [] and not call.playback.has_audio)
    rt.emit('response.audio.delta', delta=answer_audio(0.5))  # still in flight
    rt.emit('response.audio_transcript.done', transcript='Bora!')
    rt.emit('response.done', response={'status': 'cancelled', 'output': [{'id': 'a1', 'type': 'message'}]})
    await wait_for(lambda: rt.deleted)
    check('late audio dropped too', vc.plays == [] and not call.playback.has_audio)
    check('the answer is taken out of the conversation', rt.deleted == ['a1'], rt.deleted)
    await wait_for(lambda: channel.posts)
    await settle()
    check('only what people said is posted', channel.posts == ['🎙️ Beto, bora pro Nether?'], channel.posts)
    check('counted', (call.unaddressed, call.turns, call.interruptions) == (1, 0, 0))
    await call.end('fim')


async def test_wake_word_whichever_comes_first():
    call, vc, rt, channel, _ = make_call(wake_words=['Quill'])
    await call.start()
    # The transcript can beat the response...
    rt.emit('input_audio_buffer.committed', item_id='i1')
    rt.emit('conversation.item.input_audio_transcription.completed', item_id='i1', transcript='Quill, oi')
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(0.3))
    rt.emit('response.done', response={'status': 'completed'})
    await settle()
    check('decided at once', len(vc.plays) == 1)
    vc.playing = False  # it was heard
    call.playback.clear()
    # ...or the whole answer can be ready before the transcript is.
    rt.emit('input_audio_buffer.committed', item_id='i2')
    rt.emit('response.created', response={'id': 'r2'})
    rt.emit('response.audio.delta', delta=answer_audio(0.3))
    rt.emit('response.audio_transcript.done', transcript='Tudo certo.')
    rt.emit('response.done', response={'status': 'completed'})
    await settle()
    check('held', len(vc.plays) == 1 and call.turns == 1)
    rt.emit('conversation.item.input_audio_transcription.completed', item_id='i2', transcript='e você, quill?')
    await wait_for(lambda: '🔊 Tudo certo.' in channel.posts)
    check('played and counted', len(vc.plays) == 2 and call.turns == 2 and call.interruptions == 0)
    await call.end('fim')


async def test_wake_word_leaves_the_greeting_alone_and_barge_in_drops_a_held_answer():
    call, vc, rt, _, _ = make_call(wake_words=['Quill'], barge_in_ms=80)
    await call.start(greeting=voice_greeting('Ana', 'Quill'))
    check('the greeting tells people the name', 'Quill' in rt.sent[0][1])
    rt.emit('response.created', response={'id': 'r1'})
    rt.emit('response.audio.delta', delta=answer_audio(0.3))
    rt.emit('response.done', response={'status': 'completed'})
    await settle()
    check('the greeting is not held', len(vc.plays) == 1)
    vc.playing = False
    call.playback.clear()
    rt.emit('input_audio_buffer.committed', item_id='i1')
    rt.emit('response.created', response={'id': 'r2'})
    rt.emit('response.audio.delta', delta=answer_audio(0.3))
    await settle()
    talk(vc, BETO, 300)  # people kept talking before its turn was even transcribed
    await wait_for(lambda: rt.cancels == 1)
    rt.emit('conversation.item.input_audio_transcription.completed', item_id='i1', transcript='Quill, espera')
    await settle()
    check('dropped, never heard', len(vc.plays) == 1 and not call.playback.has_audio and call.interruptions == 0)
    await call.end('fim')


def test_wake_word_matching():
    check('case and accents', mentions('Ô QUÍLL, me ajuda', ['quill']))
    check('anywhere in the sentence', mentions('e aí, quill?', ['Quill']))
    check('whole words only', not mentions('o quilliam chegou', ['quill']))
    check('any of the spellings', mentions('Kill me conta', ['Quill', 'Kill']))
    check('phrases', mentions('ei bot, tudo bem', ['ei bot']) and not mentions('eita bot', ['ei bot']))
    check('nothing to match', not mentions('quill', []) and not mentions('', ['quill']) and not mentions('oi', [' , ']))


def test_voice_prompt_choosing_silence():
    silent = voice_instructions(server='S', channel='c', caller='a', participants=[], now='n',
                                name='Quill', can_stay_silent=True)
    check('the tool, and when to use it', VOICE_SILENCE_TOOL_NAME in silent and 'bem curto' not in silent)
    check('its name', 'Seu nome aqui: Quill.' in silent)
    plain = voice_instructions(server='S', channel='c', caller='a', participants=[], now='n')
    check('without the tool it keeps it short', VOICE_SILENCE_TOOL_NAME not in plain and 'bem curto' in plain)
    check('greeting without a wake word', 'está ouvindo' in voice_greeting('Ana'))
