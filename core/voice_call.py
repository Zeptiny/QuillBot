"""One live voice call: the bot in a voice channel, talking through a realtime model.

    people speak ─▶ ListeningVoiceClient (socket thread) ─▶ Downsampler per person
                 ─▶ Mixer ─▶ one 40 ms frame at a time ─▶ RealtimeVoiceClient
    model answers ─▶ audio deltas ─▶ PlaybackBuffer ─▶ VoiceClient.play
                  ─▶ transcripts ─▶ the voice channel's chat (optional)

The audio stream to the model never pauses (silence while nobody talks), so
the server's turn detection always has a clock to measure pauses against.
Someone talking over the bot cuts its playback and cancels the response
(barge-in). A watchdog ends the call when everyone left, nobody spoke for a
while, the call reached its time limit, or the voice connection is gone.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import discord

from core.realtime_voice import error_code, error_message
from core.voice_audio import INPUT_RATE, Downsampler, Mixer, PlaybackBuffer, PlaybackSource

logger = logging.getLogger(__name__)

_ATTRIBUTION_SLACK_MS = 300  # VAD timestamps vs. when each speaker's audio went out
_TIMELINE_MS = 300_000


def user_line(names: list[str], text: str) -> str:
    who = ', '.join(discord.utils.escape_markdown(n) for n in names)
    return f'🎙️ **{who}:** {text}' if who else f'🎙️ {text}'


class _Line:
    """One transcript line: queued in conversation order, posted once its text is known.

    A bot line is marked ``cut`` when someone talks over it — which usually
    happens after the model finished generating (it streams faster than it
    speaks), so a line that's already posted gets edited to show it.
    """

    __slots__ = ('ready', 'text', 'cut', 'message')

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.ready: asyncio.Future = loop.create_future()
        self.text: str | None = None
        self.cut = False
        self.message: Any = None

    @property
    def resolved(self) -> bool:
        return self.ready.done()

    def resolve(self, text: str | None) -> None:
        if not self.ready.done():
            self.text = text or None
            self.ready.set_result(None)

    def render(self) -> str | None:
        if self.text and self.cut:
            return f'{self.text} …'
        return self.text


class VoiceCall:
    MIX_INTERVAL = 0.04
    # Someone talking this long while the bot speaks cuts it off. Discord only sends
    # audio while a person's client hears speech, so this works in every turn
    # detection mode — smart_turn_v2 reports speech only once it has judged the turn,
    # usually after the person finished — and short "uhum"s and laughs don't count.
    BARGE_IN_MS = 700

    def __init__(
        self,
        *,
        voice_client: Any,
        realtime: Any,
        is_bot: Callable[[int], bool],
        name_of: Callable[[int], str],
        transcripts: bool = True,
        idle_seconds: float = 300.0,
        max_seconds: float = 1800.0,
        on_end: Callable[[VoiceCall], Awaitable[None] | None] | None = None,
        watchdog_interval: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.vc = voice_client
        self.rt = realtime
        self._is_bot = is_bot
        self._name_of = name_of
        self.transcripts = transcripts
        self.idle_seconds = idle_seconds
        self.max_seconds = max_seconds
        self._on_end = on_end
        self._watch_interval = watchdog_interval
        self._clock = clock

        self.mixer = Mixer(clock=clock)
        self.playback = PlaybackBuffer()
        self._downsamplers: dict[int, Downsampler] = {}  # socket thread only
        self._bots: dict[int, bool] = {}  # socket thread only

        self.started_at = clock()
        self.last_activity = self.started_at
        self._lost_since: float | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []
        self._ended = False
        self.end_reason = ''

        # Model state
        self._responding = False
        self._suppress = False  # drop audio of a response we cancelled
        self._talk_ms = 0  # how long people have been talking (for barge-in)
        self._bot_text: list[str] = []
        self._bot_transcript = ''

        # Who spoke when, on the stream clock the server's VAD timestamps use.
        self._sent_ms = 0
        self._timeline: deque[tuple[int, frozenset[int]]] = deque()
        self._speech_start: dict[str, int | None] = {}
        self._item_speakers: dict[str, list[str]] = {}

        # Transcript lines, posted in conversation order as their text arrives.
        self._lines: asyncio.Queue[_Line | None] = asyncio.Queue()
        self._user_lines: dict[str, _Line] = {}
        self._bot_line: _Line | None = None
        self._post_failed = False
        self._side_tasks: set[asyncio.Task] = set()

        # Stats for the end-of-call log
        self.turns = 0
        self.interruptions = 0
        self.errors = 0
        self._last_error = ''
        self.tokens_in = 0
        self.tokens_out = 0
        self._receive_warned = False

    @property
    def channel(self) -> Any:
        return self.vc.channel

    @property
    def ended(self) -> bool:
        return self._ended

    # --- lifecycle ---

    async def start(self, greeting: str | None = None) -> None:
        self._loop = asyncio.get_running_loop()
        self.vc.listen(self._on_pcm)
        spawn = self._loop.create_task
        self._tasks = [
            spawn(self._mix_loop(), name='voice-mix'),
            spawn(self._event_loop(), name='voice-events'),
            spawn(self._watchdog(), name='voice-watchdog'),
        ]
        if self.transcripts:
            self._tasks.append(spawn(self._poster(), name='voice-transcripts'))
        if greeting:
            try:
                await self.rt.add_message('user', greeting)
                await self.rt.create_response()
            except ConnectionError:
                pass  # the event loop notices the closed socket and ends the call

    async def end(self, reason: str) -> None:
        """Leave the call (idempotent): stop listening, close the model session, disconnect."""
        if self._ended:
            return
        self._ended = True
        self.end_reason = reason
        try:
            await self._shutdown(reason)
        finally:
            if self._on_end is not None:
                result = self._on_end(self)
                if inspect.isawaitable(result):
                    await result

    async def _shutdown(self, reason: str) -> None:
        current = asyncio.current_task()
        poster = None
        for task in self._tasks:
            if task.get_name() == 'voice-transcripts':
                poster = task
            elif task is not current and not task.done():
                task.cancel()
        try:
            self.vc.stop_listening()
        except Exception:
            logger.debug('[voice] stop_listening failed', exc_info=True)
        self.playback.clear()
        try:
            if self.vc.is_playing():
                self.vc.stop()
        except Exception:
            logger.debug('[voice] stopping playback failed', exc_info=True)
        try:
            await self.rt.close()
        except Exception:
            logger.debug('[voice] closing the realtime session failed', exc_info=True)
        try:
            await self.vc.disconnect(force=True)
        except Exception:
            logger.debug('[voice] disconnect failed', exc_info=True)

        # Let the transcript lines already waiting go out, then say goodbye.
        line = self._bot_line
        if line is not None and not line.resolved:
            partial = self._bot_transcript or ''.join(self._bot_text).strip()
            line.cut = True
            line.resolve(f'🔊 {partial}' if partial else None)
        for pending in self._user_lines.values():
            pending.resolve(None)
        if poster is not None and poster is not current:
            self._lines.put_nowait(None)
            try:
                await asyncio.wait_for(poster, 5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                poster.cancel()
        await self._post(f'📴 Saí da call: {reason}.')

        receiver = getattr(self.vc, 'receiver', None)
        logger.info(
            '[voice] call ended in #%s after %.0fs (%s): turns=%d interruptions=%d errors=%d '
            'tokens=%d/%d audio_sent=%.0fs receive=%s',
            getattr(self.channel, 'name', '?'), self._clock() - self.started_at, reason,
            self.turns, self.interruptions, self.errors, self.tokens_in, self.tokens_out,
            self._sent_ms / 1000, dict(getattr(receiver, 'stats', {}) or {}),
        )
        others = [t for t in self._tasks if t is not current]
        if others:
            await asyncio.gather(*others, return_exceptions=True)

    # --- people → model ---

    def _on_pcm(self, user_id: int, pcm: bytes) -> None:
        """20 ms of someone's audio (48 kHz stereo). discord.py's socket reader thread."""
        if self._ended:
            return
        bot = self._bots.get(user_id)
        if bot is None:
            bot = self._bots[user_id] = bool(self._is_bot(user_id))
        if bot:
            return  # music bots and other bots aren't part of the conversation
        down = self._downsamplers.get(user_id)
        if down is None:
            down = self._downsamplers[user_id] = Downsampler()
        self.mixer.push(user_id, down.process(pcm))

    async def _mix_loop(self) -> None:
        loop = asyncio.get_running_loop()
        silence = bytes(self.mixer.frame * 2)
        frame_ms = self.mixer.frame * 1000 // INPUT_RATE
        next_tick = loop.time()
        while not self._ended:
            next_tick += self.MIX_INTERVAL
            delay = next_tick - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -1.0:
                next_tick = loop.time()  # the loop stalled: don't burst to catch up
            frame, speakers = self.mixer.pull()
            if speakers:
                self.last_activity = self._clock()
                self._timeline.append((self._sent_ms, frozenset(speakers)))
                while self._timeline and self._timeline[0][0] < self._sent_ms - _TIMELINE_MS:
                    self._timeline.popleft()
                self._talk_ms += frame_ms
                if self._talk_ms >= self.BARGE_IN_MS and self._bot_speaking():
                    self._talk_ms = 0
                    await self._barge_in()
            else:
                self._talk_ms = max(0, self._talk_ms - 2 * frame_ms)  # a jitter gap isn't a pause
            try:
                await self.rt.append_audio(frame or silence)
            except ConnectionError:
                return
            self._sent_ms += frame_ms

    def _speakers_between(self, start_ms: int | None, end_ms: int | None) -> list[str]:
        if start_ms is None or end_ms is None:
            lo, hi = self._sent_ms - 10_000, self._sent_ms
        else:
            lo, hi = start_ms - _ATTRIBUTION_SLACK_MS, end_ms + _ATTRIBUTION_SLACK_MS
        seen: list[int] = []
        for at, speakers in self._timeline:
            if lo <= at <= hi:
                seen.extend(s for s in sorted(speakers) if s not in seen)
        return [self._name_of(s) for s in seen]

    # --- model → people ---

    async def _event_loop(self) -> None:
        async for event in self.rt.events():
            if self._ended:
                return
            try:
                await self._handle(event)
            except Exception:
                logger.exception('[voice] handling %s failed', event.get('type'))
        if not self._ended:
            if self._last_error == 'response_idle_timeout':
                await self.end('a IA de voz encerrou a sessão depois de um tempo sem conversa')
            else:
                await self.end('a conexão com a IA de voz caiu')

    async def _handle(self, event: dict) -> None:
        kind = event.get('type', '')
        item = event.get('item_id') or ''
        if kind == 'response.audio.delta':
            if not self._suppress:
                data = base64.b64decode(event.get('delta') or '')
                if data:
                    self.playback.feed(data)
                    self._ensure_playing()
        elif kind == 'response.audio_transcript.delta':
            self._bot_text.append(event.get('delta') or '')
        elif kind == 'input_audio_buffer.speech_started':
            self.last_activity = self._clock()
            self._speech_start[item] = event.get('audio_start_ms')
            await self._barge_in()
        elif kind == 'input_audio_buffer.speech_stopped':
            start = self._speech_start.pop(item, None)
            if self.transcripts:
                self._item_speakers[item] = self._speakers_between(start, event.get('audio_end_ms'))
        elif kind == 'input_audio_buffer.committed':
            if self.transcripts and self._loop is not None:
                line = self._user_lines[item] = _Line(self._loop)
                self._lines.put_nowait(line)
        elif kind == 'conversation.item.input_audio_transcription.completed':
            text = (event.get('transcript') or '').strip()
            names = self._item_speakers.pop(item, [])
            line = self._user_lines.pop(item, None)
            if line is not None:
                line.resolve(user_line(names, text) if text else None)
        elif kind == 'conversation.item.input_audio_transcription.failed':
            self._item_speakers.pop(item, None)
            line = self._user_lines.pop(item, None)
            if line is not None:
                line.resolve(None)
        elif kind == 'response.created':
            self._responding = True
            self._suppress = False
            self._talk_ms = 0  # only speech that starts over the answer interrupts it
            self._bot_text = []
            self._bot_transcript = ''
            self.playback.begin()
            self._bot_line = None
            if self.transcripts and self._loop is not None:
                self._bot_line = _Line(self._loop)
                self._lines.put_nowait(self._bot_line)
        elif kind == 'response.audio_transcript.done':
            # The whole text often arrives before the audio is cut short, so the
            # line waits for response.done to know whether it was.
            self._bot_transcript = (event.get('transcript') or '').strip()
        elif kind == 'response.done':
            self._finish_response(event.get('response') or {})
        elif kind == 'error':
            err = event.get('error')
            if isinstance(err, dict) and err.get('param') == 'response.cancel':
                # Our barge-in raced the server ending (or cancelling) the response itself.
                logger.debug('[voice] cancel had nothing to cancel: %s', error_message(event))
                return
            self.errors += 1
            self._last_error = error_code(event)
            logger.warning('[voice] realtime error in #%s: %s', getattr(self.channel, 'name', '?'), error_message(event))

    def _finish_response(self, response: dict) -> None:
        self._responding = False
        self.playback.finish()
        self._ensure_playing()  # flush a last partial frame
        status = response.get('status')
        usage = response.get('usage') or {}
        self.tokens_in += int(usage.get('input_tokens') or 0)
        self.tokens_out += int(usage.get('output_tokens') or 0)
        if status == 'completed':
            self.turns += 1
        elif status == 'failed':
            logger.warning('[voice] response failed: %s', response.get('status_details'))
        line = self._bot_line
        if line is not None:
            text = self._bot_transcript or ''.join(self._bot_text).strip()
            if status == 'cancelled':
                line.cut = True
            line.resolve(f'🔊 {text}' if text else None)

    def _bot_speaking(self) -> bool:
        return self._responding or self.playback.has_audio or self.vc.is_playing()

    async def _barge_in(self) -> None:
        """Someone started talking: stop talking over them."""
        cut = False
        if self._responding:
            cut = True
            self._suppress = True
            self._responding = False
            try:
                await self.rt.cancel_response()
            except ConnectionError:
                pass
        if self.playback.has_audio or self.vc.is_playing():
            cut = True
            self.playback.clear()
            if self.vc.is_playing():
                self.vc.stop()
        if not cut:
            return
        self.interruptions += 1
        line = self._bot_line
        if line is not None and not line.cut:
            line.cut = True
            if line.message is not None:  # already posted whole: show where it stopped
                text = line.render()
                if text:
                    task = asyncio.get_running_loop().create_task(self._edit(line.message, text))
                    self._side_tasks.add(task)  # the loop only keeps weak references
                    task.add_done_callback(self._side_tasks.discard)

    def _ensure_playing(self) -> None:
        if self._ended or not self.playback.has_audio:
            return
        vc = self.vc
        if not vc.is_connected() or vc.is_playing():
            return
        try:
            vc.play(
                PlaybackSource(self.playback), after=self._after_play,
                application='voip', signal_type='voice', bitrate=64,
            )
        except discord.ClientException:
            pass  # a player is just finishing; its after-callback starts the next one

    def _after_play(self, error: Exception | None) -> None:
        """discord.py's player thread, when a stretch of speech ends."""
        if error is not None:
            logger.warning('[voice] playback error: %s', error)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._ensure_playing)
            except RuntimeError:
                pass

    # --- transcripts ---

    async def _poster(self) -> None:
        while True:
            line = await self._lines.get()
            if line is None:
                return
            try:
                await asyncio.wait_for(asyncio.shield(line.ready), 30)
            except asyncio.TimeoutError:
                continue
            text = line.render()
            if text:
                line.message = await self._post(text)
                cut_text = line.render()
                if line.message is not None and cut_text != text:  # talked over while it was being sent
                    await self._edit(line.message, cut_text or text)

    async def _post(self, text: str) -> Any:
        if self._post_failed:
            return None
        try:
            return await self.channel.send(text[:2000], allowed_mentions=discord.AllowedMentions.none())
        except discord.Forbidden:
            self._post_failed = True
            logger.info('[voice] cannot post in the chat of #%s; transcripts off for this call',
                        getattr(self.channel, 'name', '?'))
        except Exception:
            logger.debug('[voice] posting a transcript failed', exc_info=True)
        return None

    async def _edit(self, message: Any, text: str) -> None:
        try:
            await message.edit(content=text[:2000], allowed_mentions=discord.AllowedMentions.none())
        except Exception:
            logger.debug('[voice] editing a transcript failed', exc_info=True)

    # --- watchdog ---

    async def _watchdog(self) -> None:
        while not self._ended:
            await asyncio.sleep(self._watch_interval)
            reason = self.check()
            if reason:
                await self.end(reason)
                return

    def check(self) -> str | None:
        """Why the call should end now, if it should."""
        now = self._clock()
        if not self.vc.is_connected():
            if self._lost_since is None:
                self._lost_since = now
            elif now - self._lost_since > 30:
                return 'perdi a conexão com a call'
            return None
        self._lost_since = None
        if not any(not m.bot for m in getattr(self.channel, 'members', ())):
            return 'todo mundo saiu da call'
        if now - self.last_activity > self.idle_seconds:
            return f'ninguém falou nada por {round(self.idle_seconds / 60)} min'
        if now - self.started_at > self.max_seconds:
            return f'a call chegou ao limite de {round(self.max_seconds / 60)} min'
        self._check_receive()
        return None

    def _check_receive(self) -> None:
        stats = getattr(getattr(self.vc, 'receiver', None), 'stats', None)
        if not stats or self._receive_warned:
            return
        if stats.get('packets', 0) >= 250 and not stats.get('decoded', 0):
            self._receive_warned = True
            state = getattr(self.vc, '_connection', None)
            session = getattr(state, 'dave_session', None)
            logger.warning(
                '[voice] %d packets from the voice server in #%s but none decoded: %s '
                '(dave protocol=%s ready=%s)',
                stats['packets'], getattr(self.channel, 'name', '?'), dict(stats),
                getattr(state, 'dave_protocol_version', None), getattr(session, 'ready', None),
            )
