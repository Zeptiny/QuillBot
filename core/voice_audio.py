"""Audio plumbing for voice calls: resampling, mixing and the playback buffer.

Discord carries 48 kHz stereo 16-bit PCM in 20 ms frames (once opus is
decoded). The realtime voice model listens to 16 kHz mono and talks in 24 kHz
mono, both 16-bit little-endian. Three threads touch this audio — discord.py's
voice socket reader (what people say), the asyncio loop (the model's side) and
discord.py's audio player (what the bot says) — so the stateful pieces here
take a lock, and everything is plain numpy the tests can drive without Discord.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import discord

DISCORD_RATE = 48000
# One 20 ms frame of 48 kHz stereo s16le: what discord.py's player reads each tick.
DISCORD_FRAME_BYTES = 3840
INPUT_RATE = 16000  # what the model listens to
OUTPUT_RATE = 24000  # what the model speaks

_SILENCE_FRAME = bytes(DISCORD_FRAME_BYTES)


def lowpass_taps(num_taps: int, cutoff: float) -> np.ndarray:
    """Blackman-windowed sinc low-pass, unity gain at DC.

    ``cutoff`` is a fraction of the sample rate (0–0.5).
    """
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2
    taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.blackman(num_taps)
    return (taps / taps.sum()).astype(np.float32)


# 48 → 16 kHz: speech lives below 7 kHz; the band edge stays under the new 8 kHz Nyquist.
_DOWN_TAPS = lowpass_taps(96, 7000 / DISCORD_RATE)
# 24 → 48 kHz: keeps the model's band (< 12 kHz) and removes the images zero-stuffing creates.
_UP_TAPS = lowpass_taps(64, 11000 / DISCORD_RATE)


def _int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), -32768, 32767).astype('<i2')


class Downsampler:
    """48 kHz stereo s16le → 16 kHz mono int16, one stream, chunk by chunk.

    The filter history and the decimation phase carry over between chunks, so
    feeding a stream in pieces gives the same samples as feeding it whole.
    """

    FACTOR = 3

    def __init__(self) -> None:
        self._hist = np.zeros(len(_DOWN_TAPS) - 1, dtype=np.float32)
        self._phase = 0

    def process(self, pcm: bytes) -> np.ndarray:
        x = np.frombuffer(pcm, dtype='<i2')
        if x.size % 2:
            x = x[:-1]
        mono = (x[0::2].astype(np.float32) + x[1::2]) * 0.5
        buf = np.concatenate((self._hist, mono))
        y = np.convolve(buf, _DOWN_TAPS, mode='valid')
        self._hist = buf[len(buf) - len(self._hist):]
        out = y[self._phase::self.FACTOR]
        self._phase = (self._phase - len(y)) % self.FACTOR
        return _int16(out)


class Upsampler:
    """24 kHz mono s16le → 48 kHz stereo s16le, one stream, chunk by chunk.

    Polyphase ×2: the even and odd output samples are the input filtered by
    the even and odd taps of one low-pass.
    """

    def __init__(self) -> None:
        taps = _UP_TAPS * 2  # each phase carries half the gain
        self._even, self._odd = taps[0::2], taps[1::2]
        self._hist = np.zeros(len(self._even) - 1, dtype=np.float32)

    def process(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype='<i2').astype(np.float32)
        if not x.size:
            return b''
        buf = np.concatenate((self._hist, x))
        out = np.empty(2 * x.size, dtype=np.float32)
        out[0::2] = np.convolve(buf, self._even, mode='valid')
        out[1::2] = np.convolve(buf, self._odd, mode='valid')
        self._hist = buf[len(buf) - len(self._hist):]
        return np.repeat(_int16(out), 2).tobytes()


@dataclass
class _Stream:
    buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype='<i2'))
    primed: bool = False
    last_push: float = 0.0


class Mixer:
    """Each speaker's 16 kHz stream summed into the one the model hears.

    Discord sends a speaker's audio in 20 ms packets that arrive with network
    jitter, and nothing at all while they're quiet. Each speaker gets a small
    jitter buffer: their audio is mixed in once :attr:`prime` samples are
    queued (or the burst has stopped growing), and a buffer that runs dry
    re-primes instead of leaving holes every other frame. A buffer longer than
    ``max_buffer`` (a stalled loop, clock drift) drops its oldest audio.
    """

    def __init__(
        self,
        *,
        frame: int = 640,  # 40 ms at 16 kHz
        prime: int = 960,  # 60 ms
        max_buffer: int = 8000,  # 500 ms
        flush_after: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.frame = frame
        self.prime = prime
        self.max_buffer = max_buffer
        self.flush_after = flush_after
        self._clock = clock
        self._lock = threading.Lock()
        self._streams: dict[int, _Stream] = {}

    def push(self, speaker: int, samples: np.ndarray) -> None:
        """Queue *samples* (16 kHz mono int16) from *speaker*. Any thread."""
        if not samples.size:
            return
        with self._lock:
            stream = self._streams.get(speaker)
            if stream is None:
                stream = self._streams[speaker] = _Stream()
            buf = np.concatenate((stream.buf, samples.astype('<i2', copy=False)))
            if buf.size > self.max_buffer:
                buf = buf[buf.size - self.prime:]
            stream.buf = buf
            stream.last_push = self._clock()

    def pull(self) -> tuple[bytes | None, set[int]]:
        """One frame of everyone mixed, and who is in it; ``(None, set())`` when nobody is."""
        now = self._clock()
        parts: list[np.ndarray] = []
        speakers: set[int] = set()
        with self._lock:
            for speaker, stream in list(self._streams.items()):
                n = stream.buf.size
                if not stream.primed:
                    if n >= self.prime or (n and now - stream.last_push >= self.flush_after):
                        stream.primed = True
                    else:
                        if not n and now - stream.last_push > 5.0:
                            del self._streams[speaker]
                        continue
                take = min(n, self.frame)
                if take < self.frame:
                    stream.primed = False  # ran dry: wait for a fresh cushion
                if not take:
                    continue
                parts.append(stream.buf[:take])
                stream.buf = stream.buf[take:]
                speakers.add(speaker)
        if not parts:
            return None, speakers
        mix = np.zeros(self.frame, dtype=np.int32)
        for part in parts:
            mix[: part.size] += part
        return _int16(mix).tobytes(), speakers

    def forget(self, speaker: int) -> None:
        with self._lock:
            self._streams.pop(speaker, None)


class PlaybackBuffer:
    """The model's answer, queued as 48 kHz stereo frames for discord.py's player.

    :meth:`begin` / :meth:`feed` / :meth:`finish` come from the asyncio loop as
    the response streams in; :meth:`read_frame` from the player thread every
    20 ms. While a response is still streaming, an empty buffer plays silence
    (the model sends audio faster than real time, so that's a brief gap); once
    it's finished and drained, :meth:`read_frame` returns ``b''`` and the
    player stops — the bot only "speaks" while it has something to say.
    """

    def __init__(self, *, max_underrun_frames: int = 250) -> None:
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._open = False
        self._underrun = 0
        self._max_underrun = max_underrun_frames  # 5 s of waiting for a stalled response
        self._up = Upsampler()

    def begin(self) -> None:
        with self._lock:
            self._open = True
            self._underrun = 0

    def feed(self, pcm24k: bytes) -> None:
        data = self._up.process(pcm24k)
        with self._lock:
            self._buf += data

    def finish(self) -> None:
        with self._lock:
            self._open = False

    def clear(self) -> None:
        """Drop everything queued (the bot was interrupted)."""
        with self._lock:
            self._buf.clear()
            self._open = False
            self._up = Upsampler()

    @property
    def has_audio(self) -> bool:
        with self._lock:
            return bool(self._buf)

    @property
    def queued_seconds(self) -> float:
        with self._lock:
            return len(self._buf) / (DISCORD_FRAME_BYTES * 50)

    def read_frame(self) -> bytes:
        with self._lock:
            if len(self._buf) >= DISCORD_FRAME_BYTES:
                frame = bytes(self._buf[:DISCORD_FRAME_BYTES])
                del self._buf[:DISCORD_FRAME_BYTES]
                self._underrun = 0
                return frame
            if self._open:
                self._underrun += 1
                if self._underrun <= self._max_underrun:
                    return _SILENCE_FRAME
                self._open = False
            if self._buf:
                frame = bytes(self._buf) + bytes(DISCORD_FRAME_BYTES - len(self._buf))
                self._buf.clear()
                return frame
            return b''


class PlaybackSource(discord.AudioSource):
    """What ``VoiceClient.play`` reads: frames from a shared :class:`PlaybackBuffer`.

    A new source is started for each stretch of speech; ``cleanup`` leaves the
    shared buffer alone.
    """

    def __init__(self, buffer: PlaybackBuffer) -> None:
        self.buffer = buffer

    def read(self) -> bytes:
        return self.buffer.read_frame()

    def is_opus(self) -> bool:
        return False
