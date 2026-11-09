"""Bounded audio buffers between Discord's audio threads and the async voice session.

Discord uses 48 kHz stereo int16; Qwen consumes 16 kHz mono and produces
24 kHz mono. Keep resampler state across chunks, and send silence when Discord
stops sending packets so Qwen's server VAD can detect the end of a turn.
"""

from __future__ import annotations

import audioop
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import davey
import discord
from discord.ext import voice_recv

# The extension logs packet data and transport keys at DEBUG.
logging.getLogger('discord.ext.voice_recv').setLevel(logging.WARNING)

INPUT_FRAME_BYTES = 640  # 20 ms at 16 kHz, mono int16
OUTPUT_FRAME_BYTES = 3840  # 20 ms at 48 kHz, stereo int16
OPUS_SILENCE = b'\xf8\xff\xfe'


@dataclass
class Speaker:
    decoder: discord.opus.Decoder = field(default_factory=discord.opus.Decoder)
    rate_state: tuple | None = None
    pending: bytes = b''
    frames: deque[tuple[float, bytes]] = field(default_factory=lambda: deque(maxlen=10))


class DiscordInput(voice_recv.AudioSink):
    """Receive ordered Opus, decrypt DAVE, decode and mix human speakers.

The pinned voice-recv release only removes RTP transport encryption. Asking
it for PCM would feed DAVE ciphertext into Opus and kill its receive thread.
discord.py owns DAVE key negotiation; use its live session for decryption here.
"""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._speakers: dict[int, Speaker] = {}
        self._closed = False
        self._last_activity = 0.0

    @property
    def last_activity(self) -> float:
        with self._lock:
            return self._last_activity

    def wants_opus(self) -> bool:
        return True

    def write(self, user: discord.Member | discord.User | None, data: voice_recv.VoiceData) -> None:
        if user is None or user.bot:
            return
        with self._lock:
            if self._closed:
                return
            speaker = self._speakers.get(user.id)
            if speaker is None:
                speaker = self._speakers[user.id] = Speaker()
            opus = data.opus
            if opus and opus != OPUS_SILENCE:
                connection = self.voice_client._connection
                session = connection.dave_session
                if session is not None:
                    try:
                        opus = session.decrypt(user.id, davey.MediaType.audio, opus)
                    except (ValueError, RuntimeError):
                        # Key transitions and lost encrypted frames are normal;
                        # never fall back to decoding ciphertext as Opus.
                        return
                elif connection.dave_protocol_version:
                    return
            try:
                pcm = speaker.decoder.decode(opus or None, fec=False)
            except discord.opus.OpusError:
                # A corrupt packet must not terminate every speaker's audio.
                return
            mono = audioop.tomono(pcm, 2, 0.5, 0.5)
            converted, speaker.rate_state = audioop.ratecv(mono, 2, 1, 48000, 16000, speaker.rate_state)
            speaker.pending += converted
            now = time.monotonic()
            if data.opus and data.opus != OPUS_SILENCE:
                self._last_activity = now
            while len(speaker.pending) >= INPUT_FRAME_BYTES:
                speaker.frames.append((now, speaker.pending[:INPUT_FRAME_BYTES]))
                speaker.pending = speaker.pending[INPUT_FRAME_BYTES:]

    def read(self) -> bytes:
        """One wall-clock frame, including silence; called every 20 ms."""
        mixed = bytes(INPUT_FRAME_BYTES)
        cutoff = time.monotonic() - 0.2
        with self._lock:
            for speaker in self._speakers.values():
                while speaker.frames and speaker.frames[0][0] < cutoff:
                    speaker.frames.popleft()
                if speaker.frames:
                    mixed = audioop.add(mixed, speaker.frames.popleft()[1], 2)
        return mixed

    def remove_user(self, user_id: int) -> None:
        with self._lock:
            self._speakers.pop(user_id, None)

    def cleanup(self) -> None:
        with self._lock:
            self._closed = True
            self._speakers.clear()


class DiscordOutput(discord.AudioSource):
    """Streaming PCM source. Underruns produce silence; close produces EOF."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._rate_state = None
        self._closed = False

    def append(self, pcm: bytes) -> None:
        if len(pcm) % 2:
            raise ValueError('Incomplete PCM sample')
        with self._lock:
            if self._closed:
                return
            mono, self._rate_state = audioop.ratecv(pcm, 2, 1, 24000, 48000, self._rate_state)
            stereo = audioop.tostereo(mono, 2, 1, 1)
            if len(self._buffer) + len(stereo) > OUTPUT_FRAME_BYTES * 50 * 60:
                raise BufferError('Voice playback exceeded one minute of buffered audio')
            self._buffer.extend(stereo)

    def read(self) -> bytes:
        with self._lock:
            if self._closed:
                return b''
            frame = bytes(self._buffer[:OUTPUT_FRAME_BYTES])
            del self._buffer[:OUTPUT_FRAME_BYTES]
        return frame.ljust(OUTPUT_FRAME_BYTES, b'\x00')

    def clear(self) -> None:
        """Discard unheard audio immediately on an interruption."""
        with self._lock:
            self._buffer.clear()
            self._rate_state = None

    def cleanup(self) -> None:
        with self._lock:
            self._closed = True
            self._buffer.clear()
