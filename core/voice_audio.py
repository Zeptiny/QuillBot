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
from collections import defaultdict, deque
from dataclasses import dataclass, field

import davey
import discord
from discord.ext import voice_recv

# The extension logs packet data and transport keys at DEBUG.
logging.getLogger('discord.ext.voice_recv').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

INPUT_FRAME_BYTES = 640  # 20 ms at 16 kHz, mono int16
OUTPUT_FRAME_BYTES = 3840  # 20 ms at 48 kHz, stereo int16
OPUS_SILENCE = b'\xf8\xff\xfe'
STATS_LOG_SECONDS = 15.0


@dataclass
class Speaker:
    decoder: discord.opus.Decoder = field(default_factory=discord.opus.Decoder)
    rate_state: tuple | None = None
    pending: bytes = b''
    frames: deque[tuple[float, bytes]] = field(default_factory=lambda: deque(maxlen=10))


class DiscordInput(voice_recv.AudioSink):
    """Receive ordered Opus, decrypt DAVE, decode and mix human speakers.

The pinned voice-recv release only removes RTP transport encryption and keeps
RTP padding octets in the decrypted data. Asking it for PCM would feed DAVE
ciphertext into Opus and kill its receive thread. discord.py owns DAVE key
negotiation; use its live session for decryption here, after removing the
padding or DAVE authentication fails on every padded packet.
"""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._speakers: dict[int, Speaker] = {}
        self._closed = False
        self._last_activity = 0.0
        # Privacy-safe diagnostics: counts only, never packet content. The
        # drop reasons mirror every silent `return` in write() so a live
        # session shows which stage starves Qwen of audio.
        self._dropped: dict[str, int] = defaultdict(int)
        self._speech_packets = 0
        self._voiced_frames = 0
        self._read_frames = 0
        self._last_stats_log = time.monotonic()
        self._dave_error_sample: str | None = None

    @property
    def last_activity(self) -> float:
        with self._lock:
            return self._last_activity

    def stats(self) -> dict[str, int]:
        """Snapshot of input counters for tests and diagnostics."""
        with self._lock:
            return {
                'speech_packets': self._speech_packets,
                'voiced_frames': self._voiced_frames,
                'read_frames': self._read_frames,
                'dropped_no_user': self._dropped['no_user'],
                'dropped_dave': self._dropped['dave'],
                'dropped_dave_missing': self._dropped['dave_missing'],
                'dropped_opus': self._dropped['opus'],
            }

    def wants_opus(self) -> bool:
        return True

    def write(self, user: discord.Member | discord.User | None, data: voice_recv.VoiceData) -> None:
        if user is None or user.bot:
            if user is None:
                self._dropped['no_user'] += 1
            return
        with self._lock:
            if self._closed:
                return
            speaker = self._speakers.get(user.id)
            if speaker is None:
                speaker = self._speakers[user.id] = Speaker()
            opus = data.opus
            if opus and opus != OPUS_SILENCE:
                self._speech_packets += 1
                # voice_recv decrypts the RTP transport layer but leaves the
                # RTP padding octets attached. DAVE authenticates the inner
                # frame only, so padding must be stripped before decryption
                # or every encrypted packet fails auth and audio starves.
                packet = getattr(data, 'packet', None)
                if getattr(packet, 'padding', False):
                    pad = opus[-1]
                    if 0 < pad <= len(opus):
                        opus = opus[:-pad]
                connection = self.voice_client._connection
                session = connection.dave_session
                if session is not None:
                    try:
                        opus = session.decrypt(user.id, davey.MediaType.audio, opus)
                    except (ValueError, RuntimeError) as error:
                        # Key transitions and lost encrypted frames are normal;
                        # never fall back to decoding ciphertext as Opus.
                        self._dropped['dave'] += 1
                        if self._dave_error_sample is None:
                            self._dave_error_sample = type(error).__name__ + ': ' + str(error)[:180]
                        return
                elif connection.dave_protocol_version:
                    self._dropped['dave_missing'] += 1
                    return
            try:
                pcm = speaker.decoder.decode(opus or None, fec=False)
            except discord.opus.OpusError:
                # A corrupt packet must not terminate every speaker's audio.
                self._dropped['opus'] += 1
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
        voiced = False
        with self._lock:
            for speaker in self._speakers.values():
                while speaker.frames and speaker.frames[0][0] < cutoff:
                    speaker.frames.popleft()
                if speaker.frames:
                    mixed = audioop.add(mixed, speaker.frames.popleft()[1], 2)
                    voiced = True
            self._read_frames += 1
            self._voiced_frames += voiced
        return mixed

    def _dave_diag(self) -> str:
        """Best-effort DAVE session snapshot (ids and counters only, never audio).

        davey keeps its own view of the MLS group; comparing its member list and
        per-speaker decryption stats against our drop counters distinguishes a
        missing-key problem from a frame-malformed one.
        """
        try:
            connection = self.voice_client._connection
            session = connection.dave_session
            if session is None:
                return f'dave=off(version={connection.dave_protocol_version})'
            members = sorted(session.get_user_ids())
            speakers = []
            for uid in list(self._speakers):
                stats = session.get_decryption_stats(uid)
                if stats is None:
                    speakers.append(f'{uid}=NO_KEYS_IN_GROUP')
                else:
                    speakers.append(
                        f'{uid}={stats.successes}ok/{stats.failures}fail/{stats.missing_keys}miss'
                        f'/{stats.invalid_nonces}nonce/{stats.passthroughs}pass'
                    )
            enc = session.get_encryption_stats()
            return (
                f'dave(v{connection.dave_protocol_version} {session.status.name} ready={session.ready} '
                f'epoch={session.epoch} members={members} '
                f'out={enc.successes}/{enc.attempts}enc miss={enc.missing_keys} '
                f'in[{", ".join(speakers) or "-"}]'
                + (f' first_dave_error={self._dave_error_sample}' if self._dave_error_sample else '')
                + ')'
            )
        except Exception as error:  # diagnostics must never break audio
            return f'dave_diag_failed({type(error).__name__})'

    def log_stats_if_due(self) -> None:
        """Periodic privacy-safe telemetry: counters only, never payloads."""
        now = time.monotonic()
        if now - self._last_stats_log < STATS_LOG_SECONDS:
            return
        self._last_stats_log = now
        stats = self.stats()
        logger.info(
            'Voice input: %d/%d voiced frames, %d speech packets, dropped no_user=%d dave=%d '
            'dave_missing=%d opus=%d | %s',
            stats['voiced_frames'], stats['read_frames'], stats['speech_packets'],
            stats['dropped_no_user'], stats['dropped_dave'],
            stats['dropped_dave_missing'], stats['dropped_opus'], self._dave_diag(),
        )

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
