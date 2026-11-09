"""Hearing a Discord voice channel with discord.py, which only knows how to speak.

discord.py 2.7 sends voice — including Discord's DAVE end-to-end encryption,
mandatory since March 2026 — but has no receive API (the old
``discord-ext-voice-recv`` extension predates DAVE and is unmaintained). It
does expose the three hooks a receiver needs, and :class:`ListeningVoiceClient`
uses them:

- the voice websocket ``hook``: SPEAKING (op 5) says which user sends each RTP
  ``ssrc``; CLIENT_DISCONNECT (op 13) says when they leave;
- ``add_socket_listener``: every UDP datagram from the voice server, on
  discord.py's socket reader thread;
- the connection's live ``davey.DaveSession``: the MLS group keys.

Each packet is unwrapped in three layers — the transport encryption
(``aead_xchacha20_poly1305_rtpsize``, the only mode discord.py offers that
Discord still accepts), then DAVE (keyed per sender), then opus — and the
result, 20 ms of 48 kHz stereo PCM, goes to a callback on that same thread.
"""

from __future__ import annotations

import logging
import struct
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import discord
import nacl.bindings
import nacl.exceptions
from discord import opus
from discord.gateway import DiscordVoiceWebSocket
from discord.voice_state import VoiceConnectionState

try:
    import davey  # type: ignore
except ImportError:  # discord.py refuses to open voice without it anyway
    davey = None

logger = logging.getLogger(__name__)

_RTP_HEADER = struct.Struct('>BBHII')
OPUS_PAYLOAD_TYPE = 0x78
# The one transport mode decrypt_rtp handles: the only one discord.py offers that
# Discord still accepts (the xsalsa20 modes were retired in November 2024).
TRANSPORT_MODE = 'aead_xchacha20_poly1305_rtpsize'
# What Discord clients send when they stop talking; never DAVE-encrypted.
SILENCE_FRAME = b'\xf8\xff\xfe'
_TAG = 16  # Poly1305 tag
_NONCE = 4  # rtpsize modes append a 32-bit counter instead of a full nonce

AudioCallback = Callable[[int, bytes], None]


class PacketError(ValueError):
    """Not a well-formed RTP packet."""


class DaveError(RuntimeError):
    """The E2EE layer couldn't be removed (yet)."""


@dataclass(frozen=True)
class RtpPacket:
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes  # transport-decrypted, header extension stripped


def is_rtcp(data: bytes) -> bool:
    """RTCP shares the socket; its packet types (200–204) fill the second byte."""
    return len(data) > 1 and 200 <= data[1] <= 204


def decrypt_rtp(data: bytes, key: bytes) -> RtpPacket | None:
    """Remove the transport encryption from one RTP packet.

    In the ``rtpsize`` modes the fixed header, the CSRCs and the 4-byte
    extension header are authenticated but left in clear; the extension body
    is encrypted along with the payload, so it's stripped after decrypting.
    The nonce is the 32-bit counter appended to the packet, zero-padded.
    Returns None for packets that aren't opus audio. Raises
    :class:`PacketError` or ``nacl.exceptions.CryptoError``.
    """
    if len(data) < 12 + _TAG + _NONCE:
        raise PacketError(f'{len(data)} bytes')
    b0, b1, sequence, timestamp, ssrc = _RTP_HEADER.unpack_from(data)
    if b0 >> 6 != 2:
        raise PacketError('not RTP v2')
    if b1 & 0x7F != OPUS_PAYLOAD_TYPE:
        return None
    header_len = 12 + 4 * (b0 & 0x0F)
    ext_words = 0
    if b0 & 0x10:
        if len(data) < header_len + 4:
            raise PacketError('truncated extension header')
        ext_words = struct.unpack_from('>H', data, header_len + 2)[0]
        header_len += 4
    if len(data) < header_len + _TAG + _NONCE:
        raise PacketError('no payload')
    nonce = data[-_NONCE:] + bytes(24 - _NONCE)
    plain = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
        bytes(data[header_len:-_NONCE]), bytes(data[:header_len]), nonce, key,
    )
    if ext_words:
        plain = plain[4 * ext_words:]
    if b0 & 0x20 and plain:  # RTP padding: the last byte says how much
        plain = plain[: len(plain) - plain[-1]]
    return RtpPacket(sequence, timestamp, ssrc, plain)


def strip_dave(state: Any, user_id: int, frame: bytes) -> bytes:
    """The opus frame inside DAVE's per-sender encryption.

    *state* is discord.py's ``VoiceConnectionState``, which owns the session
    and follows Discord's protocol transitions. Calls that aren't E2EE
    (protocol 0, or downgraded) pass frames through. Raises :class:`DaveError`.
    """
    if not getattr(state, 'dave_protocol_version', 0):
        return frame
    session = getattr(state, 'dave_session', None)
    if session is None or not session.ready or davey is None:
        raise DaveError('session not ready')
    try:
        return session.decrypt(user_id, davey.MediaType.audio, frame)
    except BaseException as e:  # pyo3 panics derive from BaseException
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        reason = str(e) or repr(e)
        if 'UnencryptedWhenPassthroughDisabled' in reason:
            # A sender still outside the MLS group (a transition gap) sends
            # plain opus; it decodes fine as it is.
            return frame
        raise DaveError(reason) from None


class VoiceReceiver:
    """Turns the voice server's datagrams into ``on_audio(user_id, pcm)`` calls.

    ``pcm`` is 20 ms of 48 kHz stereo s16le per packet. Runs on discord.py's
    socket reader thread; the SSRC map is filled from the voice websocket on
    the event loop (plain dict operations, atomic under the GIL). ``stats``
    counts what happened to every packet, for the logs.
    """

    def __init__(self, voice_client: Any, on_audio: AudioCallback | None = None) -> None:
        self.voice_client = voice_client
        self.on_audio = on_audio
        self.ssrc_users: dict[int, int] = {}
        self._decoders: dict[int, opus.Decoder] = {}
        self._key_src: Any = None
        self._key = b''
        self.stats: Counter[str] = Counter()
        self._logged_error = False

    # --- voice websocket side (event loop) ---

    def on_ws_message(self, msg: dict) -> None:
        op = msg.get('op')
        data = msg.get('d') or {}
        if op == DiscordVoiceWebSocket.SPEAKING:
            ssrc, user_id = data.get('ssrc'), data.get('user_id')
            if ssrc is not None and user_id is not None:
                self.ssrc_users[int(ssrc)] = int(user_id)
        elif op == DiscordVoiceWebSocket.CLIENT_DISCONNECT:
            user_id = data.get('user_id')
            if user_id is not None:
                self.forget_user(int(user_id))
        elif op == DiscordVoiceWebSocket.SESSION_DESCRIPTION:
            # A new voice session (first connect or a reconnect): the decoders'
            # state belongs to the old streams.
            self._decoders.clear()

    def forget_user(self, user_id: int) -> None:
        for ssrc in [s for s, u in self.ssrc_users.items() if u == user_id]:
            self.ssrc_users.pop(ssrc, None)
            self._decoders.pop(ssrc, None)

    # --- socket side (reader thread) ---

    def _secret_key(self) -> bytes:
        """The session's transport key ('' before Discord sent one)."""
        src = self.voice_client.secret_key
        if src is not self._key_src:
            self._key_src = src
            self._key = bytes(src) if isinstance(src, (list, bytes, bytearray)) else b''
        return self._key

    def feed(self, data: bytes) -> None:
        stats = self.stats
        stats['packets'] += 1
        try:
            if is_rtcp(data):
                stats['rtcp'] += 1
                return
            key = self._secret_key()
            if not key:
                stats['no_key'] += 1
                return
            if getattr(self.voice_client, 'mode', TRANSPORT_MODE) != TRANSPORT_MODE:
                stats['unsupported_mode'] += 1
                return
            packet = decrypt_rtp(data, key)
            if packet is None:
                stats['not_audio'] += 1
                return
            user_id = self.ssrc_users.get(packet.ssrc)
            if user_id is None:
                stats['unknown_ssrc'] += 1
                return
            if packet.payload == SILENCE_FRAME:
                stats['silence'] += 1
                return
            frame = strip_dave(self.voice_client._connection, user_id, packet.payload)
            decoder = self._decoders.get(packet.ssrc)
            if decoder is None:
                decoder = self._decoders[packet.ssrc] = opus.Decoder()
            pcm = decoder.decode(frame, fec=False)
        except nacl.exceptions.CryptoError:
            stats['transport_fail'] += 1
            return
        except DaveError:
            stats['dave_fail'] += 1
            return
        except opus.OpusError:
            stats['opus_fail'] += 1
            return
        except PacketError:
            stats['malformed'] += 1
            return
        except Exception:
            stats['errors'] += 1
            if not self._logged_error:
                self._logged_error = True
                logger.exception('[voice] receive failed on a packet')
            return
        stats['decoded'] += 1
        callback = self.on_audio
        if callback is not None:
            try:
                callback(user_id, pcm)
            except Exception:
                stats['callback_errors'] += 1
                logger.exception('[voice] audio callback failed')


class ListeningVoiceClient(discord.VoiceClient):
    """A ``VoiceClient`` that can also hear the channel.

    ``channel.connect(cls=ListeningVoiceClient)``, then :meth:`listen`.
    """

    def __init__(self, client: discord.Client, channel: discord.abc.Connectable) -> None:
        # Before super(): the connection state it creates holds our websocket hook.
        self.receiver = VoiceReceiver(self)
        super().__init__(client, channel)

    def create_connection_state(self) -> VoiceConnectionState:
        return VoiceConnectionState(self, hook=self._voice_ws_hook)

    async def _voice_ws_hook(self, ws: DiscordVoiceWebSocket, msg: dict) -> None:
        self.receiver.on_ws_message(msg)

    def listen(self, on_audio: AudioCallback) -> None:
        self.receiver.on_audio = on_audio
        self._connection.add_socket_listener(self.receiver.feed)

    def stop_listening(self) -> None:
        self._connection.remove_socket_listener(self.receiver.feed)
        self.receiver.on_audio = None
