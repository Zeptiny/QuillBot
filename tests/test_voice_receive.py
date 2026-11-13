"""Voice receive: RTP transport decryption, the DAVE layer, opus, and the discord.py hooks."""

import asyncio
import struct
from types import SimpleNamespace as NS

import davey
import discord
import nacl.exceptions
import nacl.secret
import numpy as np
import pytest

from core.voice_receive import (
    SILENCE_FRAME,
    DaveError,
    ListeningVoiceClient,
    PacketError,
    VoiceReceiver,
    decrypt_rtp,
    is_rtcp,
    strip_dave,
)

from helpers import check

KEY = bytes(range(32))

try:
    discord.opus.Decoder()
    HAVE_OPUS = True
except discord.opus.OpusNotLoaded:
    HAVE_OPUS = False
needs_opus = pytest.mark.skipif(not HAVE_OPUS, reason='libopus not installed')


def rtp(payload, *, ssrc=111, seq=1, ext=b'', csrcs=(), counter=7, pt=0x78, key=KEY):
    """A packet the way Discord sends one in aead_xchacha20_poly1305_rtpsize."""
    b0 = 0x80 | len(csrcs) | (0x10 if ext else 0)
    header = struct.pack('>BBHII', b0, pt, seq, seq * 960, ssrc)
    header += b''.join(struct.pack('>I', c) for c in csrcs)
    plain = payload
    if ext:  # the 4-byte extension header stays in clear; its body is encrypted with the payload
        header += b'\xbe\xde' + struct.pack('>H', len(ext) // 4)
        plain = ext + payload
    nonce = struct.pack('>I', counter) + bytes(20)
    return header + nacl.secret.Aead(key).encrypt(plain, header, nonce).ciphertext + nonce[:4]


def test_decrypts_the_rtpsize_transport():
    packet = decrypt_rtp(rtp(b'opus-frame', ssrc=42, seq=9), KEY)
    check('payload', packet.payload == b'opus-frame', packet)
    check('header fields', (packet.ssrc, packet.sequence, packet.timestamp) == (42, 9, 9 * 960), packet)


def test_strips_the_encrypted_header_extension():
    ext = b'\x10\xff\x00\x00' + b'\x22\x01\x02\x00'  # audio level + another element, 2 words
    check('extension body removed', decrypt_rtp(rtp(b'opus', ext=ext), KEY).payload == b'opus')


def test_skips_csrcs():
    check('payload after the CSRC list', decrypt_rtp(rtp(b'opus', csrcs=(5, 6)), KEY).payload == b'opus')


def test_reads_what_discord_py_itself_sends():
    """Same packet layout as discord.py's sender (which is Discord's)."""
    sender = NS(secret_key=list(KEY), _incr_nonce=3, checked_add=lambda *a: None)
    header = bytearray(12)
    header[0], header[1] = 0x80, 0x78
    struct.pack_into('>HII', header, 2, 7, 6720, 99)
    data = discord.VoiceClient._encrypt_aead_xchacha20_poly1305_rtpsize(sender, bytes(header), b'frame')
    packet = decrypt_rtp(data, KEY)
    check('round trip through discord.py', packet.payload == b'frame' and packet.ssrc == 99, packet)


def test_rejects_bad_packets():
    with pytest.raises(nacl.exceptions.CryptoError):
        decrypt_rtp(rtp(b'opus', key=bytes(32)), KEY)
    tampered = bytearray(rtp(b'opus'))
    tampered[14] ^= 1
    with pytest.raises(nacl.exceptions.CryptoError):
        decrypt_rtp(bytes(tampered), KEY)
    with pytest.raises(PacketError):
        decrypt_rtp(b'\x80\x78' + bytes(10), KEY)
    with pytest.raises(PacketError):
        decrypt_rtp(b'\x40' + rtp(b'opus')[1:], KEY)  # RTP version 1
    check('non-opus payload type ignored', decrypt_rtp(rtp(b'x', pt=0x60), KEY) is None)


def test_tells_rtcp_from_rtp():
    check('RTCP receiver report', is_rtcp(bytes([0x81, 201]) + bytes(30)))
    check('RTP audio', not is_rtcp(rtp(b'opus')))


class FakeDave:
    def __init__(self, *, ready=True, error=None):
        self.ready = ready
        self.error = error
        self.calls = []

    def decrypt(self, user_id, media_type, frame):
        self.calls.append((user_id, media_type))
        if self.error is not None:
            raise self.error
        return frame[2:]  # "decrypt": drop a 2-byte wrapper


class PanicException(BaseException):
    """What pyo3 raises when Rust code panics."""


def test_dave_layer():
    plain = NS(dave_protocol_version=0, dave_session=None)
    check('no E2EE: frame untouched', strip_dave(plain, 1, b'abc') == b'abc')

    session = FakeDave()
    state = NS(dave_protocol_version=1, dave_session=session)
    check('decrypted with the sender\'s id', strip_dave(state, 42, b'XXopus') == b'opus')
    check('as audio', session.calls == [(42, davey.MediaType.audio)], session.calls)

    with pytest.raises(DaveError):
        strip_dave(NS(dave_protocol_version=1, dave_session=None), 1, b'x')
    with pytest.raises(DaveError):
        strip_dave(NS(dave_protocol_version=1, dave_session=FakeDave(ready=False)), 1, b'x')

    in_transition = FakeDave(error=RuntimeError('DecryptionFailed(UnencryptedWhenPassthroughDisabled)'))
    check('plain opus from a sender not in the group yet passes',
          strip_dave(NS(dave_protocol_version=1, dave_session=in_transition), 1, b'opus') == b'opus')
    with pytest.raises(DaveError):
        strip_dave(NS(dave_protocol_version=1, dave_session=FakeDave(error=PanicException('boom'))), 1, b'x')


def test_a_real_dave_session_without_a_group_is_not_ready():
    session = davey.DaveSession(1, 1234, 5678)
    with pytest.raises(DaveError):
        strip_dave(NS(dave_protocol_version=1, dave_session=session), 1, b'frame')


def voice_client(protocol=0, session=None):
    return NS(secret_key=list(KEY), _connection=NS(dave_protocol_version=protocol, dave_session=session))


def opus_frame(freq=440):
    pcm = np.repeat((8000 * np.sin(2 * np.pi * freq * np.arange(960) / 48000)).astype('<i2'), 2)
    return discord.opus.Encoder().encode(pcm.tobytes(), 960)


@needs_opus
def test_receiver_decodes_known_speakers():
    heard = []
    receiver = VoiceReceiver(voice_client(), lambda user, pcm: heard.append((user, pcm)))
    receiver.on_ws_message({'op': 5, 'd': {'ssrc': 111, 'user_id': '42', 'speaking': 1}})
    receiver.feed(rtp(opus_frame(), ssrc=111))
    check('one 20 ms frame of 48 kHz stereo for user 42',
          len(heard) == 1 and heard[0][0] == 42 and len(heard[0][1]) == 3840, heard[:1])

    receiver.feed(rtp(opus_frame(), ssrc=222))
    receiver.feed(rtp(SILENCE_FRAME, ssrc=111))
    receiver.feed(bytes([0x81, 201]) + bytes(30))
    tampered = bytearray(rtp(opus_frame(), ssrc=111))
    tampered[-6] ^= 1
    receiver.feed(bytes(tampered))
    receiver.feed(rtp(b'\xff\xff\xff\xff\xff', ssrc=111))  # not opus
    stats = receiver.stats
    check('accounted for', (stats['unknown_ssrc'], stats['silence'], stats['rtcp'], stats['transport_fail'],
                            stats['opus_fail'], stats['decoded']) == (1, 1, 1, 1, 1, 1), dict(stats))
    check('only real audio reached the callback', len(heard) == 1)


@needs_opus
def test_receiver_removes_dave_then_decodes():
    heard = []
    receiver = VoiceReceiver(voice_client(1, FakeDave()), lambda user, pcm: heard.append(user))
    receiver.ssrc_users[111] = 7
    receiver.feed(rtp(b'XX' + opus_frame(), ssrc=111))
    check('decoded after DAVE', heard == [7], dict(receiver.stats))
    failing = VoiceReceiver(voice_client(1, FakeDave(error=RuntimeError('DecryptionFailed'))), lambda *a: None)
    failing.ssrc_users[111] = 7
    failing.feed(rtp(b'XX' + opus_frame(), ssrc=111))
    check('undecryptable frames counted, not raised', failing.stats['dave_fail'] == 1, dict(failing.stats))


@needs_opus
def test_receiver_survives_a_failing_callback():
    def boom(user, pcm):
        raise ValueError('callback bug')

    receiver = VoiceReceiver(voice_client(), boom)
    receiver.ssrc_users[111] = 7
    receiver.feed(rtp(opus_frame(), ssrc=111))
    check('counted', receiver.stats['callback_errors'] == 1, dict(receiver.stats))


def test_receiver_follows_people_joining_and_leaving():
    receiver = VoiceReceiver(voice_client())
    receiver.on_ws_message({'op': 5, 'd': {'ssrc': 1, 'user_id': '10', 'speaking': 1}})
    receiver.on_ws_message({'op': 5, 'd': {'ssrc': 2, 'user_id': '20', 'speaking': 1}})
    receiver._decoders[1] = object()
    receiver.on_ws_message({'op': 13, 'd': {'user_id': '10'}})
    check('the one who left is forgotten', receiver.ssrc_users == {2: 20} and 1 not in receiver._decoders)
    receiver._decoders[2] = object()
    receiver.on_ws_message({'op': 4, 'd': {'mode': 'aead_xchacha20_poly1305_rtpsize'}})
    check('a new voice session resets the decoders', not receiver._decoders)
    receiver.on_ws_message({'op': 8, 'd': {}})  # anything else is ignored


async def test_listening_client_plugs_into_discord_py():
    client = NS(_connection=NS(loop=asyncio.get_running_loop()))
    vc = ListeningVoiceClient(client, NS(id=1))
    state = vc._connection
    try:
        check('the voice websocket hook is ours', state.hook == vc._voice_ws_hook)
        await state.hook(None, {'op': 5, 'd': {'ssrc': 5, 'user_id': '55', 'speaking': 1}})
        check('SPEAKING maps the ssrc', vc.receiver.ssrc_users == {5: 55})

        def on_audio(user, pcm):
            return None

        vc.listen(on_audio)
        check('packets routed to the receiver', vc.receiver.feed in state._socket_reader._callbacks)
        check('callback set', vc.receiver.on_audio is on_audio)
        vc.stop_listening()
        check('unrouted', vc.receiver.feed not in state._socket_reader._callbacks and vc.receiver.on_audio is None)
    finally:
        state._socket_reader.stop()


def test_receiver_waits_for_the_session_key_and_knows_its_mode():
    receiver = VoiceReceiver(NS(secret_key=discord.utils.MISSING, _connection=NS(dave_protocol_version=0)))
    receiver.feed(rtp(b'opus'))
    check('no key yet: counted, not an error', receiver.stats['no_key'] == 1 and not receiver.stats['errors'], dict(receiver.stats))
    other_mode = VoiceReceiver(NS(secret_key=list(KEY), mode='xsalsa20_poly1305_lite',
                                  _connection=NS(dave_protocol_version=0)))
    other_mode.feed(rtp(b'opus'))
    check('a mode it cannot read is counted', other_mode.stats['unsupported_mode'] == 1, dict(other_mode.stats))


def test_strips_rtp_padding_before_the_dave_layer():
    """Padding left on the frame makes DAVE's authentication fail on every padded packet."""
    padded = b'opus' + bytes(3) + b'\x04'  # 4 padding octets, the last says how many
    header = struct.pack('>BBHII', 0x80 | 0x20, 0x78, 1, 960, 111)
    nonce = struct.pack('>I', 1) + bytes(20)
    data = header + nacl.secret.Aead(KEY).encrypt(padded, header, nonce).ciphertext + nonce[:4]
    check('padding removed', decrypt_rtp(data, KEY).payload == b'opus')
