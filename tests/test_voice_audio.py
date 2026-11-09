"""Exercise real PCM resampling/Opus decoding without a Discord connection."""

import audioop
import math
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import davey
import discord
import pytest

from core.voice_audio import DiscordInput, DiscordOutput, INPUT_FRAME_BYTES, OPUS_SILENCE, OUTPUT_FRAME_BYTES


def pcm(value, samples):
    return struct.pack('<h', value) * samples


def input_sink():
    sink = DiscordInput()
    sink._voice_client = SimpleNamespace(_connection=SimpleNamespace(dave_session=None, dave_protocol_version=0))
    return sink


def test_decode_real_opus_mix_and_supply_silence():
    sink = input_sink()
    tone = b''.join(struct.pack('<hh', *([int(4000 * math.sin(i / 10))] * 2)) for i in range(960))
    encoded = discord.opus.Encoder().encode(tone, 960)
    sink.write(SimpleNamespace(id=1, bot=False), SimpleNamespace(opus=encoded))
    frame = sink.read()
    assert len(frame) == INPUT_FRAME_BYTES
    assert audioop.rms(frame, 2) > 100
    assert sink.read() == bytes(INPUT_FRAME_BYTES)
    sink.cleanup()


def test_mix_speakers_without_doubling_audio_duration(monkeypatch):
    monkeypatch.setattr(discord.opus.Decoder, 'decode', lambda *a, **k: pcm(20000, 1920))
    sink = input_sink()
    for user_id in (1, 2):
        sink.write(SimpleNamespace(id=user_id, bot=False), SimpleNamespace(opus=b'packet'))
    assert sink.read() == pcm(32767, 320)
    assert sink.read() == bytes(INPUT_FRAME_BYTES)
    sink.cleanup()


def test_dave_is_decrypted_before_opus_and_failures_are_dropped(monkeypatch):
    decoded = []
    monkeypatch.setattr(discord.opus.Decoder, 'decode', lambda self, data, **kw: decoded.append(data) or pcm(1000, 1920))
    sink = input_sink()
    decrypt = Mock(return_value=b'opus-cleartext')
    sink._voice_client._connection.dave_session = SimpleNamespace(decrypt=decrypt)
    user = SimpleNamespace(id=42, bot=False)
    sink.write(user, SimpleNamespace(opus=b'ciphertext'))
    decrypt.assert_called_once_with(42, davey.MediaType.audio, b'ciphertext')
    assert decoded == [b'opus-cleartext']
    assert sink.read() == pcm(1000, 320)
    decrypt.side_effect = ValueError('epoch changed')
    sink.write(user, SimpleNamespace(opus=b'bad-ciphertext'))
    assert decoded == [b'opus-cleartext']
    assert sink.read() == bytes(INPUT_FRAME_BYTES)
    sink.cleanup()


def test_ignore_bots_unknown_users_and_cleanup(monkeypatch):
    monkeypatch.setattr(discord.opus.Decoder, 'decode', lambda *a, **k: pcm(1000, 1920))
    sink = input_sink()
    packet = SimpleNamespace(opus=b'packet')
    sink.write(None, packet)
    sink.write(SimpleNamespace(id=1, bot=True), packet)
    assert sink.read() == bytes(INPUT_FRAME_BYTES)
    sink.write(SimpleNamespace(id=2, bot=False), packet)
    sink.remove_user(2)
    assert sink.read() == bytes(INPUT_FRAME_BYTES)
    sink.cleanup()
    sink.write(SimpleNamespace(id=2, bot=False), packet)
    assert sink.read() == bytes(INPUT_FRAME_BYTES)


def test_input_queue_drops_stale_frames_and_bounds_backlog(monkeypatch):
    monkeypatch.setattr(discord.opus.Decoder, 'decode', lambda *a, **k: pcm(1000, 1920))
    now = [100.0]
    monkeypatch.setattr('core.voice_audio.time.monotonic', lambda: now[0])
    sink = input_sink()
    for _ in range(50):
        sink.write(SimpleNamespace(id=1, bot=False), SimpleNamespace(opus=b'packet'))
    assert sum(sink.read() != bytes(INPUT_FRAME_BYTES) for _ in range(50)) == 10
    sink.write(SimpleNamespace(id=1, bot=False), SimpleNamespace(opus=b'packet'))
    now[0] += 1
    assert sink.read() == bytes(INPUT_FRAME_BYTES)
    sink.cleanup()


def test_corrupt_opus_does_not_kill_receive_and_silence_is_not_activity(monkeypatch):
    decoder = Mock(side_effect=[discord.opus.OpusError(-4), pcm(1000, 1920), pcm(0, 1920)])
    monkeypatch.setattr(discord.opus.Decoder, 'decode', decoder)
    now = [100.0]
    monkeypatch.setattr('core.voice_audio.time.monotonic', lambda: now[0])
    sink = input_sink()
    user = SimpleNamespace(id=1, bot=False)
    sink.write(user, SimpleNamespace(opus=b'corrupt'))
    assert sink.last_activity == 0
    sink.write(user, SimpleNamespace(opus=b'audio'))
    assert sink.read() == pcm(1000, 320)
    assert sink.last_activity == 100
    now[0] += 1
    sink.write(user, SimpleNamespace(opus=OPUS_SILENCE))
    assert sink.last_activity == 100
    sink.cleanup()


def test_output_resampling_is_independent_of_chunk_boundaries_and_interruptible():
    whole, chunked = DiscordOutput(), DiscordOutput()
    signal = b''.join(struct.pack('<h', i * 10 - 5000) for i in range(960))
    whole.append(signal)
    for offset in range(0, len(signal), 62):
        chunked.append(signal[offset:offset + 62])
    for _ in range(2):
        expected = whole.read()
        assert len(expected) == OUTPUT_FRAME_BYTES
        assert chunked.read() == expected
        samples = struct.unpack('<1920h', expected)
        assert samples[::2] == samples[1::2]
    chunked.append(signal)
    chunked.clear()
    assert chunked.read() == bytes(OUTPUT_FRAME_BYTES)
    chunked.cleanup()
    chunked.append(signal)
    assert chunked.read() == b''
    whole.cleanup()


def test_output_rejects_malformed_audio_and_excessive_backlog():
    output = DiscordOutput()
    with pytest.raises(ValueError):
        output.append(b'x')
    with pytest.raises(BufferError):
        output.append(bytes(24000 * 2 * 61))
    output.cleanup()
