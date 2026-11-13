"""Voice-call audio: resampling, the per-speaker mixer and the playback buffer."""

import numpy as np

from core.voice_audio import (
    DISCORD_FRAME_BYTES,
    Downsampler,
    Mixer,
    PlaybackBuffer,
    PlaybackSource,
    Upsampler,
)

from helpers import check


def tone(freq, seconds, rate, amp=8000):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype('<i2')


def stereo(mono):
    return np.repeat(mono, 2).tobytes()


def peak_freq(x, rate):
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[np.argmax(spec)])


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def chunked(process, data, frame_bytes, sizes=(959, 961, 500, 1001)):
    """Feed *data* in uneven chunks (whole sample frames, never multiples of 3)."""
    parts, i, k = [], 0, 0
    while i < len(data):
        n = sizes[k % len(sizes)] * frame_bytes
        parts.append(process(data[i:i + n]))
        i, k = i + n, k + 1
    return parts


def test_downsampler_keeps_speech_and_drops_what_would_alias():
    out = Downsampler().process(stereo(tone(1000, 1.0, 48000)))
    check('48 kHz stereo → 16 kHz mono', out.size == 16000, out.size)
    check('1 kHz stays 1 kHz', abs(peak_freq(out[200:], 16000) - 1000) < 5, peak_freq(out[200:], 16000))
    ratio = rms(out[200:]) / rms(tone(1000, 1.0, 16000))
    check('unity gain in the speech band', 0.95 < ratio < 1.05, ratio)
    # 12 kHz can't exist at 16 kHz; decimated unfiltered, it would fold down to 4 kHz.
    high = Downsampler().process(stereo(tone(12000, 1.0, 48000)))
    check('12 kHz attenuated by more than 40 dB', rms(high[200:]) < rms(tone(12000, 1.0, 48000)) / 100, rms(high))


def test_downsampler_is_seamless_across_uneven_chunks():
    pcm = stereo(tone(440, 0.5, 48000))
    whole = Downsampler().process(pcm)
    d = Downsampler()
    pieces = np.concatenate(chunked(d.process, pcm, 4))
    check('same samples whole or in pieces', np.array_equal(pieces, whole), (pieces.size, whole.size))


def test_downsampler_averages_the_two_channels():
    left = tone(1000, 0.2, 48000)
    pcm = np.column_stack((left, np.zeros_like(left))).ravel().tobytes()
    out = Downsampler().process(pcm)
    ratio = rms(out[200:]) / rms(left)
    check('a one-sided source comes through at half level', 0.45 < ratio < 0.55, ratio)


def test_upsampler_doubles_the_rate_into_both_channels():
    x = tone(1000, 0.5, 24000)
    out = np.frombuffer(Upsampler().process(x.tobytes()), dtype='<i2')
    check('×2 rate ×2 channels', out.size == x.size * 4, out.size)
    left, right = out[0::2], out[1::2]
    check('both channels identical', np.array_equal(left, right))
    check('1 kHz stays 1 kHz', abs(peak_freq(left[200:], 48000) - 1000) < 5)
    ratio = rms(left[200:]) / rms(x)
    check('unity gain', 0.95 < ratio < 1.05, ratio)
    spec = np.abs(np.fft.rfft(left[200:].astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(left[200:].size, 1 / 48000)
    image = spec[freqs > 13000].sum() / spec.sum()
    check('the 23 kHz image of zero-stuffing is gone (< -50 dB)', image < 1e-5, image)


def test_upsampler_is_seamless_across_uneven_chunks():
    pcm = tone(700, 0.5, 24000).tobytes()
    whole = Upsampler().process(pcm)
    u = Upsampler()
    check('same samples whole or in pieces', b''.join(chunked(u.process, pcm, 2)) == whole)
    check('empty input, empty output', Upsampler().process(b'') == b'')


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def samples(*values):
    return np.array(values, dtype='<i2')


def test_mixer_sums_speakers_and_says_who_spoke():
    m = Mixer(frame=4, prime=4, clock=Clock())
    m.push(1, samples(100, 100, 100, 100))
    m.push(2, samples(50, 50, 50, 50))
    frame, who = m.pull()
    check('summed', np.frombuffer(frame, '<i2').tolist() == [150] * 4, frame)
    check('both speakers', who == {1, 2}, who)
    check('nobody left: nothing to send', m.pull() == (None, set()))


def test_mixer_clips_instead_of_wrapping_around():
    m = Mixer(frame=2, prime=2, clock=Clock())
    m.push(1, samples(30000, -30000))
    m.push(2, samples(30000, -30000))
    frame, _ = m.pull()
    check('saturates', np.frombuffer(frame, '<i2').tolist() == [32767, -32768], frame)


def test_mixer_waits_for_a_cushion_then_flushes_a_short_tail():
    clock = Clock()
    m = Mixer(frame=4, prime=8, flush_after=0.1, clock=clock)
    m.push(1, samples(1, 2, 3, 4, 5, 6))
    check('below the cushion: held back', m.pull() == (None, set()))
    clock.t += 0.2  # the burst stopped growing: it's the end of what they said
    frame, who = m.pull()
    check('flushed', np.frombuffer(frame, '<i2').tolist() == [1, 2, 3, 4] and who == {1}, frame)
    frame, _ = m.pull()
    check('last samples padded with silence', np.frombuffer(frame, '<i2').tolist() == [5, 6, 0, 0], frame)
    check('then quiet', m.pull()[0] is None)


def test_mixer_reprimes_after_running_dry_instead_of_stuttering():
    clock = Clock()
    m = Mixer(frame=4, prime=6, clock=clock)
    m.push(1, samples(*range(1, 7)))
    m.pull()
    m.pull()  # 2 left → partial frame, buffer dry
    m.push(1, samples(7, 8))
    check('a lone packet after running dry waits', m.pull()[0] is None)


def test_mixer_drops_old_audio_when_a_buffer_runs_away():
    m = Mixer(frame=4, prime=4, max_buffer=10, clock=Clock())
    m.push(1, samples(*range(12)))
    frame, _ = m.pull()
    check('keeps the newest cushion', np.frombuffer(frame, '<i2').tolist() == [8, 9, 10, 11], frame)


def test_mixer_forgets_a_speaker():
    m = Mixer(frame=2, prime=2, clock=Clock())
    m.push(1, samples(5, 5))
    m.forget(1)
    check('gone', m.pull() == (None, set()))


def loud(seconds):
    return tone(1000, seconds, 24000).tobytes()


def test_playback_plays_silence_while_waiting_and_stops_when_done():
    b = PlaybackBuffer()
    b.begin()
    check('waiting for the first audio: silence', b.read_frame() == bytes(DISCORD_FRAME_BYTES))
    b.feed(loud(0.05))  # 50 ms → two and a half 20 ms frames
    check('has audio', b.has_audio)
    first, second = b.read_frame(), b.read_frame()
    check('full frames', len(first) == len(second) == DISCORD_FRAME_BYTES and any(second))
    check('half a frame left, still streaming: wait', b.read_frame() == bytes(DISCORD_FRAME_BYTES))
    b.finish()
    last = b.read_frame()
    check('the rest, padded', len(last) == DISCORD_FRAME_BYTES and any(last[:1900]) and not any(last[1920:]))
    check('drained and finished: the player stops', b.read_frame() == b'')


def test_playback_clear_drops_everything_queued():
    b = PlaybackBuffer()
    b.begin()
    b.feed(loud(1.0))
    check('about a second queued', 0.95 < b.queued_seconds < 1.05, b.queued_seconds)
    b.clear()
    check('nothing left', not b.has_audio and b.read_frame() == b'')


def test_playback_gives_up_on_a_stalled_response():
    b = PlaybackBuffer(max_underrun_frames=3)
    b.begin()
    waits = [b.read_frame() for _ in range(3)]
    check('waits a little', all(w == bytes(DISCORD_FRAME_BYTES) for w in waits))
    check('then stops instead of playing silence forever', b.read_frame() == b'')


def test_playback_source_is_raw_pcm_from_the_buffer():
    b = PlaybackBuffer()
    source = PlaybackSource(b)
    check('PCM, encoded by discord.py', source.is_opus() is False)
    check('reads the buffer', source.read() == b'')
    source.cleanup()  # must not touch the shared buffer
    b.begin()
    check('still usable after cleanup', source.read() == bytes(DISCORD_FRAME_BYTES))
