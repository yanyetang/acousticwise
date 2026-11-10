import numpy as np
import pytest

from acousticwise.audio.preprocess import check_not_silent, make_windows, preprocess_file
from acousticwise.errors import AudioTooShortError, SilentAudioError
from tests.conftest import tone


def test_clip_of_minimum_length_is_zero_padded_to_one_window():
    samples = tone(4.0, 32000)

    windows = make_windows(samples, sr=32000, window_seconds=5.0, hop_seconds=2.5, min_seconds=4.0)

    assert windows.dtype == np.float32
    assert windows.shape == (1, 160000)
    np.testing.assert_array_equal(windows[0, :128000], samples)
    np.testing.assert_array_equal(windows[0, 128000:], 0.0)


def test_clip_shorter_than_minimum_raises_too_short():
    samples = tone(3.9, 32000)

    with pytest.raises(AudioTooShortError):
        make_windows(samples, sr=32000, window_seconds=5.0, hop_seconds=2.5, min_seconds=4.0)


def test_long_clip_is_cut_into_overlapping_windows():
    # 12 s with 5 s windows every 2.5 s: windows start at 0, 2.5 and 5 s. A window
    # starting at 7.5 s would run past the end (12.5 s), so the last 2 s are dropped.
    samples = tone(12.0, 32000)

    windows = make_windows(samples, sr=32000, window_seconds=5.0, hop_seconds=2.5, min_seconds=4.0)

    assert windows.shape == (3, 160000)
    np.testing.assert_array_equal(windows[0], samples[0:160000])
    np.testing.assert_array_equal(windows[1], samples[80000:240000])
    np.testing.assert_array_equal(windows[2], samples[160000:320000])


def test_silent_clip_raises_silent_audio():
    # Near-silent: RMS (loudness) 1e-5 is below the 1e-4 threshold from the config.
    samples = np.full(160000, 1e-5, dtype=np.float32)

    with pytest.raises(SilentAudioError):
        check_not_silent(samples, silence_rms=1.0e-4)


def test_clip_exactly_at_the_silence_threshold_is_not_silent():
    # RMS of a constant 0.5 is exactly 0.5: only audio below the threshold is silent.
    samples = np.full(160000, 0.5, dtype=np.float32)

    check_not_silent(samples, silence_rms=0.5)  # no error


def test_preprocess_file_loads_checks_and_windows_a_wav(write_wav):
    # One call does the whole path: 12 s stereo at 48 kHz becomes three 5 s mono
    # windows at 32 kHz, the same shape as in the 12 s windowing test above.
    path = write_wav(tone(12.0, 48000, channels=2), 48000)

    samples, windows = preprocess_file(
        path,
        sample_rate=32000,
        window_seconds=5.0,
        hop_seconds=2.5,
        min_seconds=4.0,
        silence_rms=1.0e-4,
        max_seconds=300.0,
    )

    assert samples.shape == (12 * 32000,)  # the whole file, mono at 32 kHz, for display
    assert windows.dtype == np.float32
    assert windows.shape == (3, 160000)
