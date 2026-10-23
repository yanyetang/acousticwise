import numpy as np
import pytest

from acousticwise.audio.preprocess import make_windows
from acousticwise.errors import AudioTooShortError
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
