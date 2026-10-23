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
