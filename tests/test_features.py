import numpy as np
import pytest

from acousticwise.audio.features import log_mel, mfcc_summary
from tests.conftest import tone


@pytest.mark.parametrize(
    ("sr", "n_fft", "hop_length"),
    [(32000, 2048, 512), (16000, 1024, 256)],
)
def test_log_mel_of_a_5s_window_is_64_by_313(sr, n_fft, hop_length):
    # Settings from the technical design, section 3: both rates give the same shape.
    samples = tone(5.0, sr)

    features = log_mel(samples, sr=sr, n_mels=64, n_fft=n_fft, hop_length=hop_length)

    assert features.shape == (64, 313)


def test_doubling_the_gain_shifts_log_mel_by_6_db_everywhere():
    # Twice the amplitude is four times the power: 10 * log10(4) = 6.02 dB in every
    # band and frame. White noise puts energy in all 64 bands, unlike a single tone.
    samples = np.random.default_rng(0).normal(0.0, 0.1, 160000).astype(np.float32)

    quiet = log_mel(samples, sr=32000, n_mels=64, n_fft=2048, hop_length=512)
    loud = log_mel(2 * samples, sr=32000, n_mels=64, n_fft=2048, hop_length=512)

    np.testing.assert_allclose(loud - quiet, 20 * np.log10(2), atol=1e-3)


def test_mfcc_summary_has_20_means_then_20_stds():
    # A log-mel that is the same in every frame: each MFCC is constant over time,
    # so its std over time is 0. That makes the last 20 values all zero.
    column = np.linspace(-60.0, -20.0, 64).reshape(64, 1)
    steady = np.tile(column, (1, 313))

    summary = mfcc_summary(steady, n_mfcc=20)

    assert summary.shape == (40,)
    assert np.any(summary[:20] != 0.0)
    np.testing.assert_allclose(summary[20:], 0.0, atol=1e-6)
