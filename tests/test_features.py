import pytest

from acousticwise.audio.features import log_mel
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
