import numpy as np

from acousticwise.audio.io import load_audio
from tests.conftest import tone


def test_mono_wav_at_target_rate_loads_unchanged(write_wav):
    samples = tone(5.0, 32000)
    path = write_wav(samples, 32000)

    audio = load_audio(path, target_sr=32000)

    assert audio.dtype == np.float32
    assert audio.shape == (160000,)
    np.testing.assert_allclose(audio, samples, atol=1e-6)


def test_stereo_wav_is_averaged_to_mono(write_wav):
    left = tone(5.0, 32000)
    right = np.zeros_like(left)
    path = write_wav(np.stack([left, right], axis=1), 32000)

    audio = load_audio(path, target_sr=32000)

    assert audio.dtype == np.float32
    assert audio.shape == (160000,)
    np.testing.assert_allclose(audio, left / 2, atol=1e-6)


def test_wav_at_other_rate_is_resampled_to_target(write_wav):
    path = write_wav(tone(5.0, 16000), 16000)

    audio = load_audio(path, target_sr=32000)

    assert audio.dtype == np.float32
    assert audio.shape == (160000,)
    # Resampling smears the first and last few samples, so compare the middle only.
    expected = tone(5.0, 32000)
    np.testing.assert_allclose(audio[1000:-1000], expected[1000:-1000], atol=1e-2)
