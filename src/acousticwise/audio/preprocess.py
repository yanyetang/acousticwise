from pathlib import Path

import numpy as np

from acousticwise.audio.io import load_audio
from acousticwise.errors import AudioTooLongError, AudioTooShortError, SilentAudioError


def make_windows(
    samples: np.ndarray, sr: int, window_seconds: float, hop_seconds: float, min_seconds: float
) -> np.ndarray:
    """Cut mono samples into fixed-length windows; shape (n_windows, window_length)."""
    min_length = int(round(min_seconds * sr))
    if len(samples) < min_length:
        raise AudioTooShortError(
            f"Audio is too short: {len(samples) / sr:.2f}s < {min_seconds:.2f}s"
        )

    window_length = int(round(window_seconds * sr))
    # Zero-pad a clip shorter than one window up to one window; a longer clip is unchanged.
    samples = np.pad(samples, (0, max(0, window_length - len(samples))))
    hop_length = int(round(hop_seconds * sr))
    starts = range(0, len(samples) - window_length + 1, hop_length)
    return np.stack([samples[s : s + window_length] for s in starts])


def check_not_silent(samples: np.ndarray, silence_rms: float) -> None:
    """Raise SilentAudioError when the RMS of samples is below silence_rms."""
    rms = np.sqrt(np.mean(samples**2))
    if rms < silence_rms:
        raise SilentAudioError(f"Audio is silent: RMS {rms:.2e} < {silence_rms:.2e}")


def check_not_too_long(samples: np.ndarray, sr: int, max_seconds: float) -> None:
    """Raise AudioTooLongError when samples last longer than max_seconds."""
    seconds = len(samples) / sr
    if seconds > max_seconds:
        raise AudioTooLongError(f"Audio is too long: {seconds:.1f}s > {max_seconds:.1f}s")


def preprocess_file(
    path: Path,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
    min_seconds: float,
    silence_rms: float,
    max_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a WAV, check it and cut it into windows; the one path for uploads in the app.

    Returns the samples (for display) and the windows.
    """
    samples = load_audio(path, sample_rate)
    check_not_too_long(samples, sample_rate, max_seconds)  # before windowing a huge file
    check_not_silent(samples, silence_rms)
    return samples, make_windows(samples, sample_rate, window_seconds, hop_seconds, min_seconds)
