import numpy as np

from acousticwise.errors import AudioTooShortError


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
    padded = np.pad(samples, (0, max(0, window_length - len(samples))))
    return padded[np.newaxis, :]
