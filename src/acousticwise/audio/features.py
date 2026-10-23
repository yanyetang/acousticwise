import librosa
import numpy as np


def log_mel(samples: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    """Log-mel spectrogram in dB; shape (n_mels, n_frames)."""

    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(mel, ref=np.max)
