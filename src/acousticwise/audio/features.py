import librosa
import numpy as np


def log_mel(samples: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    """Log-mel spectrogram in dB; shape (n_mels, n_frames)."""
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(mel, ref=1.0, top_db=None)


def mfcc_summary(log_mel_db: np.ndarray, n_mfcc: int) -> np.ndarray:
    """MFCC from a log-mel, summarised over time: n_mfcc means then n_mfcc stds."""
    mfcc = librosa.feature.mfcc(S=log_mel_db, n_mfcc=n_mfcc)
    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)
    return np.concatenate([means, stds])


def normalization_offset_db(samples: np.ndarray, method: str) -> float:
    """dB to add to a log-mel so it matches the normalized audio (peak | rms | none)."""
    if method == "peak":
        return float(-20 * np.log10(np.max(np.abs(samples))))
    elif method == "rms":
        return float(-20 * np.log10(np.sqrt(np.mean(samples**2))))
    elif method == "none":
        return 0.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")
