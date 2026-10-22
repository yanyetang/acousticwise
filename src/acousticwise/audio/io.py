import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from acousticwise.errors import UnsupportedFileError


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Read a WAV as mono float32 samples at target_sr."""
    try:
        samples, sample_rate = sf.read(path, dtype="float32")
    except sf.LibsndfileError as err:
        raise UnsupportedFileError(f"File {path} is not a supported audio file.") from err
    mono_samples = samples
    if samples.ndim == 2:
        mono_samples = np.mean(samples, axis=1).astype("float32")
    if sample_rate != target_sr:
        g = math.gcd(sample_rate, target_sr)
        mono_samples = resample_poly(mono_samples, target_sr // g, sample_rate // g)
    return mono_samples
