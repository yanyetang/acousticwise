import numpy as np


def mfcc_features(logmel: np.ndarray, offsets_db: np.ndarray, n_mfcc: int) -> np.ndarray:
    """One row per clip: the MFCC summary of the log-mel after adding the clip's offset in dB."""
    raise NotImplementedError
