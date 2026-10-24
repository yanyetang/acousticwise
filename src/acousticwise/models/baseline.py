import numpy as np
from sklearn.pipeline import Pipeline

from acousticwise.audio.features import mfcc_summary


def mfcc_features(logmel: np.ndarray, offsets_db: np.ndarray, n_mfcc: int) -> np.ndarray:
    """One row per clip: the MFCC summary of the log-mel after adding the clip's offset in dB."""
    return np.array(
        [mfcc_summary(logmel[i] + offsets_db[i], n_mfcc) for i in range(logmel.shape[0])],
        dtype=np.float32,
    )


def make_baseline(classifier: str, seed: int) -> Pipeline:
    """A scaler then the classifier ("svm" or "random_forest"), seeded."""
    raise NotImplementedError
