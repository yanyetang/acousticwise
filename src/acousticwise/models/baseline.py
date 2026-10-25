import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from acousticwise.audio.cache import IndexEntry
from acousticwise.audio.features import mfcc_summary


def mfcc_features(logmel: np.ndarray, offsets_db: np.ndarray, n_mfcc: int) -> np.ndarray:
    """One row per clip: the MFCC summary of the log-mel after adding the clip's offset in dB."""
    return np.array(
        [mfcc_summary(logmel[i] + offsets_db[i], n_mfcc) for i in range(logmel.shape[0])],
        dtype=np.float32,
    )


def split_features(
    logmel: np.ndarray, index: dict[str, IndexEntry], entries: list[dict], n_mfcc: int
) -> tuple[np.ndarray, list[str]]:
    """MFCC features and labels for one split's manifest entries, in the entries' order."""
    rows, labels, offsets = zip(*(index[e["id"]] for e in entries), strict=True)
    return mfcc_features(logmel[list(rows)], np.array(offsets), n_mfcc), list(labels)


def make_baseline(classifier: str, seed: int) -> Pipeline:
    """A scaler then the classifier ("svm" or "random_forest"), seeded."""
    if classifier == "svm":
        # The wrapper turns SVM scores into per-class probabilities; ensemble=False keeps one SVM.
        clf = CalibratedClassifierCV(SVC(random_state=seed), ensemble=False)
    elif classifier == "random_forest":
        clf = RandomForestClassifier(random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {classifier!r}. Use 'svm' or 'random_forest'.")
    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
