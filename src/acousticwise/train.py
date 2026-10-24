import csv
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score

from acousticwise.audio.cache import load_cache, settings_for
from acousticwise.data.splits import load_manifest
from acousticwise.models.baseline import make_baseline, mfcc_features


def train_baseline(cfg: dict, fold: int, run_dir: Path) -> None:
    """Train the baseline on one fold's training clips from the cache; write the run directory."""
    # The cache holds every clip of every fold; load_cache refuses one built with other settings.
    sample_rate = cfg["audio"]["sample_rate"]
    cache_dir = Path(cfg["data"]["cache_dir"]) / f"sr{sample_rate}"
    logmel = load_cache(cache_dir, settings_for(cfg, sample_rate))
    index = _read_index(cache_dir, cfg["audio"]["normalization"])

    # The manifest says which clips are this fold's train and validation clips.
    split = load_manifest(Path(cfg["data"]["splits_path"]))["folds"][str(fold)]
    n_mfcc = cfg["features"]["n_mfcc"]
    x_train, y_train = _features(logmel, index, split["train"], n_mfcc)
    x_val, y_val = _features(logmel, index, split["validation"], n_mfcc)

    model = make_baseline(cfg["baseline"]["classifier"], cfg["seed"]).fit(x_train, y_train)
    predicted = model.predict(x_val)
    metrics = {
        "validation": {
            "accuracy": float(accuracy_score(y_val, predicted)),
            "macro_f1": float(f1_score(y_val, predicted, average="macro")),
            "clips": len(y_val),
        }
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model.joblib")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }
    _write_json(run_dir / "config.json", {"config": cfg, "fold": fold, "versions": versions})
    _write_json(run_dir / "labels.json", list(model.classes_))
    _write_json(run_dir / "splits.json", split)
    _write_json(run_dir / "metrics.json", metrics)


def _read_index(cache_dir: Path, normalization: str) -> dict[str, tuple[int, str, float]]:
    """clip_id -> (row in logmel.npy, label, dB to add for the normalization method)."""
    index = {}
    with open(cache_dir / "index.csv") as f:
        for r in csv.DictReader(f):
            # The index stores the window's levels, so normalizing is subtracting one of them.
            offset = {"none": 0.0, "peak": -float(r["peak_db"]), "rms": -float(r["rms_db"])}
            index[r["clip_id"]] = (int(r["row"]), r["label"], offset[normalization])
    return index


def _features(
    logmel: np.ndarray, index: dict, entries: list[dict], n_mfcc: int
) -> tuple[np.ndarray, list[str]]:
    """MFCC features and labels for the manifest entries of one split."""
    rows, labels, offsets = zip(*(index[e["id"]] for e in entries), strict=True)
    return mfcc_features(logmel[list(rows)], np.array(offsets), n_mfcc), list(labels)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
