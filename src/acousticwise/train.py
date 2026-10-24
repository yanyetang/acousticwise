"""Train the baseline on one fold from the feature cache.

Usage:
    uv run python -m acousticwise.train [--config FILE] [--fold N] [--classifier svm|random_forest]

Writes <data.runs_dir>/baseline_fold<N>_<classifier>_<time>/ (technical design, section 5).
"""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from acousticwise.audio.cache import load_cache, settings_for
from acousticwise.data.splits import load_manifest
from acousticwise.models.baseline import make_baseline, mfcc_features

# One cache index entry: the clip's row in logmel.npy, its label and its offset in dB.
IndexEntry = tuple[int, str, float]

# The index.csv level each normalization method subtracts; "none" subtracts nothing.
LEVEL_COLUMNS = {"peak": "peak_db", "rms": "rms_db", "none": None}


def train_baseline(cfg: dict, fold: int, run_dir: Path) -> None:
    """Train the baseline on one fold's training clips from the cache; write the run directory."""
    # The cache holds every clip of every fold; load_cache refuses one built with other settings.
    sample_rate = cfg["audio"]["sample_rate"]
    cache_dir = Path(cfg["data"]["cache_dir"]) / f"sr{sample_rate}"
    logmel = load_cache(cache_dir, settings_for(cfg, sample_rate))
    index = _read_index(cache_dir, cfg["audio"]["normalization"])

    # The manifest says which clips are this fold's train and validation clips.
    manifest_path = Path(cfg["data"]["splits_path"])
    split = load_manifest(manifest_path)["folds"][str(fold)]
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
    _write_run(run_dir, cfg, fold, model, split, metrics, manifest_path)


def _write_run(
    run_dir: Path,
    cfg: dict,
    fold: int,
    model: Pipeline,
    split: dict,
    metrics: dict,
    manifest_path: Path,
) -> None:
    """Write the model and its record: config, labels, the fold's split and the metrics."""
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model.joblib")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }
    _write_json(
        run_dir / "config.json",
        {
            "config": cfg,
            "fold": fold,
            "versions": versions,
            # The manifest is not committed; its SHA-256 identifies the exact splits used.
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    )
    _write_json(run_dir / "labels.json", list(model.classes_))
    _write_json(run_dir / "splits.json", split)
    _write_json(run_dir / "metrics.json", metrics)


def _read_index(cache_dir: Path, normalization: str) -> dict[str, IndexEntry]:
    """clip_id -> (row in logmel.npy, label, dB to add for the normalization method)."""
    if normalization not in LEVEL_COLUMNS:
        raise ValueError(
            f"Unknown normalization {normalization!r}. Use {', '.join(LEVEL_COLUMNS)}."
        )
    column = LEVEL_COLUMNS[normalization]
    index = {}
    with open(cache_dir / "index.csv") as f:
        for r in csv.DictReader(f):
            # The index stores the window's levels, so normalizing is subtracting one of them.
            offset = -float(r[column]) if column else 0.0
            index[r["clip_id"]] = (int(r["row"]), r["label"], offset)
    return index


def _features(
    logmel: np.ndarray, index: dict[str, IndexEntry], entries: list[dict], n_mfcc: int
) -> tuple[np.ndarray, list[str]]:
    """MFCC features and labels for the manifest entries of one split."""
    rows, labels, offsets = zip(*(index[e["id"]] for e in entries), strict=True)
    return mfcc_features(logmel[list(rows)], np.array(offsets), n_mfcc), list(labels)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, help="official fold 1-5 (default: data.app_fold)")
    parser.add_argument("--classifier", help="svm or random_forest (default: baseline.classifier)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    fold = args.fold or cfg["data"]["app_fold"]
    if args.classifier:
        cfg["baseline"]["classifier"] = args.classifier
    name = f"baseline_fold{fold}_{cfg['baseline']['classifier']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(cfg["data"]["runs_dir"]) / name

    start = time.time()
    train_baseline(cfg, fold, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())["validation"]
    print(
        f"{run_dir}: validation accuracy {metrics['accuracy']:.3f}, "
        f"macro-F1 {metrics['macro_f1']:.3f} on {metrics['clips']} clips, "
        f"trained in {time.time() - start:.0f} s"
    )


if __name__ == "__main__":
    main()
