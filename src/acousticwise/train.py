"""Train the baseline on one fold from the feature cache.

Usage:
    uv run python -m acousticwise.train [--config FILE] [--fold N] [--classifier svm|random_forest]

Writes <data.runs_dir>/baseline_fold<N>_<classifier>_<time>/ (technical design, section 5).
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import yaml
from sklearn.pipeline import Pipeline

from acousticwise.audio.cache import load_features
from acousticwise.data.splits import load_manifest
from acousticwise.evaluate import clip_scores
from acousticwise.models.baseline import make_baseline, split_features


def train_baseline(cfg: dict, fold: int, run_dir: Path) -> None:
    """Train the baseline on one fold's training clips from the cache; write the run directory."""
    # The cache holds every clip of every fold.
    logmel, index = load_features(cfg)

    # The manifest says which clips are this fold's train and validation clips.
    manifest_path = Path(cfg["data"]["splits_path"])
    split = load_manifest(manifest_path)["folds"][str(fold)]
    n_mfcc = cfg["features"]["n_mfcc"]
    x_train, y_train = split_features(logmel, index, split["train"], n_mfcc)
    x_val, y_val = split_features(logmel, index, split["validation"], n_mfcc)

    model = make_baseline(cfg["baseline"]["classifier"], cfg["seed"]).fit(x_train, y_train)
    predicted = list(model.predict(x_val))
    # Scored like the test split (evaluate.py), so validation and test numbers are comparable.
    headline = cfg["evaluation"]["nine_class_subset"]
    validation = clip_scores(y_val, predicted, list(model.classes_), headline)
    metrics = {"validation": {**validation, "clips": len(y_val)}}
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
        f"macro-F1 headline {metrics['macro_f1_headline']:.3f}, all {metrics['macro_f1_all']:.3f} "
        f"on {metrics['clips']} clips, "
        f"trained in {time.time() - start:.0f} s"
    )


if __name__ == "__main__":
    main()
