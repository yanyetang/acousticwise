"""Train the baseline, the CNN or the CRNN on one fold from the feature cache.

Usage:
    uv run python -m acousticwise.train [--config FILE] [--fold N] [--model baseline|cnn]
        [--classifier svm|random_forest] [--time-pooling mean|lstm]

Writes <data.runs_dir>/baseline_fold<N>_<classifier>_<time>/, cnn_fold<N>_<time>/ or
crnn_fold<N>_<time>/ (technical design, section 5).
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
import torch
import yaml

from acousticwise.audio.cache import load_features
from acousticwise.data.splits import load_manifest
from acousticwise.evaluate import clip_scores
from acousticwise.models.baseline import make_baseline, split_features
from acousticwise.models.cnn import (
    ConvNet,
    pick_device,
    predict_classes,
    set_seed,
    split_logmel,
    train_epoch,
)


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

    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model.joblib")
    versions = {"scikit-learn": sklearn.__version__}
    _write_run(run_dir, cfg, fold, list(model.classes_), split, metrics, manifest_path, versions)


def train_cnn(cfg: dict, fold: int, run_dir: Path) -> None:
    """Train the CNN or CRNN (cnn.time_pooling) on one fold's training clips from the cache;
    keep the epoch with the best validation headline macro-F1; write the run directory."""
    logmel, index = load_features(cfg)
    manifest_path = Path(cfg["data"]["splits_path"])
    split = load_manifest(manifest_path)["folds"][str(fold)]
    x_train, y_train = split_logmel(logmel, index, split["train"])
    x_val, y_val = split_logmel(logmel, index, split["validation"])

    # Standardize with the training clips only, so nothing about validation or test leaks in.
    scaling = {
        "mean": float(x_train.mean(dtype=np.float64)),
        "std": float(x_train.std(dtype=np.float64)),
    }
    x_train = torch.from_numpy((x_train - scaling["mean"]) / scaling["std"])
    x_val = torch.from_numpy((x_val - scaling["mean"]) / scaling["std"])

    # PyTorch needs class indices; labels.json keeps the names in the same order.
    classes = sorted(set(y_train))
    y_train_idx = torch.tensor([classes.index(label) for label in y_train])

    params = cfg["cnn"]
    set_seed(cfg["seed"])
    model = ConvNet(len(classes), params["time_pooling"]).to(pick_device())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    # The step size shrinks smoothly from lr towards 0 over the epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
    shuffle = torch.Generator().manual_seed(cfg["seed"])
    headline = cfg["evaluation"]["nine_class_subset"]

    history, best = [], None
    for epoch in range(1, params["epochs"] + 1):
        loss = train_epoch(model, x_train, y_train_idx, optimizer, params["batch_size"], shuffle)
        scheduler.step()
        predicted = [classes[i] for i in predict_classes(model, x_val, params["batch_size"])]
        validation = clip_scores(y_val, predicted, classes, headline)
        history.append({"epoch": epoch, "loss": loss, **validation})
        if best is None or validation["macro_f1_headline"] > best[1]["macro_f1_headline"]:
            # A copy on the cpu: the model keeps training and changing on its device.
            weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = (epoch, validation, weights)

    best_epoch, validation, weights = best
    metrics = {
        "validation": {
            **validation,
            "clips": len(y_val),
            "best_epoch": best_epoch,
            "history": history,
        }
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(weights, run_dir / "model.pt")
    _write_run(
        run_dir,
        cfg,
        fold,
        classes,
        split,
        metrics,
        manifest_path,
        versions={"torch": torch.__version__},
        extra={"input_scaling": scaling},
    )


def _write_run(
    run_dir: Path,
    cfg: dict,
    fold: int,
    labels: list[str],
    split: dict,
    metrics: dict,
    manifest_path: Path,
    versions: dict[str, str],
    extra: dict | None = None,
) -> None:
    """Write a trained model's record: config, labels, the fold's split and the metrics.

    The caller saves the model itself (model.joblib or model.pt) first.
    """
    record = {
        "config": cfg,
        "fold": fold,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, **versions},
        # The manifest is not committed; its SHA-256 identifies the exact splits used.
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        **(extra or {}),
    }
    _write_json(run_dir / "config.json", record)
    _write_json(run_dir / "labels.json", labels)
    _write_json(run_dir / "splits.json", split)
    _write_json(run_dir / "metrics.json", metrics)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, help="official fold 1-5 (default: data.app_fold)")
    parser.add_argument("--model", choices=["baseline", "cnn"], default="baseline")
    parser.add_argument("--classifier", help="svm or random_forest (default: baseline.classifier)")
    parser.add_argument(
        "--time-pooling", help="mean (CNN) or lstm (CRNN) (default: cnn.time_pooling)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    fold = args.fold or cfg["data"]["app_fold"]
    if args.classifier:
        cfg["baseline"]["classifier"] = args.classifier
    if args.time_pooling:
        cfg["cnn"]["time_pooling"] = args.time_pooling
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.model == "baseline":
        name, train = f"baseline_fold{fold}_{cfg['baseline']['classifier']}_{stamp}", train_baseline
    else:
        kind = "crnn" if cfg["cnn"]["time_pooling"] == "lstm" else "cnn"
        name, train = f"{kind}_fold{fold}_{stamp}", train_cnn
    run_dir = Path(cfg["data"]["runs_dir"]) / name

    start = time.time()
    train(cfg, fold, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())["validation"]
    best = f", best epoch {metrics['best_epoch']}" if "best_epoch" in metrics else ""
    print(
        f"{run_dir}: validation accuracy {metrics['accuracy']:.3f}, "
        f"macro-F1 headline {metrics['macro_f1_headline']:.3f}, all {metrics['macro_f1_all']:.3f} "
        f"on {metrics['clips']} clips{best}, trained in {time.time() - start:.0f} s"
    )


if __name__ == "__main__":
    main()
