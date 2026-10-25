"""Score trained runs on their folds' test splits (technical design, section 8).

Usage:
    uv run python -m acousticwise.evaluate RUN_DIR [RUN_DIR ...] [--out FILE]

Adds a "test" section to each run's metrics.json, then writes one summary: a row per run
and, for two or more runs, the mean and spread across them.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from acousticwise.audio.cache import IndexEntry, load_features
from acousticwise.models.baseline import split_features
from acousticwise.models.cnn import ConvNet, predict_classes, split_logmel, standardize

# The scalar scores that go into the across-fold summary.
SUMMARY_SCORES = ("accuracy", "macro_f1_headline", "macro_f1_all")


def clip_scores(
    y_true: list[str], y_pred: list[str], classes: list[str], headline: list[str]
) -> dict:
    """Accuracy and macro-F1 per clip: over the headline classes and over all classes.

    A class with no clips in y_true cannot be scored, so it is left out of both averages
    and listed instead.
    """
    present = set(y_true)

    def macro_f1(labels: list[str]) -> float:
        # labels= picks the classes whose F1 is averaged; every clip still counts.
        scored = [label for label in labels if label in present]
        return float(f1_score(y_true, y_pred, labels=scored, average="macro"))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_headline": macro_f1(headline),
        "macro_f1_all": macro_f1(classes),
        "classes_without_clips": [label for label in classes if label not in present],
    }


def vessel_counts(labels: list[str], groups: list[str]) -> dict[str, dict]:
    """Per class: the number of vessels (days for background) and the largest one's clip share."""
    by_label = defaultdict(list)
    for label, group in zip(labels, groups, strict=True):
        by_label[label].append(group)

    counts = {}
    for label, clip_groups in by_label.items():
        clips_per_vessel = Counter(clip_groups)
        counts[label] = {
            "vessels": len(clips_per_vessel),
            "largest_vessel_share": max(clips_per_vessel.values()) / len(clip_groups),
        }
    return counts


def fold_summary(folds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per score: the mean and the sample standard deviation across folds."""
    summary = {}
    for score in folds[0]:
        values = [fold[score] for fold in folds]
        # stdev divides by n - 1: the folds are a small sample of the possible vessel splits.
        summary[score] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
    return summary


def evaluate_run(run_dir: Path) -> None:
    """Score a run's model on its fold's test clips; add the results to its metrics.json."""
    # The run's own config and split: the same preprocessing it was trained with, and the
    # test vessels it never heard.
    record = json.loads((run_dir / "config.json").read_text())
    cfg = record["config"]
    split = json.loads((run_dir / "splits.json").read_text())
    classes = json.loads((run_dir / "labels.json").read_text())

    logmel, index = load_features(cfg)
    y_test, predicted = _predict(run_dir, record, classes, logmel, index, split["test"])

    test = clip_scores(y_test, predicted, classes, cfg["evaluation"]["nine_class_subset"])
    test["clips"] = len(y_test)
    train_labels = [index[e["id"]][1] for e in split["train"]]
    test["per_class"] = _per_class(
        y_test,
        predicted,
        classes,
        train=vessel_counts(train_labels, [e["group"] for e in split["train"]]),
        test=vessel_counts(y_test, [e["group"] for e in split["test"]]),
    )
    # Rows are the true class, columns the predicted class.
    matrix = confusion_matrix(y_test, predicted, labels=classes)
    test["confusion_matrix"] = {"labels": classes, "matrix": matrix.tolist()}
    test["misclassified"] = [
        {"id": e["id"], "group": e["group"], "label": label, "predicted": guess}
        for e, label, guess in zip(split["test"], y_test, predicted, strict=True)
        if label != guess
    ]

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["test"] = test
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


def _predict(
    run_dir: Path,
    record: dict,
    classes: list[str],
    logmel: np.ndarray,
    index: dict[str, IndexEntry],
    entries: list[dict],
) -> tuple[list[str], list[str]]:
    """True and predicted labels for the entries, with the run's own model and preprocessing."""
    cfg = record["config"]
    if (run_dir / "model.pt").exists():  # CNN or CRNN: the weights, scaled like training
        x, y_true = split_logmel(logmel, index, entries)
        x = standardize(x, record["input_scaling"])
        model = ConvNet(len(classes), cfg["cnn"]["time_pooling"])
        # map_location: weights saved on a GPU still load on a machine without one.
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        indices = predict_classes(model, torch.from_numpy(x), cfg["cnn"]["batch_size"])
        return y_true, [classes[i] for i in indices]
    # Baseline: MFCC summaries into the saved scikit-learn pipeline.
    x, y_true = split_features(logmel, index, entries, cfg["features"]["n_mfcc"])
    return y_true, list(joblib.load(run_dir / "model.joblib").predict(x))


def _per_class(
    y_true: list[str], y_pred: list[str], classes: list[str], train: dict, test: dict
) -> dict[str, dict]:
    """Precision, recall, F1 and clips per class, beside its train and test vessel counts."""
    precision, recall, f1, clips = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    no_vessels = {"vessels": 0, "largest_vessel_share": 0.0}  # a class missing from a split
    return {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "clips": int(clips[i]),
            "train_vessels": train.get(label, no_vessels)["vessels"],
            "test_vessels": test.get(label, no_vessels)["vessels"],
            "largest_vessel_share": test.get(label, no_vessels)["largest_vessel_share"],
        }
        for i, label in enumerate(classes)
    }


def evaluate_runs(run_dirs: list[Path], summary_path: Path) -> None:
    """Evaluate each run; write one row per run and the mean and spread across them."""
    rows = []
    for run_dir in run_dirs:
        evaluate_run(run_dir)
        test = json.loads((run_dir / "metrics.json").read_text())["test"]
        fold = json.loads((run_dir / "config.json").read_text())["fold"]
        scores = {score: test[score] for score in SUMMARY_SCORES}
        rows.append({"run": run_dir.name, "fold": fold, **scores})

    # The details (per class, misclassified clips) stay in each run's own metrics.json.
    summary: dict[str, object] = {"runs": rows}
    if len(rows) > 1:  # one run has no spread
        summary["across_folds"] = fold_summary(
            [{s: row[s] for s in SUMMARY_SCORES} for row in rows]
        )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="run directories, one per fold")
    parser.add_argument("--out", type=Path, default=Path("artifacts/summary.json"))
    args = parser.parse_args()

    evaluate_runs(args.run_dirs, args.out)
    summary = json.loads(args.out.read_text())
    for row in summary["runs"]:
        print(
            f"{row['run']}: test accuracy {row['accuracy']:.3f}, "
            f"macro-F1 headline {row['macro_f1_headline']:.3f}, all {row['macro_f1_all']:.3f}"
        )
    for score, stats in summary.get("across_folds", {}).items():
        print(f"{score}: {stats['mean']:.3f} ± {stats['std']:.3f} over {len(summary['runs'])} runs")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
