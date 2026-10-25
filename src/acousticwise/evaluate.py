"""Score a trained run on its fold's test split (technical design, section 8)."""

import statistics
from collections import Counter, defaultdict

from sklearn.metrics import accuracy_score, f1_score


def clip_scores(
    y_true: list[str], y_pred: list[str], classes: list[str], headline: list[str]
) -> dict:
    """Accuracy and macro-F1 per clip: over the headline classes and over all classes.

    A class with no clips in y_true cannot be scored, so it is left out of the all-class
    average and listed instead.
    """
    present = set(y_true)
    scored = [label for label in classes if label in present]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # labels= picks the classes whose F1 is averaged; every clip still counts.
        "macro_f1_headline": float(f1_score(y_true, y_pred, labels=headline, average="macro")),
        "macro_f1_all": float(f1_score(y_true, y_pred, labels=scored, average="macro")),
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
