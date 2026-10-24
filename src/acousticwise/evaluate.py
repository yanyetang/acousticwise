"""Score a trained run on its fold's test split (technical design, section 8)."""

from sklearn.metrics import accuracy_score, f1_score


def clip_scores(
    y_true: list[str], y_pred: list[str], classes: list[str], headline: list[str]
) -> dict:
    """Accuracy and macro-F1 per clip: over the headline classes and over all classes."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # labels= picks the classes whose F1 is averaged; every clip still counts.
        "macro_f1_headline": float(f1_score(y_true, y_pred, labels=headline, average="macro")),
        "macro_f1_all": float(f1_score(y_true, y_pred, labels=classes, average="macro")),
    }
