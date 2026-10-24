"""Score a trained run on its fold's test split (technical design, section 8)."""

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
