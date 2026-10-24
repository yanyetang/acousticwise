"""Score a trained run on its fold's test split (technical design, section 8)."""


def clip_scores(
    y_true: list[str], y_pred: list[str], classes: list[str], headline: list[str]
) -> dict:
    """Accuracy and macro-F1 per clip: over the headline classes and over all classes."""
    raise NotImplementedError
