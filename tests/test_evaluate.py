import pytest

from acousticwise.evaluate import clip_scores


def test_clip_scores_gives_the_headline_and_all_class_macro_f1():
    # Per class F1: cargo 0.5 (P 1/2, R 1/2), tug 0.8 (P 2/3, R 1), dredger 2/3 (P 1, R 1/2).
    y_true = ["cargo", "cargo", "tug", "tug", "dredger", "dredger"]
    y_pred = ["cargo", "tug", "tug", "tug", "dredger", "cargo"]

    scores = clip_scores(
        y_true, y_pred, classes=["cargo", "dredger", "tug"], headline=["cargo", "tug"]
    )

    assert scores["accuracy"] == pytest.approx(4 / 6)
    # Headline: averaged over the headline classes only, on all clips (dredger clips included).
    assert scores["macro_f1_headline"] == pytest.approx((0.5 + 0.8) / 2)
    assert scores["macro_f1_all"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_clip_scores_leaves_out_and_lists_classes_with_no_test_clips():
    # No pilotvessel clips in this test split, but one cargo clip is predicted as pilotvessel.
    # Per class F1: cargo 2/3 (P 1, R 1/2), tug 1; pilotvessel cannot be scored.
    y_true = ["cargo", "cargo", "tug", "tug"]
    y_pred = ["cargo", "pilotvessel", "tug", "tug"]

    scores = clip_scores(
        y_true, y_pred, classes=["cargo", "pilotvessel", "tug"], headline=["cargo", "tug"]
    )

    # Averaged over cargo and tug; counting pilotvessel as 0 would give (2/3 + 1 + 0) / 3.
    assert scores["macro_f1_all"] == pytest.approx((2 / 3 + 1) / 2)
    assert scores["classes_without_clips"] == ["pilotvessel"]
