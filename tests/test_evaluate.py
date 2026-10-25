import json

import pytest

from acousticwise.evaluate import clip_scores, evaluate_run, fold_summary, vessel_counts
from acousticwise.train import train_baseline


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


def test_vessel_counts_gives_vessels_and_largest_vessel_share_per_class():
    # One entry per clip: its label (from the cache index) and its group (from the manifest).
    labels = ["cargo", "tug", "cargo", "background", "tug", "cargo", "tug", "cargo", "tug"]
    groups = [
        "mmsi:1", "mmsi:4", "mmsi:2", "day:2017-03-01", "mmsi:4",
        "mmsi:3", "mmsi:4", "mmsi:3", "mmsi:5",
    ]  # fmt: skip

    counts = vessel_counts(labels, groups)

    assert counts == {
        # cargo: 3 vessels; mmsi:3 has 2 of its 4 clips.
        "cargo": {"vessels": 3, "largest_vessel_share": 0.5},
        # tug: 2 vessels; mmsi:4 has 3 of its 4 clips, so the tug score rests mostly on one ship.
        "tug": {"vessels": 2, "largest_vessel_share": 0.75},
        # Background is grouped by day, so its "vessels" are recording days.
        "background": {"vessels": 1, "largest_vessel_share": 1.0},
    }


def test_fold_summary_gives_mean_and_spread_over_five_folds():
    folds = [
        {"accuracy": 0.4, "macro_f1_headline": 0.30},
        {"accuracy": 0.4, "macro_f1_headline": 0.32},
        {"accuracy": 0.4, "macro_f1_headline": 0.28},
        {"accuracy": 0.4, "macro_f1_headline": 0.34},
        {"accuracy": 0.4, "macro_f1_headline": 0.26},
    ]

    summary = fold_summary(folds)

    assert summary["accuracy"] == {"mean": pytest.approx(0.4), "std": pytest.approx(0.0)}
    # Spread is the sample standard deviation (divide by n - 1 = 4): squared gaps from 0.30
    # sum to 0.004, so std = sqrt(0.004 / 4) = 0.0316. Dividing by 5 would give 0.0283.
    assert summary["macro_f1_headline"] == {
        "mean": pytest.approx(0.30),
        "std": pytest.approx(0.0316228, abs=1e-6),
    }


def test_evaluate_run_adds_the_test_results_to_the_run_metrics(baseline_inputs, tmp_path):
    cfg, _ = baseline_inputs
    cfg["evaluation"]["nine_class_subset"] = ["cargo"]  # a headline smaller than all classes
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)
    validation = json.loads((run_dir / "metrics.json").read_text())["validation"]

    evaluate_run(run_dir)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["validation"] == validation  # kept as training wrote it
    test = metrics["test"]

    # Test clips cargo/8, cargo/9, tug/8, and tug/9, which sounds like cargo: 3 of 4 right.
    # cargo F1 0.8 (P 2/3, R 1), tug F1 2/3 (P 1, R 1/2).
    assert test["clips"] == 4
    assert test["accuracy"] == pytest.approx(3 / 4)
    assert test["macro_f1_headline"] == pytest.approx(0.8)
    assert test["macro_f1_all"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert test["classes_without_clips"] == []

    # Each class's scores beside the vessels behind them (6 train vessels, 2 test vessels).
    assert test["per_class"]["tug"] == {
        "precision": pytest.approx(1.0),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(2 / 3),
        "clips": 2,
        "train_vessels": 6,
        "test_vessels": 2,
        "largest_vessel_share": 0.5,
    }
    assert test["per_class"]["cargo"]["precision"] == pytest.approx(2 / 3)

    # Rows are the true class, columns the predicted class, both in labels.json order.
    assert test["confusion_matrix"] == {"labels": ["cargo", "tug"], "matrix": [[2, 0], [1, 1]]}
    # Listed with the vessel, so the error analysis can find the clip and its ship.
    assert test["misclassified"] == [
        {"id": "tug/9", "group": "mmsi:tug9", "label": "tug", "predicted": "cargo"}
    ]
