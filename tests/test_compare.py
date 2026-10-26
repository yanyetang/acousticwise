from acousticwise.compare import comparison_table


def summary(folds: list[tuple[int, float, float, float]], across: dict) -> dict:
    """A summary.json as evaluate writes it; folds are (fold, headline, all, accuracy)."""
    runs = [
        {"run": f"run{fold}", "fold": fold, "accuracy": acc,
         "macro_f1_headline": headline, "macro_f1_all": all_}
        for fold, headline, all_, acc in folds
    ]  # fmt: skip
    return {"runs": runs, "across_folds": across}


def spread(mean: float, std: float) -> dict:
    return {"mean": mean, "std": std}


def test_comparison_table_has_every_model_for_9_and_12_classes():
    summaries = {
        "Random forest": summary(
            [(1, 0.264, 0.218, 0.261), (2, 0.189, 0.142, 0.116)],
            {
                "accuracy": spread(0.1886, 0.1024),
                "macro_f1_headline": spread(0.2266, 0.0530),
                "macro_f1_all": spread(0.180, 0.0537),
            },
        ),
        "CNN": summary(
            [(1, 0.344, 0.285, 0.324), (2, 0.260, 0.201, 0.134)],
            {
                "accuracy": spread(0.229, 0.1344),
                "macro_f1_headline": spread(0.302, 0.0594),
                "macro_f1_all": spread(0.243, 0.0594),
            },
        ),
        "CRNN": summary(
            # Rows in another order: the per-fold column still runs fold 1, fold 2.
            [(2, 0.236, 0.182, 0.130), (1, 0.320, 0.265, 0.302)],
            {
                "accuracy": spread(0.216, 0.1216),
                "macro_f1_headline": spread(0.278, 0.0594),
                "macro_f1_all": spread(0.2236, 0.0587),
            },
        ),
    }

    table = comparison_table(summaries)

    # One row per model, in the order given; scores to 3 decimals, mean ± spread over folds.
    assert table == (
        "| Model | Macro-F1, 9 classes | Macro-F1, 12 classes | Accuracy | 9-class per fold |\n"
        "|---|---|---|---|---|\n"
        "| Random forest | 0.227 ± 0.053 | 0.180 ± 0.054 | 0.189 ± 0.102 | 0.264, 0.189 |\n"
        "| CNN | 0.302 ± 0.059 | 0.243 ± 0.059 | 0.229 ± 0.134 | 0.344, 0.260 |\n"
        "| CRNN | 0.278 ± 0.059 | 0.224 ± 0.059 | 0.216 ± 0.122 | 0.320, 0.236 |\n"
    )
