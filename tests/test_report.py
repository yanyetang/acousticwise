from acousticwise.data.report import build_report
from tests.conftest import clip_row


def test_report_counts_clips_groups_seconds_and_short_clips_per_split_and_class(make_bundle):
    bundle = make_bundle(
        {
            (1, "train"): [
                clip_row("cargo", "a", 111),
                clip_row("cargo", "b", 111, duration=4.5),  # same vessel, one short clip
                clip_row("cargo", "c", 222),
                clip_row("background", "d", 0, date="20220105"),
                clip_row("background", "e", 0, date="20220106"),  # another day, another group
            ],
            (1, "validation"): [clip_row("tug", "f", 333)],
            (1, "test"): [clip_row("tug", "g", 444, duration=4.0)],
        }
    )

    report = build_report(bundle, folds=[1])

    # One row per fold, split and class, in split order then class name.
    # "groups" are distinct vessels, or distinct days for background.
    assert report == [
        {"fold": 1, "split": "train", "label": "background", "clips": 2, "groups": 2,
         "seconds": 10.0, "short_clips": 0, "min_seconds": 5.0},
        {"fold": 1, "split": "train", "label": "cargo", "clips": 3, "groups": 2,
         "seconds": 14.5, "short_clips": 1, "min_seconds": 4.5},
        {"fold": 1, "split": "validation", "label": "tug", "clips": 1, "groups": 1,
         "seconds": 5.0, "short_clips": 0, "min_seconds": 5.0},
        {"fold": 1, "split": "test", "label": "tug", "clips": 1, "groups": 1,
         "seconds": 4.0, "short_clips": 1, "min_seconds": 4.0},
    ]  # fmt: skip
