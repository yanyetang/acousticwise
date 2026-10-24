from acousticwise.data.report import build_report, format_report, missing_classes
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


def report_row(fold, split, label, clips, groups, seconds, short_clips, min_seconds):
    return {"fold": fold, "split": split, "label": label, "clips": clips, "groups": groups,
            "seconds": seconds, "short_clips": short_clips, "min_seconds": min_seconds}  # fmt: skip


def test_format_report_writes_one_markdown_table_per_fold():
    rows = [
        report_row(1, "train", "cargo", 3600, 40, 17995.5, 1, 4.5),
        report_row(1, "test", "tug", 720, 9, 3600.0, 0, 5.0),
        report_row(2, "train", "cargo", 2880, 32, 14400.0, 0, 5.0),
    ]

    header = [
        "| Split | Class | Clips | Vessels or days | Hours | Short clips | Shortest (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    # Rows keep their order; hours have 2 decimals, the shortest clip 1 decimal.
    expected = "\n".join(
        ["### Fold 1", "", *header,
         "| train | cargo | 3600 | 40 | 5.00 | 1 | 4.5 |",
         "| test | tug | 720 | 9 | 1.00 | 0 | 5.0 |",
         "",
         "### Fold 2", "", *header,
         "| train | cargo | 2880 | 32 | 4.00 | 0 | 5.0 |",
         ""]
    )  # fmt: skip

    assert format_report(rows) == expected


def test_missing_classes_lists_the_classes_a_split_has_no_clips_of():
    rows = [
        report_row(1, "train", "cargo", 10, 5, 50.0, 0, 5.0),
        report_row(1, "train", "tug", 10, 5, 50.0, 0, 5.0),
        report_row(1, "validation", "cargo", 2, 1, 10.0, 0, 5.0),
        report_row(1, "validation", "tug", 2, 1, 10.0, 0, 5.0),
        report_row(1, "test", "cargo", 2, 1, 10.0, 0, 5.0),  # no tug
        report_row(2, "train", "cargo", 10, 5, 50.0, 0, 5.0),  # no tug
        report_row(2, "validation", "cargo", 2, 1, 10.0, 0, 5.0),
        report_row(2, "validation", "tug", 2, 1, 10.0, 0, 5.0),
        report_row(2, "test", "tug", 2, 1, 10.0, 0, 5.0),  # no cargo
    ]

    # Only the splits with a gap, in the order of the rows; classes sorted by name.
    assert list(missing_classes(rows).items()) == [
        ((1, "test"), ["tug"]),
        ((2, "train"), ["tug"]),
        ((2, "test"), ["cargo"]),
    ]
