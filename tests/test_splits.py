import pytest

from acousticwise.data.splits import (
    LeakageError,
    build_manifest,
    check_manifest,
    check_no_leakage,
    load_manifest,
    save_manifest,
)
from acousticwise.data.uniqueship import read_split
from tests.conftest import clip_row


def test_a_vessel_in_both_train_and_test_is_a_leak(make_bundle):
    bundle = make_bundle(
        {
            (1, "train"): [clip_row("tug", "20220103T120000_20220103T120005", 316000001)],
            (1, "validation"): [clip_row("tug", "20220104T120000_20220104T120005", 316000002)],
            (1, "test"): [clip_row("tug", "20220105T120000_20220105T120005", 316000001)],
        }
    )
    splits = {s: read_split(bundle, fold=1, split=s) for s in ("train", "validation", "test")}

    with pytest.raises(LeakageError, match="mmsi:316000001"):
        check_no_leakage(splits)


def test_a_clean_fold_passes(make_bundle):
    bundle = make_bundle(
        {
            (1, "train"): [
                clip_row("tug", "20220103T120000_20220103T120005", 316000001),
                clip_row("background", "20220103T030000_20220103T030005", 0, date="20220103"),
            ],
            (1, "validation"): [
                clip_row("tug", "20220104T120000_20220104T120005", 316000002),
                clip_row("background", "20220104T030000_20220104T030005", 0, date="20220104"),
            ],
            (1, "test"): [
                clip_row("tug", "20220105T120000_20220105T120005", 316000003),
                clip_row("background", "20220105T030000_20220105T030005", 0, date="20220105"),
            ],
        }
    )
    splits = {s: read_split(bundle, fold=1, split=s) for s in ("train", "validation", "test")}

    check_no_leakage(splits)  # no exception


def test_manifest_lists_ids_and_groups_per_fold_and_split(make_bundle):
    bundle = make_bundle(
        {
            (1, "train"): [clip_row("tug", "20220103T120000_20220103T120005", 316000001)],
            (1, "validation"): [
                clip_row("background", "20220104T030000_20220104T030005", 0, date="20220104")
            ],
            (1, "test"): [clip_row("cargo", "20220105T120000_20220105T120005", 316000003)],
        }
    )

    manifest = build_manifest(bundle, folds=[1])

    assert manifest == {
        "source": "official",
        "folds": {
            "1": {
                "train": [{"id": "tug/20220103T120000_20220103T120005", "group": "mmsi:316000001"}],
                "validation": [
                    {"id": "background/20220104T030000_20220104T030005", "group": "day:20220104"}
                ],
                "test": [
                    {"id": "cargo/20220105T120000_20220105T120005", "group": "mmsi:316000003"}
                ],
            }
        },
    }


def test_manifest_round_trips_through_json(make_bundle, tmp_path):
    bundle = make_bundle(
        {
            (1, "train"): [clip_row("tug", "20220103T120000_20220103T120005", 316000001)],
            (1, "validation"): [clip_row("tug", "20220104T120000_20220104T120005", 316000002)],
            (1, "test"): [clip_row("tug", "20220105T120000_20220105T120005", 316000003)],
        }
    )
    manifest = build_manifest(bundle, folds=[1])
    path = tmp_path / "artifacts" / "splits.json"

    save_manifest(manifest, path)

    assert load_manifest(path) == manifest


def test_manifest_check_names_the_fold_with_a_leak(make_bundle):
    clean = {
        (1, "train"): [clip_row("tug", "20220103T120000_20220103T120005", 316000001)],
        (1, "validation"): [clip_row("tug", "20220104T120000_20220104T120005", 316000002)],
        (1, "test"): [clip_row("tug", "20220105T120000_20220105T120005", 316000003)],
    }
    leaky_day = {
        (2, "train"): [
            clip_row("background", "20220106T010000_20220106T010005", 0, date="20220106")
        ],
        (2, "validation"): [clip_row("tug", "20220107T120000_20220107T120005", 316000004)],
        (2, "test"): [
            clip_row("background", "20220106T020000_20220106T020005", 0, date="20220106")
        ],
    }
    manifest = build_manifest(make_bundle(clean | leaky_day), folds=[1, 2])

    with pytest.raises(LeakageError, match=r"fold 2.*day:20220106"):
        check_manifest(manifest)
