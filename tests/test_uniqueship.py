from acousticwise.data.dataset import Recording
from acousticwise.data.uniqueship import read_split
from tests.conftest import clip_row


def test_vessel_clip_becomes_a_recording_grouped_by_mmsi(make_bundle):
    bundle = make_bundle(
        {(1, "train"): [clip_row("cargo", "20220103T120000_20220103T120005", 316001234)]}
    )

    recordings = read_split(bundle, fold=1, split="train")

    assert recordings == [
        Recording(
            id="cargo/20220103T120000_20220103T120005",
            path=bundle / "audio" / "cargo" / "20220103T120000_20220103T120005.wav",
            label="cargo",
            group="mmsi:316001234",
            meta={"duration_sec": 5.0},
        )
    ]


def test_background_clip_is_grouped_by_day(make_bundle):
    bundle = make_bundle(
        {
            (1, "train"): [
                clip_row("background", "20220105T030000_20220105T030005", 0, date="20220105")
            ]
        }
    )

    [recording] = read_split(bundle, fold=1, split="train")

    assert recording.group == "day:20220105"
