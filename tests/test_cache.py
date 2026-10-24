import csv
import json
from dataclasses import asdict

import numpy as np
import pytest

from acousticwise.audio.cache import CacheSettings, build_cache
from acousticwise.audio.features import log_mel
from acousticwise.data.dataset import Recording
from tests.conftest import tone

SETTINGS = CacheSettings(
    sample_rate=32000,
    window_seconds=5.0,
    min_seconds=4.0,
    n_fft=2048,
    hop_length=512,
    n_mels=64,
    cache_format=1,
)


def make_recording(write_wav, label, stem, samples):
    path = write_wav(samples, 32000, name=f"{label}_{stem}.wav")
    return Recording(id=f"{label}/{stem}", path=path, label=label, group="mmsi:1")


def test_build_writes_logmel_index_and_meta(write_wav, tmp_path):
    cargo = tone(5.0, 32000, freq=200.0)
    recordings = [
        make_recording(write_wav, "cargo", "a", cargo),
        make_recording(write_wav, "tug", "b", tone(5.0, 32000, freq=1000.0)),
    ]
    out = tmp_path / "cache"

    build_cache(recordings, out, SETTINGS)

    logmel = np.load(out / "logmel.npy", mmap_mode="r")
    assert logmel.shape == (2, 64, 313)
    assert logmel.dtype == np.float16
    # float16 keeps about 3 significant digits, so allow 0.1 dB.
    expected = log_mel(cargo, sr=32000, n_mels=64, n_fft=2048, hop_length=512)
    np.testing.assert_allclose(logmel[0], expected, atol=0.1)

    with open(out / "index.csv") as f:
        index = list(csv.DictReader(f))
    assert [(r["clip_id"], r["row"], r["label"]) for r in index] == [
        ("cargo/a", "0", "cargo"),
        ("tug/b", "1", "tug"),
    ]

    meta = json.loads((out / "meta.json").read_text())
    assert meta["settings"] == asdict(SETTINGS)


def test_duplicate_clip_ids_fail_the_build_before_writing(write_wav, tmp_path):
    # Two clips with one id would make the index ambiguous: training could read
    # the wrong row for a clip without any error.
    recordings = [
        make_recording(write_wav, "cargo", "a", tone(5.0, 32000)),
        make_recording(write_wav, "cargo", "a", tone(5.0, 32000, freq=1000.0)),
    ]
    out = tmp_path / "cache"

    with pytest.raises(ValueError, match="cargo/a"):
        build_cache(recordings, out, SETTINGS)

    assert not (out / "logmel.npy").exists()


def test_unreadable_clip_is_skipped_and_listed_in_meta(write_wav, tmp_path):
    broken = tmp_path / "cargo_b.wav"
    broken.write_bytes(b"not a wav file")
    recordings = [
        make_recording(write_wav, "cargo", "a", tone(5.0, 32000)),
        Recording(id="cargo/b", path=broken, label="cargo", group="mmsi:1"),
        make_recording(write_wav, "tug", "c", tone(5.0, 32000, freq=1000.0)),
    ]
    out = tmp_path / "cache"

    # 1 failure in 3 clips is far above the real limit (test 4), so raise it here.
    build_cache(recordings, out, SETTINGS, max_failed_fraction=0.5)

    assert np.load(out / "logmel.npy", mmap_mode="r").shape[0] == 2
    with open(out / "index.csv") as f:
        index = list(csv.DictReader(f))
    assert [(r["clip_id"], r["row"]) for r in index] == [("cargo/a", "0"), ("tug/c", "1")]

    meta = json.loads((out / "meta.json").read_text())
    [failed] = meta["failed"]
    assert failed["clip_id"] == "cargo/b"
    assert failed["reason"]


def test_too_many_failures_stop_the_build_before_writing(write_wav, tmp_path):
    # 1 failure in 2 clips is 50%, far above the default limit of 0.1%: that many
    # unreadable files points to a bug or a wrong path, not to a few bad recordings.
    broken = tmp_path / "cargo_b.wav"
    broken.write_bytes(b"not a wav file")
    recordings = [
        make_recording(write_wav, "cargo", "a", tone(5.0, 32000)),
        Recording(id="cargo/b", path=broken, label="cargo", group="mmsi:1"),
    ]
    out = tmp_path / "cache"

    with pytest.raises(RuntimeError, match="1 of 2"):
        build_cache(recordings, out, SETTINGS)

    assert not (out / "logmel.npy").exists()
