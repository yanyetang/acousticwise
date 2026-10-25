import csv
from pathlib import Path

import pytest

# The columns of UniqueShip's fold CSVs that the project reads. The real CSVs have
# 42 columns; the rest are optional AIS and CTD context.
COLUMNS = ["label", "path", "duration_sec", "sample_rate", "date", "MMSI", "hydrophone"]


def clip_row(label, stem, mmsi, date="20220103", duration=5.0):
    """One CSV row. `path` is relative to the fold folder, as in the real bundle."""
    return {
        "label": label,
        "path": f"../audio/{label}/{stem}.wav",
        "duration_sec": duration,
        "sample_rate": 32000,
        "date": date,
        "MMSI": mmsi,
        "hydrophone": "" if mmsi == 0 else "ICLISTENHF1234",
    }


@pytest.fixture
def make_bundle(tmp_path):
    """Write fold CSVs shaped like UniqueShip; return the bundle folder.

    Usage: make_bundle({(1, "train"): [clip_row(...), ...], (1, "test"): [...]})
    """

    def _make(splits: dict[tuple[int, str], list[dict]]) -> Path:
        root = tmp_path / "bundle"
        for (fold, split), rows in splits.items():
            folder = root / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / f"metadata_5s_{split}.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        return root

    return _make


def tone(seconds, sr, freq=440.0, amplitude=0.5, channels=1):
    """A sine tone as float32 samples; shape (n,) for mono, (n, channels) otherwise."""
    import numpy as np

    t = np.arange(int(round(seconds * sr))) / sr
    x = (amplitude * np.sin(2 * np.pi * freq * t)).astype("float32")
    return x if channels == 1 else np.stack([x] * channels, axis=1)


@pytest.fixture
def write_wav(tmp_path):
    """Write samples to a WAV (32-bit PCM, like UniqueShip) and return its path."""
    import soundfile as sf

    def _write(samples, sr, name="clip.wav", subtype="PCM_32"):
        path = tmp_path / name
        sf.write(path, samples, sr, subtype=subtype)
        return path

    return _write


FREQS = {"cargo": 200.0, "tug": 1000.0}  # one easy-to-tell tone per class


@pytest.fixture
def baseline_inputs(tmp_path, write_wav):
    """A config whose cache and manifest point at a tiny fixture: 6 train, 2 validation and
    2 test clips per class, each clip a tone of its class at its own loudness and from its
    own vessel. The last tug test clip is a cargo tone, so evaluation has one known mistake."""
    import yaml

    from acousticwise.audio.cache import build_cache, settings_for
    from acousticwise.data.dataset import Recording
    from acousticwise.data.splits import save_manifest

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["cache_dir"] = str(tmp_path / "cache")
    cfg["data"]["splits_path"] = str(tmp_path / "splits.json")
    cfg["evaluation"]["nine_class_subset"] = list(FREQS)  # the headline: the fixture's classes

    recordings, entries = [], {"train": [], "validation": [], "test": []}
    for label, freq in FREQS.items():
        for k in range(10):
            split = "train" if k < 6 else "validation" if k < 8 else "test"
            sound = FREQS["cargo"] if (label, k) == ("tug", 9) else freq
            samples = tone(5.0, 32000, freq=sound, amplitude=0.1 + 0.05 * k)
            path = write_wav(samples, 32000, name=f"{label}_{k}.wav")
            clip_id, group = f"{label}/{k}", f"mmsi:{label}{k}"
            recordings.append(Recording(id=clip_id, path=path, label=label, group=group))
            entries[split].append({"id": clip_id, "group": group})

    sample_rate = cfg["audio"]["sample_rate"]
    build_cache(recordings, tmp_path / "cache" / f"sr{sample_rate}", settings_for(cfg, sample_rate))
    manifest = {"source": "official", "folds": {"1": entries}}
    save_manifest(manifest, tmp_path / "splits.json")
    return cfg, manifest
