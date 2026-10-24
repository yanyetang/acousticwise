import json

import joblib
import pytest
import yaml

from acousticwise.audio.cache import build_cache, settings_for
from acousticwise.data.dataset import Recording
from acousticwise.data.splits import save_manifest
from acousticwise.train import train_baseline
from tests.conftest import tone

FREQS = {"cargo": 200.0, "tug": 1000.0}  # one easy-to-tell tone per class


@pytest.fixture
def baseline_inputs(tmp_path, write_wav):
    """A config whose cache and manifest point at a tiny fixture: 6 train and 2 validation
    clips per class, each clip a tone of its class at its own loudness."""
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["cache_dir"] = str(tmp_path / "cache")
    cfg["data"]["splits_path"] = str(tmp_path / "splits.json")

    recordings, entries = [], {"train": [], "validation": [], "test": []}
    for label, freq in FREQS.items():
        for k in range(8):
            split = "train" if k < 6 else "validation"
            samples = tone(5.0, 32000, freq=freq, amplitude=0.1 + 0.05 * k)
            path = write_wav(samples, 32000, name=f"{label}_{k}.wav")
            clip_id, group = f"{label}/{k}", f"mmsi:{label}{k}"
            recordings.append(Recording(id=clip_id, path=path, label=label, group=group))
            entries[split].append({"id": clip_id, "group": group})

    sample_rate = cfg["audio"]["sample_rate"]
    build_cache(recordings, tmp_path / "cache" / f"sr{sample_rate}", settings_for(cfg, sample_rate))
    manifest = {"source": "official", "folds": {"1": entries}}
    save_manifest(manifest, tmp_path / "splits.json")
    return cfg, manifest


def test_train_baseline_writes_a_complete_run_directory(baseline_inputs, tmp_path):
    cfg, manifest = baseline_inputs
    run_dir = tmp_path / "run"

    train_baseline(cfg, fold=1, run_dir=run_dir)

    model = joblib.load(run_dir / "model.joblib")
    assert list(model.classes_) == ["cargo", "tug"]
    assert json.loads((run_dir / "labels.json").read_text()) == ["cargo", "tug"]

    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["config"] == cfg
    assert saved["fold"] == 1
    assert {"python", "numpy", "scikit-learn"} <= set(saved["versions"])

    # Only the fold that was trained, so the run directory says exactly which clips it saw.
    assert json.loads((run_dir / "splits.json").read_text()) == manifest["folds"]["1"]

    # Validation scores only; the test split is scored later by evaluate.py.
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"validation"}
    assert metrics["validation"]["accuracy"] == 1.0
    assert metrics["validation"]["macro_f1"] == 1.0
    assert metrics["validation"]["clips"] == 4
