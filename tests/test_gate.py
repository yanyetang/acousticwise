"""Reproducibility gate, tier 1 (PRD section 3): the whole chain on the synthetic fixture."""

import json
import subprocess
import sys

import pytest
import yaml

from acousticwise.inference import Predictor
from acousticwise.train import train_cnn
from tests.conftest import FREQS, tone


def run(*args: str) -> None:
    """Run a project command line the way the README does, failing loudly on an error."""
    subprocess.run([sys.executable, "-m", *args], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("model", ["baseline", "cnn"])
def test_a_fresh_run_trains_evaluates_and_classifies_in_the_app(
    baseline_inputs, tmp_path, write_wav, model
):
    cfg, _ = baseline_inputs
    cfg["data"]["runs_dir"] = str(tmp_path / "runs")
    cfg["cnn"].update(epochs=2, batch_size=4)  # a few steps, for speed
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))

    run("acousticwise.train", "--config", str(config), "--fold", "1", "--model", model)
    (run_dir,) = (tmp_path / "runs").iterdir()
    run("acousticwise.evaluate", str(run_dir), "--out", str(tmp_path / "summary.json"))
    wav = write_wav(tone(5.0, 32000, freq=FREQS["tug"], amplitude=0.2), 32000, name="upload.wav")
    prediction = Predictor.load(run_dir).predict(wav)

    test = json.loads((run_dir / "metrics.json").read_text())["test"]
    assert test["clips"] == 2 * len(FREQS)  # every fixture test clip was scored
    assert set(prediction.scores) == set(FREQS)
    assert sum(prediction.scores.values()) == pytest.approx(1.0)


def losses(cfg: dict, run_dir) -> list[float]:
    """Train a CNN run and return its training loss per epoch, as saved in metrics.json."""
    train_cnn(cfg, fold=1, run_dir=run_dir)
    history = json.loads((run_dir / "metrics.json").read_text())["validation"]["history"]
    return [epoch["loss"] for epoch in history]


@pytest.mark.parametrize("time_pooling", ["mean", "lstm"])  # CNN, CRNN
def test_the_same_seed_twice_gives_the_same_training_losses(
    baseline_inputs, tmp_path, time_pooling
):
    cfg, _ = baseline_inputs
    cfg["cnn"].update(time_pooling=time_pooling, epochs=3, batch_size=4)

    first = losses(cfg, tmp_path / "first")
    second = losses(cfg, tmp_path / "second")
    cfg["seed"] += 1
    other = losses(cfg, tmp_path / "other")

    assert first == second  # exactly, not approximately: same seed, same run
    assert first != other  # and the seed is what decides it
