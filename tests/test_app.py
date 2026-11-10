import ast
import json
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from acousticwise.train import train_baseline
from tests.conftest import FREQS, tone

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"


def test_the_app_imports_only_inference():
    tree = ast.parse(APP.read_text())
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules |= {
        a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names
    }

    assert {m for m in modules if m and m.startswith("acousticwise")} == {"acousticwise.inference"}


def test_the_app_command_is_installed():
    commands = entry_points(group="console_scripts", name="acousticwise-app")

    assert len(commands) == 1
    assert callable(next(iter(commands)).load())


@pytest.fixture
def run_dir(baseline_inputs, tmp_path):
    """A trained baseline run on the cargo and tug tones."""
    cfg, _ = baseline_inputs
    path = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=path)
    return path


def classify(run_dir: Path, wav: bytes, monkeypatch) -> AppTest:
    """Open the app on run_dir, upload wav and press Classify."""
    monkeypatch.setenv("ACOUSTICWISE_RUN_DIR", str(run_dir))
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    at.file_uploader[0].upload("upload.wav", wav, "audio/wav").run()
    at.button[0].click().run()
    return at


def page_text(at: AppTest) -> str:
    """Every piece of text the page shows, banners included."""
    kinds = ["title", "header", "subheader", "markdown", "caption", "text"]
    kinds += ["info", "warning", "error", "success"]
    return "\n".join(str(element.value) for kind in kinds for element in getattr(at, kind))


def wav_bytes(write_wav, samples, sr=32000) -> bytes:
    return write_wav(samples, sr, name="made.wav").read_bytes()


def test_a_valid_upload_shows_the_predicted_class_and_model_confidence(
    run_dir, write_wav, monkeypatch
):
    at = classify(run_dir, wav_bytes(write_wav, tone(5.0, 32000, FREQS["tug"])), monkeypatch)

    assert not at.exception
    assert not at.error
    text = page_text(at)
    assert "tug" in text
    assert "model confidence" in text.lower()
    assert "one hydrophone network" in text.lower()  # PRD A2
    assert "few vessels" not in text.lower()  # tug has many training vessels


def test_a_few_vessels_class_on_top_adds_a_note(run_dir, write_wav, monkeypatch):
    # Rename the run's "tug" class to "rescue", one of the classes with few training vessels.
    (run_dir / "labels.json").write_text(json.dumps(["cargo", "rescue"]))

    at = classify(run_dir, wav_bytes(write_wav, tone(5.0, 32000, FREQS["tug"])), monkeypatch)

    assert not at.exception
    assert "few vessels" in page_text(at).lower()


INVALID_UPLOADS = {
    "unreadable": lambda write_wav: b"this is not audio",
    "silent": lambda write_wav: wav_bytes(write_wav, np.zeros(5 * 32000, "float32")),
    "too short": lambda write_wav: wav_bytes(write_wav, tone(3.0, 32000)),
}


@pytest.mark.parametrize("case", list(INVALID_UPLOADS))
def test_an_invalid_upload_shows_an_error_banner_not_a_traceback(
    run_dir, write_wav, monkeypatch, case
):
    at = classify(run_dir, INVALID_UPLOADS[case](write_wav), monkeypatch)

    assert not at.exception
    assert len(at.error) == 1
    assert "Traceback" not in at.error[0].value


def test_a_model_failure_shows_a_friendly_banner_without_internals(run_dir, write_wav, monkeypatch):
    # A broken run: the config no longer matches the model, so scikit-learn raises inside
    # the predictor, which wraps it as InferenceError.
    record = json.loads((run_dir / "config.json").read_text())
    record["config"]["features"]["n_mfcc"] = 13
    (run_dir / "config.json").write_text(json.dumps(record))

    at = classify(run_dir, wav_bytes(write_wav, tone(5.0, 32000, FREQS["tug"])), monkeypatch)

    assert not at.exception
    assert len(at.error) == 1
    assert "StandardScaler" not in page_text(at)  # scikit-learn's message stays in the log
