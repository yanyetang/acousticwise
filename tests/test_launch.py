import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from acousticwise import launch

REPO = Path(__file__).parents[1]


@pytest.fixture
def streamlit_env(monkeypatch):
    """Run launch.main with Streamlit replaced; return the env it would have started with."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append(dict(kwargs.get("env", os.environ)))  # what the child would see
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)

    def start() -> dict:
        with pytest.raises(SystemExit):
            launch.main()
        return calls[0]

    return start


def test_the_app_command_loads_the_configured_run(monkeypatch, streamlit_env):
    monkeypatch.delenv("ACOUSTICWISE_RUN_DIR", raising=False)
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())

    env = streamlit_env()

    # Absolute, so the app finds the run whichever folder the command is started from.
    assert env.get("ACOUSTICWISE_RUN_DIR") == str(REPO / cfg["data"]["app_run_dir"])


def test_a_run_dir_already_set_wins_over_the_config(monkeypatch, streamlit_env, tmp_path):
    monkeypatch.setenv("ACOUSTICWISE_RUN_DIR", str(tmp_path))

    env = streamlit_env()

    assert env.get("ACOUSTICWISE_RUN_DIR") == str(tmp_path)


def test_the_configured_run_is_a_committed_cnn_run():
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    run = REPO / cfg["data"]["app_run_dir"]

    record = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())

    # Run folders are named by kind (technical design, section 5); mean pooling is the CNN.
    assert run.name.startswith("cnn_fold")
    assert record["config"]["cnn"]["time_pooling"] == "mean"
    assert json.loads((run / "labels.json").read_text())
    assert "macro_f1_headline" in metrics["test"]
