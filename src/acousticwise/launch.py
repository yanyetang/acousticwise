import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).parents[2]
APP = REPO / "app" / "streamlit_app.py"
CONFIG = REPO / "configs" / "default.yaml"


def main() -> None:
    """`uv run acousticwise-app`: start the Streamlit app on the configured run."""
    cfg = yaml.safe_load(CONFIG.read_text())
    env = os.environ.copy()
    env.setdefault("ACOUSTICWISE_RUN_DIR", str(REPO / cfg["data"]["app_run_dir"]))
    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP)], env=env)
    sys.exit(result.returncode)
