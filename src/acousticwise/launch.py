import subprocess
import sys
from pathlib import Path

APP = Path(__file__).parents[2] / "app" / "streamlit_app.py"


def main() -> None:
    """`uv run acousticwise-app`: start the Streamlit app."""
    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP)])
    sys.exit(result.returncode)
