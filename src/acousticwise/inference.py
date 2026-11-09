from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Prediction:
    """One file's result: the top class, per-class model confidence, and what the app plots."""

    label: str
    scores: dict[str, float]  # "model confidence", not calibrated
    waveform: np.ndarray  # for display
    sr: int
    mel: np.ndarray  # for display


class Predictor:
    """The only API the app uses: load a trained run, classify a WAV."""

    @classmethod
    def load(cls, run_dir: Path) -> "Predictor":
        raise NotImplementedError

    def predict(self, wav_path: Path) -> Prediction:
        raise NotImplementedError
