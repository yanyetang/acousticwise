import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from acousticwise.audio.features import log_mel, normalization_offset_db
from acousticwise.audio.io import load_audio
from acousticwise.audio.preprocess import check_not_silent, make_windows
from acousticwise.models.baseline import mfcc_features


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

    def __init__(self, model, classes: list[str], cfg: dict):
        self.model = model
        self.classes = classes
        self.cfg = cfg

    @classmethod
    def load(cls, run_dir: Path) -> "Predictor":
        # The run's own config, so features are made exactly as in training; no feature cache.
        cfg = json.loads((run_dir / "config.json").read_text())["config"]
        classes = json.loads((run_dir / "labels.json").read_text())
        return cls(joblib.load(run_dir / "model.joblib"), classes, cfg)

    def predict(self, wav_path: Path) -> Prediction:
        """Score each window of the file; the file's scores are the mean over its windows."""
        audio, features = self.cfg["audio"], self.cfg["features"]
        sr = audio["sample_rate"]
        samples = load_audio(wav_path, sr)
        check_not_silent(samples, self.cfg["validation"]["silence_rms"])
        windows = make_windows(
            samples, sr, audio["window_seconds"], audio["hop_seconds"], audio["min_seconds"]
        )

        def mel_of(x: np.ndarray) -> np.ndarray:
            return log_mel(x, sr, features["n_mels"], features["n_fft"], features["hop_length"])

        # The same features as training: each window's log-mel shifted by its offset, as MFCCs.
        mels = np.stack([mel_of(w) for w in windows])
        offsets = np.array([normalization_offset_db(w, audio["normalization"]) for w in windows])
        x = mfcc_features(mels, offsets, features["n_mfcc"])

        # One row per window, a probability per class; labels.json was written from the
        # model's classes_ (train.py), so the columns are in its order.
        probabilities = self.model.predict_proba(x).mean(axis=0)
        scores = {c: float(p) for c, p in zip(self.classes, probabilities, strict=True)}
        label = max(scores, key=scores.__getitem__)
        return Prediction(label=label, scores=scores, waveform=samples, sr=sr, mel=mel_of(samples))
