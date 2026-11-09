import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from acousticwise.audio.features import log_mel, normalization_offset_db
from acousticwise.audio.io import load_audio
from acousticwise.audio.preprocess import check_not_silent, make_windows
from acousticwise.models.baseline import mfcc_features
from acousticwise.models.cnn import ConvNet, standardize


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

    def __init__(
        self, model, classes: list[str], cfg: dict, input_scaling: dict[str, float] | None = None
    ):
        self.model = model
        self.classes = classes
        self.cfg = cfg
        self.input_scaling = input_scaling  # CNN only: the training clips' mean and std

    @classmethod
    def load(cls, run_dir: Path) -> "Predictor":
        # The run's own config, so features are made exactly as in training; no feature cache.
        record = json.loads((run_dir / "config.json").read_text())
        cfg = record["config"]
        classes = json.loads((run_dir / "labels.json").read_text())
        if (run_dir / "model.pt").exists():  # CNN or CRNN
            model = ConvNet(len(classes), cfg["cnn"]["time_pooling"])
            # map_location: weights saved on a GPU still load on a machine without one.
            model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
            model.eval()  # dropout off, batch norm uses its saved averages
            return cls(model, classes, cfg, record["input_scaling"])
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

        mels = np.stack([mel_of(w) for w in windows])
        offsets = np.array([normalization_offset_db(w, audio["normalization"]) for w in windows])
        probabilities = self._window_probabilities(mels, offsets).mean(axis=0)
        scores = {c: float(p) for c, p in zip(self.classes, probabilities, strict=True)}
        label = max(scores, key=scores.__getitem__)
        return Prediction(label=label, scores=scores, waveform=samples, sr=sr, mel=mel_of(samples))

    def _window_probabilities(self, mels: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """One row per window, a probability per class in labels.json order, from the same
        inputs the model was trained on."""
        if self.input_scaling is not None:  # CNN: shifted log-mels, scaled like training
            x = (mels + offsets[:, None, None]).astype(np.float32)
            x = standardize(x, self.input_scaling)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(x))
            # Softmax over the classes turns each window's logits into probabilities.
            return torch.softmax(logits, dim=1).numpy()
        # Baseline: MFCC summaries; labels.json was written from the model's classes_
        # (train.py), so the columns are in its order.
        x = mfcc_features(mels, offsets, self.cfg["features"]["n_mfcc"])
        return self.model.predict_proba(x)
