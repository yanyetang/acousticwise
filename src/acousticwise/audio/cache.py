import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from acousticwise.audio.features import log_mel
from acousticwise.audio.io import load_audio
from acousticwise.audio.preprocess import make_windows
from acousticwise.data.dataset import Recording
from acousticwise.errors import UnsupportedFileError


@dataclass(frozen=True)
class CacheSettings:
    """The feature settings a cache is built with; saved in meta.json."""

    sample_rate: int
    window_seconds: float
    min_seconds: float
    n_fft: int
    hop_length: int
    n_mels: int
    cache_format: int


def build_cache(
    recordings: list[Recording],
    out_dir: Path,
    settings: CacheSettings,
    max_failed_fraction: float = 0.001,
) -> None:
    """Write logmel.npy, index.csv and meta.json for the recordings into out_dir."""
    seen = set()
    for rec in recordings:
        if rec.id in seen:
            raise ValueError(f"Duplicate clip id: {rec.id}")
        seen.add(rec.id)

    out_dir.mkdir(parents=True, exist_ok=True)
    sr = settings.sample_rate
    mels, row, failed = [], [], []
    for rec in recordings:
        try:
            samples = load_audio(rec.path, target_sr=sr)
        except UnsupportedFileError as e:
            failed.append({"clip_id": rec.id, "reason": str(e)})
            continue
        # A dataset clip is at most 5 s, so it gives exactly one window.
        padded = len(samples) < round(settings.window_seconds * sr)
        window = make_windows(
            samples, sr, settings.window_seconds, settings.window_seconds, settings.min_seconds
        )[0]
        logmel = log_mel(window, sr, settings.n_mels, settings.n_fft, settings.hop_length)
        mels.append(logmel.astype(np.float16))
        row.append({"clip_id": rec.id, "row": len(mels) - 1, "label": rec.label, "padded": padded})

    if len(failed) > 0 and len(failed) / len(recordings) > max_failed_fraction:
        raise RuntimeError(
            f"Too many clips failed to load: {len(failed)} of {len(recordings)} = "
            f"{len(failed) / len(recordings):.3f} > {max_failed_fraction:.3f}"
        )

    np.save(out_dir / "logmel.npy", np.stack(mels))
    with open(out_dir / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "row", "label", "padded"])
        writer.writeheader()
        writer.writerows(row)
    meta = {
        "settings": asdict(settings),
        "padded_count": sum(r["padded"] for r in row),
        "failed": failed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
