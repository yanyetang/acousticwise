import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from acousticwise.audio.features import log_mel, normalization_offset_db
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


# The columns of index.csv; "row" is the clip's row in logmel.npy.
INDEX_COLUMNS = ["clip_id", "row", "label", "padded", "peak_db", "rms_db"]


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

    mels, rows, failed = [], [], []
    for rec in recordings:
        try:
            mel, fields = _process_clip(rec, settings)
        except UnsupportedFileError as e:
            failed.append({"clip_id": rec.id, "reason": str(e)})
            continue
        rows.append({"clip_id": rec.id, "row": len(mels), "label": rec.label, **fields})
        mels.append(mel)

    if len(failed) > 0 and len(failed) / len(recordings) > max_failed_fraction:
        raise RuntimeError(
            f"Too many clips failed to load: {len(failed)} of {len(recordings)} = "
            f"{len(failed) / len(recordings):.3f} > {max_failed_fraction:.3f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "logmel.npy", np.stack(mels))
    with open(out_dir / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "settings": asdict(settings),
        "padded_count": sum(r["padded"] for r in rows),
        "failed": failed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _process_clip(rec: Recording, settings: CacheSettings) -> tuple[np.ndarray, dict]:
    """Load one clip; return its float16 log-mel and its padded flag and levels."""
    sr = settings.sample_rate
    samples = load_audio(rec.path, target_sr=sr)
    padded = len(samples) < round(settings.window_seconds * sr)
    # A dataset clip is at most 5 s, so it gives exactly one window.
    window = make_windows(
        samples, sr, settings.window_seconds, settings.window_seconds, settings.min_seconds
    )[0]
    mel = log_mel(window, sr, settings.n_mels, settings.n_fft, settings.hop_length)
    fields = {
        "padded": padded,
        "peak_db": -normalization_offset_db(window, method="peak"),
        "rms_db": -normalization_offset_db(window, method="rms"),
    }
    return mel.astype(np.float16), fields


def load_cache(cache_dir: Path, settings: CacheSettings) -> np.ndarray:
    """Open logmel.npy memory-mapped; refuse a cache built with other settings."""
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)
    saved, current = meta["settings"], asdict(settings)
    diffs = [
        f"{name}: cache {saved.get(name)}, config {value}"
        for name, value in current.items()
        if saved.get(name) != value
    ]
    if diffs:
        raise ValueError(
            f"Cache settings differ from the config: {'; '.join(diffs)}. Rebuild the cache."
        )

    return np.load(cache_dir / "logmel.npy", mmap_mode="r")
