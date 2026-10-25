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


def settings_for(cfg: dict, sample_rate: int) -> CacheSettings:
    """Cache settings at one sample rate, from the config's settings at its own rate."""
    audio, features = cfg["audio"], cfg["features"]
    # Scale n_fft and hop with the rate so every rate gives 64 x 313 frames per window.
    scale = sample_rate / audio["sample_rate"]
    return CacheSettings(
        sample_rate=sample_rate,
        window_seconds=audio["window_seconds"],
        min_seconds=audio["min_seconds"],
        n_fft=round(features["n_fft"] * scale),
        hop_length=round(features["hop_length"] * scale),
        n_mels=features["n_mels"],
        cache_format=features["cache_format"],
    )


# The columns of index.csv; "row" is the clip's row in logmel.npy.
INDEX_COLUMNS = ["clip_id", "row", "label", "padded", "peak_db", "rms_db"]

# One index.csv entry on load: the clip's row in logmel.npy, its label and its offset in dB.
IndexEntry = tuple[int, str, float]

# The index.csv level each normalization method subtracts; "none" subtracts nothing.
LEVEL_COLUMNS = {"peak": "peak_db", "rms": "rms_db", "none": None}


def build_cache(
    recordings: list[Recording],
    out_dir: Path,
    settings: CacheSettings,
    max_failed_fraction: float = 0.001,
) -> None:
    """Write logmel.npy, index.csv and meta.json for the recordings into out_dir."""
    _check_unique_ids(recordings)

    mels, rows, failed = [], [], []
    for rec in recordings:
        try:
            mel, fields = _process_clip(rec, settings)
        except UnsupportedFileError as e:
            failed.append({"clip_id": rec.id, "reason": str(e)})
            continue
        rows.append({"clip_id": rec.id, "row": len(mels), "label": rec.label, **fields})
        mels.append(mel)

    _check_failures(len(failed), len(recordings), max_failed_fraction)
    _write_cache(out_dir, np.stack(mels), rows, settings, failed)


def _check_unique_ids(recordings: list[Recording]) -> None:
    """Raise ValueError naming the first clip id that appears twice."""
    seen = set()
    for rec in recordings:
        if rec.id in seen:
            raise ValueError(f"Duplicate clip id: {rec.id}")
        seen.add(rec.id)


def _check_failures(n_failed: int, n_total: int, max_failed_fraction: float) -> None:
    """Raise RuntimeError when more than max_failed_fraction of the clips failed to load."""
    if n_failed > 0 and n_failed / n_total > max_failed_fraction:
        raise RuntimeError(
            f"Too many clips failed to load: {n_failed} of {n_total} = "
            f"{n_failed / n_total:.3f} > {max_failed_fraction:.3f}"
        )


def _write_cache(
    out_dir: Path,
    logmel: np.ndarray,
    rows: list[dict],
    settings: CacheSettings,
    failed: list[dict],
) -> None:
    """Write the three cache files into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "logmel.npy", logmel)
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


def read_index(cache_dir: Path, normalization: str) -> dict[str, IndexEntry]:
    """clip_id -> (row in logmel.npy, label, dB to add for the normalization method)."""
    if normalization not in LEVEL_COLUMNS:
        raise ValueError(
            f"Unknown normalization {normalization!r}. Use {', '.join(LEVEL_COLUMNS)}."
        )
    column = LEVEL_COLUMNS[normalization]
    index = {}
    with open(cache_dir / "index.csv") as f:
        for r in csv.DictReader(f):
            # The index stores the window's levels, so normalizing is subtracting one of them.
            offset = -float(r[column]) if column else 0.0
            index[r["clip_id"]] = (int(r["row"]), r["label"], offset)
    return index


def load_features(cfg: dict) -> tuple[np.ndarray, dict[str, IndexEntry]]:
    """The config's cache (refused if built with other settings) and its index.

    Training and evaluation both load through here, so they cannot read features differently.
    """
    sample_rate = cfg["audio"]["sample_rate"]
    cache_dir = Path(cfg["data"]["cache_dir"]) / f"sr{sample_rate}"
    logmel = load_cache(cache_dir, settings_for(cfg, sample_rate))
    return logmel, read_index(cache_dir, cfg["audio"]["normalization"])
