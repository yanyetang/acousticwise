from dataclasses import dataclass
from pathlib import Path

from acousticwise.data.dataset import Recording


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


def build_cache(recordings: list[Recording], out_dir: Path, settings: CacheSettings) -> None:
    """Write logmel.npy, index.csv and meta.json for the recordings into out_dir."""
    raise NotImplementedError
