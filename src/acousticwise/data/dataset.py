from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Recording:
    """One clip: the unit that is classified and split."""

    id: str  # path under audio/ without .wav, e.g. "cargo/<start>_<end>"
    path: Path  # absolute path to the WAV
    label: str  # class name
    group: str  # leakage-safe group: "mmsi:<MMSI>" or "day:<date>" for background
    meta: dict = field(default_factory=dict)
