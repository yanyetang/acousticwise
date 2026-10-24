import csv
import os
from pathlib import Path

from acousticwise.data.dataset import Recording


def read_split(bundle_dir: Path, fold: int, split: str) -> list[Recording]:
    """Read one official split (train, validation or test) of one fold into recordings."""
    bundle_dir = Path(bundle_dir)
    fold_dir = bundle_dir / f"fold_{fold}"
    metadata_file = fold_dir / f"metadata_5s_{split}.csv"

    recordings = []
    with open(metadata_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = Path(os.path.normpath(fold_dir / row["path"]))
            id = str(path.relative_to(bundle_dir / "audio").with_suffix(""))  # remove ".wav"
            group = f"mmsi:{row['MMSI']}" if row["label"] != "background" else f"day:{row['date']}"
            meta = {"duration_sec": float(row["duration_sec"])}
            recording = Recording(id=id, path=path, label=row["label"], group=group, meta=meta)
            recordings.append(recording)

    return recordings
