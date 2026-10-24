from pathlib import Path

from acousticwise.data.splits import SPLITS
from acousticwise.data.uniqueship import read_split


def build_report(bundle_dir: Path, folds: list[int], window_seconds: float = 5.0) -> list[dict]:
    """Per fold, split and class: clips, groups, seconds, short clips and the shortest clip."""
    rows = []
    for fold in folds:
        for split in SPLITS:
            recordings = read_split(bundle_dir, fold, split)
            for label in sorted({r.label for r in recordings}):
                durations = [r.meta["duration_sec"] for r in recordings if r.label == label]
                rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "label": label,
                        "clips": len(durations),
                        "groups": len({r.group for r in recordings if r.label == label}),
                        "seconds": sum(durations),
                        "short_clips": sum(d < window_seconds for d in durations),
                        "min_seconds": min(durations),
                    }
                )
    return rows
