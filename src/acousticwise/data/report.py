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


# The heading and alignment lines of each fold's table; ---: right-aligns a column.
TABLE_HEADER = [
    "| Split | Class | Clips | Vessels or days | Hours | Short clips | Shortest (s) |",
    "|---|---|---:|---:|---:|---:|---:|",
]


def format_report(rows: list[dict]) -> str:
    """Markdown: one section per fold, with one table row per split and class."""
    lines = []
    for fold in sorted({row["fold"] for row in rows}):
        if lines:
            lines.append("")  # a blank line between folds
        lines += [f"## Fold {fold}", "", *TABLE_HEADER]
        for row in [r for r in rows if r["fold"] == fold]:  # keep build_report's order
            lines.append(
                f"| {row['split']} | {row['label']} | {row['clips']} | {row['groups']} | "
                f"{row['seconds'] / 3600:.2f} | {row['short_clips']} | {row['min_seconds']:.1f} |"
            )
    return "\n".join(lines) + "\n"
