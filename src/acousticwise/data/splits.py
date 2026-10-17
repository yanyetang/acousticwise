import json
from itertools import combinations
from pathlib import Path

from acousticwise.data.dataset import Recording
from acousticwise.data.uniqueship import read_split

SPLITS = ("train", "validation", "test")


class LeakageError(Exception):
    """A clip or group appears in more than one split of the same fold."""


def _check_pairs(splits: dict[str, list[tuple[str, str]]]) -> None:
    """Raise LeakageError if two splits share a group or a clip id.

    `splits` maps a split name to its (clip id, group) pairs.
    """
    for a, b in combinations(splits, 2):
        for kind, index in (("groups", 1), ("clip ids", 0)):
            shared = {p[index] for p in splits[a]} & {p[index] for p in splits[b]}
            if shared:
                raise LeakageError(f"{a} and {b} share {kind}: {sorted(shared)}")


def check_no_leakage(splits: dict[str, list[Recording]]) -> None:
    """Raise LeakageError if any two splits share a clip id or a group."""
    _check_pairs({name: [(r.id, r.group) for r in recs] for name, recs in splits.items()})


def check_manifest(manifest: dict) -> None:
    """Raise LeakageError if any fold of the manifest shares a clip id or group between splits."""
    for fold, splits in manifest["folds"].items():
        pairs = {name: [(e["id"], e["group"]) for e in entries] for name, entries in splits.items()}
        try:
            _check_pairs(pairs)
        except LeakageError as e:
            raise LeakageError(f"fold {fold}: {e}") from e


def build_manifest(bundle_dir: Path, folds: list[int]) -> dict:
    """Turn the official fold CSVs into the split manifest (ids and groups only, no audio)."""
    return {
        "source": "official",
        "folds": {
            str(fold): {
                split: [
                    {"id": r.id, "group": r.group}
                    for r in read_split(bundle_dir, fold=fold, split=split)
                ]
                for split in SPLITS
            }
            for fold in folds
        },
    }


def save_manifest(manifest: dict, path: Path) -> None:
    """Write the manifest as readable JSON, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_manifest(path: Path) -> dict:
    """Read a manifest written by save_manifest."""
    return json.loads(path.read_text())
