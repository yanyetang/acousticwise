from pathlib import Path


def build_report(bundle_dir: Path, folds: list[int]) -> list[dict]:
    """Per fold, split and class: clips, groups, seconds, short clips and the shortest clip."""
    raise NotImplementedError
