from pathlib import Path


def train_baseline(cfg: dict, fold: int, run_dir: Path) -> None:
    """Train the baseline on one fold's training clips from the cache; write the run directory."""
    raise NotImplementedError
