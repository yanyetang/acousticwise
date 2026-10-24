"""Build the feature caches, one per sample rate, from the unzipped UniqueShip bundle.

Usage:
    uv run python scripts/cache_features.py [--bundle DIR] [--out DIR] [--sample-rates 32000 16000]

Writes <out>/sr<rate>/ with logmel.npy, index.csv and meta.json (technical design, section 5).
"""

import argparse
import time
from pathlib import Path

import yaml

from acousticwise.audio.cache import build_cache, settings_for
from acousticwise.data.uniqueship import read_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bundle", help="unzipped bundle folder (default: data.bundle_dir)")
    parser.add_argument("--out", help="cache folder (default: data.cache_dir)")
    parser.add_argument("--sample-rates", type=int, nargs="+", default=[32000, 16000])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bundle = Path(args.bundle or cfg["data"]["bundle_dir"])
    out = Path(args.out or cfg["data"]["cache_dir"])

    # Fold 1's train, validation and test splits together hold every clip once.
    recordings = [
        rec
        for split in ("train", "validation", "test")
        for rec in read_split(bundle, fold=1, split=split)
    ]
    print(f"{len(recordings)} clips in {bundle}")

    for sample_rate in args.sample_rates:
        settings = settings_for(cfg, sample_rate)
        cache_dir = out / f"sr{sample_rate}"
        start = time.time()
        build_cache(recordings, cache_dir, settings)
        print(f"{cache_dir}: built in {time.time() - start:.0f} s with {settings}")


if __name__ == "__main__":
    main()
