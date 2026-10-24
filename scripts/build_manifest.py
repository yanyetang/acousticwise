"""Build the split manifest from the bundle's official fold CSVs and check it for leakage.

Usage:
    uv run python scripts/build_manifest.py [--bundle DIR] [--out FILE]

The manifest is not committed (about 28 MB); its SHA-256 is recorded in DATA.md, so a
rebuilt manifest can be checked against it.
"""

import argparse
import hashlib
from pathlib import Path

import yaml

from acousticwise.data.splits import build_manifest, check_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bundle", help="folder with fold_1 … fold_5 (default: data.bundle_dir)")
    parser.add_argument("--out", help="manifest file (default: data.splits_path)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bundle = Path(args.bundle or cfg["data"]["bundle_dir"])
    out = Path(args.out or cfg["data"]["splits_path"])

    manifest = build_manifest(bundle, folds=[1, 2, 3, 4, 5])
    check_manifest(manifest)
    save_manifest(manifest, out)
    print(f"{out}: 5 folds, no shared clips or groups between splits")
    print(f"SHA-256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
