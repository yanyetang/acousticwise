#!/usr/bin/env bash
# Upload src/acousticwise, scripts/cache_features.py and configs/default.yaml as the private Kaggle dataset
# yanyetang/acousticwise-src, so Kaggle notebooks run the package code.
# First upload creates the dataset; later runs add a new version.
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=$(mktemp -d)
cp -R src/acousticwise "$stage/acousticwise"
find "$stage" -name __pycache__ -type d -prune -exec rm -rf {} +
mkdir -p "$stage/configs" && cp configs/default.yaml "$stage/configs/"
mkdir -p "$stage/scripts" && cp scripts/cache_features.py "$stage/scripts/"
cat > "$stage/dataset-metadata.json" <<JSON
{"title": "acousticwise-src", "id": "yanyetang/acousticwise-src", "licenses": [{"name": "other"}]}
JSON
if kaggle datasets status yanyetang/acousticwise-src >/dev/null 2>&1; then
  kaggle datasets version -p "$stage" --dir-mode zip -m "${1:-update package}"
else
  kaggle datasets create -p "$stage" --dir-mode zip
fi
rm -rf "$stage"
