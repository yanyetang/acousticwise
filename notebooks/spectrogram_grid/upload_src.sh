#!/usr/bin/env bash
# Upload the committed src/acousticwise, scripts/ and configs/default.yaml, plus the split
# manifest when it exists, as the private Kaggle dataset yanyetang/acousticwise-src, so Kaggle
# notebooks run the package code. COMMIT records the git commit the code came from.
# First upload creates the dataset; later runs add a new version.
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=$(mktemp -d)
# The last commit, not the working tree, so unfinished edits are never uploaded.
git archive HEAD src/acousticwise scripts configs/default.yaml | tar -x -C "$stage"
mv "$stage/src/acousticwise" "$stage/acousticwise" && rmdir "$stage/src"
git rev-parse HEAD > "$stage/COMMIT"
# The manifest is not committed (DATA.md has its SHA-256); notebooks check it before use.
if [ -f artifacts/splits.json ]; then
  mkdir -p "$stage/artifacts" && cp artifacts/splits.json "$stage/artifacts/"
fi
cat > "$stage/dataset-metadata.json" <<JSON
{"title": "acousticwise-src", "id": "yanyetang/acousticwise-src", "licenses": [{"name": "other"}]}
JSON
if kaggle datasets status yanyetang/acousticwise-src >/dev/null 2>&1; then
  kaggle datasets version -p "$stage" --dir-mode zip -m "${1:-update package}"
else
  kaggle datasets create -p "$stage" --dir-mode zip
fi
rm -rf "$stage"
