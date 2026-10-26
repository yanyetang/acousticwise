"""Compare models across folds in one markdown table (technical design, section 8).

Usage:
    uv run python -m acousticwise.compare NAME=SUMMARY [NAME=SUMMARY ...]

Each SUMMARY is a summary.json written by acousticwise.evaluate over one model's fold runs.
"""

import argparse
import json
from pathlib import Path


def comparison_table(summaries: dict[str, dict]) -> str:
    """One row per model: mean ± spread of the 9-class and 12-class macro-F1 and accuracy,
    then the 9-class macro-F1 of each fold, in fold order."""
    lines = [
        "| Model | Macro-F1, 9 classes | Macro-F1, 12 classes | Accuracy | 9-class per fold |",
        "|---|---|---|---|---|",
    ]
    for model, summary in summaries.items():
        across = summary["across_folds"]
        cells = [
            f"{across[score]['mean']:.3f} ± {across[score]['std']:.3f}"
            for score in ("macro_f1_headline", "macro_f1_all", "accuracy")
        ]
        runs = sorted(summary["runs"], key=lambda run: run["fold"])
        cells.append(", ".join(f"{run['macro_f1_headline']:.3f}" for run in runs))
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="NAME=SUMMARY, e.g. CNN=summary_cnn.json")
    args = parser.parse_args()

    summaries = {}
    for model in args.models:
        name, path = model.split("=", 1)
        summaries[name] = json.loads(Path(path).read_text())
    print(comparison_table(summaries), end="")


if __name__ == "__main__":
    main()
