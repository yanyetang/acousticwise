"""Compare models across folds in one markdown table (technical design, section 8).

Usage:
    uv run python -m acousticwise.compare NAME=SUMMARY [NAME=SUMMARY ...]

Each SUMMARY is a summary.json written by acousticwise.evaluate over one model's fold runs.
"""

import argparse
import json
from pathlib import Path

# Column title -> the across-fold score it shows, so a title cannot drift from its numbers.
COLUMNS = {
    "Macro-F1, 9 classes": "macro_f1_headline",
    "Macro-F1, 12 classes": "macro_f1_all",
    "Accuracy": "accuracy",
}


def comparison_table(summaries: dict[str, dict]) -> str:
    """One row per model: mean ± spread of the 9-class and 12-class macro-F1 and accuracy,
    then the 9-class macro-F1 of each fold, in fold order."""
    lines = [
        "| Model | " + " | ".join(COLUMNS) + " | 9-class per fold |",
        "|---" * (len(COLUMNS) + 2) + "|",
    ]
    for model, summary in summaries.items():
        across = summary["across_folds"]
        cells = [
            f"{across[score]['mean']:.3f} ± {across[score]['std']:.3f}"
            for score in COLUMNS.values()
        ]
        runs = sorted(summary["runs"], key=lambda run: run["fold"])
        cells.append(", ".join(f"{run['macro_f1_headline']:.3f}" for run in runs))
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def model_summary(arg: str) -> tuple[str, dict]:
    """Parse NAME=SUMMARY into the model name and its loaded summary."""
    name, sep, path = arg.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=SUMMARY, got {arg!r}")
    try:
        return name, json.loads(Path(path).read_text())
    except OSError as error:
        raise argparse.ArgumentTypeError(f"cannot read {path}: {error.strerror}") from error


def main() -> None:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "models", nargs="+", type=model_summary, help="NAME=SUMMARY, e.g. CNN=summary_cnn.json"
    )
    args = parser.parse_args()
    print(comparison_table(dict(args.models)), end="")


if __name__ == "__main__":
    main()
