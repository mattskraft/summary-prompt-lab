"""CLI tool to evaluate safety classification examples."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, Tuple

from kiso_input import classify_self_harm, load_self_harm_lexicon
from kiso_input.config import SAFETY_LEXICON_PATH


def _load_examples(path: Path) -> Iterable[Tuple[str, str]]:
    """Parse the examples file containing a Python-style list of tuples."""
    content = path.read_text(encoding="utf-8").strip()
    if "=" in content:
        _, rhs = content.split("=", 1)
    else:
        rhs = content
    data = ast.literal_eval(rhs.strip())
    return [(str(text), str(expected)) for text, expected in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate safety classification examples.")
    default_examples = Path(__file__).resolve().parents[1] / "data" / "testing" / "safety_examples.yaml"
    parser.add_argument(
        "--examples",
        type=Path,
        default=default_examples,
        help=f"Path to examples file (default: {default_examples})",
    )
    args = parser.parse_args()

    examples_path = args.examples
    if not examples_path.exists():
        raise FileNotFoundError(f"Examples file not found: {examples_path}")

    if not SAFETY_LEXICON_PATH:
        raise RuntimeError("SAFETY_LEXICON_PATH is not configured.")

    lexicon = load_self_harm_lexicon(SAFETY_LEXICON_PATH)
    tests = list(_load_examples(examples_path))

    if not tests:
        print("No tests found in the examples file.")
        return

    label_width = 40
    mismatches = 0

    for text, expected in tests:
        result = classify_self_harm(text, lexicon)
        actual = result["risk_level"]
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            mismatches += 1
        print(f"{status} {text[:label_width]:{label_width}} → {actual} (expected: {expected})")

    print(f"\nSummary: {len(tests) - mismatches} / {len(tests)} match expectations.")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


