"""CLI helper to inspect token counts for sample text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ..ai.client import ApproxByteCounter, TiktokenCounter


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect token counts for the given text input.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model identifier to use for token counting.")
    parser.add_argument(
        "--file",
        type=Path,
        help="Optional file containing the text to tokenize. Reads stdin when omitted and --text not provided.",
    )
    parser.add_argument("--text", help="Inline text to tokenize. Overrides --file when provided.")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Skip tiktoken lookups and use the byte-length estimator.",
    )
    args = parser.parse_args(argv)

    payload = _load_text(args.text, args.file)
    if not payload:
        print("No input text provided.", file=sys.stderr)
        return 1

    counter = _build_counter(args.model, estimate_only=args.estimate_only)
    precise = counter.count(payload)
    estimate = counter.estimate(payload)

    print(f"model: {args.model}")
    print(f"characters: {len(payload)}")
    print(f"tokens (precise): {precise}")
    print(f"tokens (estimate): {estimate}")
    return 0


def _load_text(inline: str | None, path: Path | None) -> str:
    if inline:
        return inline
    if path:
        return path.read_text(encoding="utf-8")
    data = sys.stdin.read()
    return data.strip()


def _build_counter(model: str, *, estimate_only: bool) -> ApproxByteCounter | TiktokenCounter:
    if estimate_only:
        return ApproxByteCounter(model_name=model)
    try:
        return TiktokenCounter(model)
    except Exception:
        return ApproxByteCounter(model_name=model)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
