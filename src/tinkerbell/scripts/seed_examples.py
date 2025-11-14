"""Seed example documents and chat transcripts."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed example documents for demos.")
    parser.add_argument("--dest", type=Path, default=Path("assets/sample_docs"))
    args = parser.parse_args([])  # placeholder consuming no CLI args yet
    args.dest.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":  # pragma: no cover
    main()
