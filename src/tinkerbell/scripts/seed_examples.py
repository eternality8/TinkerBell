"""Seed example documents and chat transcripts for demos/tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

_SAMPLE_DOCS: dict[str, str] = {
    "welcome.md": """\
# Welcome to TinkerBell

This markdown file highlights the core panels in the TinkerBell editor:

- **Editor**: write prompts, prototypes, or documentation.
- **Chat**: collaborate with the AI agent.
- **Snapshots**: keep the agent aware of your latest edits.

Use this document to validate rendering, syntax highlighting, and autosave.
""",
    "architecture.yaml": """\
editor:
  language: markdown
  autosave: 60s
ai:
  model: gpt-4o-mini
  retries: 3
  tools:
    - document_search
    - structured_edit
""",
    "todos.json": """\
{
  "inbox": [
    "Wire status bar indicators",
    "Polish chat composer",
    "Record recent files"
  ],
  "blocked": [
    "Design review"
  ]
}
""",
    "ideas.txt": """\
Brainstorm:
 - Share AI responses directly into the editor.
 - Offer theme-aware syntax preview.
 - Record tool traces in the status bar.
""",
}

_SAMPLE_TRANSCRIPTS = [
    {
        "filename": "welcome-chat.jsonl",
        "messages": [
            {
                "role": "user",
                "content": "Summarize the welcome document and highlight the agent workflow.",
            },
            {
                "role": "assistant",
                "content": (
                    "The editor hosts markdown content while the chat panel coordinates "
                    "agent prompts and tool traces."
                ),
                "metadata": {"model": "gpt-4o-mini"},
            },
        ],
    }
]


def main(argv: Sequence[str] | None = None) -> list[Path]:
    """Entry point used by the CLI and tests."""

    parser = argparse.ArgumentParser(description="Seed example documents for demos.")
    parser.add_argument("--dest", type=Path, default=Path("assets/sample_docs"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    parser.add_argument(
        "--include-chats",
        action="store_true",
        help="Generate JSONL chat transcripts alongside documents.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (useful for automated tests).",
    )
    args = parser.parse_args(argv)

    dest = args.dest.expanduser()
    created = seed_documents(dest, force=args.force)
    if args.include_chats:
        created.extend(seed_transcripts(dest / "transcripts", force=args.force))

    if not args.quiet:
        _print_summary(created, dest)
    return created


def seed_documents(destination: Path, *, force: bool = False) -> list[Path]:
    """Write the stock sample documents to ``destination``."""

    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for name, body in _SAMPLE_DOCS.items():
        target = destination / name
        if target.exists() and not force:
            continue
        target.write_text(body.strip() + "\n", encoding="utf-8")
        created.append(target)
    return created


def seed_transcripts(destination: Path, *, force: bool = False) -> list[Path]:
    """Write demo chat transcripts in JSONL format."""

    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for transcript in _SAMPLE_TRANSCRIPTS:
        target = destination / transcript["filename"]
        if target.exists() and not force:
            continue
        payload = _render_messages(transcript["messages"])
        target.write_text(payload, encoding="utf-8")
        created.append(target)
    return created


def _render_messages(messages: Iterable[dict]) -> str:
    lines = [json.dumps(message, ensure_ascii=False, sort_keys=True) for message in messages]
    return "\n".join(lines) + "\n"


def _print_summary(created: Iterable[Path], dest: Path) -> None:
    created = list(created)
    if not created:
        print(f"No files were created — existing samples in {dest} were left untouched.")
        return
    print(f"Created {len(created)} sample files under {dest}:")
    for path in created:
        print(f"  • {path.relative_to(dest)}")


if __name__ == "__main__":  # pragma: no cover
    main()
