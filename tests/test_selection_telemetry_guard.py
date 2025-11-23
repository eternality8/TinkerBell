from __future__ import annotations

from pathlib import Path

LEGACY_IDENTIFIERS: dict[str, str] = {
    "selection_snapshot_requested": "Use span_snapshot_requested instead.",
    '"selection_span"': "Use snapshot_span when serializing telemetry payloads.",
    '"selection_overlap"': "Use span_overlap for chunk metadata.",
    '"selection_length"': "Use span_length in SelectionRangeTool telemetry.",
}

TARGET_DIRS = ("src", "docs")
SCANNED_SUFFIXES = {".py", ".md"}


def test_no_legacy_selection_telemetry_identifiers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for relative_dir in TARGET_DIRS:
        target = repo_root / relative_dir
        if not target.exists():
            continue
        for file_path in target.rglob("*"):
            if not file_path.is_file() or file_path.suffix not in SCANNED_SUFFIXES:
                continue
            try:
                contents = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for identifier, guidance in LEGACY_IDENTIFIERS.items():
                if identifier in contents:
                    violations.append(f"{identifier} found in {file_path.relative_to(repo_root)} -> {guidance}")
    assert not violations, "Legacy selection telemetry identifiers detected:\n" + "\n".join(violations)
