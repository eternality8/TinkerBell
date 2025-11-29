"""Sample document discovery and data class."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .common import (
    DEFAULT_SAMPLE_LIMIT,
    humanize_bytes,
    language_from_suffix,
    project_root,
)

__all__ = [
    "SampleDocument",
    "discover_sample_documents",
]


@dataclass(slots=True)
class SampleDocument:
    """Metadata describing a built-in sample document shipped with the app."""

    name: str
    path: Path
    language: str = "markdown"
    description: str | None = None


def discover_sample_documents(limit: int | None = DEFAULT_SAMPLE_LIMIT) -> tuple[SampleDocument, ...]:
    """Return available large-file samples bundled with the repository."""

    root = project_root()
    search_dirs = [root / "test_data", root / "assets" / "sample_docs"]
    entries: list[SampleDocument] = []
    seen: set[Path] = set()

    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            files = sorted(directory.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        for candidate in files:
            if not candidate.is_file():
                continue
            language = language_from_suffix(candidate.suffix)
            if language is None:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            size_hint = humanize_bytes(candidate.stat().st_size)
            label = f"{candidate.name} ({size_hint})"
            description = f"Sample {language} document"
            entries.append(SampleDocument(name=label, path=resolved, language=language, description=description))
            seen.add(resolved)
            if limit is not None and len(entries) >= limit:
                return tuple(entries)
    return tuple(entries)
