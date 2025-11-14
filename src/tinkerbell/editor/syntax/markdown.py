"""Markdown highlighting and preview helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class MarkdownPreview:
    """Container holding rendered HTML preview content."""

    html: str
    metadata: Dict[str, Any]


def render_preview(text: str) -> MarkdownPreview:
    """Render Markdown text into HTML.

    Placeholder implementation returns escaped text wrapped inside `<pre>`.
    """

    escaped = text.replace("<", "&lt;").replace(">", "&gt;")
    return MarkdownPreview(html=f"<pre>{escaped}</pre>", metadata={"length": len(text)})
