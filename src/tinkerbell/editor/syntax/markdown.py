"""Markdown highlighting and preview helpers."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .themes import DEFAULT_THEME, Theme

try:  # pragma: no cover - dependency provided via pyproject extras
    from markdown_it import MarkdownIt
except Exception:  # pragma: no cover - graceful fallback when optional dep missing
    MarkdownIt = None  # type: ignore[assignment]

try:  # pragma: no cover - dependency provided via pyproject extras
    from ruamel.yaml import YAML
except Exception:  # pragma: no cover - graceful fallback when optional dep missing
    YAML = None  # type: ignore[assignment]

MAX_PREVIEW_CHARS = 20_000
_TRUNCATION_NOTICE = "\n\n> _Preview truncated for performance._\n"
_HEADING_PATTERN = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class MarkdownPreview:
    """Container holding rendered HTML preview content and metadata."""

    html: str
    metadata: Dict[str, Any]


def detect_frontmatter(text: str) -> Dict[str, Any]:
    """Return parsed YAML frontmatter from ``text`` if present."""

    block, _ = _split_frontmatter(text or "")
    return _parse_frontmatter_block(block)


def render_preview(
    text: str,
    *,
    theme: Theme | None = None,
    max_chars: Optional[int] = MAX_PREVIEW_CHARS,
) -> MarkdownPreview:
    """Render Markdown ``text`` into HTML using ``markdown-it-py`` when available."""

    resolved_theme = theme or DEFAULT_THEME
    raw_text = text or ""
    frontmatter_block, body = _split_frontmatter(raw_text)
    frontmatter_data = _parse_frontmatter_block(frontmatter_block)
    render_body, truncated = _maybe_truncate(body, max_chars)
    html_body = _render_markdown(render_body)
    wrapped_html = _wrap_with_theme(html_body, resolved_theme)
    metadata = {
        "length": len(raw_text),
        "frontmatter": frontmatter_data,
        "headings": _extract_headings(body),
        "stats": _calculate_stats(body),
        "truncated": truncated,
    }
    return MarkdownPreview(html=wrapped_html, metadata=metadata)


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------
def _split_frontmatter(text: str) -> tuple[Optional[str], str]:
    """Return (frontmatter_block, body) from ``text`` if fenced frontmatter exists."""

    if not text:
        return None, ""

    working = text.lstrip("\ufeff")
    if not working.startswith(("---", "+++")):
        return None, working

    lines = working.splitlines()
    if not lines:
        return None, working

    fence = lines[0].strip()
    if fence not in {"---", "+++"}:
        return None, working

    closing_index = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == fence:
            closing_index = idx
            break
    if closing_index is None:
        return None, working

    frontmatter_lines = lines[1:closing_index]
    remainder_lines = lines[closing_index + 1 :]
    frontmatter_block = "\n".join(frontmatter_lines)
    remainder = "\n".join(remainder_lines).lstrip("\r\n")
    return frontmatter_block, remainder


def _parse_frontmatter_block(block: Optional[str]) -> Dict[str, Any]:
    if not block:
        return {}
    if YAML is None:  # pragma: no cover - dependency always installed in CI
        return {}
    parser = YAML(typ="safe")
    parser.preserve_quotes = False
    parser.allow_duplicate_keys = False
    try:
        loaded = parser.load(block) or {}
    except Exception:
        return {}
    if isinstance(loaded, dict):
        return dict(loaded)
    return {}


# ---------------------------------------------------------------------------
# Rendering + metadata extraction helpers
# ---------------------------------------------------------------------------
def _maybe_truncate(text: str, max_chars: Optional[int]) -> Tuple[str, bool]:
    if max_chars is None or len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline != -1 and last_newline > max_chars * 0.5:
        truncated = truncated[:last_newline]
    truncated = truncated.rstrip() + _TRUNCATION_NOTICE
    return truncated, True


_MARKDOWN_RENDERER: Optional[MarkdownIt] = None  # type: ignore[valid-type]


def _build_renderer() -> Optional[MarkdownIt]:  # type: ignore[valid-type]
    global _MARKDOWN_RENDERER
    if _MARKDOWN_RENDERER is None and MarkdownIt is not None:
        renderer = MarkdownIt(
            "commonmark",
            {"html": False, "linkify": True, "typographer": True},
        )
        renderer.enable("table")
        renderer.enable("strikethrough")
        _MARKDOWN_RENDERER = renderer
    return _MARKDOWN_RENDERER


def _render_markdown(text: str) -> str:
    renderer = _build_renderer()
    if renderer is None:
        return f"<pre>{html.escape(text)}</pre>"
    return renderer.render(text)


def _extract_headings(text: str) -> list[Dict[str, Any]]:
    headings: list[Dict[str, Any]] = []
    for match in _HEADING_PATTERN.finditer(text):
        title = match.group("title").strip()
        if not title:
            continue
        level = len(match.group("level"))
        headings.append({"level": level, "text": title, "anchor": _slugify(title)})
    return headings


def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("-", value.lower()).strip("-")
    return slug or "section"


def _calculate_stats(text: str) -> Dict[str, Any]:
    words = _WORD_PATTERN.findall(text)
    word_count = len(words)
    char_count = len(text)
    line_count = 0 if not text else text.count("\n") + 1
    reading_time_minutes = round(word_count / 200, 2) if word_count else 0.0
    return {
        "word_count": word_count,
        "char_count": char_count,
        "line_count": line_count,
        "reading_time_minutes": reading_time_minutes,
    }


def _wrap_with_theme(html_body: str, theme: Theme) -> str:
    fg = _palette_to_css(theme.color("preview_foreground", theme.color("foreground", (235, 235, 235))))
    bg = _palette_to_css(theme.color("preview_background", theme.color("background", (30, 30, 30))))
    accent = _palette_to_css(theme.color("accent", (97, 175, 239)))
    code_bg = _palette_to_css(_adjust_color(theme.color("preview_code_background", theme.color("background", (30, 30, 30))), 1.05))
    style = f"""
<style>
.tb-markdown-preview {{
  background-color: {bg};
  color: {fg};
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  line-height: 1.6;
  padding: 1.5rem;
}}
.tb-markdown-preview h1,
.tb-markdown-preview h2,
.tb-markdown-preview h3 {{
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  padding-bottom: 0.3rem;
  margin-top: 2rem;
}}
.tb-markdown-preview pre {{
  background-color: {code_bg};
  padding: 1rem;
  border-radius: 6px;
  overflow-x: auto;
}}
.tb-markdown-preview code {{
  background-color: {code_bg};
  padding: 0.2rem 0.4rem;
  border-radius: 4px;
  font-family: 'Fira Code', 'Consolas', 'Courier New', monospace;
}}
.tb-markdown-preview a {{
  color: {accent};
  text-decoration: none;
}}
.tb-markdown-preview a:hover {{
  text-decoration: underline;
}}
</style>
"""
    return f"{style}<div class=\"tb-markdown-preview\">{html_body}</div>"


def _palette_to_css(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [max(0, min(255, channel)) for channel in rgb]
    return f"rgb({r}, {g}, {b})"


def _adjust_color(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (
        min(255, int(r * factor)),
        min(255, int(g * factor)),
        min(255, int(b * factor)),
    )
