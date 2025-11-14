"""Tests for the Markdown syntax helper utilities."""

from tinkerbell.editor.syntax.markdown import detect_frontmatter, render_preview
from tinkerbell.editor.syntax.themes import load_theme


def test_detect_frontmatter_parses_yaml_metadata():
    text = """---
    title: Sample Doc
    tags:
      - demo
    draft: true
    ---
    # Heading
    body text
    """
    metadata = detect_frontmatter(text)
    assert metadata["title"] == "Sample Doc"
    assert metadata["tags"] == ["demo"]
    assert metadata["draft"] is True


def test_render_preview_yields_html_and_headings():
    preview = render_preview("# Heading\n\nSome text here.", theme=load_theme())
    assert "tb-markdown-preview" in preview.html
    assert "<h1" in preview.html
    assert preview.metadata["headings"][0]["anchor"] == "heading"
    assert preview.metadata["stats"]["word_count"] == 4


def test_render_preview_marks_truncated_output():
    long_text = "# Title\n\n" + ("word " * 10_000)
    preview = render_preview(long_text, max_chars=500)
    assert preview.metadata["truncated"] is True
    assert "Preview truncated" in preview.html
