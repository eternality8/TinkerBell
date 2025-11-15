"""Unit tests for unified diff patch application."""

from __future__ import annotations

import pytest

from tinkerbell.ai.tools.diff_builder import DiffBuilderTool
from tinkerbell.editor.patches import PatchApplyError, apply_unified_diff


def test_apply_unified_diff_handles_multiple_hunks():
    original = "alpha\nbeta\ngamma\n"
    diff = """--- a/doc.txt
+++ b/doc.txt
@@ -1,2 +1,2 @@
-alpha
+Alpha
 beta
@@ -3 +3,2 @@
-gamma
+delta
+gamma
"""
    result = apply_unified_diff(original, diff)

    assert "Alpha" in result.text
    assert "delta" in result.text
    assert result.summary.startswith("patch:")
    assert result.spans, "expected spans to be tracked"


def test_apply_unified_diff_respects_no_newline_marker():
    original = "one\ntwo"
    diff = """--- a/x
+++ b/x
@@ -1,2 +1,2 @@
 one
-two
+two-three
\\ No newline at end of file
"""
    result = apply_unified_diff(original, diff)

    assert result.text.endswith("three"), "newline marker should be honored"


def test_apply_unified_diff_raises_on_context_mismatch():
    original = "hello\nworld\n"
    diff = """--- a/x
+++ b/x
@@ -1,2 +1,2 @@
-HELLO
+hi
 world
"""
    with pytest.raises(PatchApplyError):
        apply_unified_diff(original, diff)


def test_apply_unified_diff_reanchors_when_line_numbers_are_snippet_based():
    original = "intro\nalpha\nbeta\ngamma\n"
    diff = """--- a/snippet
+++ b/snippet
@@ -1,2 +1,2 @@
 alpha
-beta
+BETA
"""

    result = apply_unified_diff(original, diff)

    assert "BETA" in result.text
    assert result.text.count("intro") == 1


def test_apply_unified_diff_inserts_into_empty_document():
    original = ""
    diff = """--- a/doc.txt
+++ b/doc.txt
@@ -0,0 +1,2 @@
+# Title

+Line two
"""

    result = apply_unified_diff(original, diff)

    assert result.text.startswith("# Title")
    assert "Line two" in result.text


def test_apply_unified_diff_handles_blank_context_lines():
    original = "# Title\n\nLine one\n\nLine two\n"
    updated = "# Title\n\nLine one with Barnaby.\n\nLine two\n"
    diff = DiffBuilderTool().run(original, updated, filename="story.md", context=3)

    result = apply_unified_diff(original, diff)

    assert "Barnaby" in result.text
    assert result.summary.startswith("patch:")
