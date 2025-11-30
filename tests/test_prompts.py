"""Tests for structured prompt helpers."""

from __future__ import annotations

from tinkerbell.ai import prompts


def test_base_system_prompt_includes_core_elements():
    """Test that system prompt includes key guidance sections."""
    content = prompts.base_system_prompt(model_name=None)
    # Check personality/voice section
    assert "TinkerBell" in content
    assert "fairy" in content.lower()
    # Check tool sections
    assert "Available Tools" in content
    assert "read_document" in content
    assert "replace_lines" in content
    # Check workflow section
    assert "Core Workflow" in content
    assert "version_token" in content
    # Check guidelines
    assert "Guidelines" in content


def test_format_user_prompt_includes_window_metadata():
    snapshot = {
        "path": "docs/example.md",
        "text": "# Title\nSample paragraph for testing.",
        "text_range": {"start": 2, "end": 12},
        "language": "Markdown",
        "document_version": "abc123",
    }
    content = prompts.format_user_prompt("Summarize the intro", snapshot)
    assert "docs/example.md" in content
    assert "Markdown" in content
    assert "Snapshot window: 2-12" in content
    assert "abc123" in content


def test_system_prompt_v2_alias():
    """Test that base_system_prompt is an alias for system_prompt_v2."""
    base_content = prompts.base_system_prompt(model_name=None)
    v2_content = prompts.system_prompt_v2(model_name=None)
    assert base_content == v2_content


def test_format_user_prompt_no_document_shows_workspace_status():
    """Test that no_document snapshot shows workspace status message."""
    snapshot = {
        "tab_id": None,
        "text": "",
        "version": None,
        "length": 0,
        "document_id": "",
        "no_document": True,
    }
    content = prompts.format_user_prompt("Help me write a story", snapshot)
    assert "Workspace status:" in content
    assert "No documents are currently open" in content
    assert "create_document" in content


def test_format_user_prompt_with_document_no_workspace_status():
    """Test that normal snapshot does not include workspace status message."""
    snapshot = {
        "path": "story.md",
        "text": "Once upon a time...",
        "language": "Markdown",
    }
    content = prompts.format_user_prompt("Continue the story", snapshot)
    assert "Workspace status:" not in content
