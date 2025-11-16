"""Tests for structured prompt helpers."""

from __future__ import annotations

from tinkerbell.ai import prompts


def test_base_system_prompt_includes_budget_and_fallback():
    content = prompts.base_system_prompt(model_name=None)
    assert "Planning contract" in content
    assert "Tool execution contract" in content
    assert "TokenCounterRegistry" in content
    assert "approximate" in content.lower()


def test_format_user_prompt_includes_selection_metadata():
    snapshot = {
        "path": "docs/example.md",
        "text": "# Title\nSample paragraph for testing.",
        "selection": {"start": 2, "end": 12},
        "language": "Markdown",
        "document_version": "abc123",
    }
    content = prompts.format_user_prompt("Summarize the intro", snapshot)
    assert "docs/example.md" in content
    assert "Markdown" in content
    assert "Selection range: 2-12" in content
    assert "abc123" in content
    assert "excerpt" in content.lower()


def test_token_budget_hint_changes_when_model_registered():
    # Register a fake counter to flip the hint branch.
    registry = prompts.TokenCounterRegistry.global_instance()
    registry.register("fake-model", prompts.TokenCounterRegistry.global_instance().get(None))
    content = prompts.base_system_prompt(model_name="fake-model")
    assert "exact" in content.lower()
    # Clean up to avoid leaking into other tests.
    registry.unregister("fake-model")
