"""Unit tests for chat command parsing and validation."""

from __future__ import annotations

import pytest

from tinkerbell.ui.presentation.chat.commands import (
    ActionType,
    ManualCommandType,
    parse_agent_payload,
    parse_manual_command,
    resolve_tab_reference,
    validate_directive,
)


_TABS = [
    {"tab_id": "tab-a", "title": "README.md", "path": "C:/repo/README.md"},
    {"tab_id": "tab-b", "title": "Notes.md", "path": "C:/repo/docs/notes.md"},
]


def test_parse_accepts_json_string() -> None:
    payload = '{"action": "insert", "content": "hi", "target_range": [0, 0]}'

    result = parse_agent_payload(payload)

    assert result["action"] == ActionType.INSERT.value
    assert result["target_range"] == {"start": 0, "end": 0}


def test_parse_strips_code_fence_and_aliases_target() -> None:
    payload = """```json
    {"action": "replace", "content": "updated", "target": {"start": 1, "end": 4}}
    ```"""

    result = parse_agent_payload(payload)

    assert "target" not in result
    assert result["target_range"] == {"start": 1, "end": 4}


def test_parse_mapping_normalizes_action_case() -> None:
    payload = {"action": "INSERT", "content": "hi"}

    result = parse_agent_payload(payload)

    assert result["action"] == ActionType.INSERT.value


def test_parse_does_not_alias_selection_field() -> None:
    payload = {"action": "insert", "content": "hi", "selection": {"start": 0, "end": 4}}

    result = parse_agent_payload(payload)

    assert "target_range" not in result
    assert result["selection"] == {"start": 0, "end": 4}


def test_parse_invalid_payload_raises() -> None:
    with pytest.raises(ValueError):
        parse_agent_payload("no structured json here")


def test_validate_accepts_valid_payload() -> None:
    payload = {"action": "insert", "content": "hello", "target_range": [0, 0]}

    result = validate_directive(payload)

    assert result.ok


def test_validate_rejects_missing_fields() -> None:
    result = validate_directive({})

    assert not result.ok
    assert "required property" in result.message


def test_validate_rejects_unknown_action() -> None:
    result = validate_directive({"action": "delete", "content": "hi"})

    assert not result.ok
    assert "action" in result.message.lower()


def test_validate_rejects_whitespace_content() -> None:
    result = validate_directive({"action": "insert", "content": "   "})

    assert not result.ok
    assert "include non-whitespace" in result.message


def test_validate_accepts_empty_string_deletion() -> None:
    payload = {"action": "replace", "content": "", "target_range": [0, 4]}

    result = validate_directive(payload)

    assert result.ok


def test_validate_rejects_bad_target_range_type() -> None:
    result = validate_directive({"action": "insert", "content": "hi", "target_range": "oops"})

    assert not result.ok
    assert "target_range" in result.message


def test_validate_rejects_selection_field() -> None:
    payload = {"action": "insert", "content": "hi", "selection": {"start": 0, "end": 4}}

    result = validate_directive(payload)

    assert not result.ok
    assert "Deprecated range parameters" in result.message


def test_validate_accepts_patch_with_diff_and_version() -> None:
    payload = {
        "action": "patch",
        "diff": "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new",
        "document_version": "digest",
    }

    result = validate_directive(payload)

    assert result.ok is True


def test_validate_rejects_patch_without_version() -> None:
    result = validate_directive({"action": "patch", "diff": "@@ -1 +1 @@"})

    assert result.ok is False
    assert "document_version" in result.message.lower()


def test_resolve_tab_reference_matches_number() -> None:
    assert resolve_tab_reference("Tab 2", _TABS) == "tab-b"
    assert resolve_tab_reference("#1", _TABS) == "tab-a"


def test_resolve_tab_reference_matches_titles_and_paths() -> None:
    assert resolve_tab_reference("readme.md", _TABS) == "tab-a"
    assert resolve_tab_reference("notes", _TABS) == "tab-b"
    assert resolve_tab_reference("unknown", _TABS) is None


def test_parse_manual_outline_command_with_flags() -> None:
    request = parse_manual_command("/outline --doc Tab 2 --levels 3 --max 25 --no-blurbs")

    assert request is not None
    assert request.command is ManualCommandType.OUTLINE
    assert request.args["document_id"] == "Tab 2"
    assert request.args["desired_levels"] == 3
    assert request.args["max_nodes"] == 25
    assert request.args["include_blurbs"] is False


def test_parse_manual_find_command_with_query_and_flags() -> None:
    request = parse_manual_command('::find --doc "tab-a" --top 4 --confidence 0.5 "intro section"')

    assert request is not None
    assert request.command is ManualCommandType.FIND_SECTIONS
    assert request.args["document_id"] == "tab-a"
    assert request.args["top_k"] == 4
    assert pytest.approx(request.args["min_confidence"], rel=1e-6) == 0.5
    assert request.args["query"] == "intro section"


def test_parse_manual_find_requires_query() -> None:
    with pytest.raises(ValueError):
        parse_manual_command("!find --doc tab-a")


def test_parse_manual_status_command_with_flags() -> None:
    request = parse_manual_command("/status --doc Tab 2 --json")

    assert request is not None
    assert request.command is ManualCommandType.STATUS
    assert request.args["document_id"] == "Tab 2"
    assert request.args["as_json"] is True


def test_parse_manual_status_command_accepts_positional_reference() -> None:
    request = parse_manual_command("!status README.md")

    assert request is not None
    assert request.command is ManualCommandType.STATUS
    assert request.args["document_id"] == "README.md"


def test_parse_manual_command_ignored_without_prefix() -> None:
    assert parse_manual_command("outline please") is None


def test_parse_manual_command_rejects_unknown_command() -> None:
    with pytest.raises(ValueError):
        parse_manual_command("/summarize this")
