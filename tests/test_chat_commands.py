"""Unit tests for chat command parsing and validation."""

from __future__ import annotations

import pytest

from tinkerbell.chat.commands import ActionType, parse_agent_payload, validate_directive


def test_parse_accepts_json_string() -> None:
    payload = '{"action": "insert", "content": "hi", "target_range": [0, 0]}'

    result = parse_agent_payload(payload)

    assert result["action"] == ActionType.INSERT.value
    assert result["target_range"] == [0, 0]


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
    assert "content must not be empty" in result.message


def test_validate_rejects_bad_target_range_type() -> None:
    result = validate_directive({"action": "insert", "content": "hi", "target_range": "oops"})

    assert not result.ok
    assert "target_range" in result.message
