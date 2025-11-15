"""Unit tests for AI tool modules."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from tinkerbell.ai.tools.document_edit import DocumentEditTool
from tinkerbell.ai.tools.document_snapshot import DocumentSnapshotTool
from tinkerbell.ai.tools.search_replace import SearchReplaceTool
from tinkerbell.ai.tools.validation import validate_snippet
from tinkerbell.editor.syntax import yaml_json


class _EditBridgeStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_diff_summary: str | None = None
        self.last_snapshot_version: str | None = None

    def queue_edit(self, directive) -> None:  # type: ignore[override]
        self.calls.append(directive)


class _SnapshotProviderStub:
    def __init__(self) -> None:
        self.snapshot = {"text": "Hello", "selection": (0, 5), "version": "base"}
        self.delta_only_calls: list[bool] = []
        self.last_diff_summary: str | None = "+1 char"
        self.last_snapshot_version: str | None = "digest-1"

    def generate_snapshot(self, *, delta_only: bool = False):  # type: ignore[override]
        self.delta_only_calls.append(delta_only)
        return dict(self.snapshot)


class _SearchReplaceBridgeStub:
    def __init__(self, *, text: str, selection: tuple[int, int] = (0, 0), version: str = "base") -> None:
        self.snapshot = {"text": text, "selection": selection, "version": version}
        self.queue_calls: list[Mapping[str, Any]] = []
        self.last_snapshot_version: str | None = version

    def generate_snapshot(self, *, delta_only: bool = False):  # type: ignore[override]
        assert delta_only is False
        return dict(self.snapshot)

    def queue_edit(self, directive: Mapping[str, Any]):  # type: ignore[override]
        self.queue_calls.append(directive)
        self.last_snapshot_version = "updated"


def test_document_edit_tool_accepts_json_payload_and_reports_status():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "+2 chars"
    bridge.last_snapshot_version = "digest-123"
    tool = DocumentEditTool(bridge=bridge)

    status = tool.run('{"action":"insert","content":"Hi","target_range":[0,0]}')

    assert bridge.calls and isinstance(bridge.calls[0], dict)
    assert bridge.calls[0]["action"] == "insert"
    assert "applied: +2 chars" in status
    assert "digest-123" in status


def test_document_edit_tool_accepts_keyword_arguments():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "Δ0"
    tool = DocumentEditTool(bridge=bridge)

    status = tool.run(action="replace", content="Hello", target_range=(0, 5))

    assert bridge.calls and bridge.calls[0]["action"] == "replace"
    assert status.startswith("applied") or status == "queued"


def test_document_edit_tool_requires_payload():
    bridge = _EditBridgeStub()
    tool = DocumentEditTool(bridge=bridge)

    with pytest.raises(ValueError):
        tool.run()

    with pytest.raises(ValueError):
        tool.run('{"action":"insert","content":"Hi"}', action="insert")


def test_document_edit_tool_replaces_paragraph_with_formatting():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "+12 chars"
    bridge.last_snapshot_version = "digest-abc"
    tool = DocumentEditTool(bridge=bridge)

    replacement = "New intro paragraph.\n\n- bullet α\n- bullet β ✨"
    status = tool.run(
        action="replace",
        content=replacement,
        target_range={"start": 10, "end": 64},
        rationale="Refresh intro with bullets",
    )

    assert bridge.calls and bridge.calls[0]["content"] == replacement
    assert bridge.calls[0]["target_range"] == {"start": 10, "end": 64}
    assert "applied" in status and "digest-abc" in status


def test_document_edit_tool_handles_special_characters_in_sentence():
    bridge = _EditBridgeStub()
    tool = DocumentEditTool(bridge=bridge)

    payload = (
        '{"action":"replace","target_range":[25,52],'
        '"content":"“smart quotes” & emojis 🎉\\nNext line"}'
    )

    tool.run(payload)

    assert bridge.calls
    recorded = bridge.calls[-1]
    assert recorded["content"].startswith("“smart quotes” & emojis 🎉")
    assert recorded["content"].endswith("Next line")
    assert recorded["target_range"] == [25, 52]


def test_document_snapshot_tool_includes_diff_summary_and_version():
    provider = _SnapshotProviderStub()
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run(delta_only=True)

    assert provider.delta_only_calls == [True]
    assert snapshot["version"] == "base"
    assert snapshot["diff_summary"] == "+1 char"
    assert "diff_summary" not in provider.snapshot


def test_document_snapshot_tool_falls_back_to_provider_version():
    provider = _SnapshotProviderStub()
    provider.snapshot.pop("version", None)
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run()

    assert snapshot["version"] == "digest-1"


def test_search_replace_tool_updates_document_scope():
    bridge = _SearchReplaceBridgeStub(text="Hello world world", selection=(0, 0), version="digest-0")
    tool = SearchReplaceTool(bridge=bridge, preview_chars=100)

    result = tool.run(pattern="world", replacement="earth")

    assert result.replacements == 2
    assert result.applied is True
    assert result.scope == "document"
    assert bridge.queue_calls
    payload = bridge.queue_calls[-1]
    assert payload["target_range"] == (0, len("Hello world world"))
    assert "earth" in payload["content"]
    assert result.document_version == "updated"
    assert "earth" in result.preview


def test_search_replace_tool_respects_selection_scope_and_dry_run():
    bridge = _SearchReplaceBridgeStub(text="Alpha Beta Gamma", selection=(6, 10), version="digest-1")
    tool = SearchReplaceTool(bridge=bridge, preview_chars=32)

    result = tool.run(
        pattern="beta",
        replacement="delta",
        scope="selection",
        dry_run=True,
        match_case=False,
        whole_word=True,
        max_replacements=1,
    )

    assert result.replacements == 1
    assert result.applied is False
    assert result.dry_run is True
    assert result.scope == "selection"
    assert result.target_range == (6, 10)
    assert bridge.queue_calls == []
    assert result.document_version == "digest-1"
    assert result.preview


def test_search_replace_tool_validates_inputs():
    bridge = _SearchReplaceBridgeStub(text="Hello", selection=(0, 0))
    tool = SearchReplaceTool(bridge=bridge)

    with pytest.raises(ValueError):
        tool.run(pattern="", replacement="x")

    with pytest.raises(ValueError):
        tool.run(pattern="x", replacement="y", max_replacements=0)


def test_validate_snippet_handles_json_and_unknown_formats():
    success = validate_snippet('{"foo": 1}', "json")
    assert success.ok is True

    failure = validate_snippet("{foo}", "json")
    assert failure.ok is False

    unsupported = validate_snippet("value", "toml")
    assert unsupported.ok is False
    assert "Unsupported format" in unsupported.message


def test_validate_snippet_reports_yaml_errors(monkeypatch: pytest.MonkeyPatch):
    from tinkerbell.ai.tools import validation as validation_module

    errors = [
        yaml_json.ValidationError(message="first", line=2),
        yaml_json.ValidationError(message="second", line=4),
    ]

    monkeypatch.setitem(validation_module._FORMAT_VALIDATORS, "yaml", lambda _text: errors)

    outcome = validation_module.validate_snippet("bad", "yaml")

    assert outcome.ok is False
    assert "Line 2: first (+1 more issue)." == outcome.message


def test_validate_snippet_success_path_can_be_stubbed(monkeypatch: pytest.MonkeyPatch):
    from tinkerbell.ai.tools import validation as validation_module

    monkeypatch.setitem(validation_module._FORMAT_VALIDATORS, "yaml", lambda _text: [])

    outcome = validation_module.validate_snippet("foo: bar", "yaml")

    assert outcome.ok is True
