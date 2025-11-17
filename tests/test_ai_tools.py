"""Unit tests for AI tool modules."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from tinkerbell.ai.tools.diff_builder import DiffBuilderTool
from tinkerbell.ai.tools.document_apply_patch import DocumentApplyPatchTool
from tinkerbell.ai.tools.document_edit import DocumentEditTool
from tinkerbell.ai.tools.document_snapshot import DocumentSnapshotTool
from tinkerbell.ai.tools.list_tabs import ListTabsTool
from tinkerbell.ai.tools.search_replace import SearchReplaceTool
from tinkerbell.ai.tools.validation import validate_snippet
from tinkerbell.editor.syntax import yaml_json
from tinkerbell.services.bridge import DocumentVersionMismatchError


class _EditBridgeStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_diff_summary: str | None = None
        self.last_snapshot_version: str | None = None
        self.snapshot = {"text": "", "selection": (0, 0), "version": "digest-base", "path": "doc.txt"}
        self.queue_tab_ids: list[str | None] = []
        self.snapshot_requests: list[dict[str, Any]] = []
        self.tabs: list[dict[str, Any]] = [
            {"tab_id": "tab-a", "title": "README.md", "path": "C:/repo/README.md", "dirty": False},
            {"tab_id": "tab-b", "title": "Notes.md", "path": "C:/repo/notes.md", "dirty": True},
        ]
        self._active_tab_id = "tab-a"

    def queue_edit(self, directive, *, tab_id: str | None = None) -> None:  # type: ignore[override]
        self.queue_tab_ids.append(tab_id)
        self.calls.append(directive)

    def generate_snapshot(  # type: ignore[override]
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
    ):
        self.snapshot_requests.append(
            {"delta_only": delta_only, "tab_id": tab_id, "include_open_documents": include_open_documents}
        )
        assert delta_only is False
        return dict(self.snapshot)

    def get_last_diff_summary(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_diff_summary

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_snapshot_version

    def list_tabs(self) -> list[dict[str, Any]]:
        return list(self.tabs)

    def active_tab_id(self) -> str | None:
        return self._active_tab_id


class _SnapshotProviderStub:
    def __init__(self) -> None:
        self.snapshot = {"text": "Hello", "selection": (0, 5), "version": "base", "document_id": "doc-stub"}
        self.delta_only_calls: list[bool] = []
        self.last_diff_summary: str | None = "+1 char"
        self.last_snapshot_version: str | None = "digest-1"

    def generate_snapshot(  # type: ignore[override]
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
    ):
        self.delta_only_calls.append(delta_only)
        return dict(self.snapshot)

    def get_last_diff_summary(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_diff_summary

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_snapshot_version


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


class _PatchBridgeStub(_EditBridgeStub):
    def __init__(self, *, text: str = "Hello world", selection: tuple[int, int] = (0, 0), version: str = "digest-0") -> None:
        super().__init__()
        self.snapshot = {"text": text, "selection": selection, "version": version, "path": "doc.md"}
        self.last_snapshot_version = version


class _LegacyBridgeStub(_EditBridgeStub):
    def queue_edit(self, directive) -> None:  # type: ignore[override]
        # Legacy bridges do not accept tab_id keyword arguments
        self.calls.append(directive)


def test_document_edit_tool_accepts_json_payload_and_reports_status():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "+2 chars"
    bridge.last_snapshot_version = "digest-123"
    tool = DocumentEditTool(bridge=bridge, allow_inline_edits=True)

    status = tool.run('{"action":"insert","content":"Hi","target_range":[0,0]}')

    assert bridge.calls and isinstance(bridge.calls[0], dict)
    assert bridge.calls[0]["action"] == "insert"
    assert "applied: +2 chars" in status
    assert "digest-123" in status


def test_document_edit_tool_accepts_keyword_arguments():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "Δ0"
    tool = DocumentEditTool(bridge=bridge, allow_inline_edits=True)

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
    tool = DocumentEditTool(bridge=bridge, allow_inline_edits=True)

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
    tool = DocumentEditTool(bridge=bridge, allow_inline_edits=True)

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


def test_document_edit_tool_routes_to_specific_tab():
    bridge = _EditBridgeStub()
    tool = DocumentEditTool(bridge=bridge, allow_inline_edits=True)

    tool.run(action="insert", content="Hi", target_range=(0, 0), tab_id="tab-123")

    assert bridge.queue_tab_ids[-1] == "tab-123"


def test_document_edit_tool_resolves_tab_reference_in_argument():
    bridge = _EditBridgeStub()
    tool = DocumentEditTool(bridge=bridge, allow_inline_edits=True)

    tool.run(action="insert", content="Hi", target_range=(0, 0), tab_id="Tab 2")

    assert bridge.queue_tab_ids[-1] == "tab-b"


def test_document_edit_tool_resolves_tab_reference_from_payload():
    bridge = _EditBridgeStub()
    tool = DocumentEditTool(bridge=bridge, allow_inline_edits=True)

    payload = {"action": "insert", "content": "Hi", "target_range": [0, 0], "tab": "README.md"}
    tool.run(payload)

    assert bridge.queue_tab_ids[-1] == "tab-a"


def test_document_edit_tool_embeds_tab_when_bridge_lacks_routing():
    bridge = _LegacyBridgeStub()
    tool = DocumentEditTool(bridge=bridge, allow_inline_edits=True)

    tool.run(action="insert", content="Hi", target_range=(0, 0), tab_id="tab-legacy")

    payload = bridge.calls[-1]
    assert isinstance(payload, dict)
    assert payload["tab_id"] == "tab-legacy"
    metadata = payload.get("metadata") or {}
    assert metadata.get("tab_id") == "tab-legacy"


def test_document_edit_tool_accepts_patch_payload():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "patch: +4 chars"
    bridge.last_snapshot_version = "digest-xyz"
    tool = DocumentEditTool(bridge=bridge)

    diff = """--- a/doc.txt
+++ b/doc.txt
@@ -1 +1 @@
-old
+new
"""

    status = tool.run(action="patch", diff=diff, document_version="digest-xyz", content_hash="hash-xyz")

    assert bridge.calls[-1]["action"] == "patch"
    assert "patch" in status and "digest-xyz" in status


def test_document_edit_tool_rejects_patch_without_snapshot_version():
    bridge = _EditBridgeStub()
    bridge.last_snapshot_version = "digest-present"
    tool = DocumentEditTool(bridge=bridge)

    diff = """--- a/doc.txt\n+++ b/doc.txt\n@@ -1 +1 @@\n-old\n+new\n"""

    with pytest.raises(ValueError, match="document_snapshot"):
        tool.run(action="patch", diff=diff)


def test_document_edit_tool_rejects_inline_edits_when_not_allowed():
    bridge = _EditBridgeStub()
    tool = DocumentEditTool(bridge=bridge)

    with pytest.raises(ValueError, match="diff-only"):
        tool.run(action="annotate", content="Note", target_range=(0, 0))

    assert bridge.calls == []


def test_document_edit_tool_auto_converts_replace_when_patch_only_enabled():
    bridge = _PatchBridgeStub(text="Goodbye moon", selection=(0, 7), version="digest-42")
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    status = tool.run(action="replace", content="Hello", target_range=(0, 7))

    assert bridge.calls and bridge.calls[-1]["action"] == "patch"
    assert bridge.calls[-1]["document_version"] == "digest-42"
    assert "digest-42" in status


def test_document_edit_tool_allows_patches_when_patch_only_enabled():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "patch"
    bridge.last_snapshot_version = "digest"
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    diff = """--- a/doc.txt\n+++ b/doc.txt\n@@ -1 +1 @@\n-old\n+new\n"""
    status = tool.run(action="patch", diff=diff, document_version="digest", content_hash="hash-digest")

    assert bridge.calls and bridge.calls[-1]["action"] == "patch"
    assert "patch" in status


def test_document_apply_patch_tool_builds_and_applies_diff():
    bridge = _PatchBridgeStub(text="Alpha beta", selection=(6, 10), version="digest-7")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    status = tool.run(content="BETA", target_range=(6, 10))

    assert bridge.calls and bridge.calls[-1]["action"] == "patch"
    assert "digest-7" in status
    assert bridge.calls[-1]["document_version"] == "digest-7"


def test_document_apply_patch_tool_targets_specific_tab():
    bridge = _PatchBridgeStub(text="Alpha beta", selection=(0, 5), version="digest-8")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    tool.run(content="ALPHA", target_range=(0, 5), tab_id="tab-b")

    assert bridge.queue_tab_ids[-1] == "tab-b"
    assert bridge.snapshot_requests[-1]["tab_id"] == "tab-b"


def test_document_apply_patch_tool_validates_snapshot_version():
    bridge = _PatchBridgeStub(text="Hello world", selection=(0, 5), version="digest-1")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    with pytest.raises(DocumentVersionMismatchError):
        tool.run(content="Hola", target_range=(0, 5), document_version="digest-old")


def test_document_apply_patch_tool_skips_noop_edits():
    bridge = _PatchBridgeStub(text="Hello world", selection=(0, 5), version="digest-2")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    outcome = tool.run(content="Hello", target_range=(0, 5))

    assert outcome.startswith("skipped")
    assert bridge.calls == []


def test_document_apply_patch_tool_handles_missing_rationale():
    bridge = _PatchBridgeStub(text="", selection=(0, 0), version="digest-9")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    status = tool.run(content="Hello world")

    assert bridge.calls and bridge.calls[-1]["action"] == "patch"
    assert "digest-9" in status


def test_diff_builder_tool_generates_diff_and_detects_noop():
    tool = DiffBuilderTool()

    diff = tool.run("alpha\n", "beta\n")

    assert diff.startswith("--- a/")
    assert "beta" in diff

    with pytest.raises(ValueError):
        tool.run("same", "same")


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


def test_document_snapshot_tool_attaches_outline_digest_when_resolver_present():
    provider = _SnapshotProviderStub()
    calls: list[str | None] = []

    def resolver(document_id: str | None) -> str | None:
        calls.append(document_id)
        return "outline-hash" if document_id == "doc-stub" else None

    tool = DocumentSnapshotTool(provider=provider, outline_digest_resolver=resolver)

    snapshot = tool.run()

    assert snapshot["outline_digest"] == "outline-hash"
    assert calls == ["doc-stub"]


def test_document_snapshot_tool_uses_tab_identifier_when_document_id_missing():
    provider = _SnapshotProviderStub()
    provider.snapshot.pop("document_id", None)

    tool = DocumentSnapshotTool(
        provider=provider,
        outline_digest_resolver=lambda doc_id: "tab-digest" if doc_id == "tab-extra" else None,
    )

    snapshot = tool.run(tab_id="tab-extra")

    assert snapshot["outline_digest"] == "tab-digest"


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
    assert result.diff_preview and result.diff_preview.startswith("---")
    assert result.max_replacements == tool.default_max_replacements
    assert result.limited is False
    metadata = payload["metadata"]
    assert metadata["matches"] == 2
    assert metadata["limited"] is False


def test_tools_expose_expected_summarizable_flags():
    bridge = _EditBridgeStub()
    edit_tool = DocumentEditTool(bridge=bridge, allow_inline_edits=True)
    snapshot_tool = DocumentSnapshotTool(provider=_SnapshotProviderStub())
    patch_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
    list_tool = ListTabsTool(provider=bridge)
    search_tool = SearchReplaceTool(bridge=_SearchReplaceBridgeStub(text="text"))
    diff_tool = DiffBuilderTool()

    assert snapshot_tool.summarizable is True
    assert list_tool.summarizable is True
    assert diff_tool.summarizable is True
    assert edit_tool.summarizable is False
    assert patch_tool.summarizable is False
    assert search_tool.summarizable is False
    assert getattr(validate_snippet, "summarizable", None) is False


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
    assert result.diff_preview and result.diff_preview.startswith("---")
    assert result.max_replacements == 1
    assert result.limited is False


def test_search_replace_tool_honors_replacement_cap_and_reports_limited():
    bridge = _SearchReplaceBridgeStub(text="aaaaa", selection=(0, 0), version="digest-cap")
    tool = SearchReplaceTool(bridge=bridge)

    result = tool.run(pattern="a", replacement="b", max_replacements=2)

    assert result.replacements == 2
    assert result.limited is True
    assert result.max_replacements == 2
    assert result.diff_preview
    assert bridge.queue_calls
    payload = bridge.queue_calls[-1]
    metadata = payload["metadata"]
    assert metadata["limited"] is True
    assert metadata["matches"] == 2
    assert metadata["max_replacements"] == 2


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


def test_validate_snippet_flags_markdown_heading_jumps():
    markdown = """# Title\n### Skipped level"""

    outcome = validate_snippet(markdown, "markdown")

    assert outcome.ok is False
    assert "Heading level" in outcome.message
    assert "Line 2" in outcome.message


def test_validate_snippet_detects_unclosed_markdown_fence():
    markdown = """```python\ncode sample"""

    outcome = validate_snippet(markdown, "md")

    assert outcome.ok is False
    assert "fenced code" in outcome.message.lower()


def test_register_snippet_validator_supports_custom_formats(monkeypatch: pytest.MonkeyPatch):
    from tinkerbell.ai.tools import validation as validation_module

    registry_copy = dict(validation_module._FORMAT_VALIDATORS)
    monkeypatch.setattr(validation_module, "_FORMAT_VALIDATORS", registry_copy)

    def _fake_validator(text: str):
        if "invalid" in text:
            return [yaml_json.ValidationError(message="custom failure", line=3)]
        return []

    validation_module.register_snippet_validator("custom", _fake_validator)

    failure = validation_module.validate_snippet("invalid", "custom")
    assert failure.ok is False
    assert "custom failure" in failure.message

    success = validation_module.validate_snippet("valid", "custom")
    assert success.ok is True
