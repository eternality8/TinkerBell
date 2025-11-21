"""Unit tests for AIController retry handling."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, MutableMapping, cast

from tinkerbell.ai.client import AIClient
from tinkerbell.ai.orchestration import controller as controller_module
from tinkerbell.ai.orchestration.controller import AIController, ToolRegistration, _ToolCallRequest
from tinkerbell.services.bridge import DocumentVersionMismatchError


class _RetryingTool:
    def __init__(self, failures: int, *, cause: str = "hash_mismatch") -> None:
        self.failures = failures
        self.calls = 0
        self.cause = cause

    def run(self, **_: Any) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise DocumentVersionMismatchError("stale", cause=self.cause)
        return f"ok-{self.calls}"


class _SnapshotTool:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        return {"document_id": "doc-123", "version": f"digest-{self.calls}"}


def _controller_with_tools(apply_failures: int, *, cause: str = "hash_mismatch") -> tuple[AIController, _RetryingTool, _SnapshotTool]:
    flaky_tool = _RetryingTool(failures=apply_failures, cause=cause)
    snapshot_tool = _SnapshotTool()
    controller = AIController(
        client=cast(AIClient, SimpleNamespace()),
        tools=cast(
            MutableMapping[str, ToolRegistration],
            {
                "document_apply_patch": flaky_tool,
                "document_snapshot": snapshot_tool,
            },
        ),
    )
    return controller, flaky_tool, snapshot_tool


def test_ai_controller_retries_document_version_mismatch_success(monkeypatch) -> None:
    emitted: list[tuple[str, dict[str, Any] | None]] = []

    def _capture(event: str, payload: dict[str, Any] | None = None) -> None:
        emitted.append((event, payload))

    monkeypatch.setattr(controller_module.telemetry_service, "emit", _capture)
    controller, tool, snapshot = _controller_with_tools(apply_failures=1)

    call = _ToolCallRequest(
        call_id="call-1",
        name="document_apply_patch",
        index=0,
        arguments='{"tab_id":"tab-a"}',
        parsed=None,
    )

    messages, records, _ = asyncio.run(controller._handle_tool_calls([call], on_event=None))

    assert tool.calls == 2
    assert snapshot.calls == 1
    assert messages[0]["content"].startswith("ok-")
    assert "retry" in records[0]
    assert records[0]["retry"]["status"] == "success"

    retry_events = [payload for name, payload in emitted if name == "document_edit.retry" and payload]
    assert retry_events and retry_events[-1]["status"] == "success"
    assert retry_events[-1]["tab_id"] == "tab-a"


def test_ai_controller_reports_retry_failure(monkeypatch) -> None:
    emitted: list[tuple[str, dict[str, Any] | None]] = []

    def _capture(event: str, payload: dict[str, Any] | None = None) -> None:
        emitted.append((event, payload))

    monkeypatch.setattr(controller_module.telemetry_service, "emit", _capture)
    controller, tool, snapshot = _controller_with_tools(apply_failures=2)

    call = _ToolCallRequest(
        call_id="call-2",
        name="document_apply_patch",
        index=0,
        arguments='{"tab_id":"tab-b"}',
        parsed=None,
    )

    messages, records, _ = asyncio.run(controller._handle_tool_calls([call], on_event=None))

    assert tool.calls == 2
    assert snapshot.calls == 1
    assert messages[0]["content"].startswith("Tool 'document_apply_patch' failed:")
    assert records[0]["status"] == "failed"
    assert records[0]["retry"]["status"] == "failed"
    assert records[0]["retry"]["reason"] == "retry_exhausted"

    retry_events = [payload for name, payload in emitted if name == "document_edit.retry" and payload]
    assert retry_events and retry_events[-1]["status"] == "failed"
    assert retry_events[-1]["reason"] == "retry_exhausted"


def test_ai_controller_retry_event_includes_cause(monkeypatch) -> None:
    emitted: list[tuple[str, dict[str, Any] | None]] = []

    def _capture(event: str, payload: dict[str, Any] | None = None) -> None:
        emitted.append((event, payload))

    monkeypatch.setattr(controller_module.telemetry_service, "emit", _capture)
    controller, tool, snapshot = _controller_with_tools(apply_failures=2, cause="chunk_hash_mismatch")

    call = _ToolCallRequest(
        call_id="call-3",
        name="document_apply_patch",
        index=0,
        arguments='{"tab_id":"tab-c"}',
        parsed=None,
    )

    messages, records, _ = asyncio.run(controller._handle_tool_calls([call], on_event=None))

    assert tool.calls == 2
    assert snapshot.calls == 1
    assert messages[0]["content"].startswith("Tool 'document_apply_patch' failed:")
    assert records[0]["retry"]["cause"] == "chunk_hash_mismatch"

    retry_events = [payload for name, payload in emitted if name == "document_edit.retry" and payload]
    assert retry_events and retry_events[-1]["cause"] == "chunk_hash_mismatch"