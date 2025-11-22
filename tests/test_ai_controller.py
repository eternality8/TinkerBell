"""Unit tests for AIController retry handling."""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping, cast

import pytest

from tinkerbell.ai.client import AIClient
from tinkerbell.ai.orchestration import controller as controller_module
from tinkerbell.ai.orchestration.controller import AIController, ToolRegistration, _ToolCallRequest
from tinkerbell.ai.tools.document_apply_patch import NeedsRangeError
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


class _ConcurrentRetryTool:
    def __init__(self, failures_before_success: int = 1) -> None:
        self.failures_before_success = failures_before_success
        self.attempts: dict[str, int] = {}
        self.concurrent = 0
        self.max_concurrent = 0

    async def run(self, **kwargs: Any) -> str:
        request_id = str(kwargs.get("request_id") or kwargs.get("tab_id") or uuid.uuid4())
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(0)
            count = self.attempts.get(request_id, 0) + 1
            self.attempts[request_id] = count
            if count <= self.failures_before_success:
                raise DocumentVersionMismatchError("stale", cause="hash_mismatch")
            return f"ok-{request_id}-{count}"
        finally:
            self.concurrent -= 1


class _SnapshotTool:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        return {"document_id": "doc-123", "version": f"digest-{self.calls}"}


class _NeedsRangeTool:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, **_: Any) -> str:
        self.calls += 1
        raise NeedsRangeError(
            "needs_range: Provide explicit bounds",
            content_length=2_048,
            threshold=1_024,
        )


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


def test_ai_controller_surfaces_needs_range_error() -> None:
    controller = AIController(
        client=cast(AIClient, SimpleNamespace()),
        tools=cast(
            MutableMapping[str, ToolRegistration],
            {
                "document_apply_patch": _NeedsRangeTool(),
            },
        ),
    )

    call = _ToolCallRequest(
        call_id="call-needs-range",
        name="document_apply_patch",
        index=0,
        arguments='{"tab_id":"tab-z"}',
        parsed=None,
    )

    messages, records, _ = asyncio.run(controller._handle_tool_calls([call], on_event=None))

    assert messages[0]["content"].startswith("Tool 'document_apply_patch' failed: needs_range")
    payload = records[0]["raw_result"]
    assert isinstance(payload, Mapping)
    assert payload["needs_range"] is True
    assert payload["error"] == "needs_range"
    assert payload["tab_id"] == "tab-z"
    assert records[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_ai_controller_load_handles_concurrent_auto_retries() -> None:
    concurrent_tool = _ConcurrentRetryTool(failures_before_success=1)
    snapshot_tool = _SnapshotTool()
    controller = AIController(
        client=cast(AIClient, SimpleNamespace()),
        tools=cast(
            MutableMapping[str, ToolRegistration],
            {
                "document_apply_patch": concurrent_tool,
                "document_snapshot": snapshot_tool,
            },
        ),
    )

    async def _invoke(call: _ToolCallRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        return await controller._handle_tool_calls([call], on_event=None)

    call_count = 5
    calls = []
    for idx in range(call_count):
        request_id = f"req-{idx}"
        arguments = json.dumps({"tab_id": f"tab-{idx}", "request_id": request_id})
        calls.append(
            _ToolCallRequest(
                call_id=f"call-{idx}",
                name="document_apply_patch",
                index=idx,
                arguments=arguments,
                parsed=None,
            )
        )

    results = await asyncio.gather(*(_invoke(call) for call in calls))

    assert concurrent_tool.max_concurrent >= 2
    assert snapshot_tool.calls == call_count

    for idx, (messages, records, _) in enumerate(results):
        assert messages and messages[0]["content"].startswith("ok-req-")
        record = records[0]
        retry = record.get("retry")
        assert retry is not None
        assert retry["status"] == "success"
        attempt_key = f"req-{idx}"
        assert concurrent_tool.attempts[attempt_key] == concurrent_tool.failures_before_success + 1