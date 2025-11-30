"""Tests for the tool base classes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tinkerbell.ai.tools.base import (
    BaseTool,
    ReadOnlyTool,
    WriteTool,
    SubagentTool,
    ToolResult,
    ToolContext,
    DocumentProvider,
)
from tinkerbell.ai.tools.version import VersionManager, VersionToken
from tinkerbell.ai.tools.errors import (
    ToolError,
    InvalidTabIdError,
    InvalidParameterError,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

class MockDocumentProvider:
    """Mock document provider for testing."""

    def __init__(self, text: str = "Hello World", active_tab: str | None = "tab-1") -> None:
        self._text = text
        self._active_tab = active_tab
        self._documents: dict[str, str] = {}

    def get_document_text(self, tab_id: str | None = None) -> str:
        return self._text

    def get_active_tab_id(self) -> str | None:
        return self._active_tab

    def get_document_content(self, tab_id: str) -> str | None:
        return self._documents.get(tab_id, self._text)

    def set_document_content(self, tab_id: str, content: str) -> None:
        self._documents[tab_id] = content

    def get_document_metadata(self, tab_id: str) -> dict[str, Any] | None:
        return {"tab_id": tab_id, "path": f"/{tab_id}.txt", "language": "plain_text"}


class MockTelemetry:
    """Mock telemetry emitter for testing."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_name: str, payload: dict[str, Any]) -> None:
        self.events.append((event_name, dict(payload)))


@pytest.fixture
def version_manager() -> VersionManager:
    """Create a fresh VersionManager for each test."""
    manager = VersionManager()
    manager.register_tab("tab-1", "doc-abc", "hash123")
    return manager


@pytest.fixture
def doc_provider() -> MockDocumentProvider:
    """Create a mock document provider."""
    return MockDocumentProvider()


@pytest.fixture
def telemetry() -> MockTelemetry:
    """Create a mock telemetry emitter."""
    return MockTelemetry()


@pytest.fixture
def context(
    doc_provider: MockDocumentProvider,
    version_manager: VersionManager,
    telemetry: MockTelemetry,
) -> ToolContext:
    """Create a tool context for testing."""
    return ToolContext(
        document_provider=doc_provider,
        version_manager=version_manager,
        tab_id="tab-1",
        telemetry=telemetry,
        request_id="req-123",
    )


# -----------------------------------------------------------------------------
# ToolResult Tests
# -----------------------------------------------------------------------------

class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating a successful result."""
        result = ToolResult(
            success=True,
            data={"content": "Hello"},
            duration_ms=15.5,
        )

        assert result.success is True
        assert result.data == {"content": "Hello"}
        assert result.duration_ms == 15.5

    def test_success_to_dict(self) -> None:
        """Test serializing successful result."""
        result = ToolResult(
            success=True,
            data={"lines": 10, "text": "Hello"},
            metadata={"cached": True},
        )

        output = result.to_dict()

        assert output["lines"] == 10
        assert output["text"] == "Hello"
        assert output["_metadata"]["cached"] is True

    def test_error_result(self) -> None:
        """Test creating an error result."""
        error = ToolError(error_code="test_error", message="Something went wrong")
        result = ToolResult(
            success=False,
            error=error,
            duration_ms=5.0,
        )

        assert result.success is False
        assert result.error is error

    def test_error_to_dict(self) -> None:
        """Test serializing error result."""
        error = ToolError(
            error_code="test_error",
            message="Test message",
            suggestion="Try again",
        )
        result = ToolResult(success=False, error=error)

        output = result.to_dict()

        assert output["error"] == "test_error"
        assert output["message"] == "Test message"
        assert output["suggestion"] == "Try again"


# -----------------------------------------------------------------------------
# ToolContext Tests
# -----------------------------------------------------------------------------

class TestToolContext:
    """Tests for ToolContext dataclass."""

    def test_resolve_tab_id_explicit(self, context: ToolContext) -> None:
        """Test resolving explicit tab_id."""
        context.tab_id = "explicit-tab"
        assert context.resolve_tab_id() == "explicit-tab"

    def test_resolve_tab_id_from_active(
        self,
        doc_provider: MockDocumentProvider,
        version_manager: VersionManager,
    ) -> None:
        """Test resolving tab_id from active tab."""
        context = ToolContext(
            document_provider=doc_provider,
            version_manager=version_manager,
            tab_id=None,
        )
        assert context.resolve_tab_id() == "tab-1"

    def test_resolve_tab_id_no_active(
        self,
        version_manager: VersionManager,
    ) -> None:
        """Test None returned when no tab_id and no active tab."""
        provider = MockDocumentProvider(active_tab=None)
        context = ToolContext(
            document_provider=provider,
            version_manager=version_manager,
            tab_id=None,
        )

        # resolve_tab_id returns None when no tab available
        assert context.resolve_tab_id() is None

    def test_require_tab_id_no_active(
        self,
        version_manager: VersionManager,
    ) -> None:
        """Test error when no tab_id and no active tab using require_tab_id."""
        provider = MockDocumentProvider(active_tab=None)
        context = ToolContext(
            document_provider=provider,
            version_manager=version_manager,
            tab_id=None,
        )

        with pytest.raises(InvalidTabIdError, match="no active tab"):
            context.require_tab_id()


# -----------------------------------------------------------------------------
# BaseTool Tests
# -----------------------------------------------------------------------------

class TestBaseTool:
    """Tests for BaseTool abstract class."""

    def test_basic_execution(self, context: ToolContext) -> None:
        """Test basic tool execution."""

        class SimpleTool(BaseTool):
            name = "simple_tool"

            def execute(self, ctx: ToolContext, params: dict) -> dict:
                return {"result": "success", "value": params.get("value", 0)}

        tool = SimpleTool()
        result = tool.run(context, {"value": 42})

        assert result.success is True
        assert result.data["result"] == "success"
        assert result.data["value"] == 42
        assert result.duration_ms > 0

    def test_tool_error_handling(self, context: ToolContext) -> None:
        """Test that ToolError is captured properly."""

        class FailingTool(BaseTool):
            name = "failing_tool"

            def execute(self, ctx: ToolContext, params: dict) -> dict:
                raise InvalidParameterError(
                    message="Bad input",
                    parameter="value",
                )

        tool = FailingTool()
        result = tool.run(context, {})

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "invalid_parameter"
        assert result.error.message == "Bad input"

    def test_unexpected_error_handling(self, context: ToolContext) -> None:
        """Test that unexpected errors are wrapped."""

        class CrashingTool(BaseTool):
            name = "crashing_tool"

            def execute(self, ctx: ToolContext, params: dict) -> dict:
                raise RuntimeError("Unexpected crash!")

        tool = CrashingTool()
        result = tool.run(context, {})

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "internal_error"
        assert "Unexpected crash!" in result.error.message

    def test_validation_hook(self, context: ToolContext) -> None:
        """Test that validate() is called before execute()."""

        class ValidatingTool(BaseTool):
            name = "validating_tool"

            def validate(self, params: dict) -> None:
                if "required" not in params:
                    raise InvalidParameterError(
                        message="Missing required param",
                        parameter="required",
                    )

            def execute(self, ctx: ToolContext, params: dict) -> dict:
                return {"value": params["required"]}

        tool = ValidatingTool()

        # Without required param
        result = tool.run(context, {})
        assert result.success is False
        assert "required" in result.error.message

        # With required param
        result = tool.run(context, {"required": "hello"})
        assert result.success is True
        assert result.data["value"] == "hello"

    def test_telemetry_emission(
        self,
        context: ToolContext,
        telemetry: MockTelemetry,
    ) -> None:
        """Test that telemetry is emitted."""

        class TelemetryTool(BaseTool):
            name = "telemetry_tool"

            def execute(self, ctx: ToolContext, params: dict) -> dict:
                return {"done": True}

        tool = TelemetryTool()
        tool.run(context, {})

        assert len(telemetry.events) == 1
        event_name, payload = telemetry.events[0]
        assert event_name == "tool.telemetry_tool"
        assert payload["success"] is True
        assert payload["request_id"] == "req-123"
        assert payload["tab_id"] == "tab-1"

    def test_telemetry_on_error(
        self,
        context: ToolContext,
        telemetry: MockTelemetry,
    ) -> None:
        """Test telemetry includes error info on failure."""

        class ErrorTool(BaseTool):
            name = "error_tool"

            def execute(self, ctx: ToolContext, params: dict) -> dict:
                raise ToolError(error_code="test_err", message="Test")

        tool = ErrorTool()
        tool.run(context, {})

        assert len(telemetry.events) == 1
        _, payload = telemetry.events[0]
        assert payload["success"] is False
        assert payload["error_code"] == "test_err"

    def test_summarizable_flag(self) -> None:
        """Test default summarizable flag."""

        class SummarizableTool(BaseTool):
            name = "summarizable"
            summarizable = True

            def execute(self, ctx: ToolContext, params: dict) -> dict:
                return {}

        class NonSummarizableTool(BaseTool):
            name = "non_summarizable"
            summarizable = False

            def execute(self, ctx: ToolContext, params: dict) -> dict:
                return {}

        assert SummarizableTool.summarizable is True
        assert NonSummarizableTool.summarizable is False


# -----------------------------------------------------------------------------
# ReadOnlyTool Tests
# -----------------------------------------------------------------------------

class TestReadOnlyTool:
    """Tests for ReadOnlyTool base class."""

    def test_auto_includes_version(self, context: ToolContext) -> None:
        """Test that version is auto-attached to response."""

        class ReadTool(ReadOnlyTool):
            name = "read_tool"

            def read(self, ctx: ToolContext, params: dict) -> dict:
                return {"content": "data"}

        tool = ReadTool()
        result = tool.run(context, {})

        assert result.success is True
        assert "version" in result.data
        # New short format: "tab1:hash:1" (4 chars from tab, 4 from hash, version)
        version = result.data["version"]
        assert ":" in version  # Should be colon-separated
        parts = version.split(":")
        assert len(parts) == 3  # short_tab:short_hash:version_id

    def test_preserves_explicit_version(self, context: ToolContext) -> None:
        """Test that explicit version is not overwritten."""

        class ReadToolWithVersion(ReadOnlyTool):
            name = "read_tool"

            def read(self, ctx: ToolContext, params: dict) -> dict:
                return {"content": "data", "version": "custom-version"}

        tool = ReadToolWithVersion()
        result = tool.run(context, {})

        assert result.data["version"] == "custom-version"


# -----------------------------------------------------------------------------
# WriteTool Tests
# -----------------------------------------------------------------------------

class TestWriteTool:
    """Tests for WriteTool base class."""

    def test_requires_version_token(self, context: ToolContext) -> None:
        """Test that version token is required."""

        class WriteTestTool(WriteTool):
            name = "write_test"

            def write(self, ctx: ToolContext, params: dict, token: VersionToken) -> dict:
                return {"written": True}

        tool = WriteTestTool()
        result = tool.run(context, {})  # No version token

        assert result.success is False
        assert "version" in result.error.message.lower()

    def test_validates_version_token(self, context: ToolContext) -> None:
        """Test that version token is validated."""

        class WriteTestTool(WriteTool):
            name = "write_test"

            def write(self, ctx: ToolContext, params: dict, token: VersionToken) -> dict:
                return {"written": True}

        tool = WriteTestTool()

        # Valid token
        current = context.version_manager.get_current_token("tab-1")
        result = tool.run(context, {"version": current.to_string()})
        assert result.success is True

    def test_stale_token_rejected(self, context: ToolContext) -> None:
        """Test that stale tokens are rejected."""

        class WriteTestTool(WriteTool):
            name = "write_test"

            def write(self, ctx: ToolContext, params: dict, token: VersionToken) -> dict:
                return {"written": True}

        tool = WriteTestTool()

        # Get current token, then increment version
        old_token = context.version_manager.get_current_token("tab-1")
        context.version_manager.increment_version("tab-1", "newhash")

        result = tool.run(context, {"version": old_token.to_string()})

        assert result.success is False
        assert result.error.error_code == "version_mismatch"

    def test_dry_run_mode(self, context: ToolContext) -> None:
        """Test dry-run mode doesn't apply changes."""

        class WriteTestTool(WriteTool):
            name = "write_test"
            write_called = False

            def write(self, ctx: ToolContext, params: dict, token: VersionToken) -> dict:
                self.write_called = True
                return {"written": True}

            def preview(self, ctx: ToolContext, params: dict, token: VersionToken) -> dict:
                return {"preview": "Would write"}

        tool = WriteTestTool()
        current = context.version_manager.get_current_token("tab-1")

        result = tool.run(context, {"version": current.to_string(), "dry_run": True})

        assert result.success is True
        assert result.data["dry_run"] is True
        assert result.data["preview"] == "Would write"
        assert tool.write_called is False

    def test_version_incremented_on_success(self, context: ToolContext) -> None:
        """Test that version is incremented after successful write."""

        class WriteTestTool(WriteTool):
            name = "write_test"

            def write(self, ctx: ToolContext, params: dict, token: VersionToken) -> dict:
                return {"written": True, "_new_text": "Updated content"}

        tool = WriteTestTool()
        old_token = context.version_manager.get_current_token("tab-1")

        result = tool.run(context, {"version": old_token.to_string()})

        assert result.success is True
        new_version = result.data["version"]
        assert new_version != old_token.to_string()

        # Verify version was actually incremented
        current = context.version_manager.get_current_token("tab-1")
        assert current.version_id == old_token.version_id + 1


# -----------------------------------------------------------------------------
# SubagentTool Tests
# -----------------------------------------------------------------------------

class TestSubagentTool:
    """Tests for SubagentTool base class."""

    def test_basic_subagent_workflow(self, context: ToolContext) -> None:
        """Test basic plan -> execute -> aggregate flow."""

        class AnalyzeTool(SubagentTool):
            name = "analyze_tool"

            def plan(self, ctx: ToolContext, params: dict) -> list[dict]:
                return [{"chunk": 1}, {"chunk": 2}, {"chunk": 3}]

            def execute_subagent(self, ctx: ToolContext, task: dict) -> dict:
                return {"analyzed": task["chunk"]}

            def aggregate(self, results: list[dict]) -> dict:
                return {"total_chunks": len(results)}

        tool = AnalyzeTool()
        result = tool.run(context, {})

        assert result.success is True
        assert result.data["total_chunks"] == 3

    def test_empty_plan(self, context: ToolContext) -> None:
        """Test handling of empty task plan."""

        class EmptyTool(SubagentTool):
            name = "empty_tool"

            def plan(self, ctx: ToolContext, params: dict) -> list[dict]:
                return []

            def aggregate(self, results: list[dict]) -> dict:
                return {"count": len(results)}

        tool = EmptyTool()
        result = tool.run(context, {})

        assert result.success is True
        assert result.data["status"] == "no_tasks"

    def test_partial_failure_handling(self, context: ToolContext) -> None:
        """Test that partial failures are captured."""

        class PartialFailTool(SubagentTool):
            name = "partial_fail"

            def plan(self, ctx: ToolContext, params: dict) -> list[dict]:
                return [{"chunk": 1}, {"chunk": 2}, {"chunk": 3}]

            def execute_subagent(self, ctx: ToolContext, task: dict) -> dict:
                if task["chunk"] == 2:
                    raise RuntimeError("Chunk 2 failed!")
                return {"analyzed": task["chunk"]}

            def aggregate(self, results: list[dict]) -> dict:
                return {"successful_chunks": [r["analyzed"] for r in results]}

        tool = PartialFailTool()
        result = tool.run(context, {})

        assert result.success is True
        assert result.data["successful_chunks"] == [1, 3]
        assert result.data["partial_errors"] is not None
        assert len(result.data["partial_errors"]) == 1
        assert result.data["completed"] == 2
        assert result.data["total"] == 3
