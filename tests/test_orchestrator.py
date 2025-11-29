"""Unit tests for AIOrchestrator."""

from __future__ import annotations

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

from tinkerbell.ai.orchestration.orchestrator import (
    AIOrchestrator,
    OrchestratorConfig,
    ChatResult,
    _ContentDeltaEvent,
    _AIClientAdapter,
    _DispatcherToolExecutor,
    AnalysisAgentAdapter,
)
from tinkerbell.ai.orchestration.types import (
    TurnOutput,
    TurnMetrics,
    ToolCallRecord,
)
from tinkerbell.ai.tools.tool_registry import ToolRegistry, ToolSchema


# =============================================================================
# Test Fixtures
# =============================================================================


class MockAIClient:
    """Mock AI client for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Test response"]
        self._call_count = 0
        self._stream_calls: list[dict[str, Any]] = []

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float = 0.2,
        max_completion_tokens: int | None = None,
        **kwargs: Any,
    ):
        self._stream_calls.append({
            "messages": list(messages),
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        })
        response = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        
        @dataclass
        class ContentDelta:
            type: str = "content.delta"
            content: str = ""
        
        @dataclass
        class ContentDone:
            type: str = "content.done"
            content: str = ""
        
        yield ContentDelta(content=response)
        yield ContentDone(content=response)


class MockToolDispatcher:
    """Mock tool dispatcher."""

    def __init__(self):
        self.dispatches: list[tuple[str, dict]] = []

    async def dispatch(self, name: str, arguments: dict) -> Any:
        self.dispatches.append((name, arguments))
        
        @dataclass
        class MockResult:
            success: bool = True
            result: str = "Tool executed"
            error: Any = None
        
        return MockResult()


@pytest.fixture
def mock_client():
    """Create a mock AI client."""
    return MockAIClient()


@pytest.fixture
def mock_registry():
    """Create a mock tool registry."""
    return ToolRegistry()


# =============================================================================
# OrchestratorConfig Tests
# =============================================================================


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = OrchestratorConfig()
        assert config.max_iterations == 8
        assert config.max_context_tokens == 128_000
        assert config.response_token_reserve == 16_000
        assert config.temperature == 0.2
        assert config.streaming_enabled is True

    def test_custom_values(self):
        """Can create config with custom values."""
        config = OrchestratorConfig(
            max_iterations=5,
            max_context_tokens=64_000,
            response_token_reserve=8_000,
            temperature=0.5,
            streaming_enabled=False,
        )
        assert config.max_iterations == 5
        assert config.max_context_tokens == 64_000
        assert config.response_token_reserve == 8_000
        assert config.temperature == 0.5
        assert config.streaming_enabled is False

    def test_is_frozen(self):
        """Config is immutable."""
        config = OrchestratorConfig()
        with pytest.raises(AttributeError):
            config.max_iterations = 10


# =============================================================================
# ChatResult Tests
# =============================================================================


class TestChatResult:
    """Tests for ChatResult."""

    def test_default_values(self):
        """Default result has success=True."""
        result = ChatResult(response="Hello")
        assert result.response == "Hello"
        assert result.tool_calls == []
        assert result.success is True
        assert result.error is None
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.duration_ms == 0.0

    def test_from_turn_output_success(self):
        """Can convert from successful TurnOutput."""
        output = TurnOutput(
            response="Test response",
            tool_calls=(
                ToolCallRecord(
                    call_id="tc1",
                    name="read_document",
                    arguments='{"tab_id": "t1"}',
                    result="content",
                    success=True,
                ),
            ),
            metrics=TurnMetrics(
                prompt_tokens=100,
                completion_tokens=50,
                duration_ms=1500.0,
            ),
            success=True,
        )
        result = ChatResult.from_turn_output(output)
        assert result.response == "Test response"
        assert result.success is True
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.duration_ms == 1500.0
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "read_document"

    def test_from_turn_output_error(self):
        """Can convert from failed TurnOutput."""
        output = TurnOutput(
            response="",
            success=False,
            error="Something went wrong",
            metrics=TurnMetrics(duration_ms=500.0),
        )
        result = ChatResult.from_turn_output(output)
        assert result.response == ""
        assert result.success is False
        assert result.error == "Something went wrong"


# =============================================================================
# AIOrchestrator Basic Tests
# =============================================================================


class TestAIOrchestratorInit:
    """Tests for AIOrchestrator initialization."""

    def test_init_with_client(self, mock_client):
        """Can init with just a client."""
        orchestrator = AIOrchestrator(client=mock_client)
        assert orchestrator.client is mock_client
        assert orchestrator.config is not None
        assert orchestrator.tool_registry is not None
        assert orchestrator.tool_dispatcher is None
        assert orchestrator.services is None

    def test_init_with_config(self, mock_client):
        """Can init with custom config."""
        config = OrchestratorConfig(max_iterations=5)
        orchestrator = AIOrchestrator(client=mock_client, config=config)
        assert orchestrator.config.max_iterations == 5

    def test_init_with_registry(self, mock_client, mock_registry):
        """Can init with custom registry."""
        orchestrator = AIOrchestrator(
            client=mock_client,
            tool_registry=mock_registry,
        )
        assert orchestrator.tool_registry is mock_registry


class TestAIOrchestratorConfiguration:
    """Tests for AIOrchestrator configuration methods."""

    def test_configure_tool_dispatcher(self, mock_client):
        """Can configure tool dispatcher."""
        orchestrator = AIOrchestrator(client=mock_client)
        assert orchestrator.tool_dispatcher is None
        
        mock_provider = MagicMock()
        orchestrator.configure_tool_dispatcher(context_provider=mock_provider)
        
        assert orchestrator.tool_dispatcher is not None

    def test_update_client(self, mock_client):
        """Can update the AI client."""
        orchestrator = AIOrchestrator(client=mock_client)
        
        new_client = MockAIClient(responses=["New response"])
        orchestrator.update_client(new_client)
        
        assert orchestrator.client is new_client

    def test_set_config(self, mock_client):
        """Can update configuration."""
        orchestrator = AIOrchestrator(client=mock_client)
        
        new_config = OrchestratorConfig(max_iterations=10)
        orchestrator.set_config(new_config)
        
        assert orchestrator.config.max_iterations == 10


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Tests for tool registration."""

    def test_register_tool(self, mock_client, mock_registry):
        """Can register a tool."""
        orchestrator = AIOrchestrator(
            client=mock_client,
            tool_registry=mock_registry,
        )
        
        def dummy_tool(**kwargs):
            return "result"
        
        orchestrator.register_tool(
            name="dummy",
            tool=dummy_tool,
            description="A dummy tool",
            parameters={"type": "object", "properties": {}},
        )
        
        assert mock_registry.has_tool("dummy")

    def test_unregister_tool(self, mock_client, mock_registry):
        """Can unregister a tool."""
        orchestrator = AIOrchestrator(
            client=mock_client,
            tool_registry=mock_registry,
        )
        
        # First register
        mock_registry.register(
            lambda: "result",
            name="temp",
            description="Temp tool",
        )
        assert mock_registry.has_tool("temp")
        
        # Then unregister
        orchestrator.unregister_tool("temp")
        assert not mock_registry.has_tool("temp")

    def test_available_tools(self, mock_client, mock_registry):
        """Can list available tools."""
        orchestrator = AIOrchestrator(
            client=mock_client,
            tool_registry=mock_registry,
        )
        
        mock_registry.register(
            lambda: "a",
            name="tool_a",
            description="Tool A",
        )
        mock_registry.register(
            lambda: "b",
            name="tool_b",
            description="Tool B",
        )
        
        tools = orchestrator.available_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools


# =============================================================================
# Cancel and Close Tests
# =============================================================================


class TestCancelAndClose:
    """Tests for cancel and close operations."""

    def test_cancel_no_task(self, mock_client):
        """Cancel with no active task is a no-op."""
        orchestrator = AIOrchestrator(client=mock_client)
        # Should not raise
        orchestrator.cancel()

    @pytest.mark.asyncio
    async def test_aclose_closes_client(self, mock_client):
        """aclose calls client.aclose if available."""
        mock_client.aclose = AsyncMock()
        orchestrator = AIOrchestrator(client=mock_client)
        
        await orchestrator.aclose()
        
        mock_client.aclose.assert_called_once()


# =============================================================================
# Helper Class Tests
# =============================================================================


class TestContentDeltaEvent:
    """Tests for _ContentDeltaEvent."""

    def test_default_type(self):
        """Default type is content.delta."""
        event = _ContentDeltaEvent(content="Hello")
        assert event.type == "content.delta"
        assert event.content == "Hello"

    def test_custom_type(self):
        """Can set custom type."""
        event = _ContentDeltaEvent(content="Hello", type="custom")
        assert event.type == "custom"


class TestAIClientAdapter:
    """Tests for _AIClientAdapter."""

    @pytest.mark.asyncio
    async def test_stream_chat_forwards_calls(self, mock_client):
        """Adapter forwards stream_chat calls."""
        adapter = _AIClientAdapter(mock_client)
        
        messages = [{"role": "user", "content": "Hello"}]
        events = []
        async for event in adapter.stream_chat(
            messages=messages,
            tools=None,
            temperature=0.5,
            max_completion_tokens=100,
        ):
            events.append(event)
        
        assert len(events) == 2
        assert mock_client._stream_calls[0]["messages"] == messages
        assert mock_client._stream_calls[0]["temperature"] == 0.5


class TestDispatcherToolExecutor:
    """Tests for _DispatcherToolExecutor."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Execute returns result string on success."""
        dispatcher = MockToolDispatcher()
        executor = _DispatcherToolExecutor(dispatcher)
        
        result = await executor.execute("read_document", {"tab_id": "t1"})
        
        assert result == "Tool executed"
        assert len(dispatcher.dispatches) == 1
        assert dispatcher.dispatches[0] == ("read_document", {"tab_id": "t1"})

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Execute returns error message on failure."""
        @dataclass
        class MockError:
            message: str = "Tool failed"
        
        @dataclass
        class FailedResult:
            success: bool = False
            result: Any = None
            error: MockError = None
            
            def __post_init__(self):
                self.error = MockError()
        
        class FailingDispatcher:
            async def dispatch(self, name, args):
                return FailedResult()
        
        executor = _DispatcherToolExecutor(FailingDispatcher())
        result = await executor.execute("fail_tool", {})
        
        assert "Error:" in result
        assert "Tool failed" in result


class TestAnalysisAgentAdapter:
    """Tests for AnalysisAgentAdapter."""

    def test_run_analysis_success(self):
        """Adapter delegates to agent and returns advice."""
        @dataclass
        class MockAdvice:
            document_id: str
            chunk_profile: str = "auto"
            required_tools: tuple = ()
            optional_tools: tuple = ()
            must_refresh_outline: bool = False
            plot_state_status: str | None = None
            concordance_status: str | None = None
            warnings: tuple = ()

        class MockAgent:
            def __init__(self):
                self.calls: list[tuple] = []

            def analyze(self, analysis_input, *, force_refresh=False, source="pipeline"):
                self.calls.append((analysis_input, force_refresh, source))
                return MockAdvice(document_id=analysis_input.document_id)

        agent = MockAgent()
        adapter = AnalysisAgentAdapter(agent)
        
        snapshot = {
            "document_id": "doc-1",
            "version_id": "v1",
            "path": "/test/doc.txt",
            "span_start": 0,
            "span_end": 1000,
            "document_chars": 5000,  # Above threshold
        }
        
        result = adapter.run_analysis(snapshot, source="test", force_refresh=True)
        
        assert result is not None
        assert result.document_id == "doc-1"
        assert len(agent.calls) == 1
        assert agent.calls[0][1] is True  # force_refresh
        assert agent.calls[0][2] == "test"  # source

    def test_run_analysis_agent_failure(self):
        """Adapter returns None when agent raises."""
        class FailingAgent:
            def analyze(self, *args, **kwargs):
                raise RuntimeError("Analysis failed")

        adapter = AnalysisAgentAdapter(FailingAgent())
        
        result = adapter.run_analysis({
            "document_id": "doc-1",
            "document_chars": 1000,
        })
        
        assert result is None

    def test_run_analysis_skips_empty_snapshot(self):
        """Adapter skips analysis for empty snapshot (no document ID)."""
        class MockAgent:
            def __init__(self):
                self.called = False

            def analyze(self, analysis_input, **kwargs):
                self.called = True
                return None

        agent = MockAgent()
        adapter = AnalysisAgentAdapter(agent)
        
        result = adapter.run_analysis({})
        
        assert result is None
        assert agent.called is False  # Agent should not be called

    def test_run_analysis_skips_short_document(self):
        """Adapter skips analysis for documents below char threshold."""
        class MockAgent:
            def __init__(self):
                self.called = False

            def analyze(self, analysis_input, **kwargs):
                self.called = True
                return None

        agent = MockAgent()
        adapter = AnalysisAgentAdapter(agent, min_chars=500)
        
        result = adapter.run_analysis({
            "document_id": "doc-1",
            "document_chars": 100,  # Below threshold
        })
        
        assert result is None
        assert agent.called is False

    def test_run_analysis_skips_short_span(self):
        """Adapter skips analysis for spans below threshold."""
        class MockAgent:
            def __init__(self):
                self.called = False

            def analyze(self, analysis_input, **kwargs):
                self.called = True
                return None

        agent = MockAgent()
        adapter = AnalysisAgentAdapter(agent, min_span=200)
        
        result = adapter.run_analysis({
            "document_id": "doc-1",
            "document_chars": 5000,  # Document is large enough
            "span_start": 100,
            "span_end": 150,  # Span is only 50 chars
        })
        
        assert result is None
        assert agent.called is False

    def test_run_analysis_uses_content_length_fallback(self):
        """Adapter uses content length when document_chars not provided."""
        class MockAgent:
            def __init__(self):
                self.called = False

            def analyze(self, analysis_input, **kwargs):
                self.called = True
                return None

        agent = MockAgent()
        adapter = AnalysisAgentAdapter(agent, min_chars=500)
        
        # Short content, no document_chars
        result = adapter.run_analysis({
            "document_id": "doc-1",
            "content": "Short",
        })
        
        assert result is None
        assert agent.called is False
        
        # Long content
        agent.called = False
        result = adapter.run_analysis({
            "document_id": "doc-1",
            "content": "x" * 1000,
        })
        
        assert agent.called is True

    def test_run_analysis_allows_full_document_span(self):
        """Adapter allows analysis when no span is specified (full document)."""
        @dataclass
        class MockAdvice:
            document_id: str

        class MockAgent:
            def analyze(self, analysis_input, **kwargs):
                return MockAdvice(document_id=analysis_input.document_id)

        adapter = AnalysisAgentAdapter(MockAgent())
        
        # No span specified (span_start=0, span_end=0 means full document)
        result = adapter.run_analysis({
            "document_id": "doc-1",
            "document_chars": 1000,
        })
        
        assert result is not None

    def test_run_analysis_custom_thresholds(self):
        """Adapter respects custom min_chars and min_span thresholds."""
        class MockAgent:
            def __init__(self):
                self.called = False

            def analyze(self, analysis_input, **kwargs):
                self.called = True
                return None

        # Very low thresholds
        agent = MockAgent()
        adapter = AnalysisAgentAdapter(agent, min_chars=10, min_span=5)
        
        result = adapter.run_analysis({
            "document_id": "doc-1",
            "document_chars": 50,
            "span_start": 0,
            "span_end": 10,
        })
        
        assert agent.called is True

    def test_run_analysis_maps_all_fields(self):
        """Adapter correctly maps all snapshot fields to AnalysisInput."""
        class RecordingAgent:
            def __init__(self):
                self.received_input = None

            def analyze(self, analysis_input, **kwargs):
                self.received_input = analysis_input
                return None

        agent = RecordingAgent()
        adapter = AnalysisAgentAdapter(agent)
        
        snapshot = {
            "document_id": "doc-1",
            "version_id": "v1",
            "document_version": "v2",  # Should prefer version_id
            "path": "/test/doc.txt",
            "span_start": 100,
            "span_end": 500,
            "document_chars": 10000,
            "chunk_profile_hint": "large",
            "chunk_index_ready": True,
            "outline_digest": "abc123",
            "plot_state_status": "stale",
            "concordance_status": "fresh",
            "retrieval_enabled": False,
        }
        
        adapter.run_analysis(snapshot)
        
        inp = agent.received_input
        assert inp.document_id == "doc-1"
        assert inp.document_version == "v1"  # version_id takes precedence
        assert inp.document_path == "/test/doc.txt"
        assert inp.span_start == 100
        assert inp.span_end == 500
        assert inp.document_chars == 10000
        assert inp.chunk_profile_hint == "large"
        assert inp.chunk_index_ready is True
        assert inp.outline_digest == "abc123"
        assert inp.plot_state_status == "stale"
        assert inp.concordance_status == "fresh"
        assert inp.retrieval_enabled is False


# =============================================================================
# Document Snapshot Building Tests
# =============================================================================


class TestDocumentSnapshotBuilding:
    """Tests for _build_document_snapshot."""

    def test_empty_snapshot(self, mock_client):
        """Empty snapshot creates empty DocumentSnapshot."""
        orchestrator = AIOrchestrator(client=mock_client)
        snapshot = orchestrator._build_document_snapshot(None)
        
        assert snapshot.tab_id == ""
        assert snapshot.content == ""

    def test_snapshot_with_tab_id(self, mock_client):
        """Extracts tab_id from snapshot."""
        orchestrator = AIOrchestrator(client=mock_client)
        snapshot = orchestrator._build_document_snapshot({
            "tab_id": "t1",
            "content": "Hello",
        })
        
        assert snapshot.tab_id == "t1"
        assert snapshot.content == "Hello"

    def test_snapshot_with_document_id_fallback(self, mock_client):
        """Uses document_id if tab_id not present."""
        orchestrator = AIOrchestrator(client=mock_client)
        snapshot = orchestrator._build_document_snapshot({
            "document_id": "d1",
            "text": "World",
        })
        
        assert snapshot.tab_id == "d1"
        assert snapshot.content == "World"


# =============================================================================
# History Conversion Tests
# =============================================================================


class TestHistoryConversion:
    """Tests for _convert_history."""

    def test_empty_history(self, mock_client):
        """Empty history returns empty tuple."""
        orchestrator = AIOrchestrator(client=mock_client)
        messages = orchestrator._convert_history(None)
        
        assert messages == ()

    def test_history_conversion(self, mock_client):
        """Converts history dicts to Message tuples."""
        orchestrator = AIOrchestrator(client=mock_client)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        
        messages = orchestrator._convert_history(history)
        
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there"


# =============================================================================
# Suggestions Tests
# =============================================================================


class TestSuggestions:
    """Tests for suggestion generation."""

    def test_parse_suggestions_empty(self, mock_client):
        """Empty text returns empty list."""
        orchestrator = AIOrchestrator(client=mock_client)
        suggestions = orchestrator._parse_suggestions("", 4)
        assert suggestions == []

    def test_parse_suggestions_filters_short(self, mock_client):
        """Filters out too-short suggestions."""
        orchestrator = AIOrchestrator(client=mock_client)
        text = "Hi\nThis is a valid suggestion\nNo"
        
        suggestions = orchestrator._parse_suggestions(text, 4)
        
        assert len(suggestions) == 1
        assert "This is a valid suggestion" in suggestions

    def test_parse_suggestions_limits_count(self, mock_client):
        """Respects max_suggestions limit."""
        orchestrator = AIOrchestrator(client=mock_client)
        text = "\n".join([
            "1. First suggestion here",
            "2. Second suggestion here",
            "3. Third suggestion here",
            "4. Fourth suggestion here",
            "5. Fifth suggestion here",
        ])
        
        suggestions = orchestrator._parse_suggestions(text, 2)
        
        assert len(suggestions) == 2

    @pytest.mark.asyncio
    async def test_suggest_followups_empty_history(self, mock_client):
        """Empty history returns empty suggestions."""
        orchestrator = AIOrchestrator(client=mock_client)
        suggestions = await orchestrator.suggest_followups([])
        assert suggestions == []


# =============================================================================
# Tool Definition Tests
# =============================================================================


class TestToolDefinitions:
    """Tests for _get_tool_definitions."""

    def test_empty_registry(self, mock_client):
        """Empty registry returns empty list."""
        registry = ToolRegistry()
        orchestrator = AIOrchestrator(
            client=mock_client,
            tool_registry=registry,
        )
        
        definitions = orchestrator._get_tool_definitions()
        assert definitions == []

    def test_with_registered_tools(self, mock_client):
        """Returns definitions for registered tools."""
        registry = ToolRegistry()
        registry.register(
            lambda: "result",
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "First arg"},
                },
            },
        )
        
        orchestrator = AIOrchestrator(
            client=mock_client,
            tool_registry=registry,
        )
        
        definitions = orchestrator._get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["type"] == "function"
        assert definitions[0]["function"]["name"] == "test_tool"
        assert definitions[0]["function"]["description"] == "A test tool"
        assert definitions[0]["function"]["strict"] is True


# =============================================================================
# Analysis Advice Cache Tests
# =============================================================================


class TestAnalysisAdviceCache:
    """Tests for analysis advice caching."""

    def test_get_latest_analysis_advice_empty(self, mock_client):
        """Returns None when no advice cached."""
        orchestrator = AIOrchestrator(client=mock_client)
        assert orchestrator.get_latest_analysis_advice("doc-1") is None

    def test_set_and_get_analysis_advice(self, mock_client):
        """Can set and retrieve analysis advice."""
        @dataclass
        class MockAdvice:
            document_id: str
            chunk_profile: str = "auto"

        orchestrator = AIOrchestrator(client=mock_client)
        advice = MockAdvice(document_id="doc-1")
        
        orchestrator.set_analysis_advice("doc-1", advice)
        
        result = orchestrator.get_latest_analysis_advice("doc-1")
        assert result is advice

    def test_clear_analysis_advice_single(self, mock_client):
        """Can clear advice for a single document."""
        orchestrator = AIOrchestrator(client=mock_client)
        orchestrator.set_analysis_advice("doc-1", {"id": "doc-1"})
        orchestrator.set_analysis_advice("doc-2", {"id": "doc-2"})
        
        orchestrator.clear_analysis_advice("doc-1")
        
        assert orchestrator.get_latest_analysis_advice("doc-1") is None
        assert orchestrator.get_latest_analysis_advice("doc-2") is not None

    def test_clear_analysis_advice_all(self, mock_client):
        """Can clear all cached advice."""
        orchestrator = AIOrchestrator(client=mock_client)
        orchestrator.set_analysis_advice("doc-1", {"id": "doc-1"})
        orchestrator.set_analysis_advice("doc-2", {"id": "doc-2"})
        
        orchestrator.clear_analysis_advice()
        
        assert orchestrator.get_latest_analysis_advice("doc-1") is None
        assert orchestrator.get_latest_analysis_advice("doc-2") is None

    def test_set_analysis_advice_ignores_none(self, mock_client):
        """Setting None advice is a no-op."""
        orchestrator = AIOrchestrator(client=mock_client)
        orchestrator.set_analysis_advice("doc-1", None)
        
        assert orchestrator.get_latest_analysis_advice("doc-1") is None

    def test_set_analysis_advice_ignores_empty_doc_id(self, mock_client):
        """Setting advice with empty doc ID is a no-op."""
        orchestrator = AIOrchestrator(client=mock_client)
        orchestrator.set_analysis_advice("", {"id": "test"})
        
        assert orchestrator.get_latest_analysis_advice("") is None