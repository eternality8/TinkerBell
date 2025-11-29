"""Tests for Workstream 9: Subagent Execution (LLM Integration).

Tests for:
- WS9.1: SubagentExecutor Implementation
- WS9.2: Orchestrator Wiring
- WS9.3: Analysis Task Implementation
- WS9.4: Transformation Task Implementation
- WS9.5: Response Parsing and Validation
- WS9.6: Progress and Cancellation
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# WS9 imports
from tinkerbell.ai.orchestration.subagent_executor import (
    SubagentExecutor,
    SubagentExecutorConfig,
    ResponseParseError,
    extract_json_from_response,
    validate_response_schema,
    normalize_analysis_result,
    validate_transformation_output,
    create_subagent_orchestrator,
    configure_analyze_tool_executor,
    configure_transform_tool_executor,
)
from tinkerbell.ai.orchestration.subagent_prompts import (
    ANALYSIS_SYSTEM_BASE,
    ANALYSIS_PROMPTS,
    TRANSFORM_SYSTEM_BASE,
    TRANSFORM_PROMPTS,
    ANALYSIS_SCHEMAS,
    TRANSFORM_SCHEMAS,
    get_analysis_prompt,
    get_transform_prompt,
)
from tinkerbell.ai.tools.subagent import (
    SubagentTask,
    SubagentResult,
    SubagentType,
    ChunkSpec,
    TaskPriority,
    SubagentOrchestrator,
    ProgressTracker,
)
from tinkerbell.ai.client import AIClient, AIStreamEvent, ClientSettings


# =============================================================================
# Test Fixtures
# =============================================================================


class MockStreamIterator:
    """Async iterator for mock streaming that doesn't produce warnings.
    
    This avoids the 'coroutine aclose was never awaited' warning by
    implementing a proper async iterator instead of using an async generator.
    """
    
    def __init__(self, events: list[AIStreamEvent], delay: float = 0.0):
        self._events = events
        self._index = 0
        self._delay = delay
    
    def __aiter__(self) -> "MockStreamIterator":
        return self
    
    async def __anext__(self) -> AIStreamEvent:
        if self._delay > 0 and self._index == 0:
            await asyncio.sleep(self._delay)
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


@dataclass
class MockAIClient:
    """Mock AI client for testing."""
    
    responses: list[str] = field(default_factory=list)
    call_count: int = 0
    should_fail: bool = False
    fail_message: str = "Mock failure"
    stream_delay: float = 0.0
    
    def stream_chat(
        self,
        messages: Any,
        **kwargs: Any,
    ) -> AsyncIterator[AIStreamEvent]:
        """Return an async iterator for streaming.
        
        Note: This is NOT an async method. It returns an async iterator directly.
        This matches the real AIClient behavior where stream_chat returns an
        async generator (which is an async iterator).
        """
        self.call_count += 1
        
        if self.should_fail:
            raise RuntimeError(self.fail_message)
        
        # Get response for this call
        response_idx = min(self.call_count - 1, len(self.responses) - 1)
        response = self.responses[response_idx] if self.responses else '{"result": "ok"}'
        
        # Return a MockStreamIterator instead of using async generator
        # This avoids 'coroutine aclose was never awaited' warnings
        events = [
            AIStreamEvent(type="content.delta", content=response[:len(response)//2]),
            AIStreamEvent(type="content.delta", content=response[len(response)//2:]),
            AIStreamEvent(type="content.done", content=response),
        ]
        return MockStreamIterator(events, delay=self.stream_delay)
    
    def count_tokens(self, text: str, *, estimate_only: bool = False) -> int:
        return len(text) // 4


@pytest.fixture
def mock_client() -> MockAIClient:
    """Create a mock AI client."""
    return MockAIClient()


@pytest.fixture
def executor(mock_client: MockAIClient) -> SubagentExecutor:
    """Create a SubagentExecutor with mock client."""
    return SubagentExecutor(mock_client)  # type: ignore


@pytest.fixture
def sample_chunk() -> ChunkSpec:
    """Create a sample chunk for testing."""
    content = """Alice walked through the garden, admiring the roses.
She met the Cheshire Cat, who smiled mysteriously.
"Where are you going?" asked the Cat.
"I don't know," Alice replied."""
    return ChunkSpec(
        chunk_id="test-chunk-0",
        document_id="test-doc",
        content=content,
        start_char=0,
        end_char=len(content),
        start_line=0,
        end_line=3,
        token_estimate=50,
    )


@pytest.fixture
def analysis_task(sample_chunk: ChunkSpec) -> SubagentTask:
    """Create a sample analysis task."""
    return SubagentTask(
        task_id="analysis-task-1",
        subagent_type=SubagentType.CHUNK_ANALYZER,
        chunk=sample_chunk,
        instructions="Analyze this text",
        metadata={"analysis_type": "characters"},
    )


@pytest.fixture
def transform_task(sample_chunk: ChunkSpec) -> SubagentTask:
    """Create a sample transformation task."""
    return SubagentTask(
        task_id="transform-task-1",
        subagent_type=SubagentType.TRANSFORMER,
        chunk=sample_chunk,
        instructions="Transform this text",
        metadata={
            "transformation_type": "character_rename",
            "params": {"old_name": "Alice", "new_name": "Dorothy"},
        },
    )


# =============================================================================
# WS9.5: Response Parsing Tests
# =============================================================================


class TestExtractJsonFromResponse:
    """Tests for JSON extraction from LLM responses."""
    
    def test_pure_json(self):
        """Parse pure JSON response."""
        response = '{"characters": [{"name": "Alice"}]}'
        result = extract_json_from_response(response)
        assert result == {"characters": [{"name": "Alice"}]}
    
    def test_json_with_whitespace(self):
        """Parse JSON with surrounding whitespace."""
        response = '  \n{"result": "ok"}\n  '
        result = extract_json_from_response(response)
        assert result == {"result": "ok"}
    
    def test_markdown_json_block(self):
        """Parse JSON from markdown code block."""
        response = '''Here is the analysis:
```json
{"characters": ["Alice", "Cat"]}
```
That's the result.'''
        result = extract_json_from_response(response)
        assert result == {"characters": ["Alice", "Cat"]}
    
    def test_markdown_plain_block(self):
        """Parse JSON from plain markdown block."""
        response = '''```
{"summary": "A story about Alice"}
```'''
        result = extract_json_from_response(response)
        assert result == {"summary": "A story about Alice"}
    
    def test_json_in_text(self):
        """Extract JSON embedded in text."""
        response = 'The analysis is: {"themes": ["adventure"]} as shown above.'
        result = extract_json_from_response(response)
        assert result == {"themes": ["adventure"]}
    
    def test_empty_response_raises(self):
        """Empty response raises error."""
        with pytest.raises(ResponseParseError):
            extract_json_from_response("")
    
    def test_invalid_json_raises(self):
        """Invalid JSON raises error."""
        with pytest.raises(ResponseParseError):
            extract_json_from_response("This is not JSON at all")
    
    def test_nested_json(self):
        """Parse nested JSON structure."""
        response = '{"style": {"voice": {"person": "third", "tense": "past"}}}'
        result = extract_json_from_response(response)
        assert result["style"]["voice"]["person"] == "third"


class TestValidateResponseSchema:
    """Tests for schema validation."""
    
    def test_valid_characters_schema(self):
        """Valid characters response passes validation."""
        data = {
            "characters": [
                {"name": "Alice", "role": "protagonist"}
            ]
        }
        schema = ANALYSIS_SCHEMAS["characters"]
        is_valid, errors = validate_response_schema(data, schema)
        assert is_valid
        assert len(errors) == 0
    
    def test_missing_required_field(self):
        """Missing required field is detected."""
        data = {"themes": ["adventure"]}  # Missing 'characters' field
        schema = ANALYSIS_SCHEMAS["characters"]
        is_valid, errors = validate_response_schema(data, schema)
        assert not is_valid
        assert any("characters" in e for e in errors)
    
    def test_wrong_type(self):
        """Wrong type is detected."""
        data = {"characters": "not an array"}
        schema = ANALYSIS_SCHEMAS["characters"]
        is_valid, errors = validate_response_schema(data, schema)
        assert not is_valid
        assert any("array" in e for e in errors)
    
    def test_valid_summary_schema(self):
        """Valid summary response passes validation."""
        data = {"summary": "Alice goes on an adventure."}
        schema = ANALYSIS_SCHEMAS["summary"]
        is_valid, errors = validate_response_schema(data, schema)
        assert is_valid


class TestNormalizeAnalysisResult:
    """Tests for result normalization."""
    
    def test_normalize_characters_adds_defaults(self):
        """Characters normalization adds missing fields."""
        data = {"characters": [{"name": "Alice"}]}
        result = normalize_analysis_result(data, "characters")
        char = result["characters"][0]
        assert char["aliases"] == []
        assert char["role"] == "mentioned"
        assert char["traits"] == []
        assert char["relationships"] == []
        assert char["mentions"] == []
    
    def test_normalize_plot_adds_defaults(self):
        """Plot normalization adds missing fields."""
        data = {"plot_points": [{"summary": "Something happens"}]}
        result = normalize_analysis_result(data, "plot")
        assert result["tension_level"] == "unknown"
        assert result["setting"] is None
    
    def test_normalize_style_adds_defaults(self):
        """Style normalization adds missing fields."""
        data = {"style": {"tone": ["dark"]}}
        result = normalize_analysis_result(data, "style")
        assert "voice" in result["style"]
        assert "pacing" in result["style"]
    
    def test_normalize_preserves_existing(self):
        """Normalization preserves existing values."""
        data = {
            "characters": [
                {"name": "Alice", "role": "protagonist", "traits": ["curious"]}
            ]
        }
        result = normalize_analysis_result(data, "characters")
        char = result["characters"][0]
        assert char["role"] == "protagonist"
        assert char["traits"] == ["curious"]


class TestValidateTransformationOutput:
    """Tests for transformation output validation."""
    
    def test_valid_transformation_no_warnings(self):
        """Valid transformation produces no warnings."""
        original = "Alice walked through the garden."
        transformed = "Dorothy walked through the garden."
        warnings = validate_transformation_output(original, transformed)
        assert len(warnings) == 0
    
    def test_empty_output_warns(self):
        """Empty output produces warning."""
        original = "Some text"
        transformed = ""
        warnings = validate_transformation_output(original, transformed)
        assert any("empty" in w.lower() for w in warnings)
    
    def test_significant_reduction_warns(self):
        """Significant length reduction produces warning."""
        original = "A " * 100
        transformed = "B"
        warnings = validate_transformation_output(original, transformed)
        assert any("reduction" in w.lower() for w in warnings)
    
    def test_significant_expansion_warns(self):
        """Significant length expansion produces warning."""
        original = "Short."
        transformed = "Very " * 100
        warnings = validate_transformation_output(original, transformed)
        assert any("expansion" in w.lower() for w in warnings)
    
    def test_quote_imbalance_warns(self):
        """Quote imbalance produces warning."""
        original = '"Hello," said Alice. "How are you?"'
        transformed = 'Hello, said Alice.'  # Lost quotes
        warnings = validate_transformation_output(original, transformed)
        assert any("quote" in w.lower() for w in warnings)
    
    def test_truncated_output_warns(self):
        """Truncated output produces warning."""
        original = "A complete sentence."
        transformed = "An incomplete sentence,"
        warnings = validate_transformation_output(original, transformed)
        assert any("truncated" in w.lower() for w in warnings)


# =============================================================================
# WS9.3: Analysis Prompt Tests
# =============================================================================


class TestAnalysisPrompts:
    """Tests for analysis prompt generation."""
    
    def test_get_characters_prompt(self):
        """Characters prompt includes required elements."""
        prompt = get_analysis_prompt("characters")
        assert "character" in prompt.lower()
        assert "json" in prompt.lower()
        assert "name" in prompt.lower()
    
    def test_get_plot_prompt(self):
        """Plot prompt includes required elements."""
        prompt = get_analysis_prompt("plot")
        assert "plot" in prompt.lower()
        assert "tension" in prompt.lower()
    
    def test_get_style_prompt(self):
        """Style prompt includes required elements."""
        prompt = get_analysis_prompt("style")
        assert "style" in prompt.lower()
        assert "tone" in prompt.lower()
        assert "voice" in prompt.lower()
    
    def test_get_summary_prompt(self):
        """Summary prompt includes required elements."""
        prompt = get_analysis_prompt("summary")
        assert "summar" in prompt.lower()
    
    def test_get_themes_prompt(self):
        """Themes prompt includes required elements."""
        prompt = get_analysis_prompt("themes")
        assert "theme" in prompt.lower()
        assert "motif" in prompt.lower()
    
    def test_get_custom_prompt(self):
        """Custom prompt uses provided text."""
        custom = "Find all the colors mentioned in this text."
        prompt = get_analysis_prompt("custom", custom)
        assert custom in prompt
    
    def test_invalid_type_raises(self):
        """Invalid analysis type raises error."""
        with pytest.raises(ValueError):
            get_analysis_prompt("invalid_type")
    
    def test_prompts_include_json_instruction(self):
        """All prompts include JSON instruction."""
        for analysis_type in ["characters", "plot", "style", "summary", "themes"]:
            prompt = get_analysis_prompt(analysis_type)
            assert "json" in prompt.lower()


# =============================================================================
# WS9.4: Transformation Prompt Tests
# =============================================================================


class TestTransformationPrompts:
    """Tests for transformation prompt generation."""
    
    def test_style_rewrite_prompt(self):
        """Style rewrite prompt includes parameters."""
        params = {"target_style": "noir", "style_guidelines": "Use short sentences"}
        prompt = get_transform_prompt("style_rewrite", params)
        assert "noir" in prompt
        assert "short sentences" in prompt.lower()
    
    def test_setting_change_prompt(self):
        """Setting change prompt includes parameters."""
        params = {"old_setting": "London", "new_setting": "Tokyo"}
        prompt = get_transform_prompt("setting_change", params)
        assert "London" in prompt
        assert "Tokyo" in prompt
    
    def test_tense_change_prompt(self):
        """Tense change prompt includes parameters."""
        params = {"from_tense": "past", "to_tense": "present"}
        prompt = get_transform_prompt("tense_change", params)
        assert "past" in prompt
        assert "present" in prompt
    
    def test_pov_change_prompt(self):
        """POV change prompt includes parameters."""
        params = {"from_pov": "third", "to_pov": "first", "focal_character": "Alice"}
        prompt = get_transform_prompt("pov_change", params)
        assert "third" in prompt
        assert "first" in prompt
        assert "Alice" in prompt
    
    def test_tense_change_prompt_with_target(self):
        """Tense change prompt works with target_tense only."""
        params = {"target_tense": "present"}
        prompt = get_transform_prompt("tense_change", params)
        # Should infer from_tense as "past" when target is "present"
        assert "past" in prompt
        assert "present" in prompt
    
    def test_pov_change_prompt_with_target(self):
        """POV change prompt works with target_pov only."""
        params = {"target_pov": "first"}
        prompt = get_transform_prompt("pov_change", params)
        # Should use "current" as placeholder for from_pov
        assert "current" in prompt
        assert "first" in prompt

    def test_character_rename_prompt(self):
        """Character rename prompt includes parameters."""
        params = {"old_name": "Alice", "new_name": "Dorothy", "aliases": ["girl"]}
        prompt = get_transform_prompt("character_rename", params)
        assert "Alice" in prompt
        assert "Dorothy" in prompt
        assert "girl" in prompt
    
    def test_custom_transform_prompt(self):
        """Custom transform prompt uses provided text."""
        params = {"custom_prompt": "Make all dialogue formal."}
        prompt = get_transform_prompt("custom", params)
        assert "formal" in prompt.lower()
    
    def test_invalid_type_raises(self):
        """Invalid transformation type raises error."""
        with pytest.raises(ValueError):
            get_transform_prompt("invalid_type", {})


# =============================================================================
# WS9.1: SubagentExecutor Tests
# =============================================================================


class TestSubagentExecutorConfig:
    """Tests for executor configuration."""
    
    def test_default_config(self):
        """Default config has sensible values."""
        config = SubagentExecutorConfig()
        assert config.temperature == 0.3
        assert config.max_tokens == 4000
        assert config.max_retries == 3
        assert config.timeout_seconds == 120.0
    
    def test_custom_config(self):
        """Custom config values are used."""
        config = SubagentExecutorConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000,
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000


class TestSubagentExecutor:
    """Tests for SubagentExecutor."""
    
    @pytest.mark.asyncio
    async def test_execute_analysis_success(
        self,
        executor: SubagentExecutor,
        mock_client: MockAIClient,
        analysis_task: SubagentTask,
    ):
        """Successful analysis returns parsed result."""
        mock_client.responses = ['{"characters": [{"name": "Alice", "role": "protagonist"}]}']
        
        result = await executor.execute(analysis_task)
        
        assert result.success
        assert result.task_id == analysis_task.task_id
        assert "characters" in result.output
        assert result.output["characters"][0]["name"] == "Alice"
    
    @pytest.mark.asyncio
    async def test_execute_transformation_success(
        self,
        executor: SubagentExecutor,
        mock_client: MockAIClient,
        transform_task: SubagentTask,
    ):
        """Successful transformation returns content."""
        mock_client.responses = ['{"transformed_content": "Dorothy walked through the garden."}']
        
        result = await executor.execute(transform_task)
        
        assert result.success
        assert "transformed_content" in result.output
        assert "Dorothy" in result.output["transformed_content"]
    
    @pytest.mark.asyncio
    async def test_execute_handles_parse_error(
        self,
        executor: SubagentExecutor,
        mock_client: MockAIClient,
        analysis_task: SubagentTask,
    ):
        """Parse errors are handled gracefully."""
        mock_client.responses = ["This is not valid JSON at all"]
        
        result = await executor.execute(analysis_task)
        
        assert not result.success
        assert "parse" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_handles_llm_failure(
        self,
        executor: SubagentExecutor,
        mock_client: MockAIClient,
        analysis_task: SubagentTask,
    ):
        """LLM failures are handled gracefully."""
        mock_client.should_fail = True
        mock_client.fail_message = "API error"
        
        # Configure for quick retries
        executor._config.max_retries = 1
        executor._config.retry_delay_base = 0.01
        
        result = await executor.execute(analysis_task)
        
        assert not result.success
        assert "API error" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_records_latency(
        self,
        executor: SubagentExecutor,
        mock_client: MockAIClient,
        analysis_task: SubagentTask,
    ):
        """Execution latency is recorded."""
        mock_client.responses = ['{"characters": []}']
        mock_client.stream_delay = 0.05
        
        result = await executor.execute(analysis_task)
        
        assert result.latency_ms > 0
    
    @pytest.mark.asyncio
    async def test_execute_estimates_tokens(
        self,
        executor: SubagentExecutor,
        mock_client: MockAIClient,
        analysis_task: SubagentTask,
    ):
        """Token usage is estimated."""
        mock_client.responses = ['{"characters": []}']
        
        result = await executor.execute(analysis_task)
        
        assert result.tokens_used > 0
    
    @pytest.mark.asyncio
    async def test_execute_transformation_missing_content(
        self,
        executor: SubagentExecutor,
        mock_client: MockAIClient,
        transform_task: SubagentTask,
    ):
        """Missing transformed_content causes failure."""
        mock_client.responses = ['{"status": "done"}']  # No transformed_content
        
        result = await executor.execute(transform_task)
        
        assert not result.success
        assert "transformed_content" in result.error


class TestSubagentExecutorRetries:
    """Tests for retry behavior."""
    
    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self):
        """Executor retries on transient failures."""
        client = MockAIClient()
        call_count = 0
        
        # Use MockStreamIterator and a non-async method to avoid warnings
        def patched_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient error")
            # Return MockStreamIterator instead of async generator
            return MockStreamIterator([
                AIStreamEvent(type="content.done", content='{"result": "ok"}')
            ])
        
        client.stream_chat = patched_stream  # type: ignore
        
        executor = SubagentExecutor(
            client,  # type: ignore
            SubagentExecutorConfig(
                max_retries=3,
                retry_delay_base=0.01,
            ),
        )
        
        task = SubagentTask(
            task_id="test",
            subagent_type=SubagentType.CUSTOM,
            chunk=ChunkSpec(chunk_id="c", document_id="d", content="test"),
            instructions="Test",
        )
        
        result = await executor.execute(task)
        
        assert result.success
        assert call_count == 3


# =============================================================================
# WS9.2: Orchestrator Wiring Tests
# =============================================================================


class TestCreateSubagentOrchestrator:
    """Tests for orchestrator factory."""
    
    def test_creates_orchestrator_with_executor(self, mock_client: MockAIClient):
        """Factory creates orchestrator with executor."""
        orchestrator = create_subagent_orchestrator(mock_client)  # type: ignore
        
        assert orchestrator is not None
        assert orchestrator._executor is not None
    
    def test_respects_max_parallel(self, mock_client: MockAIClient):
        """Factory respects max_parallel parameter."""
        orchestrator = create_subagent_orchestrator(
            mock_client,  # type: ignore
            max_parallel=8,
        )
        
        assert orchestrator.max_parallel == 8
    
    def test_respects_config(self, mock_client: MockAIClient):
        """Factory respects executor config."""
        config = SubagentExecutorConfig(temperature=0.7)
        orchestrator = create_subagent_orchestrator(
            mock_client,  # type: ignore
            config=config,
        )
        
        assert orchestrator._executor._config.temperature == 0.7
    
    def test_attaches_progress_tracker(self, mock_client: MockAIClient):
        """Factory attaches progress tracker."""
        tracker = ProgressTracker()
        orchestrator = create_subagent_orchestrator(
            mock_client,  # type: ignore
            progress_tracker=tracker,
        )
        
        assert orchestrator.progress_tracker is tracker


class TestToolIntegration:
    """Tests for tool integration helpers."""
    
    def test_configure_analyze_tool(self, mock_client: MockAIClient):
        """Configure analyze tool with orchestrator."""
        from tinkerbell.ai.tools.analyze_document import AnalyzeDocumentTool
        
        tool = AnalyzeDocumentTool()
        orchestrator = create_subagent_orchestrator(mock_client)  # type: ignore
        
        configure_analyze_tool_executor(tool, orchestrator)
        
        assert tool.orchestrator is orchestrator
    
    def test_configure_transform_tool(self, mock_client: MockAIClient):
        """Configure transform tool with orchestrator."""
        from tinkerbell.ai.tools.transform_document import TransformDocumentTool
        
        tool = TransformDocumentTool()
        orchestrator = create_subagent_orchestrator(mock_client)  # type: ignore
        
        configure_transform_tool_executor(tool, orchestrator)
        
        assert tool.orchestrator is orchestrator


# =============================================================================
# WS9.6: Progress and Cancellation Tests
# =============================================================================


class TestProgressTracking:
    """Tests for progress tracking during execution."""
    
    @pytest.mark.asyncio
    async def test_progress_updates_during_execution(self, mock_client: MockAIClient):
        """Progress tracker receives updates."""
        mock_client.responses = ['{"result": "ok"}'] * 3
        
        tracker = ProgressTracker()
        updates: list[tuple[int, int]] = []
        
        class TestListener:
            def on_progress(self, completed: int, total: int, current: Any) -> None:
                updates.append((completed, total))
            def on_task_started(self, task: SubagentTask) -> None:
                pass
            def on_task_completed(self, task: SubagentTask, result: SubagentResult) -> None:
                pass
        
        tracker.listeners.append(TestListener())  # type: ignore
        
        orchestrator = create_subagent_orchestrator(
            mock_client,  # type: ignore
            progress_tracker=tracker,
        )
        
        tasks = [
            SubagentTask(
                task_id=f"task-{i}",
                subagent_type=SubagentType.CUSTOM,
                chunk=ChunkSpec(chunk_id=f"c{i}", document_id="d", content="test"),
                instructions="Test",
            )
            for i in range(3)
        ]
        
        await orchestrator.run_tasks(tasks, parallel=False)
        
        assert len(updates) == 3
        assert updates[-1] == (3, 3)


class TestCancellation:
    """Tests for task cancellation."""
    
    @pytest.mark.asyncio
    async def test_cancelled_task_returns_error(self):
        """Cancelled task returns appropriate error."""
        client = MockAIClient()
        client.stream_delay = 1.0  # Long delay to allow cancellation
        
        executor = SubagentExecutor(client)  # type: ignore
        
        task = SubagentTask(
            task_id="test",
            subagent_type=SubagentType.CUSTOM,
            chunk=ChunkSpec(chunk_id="c", document_id="d", content="test"),
            instructions="Test",
        )
        
        # Start task and cancel
        coro = executor.execute(task)
        task_future = asyncio.ensure_future(coro)
        
        await asyncio.sleep(0.01)
        task_future.cancel()
        
        try:
            result = await task_future
            assert not result.success
            assert "cancel" in result.error.lower()
        except asyncio.CancelledError:
            pass  # Expected


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndAnalysis:
    """End-to-end tests for analysis workflow."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, mock_client: MockAIClient):
        """Complete analysis workflow executes successfully."""
        mock_client.responses = [
            json.dumps({
                "characters": [
                    {"name": "Alice", "role": "protagonist", "traits": ["curious"]},
                    {"name": "Cheshire Cat", "role": "supporting", "traits": ["mysterious"]}
                ]
            })
        ]
        
        orchestrator = create_subagent_orchestrator(mock_client)  # type: ignore
        
        chunk = ChunkSpec(
            chunk_id="analysis-chunk",
            document_id="test-doc",
            content="Alice met the Cheshire Cat in the garden.",
            token_estimate=20,
        )
        
        task = SubagentTask(
            task_id="analysis-1",
            subagent_type=SubagentType.CHUNK_ANALYZER,
            chunk=chunk,
            instructions="Analyze characters",
            metadata={"analysis_type": "characters"},
        )
        
        results = await orchestrator.run_tasks([task])
        
        assert len(results) == 1
        assert results[0].success
        assert len(results[0].output["characters"]) == 2


class TestEndToEndTransformation:
    """End-to-end tests for transformation workflow."""
    
    @pytest.mark.asyncio
    async def test_full_transformation_workflow(self, mock_client: MockAIClient):
        """Complete transformation workflow executes successfully."""
        mock_client.responses = [
            json.dumps({
                "transformed_content": "Dorothy walked through the garden.",
                "replacements": 1,
            })
        ]
        
        orchestrator = create_subagent_orchestrator(mock_client)  # type: ignore
        
        chunk = ChunkSpec(
            chunk_id="transform-chunk",
            document_id="test-doc",
            content="Alice walked through the garden.",
            token_estimate=10,
        )
        
        task = SubagentTask(
            task_id="transform-1",
            subagent_type=SubagentType.TRANSFORMER,
            chunk=chunk,
            instructions="Rename character",
            metadata={
                "transformation_type": "character_rename",
                "params": {"old_name": "Alice", "new_name": "Dorothy"},
            },
        )
        
        results = await orchestrator.run_tasks([task])
        
        assert len(results) == 1
        assert results[0].success
        assert "Dorothy" in results[0].output["transformed_content"]


class TestSchemaDefinitions:
    """Tests for schema definitions."""
    
    def test_all_analysis_types_have_schemas(self):
        """All analysis types have corresponding schemas."""
        for analysis_type in ["characters", "plot", "style", "summary", "themes"]:
            assert analysis_type in ANALYSIS_SCHEMAS
            assert "type" in ANALYSIS_SCHEMAS[analysis_type]
    
    def test_all_transform_types_have_schemas(self):
        """All transformation types have corresponding schemas."""
        for transform_type in ["style_rewrite", "setting_change", "tense_change", "pov_change", "character_rename", "custom"]:
            assert transform_type in TRANSFORM_SCHEMAS
            assert "type" in TRANSFORM_SCHEMAS[transform_type]
    
    def test_transform_schemas_require_content(self):
        """All transform schemas require transformed_content."""
        for transform_type, schema in TRANSFORM_SCHEMAS.items():
            required = schema.get("required", [])
            assert "transformed_content" in required, f"{transform_type} should require transformed_content"
