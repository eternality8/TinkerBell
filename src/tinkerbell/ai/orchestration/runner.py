"""Turn Runner: Orchestrates the complete turn pipeline.

This module provides the TurnRunner class that wires together all pipeline
stages (prepare → analyze → execute → tools → finish) into a complete turn
execution flow with support for the tool loop.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .types import (
    AnalyzedTurn,
    Message,
    ModelResponse,
    PreparedTurn,
    ToolCallRecord,
    TurnConfig,
    TurnInput,
    TurnMetrics,
    TurnOutput,
    DocumentSnapshot,
    ParsedToolCall,
)
from .pipeline.prepare import (
    prepare_turn,
    build_messages,
    TokenCounter,
)
from .pipeline.analyze import (
    analyze_turn,
    AnalysisProvider,
)
from .pipeline.execute import (
    execute_model,
    ModelClient,
    StreamEvent,
)
from .pipeline.tools import (
    execute_tools,
    append_tool_results,
    ToolExecutor,
    ToolResults,
)
from .pipeline.finish import (
    collect_metrics,
    finish_turn,
    finish_turn_with_error,
    aggregate_tool_records,
    TurnTimer,
)
from .services import Services

__all__ = [
    "TurnRunner",
    "RunnerConfig",
    "ContentCallback",
    "ToolCallback",
]

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Callback Types
# -----------------------------------------------------------------------------

# Callback invoked when streaming content is received
ContentCallback = Callable[[str], None]

# Callback invoked when a tool call is about to be executed
ToolCallback = Callable[[str, str, Mapping[str, Any]], None]


# -----------------------------------------------------------------------------
# Runner Configuration
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RunnerConfig:
    """Configuration for the turn runner.

    Attributes:
        max_iterations: Maximum tool loop iterations (overrides TurnConfig).
        streaming_enabled: Whether to stream content.
        tool_timeout: Default timeout for tool execution in seconds.
        log_pipeline_stages: Whether to log each pipeline stage.
        raise_on_tool_error: If True, fail turn on tool errors; otherwise continue.
        allow_empty_prompt: Whether to allow empty user prompts.
    """

    max_iterations: int | None = None
    streaming_enabled: bool | None = None
    tool_timeout: float = 30.0
    log_pipeline_stages: bool = True
    raise_on_tool_error: bool = False
    allow_empty_prompt: bool = False


# -----------------------------------------------------------------------------
# Turn Runner
# -----------------------------------------------------------------------------


class TurnRunner:
    """Orchestrates the complete turn pipeline.

    The TurnRunner wires together all pipeline stages and manages the tool loop,
    iterating until the model produces a final response without tool calls or
    the maximum iteration count is reached.

    Example:
        >>> runner = TurnRunner(
        ...     client=ai_client,
        ...     tool_executor=executor,
        ... )
        >>> output = await runner.run(turn_input)
        >>> print(output.response)
    """

    def __init__(
        self,
        client: ModelClient,
        tool_executor: ToolExecutor,
        *,
        config: RunnerConfig | None = None,
        token_counter: TokenCounter | None = None,
        analysis_provider: AnalysisProvider | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        services: Services | None = None,
    ) -> None:
        """Initialize the turn runner.

        Args:
            client: Model client for executing chat completions.
            tool_executor: Executor for running tool calls.
            config: Optional runner configuration.
            token_counter: Optional token counter for budget estimation.
            analysis_provider: Optional provider for preflight analysis.
            tools: Optional list of tool definitions for the model.
            services: Optional services container for caching, telemetry, etc.
        """
        self._client = client
        self._tool_executor = tool_executor
        self._config = config or RunnerConfig()
        self._token_counter = token_counter
        self._analysis_provider = analysis_provider
        self._tools = tuple(tools) if tools else ()
        self._services = services

    @property
    def client(self) -> ModelClient:
        """The model client."""
        return self._client

    @property
    def tool_executor(self) -> ToolExecutor:
        """The tool executor."""
        return self._tool_executor

    @property
    def config(self) -> RunnerConfig:
        """The runner configuration."""
        return self._config

    @property
    def tools(self) -> tuple[Mapping[str, Any], ...]:
        """The tool definitions."""
        return self._tools

    @property
    def services(self) -> Services | None:
        """The services container."""
        return self._services

    def with_tools(
        self,
        tools: Sequence[Mapping[str, Any]],
    ) -> TurnRunner:
        """Return a new runner with the specified tools.

        Args:
            tools: Tool definitions to use.

        Returns:
            A new TurnRunner with the updated tools.
        """
        return TurnRunner(
            client=self._client,
            tool_executor=self._tool_executor,
            config=self._config,
            token_counter=self._token_counter,
            analysis_provider=self._analysis_provider,
            tools=tools,
            services=self._services,
        )

    def with_config(self, config: RunnerConfig) -> TurnRunner:
        """Return a new runner with the specified config.

        Args:
            config: Runner configuration to use.

        Returns:
            A new TurnRunner with the updated configuration.
        """
        return TurnRunner(
            client=self._client,
            tool_executor=self._tool_executor,
            config=config,
            token_counter=self._token_counter,
            analysis_provider=self._analysis_provider,
            tools=self._tools,
            services=self._services,
        )

    def with_services(self, services: Services) -> TurnRunner:
        """Return a new runner with the specified services.

        Args:
            services: Services container to use.

        Returns:
            A new TurnRunner with the updated services.
        """
        return TurnRunner(
            client=self._client,
            tool_executor=self._tool_executor,
            config=self._config,
            token_counter=self._token_counter,
            analysis_provider=self._analysis_provider,
            tools=self._tools,
            services=services,
        )

    async def run(
        self,
        input: TurnInput,
        *,
        content_callback: ContentCallback | None = None,
        tool_callback: ToolCallback | None = None,
    ) -> TurnOutput:
        """Execute a complete turn through the pipeline.

        This method orchestrates all pipeline stages:
        1. Prepare: Build messages and estimate budget
        2. Analyze: Run preflight analysis (if enabled)
        3. Execute: Call the model with prepared messages
        4. Tools: Execute tool calls and loop if needed
        5. Finish: Collect metrics and assemble output

        Args:
            input: The turn input containing prompt, snapshot, history, and config.
            content_callback: Optional callback for streaming content.
            tool_callback: Optional callback before each tool execution.

        Returns:
            TurnOutput containing the response, tool calls, metrics, and status.
        """
        timer = TurnTimer()
        timer.start()
        run_id = input.run_id or str(uuid.uuid4())

        # Resolve configuration
        turn_config = self._resolve_config(input.config)
        max_iterations = turn_config.max_iterations

        if self._config.log_pipeline_stages:
            LOGGER.debug(
                "Starting turn %s with max_iterations=%d",
                run_id,
                max_iterations,
            )

        try:
            return await self._run_pipeline(
                input=input,
                turn_config=turn_config,
                timer=timer,
                run_id=run_id,
                max_iterations=max_iterations,
                content_callback=content_callback,
                tool_callback=tool_callback,
            )
        except Exception as e:
            LOGGER.exception("Turn %s failed with exception", run_id)
            timer.stop()
            metrics = collect_metrics(
                duration_ms=timer.duration_ms,
                started_at=timer.started_at,
                finished_at=timer.finished_at,
                model_name=turn_config.model_name,
            )
            return finish_turn_with_error(
                error=str(e),
                metrics=metrics,
            )

    async def _run_pipeline(
        self,
        *,
        input: TurnInput,
        turn_config: TurnConfig,
        timer: TurnTimer,
        run_id: str,
        max_iterations: int,
        content_callback: ContentCallback | None,
        tool_callback: ToolCallback | None,
    ) -> TurnOutput:
        """Run the complete pipeline with tool loop.

        This is the main pipeline execution method, separated for testability.
        """
        # Stage 1: Prepare
        prepared = self._stage_prepare(input, turn_config)

        # Stage 2: Analyze
        analyzed = self._stage_analyze(prepared, input, turn_config)

        # Tool loop variables
        all_tool_records: list[ToolCallRecord] = []
        iteration = 0
        current_messages: tuple[Message, ...] | None = None  # None for first iteration
        total_prompt_tokens = prepared.budget.prompt_tokens
        total_completion_tokens = 0

        # Tool loop: iterate until no tool calls or max iterations
        while iteration < max_iterations:
            iteration += 1

            if self._config.log_pipeline_stages:
                LOGGER.debug("Turn %s iteration %d", run_id, iteration)

            # Stage 3: Execute model
            response = await self._stage_execute(
                analyzed=analyzed,
                turn_config=turn_config,
                content_callback=content_callback,
                override_messages=current_messages,
            )

            # Accumulate tokens
            total_completion_tokens += response.completion_tokens

            # Check if we have tool calls
            if not response.tool_calls:
                # No tool calls - we're done
                return self._finish_success(
                    response=response,
                    tool_records=tuple(all_tool_records),
                    timer=timer,
                    iteration_count=iteration,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    turn_config=turn_config,
                    analyzed=analyzed,
                )

            # Stage 4: Execute tools
            tool_results = await self._stage_tools(
                response=response,
                tool_callback=tool_callback,
            )

            # Convert to records and accumulate
            iteration_records = tool_results.to_records(response.tool_calls)
            all_tool_records.extend(iteration_records)

            # Append tool results to messages for next iteration
            # Use current_messages if we have them, otherwise start from analyzed messages
            base_messages = current_messages or analyzed.messages_with_hints()
            current_messages = append_tool_results(
                base_messages,
                response,
                tool_results,
            )

        # Max iterations reached
        LOGGER.warning(
            "Turn %s reached max iterations (%d)",
            run_id,
            max_iterations,
        )

        # Return the last response with all accumulated tool calls
        timer.stop()
        metrics = collect_metrics(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            duration_ms=timer.duration_ms,
            tool_call_count=len(all_tool_records),
            iteration_count=iteration,
            model_name=turn_config.model_name,
            analysis_ran=analyzed.analysis_ran,
            started_at=timer.started_at,
            finished_at=timer.finished_at,
        )
        return finish_turn(
            response=response,
            metrics=metrics,
            tool_records=tuple(all_tool_records),
            metadata={"max_iterations_reached": True},
        )

    def _resolve_config(self, turn_config: TurnConfig) -> TurnConfig:
        """Resolve configuration from runner config and turn config.

        Runner config takes precedence where specified.
        """
        updates: dict[str, Any] = {}

        if self._config.max_iterations is not None:
            updates["max_iterations"] = self._config.max_iterations

        if self._config.streaming_enabled is not None:
            updates["streaming_enabled"] = self._config.streaming_enabled

        if updates:
            return turn_config.with_updates(**updates)
        return turn_config

    def _stage_prepare(
        self,
        input: TurnInput,
        turn_config: TurnConfig,
    ) -> PreparedTurn:
        """Stage 1: Prepare messages and estimate budget."""
        if self._config.log_pipeline_stages:
            LOGGER.debug("Stage: Prepare")

        # Validate prompt
        if not input.prompt and not self._config.allow_empty_prompt:
            raise ValueError("Empty prompt not allowed")

        # Merge runner config into turn config for prepare
        effective_input = input
        if input.config != turn_config:
            # Create new input with updated config
            effective_input = TurnInput(
                prompt=input.prompt,
                snapshot=input.snapshot,
                history=input.history,
                config=turn_config,
                run_id=input.run_id,
                document_id=input.document_id,
            )

        return prepare_turn(
            turn_input=effective_input,
            token_counter=self._token_counter,
        )

    def _stage_analyze(
        self,
        prepared: PreparedTurn,
        input: TurnInput,
        turn_config: TurnConfig,
    ) -> AnalyzedTurn:
        """Stage 2: Run preflight analysis if enabled."""
        if self._config.log_pipeline_stages:
            LOGGER.debug("Stage: Analyze")

        return analyze_turn(
            prepared=prepared,
            snapshot=input.snapshot,
            config=turn_config,
            analysis_provider=self._analysis_provider,
        )

    async def _stage_execute(
        self,
        analyzed: AnalyzedTurn,
        turn_config: TurnConfig,
        content_callback: ContentCallback | None,
        *,
        override_messages: tuple[Message, ...] | None = None,
    ) -> ModelResponse:
        """Stage 3: Execute the model.
        
        Args:
            analyzed: The analyzed turn (used for first iteration).
            turn_config: Turn configuration.
            content_callback: Optional callback for content streaming.
            override_messages: Override messages for tool loop iterations.
        """
        if self._config.log_pipeline_stages:
            LOGGER.debug("Stage: Execute")

        # For tool loop iterations, we need to use override messages
        if override_messages is not None:
            # Create a synthetic AnalyzedTurn with the override messages
            # by wrapping them in a PreparedTurn
            from .types import BudgetEstimate
            synthetic_prepared = PreparedTurn(
                messages=override_messages,
                budget=analyzed.prepared.budget,
                system_prompt=analyzed.prepared.system_prompt,
                document_context=analyzed.prepared.document_context,
            )
            synthetic_turn = AnalyzedTurn(
                prepared=synthetic_prepared,
                hints=(),  # No hints for subsequent iterations
                analysis_ran=analyzed.analysis_ran,
            )
            turn_to_execute = synthetic_turn
        else:
            turn_to_execute = analyzed

        return await execute_model(
            turn=turn_to_execute,
            client=self._client,
            tools=self._tools if self._tools else None,
            temperature=turn_config.temperature,
            max_completion_tokens=turn_config.response_reserve,
            on_content_delta=content_callback,
        )

    async def _stage_tools(
        self,
        response: ModelResponse,
        tool_callback: ToolCallback | None,
    ) -> ToolResults:
        """Stage 4: Execute tool calls."""
        if self._config.log_pipeline_stages:
            LOGGER.debug(
                "Stage: Tools (%d calls)",
                len(response.tool_calls),
            )

        # Invoke tool callback for each call before execution
        if tool_callback:
            for tc in response.tool_calls:
                try:
                    # Parse arguments from JSON string
                    try:
                        args = json.loads(tc.arguments) if tc.arguments else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_callback(tc.call_id, tc.name, args)
                except Exception:
                    LOGGER.debug(
                        "Tool callback raised exception",
                        exc_info=True,
                    )

        return await execute_tools(
            response=response,
            executor=self._tool_executor,
            timeout_seconds=self._config.tool_timeout,
        )

    def _finish_success(
        self,
        response: ModelResponse,
        tool_records: tuple[ToolCallRecord, ...],
        timer: TurnTimer,
        iteration_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        turn_config: TurnConfig,
        analyzed: AnalyzedTurn,
    ) -> TurnOutput:
        """Finish a successful turn."""
        timer.stop()
        metrics = collect_metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=timer.duration_ms,
            tool_call_count=len(tool_records),
            iteration_count=iteration_count,
            model_name=turn_config.model_name,
            analysis_ran=analyzed.analysis_ran,
            started_at=timer.started_at,
            finished_at=timer.finished_at,
        )
        return finish_turn(
            response=response,
            metrics=metrics,
            tool_records=tool_records,
        )


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def create_runner(
    client: ModelClient,
    tool_executor: ToolExecutor,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    token_counter: TokenCounter | None = None,
    analysis_provider: AnalysisProvider | None = None,
    max_iterations: int | None = None,
    streaming_enabled: bool | None = None,
) -> TurnRunner:
    """Create a TurnRunner with the specified configuration.

    This is a convenience factory function for creating runners.

    Args:
        client: Model client for executing chat completions.
        tool_executor: Executor for running tool calls.
        tools: Optional list of tool definitions for the model.
        token_counter: Optional token counter for budget estimation.
        analysis_provider: Optional provider for preflight analysis.
        max_iterations: Maximum tool loop iterations.
        streaming_enabled: Whether to enable streaming.

    Returns:
        Configured TurnRunner instance.

    Example:
        >>> runner = create_runner(
        ...     client=ai_client,
        ...     tool_executor=executor,
        ...     tools=tool_definitions,
        ...     max_iterations=5,
        ... )
    """
    config = RunnerConfig(
        max_iterations=max_iterations,
        streaming_enabled=streaming_enabled,
    )

    return TurnRunner(
        client=client,
        tool_executor=tool_executor,
        config=config,
        token_counter=token_counter,
        analysis_provider=analysis_provider,
        tools=tools,
    )
