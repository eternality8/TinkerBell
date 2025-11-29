"""AI Orchestrator: Thin facade for AI chat orchestration.

This module provides the AIOrchestrator class, a clean facade that wires
together the pipeline components (TurnRunner, ToolDispatcher, Services)
into a coherent API for the UI layer.

This replaces the monolithic AIController with a simpler, focused design.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ..client import AIClient
from ..tools.tool_registry import (
    ToolRegistry,
    ToolSchema,
    ToolCategory,
    get_tool_registry,
)
from .tool_dispatcher import (
    ToolDispatcher,
    ToolContextProvider,
    DispatchResult,
)
from .runner import TurnRunner, RunnerConfig
from .types import (
    TurnInput,
    TurnOutput,
    TurnConfig,
    DocumentSnapshot,
    Message,
    ToolCallRecord,
)
from .pipeline.prepare import TokenCounter
from .pipeline.analyze import AnalysisProvider
from .pipeline.execute import ModelClient, StreamEvent
from .services import Services, create_services

__all__ = [
    "AIOrchestrator",
    "OrchestratorConfig",
    "ChatResult",
    "StreamCallback",
]

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

# Callback for streaming events during chat
StreamCallback = Callable[[StreamEvent], None]


@dataclass(slots=True, frozen=True)
class OrchestratorConfig:
    """Configuration for the AI orchestrator.

    Attributes:
        max_iterations: Maximum tool loop iterations.
        max_context_tokens: Maximum context window size.
        response_token_reserve: Tokens reserved for response.
        temperature: Sampling temperature.
        streaming_enabled: Whether to stream responses.
    """

    max_iterations: int = 8
    max_context_tokens: int = 128_000
    response_token_reserve: int = 16_000
    temperature: float = 0.2
    streaming_enabled: bool = True


@dataclass(slots=True)
class ChatResult:
    """Result of a chat turn.

    A simplified result type for callers that maps from TurnOutput.

    Attributes:
        response: The assistant's response text.
        tool_calls: List of tool call records.
        success: Whether the turn completed successfully.
        error: Error message if failed.
        prompt_tokens: Tokens used in prompt.
        completion_tokens: Tokens used in completion.
        duration_ms: Turn duration in milliseconds.
    """

    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0

    @classmethod
    def from_turn_output(cls, output: TurnOutput) -> ChatResult:
        """Create from TurnOutput."""
        return cls(
            response=output.response,
            tool_calls=[tc.to_dict() for tc in output.tool_calls],
            success=output.success,
            error=output.error,
            prompt_tokens=output.metrics.prompt_tokens,
            completion_tokens=output.metrics.completion_tokens,
            duration_ms=output.metrics.duration_ms,
        )


# -----------------------------------------------------------------------------
# Tool Executor Adapter
# -----------------------------------------------------------------------------


class _DispatcherToolExecutor:
    """Adapts ToolDispatcher to the ToolExecutor protocol.

    This allows the TurnRunner to use ToolDispatcher for executing tools.
    """

    def __init__(self, dispatcher: ToolDispatcher) -> None:
        self._dispatcher = dispatcher

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> str:
        """Execute a tool and return the result as a string."""
        result = await self._dispatcher.dispatch(name, dict(arguments))
        if result.success:
            return str(result.result) if result.result is not None else ""
        else:
            error_msg = result.error.message if result.error else "Unknown error"
            return f"Error: {error_msg}"


# -----------------------------------------------------------------------------
# Client Adapter
# -----------------------------------------------------------------------------


class _AIClientAdapter:
    """Adapts AIClient to the ModelClient protocol."""

    def __init__(self, client: AIClient) -> None:
        self._client = client

    async def stream_chat(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ):
        """Stream chat completion."""
        async for event in self._client.stream_chat(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield event


# -----------------------------------------------------------------------------
# AI Orchestrator
# -----------------------------------------------------------------------------


class AIOrchestrator:
    """Orchestrates AI chat turns with tool execution.

    This is a thin facade that wires together:
    - TurnRunner for pipeline execution
    - ToolDispatcher for tool routing
    - ToolRegistry for tool schemas
    - Services for caching, telemetry, etc.

    Example:
        orchestrator = AIOrchestrator(client=ai_client)
        result = await orchestrator.run_chat(
            prompt="Summarize this document",
            snapshot=doc_snapshot,
        )
        print(result.response)
    """

    def __init__(
        self,
        client: AIClient,
        *,
        config: OrchestratorConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        services: Services | None = None,
        token_counter: TokenCounter | None = None,
        analysis_provider: AnalysisProvider | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            client: The AI client for model calls.
            config: Orchestrator configuration.
            tool_registry: Tool registry (default: global registry).
            services: Services container for caching/telemetry.
            token_counter: Optional token counter.
            analysis_provider: Optional analysis provider.
        """
        self._client = client
        self._config = config or OrchestratorConfig()
        self._tool_registry = tool_registry or get_tool_registry()
        self._services = services
        self._token_counter = token_counter
        self._analysis_provider = analysis_provider

        # Tool dispatcher (will be configured later)
        self._tool_dispatcher: ToolDispatcher | None = None

        # Active task for cancellation
        self._active_task: asyncio.Task[Any] | None = None
        self._task_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> AIClient:
        """The AI client."""
        return self._client

    @property
    def config(self) -> OrchestratorConfig:
        """The orchestrator configuration."""
        return self._config

    @property
    def tool_registry(self) -> ToolRegistry:
        """The tool registry."""
        return self._tool_registry

    @property
    def tool_dispatcher(self) -> ToolDispatcher | None:
        """The tool dispatcher."""
        return self._tool_dispatcher

    @property
    def services(self) -> Services | None:
        """The services container."""
        return self._services

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_tool_dispatcher(
        self,
        *,
        context_provider: ToolContextProvider | None = None,
    ) -> None:
        """Configure the tool dispatcher.

        Args:
            context_provider: Provider for document context (usually the bridge).
        """
        self._tool_dispatcher = ToolDispatcher(
            registry=self._tool_registry,
            context_provider=context_provider,
        )

    def update_client(self, client: AIClient) -> None:
        """Update the AI client.

        Args:
            client: New AI client to use.
        """
        self._client = client

    def set_config(self, config: OrchestratorConfig) -> None:
        """Update the orchestrator configuration.

        Args:
            config: New configuration.
        """
        self._config = config

    # ------------------------------------------------------------------
    # Tool Registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        tool: Any,
        *,
        description: str,
        parameters: Mapping[str, Any],
    ) -> None:
        """Register a tool with the orchestrator.

        Args:
            name: Tool name.
            tool: Tool implementation.
            description: Tool description for the model.
            parameters: JSON Schema for tool parameters.
        """
        self._tool_registry.register(
            tool,
            name=name,
            description=description,
            parameters=parameters,
        )

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: Tool name to remove.
        """
        self._tool_registry.unregister(name)

    def available_tools(self) -> tuple[str, ...]:
        """Get names of registered tools.

        Returns:
            Tuple of tool names.
        """
        return tuple(self._tool_registry.list_tools())

    # ------------------------------------------------------------------
    # Chat Execution
    # ------------------------------------------------------------------

    async def run_chat(
        self,
        prompt: str,
        snapshot: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, str] | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
        on_event: StreamCallback | None = None,
    ) -> ChatResult:
        """Execute a chat turn.

        Args:
            prompt: User's message.
            snapshot: Document snapshot (tab_id, content, etc.).
            metadata: Additional metadata.
            history: Conversation history.
            on_event: Callback for streaming events.

        Returns:
            ChatResult with response and tool calls.
        """
        run_id = str(uuid.uuid4())

        # Build document snapshot
        doc_snapshot = self._build_document_snapshot(snapshot)

        # Convert history to Message tuples
        message_history = self._convert_history(history)

        # Build turn config
        turn_config = TurnConfig(
            max_iterations=self._config.max_iterations,
            max_context_tokens=self._config.max_context_tokens,
            response_reserve=self._config.response_token_reserve,
            temperature=self._config.temperature,
            streaming_enabled=self._config.streaming_enabled,
        )

        # Build turn input
        turn_input = TurnInput(
            prompt=prompt,
            snapshot=doc_snapshot,
            history=message_history,
            config=turn_config,
            run_id=run_id,
            document_id=doc_snapshot.tab_id or "",
        )

        # Create runner
        runner = self._create_runner()

        # Content callback adapter
        content_callback = None
        if on_event:
            def content_callback(text: str) -> None:
                # Create a minimal event-like object
                event = _ContentDeltaEvent(text)
                on_event(event)

        # Execute under task lock
        async with self._task_lock:
            task = asyncio.create_task(
                runner.run(turn_input, content_callback=content_callback)
            )
            self._active_task = task

        try:
            output = await task
            return ChatResult.from_turn_output(output)
        except asyncio.CancelledError:
            return ChatResult(
                response="",
                success=False,
                error="Turn was cancelled",
            )
        except Exception as e:
            LOGGER.exception("Chat turn failed")
            return ChatResult(
                response="",
                success=False,
                error=str(e),
            )
        finally:
            if self._active_task is task:
                self._active_task = None

    def cancel(self) -> None:
        """Cancel the active chat turn."""
        task = self._active_task
        if task and not task.done():
            LOGGER.debug("Cancelling active chat task")
            task.cancel()

    async def aclose(self) -> None:
        """Close the orchestrator and clean up resources."""
        task = self._active_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if self._active_task is task:
                self._active_task = None

        # Close client if it supports it
        close = getattr(self._client, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    async def suggest_followups(
        self,
        history: Sequence[Mapping[str, str]],
        *,
        max_suggestions: int = 4,
    ) -> list[str]:
        """Generate follow-up suggestions based on history.

        Args:
            history: Conversation history.
            max_suggestions: Maximum number of suggestions.

        Returns:
            List of suggested follow-up prompts.
        """
        if not history:
            return []

        # Build prompt for suggestions
        messages = self._build_suggestion_messages(history, max_suggestions)

        # Simple chat completion
        response_text = await self._complete_simple_chat(messages)

        # Parse suggestions
        return self._parse_suggestions(response_text, max_suggestions)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _create_runner(self) -> TurnRunner:
        """Create a TurnRunner for chat execution."""
        # Get tool definitions for the model
        tool_definitions = self._get_tool_definitions()

        # Create tool executor
        tool_executor: Any = None
        if self._tool_dispatcher:
            tool_executor = _DispatcherToolExecutor(self._tool_dispatcher)

        # Create client adapter
        client_adapter = _AIClientAdapter(self._client)

        # Runner config
        runner_config = RunnerConfig(
            max_iterations=self._config.max_iterations,
            streaming_enabled=self._config.streaming_enabled,
        )

        return TurnRunner(
            client=client_adapter,
            tool_executor=tool_executor,
            config=runner_config,
            token_counter=self._token_counter,
            analysis_provider=self._analysis_provider,
            tools=tool_definitions,
            services=self._services,
        )

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for the model."""
        definitions = []
        for name in self._tool_registry.list_tools():
            registration = self._tool_registry.get_registration(name)
            if registration and registration.enabled:
                definitions.append({
                    "type": "function",
                    "function": {
                        "name": registration.schema.name,
                        "description": registration.schema.description,
                        "parameters": registration.schema.to_json_schema(),
                    },
                })
        return definitions

    def _build_document_snapshot(
        self,
        snapshot: Mapping[str, Any] | None,
    ) -> DocumentSnapshot:
        """Build a DocumentSnapshot from a raw snapshot dict."""
        if not snapshot:
            return DocumentSnapshot(tab_id="", content="", version_token=None)

        # Only pass timestamp if present and not None
        timestamp = snapshot.get("timestamp")
        if timestamp is not None:
            return DocumentSnapshot(
                tab_id=str(snapshot.get("tab_id") or snapshot.get("document_id") or ""),
                content=str(snapshot.get("content") or snapshot.get("text") or ""),
                version_token=snapshot.get("version_token"),
                timestamp=timestamp,
            )
        else:
            return DocumentSnapshot(
                tab_id=str(snapshot.get("tab_id") or snapshot.get("document_id") or ""),
                content=str(snapshot.get("content") or snapshot.get("text") or ""),
                version_token=snapshot.get("version_token"),
            )

    def _convert_history(
        self,
        history: Sequence[Mapping[str, str]] | None,
    ) -> tuple[Message, ...]:
        """Convert history dicts to Message tuples."""
        if not history:
            return ()

        messages: list[Message] = []
        for entry in history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            messages.append(Message(role=role, content=content))
        return tuple(messages)

    def _build_suggestion_messages(
        self,
        history: Sequence[Mapping[str, str]],
        max_suggestions: int,
    ) -> list[dict[str, str]]:
        """Build messages for suggestion generation."""
        # Build transcript
        transcript_lines: list[str] = []
        for entry in history[-10:]:
            role = (entry.get("role") or "user").lower()
            label = "User" if role == "user" else "Assistant"
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            if len(content) > 400:
                content = f"{content[:397].rstrip()}…"
            transcript_lines.append(f"{label}: {content}")

        transcript = "\n".join(transcript_lines) or "(no content)"

        system_prompt = f"""You are a helpful assistant that suggests follow-up questions or commands.
Based on the conversation, suggest up to {max(1, max_suggestions)} natural, helpful follow-ups.
Return them as a simple numbered list, one per line.
Focus on actionable next steps that would help the user accomplish their goals."""

        user_prompt = (
            "Here is the recent conversation:\n"
            f"{transcript}\n\n"
            "Suggest helpful follow-up questions or commands."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _complete_simple_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> str:
        """Complete a simple chat without tools."""
        chunks: list[str] = []
        async for event in self._client.stream_chat(
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=300,
        ):
            if hasattr(event, "type"):
                if event.type == "content.delta" and event.content:
                    chunks.append(str(event.content))
                elif event.type == "content.done" and event.content:
                    if event.content not in "".join(chunks):
                        chunks.append(str(event.content))
        return "".join(chunks).strip()

    def _parse_suggestions(
        self,
        text: str,
        max_suggestions: int,
    ) -> list[str]:
        """Parse suggestion text into a list."""
        if not text:
            return []

        lines = [
            line.strip(" -*\t0123456789.")
            for line in text.splitlines()
            if line.strip()
        ]

        suggestions = [
            line for line in lines
            if 5 <= len(line) <= 200
        ]

        return suggestions[:max_suggestions]


# -----------------------------------------------------------------------------
# Helper Classes
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class _ContentDeltaEvent:
    """Minimal event for content streaming."""

    content: str
    type: str = "content.delta"
