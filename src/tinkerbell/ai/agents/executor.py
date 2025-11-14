"""Agent executor façade wrapping LangChain/LangGraph interactions."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, MutableMapping, Optional, cast

from openai.types.chat import ChatCompletionToolParam

from .. import prompts
from ..client import AIClient, AIStreamEvent
from .graph import build_agent_graph

LOGGER = logging.getLogger(__name__)

ToolCallback = Callable[[AIStreamEvent], Awaitable[None] | None]


@dataclass(slots=True)
class ToolRegistration:
    """Metadata stored for each registered tool."""

    name: str
    impl: Any
    description: str | None = None
    parameters: Mapping[str, Any] | None = None

    def as_openai_tool(self) -> ChatCompletionToolParam:
        """Return an OpenAI-compatible tool spec for the AI client."""

        parameters = self.parameters or getattr(self.impl, "args_schema", None)
        if callable(parameters):
            try:  # pragma: no cover - defensive, args_schema can be pydantic.BaseModel
                parameters = parameters()
            except TypeError:
                parameters = None
        if parameters is None:
            parameters = {"type": "object", "properties": {}}

        description = self.description or getattr(self.impl, "description", None) or (
            inspect.getdoc(self.impl) or f"Tool {self.name}"
        )

        return cast(
            ChatCompletionToolParam,
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": description,
                    "parameters": parameters,
                },
            },
        )


@dataclass(slots=True)
class AIController:
    """High-level interface invoked by the chat panel."""

    client: AIClient
    tools: MutableMapping[str, ToolRegistration] = field(default_factory=dict)
    _graph: Dict[str, Any] = field(init=False, repr=False)
    _active_task: asyncio.Task[dict] | None = field(default=None, init=False, repr=False)
    _task_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tools:
            normalized: Dict[str, ToolRegistration] = {}
            for name, value in self.tools.items():
                if isinstance(value, ToolRegistration):
                    normalized[name] = value
                else:
                    normalized[name] = ToolRegistration(name=name, impl=value)
            self.tools = normalized
        self._rebuild_graph()

    @property
    def graph(self) -> Dict[str, Any]:
        """Return the current compiled graph representation."""

        return dict(self._graph)

    def register_tool(
        self,
        name: str,
        tool: Any,
        *,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        """Register (or replace) a tool available to the agent."""

        self.tools[name] = ToolRegistration(name=name, impl=tool, description=description, parameters=parameters)
        LOGGER.debug("Registered tool: %s", name)
        self._rebuild_graph()

    def unregister_tool(self, name: str) -> None:
        """Remove a tool and rebuild the agent graph."""

        if name in self.tools:
            self.tools.pop(name)
            LOGGER.debug("Unregistered tool: %s", name)
            self._rebuild_graph()

    def update_client(self, client: AIClient) -> None:
        """Swap the underlying AI client (e.g., when settings change)."""

        self.client = client

    async def run_chat(
        self,
        prompt: str,
        doc_snapshot: Mapping[str, Any] | None,
        *,
        metadata: Mapping[str, str] | None = None,
        on_event: ToolCallback | None = None,
    ) -> dict:
        """Execute a chat turn against the compiled agent graph."""

        snapshot = dict(doc_snapshot or {})
        messages = self._build_messages(prompt, snapshot)
        merged_metadata = self._build_metadata(snapshot, metadata)
        tool_specs = [registration.as_openai_tool() for registration in self.tools.values()]

        LOGGER.debug(
            "Starting chat turn (prompt length=%s, tools=%s)",
            len(prompt),
            list(self.tools.keys()),
        )

        async def _runner() -> dict:
            deltas: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            async for event in self.client.stream_chat(
                messages=messages,
                tools=tool_specs if tool_specs else None,
                metadata=merged_metadata,
            ):
                await self._dispatch_event(event, on_event)
                if event.type == "content.delta" and event.content:
                    deltas.append(event.content)
                elif event.type == "content.done" and event.content:
                    deltas.append(str(event.content))
                elif event.type.endswith("tool_calls.function.arguments.done"):
                    tool_calls.append(
                        {
                            "name": event.tool_name,
                            "index": event.tool_index,
                            "arguments": event.tool_arguments,
                            "parsed": event.parsed,
                        }
                    )

            response_text = "".join(deltas).strip()
            LOGGER.debug("Chat turn complete (chars=%s, tools used=%s)", len(response_text), len(tool_calls))
            return {
                "prompt": prompt,
                "response": response_text,
                "doc_snapshot": snapshot,
                "tool_calls": tool_calls,
                "graph": self.graph,
            }

        async with self._task_lock:
            task = asyncio.create_task(_runner())
            self._active_task = task

        try:
            return await task
        finally:
            if self._active_task is task:
                self._active_task = None

    def cancel(self) -> None:
        """Cancel the active chat turn, if any."""

        if self._active_task and not self._active_task.done():
            LOGGER.debug("Cancelling active chat task")
            self._active_task.cancel()

    def available_tools(self) -> tuple[str, ...]:
        """Return the names of the registered tools."""

        return tuple(self.tools.keys())

    def _rebuild_graph(self) -> None:
        self._graph = build_agent_graph(tools={name: registration.impl for name, registration in self.tools.items()})

    def _build_messages(self, prompt: str, snapshot: Mapping[str, Any]) -> list[dict[str, str]]:
        user_prompt = prompts.format_user_prompt(prompt, dict(snapshot))
        return [
            {"role": "system", "content": prompts.base_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

    def _build_metadata(
        self,
        snapshot: Mapping[str, Any],
        runtime_metadata: Mapping[str, str] | None,
    ) -> Optional[Dict[str, str]]:
        metadata: Dict[str, str] = {}
        path = snapshot.get("path")
        if path:
            metadata["doc_path"] = str(path)
        selection = snapshot.get("selection")
        if isinstance(selection, Mapping):
            start = selection.get("start")
            end = selection.get("end")
            metadata["selection"] = f"{start}:{end}"
        if runtime_metadata:
            metadata.update(runtime_metadata)
        return metadata or None

    async def _dispatch_event(self, event: AIStreamEvent, handler: ToolCallback | None) -> None:
        if handler is None:
            return
        result = handler(event)
        if inspect.isawaitable(result):
            await result


