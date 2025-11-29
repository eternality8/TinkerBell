"""Declarative description of the LangGraph agent flow.

This module does not construct the full LangGraph graph just yet; instead it
produces a structured, serialisable description of the intended flow so that
the UI (and tests) can reason about the available nodes and tool wiring. The
layout follows the design captured in ``plan.md`` / ``module_plan.md``:

``ingest → planner → tool loop → guard → respond``

Each node captures its responsibility, the transitions it exposes, and any
timeout/retry configuration so we can evolve toward a fully fledged
``langgraph`` implementation without breaking consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Iterable, Mapping, MutableMapping

from ..ai_types import AgentConfig


@dataclass(frozen=True, slots=True)
class GraphNode:
    """Human-readable description of a node in the agent graph."""

    name: str
    kind: str
    description: str
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """Declarative edge definition linking graph nodes."""

    source: str
    target: str
    condition: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "condition": self.condition,
        }


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Summary of a registered tool exposed to the agent."""

    name: str
    description: str
    entry_point: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "entry_point": self.entry_point,
        }


@dataclass(slots=True)
class GraphSpec:
    """Full description of the agent graph topology."""

    entry: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    tools: list[ToolDescriptor]
    metadata: MutableMapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "tools": [tool.to_dict() for tool in self.tools],
            "metadata": dict(self.metadata),
        }


class AgentGraphBuilder:
    """Builder translating registered tools into a LangGraph-style blueprint."""

    def __init__(self, tools: Mapping[str, Any], *, config: AgentConfig | None = None) -> None:
        self._tools = dict(tools)
        self._config = (config or AgentConfig()).clamp()

    def _base_nodes(self) -> list[GraphNode]:
        policy = self._config.retry_policy
        return [
            GraphNode(
                name="plan",
                kind="planner",
                description="Summarize the request, inspect snapshot metadata, and decide the next step.",
                metadata={
                    "contracts": ["prompt", "doc_snapshot", "selection"],
                    "strategy": "react",
                    "max_tokens": 256,
                    "max_retries": policy.planner_max_retries,
                    "retry_backoff": [policy.backoff_min_seconds, policy.backoff_max_seconds],
                },
            ),
            GraphNode(
                name="select_tool",
                kind="router",
                description="Choose the next editor-aware tool or finish with a written response.",
                metadata={
                    "allow_parallel": self._config.allow_parallel_tools,
                    "diff_required": self._config.diff_required_for_edits,
                },
            ),
            GraphNode(
                name="tool_executor",
                kind="tool",
                description="Invoke snapshot/diff/edit/search tools and capture structured telemetry.",
                metadata={
                    "timeout_s": policy.tool_timeout_seconds,
                    "retry_limit": policy.tool_retry_limit,
                },
            ),
            GraphNode(
                name="safety_validator",
                kind="guard",
                description="Enforce document versioning, diff requirements, and validation hooks.",
                metadata={
                    "validation_retries": policy.validation_retry_limit,
                    "enforce_version": True,
                    "rules": ["requires_rationale", "diff_required_for_edits"],
                },
            ),
            GraphNode(
                name="response_builder",
                kind="respond",
                description="Compose the final assistant message plus any queued edit directives.",
                metadata={"outputs": ["text", "edit_directive", "tool_trace"]},
            ),
        ]

    @staticmethod
    def _base_edges() -> list[GraphEdge]:
        return [
            GraphEdge("plan", "select_tool", "tool_needed"),
            GraphEdge("plan", "response_builder", "final_response"),
            GraphEdge("select_tool", "tool_executor", "dispatch"),
            GraphEdge("tool_executor", "safety_validator", "tool_completed"),
            GraphEdge("safety_validator", "plan", "needs_follow_up"),
            GraphEdge("safety_validator", "response_builder", "validated"),
        ]

    def build(self) -> GraphSpec:
        tool_descriptors = self._build_tool_descriptors()
        nodes = self._base_nodes() + self._tool_nodes(tool_descriptors)
        edges = self._base_edges() + self._tool_edges(tool_descriptors)
        metadata: MutableMapping[str, Any] = {
            "max_iterations": self._config.max_iterations,
            "planner": {
                "strategy": "react",
                "allow_parallel_tools": self._config.allow_parallel_tools,
                "max_retries": self._config.retry_policy.planner_max_retries,
            },
            "tooling": {
                "registered": len(tool_descriptors),
                "timeout_s": self._config.retry_policy.tool_timeout_seconds,
            },
            "retry_policy": self._config.retry_policy.as_metadata(),
            "diff_required": self._config.diff_required_for_edits,
        }
        return GraphSpec(
            entry="plan",
            nodes=nodes,
            edges=edges,
            tools=tool_descriptors,
            metadata=metadata,
        )

    def _build_tool_descriptors(self) -> list[ToolDescriptor]:
        descriptors: list[ToolDescriptor] = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            description = _describe_tool(tool) or f"Tool {name}"
            entry_point = _tool_entry_point(tool)
            descriptors.append(ToolDescriptor(name=name, description=description, entry_point=entry_point))
        return descriptors

    def _tool_nodes(self, tools: Iterable[ToolDescriptor]) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        for descriptor in tools:
            nodes.append(
                GraphNode(
                    name=f"tool:{descriptor.name}",
                    kind="tool",
                    description=descriptor.description,
                    metadata={"entry_point": descriptor.entry_point},
                )
            )
        return nodes

    def _tool_edges(self, tools: Iterable[ToolDescriptor]) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        for descriptor in tools:
            node_name = f"tool:{descriptor.name}"
            edges.append(GraphEdge("tool_executor", node_name, "dispatch"))
            edges.append(GraphEdge(node_name, "safety_validator", "result"))
        return edges


def _describe_tool(tool: Any) -> str | None:
    """Attempt to craft a helpful description for a tool instance."""

    description = getattr(tool, "description", None)
    if isinstance(description, str) and description.strip():
        return description.strip()
    doc = inspect.getdoc(tool)
    if doc:
        return doc.splitlines()[0].strip()
    return None


def _tool_entry_point(tool: Any) -> str:
    """Return the dotted path for the callable entry point of a tool."""

    if callable(tool):
        module = getattr(tool, "__module__", "") or ""
        qualname = getattr(tool, "__qualname__", getattr(tool, "__name__", tool.__class__.__name__))
        return f"{module}.{qualname}".strip(".")
    run_method = getattr(tool, "run", None)
    if callable(run_method):
        module = getattr(run_method, "__module__", tool.__class__.__module__)
        qualname = getattr(run_method, "__qualname__", tool.__class__.__name__ + ".run")
        return f"{module}.{qualname}".strip(".")
    return tool.__class__.__module__ + "." + tool.__class__.__qualname__


def build_agent_graph(*, tools: Mapping[str, Any], config: AgentConfig | None = None) -> dict[str, Any]:
    """Return a declarative description of the LangGraph wiring.

    Parameters
    ----------
    tools:
        Mapping of tool name → tool implementation as registered by
        :class:`~tinkerbell.ai.orchestration.AIOrchestrator`.
    max_iterations:
        Safety guard limiting how many times the planner can loop through the
        tool invocation cycle before forcing a final response.
    """

    builder = AgentGraphBuilder(tools=tools, config=config)
    return builder.build().to_dict()

