"""Runtime helpers for subagent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping

from ..ai_types import SubagentRuntimeConfig
from ..client import AIClient
from ..memory.plot_memory import PlotStateMemory
from ..memory.character_map import CharacterMapStore
from ..memory.result_cache import SubagentResultCache
from ..services.context_policy import ContextBudgetPolicy
from ..agents.subagents import SubagentManager

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from .controller import ToolRegistration


@dataclass(slots=True)
class SubagentRuntimeManager:
    """Stores shared subagent state and wiring dependencies."""

    tool_resolver: Callable[[], Mapping[str, "ToolRegistration"]]
    manager: SubagentManager | None = None
    cache: SubagentResultCache | None = None
    plot_state_store: PlotStateMemory | None = None
    character_map_store: CharacterMapStore | None = None
    config: SubagentRuntimeConfig = field(default_factory=SubagentRuntimeConfig)

    def configure(
        self,
        *,
        client: AIClient,
        config: SubagentRuntimeConfig | None,
        budget_policy: ContextBudgetPolicy | None,
    ) -> SubagentRuntimeConfig:
        runtime = (config or SubagentRuntimeConfig()).clamp()
        self.config = runtime
        if not runtime.enabled:
            self.manager = None
            return runtime

        if self.cache is None:
            self.cache = SubagentResultCache()

        if self.manager is None:
            self.manager = SubagentManager(
                client,
                tool_resolver=self.tool_resolver,
                config=runtime,
                budget_policy=budget_policy,
                result_cache=self.cache,
            )
        else:
            self.manager.update_client(client)
            self.manager.update_config(runtime)
            self.manager.update_budget_policy(budget_policy)
            self.manager.update_cache(self.cache)

        if runtime.plot_scaffolding_enabled:
            if self.plot_state_store is None:
                self.plot_state_store = PlotStateMemory()
            if self.character_map_store is None:
                self.character_map_store = CharacterMapStore()
            extras = (
                "plot_outline",
                "document_plot_state",
                "plot_state_update",
                "character_map",
                "character_edit_planner",
            )
            current = list(runtime.allowed_tools)
            for tool in extras:
                if tool not in current:
                    current.append(tool)
            runtime.allowed_tools = tuple(current)

        return runtime

    def update_client(self, client: AIClient) -> None:
        if self.manager is not None:
            self.manager.update_client(client)

    def ensure_plot_state_store(self) -> PlotStateMemory:
        if self.plot_state_store is None:
            self.plot_state_store = PlotStateMemory()
        return self.plot_state_store

    def ensure_character_map_store(self) -> CharacterMapStore:
        if self.character_map_store is None:
            self.character_map_store = CharacterMapStore()
        return self.character_map_store
