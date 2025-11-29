"""Runtime helpers for subagent orchestration."""

from __future__ import annotations

import logging
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
    from .model_types import OpenAIToolSpec

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SubagentRuntimeManager:
    """Stores shared subagent state and wiring dependencies."""

    tool_resolver: Callable[[], Mapping[str, "OpenAIToolSpec"]]
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
        LOGGER.debug(
            "Configuring subagent runtime (enabled=%s, plot_scaffolding=%s)",
            runtime.enabled,
            runtime.plot_scaffolding_enabled,
        )
        if not runtime.enabled:
            LOGGER.debug("Subagent runtime disabled; clearing manager state")
            self.manager = None
            return runtime

        if self.cache is None:
            self.cache = SubagentResultCache()
            LOGGER.debug("Initialized subagent result cache")

        if self.manager is None:
            LOGGER.debug("Creating SubagentManager instance")
            self.manager = SubagentManager(
                client,
                tool_resolver=self.tool_resolver,
                config=runtime,
                budget_policy=budget_policy,
                result_cache=self.cache,
            )
        else:
            LOGGER.debug("Updating existing SubagentManager instance")
            self.manager.update_client(client)
            self.manager.update_config(runtime)
            self.manager.update_budget_policy(budget_policy)
            self.manager.update_cache(self.cache)

        if runtime.plot_scaffolding_enabled:
            if self.plot_state_store is None:
                self.plot_state_store = PlotStateMemory()
                LOGGER.debug("Initialized plot state memory store")
            if self.character_map_store is None:
                self.character_map_store = CharacterMapStore()
                LOGGER.debug("Initialized character map store")
            # New tool names that replace legacy plot scaffolding tools:
            # - get_outline: replaces plot_outline
            # - analyze_document: replaces document_plot_state, character_map
            # - transform_document: replaces plot_state_update, character_edit_planner
            extras = (
                "get_outline",
                "analyze_document",
                "transform_document",
            )
            current = list(runtime.allowed_tools)
            for tool in extras:
                if tool not in current:
                    current.append(tool)
            runtime.allowed_tools = tuple(current)
            LOGGER.debug(
                "Augmented subagent allowed tools with plot scaffolding extras: %s",
                extras,
            )

        return runtime

    def update_client(self, client: AIClient) -> None:
        if self.manager is not None:
            self.manager.update_client(client)

    def ensure_plot_state_store(self) -> PlotStateMemory:
        if self.plot_state_store is None:
            self.plot_state_store = PlotStateMemory()
            LOGGER.debug("Lazily created plot state store")
        return self.plot_state_store

    def ensure_character_map_store(self) -> CharacterMapStore:
        if self.character_map_store is None:
            self.character_map_store = CharacterMapStore()
            LOGGER.debug("Lazily created character map store")
        return self.character_map_store
