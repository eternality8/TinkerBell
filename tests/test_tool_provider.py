from __future__ import annotations

from types import MethodType
from typing import Any, cast

import pytest

from tinkerbell.ui.tools import provider as provider_module
from tinkerbell.ui.tools.provider import ToolProvider


class _WorkerStub:
    def __init__(self, *, pending: bool = False) -> None:
        self._pending = pending

    def is_rebuild_pending(self, document_id: str) -> bool:  # pragma: no cover - trivial
        return self._pending and bool(document_id)


def _make_provider(*, phase3: bool = True, plot: bool = True) -> ToolProvider:
    return ToolProvider(
        controller_resolver=lambda: object(),
        bridge=object(),
        document_lookup=lambda doc_id: {"id": doc_id},
        active_document_provider=lambda: {"id": "active"},
        outline_worker_resolver=lambda: cast(Any, _WorkerStub()),
        outline_memory_resolver=lambda: {"memory": True},
        embedding_index_resolver=lambda: cast(Any, {"index": True}),
        outline_digest_resolver=lambda doc_id: f"digest:{doc_id}" if doc_id else None,
        directive_schema_provider=lambda: {"type": "object"},
        plot_state_store_resolver=lambda: cast(Any, {"store": True}),
        character_map_store_resolver=lambda: cast(Any, {"characters": True}),
        phase3_outline_enabled=phase3,
        plot_scaffolding_enabled=plot,
    )


def test_build_tool_context_exposes_current_state() -> None:
    provider = _make_provider()

    context = provider.build_tool_registry_context()

    assert context.phase3_outline_enabled is True
    assert context.plot_scaffolding_enabled is True
    outline_factory = cast(MethodType, context.ensure_outline_tool)
    assert outline_factory.__self__ is provider
    assert outline_factory.__func__ is provider.ensure_outline_tool.__func__

    plot_factory = cast(MethodType, context.ensure_plot_state_tool)
    assert plot_factory.__self__ is provider
    assert plot_factory.__func__ is provider.ensure_plot_state_tool.__func__

    concordance_factory = cast(MethodType, context.ensure_character_map_tool)
    assert concordance_factory.__self__ is provider
    assert concordance_factory.__func__ is provider.ensure_character_map_tool.__func__

    planner_factory = cast(MethodType, context.ensure_character_planner_tool)
    assert planner_factory.__self__ is provider
    assert planner_factory.__func__ is provider.ensure_character_planner_tool.__func__

    update_factory = cast(MethodType, context.ensure_plot_state_update_tool)
    assert update_factory.__self__ is provider
    assert update_factory.__func__ is provider.ensure_plot_state_update_tool.__func__


def test_ensure_outline_tool_memoizes_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []

    class _OutlineStub:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    monkeypatch.setattr(provider_module, "DocumentOutlineTool", _OutlineStub)
    provider = _make_provider()

    first = provider.ensure_outline_tool()
    second = provider.ensure_outline_tool()
    assert first is second
    assert len(created) == 1

    provider.set_phase3_outline_enabled(False)
    assert provider.ensure_outline_tool() is None

    provider.set_phase3_outline_enabled(True)
    third = provider.ensure_outline_tool()
    assert third is not first
    assert len(created) == 2


def test_ensure_find_sections_tool_handles_constructor_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingSections:
        def __init__(self, **_: object) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(provider_module, "DocumentFindSectionsTool", _ExplodingSections)
    provider = _make_provider()

    assert provider.ensure_find_sections_tool() is None
    # After a failure the provider should still attempt re-instantiation when called again.
    assert provider.ensure_find_sections_tool() is None


def test_plot_state_tool_respects_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []

    class _PlotStub:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    monkeypatch.setattr(provider_module, "DocumentPlotStateTool", _PlotStub)
    provider = _make_provider(plot=False)

    assert provider.ensure_plot_state_tool() is None
    assert created == []

    provider.set_plot_scaffolding_enabled(True)
    tool = provider.ensure_plot_state_tool()
    assert tool is not None
    assert len(created) == 1

    provider.set_plot_scaffolding_enabled(False)
    assert provider.ensure_plot_state_tool() is None


def test_plot_state_update_tool_respects_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []

    class _UpdateStub:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    monkeypatch.setattr(provider_module, "PlotStateUpdateTool", _UpdateStub)
    provider = _make_provider(plot=False)

    assert provider.ensure_plot_state_update_tool() is None
    assert created == []

    provider.set_plot_scaffolding_enabled(True)
    tool = provider.ensure_plot_state_update_tool()
    assert tool is not None
    assert len(created) == 1

    provider.set_plot_scaffolding_enabled(False)
    assert provider.ensure_plot_state_update_tool() is None


def test_character_map_tool_respects_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []

    class _CharacterMapStub:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    monkeypatch.setattr(provider_module, "CharacterMapTool", _CharacterMapStub)
    provider = _make_provider(plot=False)

    assert provider.ensure_character_map_tool() is None
    assert created == []

    provider.set_plot_scaffolding_enabled(True)
    tool = provider.ensure_character_map_tool()
    assert tool is not None
    assert len(created) == 1

    provider.set_plot_scaffolding_enabled(False)
    assert provider.ensure_character_map_tool() is None


def test_character_planner_tool_respects_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []

    class _PlannerStub:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    monkeypatch.setattr(provider_module, "CharacterEditPlannerTool", _PlannerStub)
    provider = _make_provider(plot=False)

    assert provider.ensure_character_planner_tool() is None
    assert created == []

    provider.set_plot_scaffolding_enabled(True)
    tool = provider.ensure_character_planner_tool()
    assert tool is not None
    assert len(created) == 1

    provider.set_plot_scaffolding_enabled(False)
    assert provider.ensure_character_planner_tool() is None
