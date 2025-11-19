"""Declarative registration helpers for AI tools."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

from ...chat.commands import DIRECTIVE_SCHEMA
from .character_edit_planner import CharacterEditPlannerTool
from .character_map import CharacterMapTool
from .diff_builder import DiffBuilderTool
from .document_apply_patch import DocumentApplyPatchTool
from .document_chunk import DocumentChunkTool
from .document_edit import DocumentEditTool
from .document_find_sections import DocumentFindSectionsTool
from .document_outline import DocumentOutlineTool
from .document_plot_state import DocumentPlotStateTool, PlotOutlineTool
from .document_snapshot import DocumentSnapshotTool
from .list_tabs import ListTabsTool
from .plot_state_update import PlotStateUpdateTool
from .search_replace import SearchReplaceTool
from .tool_usage_advisor import ToolUsageAdvisorTool
from .validation import validate_snippet

LOGGER = logging.getLogger(__name__)

@dataclass(slots=True)
class ToolRegistrationFailure:
    """Represents a single tool registration error."""

    name: str
    error: Exception


class ToolRegistrationError(RuntimeError):
    """Aggregated exception emitted when some tools fail to register."""

    def __init__(self, failures: Sequence[ToolRegistrationFailure]):
        names = ", ".join(f.name for f in failures) or "unknown"
        super().__init__(f"Failed to register tool(s): {names}")
        self.failures = tuple(failures)


@dataclass(slots=True)
class ToolRegistryContext:
    """Runtime dependencies required to register document-aware tools."""

    controller: Any
    bridge: Any
    outline_digest_resolver: Callable[..., Any]
    directive_schema_provider: Callable[[], Mapping[str, Any]]
    phase3_outline_enabled: bool = False
    plot_scaffolding_enabled: bool = False
    ensure_outline_tool: Callable[[], DocumentOutlineTool | None] | None = None
    ensure_find_sections_tool: Callable[[], DocumentFindSectionsTool | None] | None = None
    ensure_plot_state_tool: Callable[[], PlotOutlineTool | None] | None = None
    ensure_plot_outline_tool: Callable[[], PlotOutlineTool | None] | None = None
    ensure_plot_state_update_tool: Callable[[], PlotStateUpdateTool | None] | None = None
    ensure_character_map_tool: Callable[[], CharacterMapTool | None] | None = None
    ensure_character_planner_tool: Callable[[], CharacterEditPlannerTool | None] | None = None
    auto_patch_consumer: Callable[[DocumentApplyPatchTool], None] | None = None


def register_default_tools(
    context: ToolRegistryContext,
    *,
    register_fn: Callable[..., Any] | None = None,
) -> None:
    """Register the core AI tools used by the desktop UI."""

    controller = context.controller
    if controller is None:
        return
    register = register_fn or getattr(controller, "register_tool", None)
    if not callable(register):
        LOGGER.debug("AI controller does not expose register_tool; skipping tool wiring.")
        return

    failures: list[ToolRegistrationFailure] = []
    registered: list[str] = []

    chunk_index_instance = None
    chunk_settings_resolver = getattr(controller, "get_chunking_config", None)
    if not callable(chunk_settings_resolver):
        chunk_settings_resolver = None
    ensure_chunk_index = getattr(controller, "ensure_chunk_index", None)
    if callable(ensure_chunk_index):
        try:
            chunk_index_instance = ensure_chunk_index()
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Unable to initialize chunk index", exc_info=True)

    def _record_failure(name: str, exc: Exception) -> None:
        failures.append(ToolRegistrationFailure(name=name, error=exc))
        LOGGER.warning("Failed to register tool %s: %s", name, exc, exc_info=True)

    def _safe_register(name: str, tool: Any, **kwargs: Any) -> None:
        try:
            register(name, tool, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging around controller wiring
            _record_failure(name, exc)
        else:
            registered.append(name)

    with _graph_batch_guard(controller):
        try:
            snapshot_tool = DocumentSnapshotTool(
                provider=context.bridge,
                outline_digest_resolver=context.outline_digest_resolver,
                chunk_index=chunk_index_instance,
            )
        except Exception as exc:  # pragma: no cover - rare failure constructing tool
            _record_failure("document_snapshot", exc)
        else:
            _safe_register(
                "document_snapshot",
                snapshot_tool,
                description=(
                    "Return the freshest document snapshot (text, metadata, diff summaries) for the active or specified tab."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "delta_only": {
                            "type": "boolean",
                            "description": "When true, include only the selection and surrounding context instead of the full document.",
                        },
                        "include_diff": {
                            "type": "boolean",
                            "description": "Attach the most recent diff summary when available (default true).",
                        },
                        "tab_id": {
                            "type": "string",
                            "description": "Target a specific tab instead of the active document.",
                        },
                        "source_tab_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of additional tab snapshots to gather alongside the active document.",
                        },
                        "include_open_documents": {
                            "type": "boolean",
                            "description": "When true, include a summary of all open documents in the snapshot payload.",
                        },
                        "window": {
                            "description": (
                                "Controls which portion of the document is returned. Defaults to a selection-centered window; set to 'document' to request the full text."
                            ),
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "kind": {"type": "string"},
                                        "padding": {"type": "integer", "minimum": 0},
                                        "max_chars": {"type": "integer", "minimum": 256},
                                        "max_tokens": {"type": "integer", "minimum": 256},
                                        "start": {"type": "integer", "minimum": 0},
                                        "end": {"type": "integer", "minimum": 0},
                                    },
                                    "additionalProperties": False,
                                },
                            ],
                        },
                        "chunk_profile": {
                            "type": "string",
                            "enum": ["auto", "prose", "code", "notes"],
                            "description": "Hint for chunk manifest sizing. Defaults to 'auto'.",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "minimum": 256,
                            "description": "Approximate token budget for the returned window; helps keep snapshots slim.",
                        },
                        "include_text": {
                            "type": "boolean",
                            "description": "Set false to omit the text body and request only metadata + chunk manifests.",
                        },
                    },
                    "additionalProperties": False,
                },
            )

        if chunk_index_instance is not None:
            try:
                chunk_tool = DocumentChunkTool(
                    bridge=context.bridge,
                    chunk_index=chunk_index_instance,
                    chunk_config_resolver=chunk_settings_resolver,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                _record_failure("document_chunk", exc)
            else:
                _safe_register(
                    "document_chunk",
                    chunk_tool,
                    description=(
                        "Return a specific chunk (or iterator) referenced by document_snapshot manifests, enforcing inline token caps."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "string",
                                "description": "Chunk identifier returned by document_snapshot.chunk_manifest[].id.",
                            },
                            "document_id": {
                                "type": "string",
                                "description": "Document identifier associated with the manifest entry.",
                            },
                            "cache_key": {
                                "type": "string",
                                "description": "Chunk manifest cache key; ensures the request matches the cached window.",
                            },
                            "version": {
                                "type": "string",
                                "description": "Optional version token (document_id:version_id:hash) for freshness checks.",
                            },
                            "include_text": {
                                "type": "boolean",
                                "description": "Set false to fetch only metadata/pointers without inline text.",
                            },
                            "max_tokens": {
                                "type": "integer",
                                "minimum": 256,
                                "description": "Override the inline token cap for this request (defaults to runtime setting).",
                            },
                            "iterator": {
                                "type": "object",
                                "description": "Iterate sequential chunks using a manifest cache key.",
                                "properties": {
                                    "cache_key": {
                                        "type": "string",
                                        "description": "Manifest cache key to iterate over (chunk_manifest.cache_key).",
                                    },
                                    "document_id": {
                                        "type": "string",
                                        "description": "Optional document identifier to disambiguate cache keys.",
                                    },
                                    "start_chunk_id": {
                                        "type": "string",
                                        "description": "Chunk ID to start from; omit to begin at the first chunk.",
                                    },
                                    "limit": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 16,
                                        "description": "Maximum number of chunks to return in this batch (bounded by runtime settings).",
                                    },
                                },
                                "required": ["cache_key"],
                                "additionalProperties": False,
                            },
                        },
                        "anyOf": [
                            {"required": ["chunk_id"]},
                            {"required": ["iterator"]},
                        ],
                        "additionalProperties": False,
                    },
                )

        if context.phase3_outline_enabled:
            registered.extend(
                register_phase3_tools(context, register_fn=register, failures=failures)
            )
        if context.plot_scaffolding_enabled:
            registered.extend(
                register_plot_memory_tools(context, register_fn=register, failures=failures)
            )
            registered.extend(
                register_character_map_tool(context, register_fn=register, failures=failures)
            )
            registered.extend(
                register_character_planner_tool(context, register_fn=register, failures=failures)
            )

        try:
            edit_tool = DocumentEditTool(bridge=context.bridge, patch_only=True)
        except Exception as exc:  # pragma: no cover - tool construction rarely fails
            _record_failure("document_edit", exc)
            edit_tool = None
        if edit_tool is not None:
            _safe_register(
                "document_edit",
                edit_tool,
                description=(
                    "Apply a structured edit directive (insert, replace, annotate, or unified diff patch) against the active document."
                ),
                parameters=context.directive_schema_provider(),
            )

            try:
                apply_patch_tool = DocumentApplyPatchTool(bridge=context.bridge, edit_tool=edit_tool)
            except Exception as exc:
                _record_failure("document_apply_patch", exc)
            else:
                if context.auto_patch_consumer is not None:
                    try:
                        context.auto_patch_consumer(apply_patch_tool)
                    except Exception as exc:  # pragma: no cover - defensive; consumer should not raise
                        _record_failure("document_apply_patch", exc)
                        apply_patch_tool = None
                if apply_patch_tool is not None:
                    _safe_register(
                        "document_apply_patch",
                        apply_patch_tool,
                        description=(
                            "Replace a target_range with new content by automatically building and applying a unified diff."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Replacement text that should occupy the specified target_range.",
                                },
                                "target_range": DIRECTIVE_SCHEMA["properties"]["target_range"],
                                "document_version": {
                                    "type": "string",
                                    "description": "Document snapshot version captured before drafting the edit.",
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "Optional explanation stored alongside the edit directive.",
                                },
                                "context_lines": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Override the number of context lines included in the generated diff.",
                                },
                                "tab_id": {
                                    "type": "string",
                                    "description": "Optional tab identifier; defaults to the active tab when omitted.",
                                },
                            },
                            "required": ["content"],
                            "additionalProperties": False,
                        },
                    )

        try:
            tab_listing_tool = ListTabsTool(provider=context.bridge)
        except Exception as exc:
            _record_failure("list_tabs", exc)
        else:
            _safe_register(
                "list_tabs",
                tab_listing_tool,
                description="Enumerate the open tabs (tab_id, title, path, dirty) so agents can target specific documents.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )

        try:
            diff_tool = DiffBuilderTool()
        except Exception as exc:
            _record_failure("diff_builder", exc)
        else:
            _safe_register(
                "diff_builder",
                diff_tool,
                description=(
                    "Return a unified diff given original and updated text snippets to drive patch directives."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "original": {
                            "type": "string",
                            "description": "The prior version of the text block.",
                        },
                        "updated": {
                            "type": "string",
                            "description": "The revised text block.",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Optional virtual filename used in diff headers.",
                        },
                        "context": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Number of context lines to include in the diff (default 3).",
                        },
                    },
                    "required": ["original", "updated"],
                    "additionalProperties": False,
                },
            )

        try:
            search_tool = SearchReplaceTool(bridge=context.bridge)
        except Exception as exc:
            _record_failure("search_replace", exc)
        else:
            _safe_register(
                "search_replace",
                search_tool,
                description=(
                    "Search the current document or selection and optionally apply replacements with regex/literal matching."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Text or regex pattern to find.",
                        },
                        "replacement": {
                            "type": "string",
                            "description": "Content that will replace each match.",
                        },
                        "is_regex": {
                            "type": "boolean",
                            "description": "Interpret the pattern as a regular expression.",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["document", "selection"],
                            "description": "Limit replacements to the entire document or just the current selection.",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "When true, do not apply edits—only preview the outcome.",
                        },
                        "max_replacements": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional cap on the number of replacements to perform.",
                        },
                        "match_case": {
                            "type": "boolean",
                            "description": "Respect character casing when matching (defaults to true).",
                        },
                        "whole_word": {
                            "type": "boolean",
                            "description": "Only match full words when true.",
                        },
                    },
                    "required": ["pattern", "replacement"],
                    "additionalProperties": False,
                },
            )

        advisor_runner = getattr(controller, "_advisor_tool_entrypoint", None)
        if callable(advisor_runner):
            try:
                advisor_tool = ToolUsageAdvisorTool(advisor_runner)
            except Exception as exc:  # pragma: no cover - defensive
                _record_failure("tool_usage_advisor", exc)
            else:
                _safe_register(
                    "tool_usage_advisor",
                    advisor_tool,
                    description=(
                        "Re-run the preflight analysis pipeline to retrieve up-to-date tool recommendations and warnings."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "Optional explicit target document identifier; defaults to the active tab.",
                            },
                            "selection_start": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Override the selection start used when building analysis inputs.",
                            },
                            "selection_end": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Override the selection end used when building analysis inputs.",
                            },
                            "force_refresh": {
                                "type": "boolean",
                                "description": "Bypass cached advice and recompute from scratch when true.",
                            },
                            "snapshot": {
                                "type": "object",
                                "description": "Optional snapshot payload previously returned by DocumentSnapshotTool.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Operator-provided note recorded with telemetry for this analysis run.",
                            },
                        },
                        "additionalProperties": False,
                    },
                )

        _safe_register(
            "validate_snippet",
            validate_snippet,
            description="Validate YAML/JSON snippets before inserting them into the document.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Snippet contents that should be validated.",
                    },
                    "fmt": {
                        "type": "string",
                        "description": "Declared format of the snippet.",
                        "enum": ["yaml", "yml", "json", "markdown", "md"],
                    },
                },
                "required": ["text", "fmt"],
                "additionalProperties": False,
            },
        )

    if registered:
        LOGGER.debug("Default AI tools registered: %s", ", ".join(registered))
    if failures:
        raise ToolRegistrationError(failures)


def register_phase3_tools(
    context: ToolRegistryContext,
    *,
    register_fn: Callable[..., Any] | None = None,
    failures: list[ToolRegistrationFailure] | None = None,
) -> list[str]:
    """Register outline + retrieval tools when phase3 support is enabled."""

    if not context.phase3_outline_enabled:
        return []

    controller = context.controller
    if controller is None:
        return []
    register = register_fn or getattr(controller, "register_tool", None)
    if not callable(register):
        LOGGER.debug("AI controller does not expose register_tool; skipping phase3 tool wiring.")
        return []

    registered: list[str] = []
    outline_factory = context.ensure_outline_tool
    if outline_factory is not None:
        try:
            outline_tool = outline_factory()
        except Exception as exc:
            if failures is not None:
                failures.append(ToolRegistrationFailure(name="document_outline", error=exc))
            LOGGER.warning("Failed to build document_outline tool: %s", exc, exc_info=True)
            outline_tool = None
        if outline_tool is not None:
            try:
                register(
                    "document_outline",
                    outline_tool,
                    description=(
                        "Return the most recent outline for the active document, including pointer IDs, blurbs, and staleness metadata."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "Optional explicit document identifier; defaults to the active tab.",
                            },
                            "desired_levels": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Limit the outline depth to this heading level before budgeting.",
                            },
                            "include_blurbs": {
                                "type": "boolean",
                                "description": "When false, omit excerpt blurbs to conserve tokens.",
                            },
                            "max_nodes": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 1000,
                                "description": "Cap the number of nodes returned prior to budget enforcement.",
                            },
                        },
                        "additionalProperties": False,
                    },
                )
            except Exception as exc:
                if failures is not None:
                    failures.append(ToolRegistrationFailure(name="document_outline", error=exc))
                LOGGER.warning("Failed to register document_outline: %s", exc, exc_info=True)
            else:
                registered.append("document_outline")

    find_sections_factory = context.ensure_find_sections_tool
    if find_sections_factory is not None:
        try:
            find_sections_tool = find_sections_factory()
        except Exception as exc:
            if failures is not None:
                failures.append(ToolRegistrationFailure(name="document_find_sections", error=exc))
            LOGGER.warning("Failed to build document_find_sections tool: %s", exc, exc_info=True)
            find_sections_tool = None
        if find_sections_tool is not None:
            try:
                register(
                    "document_find_sections",
                    find_sections_tool,
                    description=(
                        "Return the best-matching document chunks for a natural language query using embeddings or fallback heuristics."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "Optional target document identifier; defaults to the active document.",
                            },
                            "query": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Natural-language description of the section(s) to find.",
                            },
                            "top_k": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 12,
                                "description": "Maximum number of pointers to return.",
                            },
                            "min_confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Minimum embedding match score before falling back to heuristics.",
                            },
                            "filters": {
                                "type": "object",
                                "description": "Optional additional filtering hints understood by custom index providers.",
                            },
                            "include_outline_context": {
                                "type": "boolean",
                                "description": "When true, include nearby outline headings for each pointer.",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                )
            except Exception as exc:
                if failures is not None:
                    failures.append(ToolRegistrationFailure(name="document_find_sections", error=exc))
                LOGGER.warning("Failed to register document_find_sections: %s", exc, exc_info=True)
            else:
                registered.append("document_find_sections")

    return registered


def unregister_phase3_tools(controller: Any) -> None:
    """Remove phase3 outline/retrieval tools when disabled."""

    if controller is None:
        return
    unregister = getattr(controller, "unregister_tool", None)
    if not callable(unregister):
        return
    for name in ("document_outline", "document_find_sections"):
        try:
            unregister(name)
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("Failed to unregister tool %s", name, exc_info=True)


def register_plot_memory_tools(
    context: ToolRegistryContext,
    *,
    register_fn: Callable[..., Any] | None = None,
    failures: list[ToolRegistrationFailure] | None = None,
) -> list[str]:
    """Register plot outline + state update tools when plot scaffolding is enabled."""

    if not context.plot_scaffolding_enabled:
        return []

    controller = context.controller
    if controller is None:
        return []
    register = register_fn or getattr(controller, "register_tool", None)
    if not callable(register):
        LOGGER.debug("AI controller does not expose register_tool; skipping plot-memory wiring.")
        return []

    registered: list[str] = []
    outline_factory = context.ensure_plot_outline_tool or context.ensure_plot_state_tool
    update_factory = context.ensure_plot_state_update_tool

    outline_schema = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "Optional explicit target; defaults to the active document.",
            },
            "include_entities": {
                "type": "boolean",
                "description": "When false, omit entity payloads to conserve tokens.",
            },
            "include_arcs": {
                "type": "boolean",
                "description": "When false, omit plot arc beats from the response.",
            },
            "include_overrides": {
                "type": "boolean",
                "description": "When false, omit persisted overrides to save tokens.",
            },
            "include_dependencies": {
                "type": "boolean",
                "description": "When false, omit dependency records to save tokens.",
            },
            "max_entities": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Limit the number of entities returned before budgeting.",
            },
            "max_beats": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Limit the number of beats returned per arc before budgeting.",
            },
            "max_overrides": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Limit how many overrides are returned before budgeting.",
            },
            "max_dependencies": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Limit how many dependency entries are returned before budgeting.",
            },
        },
        "additionalProperties": False,
    }

    arc_schema = {
        "type": "object",
        "properties": {
            "arc_id": {"type": "string", "description": "Stable arc identifier (slug)."},
            "name": {"type": "string", "description": "Human-readable arc label."},
            "summary": {"type": "string", "description": "Short description of the arc state."},
            "beats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "beat_id": {"type": "string"},
                        "summary": {"type": "string"},
                        "pointer_id": {"type": "string"},
                        "chunk_hash": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary"],
        "additionalProperties": False,
    }

    override_schema = {
        "type": "object",
        "properties": {
            "override_id": {"type": "string"},
            "summary": {"type": "string"},
            "arc_id": {"type": "string"},
            "beat_id": {"type": "string"},
            "author": {"type": "string"},
            "metadata": {"type": "object"},
        },
        "required": ["summary"],
        "additionalProperties": False,
    }

    dependency_schema = {
        "type": "object",
        "properties": {
            "source_arc_id": {"type": "string"},
            "target_arc_id": {"type": "string"},
            "kind": {"type": "string"},
            "summary": {"type": "string"},
            "metadata": {"type": "object"},
        },
        "required": ["source_arc_id", "target_arc_id"],
        "additionalProperties": False,
    }

    update_schema = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "Optional explicit target; defaults to the active document.",
            },
            "version_id": {"type": "string", "description": "Document version identifier for bookkeeping."},
            "arcs": {"type": "array", "items": arc_schema},
            "overrides": {"type": "array", "items": override_schema},
            "dependencies": {"type": "array", "items": dependency_schema},
            "remove_override_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of override IDs to delete before returning the summary.",
            },
            "note": {
                "type": "string",
                "description": "Optional free-form note persisted with version metadata.",
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata stored alongside the version record.",
            },
        },
        "additionalProperties": False,
    }

    def _record_failure(name: str, exc: Exception) -> None:
        if failures is not None:
            failures.append(ToolRegistrationFailure(name=name, error=exc))
        LOGGER.warning("Failed to register plot tool %s: %s", name, exc, exc_info=True)

    if outline_factory is not None:
        try:
            outline_tool = outline_factory()
        except Exception as exc:
            _record_failure("plot_outline", exc)
            outline_tool = None
        if outline_tool is not None:
            for name in ("plot_outline", "document_plot_state"):
                try:
                    register(
                        name,
                        outline_tool,
                        description=(
                            "Return cached character/entity scaffolding, arcs, overrides, and dependencies built from plot scouting runs."
                        ),
                        parameters=outline_schema,
                    )
                except Exception as exc:
                    _record_failure(name, exc)
                else:
                    registered.append(name)

    if update_factory is not None:
        try:
            update_tool = update_factory()
        except Exception as exc:
            _record_failure("plot_state_update", exc)
            update_tool = None
        if update_tool is not None:
            try:
                register(
                    "plot_state_update",
                    update_tool,
                    description=(
                        "Persist arc/beat adjustments, overrides, and dependencies so plot scaffolding stays in sync after edits."
                    ),
                    parameters=update_schema,
                )
            except Exception as exc:
                _record_failure("plot_state_update", exc)
            else:
                registered.append("plot_state_update")

    return registered


def unregister_plot_state_tool(controller: Any) -> None:
    """Remove plot-memory tools when the feature flag is disabled."""

    if controller is None:
        return
    unregister = getattr(controller, "unregister_tool", None)
    if not callable(unregister):
        return
    for name in ("plot_outline", "document_plot_state", "plot_state_update"):
        try:
            unregister(name)
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("Failed to unregister %s", name, exc_info=True)


# Backwards-compatible aliases for legacy import sites.
register_plot_state_tool = register_plot_memory_tools


def register_character_map_tool(
    context: ToolRegistryContext,
    *,
    register_fn: Callable[..., Any] | None = None,
    failures: list[ToolRegistrationFailure] | None = None,
) -> list[str]:
    """Register the character map tool when concordance is enabled."""

    if not context.plot_scaffolding_enabled:
        return []

    controller = context.controller
    if controller is None:
        return []
    register = register_fn or getattr(controller, "register_tool", None)
    if not callable(register):
        LOGGER.debug("AI controller does not expose register_tool; skipping character-map wiring.")
        return []

    tool_factory = context.ensure_character_map_tool
    if tool_factory is None:
        return []

    try:
        tool = tool_factory()
    except Exception as exc:
        if failures is not None:
            failures.append(ToolRegistrationFailure(name="character_map", error=exc))
        LOGGER.warning("Failed to build CharacterMapTool: %s", exc, exc_info=True)
        return []

    if tool is None:
        return []

    try:
        register(
            "character_map",
            tool,
            description=(
                "Return cached character/entity concordance data—aliases, pronouns, and mention snippets—"
                "built from recent chunk summaries."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Optional explicit target; defaults to the active document.",
                    },
                    "include_mentions": {
                        "type": "boolean",
                        "description": "When false, omit mention snippets to conserve tokens.",
                    },
                    "include_stats": {
                        "type": "boolean",
                        "description": "Include ingestion counters and cache metadata in the response.",
                    },
                    "max_entities": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Limit how many entities are returned before budgeting.",
                    },
                    "max_mentions": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "description": "Limit the number of mention entries per entity.",
                    },
                },
                "additionalProperties": False,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        if failures is not None:
            failures.append(ToolRegistrationFailure(name="character_map", error=exc))
        LOGGER.warning("Failed to register CharacterMapTool: %s", exc, exc_info=True)
        return []

    return ["character_map"]


def unregister_character_map_tool(controller: Any) -> None:
    """Remove the character map tool when the feature flag is disabled."""

    if controller is None:
        return
    unregister = getattr(controller, "unregister_tool", None)
    if not callable(unregister):
        return
    try:
        unregister("character_map")
    except Exception:  # pragma: no cover - defensive
        LOGGER.debug("Failed to unregister character_map", exc_info=True)


def register_character_planner_tool(
    context: ToolRegistryContext,
    *,
    register_fn: Callable[..., Any] | None = None,
    failures: list[ToolRegistrationFailure] | None = None,
) -> list[str]:
    """Register the character edit planner when concordance is enabled."""

    if not context.plot_scaffolding_enabled:
        return []

    controller = context.controller
    if controller is None:
        return []
    register = register_fn or getattr(controller, "register_tool", None)
    if not callable(register):
        LOGGER.debug("AI controller does not expose register_tool; skipping character-planner wiring.")
        return []

    tool_factory = context.ensure_character_planner_tool
    if tool_factory is None:
        return []

    try:
        tool = tool_factory()
    except Exception as exc:
        if failures is not None:
            failures.append(ToolRegistrationFailure(name="character_edit_planner", error=exc))
        LOGGER.warning("Failed to build CharacterEditPlannerTool: %s", exc, exc_info=True)
        return []

    if tool is None:
        return []

    try:
        register(
            "character_edit_planner",
            tool,
            description=(
                "Plan document-wide character edits by listing unresolved mentions and marking tasks complete as chunks are updated."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Optional explicit target; defaults to the active document.",
                    },
                    "targets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict the planner to specific entity_ids or alias/canonical names.",
                    },
                    "completed_task_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Mark these task identifiers as completed before returning the refreshed plan.",
                    },
                    "notes": {
                        "type": "object",
                        "description": "Optional map of task_id to completion note stored with planner progress.",
                        "additionalProperties": {"type": "string"},
                    },
                    "include_completed": {
                        "type": "boolean",
                        "description": "When false, omit completed tasks from the response payload.",
                    },
                    "max_entities": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Clamp how many entities are inspected before budgeting.",
                    },
                    "max_mentions": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "description": "Limit the number of mentions scanned per entity before budgeting.",
                    },
                    "max_tasks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Limit the number of tasks returned per invocation.",
                    },
                },
                "additionalProperties": False,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        if failures is not None:
            failures.append(ToolRegistrationFailure(name="character_edit_planner", error=exc))
        LOGGER.warning("Failed to register CharacterEditPlannerTool: %s", exc, exc_info=True)
        return []

    return ["character_edit_planner"]


def unregister_character_planner_tool(controller: Any) -> None:
    """Remove the character planner tool when the feature flag is disabled."""

    if controller is None:
        return
    unregister = getattr(controller, "unregister_tool", None)
    if not callable(unregister):
        return
    try:
        unregister("character_edit_planner")
    except Exception:  # pragma: no cover - defensive
        LOGGER.debug("Failed to unregister character_edit_planner", exc_info=True)


def _graph_batch_guard(controller: Any) -> AbstractContextManager[None]:
    suspend = getattr(controller, "suspend_graph_rebuilds", None)
    if callable(suspend):
        return cast(AbstractContextManager[None], suspend())
    return nullcontext()


__all__ = [
    "ToolRegistryContext",
    "ToolRegistrationFailure",
    "ToolRegistrationError",
    "register_default_tools",
    "register_phase3_tools",
    "unregister_phase3_tools",
    "register_plot_memory_tools",
    "register_plot_state_tool",
    "unregister_plot_state_tool",
    "register_character_map_tool",
    "unregister_character_map_tool",
    "register_character_planner_tool",
    "unregister_character_planner_tool",
]
