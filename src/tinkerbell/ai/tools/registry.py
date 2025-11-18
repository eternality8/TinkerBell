"""Declarative registration helpers for AI tools."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, cast

from ...chat.commands import DIRECTIVE_SCHEMA
from .diff_builder import DiffBuilderTool
from .document_apply_patch import DocumentApplyPatchTool
from .document_edit import DocumentEditTool
from .document_find_sections import DocumentFindSectionsTool
from .document_outline import DocumentOutlineTool
from .document_plot_state import DocumentPlotStateTool
from .document_snapshot import DocumentSnapshotTool
from .list_tabs import ListTabsTool
from .search_replace import SearchReplaceTool
from .validation import validate_snippet

LOGGER = logging.getLogger(__name__)


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
    ensure_plot_state_tool: Callable[[], DocumentPlotStateTool | None] | None = None
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

    try:
        with _graph_batch_guard(controller):
            snapshot_tool = DocumentSnapshotTool(
                provider=context.bridge,
                outline_digest_resolver=context.outline_digest_resolver,
            )
            register(
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
                    },
                    "additionalProperties": False,
                },
            )

            registered = ["document_snapshot"]
            if context.phase3_outline_enabled:
                registered.extend(register_phase3_tools(context, register_fn=register))
            if context.plot_scaffolding_enabled:
                registered.extend(register_plot_state_tool(context, register_fn=register))

            edit_tool = DocumentEditTool(bridge=context.bridge, patch_only=True)
            register(
                "document_edit",
                edit_tool,
                description=(
                    "Apply a structured edit directive (insert, replace, annotate, or unified diff patch) against the active document."
                ),
                parameters=context.directive_schema_provider(),
            )

            apply_patch_tool = DocumentApplyPatchTool(bridge=context.bridge, edit_tool=edit_tool)
            if context.auto_patch_consumer is not None:
                context.auto_patch_consumer(apply_patch_tool)
            register(
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

            tab_listing_tool = ListTabsTool(provider=context.bridge)
            register(
                "list_tabs",
                tab_listing_tool,
                description="Enumerate the open tabs (tab_id, title, path, dirty) so agents can target specific documents.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )

            diff_tool = DiffBuilderTool()
            register(
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

            search_tool = SearchReplaceTool(bridge=context.bridge)
            register(
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

        register(
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

        registered.extend(
            [
                "document_edit",
                "document_apply_patch",
                "list_tabs",
                "diff_builder",
                "search_replace",
                "validate_snippet",
            ]
        )
        LOGGER.debug("Default AI tools registered: %s", ", ".join(registered))
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to register default AI tools: %s", exc)


def register_phase3_tools(
    context: ToolRegistryContext,
    *,
    register_fn: Callable[..., Any] | None = None,
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
        outline_tool = outline_factory()
        if outline_tool is not None:
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
            registered.append("document_outline")

    find_sections_factory = context.ensure_find_sections_tool
    if find_sections_factory is not None:
        find_sections_tool = find_sections_factory()
        if find_sections_tool is not None:
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


def register_plot_state_tool(
    context: ToolRegistryContext,
    *,
    register_fn: Callable[..., Any] | None = None,
) -> list[str]:
    """Register the plot state tool when the feature flag is enabled."""

    if not context.plot_scaffolding_enabled:
        return []

    controller = context.controller
    if controller is None:
        return []
    register = register_fn or getattr(controller, "register_tool", None)
    if not callable(register):
        LOGGER.debug("AI controller does not expose register_tool; skipping plot-state wiring.")
        return []

    tool_factory = context.ensure_plot_state_tool
    if tool_factory is None:
        return []

    tool = tool_factory()
    if tool is None:
        return []

    try:
        register(
            "document_plot_state",
            tool,
            description=(
                "Return cached character/entity scaffolding and plot arcs extracted from recent subagent runs."
            ),
            parameters={
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
                },
                "additionalProperties": False,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to register DocumentPlotStateTool: %s", exc)
        return []

    return ["document_plot_state"]


def unregister_plot_state_tool(controller: Any) -> None:
    """Remove the plot-state tool when the feature flag is turned off."""

    if controller is None:
        return
    unregister = getattr(controller, "unregister_tool", None)
    if not callable(unregister):
        return
    try:
        unregister("document_plot_state")
    except Exception:  # pragma: no cover - defensive
        LOGGER.debug("Failed to unregister document_plot_state", exc_info=True)


def _graph_batch_guard(controller: Any) -> AbstractContextManager[None]:
    suspend = getattr(controller, "suspend_graph_rebuilds", None)
    if callable(suspend):
        return cast(AbstractContextManager[None], suspend())
    return nullcontext()


__all__ = [
    "ToolRegistryContext",
    "register_default_tools",
    "register_phase3_tools",
    "unregister_phase3_tools",
    "register_plot_state_tool",
    "unregister_plot_state_tool",
]
