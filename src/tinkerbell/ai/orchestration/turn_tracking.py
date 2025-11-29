"""Turn tracking classes for the AI controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class SnapshotRefreshTracker:
    """Warns when the agent applies too many edits without a fresh snapshot.
    
    Recognizes both legacy and new tool names:
    - document_snapshot / read_document
    - document_apply_patch, document_edit / replace_lines, insert_lines, delete_lines
    """

    document_id: str | None
    threshold: int
    edits_since_snapshot: int = 0
    last_snapshot_version: str | None = None
    warning_active: bool = False

    # Tool name sets for matching (legacy + new names)
    _SNAPSHOT_TOOLS: frozenset[str] = frozenset({"document_snapshot", "read_document"})
    _EDIT_TOOLS: frozenset[str] = frozenset({
        "document_apply_patch", "document_edit",
        "replace_lines", "insert_lines", "delete_lines", "write_document",
    })

    def __post_init__(self) -> None:
        try:
            value = int(self.threshold)
        except (TypeError, ValueError):  # pragma: no cover - defensive cast
            value = 1
        self.threshold = max(1, value)

    def observe_tool(self, record: Mapping[str, Any], payload: Mapping[str, Any] | None) -> list[str] | None:
        name = str(record.get("name") or "").lower()
        status = str(record.get("status") or "ok").lower()
        succeeded = status == "ok"
        if name in self._SNAPSHOT_TOOLS:
            if succeeded:
                self._note_snapshot(payload)
            return None
        if name not in self._EDIT_TOOLS or not succeeded:
            return None
        self.edits_since_snapshot += 1
        if not self.warning_active and self.edits_since_snapshot >= self.threshold:
            self.warning_active = True
            return self._warning_lines()
        return None

    def _note_snapshot(self, payload: Mapping[str, Any] | None) -> None:
        self.edits_since_snapshot = 0
        self.warning_active = False
        self.last_snapshot_version = self._extract_version(payload)

    def _extract_version(self, payload: Mapping[str, Any] | None) -> str | None:
        if not isinstance(payload, Mapping):
            return None
        for key in ("version", "document_version", "version_id"):
            value = payload.get(key)
            if value in (None, ""):
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _warning_lines(self) -> list[str]:
        target = self.document_id or "this document"
        version = self.last_snapshot_version or "unknown"
        return [
            f"{self.edits_since_snapshot} edits have landed on {target} since the last read_document (version {version}).",
            "Offsets drift after multiple edits off a single snapshot. Call read_document to capture a fresh span before drafting another edit and narrate the refresh to the user.",
        ]


@dataclass(slots=True)
class PlotLoopTracker:
    """Ensures the agent follows the outline → edit → update contract.
    
    Tool name mapping (legacy → new):
    - plot_outline → get_outline
    - document_plot_state → analyze_document
    - plot_state_update → transform_document
    - document_apply_patch, document_edit → replace_lines, insert_lines, delete_lines
    """

    document_id: str | None
    outline_called: bool = False
    pending_update: bool = False
    blocked_edits: int = 0
    snapshot_prompt_pending: bool = False

    # Tool name sets for matching
    _OUTLINE_TOOLS: frozenset[str] = frozenset({
        "get_outline", "analyze_document",
        # Legacy names for backward compatibility
        "plot_outline", "document_plot_state",
    })
    _EDIT_TOOLS: frozenset[str] = frozenset({
        "replace_lines", "insert_lines", "delete_lines", "write_document",
        # Legacy names for backward compatibility
        "document_apply_patch", "document_edit",
    })
    _UPDATE_TOOLS: frozenset[str] = frozenset({
        "transform_document",
        # Legacy name for backward compatibility
        "plot_state_update",
    })

    def before_tool(self, tool_name: str | None) -> str | None:
        name = (tool_name or "").strip().lower()
        if name in self._EDIT_TOOLS and not self.outline_called:
            self.blocked_edits += 1
            self.snapshot_prompt_pending = True
            return (
                "Plot loop guardrail: call get_outline or analyze_document for continuity context before applying edits."
            )
        return None

    def observe_tool(self, record: Mapping[str, Any], payload: Mapping[str, Any] | None) -> list[str] | None:
        name = str(record.get("name") or "").lower()
        status = str(record.get("status") or "ok").lower()
        succeeded = status == "ok"
        if name in self._OUTLINE_TOOLS and succeeded:
            self.outline_called = True
            if self.snapshot_prompt_pending:
                self.snapshot_prompt_pending = False
                target = self.document_id or "this document"
                return [
                    f"Plot loop guardrail satisfied for {target}: capture a fresh read_document before drafting your next edit so offsets stay in sync.",
                ]
            return None
        if name in self._UPDATE_TOOLS:
            if succeeded and self.pending_update:
                self.pending_update = False
                return [
                    "transform_document received your changes. You may proceed to the next edit after reading the outline if needed.",
                ]
            if succeeded:
                self.pending_update = False
            return None
        if name in self._EDIT_TOOLS and succeeded:
            self.pending_update = True
            return [
                "Plot loop reminder: call transform_document to log the changes you just applied so downstream agents stay in sync.",
            ]
        return None

    def needs_update_prompt(self) -> bool:
        return self.pending_update

    def update_prompt(self) -> str:
        target = self.document_id or "this document"
        return (
            f"Plot loop requirement: run transform_document for {target} before finishing this turn so plot scaffolding reflects your edits."
        )


__all__ = [
    "SnapshotRefreshTracker",
    "PlotLoopTracker",
]
