"""Tool that builds and applies unified diffs in a single call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .diff_builder import DiffBuilderTool
from .document_edit import DocumentEditTool, Bridge as EditBridge
from .document_snapshot import SnapshotProvider


class PatchBridge(EditBridge, SnapshotProvider, Protocol):
    """Protocol describing the bridge required by DocumentApplyPatchTool."""

    ...


@dataclass(slots=True)
class DocumentApplyPatchTool:
    """Build a diff from the live snapshot and route it through DocumentEdit."""

    bridge: PatchBridge
    edit_tool: DocumentEditTool
    diff_builder: DiffBuilderTool = field(default_factory=DiffBuilderTool)
    filename_fallback: str = "document.txt"

    def run(
        self,
        *,
        content: str,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None = None,
        document_version: str | None = None,
        rationale: str | None = None,
        context_lines: int = 3,
    ) -> str:
        snapshot = dict(self.bridge.generate_snapshot(delta_only=False))
        base_text = snapshot.get("text", "")
        if not isinstance(base_text, str):
            raise ValueError("Snapshot did not provide document text")

        selection_tuple = snapshot.get("selection") or (0, 0)
        start, end = self._resolve_range(target_range, selection_tuple, len(base_text))
        new_text = str(content)
        if base_text[start:end] == new_text:
            return "skipped: content already matches selection"

        updated_text = base_text[:start] + new_text + base_text[end:]
        filename = str(snapshot.get("path") or self.filename_fallback)
        diff = self.diff_builder.run(
            base_text,
            updated_text,
            filename=filename,
            context=max(0, int(context_lines)),
        )
        version = self._resolve_version(snapshot, document_version)
        payload: dict[str, Any] = {
            "action": "patch",
            "diff": diff,
            "document_version": version,
        }
        if rationale is not None:
            payload["rationale"] = rationale
        return self.edit_tool.run(**payload)

    def _resolve_range(
        self,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        selection: Sequence[int],
        length: int,
    ) -> tuple[int, int]:
        if target_range is None:
            start, end = selection if len(selection) == 2 else (0, 0)
        elif isinstance(target_range, Mapping):
            start = int(target_range.get("start", 0))
            end = int(target_range.get("end", 0))
        elif isinstance(target_range, Sequence) and len(target_range) == 2:
            start = int(target_range[0])
            end = int(target_range[1])
        else:
            raise ValueError("target_range must be a [start, end] sequence or {'start','end'} mapping")

        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end < start:
            start, end = end, start
        return start, end

    def _resolve_version(self, snapshot: Mapping[str, Any], explicit: str | None) -> str:
        snapshot_version = snapshot.get("version") or getattr(self.bridge, "last_snapshot_version", None)
        candidate = explicit or snapshot_version
        if not candidate:
            raise ValueError("Document version is required; call document_snapshot before applying edits")
        candidate_text = str(candidate).strip()
        if not candidate_text:
            raise ValueError("Document version is required; call document_snapshot before applying edits")
        if explicit and candidate_text != str(snapshot_version).strip():
            raise ValueError("Provided document_version does not match the latest snapshot; refresh first")
        return candidate_text


__all__ = ["DocumentApplyPatchTool"]
