"""Tool that builds and applies unified diffs in a single call."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, cast

from .diff_builder import DiffBuilderTool
from .document_edit import DocumentEditTool, Bridge as EditBridge
from .document_snapshot import SnapshotProvider
from ...services.bridge import DocumentVersionMismatchError


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
    default_context_lines: int = 5

    def run(
        self,
        *,
        content: str,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None = None,
        document_version: str | None = None,
        rationale: str | None = None,
        context_lines: int | None = None,
        tab_id: str | None = None,
    ) -> str:
        snapshot = dict(self._generate_snapshot(tab_id=tab_id))
        base_text = snapshot.get("text", "")
        if not isinstance(base_text, str):
            raise ValueError("Snapshot did not provide document text")

        selection_tuple = snapshot.get("selection") or (0, 0)
        start, end = self._resolve_range(target_range, selection_tuple, len(base_text))
        new_text = str(content)
        if base_text[start:end] == new_text:
            return "skipped: content already matches selection"

        updated_text = base_text[:start] + new_text + base_text[end:]
        filename = self._normalize_filename(snapshot)
        diff = self.diff_builder.run(
            base_text,
            updated_text,
            filename=filename,
            context=context_lines if context_lines is not None else self.default_context_lines,
        )
        version = self._resolve_version(snapshot, document_version, tab_id=tab_id)
        content_hash = self._resolve_content_hash(snapshot, base_text)
        payload: dict[str, Any] = {
            "action": "patch",
            "diff": diff,
            "document_version": version,
            "content_hash": content_hash,
        }
        if rationale is not None:
            payload["rationale"] = rationale
        return self.edit_tool.run(tab_id=tab_id, **payload)

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

    def _resolve_version(self, snapshot: Mapping[str, Any], explicit: str | None, *, tab_id: str | None) -> str:
        snapshot_version = snapshot.get("version") or self._last_snapshot_version(tab_id)
        candidate = explicit or snapshot_version
        if not candidate:
            raise ValueError("Document version is required; call document_snapshot before applying edits")
        candidate_text = str(candidate).strip()
        if not candidate_text:
            raise ValueError("Document version is required; call document_snapshot before applying edits")
        if explicit and candidate_text != str(snapshot_version).strip():
            raise DocumentVersionMismatchError(
                "Provided document_version does not match the latest snapshot; refresh document_snapshot and rebuild your diff."
            )
        return candidate_text

    @staticmethod
    def _resolve_content_hash(snapshot: Mapping[str, Any], base_text: str) -> str:
        token = snapshot.get("content_hash")
        if isinstance(token, str) and token.strip():
            return token.strip()
        return hashlib.sha1(base_text.encode("utf-8")).hexdigest()

    def _generate_snapshot(self, *, tab_id: str | None) -> Mapping[str, Any]:
        snapshot_fn = getattr(self.bridge, "generate_snapshot", None)
        if not callable(snapshot_fn):  # pragma: no cover - defensive
            raise ValueError("Bridge does not expose generate_snapshot")
        try:
            result = snapshot_fn(delta_only=False, tab_id=tab_id)
        except TypeError:
            result = snapshot_fn(delta_only=False)
        return cast(Mapping[str, Any], result)

    def _last_snapshot_version(self, tab_id: str | None) -> str | None:
        getter = getattr(self.bridge, "get_last_snapshot_version", None)
        if callable(getter):
            return cast(str | None, getter(tab_id=tab_id))
        return cast(str | None, getattr(self.bridge, "last_snapshot_version", None))

    def _normalize_filename(self, snapshot: Mapping[str, Any]) -> str:
        path = snapshot.get("path")
        if isinstance(path, str) and path.strip():
            return path.strip()
        document_id = snapshot.get("document_id") or "document"
        return f"tab://{document_id}" if document_id else self.filename_fallback


__all__ = ["DocumentApplyPatchTool"]
