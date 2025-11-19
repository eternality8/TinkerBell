"""Tool that builds and applies unified diffs in a single call."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, cast

from .diff_builder import DiffBuilderTool, StreamedDiffBuilder, StreamedEditRequest, StreamedDiffResult
from .document_edit import DocumentEditTool, Bridge as EditBridge
from .document_snapshot import SnapshotProvider
from ...chat.commands import ActionType
from ...services.bridge import DocumentVersionMismatchError
from ...services.telemetry import emit as telemetry_emit


class PatchBridge(EditBridge, SnapshotProvider, Protocol):
    """Protocol describing the bridge required by DocumentApplyPatchTool."""

    ...


@dataclass(slots=True)
class DocumentApplyPatchTool:
    """Build a diff from the live snapshot and route it through DocumentEdit."""

    bridge: PatchBridge
    edit_tool: DocumentEditTool
    diff_builder: DiffBuilderTool = field(default_factory=DiffBuilderTool)
    streamed_diff_builder: StreamedDiffBuilder | None = None
    filename_fallback: str = "document.txt"
    default_context_lines: int = 5
    use_streamed_diffs: bool = True
    stream_window_padding: int = 64
    streamed_diff_event: str = "diff.streamed"
    anchor_event: str = "patch.anchor"
    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        content: str | None = None,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None = None,
        document_version: str | None = None,
        rationale: str | None = None,
        context_lines: int | None = None,
        tab_id: str | None = None,
        patches: Sequence[Mapping[str, Any]] | None = None,
        match_text: str | None = None,
        expected_text: str | None = None,
        selection_fingerprint: str | None = None,
    ) -> str:
        streaming_mode = patches is not None and self.use_streamed_diffs
        snapshot = dict(self._generate_snapshot(tab_id=tab_id, include_text=not streaming_mode))
        base_text = snapshot.get("text", "")
        if not streaming_mode and not isinstance(base_text, str):
            raise ValueError("Snapshot did not provide document text")

        document_length = int(snapshot.get("length") or len(base_text))
        selection_tuple = snapshot.get("selection") or (0, 0)
        selection_text = self._resolve_snapshot_selection_text(snapshot)

        anchor_text, anchor_from_user = self._normalize_anchor_text(match_text, expected_text)
        if streaming_mode:
            requests = self._coerce_streaming_requests(patches or (), length=document_length)
            if not requests:
                raise ValueError("Patch ranges are required when streaming diffs are enabled")
            streamed_result = self._build_streamed_ranges(requests, snapshot, tab_id=tab_id)
            payload = self._build_streamed_payload(
                streamed_result,
                snapshot,
                document_version=document_version,
                rationale=rationale,
                tab_id=tab_id,
            )
            return self.edit_tool.run(tab_id=tab_id, **payload)

        if content is None:
            raise ValueError("content is required when patches are not provided")
        selection_span = self._selection_span(selection_tuple)
        anchor_source = self._anchor_source(anchor_text, selection_text, selection_fingerprint)
        try:
            self._enforce_range_requirements(
                target_range,
                selection_tuple,
                anchor_text,
                selection_fingerprint,
                selection_text,
            )
        except ValueError as exc:
            self._emit_anchor_event(
                snapshot=snapshot,
                tab_id=tab_id,
                status="reject",
                phase="requirements",
                reason=str(exc),
                anchor_source=anchor_source,
                range_provided=target_range is not None,
                selection_span=selection_span,
            )
            raise
        try:
            start, end = self._resolve_anchored_range(
                base_text,
                snapshot,
                target_range,
                selection_tuple,
                anchor_text,
                anchor_from_user,
                selection_fingerprint,
                selection_text,
            )
        except ValueError as exc:
            self._emit_anchor_event(
                snapshot=snapshot,
                tab_id=tab_id,
                status="reject",
                phase="alignment",
                reason=str(exc),
                anchor_source=anchor_source,
                range_provided=target_range is not None,
                selection_span=selection_span,
            )
            raise
        self._emit_anchor_event(
            snapshot=snapshot,
            tab_id=tab_id,
            status="success",
            phase="alignment",
            anchor_source=anchor_source,
            range_provided=target_range is not None,
            selection_span=selection_span,
            resolved_range=(start, end),
        )
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

    def _normalize_anchor_text(
        self,
        match_text: str | None,
        expected_text: str | None,
    ) -> tuple[str | None, bool]:
        values = [value for value in (match_text, expected_text) if value is not None]
        if not values:
            return None, False
        first = str(values[0])
        for candidate in values[1:]:
            if str(candidate) != first:
                raise ValueError("match_text and expected_text must match when both are provided")
        return first, True

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

    def _enforce_range_requirements(
        self,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        selection: Sequence[int],
        anchor_text: str | None,
        selection_fingerprint: str | None,
        selection_text: str | None,
    ) -> None:
        if target_range is not None:
            return
        if len(selection) == 2:
            try:
                sel_start = int(selection[0])
                sel_end = int(selection[1])
            except (TypeError, ValueError):
                sel_start = sel_end = 0
            if (sel_start, sel_end) != (0, 0):
                return
        if anchor_text:
            return
        if selection_fingerprint:
            return
        if selection_text:
            return
        raise ValueError(
            "Edits must include target_range or match_text; call document_snapshot to capture the intended selection before editing."
        )

    def _resolve_anchored_range(
        self,
        base_text: str,
        snapshot: Mapping[str, Any],
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        selection: Sequence[int],
        anchor_text: str | None,
        anchor_from_user: bool,
        selection_fingerprint: str | None,
        selection_text: str | None,
    ) -> tuple[int, int]:
        start, end = self._resolve_range(target_range, selection, len(base_text))
        selection_hash = self._resolve_selection_hash(snapshot, selection_text)

        if selection_fingerprint is not None:
            fingerprint = selection_fingerprint.strip()
            if not fingerprint:
                raise ValueError("selection_fingerprint cannot be empty")
            if not selection_hash:
                raise ValueError("Snapshot did not expose selection_text required to validate selection_fingerprint")
            if fingerprint != selection_hash:
                raise ValueError(
                    "selection_fingerprint does not match the latest snapshot; refresh document_snapshot before applying this edit."
                )

        anchor_candidate = anchor_text
        from_snapshot = False
        if anchor_candidate is None and selection_text is not None:
            anchor_candidate = selection_text
            from_snapshot = True

        if anchor_candidate is None or anchor_candidate == "":
            return start, end

        selection_slice = base_text[start:end]
        if selection_slice == anchor_candidate:
            return start, end

        if from_snapshot and not anchor_from_user:
            raise ValueError(
                "Snapshot selection_text no longer matches document content; provide match_text or selection_fingerprint to re-anchor the edit."
            )

        relocated = self._locate_unique_anchor(base_text, anchor_candidate)
        if relocated is None:
            raise ValueError("match_text did not match any content in the current document; refresh document_snapshot")
        return relocated

    @staticmethod
    def _locate_unique_anchor(base_text: str, anchor: str) -> tuple[int, int] | None:
        position = base_text.find(anchor)
        if position < 0:
            return None
        duplicate = base_text.find(anchor, position + 1)
        if duplicate >= 0:
            raise ValueError("match_text matched multiple ranges; narrow the selection to proceed")
        return position, position + len(anchor)

    @staticmethod
    @staticmethod
    def _resolve_snapshot_selection_text(snapshot: Mapping[str, Any]) -> str | None:
        snapshot_value = snapshot.get("selection_text")
        if isinstance(snapshot_value, str):
            return snapshot_value
        return None

    def _resolve_selection_hash(self, snapshot: Mapping[str, Any], selection_text: str | None) -> str | None:
        token = snapshot.get("selection_hash")
        if isinstance(token, str) and token.strip():
            return token.strip()
        if selection_text:
            return self._hash_text(selection_text)
        return None

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _coerce_streaming_requests(
        self,
        patches: Sequence[Mapping[str, Any]],
        *,
        length: int,
    ) -> list[StreamedEditRequest]:
        requests: list[StreamedEditRequest] = []
        for entry in patches:
            if not isinstance(entry, Mapping):
                raise ValueError("Patch entries must be objects")
            if "start" not in entry or "end" not in entry:
                raise ValueError("Patch entries must include 'start' and 'end' offsets")
            start, end = self._clamp_bounds(entry.get("start", 0), entry.get("end", 0), length)
            replacement = entry.get("replacement")
            if replacement is None:
                replacement = entry.get("content") or entry.get("text")
            if replacement is None:
                raise ValueError("Patch entries must include replacement text")
            match_text = entry.get("match_text")
            match_text_str = None if match_text is None else str(match_text)
            chunk_id = entry.get("chunk_id")
            chunk_hash = entry.get("chunk_hash")
            requests.append(
                StreamedEditRequest(
                    start=start,
                    end=end,
                    replacement=str(replacement),
                    match_text=match_text_str,
                    chunk_id=str(chunk_id) if isinstance(chunk_id, str) else None,
                    chunk_hash=str(chunk_hash) if isinstance(chunk_hash, str) else None,
                )
            )
        return requests

    def _build_streamed_ranges(
        self,
        requests: Sequence[StreamedEditRequest],
        snapshot: Mapping[str, Any],
        *,
        tab_id: str | None,
    ) -> StreamedDiffResult:
        builder = self.streamed_diff_builder or StreamedDiffBuilder()
        self.streamed_diff_builder = builder
        manifest = snapshot.get("chunk_manifest")
        length = int(snapshot.get("length") or 0)

        def _window_loader(start: int, end: int) -> Mapping[str, object]:
            return self._load_window_snapshot(start, end, tab_id=tab_id, length=length)

        manifest_mapping = cast(Mapping[str, object], manifest) if isinstance(manifest, Mapping) else None
        result = builder.build(requests, window_loader=_window_loader, manifest=manifest_mapping)
        self._emit_streaming_stats(result, snapshot, tab_id=tab_id)
        return result

    def _build_streamed_payload(
        self,
        result: StreamedDiffResult,
        snapshot: Mapping[str, Any],
        *,
        document_version: str | None,
        rationale: str | None,
        tab_id: str | None,
    ) -> dict[str, Any]:
        version = self._resolve_version(snapshot, document_version, tab_id=tab_id)
        content_hash = self._resolve_content_hash(snapshot, None)
        ranges_payload: list[dict[str, Any]] = []
        for entry in result.ranges:
            payload: dict[str, Any] = {
                "start": entry.start,
                "end": entry.end,
                "replacement": entry.replacement,
                "match_text": entry.match_text,
            }
            if entry.chunk_id:
                payload["chunk_id"] = entry.chunk_id
            if entry.chunk_hash:
                payload["chunk_hash"] = entry.chunk_hash
            ranges_payload.append(payload)

        metadata: dict[str, Any] = {
            "range_count": result.stats.range_count,
            "replaced_chars": result.stats.replaced_chars,
            "inserted_chars": result.stats.inserted_chars,
        }

        payload: dict[str, Any] = {
            "action": ActionType.PATCH.value,
            "ranges": ranges_payload,
            "document_version": version,
            "content_hash": content_hash,
            "metadata": {"streamed_diff": metadata},
        }
        if rationale is not None:
            payload["rationale"] = rationale
        return payload

    def _load_window_snapshot(
        self,
        start: int,
        end: int,
        *,
        tab_id: str | None,
        length: int,
    ) -> Mapping[str, object]:
        snapshot_fn = getattr(self.bridge, "generate_snapshot", None)
        if not callable(snapshot_fn):
            raise ValueError("Bridge does not expose generate_snapshot")

        padded_start = max(0, start - self.stream_window_padding)
        padded_end = min(length, end + self.stream_window_padding)
        if padded_end < padded_start:
            padded_end = padded_start
        window_payload = {"start": padded_start, "end": padded_end}
        call_kwargs: dict[str, Any] = {
            "delta_only": False,
            "tab_id": tab_id,
            "include_text": True,
            "window": window_payload,
        }
        try:
            window_snapshot = snapshot_fn(**call_kwargs)
        except TypeError:
            try:
                window_snapshot = snapshot_fn(delta_only=False, tab_id=tab_id)
            except TypeError:
                window_snapshot = snapshot_fn(delta_only=False)
        mapping = dict(cast(Mapping[str, Any], window_snapshot))
        mapping.setdefault("window", window_payload)
        if "text" not in mapping or not isinstance(mapping.get("text"), str):
            raise ValueError("Bridge window snapshot did not return text content")
        return mapping

    def _emit_streaming_stats(
        self,
        result: StreamedDiffResult,
        snapshot: Mapping[str, Any],
        *,
        tab_id: str | None,
    ) -> None:
        if not self.streamed_diff_event:
            return
        payload = {
            "document_id": snapshot.get("document_id"),
            "tab_id": tab_id,
            "range_count": result.stats.range_count,
            "replaced_chars": result.stats.replaced_chars,
            "inserted_chars": result.stats.inserted_chars,
        }
        telemetry_emit(self.streamed_diff_event, payload)

    def _emit_anchor_event(
        self,
        *,
        snapshot: Mapping[str, Any],
        tab_id: str | None,
        status: str,
        phase: str,
        anchor_source: str,
        range_provided: bool,
        selection_span: tuple[int, int] | None,
        resolved_range: tuple[int, int] | None = None,
        reason: str | None = None,
    ) -> None:
        if not self.anchor_event:
            return
        payload: dict[str, Any] = {
            "document_id": snapshot.get("document_id"),
            "tab_id": tab_id,
            "status": status,
            "phase": phase,
            "source": "document_apply_patch",
            "anchor_source": anchor_source,
            "range_provided": range_provided,
        }
        if selection_span is not None:
            payload["selection_span"] = {"start": selection_span[0], "end": selection_span[1]}
        if resolved_range is not None:
            payload["resolved_range"] = {"start": resolved_range[0], "end": resolved_range[1]}
        if reason:
            payload["reason"] = reason
        telemetry_emit(self.anchor_event, payload)

    @staticmethod
    def _selection_span(selection: Sequence[int]) -> tuple[int, int] | None:
        if len(selection) != 2:
            return None
        try:
            start = int(selection[0])
            end = int(selection[1])
        except (TypeError, ValueError):
            return None
        if end < start:
            start, end = end, start
        return (start, end)

    @staticmethod
    def _anchor_source(
        anchor_text: str | None,
        selection_text: str | None,
        selection_fingerprint: str | None,
    ) -> str:
        if selection_fingerprint:
            return "fingerprint"
        if anchor_text:
            return "match_text"
        if selection_text:
            return "selection_text"
        return "range_only"

    @staticmethod
    def _clamp_bounds(start: Any, end: Any, length: int) -> tuple[int, int]:
        start_int = int(start)
        end_int = int(end)
        start_int = max(0, min(start_int, length))
        end_int = max(0, min(end_int, length))
        if end_int < start_int:
            start_int, end_int = end_int, start_int
        return start_int, end_int

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
    def _resolve_content_hash(snapshot: Mapping[str, Any], base_text: str | None) -> str:
        token = snapshot.get("content_hash")
        if isinstance(token, str) and token.strip():
            return token.strip()
        if base_text is None:
            raise ValueError("Snapshot did not expose content_hash; call document_snapshot before editing")
        return hashlib.sha1(base_text.encode("utf-8")).hexdigest()

    def _generate_snapshot(self, *, tab_id: str | None, include_text: bool = True) -> Mapping[str, Any]:
        snapshot_fn = getattr(self.bridge, "generate_snapshot", None)
        if not callable(snapshot_fn):  # pragma: no cover - defensive
            raise ValueError("Bridge does not expose generate_snapshot")
        call_kwargs = {"delta_only": False, "tab_id": tab_id, "include_text": include_text}
        try:
            result = snapshot_fn(**call_kwargs)
        except TypeError:
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
