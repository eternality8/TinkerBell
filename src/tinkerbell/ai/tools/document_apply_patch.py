"""Tool that builds and applies unified diffs in a single call."""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, cast

from .diff_builder import (
    DiffBuilderTool,
    StreamedDiffBuilder,
    StreamedDiffResult,
    StreamedEditRequest,
    StreamedPatchRange,
)
from .document_edit import DocumentEditTool, Bridge as EditBridge
from .document_snapshot import SnapshotProvider
from ...chat.commands import ActionType
from ...documents.range_normalizer import NormalizedTextRange, compose_normalized_replacement, normalize_text_range
from ...documents.ranges import LineRange, TextRange
from ..memory.plot_state import DocumentPlotStateStore
from ...services.bridge import DocumentVersionMismatchError
from ...services.telemetry import emit as telemetry_emit


class PatchBridge(EditBridge, SnapshotProvider, Protocol):
    """Protocol describing the bridge required by DocumentApplyPatchTool."""

    ...


class NeedsRangeError(ValueError):
    """Raised when large insertions lack an explicit range."""

    code = "needs_range"

    def __init__(self, message: str, *, content_length: int | None = None, threshold: int | None = None) -> None:
        super().__init__(message)
        self.content_length = content_length
        self.threshold = threshold
        self.details: dict[str, Any] = {"code": self.code}
        if content_length is not None:
            self.details["content_length"] = content_length
        if threshold is not None:
            self.details["threshold"] = threshold


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
    needs_range_event: str = "needs_range"
    hash_mismatch_event: str = "hash_mismatch"
    insert_needs_range_threshold: int = 1024
    replace_all_length_tolerance: float = 0.05
    plot_state_store_resolver: Callable[[], DocumentPlotStateStore | None] | None = None
    plot_warning_event: str | None = "plot_state.warning"
    require_line_spans: bool = False
    legacy_range_adapter_enabled: bool = True
    legacy_range_event: str | None = "target_range.legacy_adapter"
    summarizable: ClassVar[bool] = False

    def configure_line_span_policy(
        self,
        *,
        require_line_spans: bool,
        adapt_legacy_ranges: bool | None = None,
    ) -> None:
        """Toggle whether the tool requires line spans or adapts legacy offsets."""

        self.require_line_spans = bool(require_line_spans)
        if adapt_legacy_ranges is not None:
            self.legacy_range_adapter_enabled = bool(adapt_legacy_ranges)
        elif self.require_line_spans:
            # When spans are required we default to disabling the adapter unless explicitly overridden.
            self.legacy_range_adapter_enabled = False

    def run(
        self,
        *,
        content: str | None = None,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None = None,
        target_span: Mapping[str, Any] | Sequence[int] | tuple[int, int] | LineRange | None = None,
        document_version: str | None = None,
        version_id: str | int | None = None,
        content_hash: str | None = None,
        rationale: str | None = None,
        context_lines: int | None = None,
        tab_id: str | None = None,
        patches: Sequence[Mapping[str, Any]] | None = None,
        match_text: str | None = None,
        expected_text: str | None = None,
        selection_fingerprint: str | None = None,
        replace_all: bool | None = None,
        operation: str | None = None,
    ) -> str:
        streaming_mode = patches is not None and self.use_streamed_diffs
        snapshot = dict(self._generate_snapshot(tab_id=tab_id, include_text=not streaming_mode))
        base_text = snapshot.get("text", "")
        if streaming_mode:
            document_text = self._resolve_document_text_for_streaming(snapshot, tab_id=tab_id)
        else:
            if not isinstance(base_text, str):
                raise ValueError("Snapshot did not provide document text")
            document_text = base_text
        normalization_text = document_text

        line_span = self._coerce_line_span(target_span)
        span_from_request = line_span is not None
        resolved_target_range: tuple[int, int] | None = None
        if not streaming_mode:
            line_span, resolved_target_range = self._resolve_target_inputs(
                line_span=line_span,
                target_range=target_range,
                snapshot=snapshot,
                document_text=document_text,
                tab_id=tab_id,
            )
        range_provided = resolved_target_range is not None
        event_range_payload: Mapping[str, Any] | Sequence[int] | tuple[int, int] | LineRange | None = target_range
        if event_range_payload is None:
            event_range_payload = target_span
        if event_range_payload is None and line_span is not None:
            event_range_payload = line_span.to_dict()

        version_token = self._resolve_version(snapshot, document_version, tab_id=tab_id)
        self._verify_version_id(snapshot, version_id, tab_id=tab_id)
        snapshot_hash = self._verify_content_hash(
            snapshot,
            content_hash,
            tab_id=tab_id,
            base_text=document_text,
        )

        document_length = int(snapshot.get("length") or len(document_text))
        selection_tuple = snapshot.get("selection") or (0, 0)
        selection_span = self._selection_span(selection_tuple)
        selection_text = self._resolve_snapshot_selection_text(snapshot)
        replace_all_flag = bool(replace_all)
        operation_token = (operation or "").strip().lower()
        if operation_token and operation_token not in {ActionType.REPLACE.value}:
            raise ValueError("document_apply_patch only supports operation='replace' or replace_all=true")

        anchor_text, anchor_from_user = self._normalize_anchor_text(match_text, expected_text)
        if self._should_force_replace_all(
            content=content,
            target_range=resolved_target_range,
            replace_all=replace_all_flag,
            document_length=document_length,
            streaming_mode=streaming_mode,
        ):
            replace_all_flag = True
        if self._should_require_explicit_range(
            content=content,
            target_range=resolved_target_range,
            anchor_text=anchor_text,
            selection_span=selection_span,
            replace_all=replace_all_flag,
            streaming_mode=streaming_mode,
        ):
            reason = (
                "needs_range: Large inserts (>1 KB) require target_span (preferred), target_range, match_text, or replace_all=true; capture a snapshot and retry with explicit bounds."
            )
            self._emit_needs_range_event(
                snapshot=snapshot,
                tab_id=tab_id,
                selection_span=selection_span,
                content_length=len(content or ""),
                reason=reason,
            )
            raise NeedsRangeError(
                reason,
                content_length=len(content or ""),
                threshold=self.insert_needs_range_threshold,
            )
        if streaming_mode:
            requests = self._coerce_streaming_requests(patches or (), length=document_length)
            if not requests:
                raise ValueError("Patch ranges are required when streaming diffs are enabled")
            requests = self._normalize_streaming_requests(requests, text=normalization_text)
            streamed_result = self._build_streamed_ranges(requests, snapshot, tab_id=tab_id)
            updated_text = self._apply_streamed_ranges(document_text, streamed_result.ranges)
            payload = self._build_streamed_payload(
                streamed_result,
                snapshot,
                document_version=version_token,
                content_hash=snapshot_hash,
                rationale=rationale,
                tab_id=tab_id,
            )
            status = self.edit_tool.run(tab_id=tab_id, **payload)
            warnings = self._detect_plot_state_warnings(snapshot, document_text, updated_text)
            return self._finalize_status_with_plot_warning(status, warnings, snapshot, tab_id=tab_id)

        if content is None:
            raise ValueError("content is required when patches are not provided")
        anchor_source = self._anchor_source(anchor_text, selection_text, selection_fingerprint)
        selection_authoritative = bool(selection_fingerprint)
        try:
            self._enforce_range_requirements(
                resolved_target_range,
                selection_tuple,
                anchor_text,
                selection_authoritative,
                replace_all_flag,
            )
        except ValueError as exc:
            self._emit_caret_guard_event(
                snapshot=snapshot,
                tab_id=tab_id,
                selection_span=selection_span,
                range_provided=range_provided,
                range_payload=event_range_payload,
                replace_all=replace_all_flag,
                reason=str(exc),
            )
            self._emit_anchor_event(
                snapshot=snapshot,
                tab_id=tab_id,
                status="reject",
                phase="requirements",
                reason=str(exc),
                anchor_source=anchor_source,
                range_provided=range_provided or replace_all_flag,
                selection_span=selection_span,
            )
            raise
        try:
            start, end = self._resolve_anchored_range(
                base_text,
                snapshot,
                resolved_target_range,
                selection_tuple,
                anchor_text,
                anchor_from_user,
                selection_fingerprint,
                selection_text,
                replace_all_flag,
            )
        except ValueError as exc:
            self._emit_anchor_event(
                snapshot=snapshot,
                tab_id=tab_id,
                status="reject",
                phase="alignment",
                reason=str(exc),
                anchor_source=anchor_source,
                range_provided=range_provided or replace_all_flag,
                selection_span=selection_span,
            )
            raise
        self._emit_anchor_event(
            snapshot=snapshot,
            tab_id=tab_id,
            status="success",
            phase="alignment",
            anchor_source=anchor_source,
            range_provided=range_provided or replace_all_flag,
            selection_span=selection_span,
            resolved_range=(start, end),
        )
        new_text = str(content)
        if span_from_request and resolved_target_range is not None:
            normalized = NormalizedTextRange(
                start=resolved_target_range[0],
                end=resolved_target_range[1],
                slice_text=base_text[resolved_target_range[0] : resolved_target_range[1]],
            )
        else:
            normalized = normalize_text_range(base_text, start, end, replacement=new_text)
        normalized_replacement = compose_normalized_replacement(
            base_text,
            normalized,
            new_text,
            original_start=start,
            original_end=end,
        )
        if normalized.slice_text == normalized_replacement:
            return "skipped: content already matches selection"

        match_text_payload = normalized.slice_text
        updated_text = (
            base_text[: normalized.start]
            + normalized_replacement
            + base_text[normalized.end :]
        )
        filename = self._normalize_filename(snapshot)
        diff = self.diff_builder.run(
            base_text,
            updated_text,
            filename=filename,
            context=context_lines if context_lines is not None else self.default_context_lines,
        )
        payload: dict[str, Any] = {
            "action": "patch",
            "diff": diff,
            "document_version": version_token,
            "content_hash": snapshot_hash,
            "ranges": [
                {
                    "start": normalized.start,
                    "end": normalized.end,
                    "replacement": normalized_replacement,
                    "match_text": match_text_payload,
                }
            ],
        }
        if rationale is not None:
            payload["rationale"] = rationale
        status = self.edit_tool.run(tab_id=tab_id, **payload)
        warnings = self._detect_plot_state_warnings(snapshot, document_text, updated_text)
        return self._finalize_status_with_plot_warning(status, warnings, snapshot, tab_id=tab_id)

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

    def _coerce_line_span(
        self,
        value: Mapping[str, Any] | Sequence[int] | tuple[int, int] | LineRange | None,
    ) -> LineRange | None:
        if value is None:
            return None
        try:
            return LineRange.from_value(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_span must include start_line and end_line values") from exc

    def _resolve_target_inputs(
        self,
        *,
        line_span: LineRange | None,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        snapshot: Mapping[str, Any],
        document_text: str,
        tab_id: str | None,
    ) -> tuple[LineRange | None, tuple[int, int] | None]:
        resolved_span = line_span
        resolved_range: tuple[int, int] | None = None
        offsets: Sequence[int] | None = None
        if resolved_span is None and target_range is not None:
            if self.require_line_spans and not self.legacy_range_adapter_enabled:
                raise ValueError(
                    "target_range offsets are no longer supported; call document_snapshot and pass target_span with start_line/end_line bounds."
                )
            try:
                text_range = TextRange.from_value(target_range)
            except (TypeError, ValueError) as exc:
                raise ValueError("target_range must provide numeric start/end offsets") from exc
            length = len(document_text or "")
            start = max(0, min(text_range.start, length))
            end = max(0, min(text_range.end, length))
            if end < start:
                start, end = end, start
            offsets = self._resolve_line_offsets(snapshot, document_text)
            resolved_span = self._line_span_from_offsets((start, end), offsets)
            self._emit_legacy_range_event(
                snapshot=snapshot,
                tab_id=tab_id,
                span=resolved_span,
                range_start=start,
                range_end=end,
            )
            resolved_range = (start, end)
        if resolved_span is not None:
            if offsets is None:
                offsets = self._resolve_line_offsets(snapshot, document_text)
            if resolved_range is None:
                resolved_range = self._line_span_to_offsets(resolved_span, offsets)
        return resolved_span, resolved_range

    def _resolve_range(
        self,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        selection: Sequence[int],
        length: int,
        replace_all: bool = False,
    ) -> tuple[int, int]:
        if replace_all:
            return (0, max(0, int(length)))
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

    def _resolve_line_offsets(self, snapshot: Mapping[str, Any], text: str) -> Sequence[int]:
        raw = snapshot.get("line_offsets")
        offsets: list[int] = []
        if isinstance(raw, Sequence):
            for value in raw:
                try:
                    cursor = int(value)
                except (TypeError, ValueError):
                    continue
                cursor = max(0, cursor)
                if offsets and cursor < offsets[-1]:
                    cursor = offsets[-1]
                offsets.append(cursor)
        if not offsets:
            offsets = self._build_line_offsets(text)
        if not offsets:
            offsets = [0]
        if offsets[0] != 0:
            offsets.insert(0, 0)
        length = len(text or "")
        if offsets[-1] < length:
            offsets.append(length)
        elif offsets[-1] > length:
            offsets[-1] = length
        return offsets

    def _line_span_from_offsets(self, bounds: tuple[int, int], offsets: Sequence[int]) -> LineRange:
        start, end = bounds
        if end < start:
            start, end = end, start
        anchor = end - 1 if end > start else end
        start_line = self._line_for_offset(start, offsets)
        end_line = self._line_for_offset(max(anchor, start), offsets)
        return LineRange(start_line, end_line)

    def _line_span_to_offsets(self, span: LineRange, offsets: Sequence[int]) -> tuple[int, int]:
        if not offsets:
            return (0, 0)
        max_index = max(0, len(offsets) - 1)
        start_index = min(span.start_line, max_index)
        start = offsets[start_index]
        end_index = min(span.end_line + 1, len(offsets) - 1)
        end = offsets[end_index]
        return (start, end)

    @staticmethod
    def _line_for_offset(offset: int, offsets: Sequence[int]) -> int:
        if not offsets:
            return 0
        cursor = max(0, offset)
        index = bisect_right(offsets, cursor) - 1
        return max(0, index)

    @staticmethod
    def _build_line_offsets(text: str) -> list[int]:
        offsets = [0]
        if not text:
            return offsets
        cursor = 0
        for segment in text.splitlines(keepends=True):
            cursor += len(segment)
            offsets.append(cursor)
        if offsets[-1] < len(text):
            offsets.append(len(text))
        return offsets

    def _enforce_range_requirements(
        self,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        selection: Sequence[int],
        anchor_text: str | None,
        selection_authoritative: bool,
        replace_all: bool,
    ) -> None:
        if replace_all:
            return
        if target_range is not None:
            return
        if selection_authoritative and len(selection) == 2:
            try:
                sel_start = int(selection[0])
                sel_end = int(selection[1])
            except (TypeError, ValueError):
                sel_start = sel_end = 0
            if (sel_start, sel_end) != (0, 0):
                return
        if anchor_text:
            return
        raise ValueError(
            "Edits must include target_span (preferred), target_range, match_text, or replace_all=true; call document_snapshot to capture the intended selection before editing."
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
        replace_all: bool,
    ) -> tuple[int, int]:
        start, end = self._resolve_range(target_range, selection, len(base_text), replace_all)
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
        use_snapshot_anchor = selection_fingerprint is not None and selection_text is not None
        if anchor_candidate is None and use_snapshot_anchor:
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
        document_version: str,
        content_hash: str,
        rationale: str | None,
        tab_id: str | None,
    ) -> dict[str, Any]:
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
            "document_version": document_version,
            "content_hash": content_hash,
            "metadata": {"streamed_diff": metadata},
        }
        if rationale is not None:
            payload["rationale"] = rationale
        return payload

    def _resolve_document_text_for_streaming(
        self,
        snapshot: Mapping[str, Any],
        *,
        tab_id: str | None,
    ) -> str:
        text_value = snapshot.get("text")
        if isinstance(text_value, str) and text_value:
            return text_value
        refreshed = dict(self._generate_snapshot(tab_id=tab_id, include_text=True))
        refreshed_text = refreshed.get("text")
        if isinstance(refreshed_text, str):
            return refreshed_text
        raise ValueError("Unable to load document text for streamed patch normalization")

    def _normalize_streaming_requests(
        self,
        requests: Sequence[StreamedEditRequest],
        *,
        text: str,
    ) -> list[StreamedEditRequest]:
        if not requests:
            return []
        document = text or ""
        normalized_requests: list[StreamedEditRequest] = []
        for entry in requests:
            normalized = normalize_text_range(document, entry.start, entry.end, replacement=entry.replacement)
            normalized_replacement = compose_normalized_replacement(
                document,
                normalized,
                entry.replacement,
                original_start=entry.start,
                original_end=entry.end,
            )
            chunk_id = entry.chunk_id
            chunk_hash = entry.chunk_hash
            if normalized.start != entry.start or normalized.end != entry.end:
                chunk_id = None
                chunk_hash = None
            normalized_requests.append(
                StreamedEditRequest(
                    start=normalized.start,
                    end=normalized.end,
                    replacement=normalized_replacement,
                    match_text=normalized.slice_text,
                    chunk_id=chunk_id,
                    chunk_hash=chunk_hash,
                )
            )
        return normalized_requests

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

    def _emit_legacy_range_event(
        self,
        *,
        snapshot: Mapping[str, Any],
        tab_id: str | None,
        span: LineRange,
        range_start: int,
        range_end: int,
    ) -> None:
        if not self.legacy_range_event:
            return
        telemetry_emit(
            self.legacy_range_event,
            {
                "document_id": snapshot.get("document_id"),
                "tab_id": tab_id,
                "source": "document_apply_patch",
                "start_line": span.start_line,
                "end_line": span.end_line,
                "range_start": range_start,
                "range_end": range_end,
            },
        )

    def _emit_caret_guard_event(
        self,
        *,
        snapshot: Mapping[str, Any],
        tab_id: str | None,
        selection_span: tuple[int, int] | None,
        range_provided: bool,
        range_payload: Mapping[str, Any] | Sequence[int] | tuple[int, int] | LineRange | None,
        replace_all: bool,
        reason: str,
    ) -> None:
        payload: dict[str, Any] = {
            "document_id": snapshot.get("document_id"),
            "tab_id": tab_id,
            "source": "document_apply_patch",
            "range_provided": range_provided or replace_all,
            "replace_all": replace_all,
            "selection_span": self._span_payload(selection_span),
            "reason": reason,
        }
        if range_payload is not None:
            payload["range_payload"] = self._coerce_range_payload(range_payload)
        telemetry_emit(
            "caret_call_blocked",
            payload,
        )

    def _emit_needs_range_event(
        self,
        *,
        snapshot: Mapping[str, Any],
        tab_id: str | None,
        selection_span: tuple[int, int] | None,
        content_length: int,
        reason: str,
    ) -> None:
        if not self.needs_range_event:
            return
        telemetry_emit(
            self.needs_range_event,
            {
                "document_id": snapshot.get("document_id"),
                "tab_id": tab_id,
                "source": "document_apply_patch",
                "selection_span": self._span_payload(selection_span),
                "content_length": content_length,
                "threshold": self.insert_needs_range_threshold,
                "reason": reason,
            },
        )

    def _emit_hash_mismatch_event(
        self,
        *,
        snapshot: Mapping[str, Any],
        tab_id: str | None,
        stage: str,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.hash_mismatch_event:
            return
        payload: dict[str, Any] = {
            "document_id": snapshot.get("document_id"),
            "version": snapshot.get("version"),
            "version_id": snapshot.get("version_id"),
            "content_hash": snapshot.get("content_hash"),
            "tab_id": tab_id,
            "stage": stage,
            "reason": reason,
            "source": "document_apply_patch",
        }
        if details:
            payload["details"] = dict(details)
        telemetry_emit(self.hash_mismatch_event, payload)

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
    def _span_payload(span: tuple[int, int] | None) -> dict[str, int] | None:
        if span is None:
            return None
        return {"start": span[0], "end": span[1]}

    @staticmethod
    def _coerce_range_payload(
        value: Mapping[str, Any] | Sequence[int] | tuple[int, int] | LineRange | None,
    ) -> Mapping[str, Any] | Sequence[int] | tuple[int, int]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, LineRange):
            return value.to_dict()
        if isinstance(value, Sequence) and len(value) == 2:
            try:
                start = int(value[0])
                end = int(value[1])
            except (TypeError, ValueError):
                return list(value)
            return {"start": start, "end": end}
        return value

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
        if explicit is None:
            raise ValueError("document_version is required; call document_snapshot before applying edits")
        candidate_text = str(explicit).strip()
        if not candidate_text:
            raise ValueError("document_version is required; call document_snapshot before applying edits")
        snapshot_version = snapshot.get("version")
        snapshot_text = str(snapshot_version).strip() if snapshot_version else None
        if snapshot_text and candidate_text != snapshot_text:
            reason = (
                "Provided document_version does not match the latest snapshot; refresh document_snapshot and rebuild your diff."
            )
            self._emit_hash_mismatch_event(
                snapshot=snapshot,
                tab_id=tab_id,
                stage="document_version",
                reason=reason,
                details={"provided_version": candidate_text, "expected_version": snapshot_text},
            )
            raise DocumentVersionMismatchError(reason, cause="hash_mismatch")
        return candidate_text

    def _verify_version_id(
        self,
        snapshot: Mapping[str, Any],
        provided: str | int | None,
        *,
        tab_id: str | None,
    ) -> str:
        token = self._normalize_version_id(provided)
        if token is None:
            raise ValueError("version_id is required; call document_snapshot before applying edits")
        snapshot_token = self._normalize_version_id(snapshot.get("version_id"))
        if snapshot_token is None:
            raise ValueError("Snapshot did not expose version_id; call document_snapshot before editing")
        if token != snapshot_token:
            reason = "Provided version_id does not match the latest snapshot; refresh document_snapshot and rebuild your diff."
            self._emit_hash_mismatch_event(
                snapshot=snapshot,
                tab_id=tab_id,
                stage="version_id",
                reason=reason,
                details={"provided_version_id": token, "expected_version_id": snapshot_token},
            )
            raise DocumentVersionMismatchError(reason, cause="hash_mismatch")
        return snapshot_token

    def _verify_content_hash(
        self,
        snapshot: Mapping[str, Any],
        provided: str | None,
        *,
        tab_id: str | None,
        base_text: str | None,
    ) -> str:
        normalized = str(provided).strip() if isinstance(provided, str) else None
        if not normalized:
            raise ValueError("content_hash is required; call document_snapshot before applying edits")
        snapshot_hash = self._resolve_snapshot_content_hash(snapshot, base_text)
        if normalized != snapshot_hash:
            reason = (
                "Provided content_hash does not match the latest snapshot; refresh document_snapshot and rebuild your diff."
            )
            self._emit_hash_mismatch_event(
                snapshot=snapshot,
                tab_id=tab_id,
                stage="content_hash",
                reason=reason,
                details={"provided_content_hash": normalized, "expected_content_hash": snapshot_hash},
            )
            raise DocumentVersionMismatchError(reason, cause="hash_mismatch")
        return snapshot_hash

    @staticmethod
    def _resolve_snapshot_content_hash(snapshot: Mapping[str, Any], base_text: str | None) -> str:
        token = snapshot.get("content_hash")
        if isinstance(token, str) and token.strip():
            return token.strip()
        if base_text is None:
            raise ValueError("Snapshot did not expose content_hash; call document_snapshot before editing")
        return hashlib.sha1(base_text.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_version_id(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return str(int(value))
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return None

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

    def _normalize_filename(self, snapshot: Mapping[str, Any]) -> str:
        path = snapshot.get("path")
        if isinstance(path, str) and path.strip():
            return path.strip()
        document_id = snapshot.get("document_id") or "document"
        return f"tab://{document_id}" if document_id else self.filename_fallback

    def _should_force_replace_all(
        self,
        *,
        content: str | None,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        replace_all: bool,
        document_length: int,
        streaming_mode: bool,
    ) -> bool:
        if streaming_mode:
            return False
        if replace_all or target_range is not None:
            return False
        if content is None or document_length <= 0:
            return False
        tolerance = max(1, int(document_length * self.replace_all_length_tolerance))
        return abs(len(content) - document_length) <= tolerance

    def _should_require_explicit_range(
        self,
        *,
        content: str | None,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        anchor_text: str | None,
        selection_span: tuple[int, int] | None,
        replace_all: bool,
        streaming_mode: bool,
    ) -> bool:
        if streaming_mode:
            return False
        if replace_all:
            return False
        if content is None or len(content) < self.insert_needs_range_threshold:
            return False
        if target_range is not None or anchor_text:
            return False
        if selection_span is not None and selection_span[0] != selection_span[1]:
            return False
        return True

    @staticmethod
    def _apply_streamed_ranges(text: str, ranges: Sequence[StreamedPatchRange]) -> str:
        if not ranges:
            return text
        document = text or ""
        cursor = 0
        parts: list[str] = []
        for entry in ranges:
            parts.append(document[cursor : entry.start])
            parts.append(entry.replacement)
            cursor = entry.end
        parts.append(document[cursor:])
        return "".join(parts)

    def _detect_plot_state_warnings(
        self,
        snapshot: Mapping[str, Any],
        before_text: str,
        after_text: str,
    ) -> Mapping[str, Any] | None:
        store = self._resolve_plot_state_store()
        if store is None:
            return None
        document_id = snapshot.get("document_id")
        doc_id = str(document_id).strip() if isinstance(document_id, str) else None
        if not doc_id:
            return None
        try:
            plot_snapshot = store.snapshot(doc_id)
        except Exception:
            return None
        if not isinstance(plot_snapshot, Mapping):
            return None
        tracked_entities = self._extract_tracked_entities(plot_snapshot)
        tracked_beats = self._extract_tracked_beats(plot_snapshot)
        if not tracked_entities and not tracked_beats:
            return None
        before_lower = before_text.lower()
        after_lower = after_text.lower()
        removed_entities = sorted(
            {name for name in tracked_entities if name.lower() in before_lower and name.lower() not in after_lower}
        )
        removed_beats = sorted(
            {summary for summary in tracked_beats if summary.lower() in before_lower and summary.lower() not in after_lower}
        )
        if not removed_entities and not removed_beats:
            return None
        warning: dict[str, Any] = {"warning": "plot_outline_drift"}
        if removed_entities:
            warning["entities"] = removed_entities
        if removed_beats:
            warning["beats"] = removed_beats
        return warning

    def _finalize_status_with_plot_warning(
        self,
        status: str,
        warning: Mapping[str, Any] | None,
        snapshot: Mapping[str, Any],
        *,
        tab_id: str | None,
    ) -> str:
        if not warning:
            return status
        self._emit_plot_warning(snapshot=snapshot, tab_id=tab_id, warning=warning)
        notes: list[str] = []
        entities = warning.get("entities") if isinstance(warning, Mapping) else None
        beats = warning.get("beats") if isinstance(warning, Mapping) else None
        if isinstance(entities, Sequence) and entities:
            joined = ", ".join(entities[:5])
            notes.append(f"tracked entities removed: {joined}")
        if isinstance(beats, Sequence) and beats:
            joined = ", ".join(beats[:3])
            notes.append(f"tracked beats missing: {joined}")
        details = "; ".join(notes) if notes else "tracked plot elements changed"
        guidance = "Call plot_outline to confirm continuity or update plot_state before retrying."
        suffix = f"Plot continuity warning – {details}. {guidance}"
        separator = "\n" if "\n" in status else "\n"
        return f"{status}{separator}{suffix}"

    def _emit_plot_warning(
        self,
        *,
        snapshot: Mapping[str, Any],
        tab_id: str | None,
        warning: Mapping[str, Any],
    ) -> None:
        if not self.plot_warning_event:
            return
        payload: dict[str, Any] = {
            "document_id": snapshot.get("document_id"),
            "tab_id": tab_id,
            "source": "document_apply_patch",
        }
        for key in ("entities", "beats"):
            value = warning.get(key)
            if isinstance(value, Sequence) and value:
                payload[key] = list(value)
        telemetry_emit(self.plot_warning_event, payload)

    def _resolve_plot_state_store(self) -> DocumentPlotStateStore | None:
        resolver = self.plot_state_store_resolver
        if not callable(resolver):
            return None
        try:
            return resolver()
        except Exception:
            return None

    @staticmethod
    def _extract_tracked_entities(snapshot: Mapping[str, Any]) -> list[str]:
        entities = snapshot.get("entities")
        names: list[str] = []
        if isinstance(entities, Sequence):
            for entry in entities:
                if not isinstance(entry, Mapping):
                    continue
                name = entry.get("name")
                if isinstance(name, str) and name.strip():
                    token = name.strip()
                    if token not in names:
                        names.append(token)
        return names

    @staticmethod
    def _extract_tracked_beats(snapshot: Mapping[str, Any]) -> list[str]:
        arcs = snapshot.get("arcs")
        summaries: list[str] = []
        if not isinstance(arcs, Sequence):
            return summaries
        for arc in arcs:
            beats = arc.get("beats") if isinstance(arc, Mapping) else None
            if not isinstance(beats, Sequence):
                continue
            for beat in beats:
                if not isinstance(beat, Mapping):
                    continue
                summary = beat.get("summary")
                if isinstance(summary, str) and summary.strip():
                    token = summary.strip()
                    if token not in summaries:
                        summaries.append(token)
        return summaries


__all__ = ["DocumentApplyPatchTool", "NeedsRangeError"]
