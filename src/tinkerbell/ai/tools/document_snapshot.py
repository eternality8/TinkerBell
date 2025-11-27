"""Tool returning the current document snapshot."""

from __future__ import annotations

import ast
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Mapping, Protocol, cast

from ..memory.chunk_index import ChunkIndex
from ...services.telemetry import emit

LOGGER = logging.getLogger(__name__)


class SnapshotProvider(Protocol):
    """Protocol implemented by the document bridge for retrieving snapshots."""

    def generate_snapshot(
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
        window: Mapping[str, Any] | str | None = None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
    ) -> Mapping[str, Any]:
        ...

    def get_last_diff_summary(self, tab_id: str | None = None) -> str | None:
        ...

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:
        ...


@dataclass(slots=True)
class DocumentSnapshotTool:
    """Simple synchronous tool returning document snapshots."""

    provider: SnapshotProvider
    outline_digest_resolver: Callable[[str | None], str | None] | None = None
    chunk_index: ChunkIndex | None = None
    summarizable: ClassVar[bool] = True
    DEFAULT_WINDOW: ClassVar[dict[str, Any]] = {"kind": "document", "max_chars": 8192}

    def run(
        self,
        request: Mapping[str, Any] | str | None = None,
        *,
        delta_only: bool = False,
        include_diff: bool = True,
        tab_id: str | None = None,
        source_tab_ids: Iterable[str] | None = None,
        include_open_documents: bool = False,
        window: Mapping[str, Any] | str | None = None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
        offset: int | None = None,
    ) -> dict:
        request_kwargs = self._coerce_request_mapping(request)
        ignored_keys: list[str] = []  # WS3 4.4.1: Track ignored keys
        if request_kwargs:
            delta_only = request_kwargs.pop("delta_only", delta_only)
            include_diff = request_kwargs.pop("include_diff", include_diff)
            source_tab_ids = request_kwargs.pop("source_tab_ids", source_tab_ids)
            include_open_documents = request_kwargs.pop("include_open_documents", include_open_documents)
            window = request_kwargs.pop("window", window)
            chunk_profile = request_kwargs.pop("chunk_profile", chunk_profile)
            max_tokens = request_kwargs.pop("max_tokens", max_tokens)
            include_text = request_kwargs.pop("include_text", include_text)
            offset = request_kwargs.pop("offset", offset)
            tab_id = request_kwargs.pop("tab_id", tab_id)
            document_id = request_kwargs.pop("document_id", None)
            alias = str(document_id).strip() if document_id is not None else ""
            if (tab_id is None or not str(tab_id).strip()) and alias:
                tab_id = alias
            if request_kwargs:
                ignored_keys = sorted(request_kwargs.keys())  # WS3 4.4.1
                LOGGER.debug(
                    "DocumentSnapshotTool ignoring unsupported request keys: %s",
                    ", ".join(ignored_keys),
                )
                # WS3 4.4.2: Telemetry for ignored fields
                self._emit_ignored_keys(ignored_keys, tab_id=tab_id)
        
        # If offset is provided, build a window starting from that offset
        if offset is not None and offset > 0:
            window = {"start": offset, "kind": "range"}
        
        resolved_window = self._resolve_window(window)
        snapshot = self._build_snapshot(
            delta_only=delta_only,
            include_diff=include_diff,
            tab_id=tab_id,
            include_open_documents=include_open_documents,
            window=resolved_window,
            chunk_profile=chunk_profile,
            max_tokens=max_tokens,
            include_text=include_text,
        )

        digest = self._resolve_outline_digest(snapshot, tab_id)
        if digest:
            snapshot["outline_digest"] = digest

        extras = self._build_additional_snapshots(
            source_tab_ids,
            delta_only=delta_only,
            include_diff=include_diff,
            window=resolved_window,
            chunk_profile=chunk_profile,
            max_tokens=max_tokens,
            include_text=include_text,
        )
        if extras:
            snapshot["source_tab_snapshots"] = extras

        # WS3 4.4.1: Add ignored_keys to response with warning
        if ignored_keys:
            snapshot["ignored_keys"] = ignored_keys
            snapshot["warning"] = f"The following request parameters were not recognized and ignored: {', '.join(ignored_keys)}"

        # Clean up snapshot to remove fields that confuse the AI (after all internal processing)
        self._simplify_snapshot_for_ai(snapshot)

        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_snapshot(
        self,
        *,
        delta_only: bool,
        include_diff: bool,
        tab_id: str | None,
        include_open_documents: bool,
        window: Mapping[str, Any] | str | None,
        chunk_profile: str | None,
        max_tokens: int | None,
        include_text: bool,
    ) -> dict:
        snapshot = deepcopy(
            dict(
                self._invoke_generate_snapshot(
                    delta_only=delta_only,
                    tab_id=tab_id,
                    include_open_documents=include_open_documents,
                    window=self._resolve_window(window),
                    chunk_profile=chunk_profile,
                    max_tokens=max_tokens,
                    include_text=include_text,
                )
            )
        )

        if include_diff:
            diff = self._last_diff(tab_id)
            if diff is not None:
                snapshot["diff_summary"] = diff

        version = snapshot.get("version") or self._last_version(tab_id)
        if version is not None:
            snapshot["version"] = version

        self._ingest_chunk_manifest(snapshot)

        # Add snapshot_token for simplified versioning (WS1.1.1)
        self._add_snapshot_token(snapshot, tab_id)

        # Add suggested_span for auto-fill (WS1.4.2)
        self._add_suggested_span(snapshot)

        return snapshot

    def _simplify_snapshot_for_ai(self, snapshot: dict) -> None:
        """Remove or simplify fields that are internal or confuse the AI.
        
        The AI should use snapshot_token (contains tab_id:version_id) for all
        edit operations. Fields like document_id are removed because the AI
        often confuses them with tab_id when constructing snapshot_token.
        """
        # Remove document_id - AI often confuses it with tab_id
        # The snapshot_token contains the correct tab_id
        snapshot.pop("document_id", None)
        
        # Remove internal line offset array - not needed by AI
        snapshot.pop("line_start_offsets", None)
        snapshot.pop("line_offsets", None)
        
        # Simplify chunk_manifest - keep only cache_key and count for iteration
        manifest = snapshot.get("chunk_manifest")
        if isinstance(manifest, Mapping):
            simplified_manifest = {}
            if "cache_key" in manifest:
                simplified_manifest["cache_key"] = manifest["cache_key"]
            if "chunk_count" in manifest:
                simplified_manifest["chunk_count"] = manifest["chunk_count"]
            elif "chunks" in manifest and isinstance(manifest["chunks"], list):
                simplified_manifest["chunk_count"] = len(manifest["chunks"])
            snapshot["chunk_manifest"] = simplified_manifest if simplified_manifest else None
            if not snapshot["chunk_manifest"]:
                snapshot.pop("chunk_manifest", None)
        
        # Simplify window - just keep essential info
        window = snapshot.get("window")
        if isinstance(window, Mapping):
            simplified_window = {
                "start": window.get("start", 0),
                "end": window.get("end", 0),
            }
            if window.get("includes_full_document"):
                simplified_window["full_document"] = True
            snapshot["window"] = simplified_window

        # Add continuation hint when document is truncated
        doc_length = snapshot.get("length", 0)
        window_end = snapshot.get("window", {}).get("end", 0)
        if doc_length and window_end and doc_length > window_end:
            # Document is truncated - add simple hint to use offset parameter
            snapshot["continuation_hint"] = (
                f"Document truncated at {window_end}/{doc_length} chars. "
                f"To read more, call document_snapshot with offset={window_end}."
            )

    def _add_snapshot_token(self, snapshot: dict, tab_id: str | None) -> None:
        """Add compact snapshot_token combining tab_id and version_id."""
        resolved_tab_id = tab_id or snapshot.get("tab_id") or snapshot.get("document_id") or ""
        version_id = snapshot.get("version_id")
        if resolved_tab_id and version_id is not None:
            snapshot["snapshot_token"] = f"{resolved_tab_id}:{version_id}"

    def _add_suggested_span(self, snapshot: dict) -> None:
        """Derive suggested_span from text_range for downstream tools."""
        text_range = snapshot.get("text_range")
        if not isinstance(text_range, Mapping):
            return
        line_offsets = snapshot.get("line_start_offsets") or snapshot.get("line_offsets")
        if not isinstance(line_offsets, (list, tuple)) or not line_offsets:
            return
        start_offset = text_range.get("start", 0)
        end_offset = text_range.get("end", 0)
        start_line = self._line_for_offset(start_offset, line_offsets)
        # For end_line, use the last character position (end - 1) to get the inclusive line
        end_line = self._line_for_offset(max(start_offset, end_offset - 1), line_offsets) if end_offset > start_offset else start_line
        snapshot["suggested_span"] = {"start_line": start_line, "end_line": end_line}

    @staticmethod
    def _line_for_offset(offset: int, offsets: list | tuple) -> int:
        """Return the 0-based line index containing the given offset."""
        from bisect import bisect_right
        cursor = max(0, offset)
        index = bisect_right(offsets, cursor) - 1
        return max(0, index)

    def _build_additional_snapshots(
        self,
        tab_ids: Iterable[str] | None,
        *,
        delta_only: bool,
        include_diff: bool,
        window: Mapping[str, Any] | str | None,
        chunk_profile: str | None,
        max_tokens: int | None,
        include_text: bool,
    ) -> list[dict]:
        if not tab_ids:
            return []
        if isinstance(tab_ids, (str, bytes)):
            iterable: Iterable[str | bytes] = [tab_ids]
        else:
            iterable = tab_ids
        snapshots: list[dict] = []
        for source_id in iterable:
            candidate = str(source_id).strip()
            if not candidate:
                continue
            snapshots.append(
                self._build_snapshot(
                    delta_only=delta_only,
                    include_diff=include_diff,
                    tab_id=candidate,
                    include_open_documents=False,
                    window=window,
                    chunk_profile=chunk_profile,
                    max_tokens=max_tokens,
                    include_text=include_text,
                )
            )
        return snapshots

    def _ingest_chunk_manifest(self, snapshot: Mapping[str, Any]) -> None:
        if self.chunk_index is None:
            return
        manifest = snapshot.get("chunk_manifest")
        if not isinstance(manifest, Mapping):
            return
        try:
            self.chunk_index.ingest_manifest(manifest)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Chunk manifest ingestion failed", exc_info=True)

    def _invoke_generate_snapshot(
        self,
        *,
        delta_only: bool,
        tab_id: str | None,
        include_open_documents: bool,
        window: Mapping[str, Any] | str | None,
        chunk_profile: str | None,
        max_tokens: int | None,
        include_text: bool,
    ) -> Mapping[str, Any]:
        generate = getattr(self.provider, "generate_snapshot", None)
        if not callable(generate):  # pragma: no cover - defensive guard
            raise ValueError("Snapshot provider is missing generate_snapshot()")

        try:
            result = generate(
                delta_only=delta_only,
                tab_id=tab_id,
                include_open_documents=include_open_documents,
                window=window,
                chunk_profile=chunk_profile,
                max_tokens=max_tokens,
                include_text=include_text,
            )
        except TypeError:
            if tab_id is not None or include_open_documents:
                raise ValueError("Snapshot provider does not support tab_id or open document metadata")
            result = generate(delta_only=delta_only)
        return cast(Mapping[str, Any], result)

    def _last_diff(self, tab_id: str | None) -> str | None:
        getter = getattr(self.provider, "get_last_diff_summary", None)
        if callable(getter):
            return cast(str | None, getter(tab_id=tab_id))
        return cast(str | None, getattr(self.provider, "last_diff_summary", None))

    def _last_version(self, tab_id: str | None) -> str | None:
        getter = getattr(self.provider, "get_last_snapshot_version", None)
        if callable(getter):
            return cast(str | None, getter(tab_id=tab_id))
        return cast(str | None, getattr(self.provider, "last_snapshot_version", None))

    def _resolve_outline_digest(self, snapshot: Mapping[str, Any], tab_id: str | None) -> str | None:
        resolver = self.outline_digest_resolver
        if not callable(resolver):
            return None
        raw_id = snapshot.get("document_id")
        if (raw_id is None or not str(raw_id).strip()) and tab_id:
            raw_id = tab_id
        document_id = str(raw_id).strip() if raw_id is not None else None
        return resolver(document_id or None)

    def _resolve_window(self, window: Mapping[str, Any] | str | None) -> Mapping[str, Any] | str | None:
        if window is None:
            return dict(self.DEFAULT_WINDOW)
        if isinstance(window, str):
            normalized = window.strip().lower()
            if not normalized or normalized == "default":
                return dict(self.DEFAULT_WINDOW)
            return normalized
        if isinstance(window, Mapping):
            return dict(window)
        return window

    def _coerce_request_mapping(self, request: Mapping[str, Any] | str | None) -> dict[str, Any]:
        if request is None:
            return {}
        if isinstance(request, Mapping):
            return dict(request)
        if isinstance(request, str):
            text = request.strip()
            if not text:
                return {}
            for parser in (self._parse_json, self._parse_literal):
                parsed = parser(text)
                if isinstance(parsed, Mapping):
                    return dict(parsed)
            LOGGER.debug("DocumentSnapshotTool received unparseable request payload: %s", text)
            return {}
        LOGGER.debug("DocumentSnapshotTool received unsupported request payload type: %s", type(request).__name__)
        return {}

    @staticmethod
    def _parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_literal(text: str) -> Any:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    @staticmethod
    def _emit_ignored_keys(keys: list[str], *, tab_id: str | None) -> None:
        """WS3 4.4.2: Telemetry for ignored request fields."""
        emit(
            "snapshot_tool.ignored_keys",
            {
                "tab_id": tab_id,
                "ignored_keys": keys,
                "count": len(keys),
            },
        )