"""Tool exposing embedding-backed document span retrieval."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence

from ...editor.document_model import DocumentState
from ..memory.buffers import DocumentSummaryMemory, OutlineNode
from ..memory.embeddings import DocumentEmbeddingIndex, EmbeddingMatch
from ...services.telemetry import count_text_tokens, emit
from ..utils.document_checks import unsupported_format_reason
from .validation import parse_snapshot_token

LOGGER = logging.getLogger(__name__)

DocumentResolver = Callable[[str], DocumentState | None]
EmbeddingIndexResolver = Callable[[], DocumentEmbeddingIndex | None]
OutlineMemoryResolver = Callable[[], DocumentSummaryMemory | None]


@dataclass(slots=True)
class DocumentFindTextTool:
    """Retrieve document chunk pointers by querying the embedding index."""

    embedding_index: DocumentEmbeddingIndex | None = None
    embedding_index_resolver: EmbeddingIndexResolver | None = None
    document_lookup: DocumentResolver | None = None
    active_document_provider: Callable[[], DocumentState | None] | None = None
    outline_memory: DocumentSummaryMemory | OutlineMemoryResolver | None = None
    max_preview_chars: int = 360
    default_top_k: int = 6

    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        document_id: str | None = None,
        tab_id: str | None = None,
        snapshot_token: str | None = None,
        query: str | None = None,
        top_k: int | None = None,
        min_confidence: float | None = None,
        filters: Mapping[str, Any] | None = None,
        include_outline_context: bool = True,
    ) -> dict[str, Any]:
        # Parse snapshot_token if provided (WS1.1.4)
        parsed_tab_id, parsed_version_id = self._parse_snapshot_token(snapshot_token)
        if parsed_tab_id is not None:
            if tab_id is None:
                tab_id = parsed_tab_id
            if document_id is None:
                document_id = parsed_tab_id

        query_text = (query or "").strip()
        if not query_text:
            return {
                "status": "invalid_request",
                "reason": "query_required",
            }

        document = self._resolve_document(document_id or tab_id)
        if document is None:
            return {
                "status": "no_document",
                "reason": "document_unavailable",
                "document_id": document_id,
                "tab_id": tab_id,
            }
        unsupported_reason = unsupported_format_reason(document)
        if unsupported_reason:
            return {
                "status": "unsupported_format",
                "reason": unsupported_reason,
                "document_id": document.document_id,
                "tab_id": tab_id,
            }

        limit = self._sanitize_top_k(top_k)
        confidence = self._sanitize_confidence(min_confidence)
        outline_record = self._lookup_outline(document.document_id)
        outline_index = self._build_outline_index(outline_record.nodes if outline_record else None)
        outline_digest = outline_record.outline_hash if outline_record else None

        start = time.perf_counter()
        pointers: list[dict[str, Any]] = []
        strategy = "embedding"
        fallback_reason: str | None = None
        provider_error: str | None = None

        index = self._resolve_embedding_index()
        provider_label = getattr(index, "provider_name", None) if index is not None else None
        if index is not None:
            try:
                matches = self._similarity_search(index, document.document_id, query_text, limit, confidence)
            except Exception as exc:
                import traceback
                provider_error = f"{exc.__class__.__name__}: {exc}"
                LOGGER.warning(
                    "Embedding search failed for document %s: %s\n%s",
                    document.document_id,
                    provider_error,
                    traceback.format_exc(),
                )
                emit(
                    "retrieval.provider.error",
                    {
                        "document_id": document.document_id,
                        "provider": provider_label,
                        "query_length": len(query_text),
                        "error": provider_error[:200],
                    },
                )
                matches = []
            if matches:
                pointers.extend(
                    self._pointers_from_matches(
                        matches,
                        document=document,
                        include_outline_context=include_outline_context,
                        outline_index=outline_index,
                    )
                )
        else:
            matches = []

        if not pointers:
            strategy = "fallback"
            if provider_error is not None:
                fallback_reason = "provider_error"
            else:
                fallback_reason = "embedding_unavailable" if index is None else "no_embedding_matches"
            pointers.extend(
                self._fallback_search(
                    document=document,
                    query=query_text,
                    limit=limit,
                    include_outline_context=include_outline_context,
                    outline_index=outline_index,
                )
            )

        latency_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
        offline_mode = index is None
        if offline_mode:
            status = "offline_fallback" if pointers else "offline_no_results"
        else:
            status = "ok" if pointers else "no_results"
        document_tokens = count_text_tokens(document.text or "", estimate_only=True)
        pointer_tokens = self._total_pointer_tokens(pointers)
        tokens_saved = max(0, document_tokens - pointer_tokens)

        # WS3 4.1.x: Add confidence level based on strategy and mode
        result_confidence = self._determine_confidence(strategy, offline_mode, pointers)
        warning = self._build_confidence_warning(result_confidence, fallback_reason, offline_mode)

        response = {
            "status": status,
            "document_id": document.document_id,
            "tab_id": tab_id or document.document_id,
            "version_id": document.version_id,
            "snapshot_token": f"{document.document_id}:{document.version_id}" if document.version_id else None,
            "query": query_text,
            "strategy": strategy,
            "fallback_reason": fallback_reason,
            "provider_error": provider_error[:200] if provider_error else None,
            "confidence": result_confidence,
            "warning": warning,
            "latency_ms": round(latency_ms, 3),
            "top_k": limit,
            "min_confidence": confidence,
            "filters": dict(filters or {}),
            "pointers": pointers,
            "outline_digest": outline_digest,
            "outline_version_id": outline_record.version_id if outline_record else None,
            "offline_mode": offline_mode,
        }

        emit(
            "retrieval.query",
            {
                "document_id": document.document_id,
                "strategy": strategy,
                "query_length": len(query_text),
                "pointer_count": len(pointers),
                "top_k": limit,
                "min_confidence": confidence,
                "result_confidence": result_confidence,
                "status": status,
                "latency_ms": round(latency_ms, 3),
                "fallback_reason": fallback_reason,
                "outline_digest": outline_digest,
                "outline_version_id": outline_record.version_id if outline_record else None,
                "document_tokens": document_tokens,
                "pointer_tokens": pointer_tokens,
                "tokens_saved": tokens_saved,
                "provider": provider_label,
                "provider_error": provider_error[:200] if provider_error else None,
                "offline_mode": offline_mode,
            },
        )
        return response

    # ------------------------------------------------------------------
    # Resolvers
    # ------------------------------------------------------------------
    def _resolve_document(self, requested: str | None) -> DocumentState | None:
        if requested:
            return self._lookup_document(requested)
        active = self._active_document()
        if active is not None:
            return active
        return None

    def _lookup_document(self, document_id: str) -> DocumentState | None:
        if callable(self.document_lookup):
            return self.document_lookup(document_id)
        return None

    def _active_document(self) -> DocumentState | None:
        if callable(self.active_document_provider):
            return self.active_document_provider()
        return None

    def _resolve_outline_memory(self) -> DocumentSummaryMemory | None:
        if isinstance(self.outline_memory, DocumentSummaryMemory):
            return self.outline_memory
        if callable(self.outline_memory):
            return self.outline_memory()
        return None

    def _lookup_outline(self, document_id: str):
        memory = self._resolve_outline_memory()
        if memory is None:
            return None
        return memory.get(document_id)

    def _resolve_embedding_index(self) -> DocumentEmbeddingIndex | None:
        if self.embedding_index is not None:
            return self.embedding_index
        if callable(self.embedding_index_resolver):
            return self.embedding_index_resolver()
        return None

    def _parse_snapshot_token(self, token: str | None) -> tuple[str | None, str | None]:
        """Parse snapshot_token into (tab_id, version_id) components.

        Uses non-strict mode to gracefully handle malformed tokens.
        """
        return parse_snapshot_token(token, strict=False)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _sanitize_top_k(self, value: int | None) -> int:
        limit = self.default_top_k if value is None else int(value)
        return max(1, min(limit, 12))

    def _sanitize_confidence(self, value: float | None) -> float:
        if value is None:
            return 0.3
        return max(0.0, min(float(value), 1.0))

    def _determine_confidence(
        self,
        strategy: str,
        offline_mode: bool,
        pointers: list[dict[str, Any]],
    ) -> str:
        """WS3 4.1.1: Return 'high' or 'low' based on retrieval quality."""
        # High confidence only when using semantic embeddings with results
        if strategy == "embedding" and not offline_mode and pointers:
            return "high"
        # Low confidence for fallback, offline mode, or no results
        return "low"

    def _build_confidence_warning(
        self,
        confidence: str,
        fallback_reason: str | None,
        offline_mode: bool,
    ) -> str | None:
        """WS3 4.1.1: Build warning message for low-confidence results."""
        if confidence == "high":
            return None
        # All low-confidence warnings now include explicit next-step guidance
        base_msg = ""
        if offline_mode:
            base_msg = "Embeddings unavailable; using heuristic search."
        elif fallback_reason == "no_embedding_matches":
            base_msg = "No semantic matches found; using regex/outline fallback."
        elif fallback_reason == "embedding_unavailable":
            base_msg = "Embedding index not ready; using heuristic search."
        elif fallback_reason == "provider_error":
            base_msg = "Embedding provider error; using regex fallback."
        else:
            base_msg = "Low confidence results."
        # Always add next-step guidance for low-confidence results
        return f"{base_msg} Use line_span with document_snapshot to verify content. Do NOT repeat this query."

    def _similarity_search(
        self,
        index: DocumentEmbeddingIndex,
        document_id: str,
        query: str,
        top_k: int,
        min_confidence: float,
    ) -> Sequence[EmbeddingMatch]:
        async def _invoke() -> Sequence[EmbeddingMatch]:
            return await index.similarity_search(
                document_id,
                query_text=query,
                top_k=top_k,
                min_score=min_confidence,
            )

        # Check if there's already a running event loop (common in Qt apps with qasync).
        try:
            asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        if has_running_loop:
            # We're inside an existing event loop - run in a separate thread with its own loop.
            # This avoids "Cannot run the event loop while another loop is running".
            import concurrent.futures

            def _run_in_thread() -> Sequence[EmbeddingMatch]:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(_invoke())
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_in_thread)
                return future.result(timeout=60.0)
        else:
            # No running loop - safe to use asyncio.run()
            return asyncio.run(_invoke())

    def _pointers_from_matches(
        self,
        matches: Sequence[EmbeddingMatch],
        *,
        document: DocumentState,
        include_outline_context: bool,
        outline_index: Mapping[str, OutlineNode] | None,
        line_offsets: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        text = document.text or ""
        # WS3 4.2.2: Compute line offsets if not provided
        if line_offsets is None:
            line_offsets = self._compute_line_offsets(text)
        pointers: list[dict[str, Any]] = []
        for match in matches:
            record = match.record
            preview = self._extract_preview(text, record.start_offset, record.end_offset)
            char_range = [record.start_offset, record.end_offset]
            pointer: dict[str, Any] = {
                "chunk_id": record.chunk_id,
                "pointer_id": self._pointer_id(document.document_id, record.chunk_id),
                "score": round(match.score, 4),
                "preview": preview,
                "token_estimate": max(1, int(record.token_count or 0)),
                "char_range": char_range,
                "line_span": self._line_span_for_char_range(char_range, line_offsets),
                "outline_node_id": record.outline_node_id,
            }
            if include_outline_context and outline_index and record.outline_node_id:
                outline_context = self._outline_context_from_id(
                    record.outline_node_id,
                    document.document_id,
                    outline_index,
                )
                if outline_context:
                    pointer["outline_context"] = outline_context
            pointers.append(pointer)
        return pointers

    def _fallback_search(
        self,
        *,
        document: DocumentState,
        query: str,
        limit: int,
        include_outline_context: bool,
        outline_index: Mapping[str, OutlineNode] | None,
        line_offsets: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        text = document.text or ""
        if not text:
            return []
        # WS3 4.2.2: Compute line offsets if not provided
        if line_offsets is None:
            line_offsets = self._compute_line_offsets(text)
        pattern = self._fallback_pattern(query)
        matches: list[dict[str, Any]] = []
        if pattern is not None:
            for match in pattern.finditer(text):
                start = max(0, match.start() - self.max_preview_chars // 2)
                end = min(len(text), match.end() + self.max_preview_chars // 2)
                matches.append(self._fallback_pointer(document, start, end, "regex", line_offsets=line_offsets))
                if len(matches) >= limit:
                    break
        if not matches and outline_index:
            lower_query = query.lower()
            for node in outline_index.values():
                if lower_query in node.text.lower() or lower_query in (node.blurb or "").lower():
                    matches.append(
                        self._fallback_pointer(
                            document,
                            node.char_range[0],
                            node.char_range[1],
                            "outline",
                            outline_node=node,
                            line_offsets=line_offsets,
                        )
                    )
                    if len(matches) >= limit:
                        break
        if not matches:
            matches.append(self._fallback_pointer(document, 0, self.max_preview_chars, "document", line_offsets=line_offsets))
        if include_outline_context and outline_index:
            for pointer in matches:
                if "outline_context" in pointer:
                    continue
                context = self._outline_context_for_range(
                    pointer["char_range"],
                    document.document_id,
                    outline_index,
                )
                if context:
                    pointer["outline_context"] = context
        return matches[:limit]

    def _fallback_pattern(self, query: str) -> re.Pattern[str] | None:
        candidate = query.strip()
        if len(candidate) < 3:
            return None
        escaped = re.escape(candidate[:128])
        return re.compile(escaped, re.IGNORECASE)

    def _fallback_pointer(
        self,
        document: DocumentState,
        start: int,
        end: int,
        match_type: str,
        *,
        outline_node: OutlineNode | None = None,
        line_offsets: list[int] | None = None,
    ) -> dict[str, Any]:
        text = document.text or ""
        start = max(0, min(len(text), start))
        end = max(start + 1, min(len(text), end))
        preview = self._extract_preview(text, start, end)
        pointer_id = self._pointer_id(document.document_id, f"fallback-{start}-{end}")
        char_range = [start, end]
        # WS3 4.2.2: Compute line offsets if not provided
        if line_offsets is None:
            line_offsets = self._compute_line_offsets(text)
        pointer: dict[str, Any] = {
            "chunk_id": f"fallback-{start}-{end}",
            "pointer_id": pointer_id,
            "score": 0.2,
            "preview": preview,
            "token_estimate": self._estimate_tokens(preview),
            "char_range": char_range,
            "line_span": self._line_span_for_char_range(char_range, line_offsets),
            "match_type": match_type,
        }
        if outline_node is not None:
            pointer["outline_context"] = self._outline_payload(outline_node, document.document_id)
        return pointer

    def _total_pointer_tokens(self, pointers: Sequence[Mapping[str, Any]]) -> int:
        total = 0
        for pointer in pointers:
            try:
                total += max(1, int(pointer.get("token_estimate", 0)))
            except (TypeError, ValueError):
                total += 1
        return total

    def _extract_preview(self, text: str, start: int, end: int) -> str:
        snippet = text[start:end]
        if len(snippet) <= self.max_preview_chars:
            return snippet.strip()
        return snippet[: self.max_preview_chars].rstrip() + "…"

    def _estimate_tokens(self, text: str) -> int:
        words = text.split()
        return max(1, len(words))

    # ------------------------------------------------------------------
    # WS3 4.2.x: Line span helpers
    # ------------------------------------------------------------------
    def _compute_line_offsets(self, text: str) -> list[int]:
        """Build list of line start offsets for offset-to-line conversion."""
        offsets = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                offsets.append(i + 1)
        return offsets

    def _line_for_offset(self, offset: int, line_offsets: list[int]) -> int:
        """Return the 0-based line index containing the given offset."""
        from bisect import bisect_right
        cursor = max(0, offset)
        index = bisect_right(line_offsets, cursor) - 1
        return max(0, index)

    def _line_span_for_char_range(
        self,
        char_range: Sequence[int],
        line_offsets: list[int],
    ) -> dict[str, int]:
        """WS3 4.2.1: Convert char_range to line_span."""
        start_offset = int(char_range[0]) if char_range else 0
        end_offset = int(char_range[1]) if len(char_range) > 1 else start_offset
        start_line = self._line_for_offset(start_offset, line_offsets)
        # Use end_offset - 1 for inclusive line calculation (same as document_snapshot)
        end_line = (
            self._line_for_offset(max(start_offset, end_offset - 1), line_offsets)
            if end_offset > start_offset
            else start_line
        )
        return {"start_line": start_line, "end_line": end_line}

    def _pointer_id(self, document_id: str, chunk_id: str) -> str:
        safe_chunk = chunk_id.replace(" ", "_")
        return f"chunk:{document_id}:{safe_chunk}"

    def _build_outline_index(self, nodes: Iterable[OutlineNode] | None) -> dict[str, OutlineNode]:
        index: dict[str, OutlineNode] = {}
        if not nodes:
            return index

        def visit(node: OutlineNode) -> None:
            index[node.id] = node
            for child in node.children:
                visit(child)

        for node in nodes:
            visit(node)
        return index

    def _outline_context_from_id(
        self,
        node_id: str,
        document_id: str,
        outline_index: Mapping[str, OutlineNode],
    ) -> dict[str, Any] | None:
        node = outline_index.get(node_id)
        if node is None:
            return None
        return self._outline_payload(node, document_id)

    def _outline_context_for_range(
        self,
        char_range: Sequence[int],
        document_id: str,
        outline_index: Mapping[str, OutlineNode],
    ) -> dict[str, Any] | None:
        if not char_range:
            return None
        start = int(char_range[0])
        end = int(char_range[1]) if len(char_range) > 1 else start
        best: OutlineNode | None = None
        for node in outline_index.values():
            node_start, node_end = node.char_range
            if start >= node_start and end <= node_end:
                if best is None or node.level >= best.level:
                    best = node
        if best is None:
            return None
        return self._outline_payload(best, document_id)

    def _outline_payload(self, node: OutlineNode, document_id: str) -> dict[str, Any]:
        return {
            "node_id": node.id,
            "heading": node.text,
            "level": node.level,
            "pointer_id": f"outline:{document_id}:{node.id}",
            "char_range": [node.char_range[0], node.char_range[1]],
        }
