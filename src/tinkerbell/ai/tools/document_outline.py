"""Tool returning cached document outlines with budget-aware trimming."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence

from ...editor.document_model import DocumentState
from ..memory.buffers import DocumentSummaryMemory, OutlineNode, SummaryRecord
from ..services.context_policy import BudgetDecision, ContextBudgetPolicy
from ...services.telemetry import count_text_tokens, emit
from ..utils.document_checks import document_size_bytes, unsupported_format_reason

DocumentResolver = Callable[[str], DocumentState | None]
MemoryResolver = Callable[[], DocumentSummaryMemory | None] | DocumentSummaryMemory


@dataclass(slots=True)
class DocumentOutlineTool:
    """Expose cached outlines with pointer metadata and budget-aware trimming.

    Parameters
    ----------
    document_id: str | None
        Optional explicit target. When omitted the active document is used.
    desired_levels: int | None
        Depth limit (heading level) applied before budgeting. ``None`` means all.
    include_blurbs: bool
        Attach per-node blurbs/excerpts when true (default).
    max_nodes: int | None
        Hard cap on node count prior to budget enforcement; defaults to
        :attr:`default_max_nodes`.

    Returns
    -------
    dict
        Response payload containing ``outline_digest`` (hash), ``version_id``,
        ``generated_at`` ISO timestamp, ``is_stale`` flag, and ``nodes`` where
        each entry includes ``pointer_id = "outline:{document_id}:{node_id}"``.
        Additional metadata calls out trimming reasons, token budget usage, and
        document staleness deltas to help the controller decide next steps.
    """

    memory_resolver: MemoryResolver
    document_lookup: DocumentResolver | None = None
    active_document_provider: Callable[[], DocumentState | None] | None = None
    budget_policy: ContextBudgetPolicy | None = None
    max_outline_tokens: int = 4096
    default_max_nodes: int = 120
    pending_outline_checker: Callable[[str], bool] | None = None
    pending_retry_ms: int = 750

    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        document_id: str | None = None,
        desired_levels: int | None = None,
        include_blurbs: bool = True,
        max_nodes: int | None = None,
    ) -> dict[str, Any]:
        memory = self._resolve_memory()
        if memory is None:
            self._emit_outline_miss(document_id, "outline_memory_uninitialized", "outline_unavailable")
            return {
                "status": "outline_unavailable",
                "reason": "outline_memory_uninitialized",
            }

        target_id = self._resolve_document_id(document_id)
        if not target_id:
            self._emit_outline_miss(document_id, "no_document_id", "no_document")
            return {
                "status": "no_document",
                "reason": "no_document_id",
            }

        record = memory.get(target_id)
        document = self._lookup_document(target_id)
        unsupported_reason = self._resolve_unsupported_reason(document, record)
        if unsupported_reason:
            self._emit_outline_miss(target_id, "unsupported_format", "unsupported_format")
            return {
                "status": "unsupported_format",
                "reason": unsupported_reason,
                "document_id": target_id,
                "outline_available": False,
            }

        if record is None or not record.nodes:
            self._emit_outline_miss(target_id, "outline_missing", "outline_missing")
            return {
                "status": "outline_missing",
                "document_id": target_id,
                "outline_available": False,
            }

        if self._should_report_pending(document, record):
            retry_after = self._pending_retry_delay()
            self._emit_outline_miss(target_id, "outline_pending", "pending")
            return {
                "status": "pending",
                "reason": "outline_pending",
                "document_id": target_id,
                "outline_available": False,
                "retry_after_ms": retry_after,
            }

        level_limit = self._sanitize_level_limit(desired_levels)
        node_limit = self._sanitize_node_limit(max_nodes)
        nodes_copy, truncated_by_request = self._copy_nodes(
            record.nodes,
            document_id=target_id,
            include_blurbs=include_blurbs,
            level_limit=level_limit,
            node_limit=node_limit,
        )

        token_budget = self._token_budget()
        token_count = self._estimate_total_tokens(nodes_copy)
        trimmed_reason: str | None = "request_limit" if truncated_by_request else None
        if token_budget is not None and token_count > token_budget:
            nodes_copy, token_count, trimmed = self._trim_to_budget(nodes_copy, token_budget)
            if trimmed:
                truncated_by_request = True
                trimmed_reason = "token_budget"

        document_version = document.version_id if document else None
        is_stale = (
            document_version is not None
            and record.version_id is not None
            and document_version > record.version_id
        )
        metadata = dict(getattr(record, "metadata", {}) or {})
        document_bytes = metadata.get("document_bytes")
        if document_bytes is None and document is not None:
            document_bytes = document_size_bytes(document)
        guardrails = self._guardrail_messages(metadata)

        decision_payload: dict[str, Any] | None = None
        if self.budget_policy is not None:
            decision = self._record_budget_decision(target_id, token_count)
            decision_payload = decision.as_payload()

        node_count = self._count_nodes(nodes_copy)
        response = {
            "status": "ok" if not is_stale else "stale",
            "document_id": target_id,
            "version_id": record.version_id,
            "generated_at": self._format_timestamp(record.updated_at),
            "outline_digest": record.outline_hash,
            "outline_available": True,
            "nodes": nodes_copy,
            "node_count": node_count,
            "levels_returned": self._max_level(nodes_copy),
            "desired_levels": level_limit,
            "max_nodes_requested": max_nodes,
            "max_nodes_applied": node_limit,
            "include_blurbs": bool(include_blurbs),
            "token_budget": token_budget,
            "token_count": token_count,
            "trimmed": truncated_by_request,
            "trimmed_reason": trimmed_reason,
            "is_stale": is_stale,
            "document_version": document_version,
            "stale_delta": self._stale_delta(document_version, record.version_id),
            "budget_decision": decision_payload,
            "document_bytes": document_bytes,
            "outline_metadata": metadata,
        }
        if guardrails:
            response["guardrails"] = guardrails
        document_tokens = self._estimate_document_tokens(document)
        tokens_saved = self._compute_tokens_saved(document_tokens, token_count)
        self._emit_outline_hit(record, response, document_tokens=document_tokens, tokens_saved=tokens_saved)
        if is_stale:
            self._emit_outline_stale(record, response)
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_memory(self) -> DocumentSummaryMemory | None:
        if isinstance(self.memory_resolver, DocumentSummaryMemory):
            return self.memory_resolver
        if callable(self.memory_resolver):
            return self.memory_resolver()
        return None

    def _resolve_document_id(self, requested: str | None) -> str | None:
        candidate = (requested or "").strip()
        if candidate:
            return candidate
        document = self._active_document()
        if document is not None:
            return document.document_id
        return None

    def _active_document(self) -> DocumentState | None:
        if callable(self.active_document_provider):
            return self.active_document_provider()
        return None

    def _lookup_document(self, document_id: str) -> DocumentState | None:
        if callable(self.document_lookup):
            return self.document_lookup(document_id)
        return None

    def _resolve_unsupported_reason(
        self,
        document: DocumentState | None,
        record: SummaryRecord | None,
    ) -> str | None:
        if document is not None:
            reason = unsupported_format_reason(document)
            if reason:
                return reason
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, Mapping):
            reason = metadata.get("unsupported_format")
            if isinstance(reason, str) and reason:
                return reason
        return None

    def _should_report_pending(self, document: DocumentState | None, record: SummaryRecord) -> bool:
        if document is None:
            return False
        checker = self.pending_outline_checker
        if not callable(checker):
            return False
        document_version = getattr(document, "version_id", None)
        record_version = getattr(record, "version_id", None)
        if document_version is None or record_version is None:
            return False
        if document_version <= record_version:
            return False
        try:
            return bool(checker(record.document_id))
        except Exception:  # pragma: no cover - defensive guard
            return False

    def _pending_retry_delay(self) -> int:
        delay = max(250, min(int(self.pending_retry_ms), 4000))
        return delay

    def _sanitize_level_limit(self, desired: int | None) -> int | None:
        if desired is None:
            return None
        level = max(1, int(desired))
        return level

    def _sanitize_node_limit(self, requested: int | None) -> int:
        limit = self.default_max_nodes if requested is None else requested
        limit = max(1, min(int(limit), 1000))
        return limit

    def _copy_nodes(
        self,
        nodes: Sequence[OutlineNode],
        *,
        document_id: str,
        include_blurbs: bool,
        level_limit: int | None,
        node_limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        state = {"count": 0, "truncated": False}
        copied: list[dict[str, Any]] = []
        for node in nodes:
            clone = self._copy_node(
                node,
                document_id=document_id,
                include_blurbs=include_blurbs,
                level_limit=level_limit,
                node_limit=node_limit,
                state=state,
            )
            if clone is not None:
                copied.append(clone)
        return copied, bool(state["truncated"])

    def _copy_node(
        self,
        node: OutlineNode,
        *,
        document_id: str,
        include_blurbs: bool,
        level_limit: int | None,
        node_limit: int,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if level_limit is not None and node.level > level_limit:
            state["truncated"] = True
            return None
        if state["count"] >= node_limit:
            state["truncated"] = True
            return None
        state["count"] += 1
        payload: dict[str, Any] = {
            "id": node.id,
            "pointer_id": self._pointer_id(document_id, node.id),
            "level": node.level,
            "text": node.text,
            "char_range": [node.char_range[0], node.char_range[1]],
            "chunk_id": node.chunk_id,
            "token_estimate": max(1, int(node.token_estimate or 0)),
            "truncated": bool(node.truncated),
        }
        if include_blurbs and node.blurb:
            payload["blurb"] = node.blurb
        children_payload: list[dict[str, Any]] = []
        for child in node.children:
            child_clone = self._copy_node(
                child,
                document_id=document_id,
                include_blurbs=include_blurbs,
                level_limit=level_limit,
                node_limit=node_limit,
                state=state,
            )
            if child_clone is not None:
                children_payload.append(child_clone)
        payload["children"] = children_payload
        return payload

    def _pointer_id(self, document_id: str, node_id: str) -> str:
        return f"outline:{document_id}:{node_id}"

    def _guardrail_messages(self, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
        guardrails: list[dict[str, Any]] = []
        if metadata.get("huge_document_guardrail"):
            guardrails.append(
                {
                    "type": "huge_document",
                    "message": "Document exceeds safe outline size; only top-level sections included.",
                    "action": "Use targeted scan tools for deeper sections.",
                }
            )
        return guardrails

    def _token_budget(self) -> int | None:
        budget = max(0, int(self.max_outline_tokens))
        policy = self.budget_policy
        if policy is not None and policy.prompt_budget:
            policy_budget = max(0, int(policy.prompt_budget))
            budget = min(budget, policy_budget)
        return budget if budget > 0 else None

    def _estimate_total_tokens(self, nodes: Sequence[dict[str, Any]]) -> int:
        total = 0
        for node in nodes:
            total += int(node.get("token_estimate", 0))
            children = node.get("children")
            if children:
                total += self._estimate_total_tokens(children)
        return total

    def _trim_to_budget(
        self,
        nodes: list[dict[str, Any]],
        token_budget: int,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        flat = self._flatten_nodes(nodes)
        token_count = sum(self._node_token_cost(node) for node, _ in flat)
        trimmed = False
        for node, container in reversed(flat):
            if token_count <= token_budget:
                break
            if node not in container:
                continue
            container.remove(node)
            token_count -= self._subtree_tokens(node)
            trimmed = True
        self._cleanup_subtree_tokens(nodes)
        return nodes, token_count, trimmed

    def _flatten_nodes(self, nodes: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        flat: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

        def visit(node: dict[str, Any], container: list[dict[str, Any]]) -> int:
            flat.append((node, container))
            subtotal = self._node_token_cost(node)
            children = node.get("children") or []
            for child in children:
                subtotal += visit(child, children)
            node["_subtree_tokens"] = subtotal
            return subtotal

        for item in nodes:
            visit(item, nodes)
        return flat

    def _node_token_cost(self, node: dict[str, Any]) -> int:
        return max(1, int(node.get("token_estimate", 1)))

    def _subtree_tokens(self, node: dict[str, Any]) -> int:
        return int(node.get("_subtree_tokens", self._node_token_cost(node)))

    def _cleanup_subtree_tokens(self, nodes: Iterable[dict[str, Any]]) -> None:
        for node in nodes:
            node.pop("_subtree_tokens", None)
            children = node.get("children") or []
            if children:
                self._cleanup_subtree_tokens(children)

    def _count_nodes(self, nodes: Sequence[dict[str, Any]]) -> int:
        total = 0
        for node in nodes:
            total += 1
            children = node.get("children") or []
            if children:
                total += self._count_nodes(children)
        return total

    def _max_level(self, nodes: Sequence[dict[str, Any]]) -> int:
        max_level = 0
        for node in nodes:
            level = int(node.get("level", 0))
            max_level = max(max_level, level)
            children = node.get("children") or []
            if children:
                max_level = max(max_level, self._max_level(children))
        return max_level

    def _stale_delta(self, document_version: int | None, record_version: int | None) -> int | None:
        if document_version is None or record_version is None:
            return None
        return max(0, document_version - record_version)

    def _record_budget_decision(self, document_id: str, pending_tokens: int) -> BudgetDecision:
        policy = self.budget_policy or ContextBudgetPolicy.disabled(model_name=None)
        return policy.tokens_available(
            prompt_tokens=0,
            pending_tool_tokens=max(0, int(pending_tokens)),
            document_id=document_id,
        )

    def _format_timestamp(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat()

    def _estimate_document_tokens(self, document: DocumentState | None) -> int | None:
        if document is None:
            return None
        text = document.text or ""
        if not text.strip():
            return 0
        return count_text_tokens(text, estimate_only=True)

    def _compute_tokens_saved(self, document_tokens: int | None, outline_tokens: int | None) -> int | None:
        if document_tokens is None or outline_tokens is None:
            return None
        return max(0, int(document_tokens) - int(outline_tokens))

    def _outline_age_seconds(self, record: SummaryRecord) -> float | None:
        updated = getattr(record, "updated_at", None)
        if updated is None:
            return None
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - updated
        return max(0.0, float(delta.total_seconds()))

    def _emit_outline_hit(
        self,
        record: SummaryRecord,
        response: Mapping[str, Any],
        *,
        document_tokens: int | None,
        tokens_saved: int | None,
    ) -> None:
        payload = {
            "document_id": record.document_id,
            "outline_digest": record.outline_hash,
            "outline_version_id": record.version_id,
            "document_version": response.get("document_version"),
            "status": response.get("status"),
            "node_count": response.get("node_count"),
            "token_count": response.get("token_count"),
            "token_budget": response.get("token_budget"),
            "trimmed": response.get("trimmed"),
            "trimmed_reason": response.get("trimmed_reason"),
            "levels_returned": response.get("levels_returned"),
            "desired_levels": response.get("desired_levels"),
            "document_tokens": document_tokens,
            "tokens_saved": tokens_saved,
            "outline_age_seconds": self._outline_age_seconds(record),
            "is_stale": response.get("is_stale"),
            "stale_delta": response.get("stale_delta"),
        }
        emit("outline.tool.hit", payload)

    def _emit_outline_miss(self, document_id: str | None, reason: str, status: str) -> None:
        emit(
            "outline.tool.miss",
            {
                "document_id": document_id,
                "reason": reason,
                "status": status,
            },
        )

    def _emit_outline_stale(self, record: SummaryRecord, response: Mapping[str, Any]) -> None:
        emit(
            "outline.stale",
            {
                "document_id": record.document_id,
                "outline_digest": record.outline_hash,
                "outline_version_id": record.version_id,
                "document_version": response.get("document_version"),
                "stale_delta": response.get("stale_delta"),
                "node_count": response.get("node_count"),
                "token_count": response.get("token_count"),
                "outline_age_seconds": self._outline_age_seconds(record),
            },
        )
