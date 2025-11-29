"""Subagent pipeline coordination for chunk-level analysis."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TYPE_CHECKING

from ...services import telemetry as telemetry_service
from ..ai_types import (
    ChunkReference,
    SubagentBudget,
    SubagentJob,
    SubagentJobState,
    SubagentRuntimeConfig,
)
from .chunk_flow import ChunkContext
from .subagent_state import SubagentDocumentState

if TYPE_CHECKING:
    from .subagent_runtime import SubagentRuntimeManager
    from .controller import OpenAIToolSpec

LOGGER = logging.getLogger(__name__)

# File extensions treated as code documents
_CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".rs", ".go",
    ".rb", ".swift", ".kt", ".scala", ".cs", ".php", ".jsx", ".tsx",
})


class SubagentCoordinator:
    """Coordinates subagent pipeline execution for chat turns.
    
    Manages the lifecycle of subagent jobs, including:
    - Job planning based on document state and heuristics
    - Pipeline execution through SubagentRuntimeManager
    - Plot state and character map updates from job results
    - Document-level state tracking (edit churn, job history)
    """

    def __init__(
        self,
        config: SubagentRuntimeConfig,
        runtime_manager: "SubagentRuntimeManager | None" = None,
        tool_registry: Mapping[str, "OpenAIToolSpec"] | None = None,
        token_estimator: Callable[[str], int] | None = None,
        chunk_hydrator: Callable[[str, str, str | None, str | None], str] | None = None,
        document_id_resolver: Callable[[Mapping[str, Any]], str | None] | None = None,
        snapshot_span_resolver: Callable[[Mapping[str, Any]], tuple[int, int] | None] | None = None,
        max_context_tokens: int = 128_000,
        response_token_reserve: int = 16_000,
    ) -> None:
        """Initialize the subagent coordinator.
        
        Args:
            config: Subagent runtime configuration.
            runtime_manager: Manager for executing subagent jobs.
            tool_registry: Registry of available tools.
            token_estimator: Callable to estimate tokens in text.
            chunk_hydrator: Callable to hydrate chunk text from cache.
            document_id_resolver: Callable to resolve document ID from snapshot.
            snapshot_span_resolver: Callable to resolve snapshot span.
            max_context_tokens: Maximum context tokens for budget calculation.
            response_token_reserve: Reserved tokens for response.
        """
        self._config = config
        self._runtime = runtime_manager
        self._tools = tool_registry or {}
        self._estimate_tokens = token_estimator or (lambda x: len(x) // 4)
        self._hydrate_chunk = chunk_hydrator
        self._resolve_document_id = document_id_resolver or (lambda s: s.get("document_id"))
        self._resolve_snapshot_span = snapshot_span_resolver
        self._max_context_tokens = max_context_tokens
        self._response_token_reserve = response_token_reserve
        self._doc_states: dict[str, SubagentDocumentState] = {}

    @property
    def config(self) -> SubagentRuntimeConfig:
        """Return the current subagent configuration."""
        return self._config

    def update_config(self, config: SubagentRuntimeConfig) -> None:
        """Update the subagent configuration.
        
        Args:
            config: New configuration to apply.
        """
        self._config = config

    def update_runtime(self, runtime: "SubagentRuntimeManager | None") -> None:
        """Update the runtime manager.
        
        Args:
            runtime: New runtime manager or None.
        """
        self._runtime = runtime

    def update_tools(self, tools: Mapping[str, "OpenAIToolSpec"]) -> None:
        """Update the tool registry reference.
        
        Args:
            tools: New tool registry mapping.
        """
        self._tools = tools

    async def run_pipeline(
        self,
        *,
        prompt: str,
        snapshot: Mapping[str, Any],
        turn_context: dict[str, Any],
    ) -> tuple[list[SubagentJob], list[dict[str, str]]]:
        """Run the subagent pipeline for a chat turn.
        
        Args:
            prompt: User prompt text.
            snapshot: Document snapshot.
            turn_context: Mutable turn context for metrics.
            
        Returns:
            Tuple of (completed_jobs, system_messages).
        """
        runtime = self._runtime
        manager = runtime.manager if runtime else None
        document_id = self._resolve_document_id(snapshot) if self._resolve_document_id else None
        focus_span = self._resolve_snapshot_span(snapshot) if self._resolve_snapshot_span else None
        
        if manager is None or not self._config.enabled:
            LOGGER.debug(
                "Subagent pipeline skipped (manager=%s, enabled=%s, document_id=%s, span=%s)",
                bool(manager),
                self._config.enabled,
                document_id,
                focus_span,
            )
            return [], []

        jobs = self.plan_jobs(prompt, snapshot, turn_context)
        if not jobs:
            LOGGER.debug(
                "Subagent pipeline planned zero jobs (document_id=%s, span=%s)",
                document_id,
                focus_span,
            )
            return [], []

        results = await manager.run_jobs(jobs)
        LOGGER.debug(
            "Subagent pipeline completed %s job(s) (document_id=%s)",
            len(results),
            document_id,
        )
        
        summary_text = self._summary_message(results)
        messages: list[dict[str, str]] = []
        if summary_text:
            messages.append({"role": "system", "content": summary_text})
            
        plot_hint = self.maybe_update_plot_state(snapshot, results)
        if plot_hint:
            messages.append({"role": "system", "content": plot_hint})
            
        concordance_hint = self.maybe_update_character_map(snapshot, results)
        if concordance_hint:
            messages.append({"role": "system", "content": concordance_hint})
            
        turn_context["subagent_jobs"] = len(results)
        return results, messages

    def plan_jobs(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        turn_context: Mapping[str, Any],
    ) -> list[SubagentJob]:
        """Plan subagent jobs based on document state.
        
        Args:
            prompt: User prompt text.
            snapshot: Document snapshot with chunk manifest.
            turn_context: Turn context for metadata.
            
        Returns:
            List of planned SubagentJob instances.
        """
        config = self._config
        base_document_id = (
            self._resolve_document_id(snapshot) if self._resolve_document_id else None
        ) or "document"
        
        if not config.enabled:
            LOGGER.debug("Subagent planning skipped: config disabled (document_id=%s)", base_document_id)
            return []
            
        manifest = snapshot.get("chunk_manifest")
        if not isinstance(manifest, Mapping):
            LOGGER.debug("Subagent planning skipped: missing chunk manifest (document_id=%s)", base_document_id)
            return []
            
        chunks = manifest.get("chunks")
        if not isinstance(chunks, Sequence) or not chunks:
            LOGGER.debug("Subagent planning skipped: empty chunk manifest (document_id=%s)", base_document_id)
            return []
            
        document_id = str(
            manifest.get("document_id")
            or snapshot.get("document_id")
            or base_document_id
        )
        
        state = self._state_for_document(document_id)
        now = time.monotonic()
        
        # Check cooldown
        cooldown = max(0.0, config.helper_cooldown_seconds)
        if state.last_job_ts and now - state.last_job_ts < cooldown:
            LOGGER.debug(
                "Subagent planning skipped: helper cooldown active (document_id=%s, remaining=%.2fs)",
                document_id,
                cooldown - (now - state.last_job_ts),
            )
            return []
            
        # Check debounce
        debounce = max(0.0, config.edit_debounce_seconds)
        if state.last_edit_ts and now - state.last_edit_ts < debounce:
            LOGGER.debug(
                "Subagent planning skipped: edit debounce active (document_id=%s, remaining=%.2fs)",
                document_id,
                debounce - (now - state.last_edit_ts),
            )
            return []
            
        dirty_chunks = self._dirty_manifest_chunks(chunks, state)
        prioritized_source = dirty_chunks if dirty_chunks else chunks
        prioritized = self._prioritize_chunks(snapshot, prioritized_source)
        
        if not prioritized:
            LOGGER.debug("Subagent planning skipped: unable to prioritize chunks (document_id=%s)", document_id)
            return []
            
        reasons = self._trigger_reasons(snapshot, manifest, state, dirty_chunks)
        if not reasons:
            LOGGER.debug(
                "Subagent planning skipped: heuristics not satisfied (document_id=%s, chunk_count=%s, churn=%s)",
                document_id,
                len(chunks),
                state.edit_churn,
            )
            return []
            
        target_entry = prioritized[0]
        chunk_context = self._build_chunk_context(
            snapshot,
            manifest,
            target_entry,
            hydrate_text=True,
        )
        
        if chunk_context is None:
            LOGGER.debug(
                "Subagent planning skipped: manifest entry lacked context (document_id=%s)",
                document_id,
            )
            return []
            
        document_id = chunk_context.document_id or document_id
        preview_source = (chunk_context.text or "").strip()
        
        if not preview_source:
            LOGGER.debug(
                "Subagent planning skipped: unable to hydrate chunk text (document_id=%s, chunk_id=%s)",
                document_id,
                chunk_context.chunk_id,
            )
            return []
            
        preview = preview_source[: config.chunk_preview_chars].strip()
        if not preview:
            LOGGER.debug(
                "Subagent planning skipped: preview trimmed to empty string (document_id=%s, chunk_id=%s)",
                document_id,
                chunk_context.chunk_id,
            )
            return []
            
        version = manifest.get("version") or snapshot.get("version") or snapshot.get("document_version")
        chunk_hash = chunk_context.chunk_hash or self._hash_chunk(document_id, version, preview_source)
        pointer_id = chunk_context.pointer_id or f"chunk:{document_id}:{chunk_context.chunk_id}"
        token_estimate = self._estimate_tokens(preview_source)
        
        chunk_ref = ChunkReference(
            document_id=document_id,
            chunk_id=chunk_context.chunk_id,
            version_id=str(version) if version else None,
            pointer_id=pointer_id,
            char_range=chunk_context.char_range,
            token_estimate=token_estimate,
            chunk_hash=chunk_hash,
            preview=preview,
        )
        
        allowed_tools = tuple(tool for tool in config.allowed_tools if tool in self._tools)
        budget = self._build_budget(token_estimate)
        instructions = self._render_instructions(prompt, chunk_ref)
        
        job = SubagentJob(
            job_id=uuid.uuid4().hex,
            parent_run_id=str(turn_context.get("run_id") or uuid.uuid4().hex),
            instructions=instructions,
            chunk_ref=chunk_ref,
            allowed_tools=allowed_tools,
            budget=budget,
            dedup_hash=chunk_hash,
        )
        
        # Update state
        state.last_job_ts = now
        state.edit_churn = 0
        if chunk_hash:
            state.last_job_hashes[chunk_context.chunk_id] = chunk_hash
            
        # Update turn context
        if isinstance(turn_context, MutableMapping):
            turn_context.setdefault("subagent_trigger_reasons", list(reasons))
            turn_context["subagent_focus_chunk"] = chunk_context.chunk_id
            turn_context["subagent_focus_document"] = document_id
            
        self._emit_queue_event(document_id, (job,), reasons)
        
        LOGGER.debug(
            "Subagent planning created job (document_id=%s, chunk_id=%s, tokens=%s, tools=%s, reasons=%s)",
            document_id,
            chunk_context.chunk_id,
            token_estimate,
            len(allowed_tools),
            ",".join(reasons),
        )
        return [job]

    def _state_for_document(self, document_id: str) -> SubagentDocumentState:
        """Get or create state for a document.
        
        Args:
            document_id: Document identifier.
            
        Returns:
            SubagentDocumentState for the document.
        """
        state = self._doc_states.get(document_id)
        if state is None:
            state = SubagentDocumentState()
            self._doc_states[document_id] = state
        return state

    def _infer_document_kind(self, snapshot: Mapping[str, Any]) -> str:
        """Infer document type (code vs prose).
        
        Args:
            snapshot: Document snapshot.
            
        Returns:
            'code' or 'prose'.
        """
        format_hint = str(snapshot.get("document_format") or snapshot.get("format") or "").lower()
        if format_hint in {"code", "python", "notebook"}:
            return "code"
        path_value = snapshot.get("path") or snapshot.get("tab_name")
        if isinstance(path_value, str) and path_value:
            suffix = Path(path_value).suffix.lower()
            if suffix in _CODE_EXTENSIONS:
                return "code"
        return "prose"

    def _prioritize_chunks(
        self,
        snapshot: Mapping[str, Any],
        chunks: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """Prioritize chunks by distance from focus center.
        
        Args:
            snapshot: Document snapshot with focus info.
            chunks: List of chunk entries to prioritize.
            
        Returns:
            Chunks sorted by proximity to focus.
        """
        center = self._span_center(snapshot)
        if center is None:
            center = self._window_focus_center(snapshot)
        if center is None:
            return [dict(chunk) for chunk in chunks]

        def _chunk_center(entry: Mapping[str, Any]) -> int:
            start = self._coerce_index(entry.get("start"), 0)
            end = self._coerce_index(entry.get("end"), start)
            width = max(1, end - start)
            return start + width // 2

        return sorted(chunks, key=lambda entry: abs(_chunk_center(entry) - center))

    def _dirty_manifest_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        state: SubagentDocumentState,
    ) -> list[Mapping[str, Any]]:
        """Find chunks that have changed since last job.
        
        Args:
            chunks: All chunks in manifest.
            state: Document state with job history.
            
        Returns:
            List of dirty chunk entries.
        """
        dirty: list[Mapping[str, Any]] = []
        for entry in chunks:
            if not isinstance(entry, Mapping):
                continue
            chunk_id = str(entry.get("id") or "").strip()
            if not chunk_id:
                continue
            chunk_hash = entry.get("hash")
            normalized_hash = str(chunk_hash).strip() if isinstance(chunk_hash, str) and chunk_hash.strip() else None
            if not normalized_hash or state.last_job_hashes.get(chunk_id) != normalized_hash:
                dirty.append(entry)
        return dirty

    def _build_chunk_context(
        self,
        snapshot: Mapping[str, Any],
        manifest: Mapping[str, Any],
        entry: Mapping[str, Any],
        *,
        hydrate_text: bool,
    ) -> ChunkContext | None:
        """Build chunk context from manifest entry.
        
        Args:
            snapshot: Document snapshot.
            manifest: Chunk manifest.
            entry: Chunk entry from manifest.
            hydrate_text: Whether to hydrate chunk text.
            
        Returns:
            ChunkContext or None if invalid.
        """
        chunk_id = str(entry.get("id") or "").strip()
        if not chunk_id:
            return None
            
        start = self._coerce_index(entry.get("start"), 0)
        end = self._coerce_index(entry.get("end"), start + 1)
        if end <= start:
            end = start + 1
            
        document_id = str(
            manifest.get("document_id")
            or (self._resolve_document_id(snapshot) if self._resolve_document_id else None)
            or snapshot.get("document_id")
            or "document"
        )
        
        chunk_hash = entry.get("hash")
        pointer_id = entry.get("outline_pointer_id")
        
        context = ChunkContext(
            chunk_id=chunk_id,
            document_id=document_id,
            char_range=(start, end),
            chunk_hash=str(chunk_hash).strip() if isinstance(chunk_hash, str) and chunk_hash.strip() else None,
            pointer_id=str(pointer_id).strip() if isinstance(pointer_id, str) and pointer_id.strip() else None,
        )
        
        if hydrate_text and self._hydrate_chunk is not None:
            cache_key = manifest.get("cache_key")
            version = manifest.get("version") or snapshot.get("version") or snapshot.get("document_version")
            text = self._hydrate_chunk(
                chunk_id,
                document_id,
                str(cache_key).strip() if isinstance(cache_key, str) and cache_key.strip() else None,
                str(version).strip() if isinstance(version, str) and version else None,
            )
            context.text = text or None
            
        return context

    def _trigger_reasons(
        self,
        snapshot: Mapping[str, Any],
        manifest: Mapping[str, Any],
        state: SubagentDocumentState,
        dirty_chunks: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        """Determine reasons for triggering subagent jobs.
        
        Args:
            snapshot: Document snapshot.
            manifest: Chunk manifest.
            state: Document state.
            dirty_chunks: List of dirty chunks.
            
        Returns:
            List of trigger reason strings.
        """
        config = self._config
        chunk_entries = manifest.get("chunks")
        chunk_count = len(chunk_entries) if isinstance(chunk_entries, Sequence) else 0
        reasons: list[str] = []
        
        if dirty_chunks:
            reasons.append("dirty_chunks")
        if chunk_count >= max(1, config.chunk_trigger_threshold):
            reasons.append("chunk_threshold")
        if self._infer_document_kind(snapshot) == "code" and chunk_count >= max(1, config.code_chunk_trigger):
            reasons.append("code_format")
        if state.edit_churn >= max(0, config.edit_churn_threshold):
            reasons.append("edit_churn")
            
        return reasons

    def _build_budget(self, token_estimate: int) -> SubagentBudget:
        """Build subagent budget from token estimate.
        
        Args:
            token_estimate: Estimated tokens in chunk.
            
        Returns:
            SubagentBudget instance.
        """
        prompt_cap = min(self._max_context_tokens // 2, max(512, token_estimate + 256))
        reserve_slice = max(256, self._response_token_reserve // 4 if self._response_token_reserve else 256)
        completion_cap = min(reserve_slice, self._response_token_reserve or reserve_slice)
        budget = SubagentBudget(
            max_prompt_tokens=prompt_cap,
            max_completion_tokens=max(256, completion_cap),
            max_runtime_seconds=45.0,
            max_tool_iterations=0,
        )
        return budget.clamp()

    def _render_instructions(self, prompt: str, chunk_ref: ChunkReference) -> str:
        """Render subagent instructions.
        
        Args:
            prompt: User prompt.
            chunk_ref: Chunk reference.
            
        Returns:
            Instruction string for subagent.
        """
        base_prompt = self._config.instructions_template.strip()
        user_prompt = prompt.strip() or "(no additional user guidance provided)"
        document_label = chunk_ref.document_id or "document"
        return (
            f"{base_prompt}\n\nUser prompt:\n{user_prompt}\n\n"
            f"Focus on document '{document_label}' chunk {chunk_ref.chunk_id}. "
            "Summaries must stay under 200 tokens and include:"
            "\n1. Current intent\n2. Risks or continuity gaps\n3. Recommended follow-up tools."
        )

    def _summary_message(self, jobs: Sequence[SubagentJob]) -> str:
        """Build summary message from completed jobs.
        
        Args:
            jobs: Completed subagent jobs.
            
        Returns:
            Summary message string.
        """
        if not jobs:
            return ""
        lines: list[str] = []
        for job in jobs:
            if job.state != SubagentJobState.SUCCEEDED or job.result is None:
                continue
            summary = (job.result.summary or "").strip()
            if not summary:
                continue
            label = job.chunk_ref.chunk_id or job.job_id
            trimmed = summary if len(summary) <= 280 else f"{summary[:277].rstrip()}…"
            lines.append(f"- {label}: {trimmed}")
            if len(lines) >= 4:
                break
        if not lines:
            if any(job.state == SubagentJobState.FAILED for job in jobs):
                return "Subagent scouting report: helper job failed; rerun if deeper analysis is required."
            return ""
        header = "Subagent scouting report (chunk-level analysis):"
        return "\n".join([header, *lines])

    def maybe_update_plot_state(
        self,
        snapshot: Mapping[str, Any],
        jobs: Sequence[SubagentJob],
    ) -> str | None:
        """Update plot state from job results.
        
        Args:
            snapshot: Document snapshot.
            jobs: Completed subagent jobs.
            
        Returns:
            Hint message or None.
        """
        document_id = self._resolve_document_id(snapshot) if self._resolve_document_id else None
        if not jobs:
            LOGGER.debug(
                "Plot state update skipped: no subagent jobs (document_id=%s)",
                document_id,
            )
            return None
        if not self._config.plot_scaffolding_enabled:
            LOGGER.debug(
                "Plot state update skipped: plot scaffolding disabled (document_id=%s)",
                document_id,
            )
            return None
        runtime = self._runtime
        if runtime is None:
            LOGGER.debug(
                "Plot state update skipped: subagent runtime unavailable (document_id=%s)",
                document_id,
            )
            return None
        store = runtime.ensure_plot_state_store()

        ingested = 0
        for job in jobs:
            if job.state != SubagentJobState.SUCCEEDED or job.result is None:
                continue
            summary = (job.result.summary or "").strip()
            if not summary:
                continue
            chunk = job.chunk_ref
            target_document_id = chunk.document_id or document_id
            if not target_document_id:
                continue
            store.ingest_chunk_summary(
                target_document_id,
                summary,
                version_id=chunk.version_id,
                chunk_hash=chunk.chunk_hash,
                pointer_id=chunk.pointer_id,
                metadata={
                    "source_job_id": job.job_id,
                    "tokens_used": job.result.tokens_used,
                    "latency_ms": job.result.latency_ms,
                },
            )
            ingested += 1

        if not ingested:
            LOGGER.debug(
                "Plot state update skipped: no ingested summaries (document_id=%s)",
                document_id,
            )
            return None

        doc_label = document_id or jobs[0].chunk_ref.document_id or "document"
        LOGGER.debug(
            "Plot state updated via %s ingested chunk(s) (document_id=%s)",
            ingested,
            doc_label,
        )
        return (
            f"Plot scaffolding refreshed for '{doc_label}'. Call PlotOutlineTool before editing for continuity "
            "and follow up with PlotStateUpdateTool after applying chunk edits."
        )

    def maybe_update_character_map(
        self,
        snapshot: Mapping[str, Any],
        jobs: Sequence[SubagentJob],
    ) -> str | None:
        """Update character map from job results.
        
        Args:
            snapshot: Document snapshot.
            jobs: Completed subagent jobs.
            
        Returns:
            Hint message or None.
        """
        document_id = self._resolve_document_id(snapshot) if self._resolve_document_id else None
        if not jobs:
            LOGGER.debug(
                "Character map update skipped: no subagent jobs (document_id=%s)",
                document_id,
            )
            return None
        if not self._config.plot_scaffolding_enabled:
            LOGGER.debug(
                "Character map update skipped: plot scaffolding disabled (document_id=%s)",
                document_id,
            )
            return None
        runtime = self._runtime
        if runtime is None:
            LOGGER.debug(
                "Character map update skipped: subagent runtime unavailable (document_id=%s)",
                document_id,
            )
            return None
        store = runtime.ensure_character_map_store()

        ingested = 0
        for job in jobs:
            if job.state != SubagentJobState.SUCCEEDED or job.result is None:
                continue
            summary = (job.result.summary or "").strip()
            if not summary:
                continue
            chunk = job.chunk_ref
            target_document_id = chunk.document_id or document_id
            if not target_document_id:
                continue
            store.ingest_summary(
                target_document_id,
                summary,
                version_id=chunk.version_id,
                chunk_id=chunk.chunk_id,
                pointer_id=chunk.pointer_id,
                chunk_hash=chunk.chunk_hash,
                char_range=chunk.char_range,
            )
            ingested += 1

        if not ingested:
            LOGGER.debug(
                "Character map update skipped: no ingested summaries (document_id=%s)",
                document_id,
            )
            return None

        doc_label = document_id or jobs[0].chunk_ref.document_id or "document"
        LOGGER.debug(
            "Character map updated via %s ingested chunk(s) (document_id=%s)",
            ingested,
            doc_label,
        )
        return (
            f"Character concordance refreshed for '{doc_label}'. Call CharacterMapTool to review "
            "entity mentions before editing across scenes."
        )

    def handle_document_changed(self, document_id: str) -> None:
        """Handle document change event.
        
        Updates edit churn tracking for the document.
        
        Args:
            document_id: Changed document identifier.
        """
        state = self._doc_states.setdefault(document_id, SubagentDocumentState())
        state.edit_churn += 1
        state.last_edit_ts = time.time()

    def handle_document_closed(self, document_id: str) -> None:
        """Handle document close event.
        
        Cleans up state for the closed document.
        
        Args:
            document_id: Closed document identifier.
        """
        self._doc_states.pop(document_id, None)

    def _emit_queue_event(
        self,
        document_id: str,
        jobs: Sequence[SubagentJob],
        reasons: Sequence[str],
    ) -> None:
        """Emit telemetry for queued jobs.
        
        Args:
            document_id: Document identifier.
            jobs: Queued jobs.
            reasons: Trigger reasons.
        """
        if not jobs:
            return
        payload = {
            "document_id": document_id,
            "job_count": len(jobs),
            "job_ids": [job.job_id for job in jobs if job.job_id],
            "chunk_ids": [job.chunk_id for job in jobs if job.chunk_id],
            "chunk_hashes": [job.chunk_hash for job in jobs if job.chunk_hash],
            "reasons": list(reasons),
        }
        telemetry_service.emit("subagent.jobs_queued", payload)

    def _span_center(self, snapshot: Mapping[str, Any]) -> int | None:
        """Get center of snapshot span.
        
        Args:
            snapshot: Document snapshot.
            
        Returns:
            Center index or None.
        """
        if self._resolve_snapshot_span is None:
            return None
        span = self._resolve_snapshot_span(snapshot)
        if span is None:
            return None
        start, end = span
        if end <= start:
            return None
        return start + (end - start) // 2

    def _window_focus_center(self, snapshot: Mapping[str, Any]) -> int | None:
        """Get center of window focus.
        
        Args:
            snapshot: Document snapshot.
            
        Returns:
            Center index or None.
        """
        candidates = (
            snapshot.get("text_range"),
            snapshot.get("window"),
        )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            start = self._coerce_index(candidate.get("start"), 0)
            end = self._coerce_index(candidate.get("end"), start)
            if end > start:
                return start + (end - start) // 2
        return None

    @staticmethod
    def _coerce_index(value: Any, default: int) -> int:
        """Coerce value to int index.
        
        Args:
            value: Value to coerce.
            default: Default if coercion fails.
            
        Returns:
            Integer index.
        """
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _hash_chunk(document_id: str, version: Any, chunk_text: str) -> str:
        """Hash chunk content for deduplication.
        
        Args:
            document_id: Document identifier.
            version: Document version.
            chunk_text: Chunk content.
            
        Returns:
            SHA1 hash string.
        """
        token = f"{document_id}:{version}:{chunk_text}".encode("utf-8", errors="ignore")
        return hashlib.sha1(token).hexdigest()
