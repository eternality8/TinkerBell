"""Subagent sandbox controller used by :class:`AIController`."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable, Deque, Mapping, MutableMapping, Sequence, cast

from ...ai_types import SubagentJob, SubagentJobResult, SubagentJobState, SubagentRuntimeConfig
from ...client import AIClient
from ...memory.result_cache import SubagentResultCache
from ...services import telemetry as telemetry_service
from ...services.context_policy import BudgetDecision, ContextBudgetPolicy

LOGGER = logging.getLogger(__name__)


class SubagentJobQueue:
    """Bounded FIFO queue that deduplicates jobs per turn."""

    def __init__(self, *, max_jobs: int) -> None:
        self.max_jobs = max(0, int(max_jobs))
        self._queue: Deque[SubagentJob] = deque()
        self._dedup: set[str] = set()

    def enqueue(self, job: SubagentJob) -> bool:
        if self.max_jobs and len(self._queue) >= self.max_jobs:
            return False
        dedup = job.dedup_hash
        if dedup and dedup in self._dedup:
            return False
        self._queue.append(job)
        if dedup:
            self._dedup.add(dedup)
        return True

    def popleft(self) -> SubagentJob | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return len(self._queue)


class SubagentExecutor:
    """Runs a single subagent job using the shared AI client."""

    def __init__(
        self,
        client: AIClient,
        *,
        tool_resolver: Callable[[], Mapping[str, Any]],
        budget_policy: ContextBudgetPolicy | None = None,
        config: SubagentRuntimeConfig | None = None,
    ) -> None:
        self._client = client
        self._tool_resolver = tool_resolver
        self._budget_policy = budget_policy
        self._config = (config or SubagentRuntimeConfig()).clamp()

    def update_client(self, client: AIClient) -> None:
        self._client = client

    def update_budget_policy(self, policy: ContextBudgetPolicy | None) -> None:
        self._budget_policy = policy

    def update_config(self, config: SubagentRuntimeConfig) -> None:
        self._config = config.clamp()

    async def execute(self, job: SubagentJob) -> SubagentJob:
        chunk_preview = (job.chunk_ref.preview or "").strip()
        if not chunk_preview:
            job.state = SubagentJobState.SKIPPED
            job.result = SubagentJobResult(
                status="skipped",
                summary="Subagent skipped because the chunk preview was empty.",
            )
            telemetry_service.emit(
                "subagent.job_skipped",
                {
                    "job_id": job.job_id,
                    "document_id": job.document_id,
                    "chunk_id": job.chunk_id,
                    "reason": "empty_chunk",
                },
            )
            return job

        chunk_text = self._trim_chunk(chunk_preview, job.budget.max_prompt_tokens)
        messages = self._build_messages(job, chunk_text)
        prompt_tokens = sum(self._estimate_tokens(msg.get("content", "")) + 4 for msg in messages)
        decision = self._evaluate_budget(job, prompt_tokens)
        if decision and decision.verdict == "reject" and not decision.dry_run:
            job.state = SubagentJobState.SKIPPED
            job.result = SubagentJobResult(
                status="skipped",
                summary="Skipped due to context budget limits.",
                details=decision.reason,
            )
            telemetry_service.emit(
                "subagent.job_skipped",
                {
                    "job_id": job.job_id,
                    "document_id": job.document_id,
                    "chunk_id": job.chunk_id,
                    "reason": decision.reason,
                },
            )
            return job

        telemetry_service.emit(
            "subagent.job_started",
            {
                "job_id": job.job_id,
                "document_id": job.document_id,
                "chunk_id": job.chunk_id,
                "token_estimate": job.chunk_ref.token_estimate,
                "prompt_tokens": prompt_tokens,
            },
        )

        start = time.perf_counter()
        try:
            summary, tool_calls = await self._complete_job(job, messages)
            duration_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
            job.state = SubagentJobState.SUCCEEDED
            job.result = SubagentJobResult(
                status="ok",
                summary=summary or "(no findings)",
                tokens_used=self._estimate_tokens(summary),
                latency_ms=duration_ms,
                tool_calls=tuple(tool_calls),
            )
            telemetry_service.emit(
                "subagent.job_completed",
                {
                    "job_id": job.job_id,
                    "document_id": job.document_id,
                    "chunk_id": job.chunk_id,
                    "latency_ms": round(duration_ms, 3),
                    "tokens_used": job.result.tokens_used,
                },
            )
        except Exception as exc:  # pragma: no cover - guarded runtime path
            duration_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
            job.state = SubagentJobState.FAILED
            job.result = SubagentJobResult(
                status="error",
                summary="Subagent failed",
                details=str(exc),
                latency_ms=duration_ms,
            )
            telemetry_service.emit(
                "subagent.job_failed",
                {
                    "job_id": job.job_id,
                    "document_id": job.document_id,
                    "chunk_id": job.chunk_id,
                    "error": str(exc)[:200],
                },
            )
            LOGGER.debug("Subagent job %s failed", job.job_id, exc_info=True)
        return job

    async def _complete_job(
        self,
        job: SubagentJob,
        messages: Sequence[Mapping[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        tool_specs = self._tool_specs(job.allowed_tools)
        metadata = {
            "subagent_job_id": job.job_id,
            "document_id": job.document_id,
        }
        stream_kwargs: MutableMapping[str, Any] = {
            "messages": list(messages),
            "metadata": metadata,
            "max_completion_tokens": job.budget.max_completion_tokens,
            "temperature": 0.0,
        }
        if tool_specs:
            stream_kwargs["tools"] = tool_specs

        async def _stream_completion() -> tuple[str, list[dict[str, Any]]]:
            deltas: list[str] = []
            final_chunk: str | None = None
            tool_calls: list[dict[str, Any]] = []
            async for event in self._client.stream_chat(**stream_kwargs):
                if event.type == "content.delta" and event.content:
                    deltas.append(str(event.content))
                elif event.type == "content.done" and event.content:
                    final_chunk = str(event.content)
                elif event.type.endswith("tool_calls.function.arguments.done"):
                    tool_calls.append(
                        {
                            "name": event.tool_name,
                            "arguments": event.tool_arguments,
                            "tool_call_id": event.tool_call_id,
                        }
                    )
            response = "".join(deltas)
            if final_chunk:
                if not response:
                    response = final_chunk
                elif not response.endswith(final_chunk):
                    response += final_chunk
            return response.strip(), tool_calls

        timeout = max(1.0, job.budget.max_runtime_seconds)
        return await asyncio.wait_for(_stream_completion(), timeout=timeout)

    def _build_messages(self, job: SubagentJob, chunk_text: str) -> list[dict[str, str]]:
        chunk = job.chunk_ref
        header_lines = [
            f"Document: {chunk.document_id}",
            f"Chunk: {chunk.chunk_id}",
        ]
        if chunk.char_range:
            header_lines.append(f"Character range: {chunk.char_range[0]}-{chunk.char_range[1]}")
        header = "\n".join(header_lines)
        content = f"{header}\n\n{chunk_text}"
        system_prompt = job.instructions.strip() or self._config.instructions_template
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _evaluate_budget(self, job: SubagentJob, prompt_tokens: int) -> BudgetDecision | None:
        if self._budget_policy is None:
            return None
        try:
            return self._budget_policy.tokens_available(
                prompt_tokens=prompt_tokens,
                response_reserve=job.budget.max_completion_tokens,
                document_id=job.document_id,
            )
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Budget policy evaluation failed for subagent job %s", job.job_id, exc_info=True)
            return None

    def _tool_specs(self, allowed_tools: Sequence[str]) -> list[Mapping[str, Any]]:
        if not allowed_tools:
            return []
        try:
            registry = self._tool_resolver()
        except Exception:  # pragma: no cover - defensive guard
            return []
        specs: list[Mapping[str, Any]] = []
        for name in allowed_tools:
            registration = registry.get(name)
            spec_builder = getattr(registration, "as_openai_tool", None)
            if callable(spec_builder):
                try:
                    spec = cast(Mapping[str, Any], spec_builder())
                except Exception:
                    LOGGER.debug("Failed to serialize tool %s for subagent", name, exc_info=True)
                    continue
                specs.append(spec)
        return specs

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        counter = getattr(self._client, "count_tokens", None)
        if callable(counter):
            try:
                value: Any = counter(text, estimate_only=True)
                return int(value)
            except TypeError:
                pass
        return max(1, len(text.encode("utf-8")) // 4)

    def _trim_chunk(self, chunk_text: str, token_budget: int) -> str:
        limit = max(256, token_budget)
        approx_tokens = self._estimate_tokens(chunk_text)
        if approx_tokens <= limit:
            return chunk_text
        ratio = limit / float(max(1, approx_tokens))
        char_budget = max(200, int(len(chunk_text) * ratio))
        return chunk_text[:char_budget].rstrip()+"…"


class SubagentManager:
    """Coordinates job queues and executor wiring for AIController."""

    def __init__(
        self,
        client: AIClient,
        *,
        tool_resolver: Callable[[], Mapping[str, Any]],
        config: SubagentRuntimeConfig,
        budget_policy: ContextBudgetPolicy | None = None,
        result_cache: SubagentResultCache | None = None,
    ) -> None:
        self._config = config.clamp()
        self._executor = SubagentExecutor(
            client,
            tool_resolver=tool_resolver,
            budget_policy=budget_policy,
            config=self._config,
        )
        self._cache = result_cache

    @property
    def config(self) -> SubagentRuntimeConfig:
        return self._config

    def update_client(self, client: AIClient) -> None:
        self._executor.update_client(client)

    def update_budget_policy(self, policy: ContextBudgetPolicy | None) -> None:
        self._executor.update_budget_policy(policy)

    def update_config(self, config: SubagentRuntimeConfig) -> None:
        self._config = config.clamp()
        self._executor.update_config(self._config)

    def update_cache(self, cache: SubagentResultCache | None) -> None:
        self._cache = cache

    async def run_jobs(self, jobs: Sequence[SubagentJob]) -> list[SubagentJob]:
        if not self._config.enabled or not jobs:
            return []
        limit = self._config.max_jobs_per_turn or len(jobs)
        if limit <= 0:
            return []
        queue = SubagentJobQueue(max_jobs=limit)
        scheduled: list[SubagentJob] = []
        cache_hits = 0
        for job in jobs:
            job.budget = job.budget.clamp()
            if queue.enqueue(job):
                scheduled.append(job)
        results: list[SubagentJob] = []
        for job in scheduled:
            cached_result = self._cache.get(job) if self._cache is not None else None
            if cached_result is not None:
                job.state = SubagentJobState.SUCCEEDED
                job.result = cached_result
                results.append(job)
                cache_hits += 1
                continue
            executed = await self._executor.execute(job)
            results.append(executed)
            if (
                self._cache is not None
                and executed.state == SubagentJobState.SUCCEEDED
                and executed.result is not None
            ):
                self._cache.store(executed)
        if scheduled:
            self._emit_turn_summary(
                requested=len(jobs),
                scheduled=scheduled,
                results=results,
                cache_hits=cache_hits,
            )
        return results

    def _emit_turn_summary(
        self,
        *,
        requested: int,
        scheduled: Sequence[SubagentJob],
        results: Sequence[SubagentJob],
        cache_hits: int,
    ) -> None:
        succeeded = sum(1 for job in results if job.state == SubagentJobState.SUCCEEDED)
        failed = sum(1 for job in results if job.state == SubagentJobState.FAILED)
        skipped = sum(1 for job in results if job.state == SubagentJobState.SKIPPED)
        total_latency = sum((job.result.latency_ms if job.result else 0.0) for job in results)
        total_tokens = sum((job.result.tokens_used if job.result else 0) for job in results)
        document_ids = sorted({job.document_id for job in scheduled if job.document_id})
        telemetry_service.emit(
            "subagent.turn_summary",
            {
                "jobs_requested": requested,
                "jobs_scheduled": len(scheduled),
                "jobs_succeeded": succeeded,
                "jobs_failed": failed,
                "jobs_skipped": skipped,
                "cache_hits": cache_hits,
                "tokens_used": round(total_tokens, 3),
                "latency_ms_total": round(total_latency, 3),
                "config_max_jobs": self._config.max_jobs_per_turn,
                "documents": document_ids,
            },
        )


__all__ = ["SubagentExecutor", "SubagentJobQueue", "SubagentManager"]
