"""Regression tests for the subagent result cache."""

from __future__ import annotations

from typing import cast

import pytest

from tinkerbell.ai.agents.subagents.manager import SubagentExecutor, SubagentManager
from tinkerbell.ai.ai_types import (
    ChunkReference,
    SubagentBudget,
    SubagentJob,
    SubagentJobResult,
    SubagentJobState,
    SubagentRuntimeConfig,
)
from tinkerbell.ai.memory.cache_bus import DocumentCacheBus, DocumentChangedEvent
from tinkerbell.ai.memory.result_cache import SubagentResultCache
from tinkerbell.ai.client import AIClient


class _DummyClient:
    settings = type("S", (), {"model": "test-model"})()


def _make_job(*, chunk_hash: str = "chunk-1", summary: str = "cached") -> SubagentJob:
    chunk = ChunkReference(
        document_id="doc-1",
        chunk_id="selection:0-10",
        version_id="v1",
        pointer_id="selection:doc-1",
        char_range=(0, 10),
        token_estimate=200,
        chunk_hash=chunk_hash,
        preview="Sample text",
    )
    job = SubagentJob(
        job_id=f"job-{chunk_hash}",
        parent_run_id="run-1",
        instructions="Analyze chunk",
        chunk_ref=chunk,
        allowed_tools=("document_snapshot",),
        budget=SubagentBudget(
            max_prompt_tokens=400,
            max_completion_tokens=200,
            max_runtime_seconds=30.0,
        ),
        dedup_hash=chunk_hash,
    )
    job.result = SubagentJobResult(status="ok", summary=summary, tokens_used=42, latency_ms=12.3)
    return job


def test_result_cache_hits_and_invalidates_via_bus() -> None:
    bus = DocumentCacheBus()
    cache = SubagentResultCache(max_entries=4, ttl_seconds=None, bus=bus)
    job = _make_job(summary="original")

    assert cache.get(job) is None

    cache.store(job)

    assert job.result is not None
    job.result.summary = "mutated"
    cached = cache.get(job)
    assert cached is not None
    assert cached.summary == "original"
    assert cached is not job.result

    bus.publish(DocumentChangedEvent(document_id="doc-1", version_id=2, content_hash="zzz"))

    assert cache.get(job) is None


@pytest.mark.asyncio
async def test_subagent_manager_reuses_cached_results(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = DocumentCacheBus()
    cache = SubagentResultCache(max_entries=4, ttl_seconds=None, bus=bus)
    manager = SubagentManager(
        cast(AIClient, _DummyClient()),
        tool_resolver=lambda: {},
        config=SubagentRuntimeConfig(enabled=True, max_jobs_per_turn=2),
        result_cache=cache,
    )

    call_counter = {"count": 0}

    async def _fake_execute(self: SubagentExecutor, job: SubagentJob) -> SubagentJob:
        call_counter["count"] += 1
        job.state = SubagentJobState.SUCCEEDED
        job.result = SubagentJobResult(status="ok", summary="fresh", tokens_used=5, latency_ms=1.0)
        return job

    monkeypatch.setattr(SubagentExecutor, "execute", _fake_execute, raising=False)

    first_job = _make_job(summary="ignored")
    first_job.result = None
    results = await manager.run_jobs([first_job])
    assert call_counter["count"] == 1
    assert results[0].result is not None

    second_job = _make_job(summary="will_use_cache")
    second_job.result = None
    second_results = await manager.run_jobs([second_job])

    assert call_counter["count"] == 1, "Executor should not run when cache hits"
    assert second_results[0].result is not None
    assert second_results[0].result.summary == results[0].result.summary
