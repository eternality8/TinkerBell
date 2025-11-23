from __future__ import annotations

import asyncio
from typing import cast

import pytest

from tinkerbell.ai.agents.subagents.manager import SubagentExecutor, SubagentManager
from tinkerbell.ai.orchestration.subagent_runtime import SubagentRuntimeManager
from tinkerbell.ai.ai_types import (
    ChunkReference,
    SubagentBudget,
    SubagentJob,
    SubagentJobResult,
    SubagentJobState,
    SubagentRuntimeConfig,
)
from tinkerbell.ai.client import AIClient
from tinkerbell.ai.services.context_policy import BudgetDecision


class _DummyClient:
    settings = type("S", (), {"model": "test-model"})()


def _build_job(job_id: str, chunk_hash: str) -> SubagentJob:
    chunk = ChunkReference(
        document_id="doc-1",
        chunk_id=f"selection:{job_id}",
        version_id="v1",
        pointer_id=f"selection:{job_id}",
        char_range=(0, 10),
        token_estimate=200,
        chunk_hash=chunk_hash,
        preview="Sample text",
    )
    job = SubagentJob(
        job_id=job_id,
        parent_run_id="run-1",
        instructions="Analyze chunk",
        chunk_ref=chunk,
        allowed_tools=(),
        budget=SubagentBudget(
            max_prompt_tokens=400,
            max_completion_tokens=200,
            max_runtime_seconds=30.0,
        ),
        dedup_hash=chunk_hash,
    )
    job.result = None
    return job


def test_subagent_job_exposes_chunk_metadata() -> None:
    job = _build_job("job-meta", "chunk-meta")
    assert job.document_id == "doc-1"
    assert job.chunk_id == "selection:job-meta"
    assert job.chunk_hash == "chunk-meta"


@pytest.mark.asyncio
async def test_subagent_manager_runs_jobs_sequentially(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = SubagentManager(
        cast(AIClient, _DummyClient()),
        tool_resolver=lambda: {},
        config=SubagentRuntimeConfig(enabled=True, max_jobs_per_turn=4),
    )

    invocation_order: list[str] = []

    async def _fake_execute(self: SubagentExecutor, job: SubagentJob) -> SubagentJob:
        invocation_order.append(f"{job.job_id}:start")
        await asyncio.sleep(0)
        invocation_order.append(f"{job.job_id}:end")
        job.state = SubagentJobState.SUCCEEDED
        job.result = SubagentJobResult(status="ok", summary="done", tokens_used=5, latency_ms=1.0)
        return job

    monkeypatch.setattr(SubagentExecutor, "execute", _fake_execute, raising=False)

    jobs = [_build_job("job-1", "chunk-1"), _build_job("job-2", "chunk-2")]

    results = await manager.run_jobs(jobs)

    assert invocation_order == ["job-1:start", "job-1:end", "job-2:start", "job-2:end"]
    assert all(job.state == SubagentJobState.SUCCEEDED for job in results)


class _RejectingPolicy:
    def __init__(self) -> None:
        self.enabled = True
        self.dry_run = False
        self.prompt_budget = 200
        self.response_reserve = 0
        self.emergency_buffer = 0
        self.model_name = "stub"

    def tokens_available(
        self,
        *,
        prompt_tokens: int,
        response_reserve: int | None = None,
        pending_tool_tokens: int = 0,
        document_id: str | None = None,
        **_kwargs: object,
    ) -> BudgetDecision:
        return BudgetDecision(
            verdict="reject",
            reason="forced",
            prompt_tokens=prompt_tokens + pending_tool_tokens,
            prompt_budget=self.prompt_budget,
            response_reserve=response_reserve or self.response_reserve,
            pending_tool_tokens=pending_tool_tokens,
            deficit=1,
            dry_run=False,
            model=self.model_name,
            document_id=document_id,
        )


@pytest.mark.asyncio
async def test_subagent_manager_respects_budget_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _RejectingPolicy()
    manager = SubagentManager(
        cast(AIClient, _DummyClient()),
        tool_resolver=lambda: {},
        config=SubagentRuntimeConfig(enabled=True, max_jobs_per_turn=2),
        budget_policy=policy,
    )

    jobs = [_build_job("job-budget", "chunk-budget")]

    results = await manager.run_jobs(jobs)

    assert results[0].state == SubagentJobState.SKIPPED
    assert results[0].result is not None
    assert results[0].result.status == "skipped"
    assert "context budget" in results[0].result.summary.lower()

    # New test for SubagentRuntimeManager

def test_subagent_runtime_manager_exposes_character_map_store() -> None:
    runtime = SubagentRuntimeManager(tool_resolver=lambda: {})
    runtime.configure(
        client=cast(AIClient, _DummyClient()),
        config=SubagentRuntimeConfig(enabled=True, plot_scaffolding_enabled=True),
        budget_policy=None,
    )

    store = runtime.ensure_character_map_store()

    assert store is runtime.character_map_store
