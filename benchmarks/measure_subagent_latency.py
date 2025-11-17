from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Sequence, cast

from tinkerbell.ai.agents.subagents.manager import SubagentManager
from tinkerbell.ai.ai_types import (
    ChunkReference,
    SubagentBudget,
    SubagentJob,
    SubagentJobResult,
    SubagentJobState,
    SubagentRuntimeConfig,
)
from tinkerbell.ai.client import AIClient


class _DummyClient:
    settings = type("S", (), {"model": "benchmark-model"})()


class _BenchmarkExecutor:
    def __init__(self, latency_ms: float) -> None:
        self._latency = max(0.0, latency_ms) / 1000.0

    async def execute(self, job: SubagentJob) -> SubagentJob:
        await asyncio.sleep(self._latency)
        job.state = SubagentJobState.SUCCEEDED
        job.result = SubagentJobResult(
            status="ok",
            summary="benchmark",
            tokens_used=64,
            latency_ms=self._latency * 1_000,
        )
        return job

    def update_client(self, _client: AIClient) -> None:  # pragma: no cover - helper
        return

    def update_budget_policy(self, _policy) -> None:  # pragma: no cover - helper
        return

    def update_config(self, _config: SubagentRuntimeConfig) -> None:  # pragma: no cover - helper
        return


def _build_job(job_id: int) -> SubagentJob:
    chunk = ChunkReference(
        document_id="doc-bench",
        chunk_id=f"selection:{job_id}",
        version_id="v1",
        pointer_id=f"selection:{job_id}",
        char_range=(0, 600),
        token_estimate=600,
        chunk_hash=f"chunk-{job_id}",
        preview="Benchmark text",
    )
    job = SubagentJob(
        job_id=f"job-{job_id}",
        parent_run_id="run-bench",
        instructions="Analyze",
        chunk_ref=chunk,
        allowed_tools=(),
        budget=SubagentBudget(max_prompt_tokens=800, max_completion_tokens=256, max_runtime_seconds=30.0),
        dedup_hash=chunk.chunk_hash,
    )
    job.result = None
    return job


def _build_manager(config: SubagentRuntimeConfig) -> SubagentManager:
    manager = SubagentManager(
        cast(AIClient, _DummyClient()),
        tool_resolver=lambda: {},
        config=config,
    )
    return manager


def _reset_jobs(jobs: Sequence[SubagentJob]) -> None:
    for job in jobs:
        job.state = SubagentJobState.QUEUED
        job.result = None


async def _measure(job_count: int, *, iterations: int, latency_ms: float, config: SubagentRuntimeConfig) -> dict[str, float]:
    manager = _build_manager(config)
    setattr(manager, "_executor", _BenchmarkExecutor(latency_ms))  # type: ignore[attr-defined]
    jobs = tuple(_build_job(idx) for idx in range(job_count))
    samples: list[float] = []
    for _ in range(iterations):
        _reset_jobs(jobs)
        start = time.perf_counter()
        await manager.run_jobs(jobs)
        samples.append((time.perf_counter() - start) * 1_000)
    avg = statistics.fmean(samples) if samples else 0.0
    minimum = min(samples) if samples else 0.0
    maximum = max(samples) if samples else 0.0
    if len(samples) >= 2:
        p95 = statistics.quantiles(samples, n=20, method="inclusive")[18]
    else:
        p95 = samples[0] if samples else 0.0
    return {
        "job_count": float(job_count),
        "avg_ms": avg,
        "p95_ms": p95,
        "min_ms": minimum,
        "max_ms": maximum,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure SubagentManager scheduling overhead")
    parser.add_argument("--iterations", type=int, default=60, help="Samples per configuration")
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=5.0,
        help="Simulated latency per helper job (ms)",
    )
    parser.add_argument(
        "--job-counts",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4),
        help="Job counts to benchmark",
    )
    parser.add_argument(
        "--max-jobs-per-turn",
        type=int,
        default=4,
        help="Runtime cap applied to SubagentRuntimeConfig",
    )
    return parser.parse_args()


def _render_table(rows: Sequence[dict[str, float]]) -> str:
    headers = ("jobs", "avg ms", "p95 ms", "min ms", "max ms")
    lines = [" | ".join(headers), " | ".join("-" * len(h) for h in headers)]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"{int(row['job_count']):>4}",
                    f"{row['avg_ms']:>7.2f}",
                    f"{row['p95_ms']:>7.2f}",
                    f"{row['min_ms']:>7.2f}",
                    f"{row['max_ms']:>7.2f}",
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    config = SubagentRuntimeConfig(enabled=True, max_jobs_per_turn=args.max_jobs_per_turn)

    async def _runner() -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for job_count in args.job_counts:
            rows.append(
                await _measure(
                    job_count,
                    iterations=max(1, args.iterations),
                    latency_ms=args.latency_ms,
                    config=config,
                )
            )
        return rows

    rows = asyncio.run(_runner())
    print("Subagent scheduling latency (simulated AI latency = %.2f ms)" % args.latency_ms)
    print(_render_table(rows))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
