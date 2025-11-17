# Subagent helper latency snapshot

Phase 4.4 adds a repeatable benchmark that measures the scheduling overhead of the `SubagentManager` compared to a manager-only baseline. The script simulates helper jobs with a fixed per-job latency so we can reason about orchestration costs without calling a real model.

## Methodology

- Script: `benchmarks/measure_subagent_latency.py`
- Config: `SubagentRuntimeConfig(enabled=True, max_jobs_per_turn=4)`
- Simulation: helper jobs sleep for 6 ms to emulate a fast reasoning call.
- Samples: 40 iterations per job-count configuration; each sample runs all jobs sequentially.
- Environment: Windows 11, Python 3.13.3 (uv-managed virtualenv), no GPU acceleration.

## Results

| jobs | avg ms | p95 ms | min ms | max ms |
| ---- | ------ | ------ | ------ | ------ |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 9.99 | 16.12 | 7.14 | 16.56 |
| 2 | 20.06 | 24.79 | 15.14 | 26.98 |
| 4 | 41.01 | 50.32 | 33.39 | 53.85 |

**Takeaways**

- Scheduling overhead remains below 4 ms/job when helpers succeed (~6 ms simulated), so runtime remains dominated by the model call.
- Doubling helper count scales linearly, confirming the sequential execution guarantee and validating the Phase 4.4 smoke tests.
- Zero-job baseline shows the manager short-circuits immediately when no work is enqueued.

## Reproducing the run

```powershell
uv run python benchmarks/measure_subagent_latency.py --iterations 40 --latency-ms 6
```

Customize `--job-counts` or `--max-jobs-per-turn` to explore other queues. The helper flag remains **default-off** in production builds; run this benchmark from the AI_v2 branch before toggling it on in production.
