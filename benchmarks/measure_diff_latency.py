"""Quick benchmark helper for DiffBuilderTool latency on large fixtures."""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

from tinkerbell.ai.services.summarizer import ToolPayload, build_pointer, summarize_tool_content
from tinkerbell.ai.tools.diff_builder import DiffBuilderTool
from tinkerbell.services.telemetry import count_text_tokens


@dataclass(slots=True)
class BenchmarkResult:
    label: str
    path: Path
    tokens: int
    size_bytes: int
    diff_chars: int
    runtime_ms: float
    diff_tokens: int
    pointer_tokens: int
    tokens_saved: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def _default_cases() -> Sequence[tuple[str, Path]]:
    root = Path("test_data")
    return (
        ("War and Peace", root / "War and Peace.txt"),
        ("Twenty Thousand Leagues", root / "Twenty Thousand Leagues under the Sea.txt"),
        ("Large JSON", root / "5MB.json"),
    )


def _mutate_text(text: str) -> str:
    insert_size = min(max(10_000, len(text) // 10), len(text))
    insert_block = text[:insert_size]
    amplified_block = insert_block.replace(" the ", " the AI ")
    return f"{text}\n\n<!-- benchmark mutation -->\n{amplified_block}"


POINTER_SUMMARY_TOKENS = 512


def _estimate_pointer_savings(
    diff_text: str,
    *,
    label: str,
    context_lines: int,
) -> tuple[int, int, int]:
    if not diff_text.strip():
        return 0, 0, 0
    payload = ToolPayload(
        name="DiffBuilderTool",
        content=diff_text,
        arguments={"context": context_lines, "benchmark_case": label},
        metadata={"benchmark_case": label},
    )
    summary = summarize_tool_content(payload, budget_tokens=POINTER_SUMMARY_TOKENS)
    pointer = build_pointer(summary, tool_name="DiffBuilderTool")
    pointer_tokens = count_text_tokens(pointer.as_chat_content())
    diff_tokens = summary.original_tokens or count_text_tokens(diff_text)
    tokens_saved = max(0, diff_tokens - pointer_tokens)
    return diff_tokens, pointer_tokens, tokens_saved


def run_benchmarks(
    cases: Iterable[tuple[str, Path]],
    *,
    context_lines: int,
) -> list[BenchmarkResult]:
    builder = DiffBuilderTool(default_context_lines=context_lines)
    results: list[BenchmarkResult] = []

    for label, path in cases:
        text = path.read_text(encoding="utf-8")
        updated = _mutate_text(text)
        start = perf_counter()
        diff = builder.run(text, updated, filename=path.name, context=context_lines)
        runtime_ms = (perf_counter() - start) * 1000
        diff_tokens, pointer_tokens, tokens_saved = _estimate_pointer_savings(
            diff,
            label=label,
            context_lines=context_lines,
        )
        results.append(
            BenchmarkResult(
                label=label,
                path=path,
                tokens=count_text_tokens(text),
                size_bytes=len(text.encode("utf-8")),
                diff_chars=len(diff),
                runtime_ms=runtime_ms,
                diff_tokens=diff_tokens,
                pointer_tokens=pointer_tokens,
                tokens_saved=tokens_saved,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure DiffBuilderTool latency on large fixtures.")
    parser.add_argument(
        "--case",
        action="append",
        metavar="LABEL=PATH",
        help="Optional case override; can be supplied multiple times.",
    )
    parser.add_argument("--context", type=int, default=3, help="Context lines passed to unified diff.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON results.")
    args = parser.parse_args()

    if args.case:
        cases: list[tuple[str, Path]] = []
        for raw in args.case:
            if "=" not in raw:
                parser.error(f"Invalid --case '{raw}'. Expected LABEL=PATH format.")
            label, value = raw.split("=", 1)
            cases.append((label.strip(), Path(value).expanduser()))
    else:
        cases = list(_default_cases())

    results = run_benchmarks(cases, context_lines=max(0, args.context))

    if args.json:
        import json

        payload = [
            {
                "label": result.label,
                "path": str(result.path),
                "tokens": result.tokens,
                "size_mb": result.size_mb,
                "diff_chars": result.diff_chars,
                "diff_tokens": result.diff_tokens,
                "pointer_tokens": result.pointer_tokens,
                "tokens_saved": result.tokens_saved,
                "runtime_ms": result.runtime_ms,
            }
            for result in results
        ]
        print(json.dumps(payload, indent=2))
        return

    max_label = max(len(result.label) for result in results)
    header = (
        f"{'Document':<{max_label}}  Size (MB)  Tokens       Diff chars  Diff tokens  "
        "Pointer tokens  Tokens saved  Runtime (ms)"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.label:<{max_label}}  "
            f"{result.size_mb:>8.2f}  "
            f"{result.tokens:>10,}  "
            f"{result.diff_chars:>10,}  "
            f"{result.diff_tokens:>11,}  "
            f"{result.pointer_tokens:>14,}  "
            f"{result.tokens_saved:>12,}  "
            f"{result.runtime_ms:>11.2f}"
        )

    runtimes = [result.runtime_ms for result in results]
    print()
    print(
        "Runtime stats → min: "
        f"{min(runtimes):.2f} ms · median: {statistics.median(runtimes):.2f} ms · max: {max(runtimes):.2f} ms"
    )


if __name__ == "__main__":
    main()
