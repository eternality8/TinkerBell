"""Benchmark helper for diff + bridge guardrail latency."""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

from tinkerbell.ai.tools.document_apply_patch import DocumentApplyPatchTool
from tinkerbell.ai.tools.document_edit import DocumentEditTool
from tinkerbell.ai.services.summarizer import ToolPayload, build_pointer, summarize_tool_content
from tinkerbell.ai.tools.diff_builder import DiffBuilderTool
from tinkerbell.chat.message_model import EditDirective
from tinkerbell.editor.document_model import DocumentState
from tinkerbell.editor.patches import PatchResult
from tinkerbell.services.bridge import DocumentBridge, DocumentVersionMismatchError
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


@dataclass(slots=True)
class PipelineBenchmarkResult:
    label: str
    path: Path
    safe_edits: bool
    runtime_ms: float
    status: str
    detail: str
    diff_summary: str | None
    patch_latency_ms: float


class _BenchmarkEditor:
    """Minimal bridge adapter for benchmarking."""

    def __init__(self, text: str) -> None:
        self.state = DocumentState(text=text)

    def load_document(self, document: DocumentState) -> None:
        self.state = document

    def to_document(self) -> DocumentState:
        return self.state

    def apply_ai_edit(self, directive: EditDirective, *, preserve_selection: bool = False) -> DocumentState:  # pragma: no cover - not used in benchmark
        start, end = directive.target_range
        text = self.state.text
        content = directive.content or ""
        if directive.action == "insert":
            new_text = text[:start] + content + text[start:]
        elif directive.action == "replace":
            new_text = text[:start] + content + text[end:]
        else:
            new_text = text
        self.state.update_text(new_text)
        return self.state

    def apply_patch_result(self, result: PatchResult, selection_hint=None, *, preserve_selection: bool = False) -> DocumentState:
        self.state.update_text(result.text)
        return self.state

    def restore_document(self, document: DocumentState) -> DocumentState:
        self.state = document
        return self.state


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


def _mutation_suffix(text: str) -> str:
    mutated = _mutate_text(text)
    suffix = mutated[len(text) :]
    if suffix:
        return suffix
    return "\n\n<!-- benchmark mutation noop -->\n"


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


def _run_pipeline_case(
    *,
    label: str,
    path: Path,
    text: str,
    suffix: str,
    safe_edits: bool,
    duplicate_threshold: int,
    token_drift: float,
) -> PipelineBenchmarkResult:
    editor = _BenchmarkEditor(text)
    bridge = DocumentBridge(editor=editor)
    bridge.configure_safe_editing(
        enabled=safe_edits,
        duplicate_threshold=duplicate_threshold,
        token_drift=token_drift,
    )
    edit_tool = DocumentEditTool(bridge=bridge)
    apply_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
    patches = [
        {
            "start": len(text),
            "end": len(text),
            "replacement": suffix,
            "match_text": "",
        }
    ]
    start = perf_counter()
    try:
        detail = apply_tool.run(patches=patches)
        status = "applied"
    except DocumentVersionMismatchError as exc:
        status = "rejected"
        detail = str(exc)
    except Exception as exc:  # pragma: no cover - defensive benchmark guard
        status = "error"
        detail = f"{exc.__class__.__name__}: {exc}"
    runtime_ms = (perf_counter() - start) * 1000.0
    diff_summary = bridge.last_diff_summary
    patch_latency_ms = bridge.patch_metrics.avg_latency_ms or runtime_ms
    return PipelineBenchmarkResult(
        label=label,
        path=path,
        safe_edits=safe_edits,
        runtime_ms=runtime_ms,
        status=status,
        detail=detail,
        diff_summary=diff_summary,
        patch_latency_ms=patch_latency_ms,
    )


def run_pipeline_benchmarks(
    cases: Iterable[tuple[str, Path]],
    *,
    duplicate_threshold: int,
    token_drift: float,
) -> list[PipelineBenchmarkResult]:
    results: list[PipelineBenchmarkResult] = []
    for label, path in cases:
        text = path.read_text(encoding="utf-8")
        suffix = _mutation_suffix(text)
        for safe_edits in (False, True):
            results.append(
                _run_pipeline_case(
                    label=label,
                    path=path,
                    text=text,
                    suffix=suffix,
                    safe_edits=safe_edits,
                    duplicate_threshold=duplicate_threshold,
                    token_drift=token_drift,
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
    parser.add_argument(
        "--mode",
        choices=("diff", "pipeline", "both"),
        default="both",
        help="Select which benchmark suites to run (diff builder, apply pipeline, or both).",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=int,
        default=2,
        help="Duplicate paragraph threshold for safe edit inspections during pipeline benchmarks.",
    )
    parser.add_argument(
        "--token-drift",
        type=float,
        default=0.05,
        help="Token drift tolerance for safe edit inspections during pipeline benchmarks.",
    )
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

    run_diff = args.mode in {"diff", "both"}
    run_pipeline = args.mode in {"pipeline", "both"}
    diff_results: list[BenchmarkResult] = []
    pipeline_results: list[PipelineBenchmarkResult] = []

    if run_diff:
        diff_results = run_benchmarks(cases, context_lines=max(0, args.context))
    if run_pipeline:
        duplicate_threshold = max(2, int(args.duplicate_threshold))
        token_drift = max(0.0, float(args.token_drift))
        pipeline_results = run_pipeline_benchmarks(
            cases,
            duplicate_threshold=duplicate_threshold,
            token_drift=token_drift,
        )

    if args.json:
        import json

        payload: list[dict[str, object]] = []
        for result in diff_results:
            payload.append(
                {
                    "suite": "diff",
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
            )
        for result in pipeline_results:
            payload.append(
                {
                    "suite": "pipeline",
                    "label": result.label,
                    "path": str(result.path),
                    "safe_edits": result.safe_edits,
                    "runtime_ms": result.runtime_ms,
                    "patch_latency_ms": result.patch_latency_ms,
                    "status": result.status,
                    "detail": result.detail,
                    "diff_summary": result.diff_summary,
                }
            )
        print(json.dumps(payload, indent=2))
        return

    if diff_results:
        max_label = max(len(result.label) for result in diff_results)
        header = (
            f"{'Document':<{max_label}}  Size (MB)  Tokens       Diff chars  Diff tokens  "
            "Pointer tokens  Tokens saved  Runtime (ms)"
        )
        print(header)
        print("-" * len(header))
        for result in diff_results:
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

        runtimes = [result.runtime_ms for result in diff_results]
        print()
        print(
            "DiffBuilder runtime stats → min: "
            f"{min(runtimes):.2f} ms · median: {statistics.median(runtimes):.2f} ms · max: {max(runtimes):.2f} ms"
        )

    if pipeline_results:
        if diff_results:
            print()
        max_label = max(len(result.label) for result in pipeline_results)
        header = (
            f"{'Document':<{max_label}}  {'Safe':>4}  {'Runtime (ms)':>12}  "
            f"{'Patch avg (ms)':>14}  {'Status':>8}  Diff summary"
        )
        print(header)
        print("-" * len(header))
        for result in pipeline_results:
            safe_label = "on" if result.safe_edits else "off"
            summary = result.diff_summary or "-"
            print(
                f"{result.label:<{max_label}}  "
                f"{safe_label:>4}  "
                f"{result.runtime_ms:>12.2f}  "
                f"{result.patch_latency_ms:>14.2f}  "
                f"{result.status:>8}  "
                f"{summary}"
            )

        print()
        for safe_flag in (False, True):
            runtimes = [r.runtime_ms for r in pipeline_results if r.safe_edits == safe_flag]
            if not runtimes:
                continue
            label = "Safe edits on" if safe_flag else "Safe edits off"
            print(
                f"{label} → min: {min(runtimes):.2f} ms · median: {statistics.median(runtimes):.2f} ms · "
                f"max: {max(runtimes):.2f} ms"
            )


if __name__ == "__main__":
    main()
