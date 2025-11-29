"""Pipeline stages for the orchestration turn flow.

This package contains the individual stages of the turn pipeline:
- prepare: Build messages and estimate budget
- analyze: Run preflight analysis and generate hints  
- execute: Execute model calls and parse responses
- tools: Execute tool calls and collect results
- finish: Assemble final output and metrics
"""

from .prepare import (
    build_messages,
    estimate_budget,
    prepare_turn,
    estimate_message_tokens,
    estimate_text_tokens,
    sanitize_history,
)

from .analyze import (
    analyze_turn,
    generate_hints,
    format_hints_block,
    AnalysisProvider,
)

from .execute import (
    ModelClient,
    StreamEvent,
    execute_model,
    parse_response,
    aggregate_streaming_events,
    extract_tool_calls_from_events,
    merge_tool_calls,
)

from .tools import (
    ToolExecutor,
    ToolExecutionResult,
    ToolResults,
    execute_tools,
    execute_tool_call,
    append_tool_results,
    tool_call_to_record,
    format_tool_result_content,
)

from .finish import (
    collect_metrics,
    finish_turn,
    finish_turn_with_error,
    aggregate_tool_records,
    calculate_total_tokens,
    TurnTimer,
)

__all__ = [
    # prepare.py exports
    "build_messages",
    "estimate_budget",
    "prepare_turn",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "sanitize_history",
    # analyze.py exports
    "analyze_turn",
    "generate_hints",
    "format_hints_block",
    "AnalysisProvider",
    # execute.py exports
    "ModelClient",
    "StreamEvent",
    "execute_model",
    "parse_response",
    "aggregate_streaming_events",
    "extract_tool_calls_from_events",
    "merge_tool_calls",
    # tools.py exports
    "ToolExecutor",
    "ToolExecutionResult",
    "ToolResults",
    "execute_tools",
    "execute_tool_call",
    "append_tool_results",
    "tool_call_to_record",
    "format_tool_result_content",
    # finish.py exports
    "collect_metrics",
    "finish_turn",
    "finish_turn_with_error",
    "aggregate_tool_records",
    "calculate_total_tokens",
    "TurnTimer",
]
