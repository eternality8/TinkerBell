"""Compatibility wrapper for telemetry utilities (deprecated import path)."""

from __future__ import annotations

from ...services import telemetry as _telemetry

ContextUsageEvent = _telemetry.ContextUsageEvent
TelemetrySink = _telemetry.TelemetrySink
InMemoryTelemetrySink = _telemetry.InMemoryTelemetrySink
PersistentTelemetrySink = _telemetry.PersistentTelemetrySink
TokenCounterStatus = _telemetry.TokenCounterStatus
TokenUsageSummary = _telemetry.TokenUsageSummary
TokenUsageTotals = _telemetry.TokenUsageTotals
UsageDashboard = _telemetry.UsageDashboard
count_text_tokens = _telemetry.count_text_tokens
default_telemetry_path = _telemetry.default_telemetry_path
format_usage_summary = _telemetry.format_usage_summary
get_token_counter = _telemetry.get_token_counter
get_token_counter_registry = _telemetry.get_token_counter_registry
load_persistent_events = _telemetry.load_persistent_events
build_usage_dashboard = _telemetry.build_usage_dashboard
summarize_usage_event = _telemetry.summarize_usage_event
summarize_usage_events = _telemetry.summarize_usage_events
summarize_usage_totals = _telemetry.summarize_usage_totals
token_counter_status = _telemetry.token_counter_status
snapshot_events = _telemetry.snapshot_events
emit = _telemetry.emit
register_event_listener = _telemetry.register_event_listener

__all__ = (
	"ContextUsageEvent",
	"TelemetrySink",
	"InMemoryTelemetrySink",
	"PersistentTelemetrySink",
	"TokenCounterStatus",
	"TokenUsageSummary",
	"TokenUsageTotals",
	"UsageDashboard",
	"count_text_tokens",
	"default_telemetry_path",
	"format_usage_summary",
	"get_token_counter",
	"get_token_counter_registry",
	"load_persistent_events",
	"build_usage_dashboard",
	"summarize_usage_event",
	"summarize_usage_events",
	"summarize_usage_totals",
	"token_counter_status",
	"snapshot_events",
	"emit",
	"register_event_listener",
)
