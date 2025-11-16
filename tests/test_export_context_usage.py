"""Tests for the export_context_usage CLI script."""

from __future__ import annotations

import json
from pathlib import Path

from tinkerbell.services import telemetry as telemetry_service
from tinkerbell.scripts import export_context_usage as exporter

from tinkerbell.ai.services.telemetry import ContextUsageEvent


def _event(run_id: str, prompt: int, tool: int) -> ContextUsageEvent:
    return ContextUsageEvent(
        document_id="doc",
        model="gpt-test",
        prompt_tokens=prompt,
        tool_tokens=tool,
        response_reserve=None,
        timestamp=1.0,
        conversation_length=1,
        tool_names=("diff",),
        run_id=run_id,
    )


def test_export_context_usage_json(tmp_path: Path) -> None:
    source = tmp_path / "telemetry" / "context_usage.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    sink = telemetry_service.PersistentTelemetrySink(path=source, capacity=5)
    sink.record(_event("run-json", 10, 5))

    target = tmp_path / "export.json"
    exit_code = exporter.main(["--source", str(source), "--output", str(target), "--format", "json"])
    assert exit_code == 0

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload and payload[0]["run_id"] == "run-json"


def test_export_context_usage_csv(tmp_path: Path) -> None:
    source = tmp_path / "telemetry2" / "context_usage.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    sink = telemetry_service.PersistentTelemetrySink(path=source, capacity=5)
    sink.record(_event("run-csv", 20, 7))

    target = tmp_path / "export.csv"
    exit_code = exporter.main(["--source", str(source), "--output", str(target), "--format", "csv"])
    assert exit_code == 0

    rows = target.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].startswith("timestamp,document_id")
    assert any("run-csv" in row for row in rows)
