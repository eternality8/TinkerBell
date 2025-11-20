"""Tests for the chat event logging helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tinkerbell.ai.orchestration.event_log import ChatEventLogger


def _read_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_chat_event_logger_writes_entries(tmp_path: Path) -> None:
    logger = ChatEventLogger(enabled=True, base_dir=tmp_path)
    run = logger.start_run(
        run_id="run-test",
        prompt="Summarize",
        document_id="doc-1",
        document_path="story.md",
        snapshot={"text": "Hello"},
        metadata={"selection": "intro"},
        history=[{"role": "user", "content": "Hi"}],
    )

    with run:
        run.log_snapshot({"text": "Hello"}, label="manual")
        run.log_assistant_message(
            turn_index=1,
            message={"role": "assistant", "content": "Working"},
            response_text="Working",
            tool_calls=[{"id": "call-1", "name": "document_snapshot", "index": 0}],
        )
        run.log_tool_batch(
            turn_index=1,
            records=[{"id": "call-1", "name": "document_snapshot", "raw_result": {"text": "Hello"}}],
            messages=[{"role": "tool", "content": "payload"}],
        )
        run.log_completion(response_text="Done", tool_call_count=1, warnings=["test"], trace_compaction={})

    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1
    entries = _read_entries(log_files[0])
    assert entries[0]["event"] == "start"
    assert any(entry.get("event") == "assistant" for entry in entries)
    assert entries[-1]["event"] == "completion"
    assert entries[-1]["status"] == "success"


def test_chat_event_logger_disabled_is_noop(tmp_path: Path) -> None:
    logger = ChatEventLogger(enabled=False, base_dir=tmp_path)
    run = logger.start_run(
        run_id="no-log",
        prompt="noop",
        document_id=None,
        document_path=None,
        snapshot={},
        metadata=None,
        history=None,
    )
    with run:
        run.log_completion(response_text="", tool_call_count=0)

    log_files = list(tmp_path.glob("*.jsonl"))
    assert log_files == []
