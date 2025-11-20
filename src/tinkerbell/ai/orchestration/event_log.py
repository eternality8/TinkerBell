"""Debug event logging utilities for AI chat runs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)


def _default_event_dir() -> Path:
    log_path = logging_utils.get_log_path()
    if log_path is not None:
        return log_path.parent / "events"
    return Path.home() / ".tinkerbell" / "logs" / "events"


@dataclass(slots=True)
class _NullChatEventLogRun:
    """No-op implementation used when event logging is disabled."""

    path: Path | None = None

    def __enter__(self) -> "_NullChatEventLogRun":  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401
        return False

    def log_snapshot(self, *_: Any, **__: Any) -> None:
        return

    def log_assistant_message(self, *_: Any, **__: Any) -> None:
        return

    def log_tool_batch(self, *_: Any, **__: Any) -> None:
        return

    def log_completion(self, *_: Any, **__: Any) -> None:
        return

    def log_failure(self, *_: Any, **__: Any) -> None:
        return


class ChatEventLogRun:
    """Context manager that writes structured JSONL entries for a chat turn."""

    def __init__(self, path: Path, *, context: Mapping[str, Any]) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8")
        self._finalized = False
        self._write_entry("start", context)

    def __enter__(self) -> "ChatEventLogRun":  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401
        if exc is not None and not self._finalized:
            self.log_failure(message=str(exc))
        elif not self._finalized:
            self.log_failure(message="run aborted without completion")
        return False

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:  # pragma: no cover - defensive guard
            pass

    def log_snapshot(self, snapshot: Mapping[str, Any], *, label: str = "snapshot") -> None:
        payload = {
            "label": label,
            "snapshot": self._safe_json(dict(snapshot)),
        }
        self._write_entry("snapshot", payload)

    def log_assistant_message(
        self,
        *,
        turn_index: int,
        message: Mapping[str, Any],
        response_text: str,
        tool_calls: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        payload = {
            "turn_index": turn_index,
            "message": self._safe_json(dict(message)),
            "response_text": response_text,
            "tool_calls": self._safe_json(list(tool_calls or [])),
        }
        self._write_entry("assistant", payload)

    def log_tool_batch(
        self,
        *,
        turn_index: int,
        records: Sequence[Mapping[str, Any]],
        messages: Sequence[Mapping[str, Any]],
    ) -> None:
        if not records and not messages:
            return
        payload = {
            "turn_index": turn_index,
            "tool_records": self._safe_json(list(records)),
            "tool_messages": self._safe_json(list(messages)),
        }
        self._write_entry("tools", payload)

    def log_completion(
        self,
        *,
        response_text: str,
        tool_call_count: int,
        warnings: Sequence[str] | None = None,
        trace_compaction: Mapping[str, Any] | None = None,
    ) -> None:
        if self._finalized:
            return
        payload = {
            "response_text": response_text,
            "tool_call_count": tool_call_count,
            "warnings": list(warnings or ()),
            "trace_compaction": self._safe_json(dict(trace_compaction or {})),
            "status": "success",
        }
        self._write_entry("completion", payload)
        self._finalized = True
        self.close()

    def log_failure(self, *, message: str, details: Mapping[str, Any] | None = None) -> None:
        if self._finalized:
            return
        payload = {
            "status": "failure",
            "message": message,
        }
        if details:
            payload["details"] = self._safe_json(dict(details))
        self._write_entry("failure", payload)
        self._finalized = True
        self.close()

    def _write_entry(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {
            "event": event,
            "timestamp": time.time(),
        }
        if payload:
            for key, value in payload.items():
                entry[key] = self._safe_json(value)
        json.dump(entry, self._file, ensure_ascii=False)
        self._file.write("\n")
        self._file.flush()

    def _safe_json(self, value: Any, *, depth: int = 0) -> Any:
        if depth > 6:
            return repr(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): self._safe_json(val, depth=depth + 1) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._safe_json(item, depth=depth + 1) for item in value]
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - defensive conversion
                return repr(value)
        return repr(value)


class ChatEventLogger:
    """Factory for per-chat event logs when debug logging is enabled."""

    def __init__(self, *, enabled: bool, base_dir: Path | str | None = None) -> None:
        self.enabled = bool(enabled)
        self._base_dir = Path(base_dir) if base_dir else _default_event_dir()

    def start_run(
        self,
        *,
        run_id: str,
        prompt: str,
        document_id: str | None,
        document_path: str | None,
        snapshot: Mapping[str, Any],
        metadata: Mapping[str, Any] | None,
        history: Sequence[Mapping[str, Any]] | None,
    ) -> ChatEventLogRun | _NullChatEventLogRun:
        if not self.enabled:
            return _NullChatEventLogRun()
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            path = self._allocate_path(run_id)
            context = {
                "run_id": run_id,
                "prompt": prompt,
                "document_id": document_id,
                "document_path": document_path,
                "metadata": dict(metadata or {}),
                "history": list(history or ()),
            }
            log_run = ChatEventLogRun(path, context=context)
            log_run.log_snapshot(snapshot, label="initial")
            LOGGER.debug("AI event log started: %s", path)
            return log_run
        except Exception:  # pragma: no cover - best effort logging
            LOGGER.debug("Failed to start chat event log", exc_info=True)
            return _NullChatEventLogRun()

    def _allocate_path(self, run_id: str) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        safe_run_id = "".join(ch for ch in run_id if ch.isalnum())[:12] or "run"
        return self._base_dir / f"chat-{timestamp}-{safe_run_id}.jsonl"


__all__ = [
    "ChatEventLogger",
    "ChatEventLogRun",
]
