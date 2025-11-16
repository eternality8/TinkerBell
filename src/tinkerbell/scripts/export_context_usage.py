"""CLI utility to export recorded ContextUsageEvent data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from tinkerbell.ai.services.telemetry import ContextUsageEvent
from tinkerbell.services import telemetry as telemetry_service

_FIELDNAMES = [
    "timestamp",
    "document_id",
    "model",
    "prompt_tokens",
    "tool_tokens",
    "response_reserve",
    "conversation_length",
    "run_id",
    "tool_names",
]


def _event_to_dict(event: ContextUsageEvent) -> dict:
    payload = asdict(event)
    payload["tool_names"] = list(event.tool_names)
    return payload


def _write_json(events: Sequence[ContextUsageEvent], destination) -> None:
    json.dump([_event_to_dict(event) for event in events], destination, indent=2)
    destination.write("\n")


def _write_csv(events: Sequence[ContextUsageEvent], destination) -> None:
    writer = csv.DictWriter(destination, fieldnames=_FIELDNAMES)
    writer.writeheader()
    for event in events:
        writer.writerow(
            {
                "timestamp": f"{event.timestamp:.6f}",
                "document_id": event.document_id or "",
                "model": event.model,
                "prompt_tokens": event.prompt_tokens,
                "tool_tokens": event.tool_tokens,
                "response_reserve": event.response_reserve or "",
                "conversation_length": event.conversation_length,
                "run_id": event.run_id,
                "tool_names": ",".join(event.tool_names),
            }
        )


def _load_events(path: Path, limit: int | None) -> list[ContextUsageEvent]:
    return telemetry_service.load_persistent_events(path, limit=limit)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export recent ContextUsageEvent telemetry data")
    parser.add_argument("--format", choices={"json", "csv"}, default="json", help="Output format")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of events exported (default: entire buffer)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=telemetry_service.default_telemetry_path(),
        help="Path to the telemetry buffer file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path (defaults to stdout)",
    )
    args = parser.parse_args(argv)

    limit = args.limit if args.limit and args.limit > 0 else None
    events = _load_events(args.source, limit)

    if args.output is None:
        destination = sys.stdout
        close_handle = False
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        destination = args.output.open("w", encoding="utf-8", newline="")
        close_handle = True

    try:
        if args.format == "json":
            _write_json(events, destination)
        else:
            _write_csv(events, destination)
    finally:
        if close_handle:
            destination.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
