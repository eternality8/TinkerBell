"""Parsing and validation helpers for AI-issued commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class ActionType(str, Enum):
    """Supported edit directive actions."""

    INSERT = "insert"
    REPLACE = "replace"
    ANNOTATE = "annotate"


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating an edit directive."""

    ok: bool
    message: str = ""


def parse_agent_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize agent payload dictionaries."""

    return payload


def validate_directive(payload: Dict[str, Any]) -> ValidationResult:
    """Validate the given directive against basic schema rules."""

    required = {"action", "content"}
    missing = [field for field in required if field not in payload]
    if missing:
        return ValidationResult(ok=False, message=f"Missing fields: {', '.join(missing)}")
    if payload.get("action") not in {item.value for item in ActionType}:
        return ValidationResult(ok=False, message="Unsupported action")
    return ValidationResult(ok=True)
