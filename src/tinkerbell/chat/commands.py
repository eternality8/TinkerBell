"""Parsing and validation helpers for AI-issued commands."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from json import JSONDecodeError
from typing import Any, Dict

from jsonschema import Draft7Validator, ValidationError


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


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*)```$", re.IGNORECASE | re.DOTALL)
_JSON_DECODER = json.JSONDecoder()

_DIRECTIVE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["action", "content"],
    "properties": {
        "action": {
            "type": "string",
            "enum": [item.value for item in ActionType],
        },
        "content": {"type": "string", "minLength": 1},
        "rationale": {"type": "string"},
        "target_range": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 0},
                    },
                    "additionalProperties": False,
                },
                {"type": "null"},
            ]
        },
        "document_version": {"type": "string"},
        "snapshot_version": {"type": "string"},
        "version": {"type": "string"},
        "document_digest": {"type": "string"},
        "metadata": {"type": "object"},
    },
    "additionalProperties": True,
}

_DIRECTIVE_VALIDATOR = Draft7Validator(_DIRECTIVE_SCHEMA)


def parse_agent_payload(payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
    """Normalize agent payload dictionaries or JSON-encoded command strings."""

    mapping = _coerce_payload(payload)
    return _normalize_payload(mapping)


def validate_directive(payload: Mapping[str, Any]) -> ValidationResult:
    """Validate the given directive against schema and semantic rules."""

    if not isinstance(payload, Mapping):
        return ValidationResult(ok=False, message="Directive payload must be a mapping")

    candidate: Dict[str, Any] = dict(payload)
    action = candidate.get("action")
    if isinstance(action, str):
        candidate["action"] = action.strip().lower()

    try:
        _DIRECTIVE_VALIDATOR.validate(candidate)
    except ValidationError as error:
        return ValidationResult(ok=False, message=_format_validation_error(error))

    content = candidate.get("content")
    if isinstance(content, str) and not content.strip():
        return ValidationResult(ok=False, message="content must not be empty")

    return ValidationResult(ok=True)


def _coerce_payload(payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise ValueError("Directive payload is empty")
        text = _strip_code_fence(text)
        parsed = _loads_with_fallback(text)
        if not isinstance(parsed, Mapping):
            raise ValueError("Directive payload must decode to an object")
        return dict(parsed)
    raise TypeError("Directive payload must be a mapping or JSON string")


def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)

    action = normalized.get("action")
    if isinstance(action, str):
        normalized["action"] = action.strip().lower()

    if "target_range" not in normalized:
        for alias in ("target", "range", "selection"):
            if alias in normalized:
                normalized["target_range"] = normalized.pop(alias)
                break

    target_range = normalized.get("target_range")
    if isinstance(target_range, tuple):
        normalized["target_range"] = list(target_range)

    return normalized


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match:
        return match.group("body").strip()
    return text


def _loads_with_fallback(text: str) -> Any:
    try:
        return json.loads(text)
    except JSONDecodeError as exc:
        for idx, char in enumerate(text):
            if char in "{[":
                try:
                    parsed, _ = _JSON_DECODER.raw_decode(text[idx:])
                except JSONDecodeError:
                    continue
                return parsed
        raise ValueError("Unable to parse directive payload as JSON") from exc


def _format_validation_error(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.path)
    if path:
        return f"{path}: {error.message}"
    return error.message

