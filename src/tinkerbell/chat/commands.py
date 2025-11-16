"""Parsing and validation helpers for AI-issued commands."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft7Validator, ValidationError


class ActionType(str, Enum):
    """Supported edit directive actions."""

    INSERT = "insert"
    REPLACE = "replace"
    ANNOTATE = "annotate"
    PATCH = "patch"


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating an edit directive."""

    ok: bool
    message: str = ""


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*)```$", re.IGNORECASE | re.DOTALL)
_JSON_DECODER = json.JSONDecoder()
_TAB_REFERENCE_KEYS = (
    "tab",
    "tab_reference",
    "tab_label",
    "tab_name",
    "target_tab",
    "document",
    "document_title",
    "document_name",
    "file",
    "file_name",
)
_UNTITLED_REFERENCE_RE = re.compile(r"^untitled\s*(?P<index>\d+)$", re.IGNORECASE)
_TAB_NUMBER_RE = re.compile(r"^(?:tab\s*#?|#)(?P<number>\d+)$", re.IGNORECASE)

DIRECTIVE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["action"],
    "properties": {
        "action": {
            "type": "string",
            "enum": [item.value for item in ActionType],
        },
        "content": {"type": "string", "minLength": 1},
        "diff": {"type": "string", "minLength": 1},
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
        "tab_id": {"type": "string", "minLength": 1},
        "selection_fingerprint": {"type": "string", "minLength": 1},
    },
    "additionalProperties": True,
    "allOf": [
        {
            "if": {"properties": {"action": {"const": ActionType.PATCH.value}}},
            "then": {
                "required": ["diff"],
                "not": {
                    "anyOf": [
                        {"required": ["content"]},
                        {"required": ["target_range"]},
                    ]
                },
            },
            "else": {"required": ["content"]},
        }
    ],
}

_DIRECTIVE_VALIDATOR = Draft7Validator(DIRECTIVE_SCHEMA)


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

    action = candidate.get("action")
    if action == ActionType.PATCH.value:
        diff = candidate.get("diff")
        if not isinstance(diff, str) or not diff.strip():
            return ValidationResult(ok=False, message="diff must not be empty for patch directives")
        version_token = _extract_version_token(candidate)
        if not version_token:
            return ValidationResult(ok=False, message="patch directives require a document_version token")

    return ValidationResult(ok=True)


def create_patch_directive(diff: str, version: str, rationale: str | None = None) -> Dict[str, Any]:
    """Helper used by agents/tests to construct a valid patch directive payload."""

    if not diff or not diff.strip():
        raise ValueError("diff must be a non-empty unified diff string")
    version_token = str(version).strip()
    if not version_token:
        raise ValueError("version must be a non-empty string")
    directive: Dict[str, Any] = {
        "action": ActionType.PATCH.value,
        "diff": diff,
        "document_version": version_token,
    }
    if rationale:
        directive["rationale"] = rationale
    return directive


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

    tab_reference = extract_tab_reference(normalized)
    if tab_reference and not normalized.get("tab_reference"):
        normalized["tab_reference"] = tab_reference

    return normalized


def resolve_tab_reference(
    reference: str,
    tabs: Sequence[Mapping[str, Any]],
    *,
    active_tab_id: str | None = None,
) -> str | None:
    """Resolve ``reference`` to a concrete ``tab_id`` using ``tabs`` metadata."""

    text = (reference or "").strip()
    if not text:
        return None
    normalized = text.lower()
    normalized_tabs = list(tabs or [])
    if not normalized_tabs:
        return text

    resolved_tabs: list[tuple[int, str, Mapping[str, Any]]] = []
    for idx, entry in enumerate(normalized_tabs, start=1):
        tab_id = _coerce_tab_id(entry)
        if not tab_id:
            continue
        resolved_tabs.append((idx, tab_id, entry))
        if tab_id.lower() == normalized:
            return tab_id

    index_match = _TAB_NUMBER_RE.match(normalized)
    if index_match:
        number = int(index_match.group("number"))
        for idx, tab_id, _ in resolved_tabs:
            if idx == number:
                return tab_id

    untitled_match = _UNTITLED_REFERENCE_RE.match(normalized)
    if untitled_match:
        target = int(untitled_match.group("index"))
        for _, tab_id, entry in resolved_tabs:
            if int(entry.get("untitled_index") or -1) == target:
                return tab_id

    exact_matches = _match_tabs_by_fields(normalized, resolved_tabs, exact=True)
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        preferred = _pick_preferred_tab(exact_matches, active_tab_id)
        if preferred:
            return preferred

    substring_matches = _match_tabs_by_fields(normalized, resolved_tabs, exact=False)
    if substring_matches:
        preferred = _pick_preferred_tab(substring_matches, active_tab_id)
        if preferred:
            return preferred

    return None


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


def _extract_version_token(payload: Mapping[str, Any]) -> str | None:
    for key in ("document_version", "snapshot_version", "version", "document_digest"):
        token = payload.get(key)
        if token is None:
            continue
        token_str = str(token).strip()
        if token_str:
            return token_str
    return None


def extract_tab_reference(payload: Mapping[str, Any]) -> str | None:
    for key in _TAB_REFERENCE_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, (int, float)):
            return f"Tab {int(candidate)}"
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        return extract_tab_reference(dict(metadata))
    return None


def _coerce_tab_id(entry: Mapping[str, Any]) -> str:
    return str(entry.get("tab_id") or entry.get("id") or "").strip()


def _match_tabs_by_fields(
    normalized_reference: str,
    tabs: Sequence[tuple[int, str, Mapping[str, Any]]],
    *,
    exact: bool,
) -> list[str]:
    matches: list[str] = []
    for _, tab_id, entry in tabs:
        for field in _iter_tab_fields(entry):
            candidate = field.lower()
            if not candidate:
                continue
            if exact and candidate == normalized_reference:
                matches.append(tab_id)
                break
            if not exact and normalized_reference in candidate:
                matches.append(tab_id)
                break
    return matches


def _iter_tab_fields(entry: Mapping[str, Any]) -> list[str]:
    fields: list[str] = []
    title = entry.get("title")
    if isinstance(title, str):
        fields.append(title.strip())
    path = entry.get("path")
    if isinstance(path, str) and path.strip():
        normalized_path = path.strip()
        fields.append(normalized_path)
        basename = os.path.basename(normalized_path) or Path(normalized_path).name
        if basename:
            fields.append(basename)
    label = entry.get("label")
    if isinstance(label, str):
        fields.append(label.strip())
    return [field for field in fields if field]


def _pick_preferred_tab(candidates: list[str], active_tab_id: str | None) -> str | None:
    if not candidates:
        return None
    if active_tab_id and active_tab_id in candidates:
        return active_tab_id
    return candidates[0]

