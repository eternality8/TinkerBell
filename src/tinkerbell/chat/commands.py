"""Parsing and validation helpers for AI-issued commands."""

from __future__ import annotations

import json
import os
import re
import shlex
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft7Validator, ValidationError

from ..documents.ranges import TextRange
from ..services.telemetry import emit as telemetry_emit


class ActionType(str, Enum):
    """Supported edit directive actions."""

    INSERT = "insert"
    REPLACE = "replace"
    ANNOTATE = "annotate"
    PATCH = "patch"


class ManualCommandType(str, Enum):
    """Manual chat commands handled locally without the AI model."""

    OUTLINE = "outline"
    FIND_SECTIONS = "find_sections"
    ANALYZE = "analyze"
    STATUS = "status"


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating an edit directive."""

    ok: bool
    message: str = ""


@dataclass(slots=True)
class ManualCommandRequest:
    """Parsed representation of a manual chat command string."""

    command: ManualCommandType
    args: dict[str, Any]
    raw: str


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
_MANUAL_PREFIXES = ("/", "::", "!")
_OUTLINE_FLAG_ALIASES = {
    "--doc": "document_id",
    "--document": "document_id",
    "--tab": "document_id",
    "--levels": "desired_levels",
    "-l": "desired_levels",
    "--max-nodes": "max_nodes",
    "--max": "max_nodes",
}
_FIND_FLAG_ALIASES = {
    "--doc": "document_id",
    "--document": "document_id",
    "--tab": "document_id",
    "--top": "top_k",
    "--top-k": "top_k",
    "--k": "top_k",
    "--confidence": "min_confidence",
    "--min-confidence": "min_confidence",
    "--query": "query",
    "-q": "query",
}
_ANALYZE_FLAG_ALIASES = {
    "--doc": "document_id",
    "--document": "document_id",
    "--tab": "document_id",
    "--start": "selection_start",
    "--end": "selection_end",
    "--reason": "reason",
}
_ANALYZE_BOOLEAN_FLAGS = {
    "--refresh": True,
    "--force-refresh": True,
}
_STATUS_FLAG_ALIASES = {
    "--doc": "document_id",
    "--document": "document_id",
    "--tab": "document_id",
}
_STATUS_BOOLEAN_FLAGS = {
    "--json": True,
    "--as-json": True,
}
_MANUAL_COMMAND_ALIASES = {
    "outline": ManualCommandType.OUTLINE,
    "out": ManualCommandType.OUTLINE,
    "ol": ManualCommandType.OUTLINE,
    "find": ManualCommandType.FIND_SECTIONS,
    "search": ManualCommandType.FIND_SECTIONS,
    "sections": ManualCommandType.FIND_SECTIONS,
    "retrieve": ManualCommandType.FIND_SECTIONS,
    "retrieval": ManualCommandType.FIND_SECTIONS,
    "analyze": ManualCommandType.ANALYZE,
    "analysis": ManualCommandType.ANALYZE,
    "status": ManualCommandType.STATUS,
    "stat": ManualCommandType.STATUS,
}
_OUTLINE_BOOLEAN_FLAGS = {
    "--blurbs": True,
    "--include-blurbs": True,
    "--with-blurbs": True,
    "--no-blurbs": False,
    "--without-blurbs": False,
    "--brief": False,
}
_FIND_BOOLEAN_FLAGS = {
    "--outline": True,
    "--outline-context": True,
    "--with-outline": True,
    "--no-outline": False,
    "--no-outline-context": False,
    "--without-outline": False,
}
_DEPRECATED_CARET_FIELDS = (
    "insert_at_cursor",
    "cursor_offset",
    "selection_start",
    "selection_end",
)


def is_manual_command(text: str) -> bool:
    """Return ``True`` when ``text`` starts with a manual command prefix."""

    normalized = (text or "").strip()
    if not normalized:
        return False
    return _split_manual_prefix(normalized) is not None


def parse_manual_command(text: str) -> ManualCommandRequest | None:
    """Parse ``text`` into a :class:`ManualCommandRequest` when prefixed."""

    normalized = (text or "").strip()
    if not normalized:
        return None
    prefix = _split_manual_prefix(normalized)
    if prefix is None:
        return None
    _, remainder = prefix
    if not remainder:
        raise ValueError("Manual command is missing a verb. Try /outline or /find.")
    tokens = _tokenize_manual_command(remainder)
    if not tokens:
        raise ValueError("Manual command is missing a verb. Try /outline or /find.")
    command_token = tokens.popleft().lower()
    command = _MANUAL_COMMAND_ALIASES.get(command_token)
    if command is None:
        raise ValueError(f"Unknown manual command '{command_token}'. Try /outline or /find.")
    if command is ManualCommandType.OUTLINE:
        args = _parse_outline_command(tokens)
    elif command is ManualCommandType.ANALYZE:
        args = _parse_analyze_command(tokens)
    elif command is ManualCommandType.STATUS:
        args = _parse_status_command(tokens)
    else:
        args = _parse_find_sections_command(tokens)
    return ManualCommandRequest(command=command, args=args, raw=normalized)


def _split_manual_prefix(text: str) -> tuple[str, str] | None:
    candidate = text.lstrip()
    for prefix in _MANUAL_PREFIXES:
        if candidate.startswith(prefix):
            remainder = candidate[len(prefix) :].lstrip()
            return prefix, remainder
    return None


def _tokenize_manual_command(text: str) -> deque[str]:
    try:
        parts = shlex.split(text, posix=True)
    except ValueError as exc:  # pragma: no cover - shlex provides the details
        raise ValueError(f"Unable to parse manual command: {exc}") from exc
    return deque(parts)


def _parse_outline_command(tokens: deque[str]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    positional: list[str] = []
    while tokens:
        token = tokens.popleft()
        if token == "--":
            positional.extend(tokens)
            tokens.clear()
            break
        if _is_flag_token(token):
            flag, inline_value = _split_flag_value(token)
            bool_value = _OUTLINE_BOOLEAN_FLAGS.get(flag)
            if bool_value is not None:
                args["include_blurbs"] = bool_value
                continue
            normalized = _OUTLINE_FLAG_ALIASES.get(flag)
            if normalized is None:
                raise ValueError(f"Unknown outline flag '{token}'")
            value = inline_value if inline_value is not None else _pop_flag_value(tokens, token)
            cleaned = value.strip()
            if inline_value is None and normalized == "document_id":
                cleaned = _collect_reference_value(cleaned, tokens)
            if not cleaned:
                raise ValueError(f"Flag '{token}' requires a value")
            if normalized == "desired_levels":
                args[normalized] = _coerce_int_flag(cleaned, token, minimum=1)
            elif normalized == "max_nodes":
                args[normalized] = _coerce_int_flag(cleaned, token, minimum=1, maximum=1000)
            else:
                args[normalized] = cleaned
            continue
        positional.append(token)
    if positional and "document_id" not in args:
        reference = " ".join(positional).strip()
        if reference:
            args["document_id"] = reference
    return args


def _parse_find_sections_command(tokens: deque[str]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    query_tokens: list[str] = []
    while tokens:
        token = tokens.popleft()
        if token == "--":
            query_tokens.extend(tokens)
            tokens.clear()
            break
        if _is_flag_token(token):
            flag, inline_value = _split_flag_value(token)
            bool_value = _FIND_BOOLEAN_FLAGS.get(flag)
            if bool_value is not None:
                args["include_outline_context"] = bool_value
                continue
            normalized = _FIND_FLAG_ALIASES.get(flag)
            if normalized is None:
                raise ValueError(f"Unknown retrieval flag '{token}'")
            value = inline_value if inline_value is not None else _pop_flag_value(tokens, token)
            cleaned = value.strip()
            if inline_value is None and normalized == "document_id":
                cleaned = _collect_reference_value(cleaned, tokens)
            if not cleaned:
                raise ValueError(f"Flag '{token}' requires a value")
            if normalized == "top_k":
                args[normalized] = _coerce_int_flag(cleaned, token, minimum=1, maximum=12)
            elif normalized == "min_confidence":
                args[normalized] = _coerce_float_flag(cleaned, token, minimum=0.0, maximum=1.0)
            else:
                args[normalized] = cleaned
            continue
        if "document_id" not in args and _looks_like_tab_reference(token):
            args["document_id"] = token.strip()
            continue
        query_tokens.append(token)
    if query_tokens:
        existing = (args.get("query") or "").strip()
        suffix = " ".join(query_tokens).strip()
        args["query"] = suffix if not existing else f"{existing} {suffix}".strip()
    query_text = (args.get("query") or "").strip()
    if not query_text:
        raise ValueError("Find sections command requires a query, e.g., /find introduction paragraph")
    args["query"] = query_text
    return args


def _parse_analyze_command(tokens: deque[str]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    positional: list[str] = []
    while tokens:
        token = tokens.popleft()
        if token == "--":
            positional.extend(tokens)
            tokens.clear()
            break
        if _is_flag_token(token):
            flag, inline_value = _split_flag_value(token)
            if flag in _ANALYZE_BOOLEAN_FLAGS:
                args["force_refresh"] = _ANALYZE_BOOLEAN_FLAGS[flag]
                continue
            normalized = _ANALYZE_FLAG_ALIASES.get(flag)
            if normalized is None:
                raise ValueError(f"Unknown analyze flag '{token}'")
            value = inline_value if inline_value is not None else _pop_flag_value(tokens, token)
            cleaned = value.strip()
            if not cleaned:
                raise ValueError(f"Flag '{token}' requires a value")
            if normalized in {"selection_start", "selection_end"}:
                args[normalized] = _coerce_int_flag(cleaned, token, minimum=0)
            else:
                if inline_value is None and normalized == "document_id":
                    cleaned = _collect_reference_value(cleaned, tokens)
                args[normalized] = cleaned
            continue
        positional.append(token)
    if positional and "document_id" not in args:
        reference = " ".join(positional).strip()
        if reference:
            args["document_id"] = reference
    return args


def _parse_status_command(tokens: deque[str]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    positional: list[str] = []
    while tokens:
        token = tokens.popleft()
        if token == "--":
            positional.extend(tokens)
            tokens.clear()
            break
        if _is_flag_token(token):
            flag, inline_value = _split_flag_value(token)
            if flag in _STATUS_BOOLEAN_FLAGS:
                args["as_json"] = True
                continue
            normalized = _STATUS_FLAG_ALIASES.get(flag)
            if normalized is None:
                raise ValueError(f"Unknown status flag '{token}'")
            value = inline_value if inline_value is not None else _pop_flag_value(tokens, token)
            cleaned = value.strip()
            if inline_value is None and normalized == "document_id":
                cleaned = _collect_reference_value(cleaned, tokens)
            if not cleaned:
                raise ValueError(f"Flag '{token}' requires a value")
            args[normalized] = cleaned
            continue
        positional.append(token)
    if positional and "document_id" not in args:
        reference = " ".join(positional).strip()
        if reference:
            args["document_id"] = reference
    return args


def _is_flag_token(token: str) -> bool:
    if not token or token == "--":
        return token == "--"
    return token.startswith("-") and token not in {"-"}


def _split_flag_value(token: str) -> tuple[str, str | None]:
    if "=" in token:
        flag, value = token.split("=", 1)
        return flag.lower(), value
    return token.lower(), None


def _pop_flag_value(tokens: deque[str], flag: str) -> str:
    if not tokens:
        raise ValueError(f"Flag '{flag}' requires a value")
    value = tokens.popleft()
    if value == "--":
        raise ValueError(f"Flag '{flag}' requires a value")
    if _is_flag_token(value) and not _looks_numeric(value):
        raise ValueError(f"Flag '{flag}' requires a value")
    return value


def _collect_reference_value(initial: str, tokens: deque[str]) -> str:
    parts: list[str] = [initial]
    while tokens:
        peek = tokens[0]
        if peek == "--" or _is_flag_token(peek):
            break
        parts.append(tokens.popleft())
    return " ".join(part for part in parts if part).strip()


def _coerce_int_flag(value: str, flag: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag} expects an integer value") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{flag} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{flag} must be <= {maximum}")
    return number


def _coerce_float_flag(value: str, flag: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag} expects a numeric value") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{flag} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{flag} must be <= {maximum}")
    return number


def _looks_like_tab_reference(token: str) -> bool:
    normalized = (token or "").strip().lower()
    if not normalized:
        return False
    if _TAB_NUMBER_RE.match(normalized):
        return True
    if _UNTITLED_REFERENCE_RE.match(normalized):
        return True
    return normalized in {"active", "current"}


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False

_TEXT_RANGE_OBJECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "start": {"type": "integer", "minimum": 0},
        "end": {"type": "integer", "minimum": 0},
    },
    "required": ["start", "end"],
    "additionalProperties": False,
}

_TEXT_RANGE_SEQUENCE_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
}


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
                _TEXT_RANGE_OBJECT_SCHEMA,
                _TEXT_RANGE_SEQUENCE_SCHEMA,
                {"type": "string", "enum": ["document"]},
                {"type": "null"},
            ]
        },
        "document_version": {"type": "string"},
        "snapshot_version": {"type": "string"},
        "version": {"type": "string"},
        "document_digest": {"type": "string"},
        "metadata": {"type": "object"},
        "tab_id": {"type": "string", "minLength": 1},
        "replace_all": {"type": "boolean"},
        "selection_fingerprint": {"type": "string", "minLength": 1},
        "match_text": {"type": "string"},
        "expected_text": {"type": "string"},
        "ranges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 0},
                    "replacement": {"type": "string"},
                    "match_text": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "chunk_hash": {"type": "string"},
                },
                "required": ["start", "end", "replacement", "match_text"],
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": True,
    "allOf": [
        {
            "if": {"properties": {"action": {"const": ActionType.PATCH.value}}},
            "then": {
                "anyOf": [
                    {"required": ["diff"]},
                    {"required": ["ranges"]},
                ],
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

    deprecated_fields = _collect_deprecated_caret_fields(candidate)
    if deprecated_fields:
        _emit_caret_block_event(candidate, deprecated_fields, source="directive_schema")
        field_list = ", ".join(sorted(deprecated_fields))
        return ValidationResult(
            ok=False,
            message=(
                f"Deprecated caret parameters ({field_list}) are no longer supported. "
                "Call document_snapshot/selection_range and send target_range or replace_all=true."
            ),
        )

    action = candidate.get("action")
    if action == ActionType.PATCH.value:
        diff = candidate.get("diff")
        ranges = candidate.get("ranges")
        has_diff = isinstance(diff, str) and diff.strip()
        has_ranges = isinstance(ranges, Sequence) and len(ranges) > 0
        if not has_diff and not has_ranges:
            return ValidationResult(ok=False, message="patch directives require a diff or ranges payload")
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
    if "target_range" in normalized:
        normalized["target_range"] = _normalize_target_range_value(normalized["target_range"])

    target_range = normalized.get("target_range")
    if isinstance(target_range, tuple):
        normalized["target_range"] = list(target_range)

    tab_reference = extract_tab_reference(normalized)
    if tab_reference and not normalized.get("tab_reference"):
        normalized["tab_reference"] = tab_reference

    return normalized


def _collect_deprecated_caret_fields(payload: Mapping[str, Any]) -> set[str]:
    hits: set[str] = set()
    for field in _DEPRECATED_CARET_FIELDS:
        if field in payload:
            hits.add(field)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for field in _DEPRECATED_CARET_FIELDS:
            if field in metadata:
                hits.add(field)
    return hits


def _emit_caret_block_event(payload: Mapping[str, Any], fields: set[str], *, source: str) -> None:
    if not fields:
        return
    telemetry_payload = {
        "document_id": payload.get("document_id"),
        "tab_id": payload.get("tab_id"),
        "source": source,
        "blocked_fields": sorted(fields),
    }
    telemetry_emit("caret_call_blocked", telemetry_payload)


def _normalize_target_range_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, TextRange):
        return value.to_dict()
    try:
        return TextRange.from_value(value).to_dict()
    except (TypeError, ValueError):
        return value


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

