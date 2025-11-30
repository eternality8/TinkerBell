"""YAML/JSON syntax helpers and validation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - dependency provided via pyproject extras
    import jsonschema
except Exception:  # pragma: no cover - graceful fallback when optional dep missing
    jsonschema = None  # type: ignore[assignment]

try:  # pragma: no cover - dependency provided via pyproject extras
    from ruamel.yaml import YAML
    from ruamel.yaml.error import MarkedYAMLError
except Exception:  # pragma: no cover - graceful fallback when optional dep missing
    YAML = None  # type: ignore[assignment]
    MarkedYAMLError = None  # type: ignore[assignment]

MAX_SCHEMA_ERRORS = 25


@dataclass(slots=True)
class ValidationError:
    """Represents a schema validation error returned to the UI."""

    message: str
    line: int | None = None


def validate_yaml(text: str) -> list[ValidationError]:
    """Validate YAML content and return any issues."""

    raw = (text or "").strip()
    if not raw:
        return []

    parser = _create_yaml_parser()
    if parser is None:
        return [ValidationError(message="YAML validation requires the 'ruamel.yaml' dependency.")]

    try:
        # ``load_all`` ensures multi-document files are processed entirely.
        list(parser.load_all(text))
        return []
    except MarkedYAMLError as exc:  # type: ignore[misc]
        return [_validation_error_from_yaml(exc)]
    except Exception as exc:  # pragma: no cover - defensive guard
        return [ValidationError(message=str(exc) or "Invalid YAML content")]


def validate_json(text: str, *, schema: dict[str, Any] | None = None) -> list[ValidationError]:
    """Validate JSON content and return any issues."""

    raw = (text or "").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw, object_pairs_hook=_NoDuplicateKeys.dict_factory)
    except DuplicateJSONKeyError as exc:
        return [ValidationError(message=str(exc))]
    except JSONDecodeError as exc:
        return [ValidationError(message=_format_json_decode_message(exc), line=exc.lineno)]

    if not schema:
        return []

    if jsonschema is None:  # pragma: no cover - dependency always installed in CI
        return [ValidationError(message="JSON schema validation requires the 'jsonschema' dependency.")]

    errors: list[ValidationError] = []
    validator_cls = _resolve_json_validator()
    if validator_cls is None:  # pragma: no cover - defensive guard
        return [ValidationError(message="No compatible JSON Schema validator available.")]

    try:
        validator = validator_cls(schema)  # type: ignore[call-arg]
    except jsonschema.exceptions.SchemaError as exc:  # type: ignore[union-attr]
        return [ValidationError(message=f"Invalid JSON schema: {exc.message}")]

    for issue in validator.iter_errors(parsed):
        path = _format_schema_path(issue.absolute_path)
        msg = issue.message
        if path:
            msg = f"{path}: {msg}"
        errors.append(ValidationError(message=msg))
        if len(errors) >= MAX_SCHEMA_ERRORS:
            errors.append(ValidationError(message="Too many validation errors; stopping early."))
            break

    return errors


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------
def _create_yaml_parser() -> YAML | None:  # type: ignore[valid-type]
    if YAML is None:  # pragma: no cover - dependency always installed in CI
        return None
    parser = YAML(typ="safe")
    parser.allow_duplicate_keys = False
    parser.preserve_quotes = False
    parser.default_flow_style = False
    parser.width = 4096
    return parser


def _validation_error_from_yaml(exc: MarkedYAMLError) -> ValidationError:  # type: ignore[valid-type]
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    line = _extract_mark_line(mark)
    detail = exc.problem or str(exc) or "Invalid YAML content"
    if exc.context:
        detail = f"{detail}; {exc.context}"
    return ValidationError(message=detail, line=line)


def _extract_mark_line(mark: Any) -> int | None:
    if mark is None:
        return None
    try:
        return int(mark.line) + 1
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
class DuplicateJSONKeyError(ValueError):
    """Raised when a duplicate key is encountered during JSON parsing."""

    def __init__(self, key: str) -> None:
        super().__init__(f"Duplicate key '{key}' found in JSON object.")
        self.key = key


class _NoDuplicateKeys(dict):
    @staticmethod
    def dict_factory(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        sentinel: dict[str, Any] = {}
        for key, value in pairs:
            if key in sentinel:
                raise DuplicateJSONKeyError(key)
            sentinel[key] = value
        return sentinel


def _resolve_json_validator():
    if jsonschema is None:  # pragma: no cover
        return None
    if hasattr(jsonschema, "Draft202012Validator"):
        return jsonschema.Draft202012Validator
    if hasattr(jsonschema, "Draft7Validator"):
        return jsonschema.Draft7Validator
    return None


def _format_json_decode_message(exc: JSONDecodeError) -> str:
    snippet = exc.doc.splitlines()[exc.lineno - 1].strip() if exc.doc and exc.lineno else ""
    detail = exc.msg
    if snippet:
        return f"{detail} (line {exc.lineno}, column {exc.colno}): {snippet}"
    return f"{detail} (line {exc.lineno}, column {exc.colno})"


def _format_schema_path(path: Sequence[Any]) -> str:
    if not path:
        return ""
    components: list[str] = []
    for segment in path:
        if isinstance(segment, int):
            if components:
                components[-1] = f"{components[-1]}[{segment}]"
            else:
                components.append(f"[{segment}]")
        else:
            components.append(str(segment))
    return ".".join(filter(None, components))
