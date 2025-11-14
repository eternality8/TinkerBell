"""Validation helpers for YAML/JSON snippets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from tinkerbell.editor.syntax import yaml_json


@dataclass(slots=True)
class ValidationOutcome:
    """Represents the result of validating a snippet."""

    ok: bool
    message: str = ""


_FORMAT_VALIDATORS: Dict[str, Callable[[str], List[yaml_json.ValidationError]]] = {
    "yaml": yaml_json.validate_yaml,
    "yml": yaml_json.validate_yaml,
    "json": yaml_json.validate_json,
}


def validate_snippet(text: str, fmt: str) -> ValidationOutcome:
    """Validate YAML/JSON text and return an outcome."""

    validator = _FORMAT_VALIDATORS.get((fmt or "").strip().lower())
    if validator is None:
        return ValidationOutcome(ok=False, message=f"Unsupported format '{fmt}'.")

    errors = validator(text)
    if not errors:
        return ValidationOutcome(ok=True, message="Snippet is valid.")

    head = errors[0]
    message = head.message
    if head.line:
        message = f"Line {head.line}: {message}"
    if len(errors) > 1:
        remaining = len(errors) - 1
        message = f"{message} (+{remaining} more issue{'s' if remaining > 1 else ''})."
    return ValidationOutcome(ok=False, message=message)

