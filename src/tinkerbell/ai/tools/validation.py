"""Validation helpers for YAML/JSON snippets with markdown lint stubs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List, MutableMapping

from tinkerbell.editor.syntax import yaml_json


@dataclass(slots=True)
class ValidationOutcome:
    """Represents the result of validating a snippet."""

    ok: bool
    message: str = ""


SnippetValidator = Callable[[str], List[yaml_json.ValidationError]]


_HEADING_PATTERN = re.compile(r"^(?P<hashes>#{1,6})\s+\S")
_FENCE_PATTERN = re.compile(r"^(```+|~~~+)(.*)$")


def _validate_markdown(text: str) -> List[yaml_json.ValidationError]:
    raw = text or ""
    if not raw.strip():
        return []

    errors: List[yaml_json.ValidationError] = []
    fence_stack: List[tuple[str, int]] = []
    previous_heading_level: int | None = None

    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.lstrip()
        fence_match = _FENCE_PATTERN.match(stripped)
        if fence_match:
            marker_sequence = fence_match.group(1)
            marker_char = marker_sequence[0]
            trailing = stripped[len(marker_sequence) :].strip()
            if fence_stack and fence_stack[-1][0] == marker_char and not trailing:
                fence_stack.pop()
            else:
                fence_stack.append((marker_char, line_number))
            continue

        if fence_stack:
            continue

        heading_match = _HEADING_PATTERN.match(stripped)
        if not heading_match:
            continue

        level = len(heading_match.group("hashes"))
        if previous_heading_level is None:
            previous_heading_level = level
            continue
        if level - previous_heading_level > 1:
            errors.append(
                yaml_json.ValidationError(
                    message=f"Heading level jumps from H{previous_heading_level} to H{level}.",
                    line=line_number,
                )
            )
        previous_heading_level = level

    if fence_stack:
        _, start_line = fence_stack[-1]
        errors.append(
            yaml_json.ValidationError(
                message="Unclosed fenced code block detected; add a closing fence.",
                line=start_line,
            )
        )

    return errors


_FORMAT_VALIDATORS: MutableMapping[str, SnippetValidator] = {
    "yaml": yaml_json.validate_yaml,
    "yml": yaml_json.validate_yaml,
    "json": yaml_json.validate_json,
    "markdown": _validate_markdown,
    "md": _validate_markdown,
}


def register_snippet_validator(fmt: str, validator: SnippetValidator) -> None:
    """Register or replace a snippet validator for custom formats."""

    key = (fmt or "").strip().lower()
    if not key:
        raise ValueError("Format identifier must be a non-empty string.")
    if not callable(validator):
        raise ValueError("Validator callable is required.")
    _FORMAT_VALIDATORS[key] = validator


def validate_snippet(text: str, fmt: str, *, schema: dict[str, Any] | None = None) -> ValidationOutcome:
    """Validate text for the requested format and summarize the first issue."""

    key = (fmt or "").strip().lower()
    validator = _FORMAT_VALIDATORS.get(key)
    if validator is None:
        return ValidationOutcome(ok=False, message=f"Unsupported format '{fmt}'.")

    if key == "json" and schema is not None:
        errors = yaml_json.validate_json(text, schema=schema)
    else:
        errors = validator(text)
    return _outcome_from_errors(errors)


def _outcome_from_errors(errors: List[yaml_json.ValidationError]) -> ValidationOutcome:
    if not errors:
        return ValidationOutcome(ok=True, message="Snippet is valid.")

    head = errors[0]
    message = head.message
    if head.line:
        message = f"Line {head.line}: {message}"
    if len(errors) > 1:
        remaining = len(errors) - 1
        plural = "s" if remaining > 1 else ""
        message = f"{message} (+{remaining} more issue{plural})."
    return ValidationOutcome(ok=False, message=message)


validate_snippet.summarizable = False


__all__ = ["ValidationOutcome", "validate_snippet", "register_snippet_validator"]

