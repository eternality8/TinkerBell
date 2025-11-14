"""Validation helpers for YAML/JSON snippets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ValidationOutcome:
    """Represents the result of validating a snippet."""

    ok: bool
    message: str = ""


def validate_snippet(text: str, fmt: str) -> ValidationOutcome:
    """Validate YAML/JSON text and return an outcome."""

    del text, fmt
    return ValidationOutcome(ok=True, message="validation stub")

