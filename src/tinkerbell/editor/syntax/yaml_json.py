"""YAML/JSON syntax helpers and validation stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class ValidationError:
    """Represents a schema validation error returned to the UI."""

    message: str
    line: int | None = None


def validate_yaml(text: str) -> List[ValidationError]:
    """Validate YAML content and return any issues."""

    del text
    return []


def validate_json(text: str) -> List[ValidationError]:
    """Validate JSON content and return any issues."""

    del text
    return []
