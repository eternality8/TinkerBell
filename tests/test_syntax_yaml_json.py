"""Tests for YAML/JSON syntax helpers."""

from __future__ import annotations

from tinkerbell.editor.syntax.yaml_json import validate_json, validate_yaml


def test_validate_yaml_accepts_valid_documents() -> None:
    yaml_text = """
    name: Example
    tags:
      - alpha
      - beta
    metadata:
      nested:
        value: 42
    """

    errors = validate_yaml(yaml_text)

    assert errors == []


def test_validate_yaml_reports_duplicate_keys_with_line_numbers() -> None:
    yaml_text = """
    foo: 1
    foo: 2
    """

    errors = validate_yaml(yaml_text)

    assert len(errors) == 1
    assert errors[0].line == 3
    assert "duplicate key" in errors[0].message.lower()


def test_validate_json_flags_decode_errors_with_line_numbers() -> None:
    json_text = """
    {
      "foo": 1,,
      "bar": 2
    }
    """

    errors = validate_json(json_text)

    assert len(errors) == 1
    assert errors[0].line == 2
    assert "line 2" in errors[0].message.lower()


def test_validate_json_detects_duplicate_keys() -> None:
    json_text = '{"foo": 1, "foo": 2}'

    errors = validate_json(json_text)

    assert len(errors) == 1
    assert "duplicate key" in errors[0].message.lower()


def test_validate_json_schema_validation_reports_paths() -> None:
    payload = """
    {
      "items": [
        {"id": 1, "quantity": -1}
      ]
    }
    """
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "quantity": {"type": "integer", "minimum": 0},
                    },
                    "required": ["id", "quantity"],
                },
            }
        },
        "required": ["items"],
    }

    errors = validate_json(payload, schema=schema)

    assert errors
    assert any("items[0].quantity" in err.message for err in errors)