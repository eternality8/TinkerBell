"""Tests covering the utilities modules."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tinkerbell.editor.document_model import DocumentState
from tinkerbell.utils import file_io, logging as logging_utils, telemetry


def test_read_text_detects_bom_and_normalizes_newlines(tmp_path: Path) -> None:
    target = tmp_path / "utf16.txt"
    target.write_bytes("Line1\r\nLine2".encode("utf-16"))

    result = file_io.read_text(target)

    assert result == "Line1\nLine2"


def test_write_text_enforces_newline_policy(tmp_path: Path) -> None:
    target = tmp_path / "output.txt"

    returned = file_io.write_text(target, "Line1\nLine2", newline="\r\n")

    assert returned == target
    assert target.read_bytes() == b"Line1\r\nLine2"


def test_detect_format_uses_suffix_and_content() -> None:
    assert file_io.detect_format(path=Path("note.md")) is file_io.DocumentFormat.MARKDOWN
    assert file_io.detect_format(text="{\"a\": 1}") is file_io.DocumentFormat.JSON
    assert file_io.detect_format(text="---\nfoo: bar\n") is file_io.DocumentFormat.YAML
    assert file_io.detect_format(text="# Heading") is file_io.DocumentFormat.MARKDOWN


def test_ensure_autosave_dir_respects_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    autosave_dir = tmp_path / "autosave"
    monkeypatch.setenv("TINKERBELL_AUTOSAVE_DIR", str(autosave_dir))

    resolved = file_io.ensure_autosave_dir()

    assert resolved == autosave_dir
    assert resolved.exists()


def test_write_autosave_uses_document_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    autosave_dir = tmp_path / "autosave"
    monkeypatch.setenv("TINKERBELL_AUTOSAVE_DIR", str(autosave_dir))
    document = DocumentState(text="Autosave body")
    document.metadata.path = tmp_path / "My Notes.md"

    autosave_path = file_io.write_autosave(document)

    assert autosave_path.parent == autosave_dir
    assert autosave_path.exists()
    assert autosave_path.read_text(encoding="utf-8") == "Autosave body"
    assert autosave_path.name.startswith("My_Notes.md-")


def test_file_signature_detects_changes(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("alpha", encoding="utf-8")
    signature = file_io.snapshot_file(target)

    assert not file_io.file_has_changed(signature)

    target.write_text("beta", encoding="utf-8")

    assert file_io.file_has_changed(signature)


def test_compute_text_digest_changes_with_content() -> None:
    digest_one = file_io.compute_text_digest("hello")
    digest_two = file_io.compute_text_digest("world")

    assert digest_one != digest_two


def test_setup_logging_creates_rotating_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    log_path = logging_utils.setup_logging(
        level=logging.INFO,
        log_dir=log_dir,
        console=False,
        force=True,
    )

    logger = logging_utils.get_logger("tinkerbell.tests")
    logger.info("Logging smoke test")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    assert "Logging smoke test" in contents


def test_telemetry_client_flushes_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    storage_dir = tmp_path / "telemetry"
    client = telemetry.TelemetryClient(enabled=True, storage_dir=storage_dir)

    client.track_event("launch", feature="autosave")
    output_path = client.flush()

    assert output_path is not None
    body = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(body) == 1
    event = json.loads(body[0])
    assert event["name"] == "launch"
    assert event["properties"]["feature"] == "autosave"


def test_telemetry_enabled_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINKERBELL_TELEMETRY", "true")

    assert telemetry.telemetry_enabled(settings=None)
