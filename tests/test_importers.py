"""Unit tests for the file import registry and handlers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tinkerbell.services.importers import (
    FileImporter,
    ImportResult,
    ImporterError,
    PDFImportHandler,
)


class _DummyHandler:
    name = "dummy"
    extensions = (".foo",)

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".foo"

    def import_file(self, path: Path) -> ImportResult:
        self.calls.append(path)
        return ImportResult(text="Converted", title="Converted Foo", language="text")


class _StubPdfReader:
    def __init__(self, _path: str) -> None:
        self.pages = [
            SimpleNamespace(extract_text=lambda: "First page"),
            SimpleNamespace(extract_text=lambda: ""),
            SimpleNamespace(extract_text=lambda: "Last page"),
        ]


def test_file_importer_selects_matching_handler(tmp_path: Path) -> None:
    handler = _DummyHandler()
    importer = FileImporter(handlers=(handler,))
    target = tmp_path / "note.foo"
    target.write_text("seed", encoding="utf-8")

    result = importer.import_file(target)

    assert result.text == "Converted"
    assert handler.calls == [target]


def test_file_importer_raises_for_unknown_extension(tmp_path: Path) -> None:
    importer = FileImporter(handlers=())
    target = tmp_path / "doc.unknown"
    target.write_text("content", encoding="utf-8")

    with pytest.raises(ImporterError):
        importer.import_file(target)


def test_pdf_import_handler_extracts_text(tmp_path: Path) -> None:
    handler = PDFImportHandler(reader_cls=_StubPdfReader)
    target = tmp_path / "sample.pdf"
    target.write_text("binary", encoding="utf-8")

    result = handler.import_file(target)

    assert "First page" in result.text
    assert "Last page" in result.text
    assert result.title == "sample"