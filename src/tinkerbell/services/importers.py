"""File import helpers that convert non-native formats into editable text."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Protocol, Sequence

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency resolved at runtime
    from pypdf import PdfReader as _PdfReader
except Exception:  # pragma: no cover - dependency may be missing in tests
    _PdfReader = None  # type: ignore[assignment]


class ImporterError(RuntimeError):
    """Raised when a file import operation fails."""


def _normalize_extension(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext:
        return ext
    if not ext.startswith("."):
        return f".{ext}"
    return ext


@dataclass(slots=True)
class ImportResult:
    """Outcome returned by a file import handler."""

    text: str
    title: str | None = None
    language: str = "text"
    notes: str | None = None


class ImportHandler(Protocol):
    """Protocol implemented by concrete import handlers."""

    name: str
    extensions: tuple[str, ...]

    def supports(self, path: Path) -> bool:
        """Return True if the handler can process the provided path."""
        ...

    def import_file(self, path: Path) -> ImportResult:
        """Convert the file into editable text."""
        ...


class FileImporter:
    """Registry-driven facade for converting external file formats."""

    def __init__(self, handlers: Sequence[ImportHandler] | None = None) -> None:
        registry = list(handlers or [])
        if not registry:
            registry.append(PDFImportHandler())
        self._handlers: list[ImportHandler] = registry

    def register_handler(self, handler: ImportHandler) -> None:
        if handler not in self._handlers:
            self._handlers.append(handler)

    def handlers(self) -> tuple[ImportHandler, ...]:
        return tuple(self._handlers)

    def supported_extensions(self) -> tuple[str, ...]:
        seen: list[str] = []
        for handler in self._handlers:
            for extension in handler.extensions:
                normalized = _normalize_extension(extension)
                if normalized and normalized not in seen:
                    seen.append(normalized)
        return tuple(seen)

    def dialog_filter(self) -> str:
        extensions = self.supported_extensions()
        if not extensions:
            return "All Files (*)"
        patterns = " ".join(f"*{ext}" for ext in extensions)
        return f"Supported Imports ({patterns});;All Files (*)"

    def import_file(self, path: Path | str) -> ImportResult:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)
        handler = self._select_handler(target)
        if handler is None:
            raise ImporterError(f"No import handler registered for '{target.suffix or target}'.")
        return handler.import_file(target)

    def _select_handler(self, path: Path) -> ImportHandler | None:
        for handler in self._handlers:
            try:
                if handler.supports(path):
                    return handler
            except Exception as exc:  # pragma: no cover - handler bugs are logged
                _LOGGER.debug("Import handler %s failed during supports(): %s", handler, exc)
        return None


class PDFImportHandler:
    """Convert PDF files into plain text using pypdf."""

    name: str = "pdf"
    extensions: tuple[str, ...] = (".pdf",)

    def __init__(self, *, reader_cls: type | None = None) -> None:
        self._reader_cls = reader_cls or _PdfReader

    def supports(self, path: Path) -> bool:  # pragma: no cover - trivial logic
        return path.suffix.lower() in self.extensions

    def import_file(self, path: Path) -> ImportResult:
        reader_cls = self._reader_cls
        if reader_cls is None:  # pragma: no cover - dependency guard
            raise ImporterError("PDF support requires the 'pypdf' dependency.")

        try:
            reader = reader_cls(str(path))
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            raise ImporterError(f"Unable to open PDF: {exc}") from exc

        text_chunks: list[str] = []
        pages = getattr(reader, "pages", [])
        for index, page in enumerate(pages):
            extractor = getattr(page, "extract_text", None)
            if not callable(extractor):
                _LOGGER.debug("PDF page %s missing extract_text; skipping.", index)
                continue
            try:
                chunk = str(extractor() or "")
            except Exception as exc:  # pragma: no cover - defensive logging
                _LOGGER.debug("Failed to extract page %s: %s", index, exc)
                continue
            chunk = chunk.strip()
            if chunk:
                text_chunks.append(chunk)

        text = "\n\n".join(text_chunks).strip()
        if not text:
            text = "(No extractable text found in PDF.)"
        return ImportResult(
            text=text,
            title=path.stem,
            language="text",
            notes="Imported from PDF",
        )