"""Document guardrail helpers for outline/retrieval tools."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from ...editor.document_model import DocumentState
from ...utils.file_io import DocumentFormat, detect_format

HUGE_DOCUMENT_BYTES = 5 * 1024 * 1024
_BINARY_EXTENSION_HINTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".psd",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".doc",
    ".docx",
    ".exe",
    ".dll",
    ".bin",
    ".mp3",
    ".mp4",
}
_PRINTABLE_EXTRA = {"\n", "\r", "\t", "\f", "\v"}
_SUPPORTED_FORMATS: set[DocumentFormat] = {
    DocumentFormat.MARKDOWN,
    DocumentFormat.YAML,
    DocumentFormat.JSON,
    DocumentFormat.TEXT,
}

__all__ = [
    "HUGE_DOCUMENT_BYTES",
    "document_size_bytes",
    "is_huge_document",
    "unsupported_format_reason",
    "is_probably_binary_text",
]

def document_size_bytes(document: DocumentState) -> int:
    text = document.text or ""
    return len(text.encode("utf-8"))

def is_huge_document(document: DocumentState, threshold: int = HUGE_DOCUMENT_BYTES) -> bool:
    return document_size_bytes(document) > threshold

def unsupported_format_reason(document: DocumentState) -> str | None:
    path = getattr(document.metadata, "path", None)
    suffix = Path(path).suffix.lower() if path else ""
    if suffix in _BINARY_EXTENSION_HINTS:
        return f"binary_extension:{suffix or 'unknown'}"
    text = document.text or ""
    if is_probably_binary_text(text):
        return "binary_content"
    fmt = detect_format(path, text)
    if fmt == DocumentFormat.UNKNOWN and not text.strip():
        fmt = DocumentFormat.TEXT
    if fmt not in _SUPPORTED_FORMATS:
        if path:
            mime, _ = mimetypes.guess_type(str(path))
            if mime:
                return f"unsupported_mime:{mime}"
        return f"unsupported_format:{fmt.value}"
    return None

def is_probably_binary_text(text: str, *, sample_size: int = 4096) -> bool:
    if not text:
        return False
    sample = text[:sample_size]
    if "\x00" in sample:
        return True
    non_printable = sum(1 for ch in sample if not _is_printable(ch))
    return non_printable / max(1, len(sample)) >= 0.3

def _is_printable(ch: str) -> bool:
    if ch in _PRINTABLE_EXTRA:
        return True
    code = ord(ch)
    return 32 <= code <= 126
