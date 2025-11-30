"""Robust file IO helpers used throughout the editor stack."""

from __future__ import annotations

import codecs
import hashlib
import locale
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DocumentFormat",
    "FileSignature",
    "read_text",
    "write_text",
    "detect_format",
    "ensure_autosave_dir",
    "write_autosave",
    "snapshot_file",
    "file_has_changed",
    "compute_text_digest",
]

_BOM_MAP: dict[bytes, str] = {
    codecs.BOM_UTF8: "utf-8-sig",
    codecs.BOM_UTF16_LE: "utf-16-le",
    codecs.BOM_UTF16_BE: "utf-16-be",
    codecs.BOM_UTF32_LE: "utf-32-le",
    codecs.BOM_UTF32_BE: "utf-32-be",
}
_FORMAT_EXTENSIONS = {
    "markdown": {".md", ".markdown"},
    "yaml": {".yaml", ".yml"},
    "json": {".json"},
    "text": {".txt"},
}
_DEFAULT_AUTOSAVE_DIR = Path.home() / ".tinkerbell" / "autosave"


class DocumentFormat(Enum):
    """Detectable document formats supported by the editor."""

    MARKDOWN = "markdown"
    YAML = "yaml"
    JSON = "json"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class FileSignature:
    """Represents a file fingerprint for change detection."""

    path: Path
    digest: str
    size: int
    modified_at: float


@runtime_checkable
class SupportsAutosave(Protocol):
    """Protocol representing the minimal interface needed for autosave."""

    text: str

    @property
    def metadata(self) -> Any:  # pragma: no cover - Protocol placeholder
        """Return metadata containing an optional ``path`` attribute."""


def read_text(
    path: Path | str,
    *,
    encoding: str | None = None,
    errors: str = "strict",
    normalize_newlines: bool = True,
) -> str:
    """Read a text file with encoding detection and optional newline normalization."""

    target = Path(path)
    raw = target.read_bytes()
    detected_encoding = encoding or _detect_encoding(raw)
    text = raw.decode(detected_encoding, errors=errors)
    text = _strip_bom(text)
    return _normalize_newlines(text) if normalize_newlines else text


def write_text(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    newline: str = "\n",
    atomic: bool = True,
) -> Path:
    """Write text to disk using atomic semantics and configurable newline style."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = _apply_newline_policy(content, newline)
    if not atomic:
        with target.open("w", encoding=encoding, newline="") as handle:
            handle.write(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        return target

    descriptor, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):  # pragma: no cover - cleanup path
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    return target


def detect_format(path: Path | str | None = None, text: str | None = None) -> DocumentFormat:
    """Infer a document format using file suffix heuristics and contents."""

    suffix = Path(path).suffix.lower() if path else ""
    for name, extensions in _FORMAT_EXTENSIONS.items():
        if suffix in extensions:
            return DocumentFormat(name)

    if text is None:
        return DocumentFormat.UNKNOWN

    stripped = text.strip()
    if _looks_like_json(stripped):
        return DocumentFormat.JSON
    if _looks_like_yaml(stripped):
        return DocumentFormat.YAML
    if _looks_like_markdown(stripped):
        return DocumentFormat.MARKDOWN
    return DocumentFormat.TEXT if stripped else DocumentFormat.UNKNOWN


def ensure_autosave_dir(base_dir: Path | str | None = None) -> Path:
    """Ensure the autosave directory exists and return its path."""

    env_override = os.environ.get("TINKERBELL_AUTOSAVE_DIR")
    resolved = Path(base_dir or env_override or _DEFAULT_AUTOSAVE_DIR).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_autosave(
    document: SupportsAutosave,
    *,
    autosave_dir: Path | str | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Persist the provided document to the autosave directory."""

    autosave_root = ensure_autosave_dir(autosave_dir)
    source_name = _autosave_slug(_document_path(document))
    instant = timestamp or datetime.now(timezone.utc)
    filename = f"{source_name}-{instant:%Y%m%d%H%M%S%f}.autosave"
    target = autosave_root / filename
    write_text(target, document.text, atomic=True)
    return target


def snapshot_file(path: Path | str) -> FileSignature:
    """Compute a :class:`FileSignature` for the provided path."""

    target = Path(path)
    data = target.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    stat = target.stat()
    return FileSignature(path=target, digest=digest, size=stat.st_size, modified_at=stat.st_mtime)


def file_has_changed(signature: FileSignature) -> bool:
    """Return ``True`` if the file represented by ``signature`` has changed on disk."""

    try:
        stat = signature.path.stat()
    except FileNotFoundError:
        return True

    if stat.st_mtime != signature.modified_at or stat.st_size != signature.size:
        return True

    current_digest = hashlib.sha256(signature.path.read_bytes()).hexdigest()
    return current_digest != signature.digest


def compute_text_digest(text: str) -> str:
    """Return a SHA-256 digest for the provided text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_encoding(raw: bytes) -> str:
    for bom, encoding in _BOM_MAP.items():
        if raw.startswith(bom):
            return encoding

    preferred = locale.getpreferredencoding(False) or "utf-8"
    seen: set[str] = set()
    for candidate in ("utf-8", preferred, "latin-1"):
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            raw.decode(candidate)
            return candidate
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _normalize_newlines(text: str) -> str:
    if "\r" not in text:
        return text
    text = text.replace("\r\n", "\n")
    return text.replace("\r", "\n")


def _strip_bom(text: str) -> str:
    return text[1:] if text.startswith("\ufeff") else text


def _apply_newline_policy(content: str, newline: str) -> str:
    normalized = _normalize_newlines(content)
    if newline == "\n":
        return normalized
    if newline == "\r\n":
        return normalized.replace("\n", "\r\n")
    if newline == "\r":
        return normalized.replace("\n", "\r")
    raise ValueError(f"Unsupported newline policy: {newline!r}")


def _looks_like_json(text: str) -> bool:
    return bool(text) and text[0] in "[{"


def _looks_like_yaml(text: str) -> bool:
    if text.startswith("---"):
        return True
    return ":" in text.splitlines()[0] if text.splitlines() else False


def _looks_like_markdown(text: str) -> bool:
    if text.startswith("#") or text.startswith(("* ", "- ", ">")):
        return True
    return bool(re.search(r"`{1,3}", text))


def _document_path(document: SupportsAutosave | None) -> Path | None:
    if document is None:
        return None
    metadata = getattr(document, "metadata", None)
    path = getattr(metadata, "path", None)
    return Path(path) if path else None


def _autosave_slug(path: Path | None) -> str:
    if path is None:
        return "untitled"
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", path.name).strip("._")
    return slug or "untitled"

