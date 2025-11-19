"""Input normalization helpers for the analyzer."""

from __future__ import annotations

from dataclasses import dataclass

_CHUNK_PROFILES = {"auto", "prose", "code", "notes"}
_CODE_EXTENSIONS = {".py", ".rs", ".ts", ".tsx", ".js", ".cs", ".cpp", ".c", ".java"}


def normalize_chunk_profile(profile: str | None) -> str | None:
    if not profile:
        return None
    normalized = profile.strip().lower()
    return normalized if normalized in _CHUNK_PROFILES else None


def infer_profile_from_path(path: str | None) -> str | None:
    if not path:
        return None
    lowered = path.lower()
    for ext in _CODE_EXTENSIONS:
        if lowered.endswith(ext):
            return "code"
    return None


@dataclass(slots=True)
class DocumentStats:
    """Lightweight struct shared with callers while collecting inputs."""

    char_count: int | None = None
    word_count: int | None = None
    line_count: int | None = None

    def size_bucket(self) -> str:
        chars = self.char_count or 0
        if chars >= 120_000:
            return "max"
        if chars >= 60_000:
            return "large"
        if chars >= 20_000:
            return "medium"
        return "small"
