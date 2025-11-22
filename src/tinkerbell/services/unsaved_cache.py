"""Persistence helpers for unsaved editor snapshots."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .settings import _SETTINGS_DIR

__all__ = ["UnsavedCache", "UnsavedCacheStore"]

LOGGER = logging.getLogger(__name__)
_CACHE_FILENAME = "unsaved_cache.json"
_CACHE_VERSION = 1


def _default_cache_path() -> Path:
    return _SETTINGS_DIR / _CACHE_FILENAME


@dataclass(slots=True)
class UnsavedCache:
    """Transient editor state kept separate from user settings."""

    unsaved_snapshot: dict[str, Any] | None = None
    unsaved_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    untitled_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)


class UnsavedCacheStore:
    """Persistence adapter for :class:`UnsavedCache`."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _default_cache_path()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> UnsavedCache:
        payload = self._read_payload()
        snapshot = _coerce_snapshot(payload.get("unsaved_snapshot"))
        snapshots = _coerce_snapshot_mapping(payload.get("unsaved_snapshots"))
        untitled = _coerce_snapshot_mapping(payload.get("untitled_snapshots"))
        return UnsavedCache(
            unsaved_snapshot=snapshot,
            unsaved_snapshots=snapshots,
            untitled_snapshots=untitled,
        )

    def save(self, cache: UnsavedCache | None) -> Path:
        if cache is None:
            return self._path
        payload = {
            "version": _CACHE_VERSION,
            "unsaved_snapshot": cache.unsaved_snapshot,
            "unsaved_snapshots": cache.unsaved_snapshots,
            "untitled_snapshots": cache.untitled_snapshots,
        }
        body = json.dumps(payload, indent=2, sort_keys=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(body, encoding="utf-8")
        tmp_path.replace(self._path)
        return self._path

    def _read_payload(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, Mapping):
                return dict(data)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            LOGGER.warning("Unsaved cache %s is not valid JSON: %s", self._path, exc)
        return {}


def _coerce_snapshot(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _coerce_snapshot_mapping(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for key, entry in value.items():
        if not isinstance(key, str):
            continue
        snapshot = _coerce_snapshot(entry)
        if snapshot is None:
            continue
        result[key] = snapshot
    return result
