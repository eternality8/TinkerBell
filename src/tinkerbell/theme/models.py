"""Data structures describing TinkerBell application themes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

ColorTuple = Tuple[int, int, int]
PaletteLike = Mapping[str, Any] | Sequence[tuple[str, Any]]


def _clamp_channel(value: Any) -> int:
    channel = int(value)
    if channel < 0:
        return 0
    if channel > 255:
        return 255
    return channel


def normalize_color(value: Any) -> ColorTuple:
    """Convert ``value`` into an RGB tuple, accepting hex strings or sequences."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Color strings cannot be empty")
        if text.startswith("#"):
            text = text[1:]
        if "," in text:
            parts = [part.strip() for part in text.split(",") if part.strip()]
            if len(parts) != 3:
                raise ValueError(f"Color '{value}' must have exactly 3 components")
            return tuple(_clamp_channel(int(part, 0)) for part in parts)  # type: ignore[return-value]
        if len(text) in (3, 6):
            if len(text) == 3:
                text = "".join(ch * 2 for ch in text)
            return tuple(int(text[i : i + 2], 16) for i in range(0, 6, 2))  # type: ignore[return-value]
        raise ValueError(f"Unsupported color format: {value!r}")

    if isinstance(value, Sequence):
        items = list(value)
        if len(items) != 3:
            raise ValueError(f"RGB sequences must contain 3 values, received {value!r}")
        return tuple(_clamp_channel(component) for component in items)  # type: ignore[return-value]

    raise TypeError(f"Cannot convert {type(value)!r} to an RGB color")


def _normalize_palette(palette: PaletteLike | None) -> Dict[str, ColorTuple]:
    normalized: Dict[str, ColorTuple] = {}
    if palette is None:
        return normalized
    items: Sequence[tuple[str, Any]]
    if isinstance(palette, Mapping):
        items = list(palette.items())
    else:
        items = list(palette)
    for key, value in items:
        if key is None:
            continue
        normalized[key.strip().lower()] = normalize_color(value)
    return normalized


def _tuple_to_hex(value: ColorTuple) -> str:
    return "#" + "".join(f"{component:02x}" for component in value)


@dataclass(slots=True)
class Theme:
    """Serializable structure describing a UI theme."""

    name: str
    title: str
    palette: Dict[str, ColorTuple] = field(default_factory=dict)
    description: str | None = None
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = (self.name or "default").strip().lower() or "default"
        self.title = (self.title or self.name.title()).strip()
        self.palette = _normalize_palette(self.palette)
        self.metadata = dict(self.metadata or {})

    def color(self, key: str, fallback: ColorTuple | None = None) -> ColorTuple:
        lookup = key.strip().lower()
        if lookup in self.palette:
            return self.palette[lookup]
        if fallback is not None:
            return fallback
        if not self.palette:
            raise KeyError(f"Theme '{self.name}' has no palette entries")
        # Deterministic fallback by returning first color
        first_key = next(iter(self.palette))
        return self.palette[first_key]

    def as_css_hex(self, key: str, fallback: str | None = None) -> str:
        try:
            value = self.color(key)
        except KeyError:
            if fallback is None:
                raise
            return fallback
        return _tuple_to_hex(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "metadata": dict(self.metadata),
            "palette": {key: list(value) for key, value in self.palette.items()},
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Theme":
        if "name" not in payload:
            raise ValueError("Theme payload missing 'name'")
        palette = payload.get("palette")
        description = payload.get("description")
        return cls(
            name=str(payload["name"]),
            title=str(payload.get("title") or payload["name"]),
            palette=palette or {},
            description=str(description) if description is not None else None,
            version=str(payload.get("version") or "1.0.0"),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "Theme":
        data = json.loads(text)
        if not isinstance(data, Mapping):  # pragma: no cover - defensive guard
            raise ValueError("Theme JSON root must be an object")
        return cls.from_dict(data)


__all__ = ["ColorTuple", "Theme", "normalize_color"]
