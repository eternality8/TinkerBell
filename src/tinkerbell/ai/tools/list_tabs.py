"""Lightweight tool returning summaries of all open tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Protocol, Sequence


class TabListingProvider(Protocol):
    """Protocol describing the workspace facade required to enumerate tabs."""

    def list_tabs(self) -> Sequence[Mapping[str, Any]]:
        ...

    def active_tab_id(self) -> str | None:
        ...


@dataclass(slots=True)
class ListTabsTool:
    """Return sanitized metadata for every open editor tab."""

    provider: TabListingProvider
    summarizable: ClassVar[bool] = True

    def run(self) -> dict[str, Any]:
        raw_tabs = self.provider.list_tabs()
        tabs = [self._normalize(entry, index=idx) for idx, entry in enumerate(raw_tabs, start=1)]
        return {
            "tabs": tabs,
            "active_tab_id": self.provider.active_tab_id(),
            "total": len(tabs),
        }

    def _normalize(self, entry: Mapping[str, Any], *, index: int | None = None) -> dict[str, Any]:
        tab_id = str(entry.get("tab_id") or entry.get("id") or "").strip()
        title = str(entry.get("title") or "Untitled").strip()
        path = entry.get("path")
        if path is not None:
            path = str(path)
        payload = {
            "tab_id": tab_id,
            "title": title,
            "path": path,
            "dirty": bool(entry.get("dirty", False)),
            "language": entry.get("language"),
            "untitled_index": entry.get("untitled_index"),
        }
        if index is not None:
            payload["tab_number"] = index
            payload["label"] = f"Tab {index}: {title}"
        return payload
