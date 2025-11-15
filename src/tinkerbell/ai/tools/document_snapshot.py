"""Tool returning the current document snapshot."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, cast


class SnapshotProvider(Protocol):
    """Protocol implemented by the document bridge for retrieving snapshots."""

    def generate_snapshot(
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
    ) -> Mapping[str, Any]:
        ...

    def get_last_diff_summary(self, tab_id: str | None = None) -> str | None:
        ...

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:
        ...


@dataclass(slots=True)
class DocumentSnapshotTool:
    """Simple synchronous tool returning document snapshots."""

    provider: SnapshotProvider

    def run(
        self,
        *,
        delta_only: bool = False,
        include_diff: bool = True,
        tab_id: str | None = None,
        source_tab_ids: Iterable[str] | None = None,
        include_open_documents: bool = False,
    ) -> dict:
        snapshot = self._build_snapshot(
            delta_only=delta_only,
            include_diff=include_diff,
            tab_id=tab_id,
            include_open_documents=include_open_documents,
        )

        extras = self._build_additional_snapshots(
            source_tab_ids,
            delta_only=delta_only,
            include_diff=include_diff,
        )
        if extras:
            snapshot["source_tab_snapshots"] = extras

        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_snapshot(
        self,
        *,
        delta_only: bool,
        include_diff: bool,
        tab_id: str | None,
        include_open_documents: bool,
    ) -> dict:
        snapshot = deepcopy(
            dict(
                self._invoke_generate_snapshot(
                    delta_only=delta_only,
                    tab_id=tab_id,
                    include_open_documents=include_open_documents,
                )
            )
        )

        if include_diff:
            diff = self._last_diff(tab_id)
            if diff is not None:
                snapshot["diff_summary"] = diff

        version = snapshot.get("version") or self._last_version(tab_id)
        if version is not None:
            snapshot["version"] = version

        return snapshot

    def _build_additional_snapshots(
        self,
        tab_ids: Iterable[str] | None,
        *,
        delta_only: bool,
        include_diff: bool,
    ) -> list[dict]:
        if not tab_ids:
            return []
        if isinstance(tab_ids, (str, bytes)):
            iterable: Iterable[str | bytes] = [tab_ids]
        else:
            iterable = tab_ids
        snapshots: list[dict] = []
        for source_id in iterable:
            candidate = str(source_id).strip()
            if not candidate:
                continue
            snapshots.append(
                self._build_snapshot(
                    delta_only=delta_only,
                    include_diff=include_diff,
                    tab_id=candidate,
                    include_open_documents=False,
                )
            )
        return snapshots

    def _invoke_generate_snapshot(
        self,
        *,
        delta_only: bool,
        tab_id: str | None,
        include_open_documents: bool,
    ) -> Mapping[str, Any]:
        generate = getattr(self.provider, "generate_snapshot", None)
        if not callable(generate):  # pragma: no cover - defensive guard
            raise ValueError("Snapshot provider is missing generate_snapshot()")

        try:
            result = generate(
                delta_only=delta_only,
                tab_id=tab_id,
                include_open_documents=include_open_documents,
            )
        except TypeError:
            if tab_id is not None or include_open_documents:
                raise ValueError("Snapshot provider does not support tab_id or open document metadata")
            result = generate(delta_only=delta_only)
        return cast(Mapping[str, Any], result)

    def _last_diff(self, tab_id: str | None) -> str | None:
        getter = getattr(self.provider, "get_last_diff_summary", None)
        if callable(getter):
            return cast(str | None, getter(tab_id=tab_id))
        return cast(str | None, getattr(self.provider, "last_diff_summary", None))

    def _last_version(self, tab_id: str | None) -> str | None:
        getter = getattr(self.provider, "get_last_snapshot_version", None)
        if callable(getter):
            return cast(str | None, getter(tab_id=tab_id))
        return cast(str | None, getattr(self.provider, "last_snapshot_version", None))

