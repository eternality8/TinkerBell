"""Tool returning the current document snapshot."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Mapping, Protocol, cast

from ..memory.chunk_index import ChunkIndex

LOGGER = logging.getLogger(__name__)


class SnapshotProvider(Protocol):
    """Protocol implemented by the document bridge for retrieving snapshots."""

    def generate_snapshot(
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
        window: Mapping[str, Any] | str | None = None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
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
    outline_digest_resolver: Callable[[str | None], str | None] | None = None
    chunk_index: ChunkIndex | None = None
    summarizable: ClassVar[bool] = True
    DEFAULT_WINDOW: ClassVar[dict[str, Any]] = {"kind": "selection", "padding": 2048, "max_chars": 8192}

    def run(
        self,
        *,
        delta_only: bool = False,
        include_diff: bool = True,
        tab_id: str | None = None,
        source_tab_ids: Iterable[str] | None = None,
        include_open_documents: bool = False,
        window: Mapping[str, Any] | str | None = None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
    ) -> dict:
        resolved_window = self._resolve_window(window)
        snapshot = self._build_snapshot(
            delta_only=delta_only,
            include_diff=include_diff,
            tab_id=tab_id,
            include_open_documents=include_open_documents,
            window=resolved_window,
            chunk_profile=chunk_profile,
            max_tokens=max_tokens,
            include_text=include_text,
        )

        digest = self._resolve_outline_digest(snapshot, tab_id)
        if digest:
            snapshot["outline_digest"] = digest

        extras = self._build_additional_snapshots(
            source_tab_ids,
            delta_only=delta_only,
            include_diff=include_diff,
            window=resolved_window,
            chunk_profile=chunk_profile,
            max_tokens=max_tokens,
            include_text=include_text,
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
        window: Mapping[str, Any] | str | None,
        chunk_profile: str | None,
        max_tokens: int | None,
        include_text: bool,
    ) -> dict:
        snapshot = deepcopy(
            dict(
                self._invoke_generate_snapshot(
                    delta_only=delta_only,
                    tab_id=tab_id,
                    include_open_documents=include_open_documents,
                    window=self._resolve_window(window),
                    chunk_profile=chunk_profile,
                    max_tokens=max_tokens,
                    include_text=include_text,
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

        self._ingest_chunk_manifest(snapshot)

        return snapshot

    def _build_additional_snapshots(
        self,
        tab_ids: Iterable[str] | None,
        *,
        delta_only: bool,
        include_diff: bool,
        window: Mapping[str, Any] | str | None,
        chunk_profile: str | None,
        max_tokens: int | None,
        include_text: bool,
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
                    window=window,
                    chunk_profile=chunk_profile,
                    max_tokens=max_tokens,
                    include_text=include_text,
                )
            )
        return snapshots

    def _ingest_chunk_manifest(self, snapshot: Mapping[str, Any]) -> None:
        if self.chunk_index is None:
            return
        manifest = snapshot.get("chunk_manifest")
        if not isinstance(manifest, Mapping):
            return
        try:
            self.chunk_index.ingest_manifest(manifest)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Chunk manifest ingestion failed", exc_info=True)

    def _invoke_generate_snapshot(
        self,
        *,
        delta_only: bool,
        tab_id: str | None,
        include_open_documents: bool,
        window: Mapping[str, Any] | str | None,
        chunk_profile: str | None,
        max_tokens: int | None,
        include_text: bool,
    ) -> Mapping[str, Any]:
        generate = getattr(self.provider, "generate_snapshot", None)
        if not callable(generate):  # pragma: no cover - defensive guard
            raise ValueError("Snapshot provider is missing generate_snapshot()")

        try:
            result = generate(
                delta_only=delta_only,
                tab_id=tab_id,
                include_open_documents=include_open_documents,
                window=window,
                chunk_profile=chunk_profile,
                max_tokens=max_tokens,
                include_text=include_text,
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

    def _resolve_outline_digest(self, snapshot: Mapping[str, Any], tab_id: str | None) -> str | None:
        resolver = self.outline_digest_resolver
        if not callable(resolver):
            return None
        raw_id = snapshot.get("document_id")
        if (raw_id is None or not str(raw_id).strip()) and tab_id:
            raw_id = tab_id
        document_id = str(raw_id).strip() if raw_id is not None else None
        return resolver(document_id or None)

    def _resolve_window(self, window: Mapping[str, Any] | str | None) -> Mapping[str, Any] | str | None:
        if window is None:
            return dict(self.DEFAULT_WINDOW)
        if isinstance(window, str):
            normalized = window.strip().lower()
            if not normalized or normalized in {"selection", "default"}:
                return dict(self.DEFAULT_WINDOW)
            return normalized
        if isinstance(window, Mapping):
            return dict(window)
        return window

