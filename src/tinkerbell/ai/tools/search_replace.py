"""Search/replace helper tool with dry-run previews."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol


class SearchReplaceBridge(Protocol):
    """Subset of the document bridge used by the search/replace tool."""

    def generate_snapshot(self, *, delta_only: bool = False) -> Mapping[str, Any]:
        ...

    def queue_edit(self, directive: Mapping[str, Any]) -> None:
        ...

    @property
    def last_snapshot_version(self) -> str | None:
        ...


@dataclass(slots=True)
class SearchReplaceResult:
    """Outcome summary for a search/replace request."""

    replacements: int
    preview: str
    applied: bool
    dry_run: bool
    scope: Literal["document", "selection"]
    target_range: tuple[int, int]
    document_version: str | None = None


@dataclass(slots=True)
class SearchReplaceTool:
    """Perform scoped search/replace operations through the document bridge."""

    bridge: SearchReplaceBridge
    preview_chars: int = 240

    def run(
        self,
        pattern: str,
        replacement: str,
        *,
        is_regex: bool = False,
        scope: Literal["document", "selection"] = "document",
        dry_run: bool = False,
        max_replacements: int | None = None,
        match_case: bool = True,
        whole_word: bool = False,
    ) -> SearchReplaceResult:
        pattern = (pattern or "").strip()
        if not pattern:
            raise ValueError("pattern must not be empty")

        normalized_scope = scope.lower()
        if normalized_scope not in {"document", "selection"}:
            raise ValueError("scope must be either 'document' or 'selection'")

        snapshot = self.bridge.generate_snapshot(delta_only=False)
        text = str(snapshot.get("text") or "")
        selection = snapshot.get("selection") or (0, 0)
        selection_range = self._clamp_range(selection, len(text))

        use_selection = normalized_scope == "selection" and selection_range[0] != selection_range[1]
        segment_start, segment_end = selection_range if use_selection else (0, len(text))
        active_scope = "selection" if use_selection else "document"
        segment_text = text[segment_start:segment_end]

        regex = self._compile_pattern(pattern, is_regex=is_regex, match_case=match_case, whole_word=whole_word)
        first_match = regex.search(segment_text)
        count_limit = self._normalize_replacement_limit(max_replacements)
        updated_segment, replacements = regex.subn(replacement, segment_text, count=count_limit)

        updated_document = text[:segment_start] + updated_segment + text[segment_end:]
        focus_index = self._compute_focus_index(first_match, segment_start if active_scope == "document" else 0)

        preview_source = updated_document if active_scope == "document" else updated_segment
        preview = self._build_preview(preview_source, focus_index)

        directive_version = snapshot.get("version") or getattr(self.bridge, "last_snapshot_version", None)
        applied = False

        if replacements and not dry_run:
            payload: dict[str, Any] = {
                "action": "replace",
                "target_range": (segment_start, segment_end),
                "content": updated_segment if active_scope == "selection" else updated_document,
            }
            if directive_version:
                payload["document_version"] = directive_version

            self.bridge.queue_edit(payload)
            applied = True
            directive_version = getattr(self.bridge, "last_snapshot_version", None) or directive_version

        return SearchReplaceResult(
            replacements=replacements,
            preview=preview,
            applied=applied,
            dry_run=dry_run,
            scope=active_scope,
            target_range=(segment_start, segment_end),
            document_version=directive_version,
        )

    def _compile_pattern(
        self,
        pattern: str,
        *,
        is_regex: bool,
        match_case: bool,
        whole_word: bool,
    ) -> re.Pattern[str]:
        flags = re.MULTILINE
        if not match_case:
            flags |= re.IGNORECASE

        pattern_text = pattern if is_regex else re.escape(pattern)
        if whole_word:
            pattern_text = rf"\b{pattern_text}\b"

        try:
            return re.compile(pattern_text, flags)
        except re.error as exc:  # pragma: no cover - invalid regex path is already guarded in tests
            raise ValueError(f"Invalid regex pattern: {exc}") from exc

    @staticmethod
    def _normalize_replacement_limit(max_replacements: int | None) -> int:
        if max_replacements is None:
            return 0
        if max_replacements <= 0:
            raise ValueError("max_replacements must be a positive integer")
        return int(max_replacements)

    def _build_preview(self, text: str, focus_index: int | None) -> str:
        if not text:
            return ""
        focus = 0 if focus_index is None else max(0, min(focus_index, len(text)))
        start = max(0, focus - 40)
        end = min(len(text), start + self.preview_chars)
        return text[start:end]

    @staticmethod
    def _clamp_range(selection: tuple[int, int] | list[int], length: int) -> tuple[int, int]:
        try:
            start = int(selection[0])
            end = int(selection[1])
        except (TypeError, ValueError, IndexError):
            return (0, 0)

        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end < start:
            start, end = end, start
        return (start, end)

    @staticmethod
    def _compute_focus_index(match: re.Match[str] | None, offset: int) -> int | None:
        if match is None:
            return None
        return offset + match.start()

