"""Search/replace helper tool with dry-run previews and diff summaries."""

from __future__ import annotations

import difflib
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
    max_replacements: int | None = None
    limited: bool = False
    diff_preview: str | None = None


@dataclass(slots=True)
class SearchReplaceTool:
    """Perform scoped search/replace operations through the document bridge."""

    bridge: SearchReplaceBridge
    preview_chars: int = 240
    default_max_replacements: int = 200

    def __post_init__(self) -> None:
        if self.default_max_replacements <= 0:
            raise ValueError("default_max_replacements must be positive")

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
        count_limit = self._normalize_replacement_limit(max_replacements)
        updated_segment, replacements, limited, first_match = self._apply_replacements(
            regex,
            segment_text,
            replacement,
            limit=count_limit,
        )

        updated_document = text if replacements == 0 else text[:segment_start] + updated_segment + text[segment_end:]
        focus_index = self._compute_focus_index(first_match, segment_start if active_scope == "document" else 0)

        preview_source = updated_document if active_scope == "document" else updated_segment
        preview = self._build_preview(preview_source, focus_index)

        diff_preview = None
        if replacements:
            diff_preview = self._build_diff_preview(
                text if active_scope == "document" else segment_text,
                updated_document if active_scope == "document" else updated_segment,
                document_label=self._resolve_document_label(snapshot),
            )

        directive_version = snapshot.get("version") or getattr(self.bridge, "last_snapshot_version", None)
        applied = False

        if replacements and not dry_run:
            payload: dict[str, Any] = {
                "action": "replace",
                "target_range": (segment_start, segment_end),
                "content": updated_segment if active_scope == "selection" else updated_document,
                "metadata": {
                    "matches": replacements,
                    "limited": limited,
                    "max_replacements": count_limit,
                },
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
            max_replacements=count_limit,
            limited=limited,
            diff_preview=diff_preview,
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

    def _normalize_replacement_limit(self, max_replacements: int | None) -> int:
        if max_replacements is None:
            return self.default_max_replacements
        if max_replacements <= 0:
            raise ValueError("max_replacements must be a positive integer")
        return int(max_replacements)

    def _apply_replacements(
        self,
        regex: re.Pattern[str],
        text: str,
        replacement: str,
        *,
        limit: int,
    ) -> tuple[str, int, bool, re.Match[str] | None]:
        cursor = 0
        chunks: list[str] = []
        replacements = 0
        limited = False
        first_match: re.Match[str] | None = None

        for match in regex.finditer(text):
            if replacements >= limit:
                limited = True
                break

            if first_match is None:
                first_match = match

            start, end = match.span()
            chunks.append(text[cursor:start])
            chunks.append(replacement)
            cursor = end
            replacements += 1

        if replacements == 0:
            return text, 0, False, None

        chunks.append(text[cursor:])
        return ("".join(chunks), replacements, limited, first_match)

    def _build_diff_preview(
        self,
        original: str,
        updated: str,
        *,
        document_label: str,
        context_lines: int = 3,
        max_lines: int = 200,
    ) -> str:
        diff_iter = difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile=f"{document_label}:before",
            tofile=f"{document_label}:after",
            n=context_lines,
            lineterm="",
        )
        lines: list[str] = []
        for line in diff_iter:
            lines.append(line)
            if len(lines) >= max_lines:
                lines.append("... (diff truncated)")
                break
        return "\n".join(lines)

    @staticmethod
    def _resolve_document_label(snapshot: Mapping[str, Any]) -> str:
        path = snapshot.get("path")
        if path:
            return str(path)
        document_id = snapshot.get("document_id") or "document"
        return str(document_id)

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

