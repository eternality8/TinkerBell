"""Search/replace helper tool with dry-run previews and diff summaries."""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping, Protocol, Sequence

from .diff_builder import DiffBuilderTool
from ...documents.ranges import TextRange


class SearchReplaceBridge(Protocol):
    """Subset of the document bridge used by the search/replace tool."""

    def generate_snapshot(self, *, delta_only: bool = False, **_: Any) -> Mapping[str, Any]:
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
    scope: Literal["document", "target_range"]
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
    diff_builder: DiffBuilderTool = field(default_factory=DiffBuilderTool)
    summarizable: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.default_max_replacements <= 0:
            raise ValueError("default_max_replacements must be positive")

    def run(
        self,
        pattern: str,
        replacement: str,
        *,
        is_regex: bool = False,
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None = None,
        dry_run: bool = False,
        max_replacements: int | None = None,
        match_case: bool = True,
        whole_word: bool = False,
    ) -> SearchReplaceResult:
        pattern = (pattern or "").strip()
        if not pattern:
            raise ValueError("pattern must not be empty")

        snapshot = self.bridge.generate_snapshot(delta_only=False)
        text = str(snapshot.get("text") or "")
        segment_start, segment_end = self._resolve_target_range(target_range, len(text))
        active_scope = "document" if target_range is None else "target_range"
        segment_text = text[segment_start:segment_end]
        range_hint = TextRange(segment_start, segment_end)

        regex = self._compile_pattern(pattern, is_regex=is_regex, match_case=match_case, whole_word=whole_word)
        count_limit = self._normalize_replacement_limit(max_replacements)
        updated_segment, replacements, limited, first_match = self._apply_replacements(
            regex,
            segment_text,
            replacement,
            limit=count_limit,
        )

        updated_document = text if replacements == 0 else text[:segment_start] + updated_segment + text[segment_end:]
        focus_index = self._compute_focus_index(first_match, segment_start)

        preview_source = updated_document if active_scope == "document" else updated_segment
        preview_focus = focus_index
        if preview_focus is not None and target_range is not None:
            preview_focus -= segment_start
        preview = self._build_preview(preview_source, preview_focus)

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
            document_version = self._resolve_document_version(snapshot, directive_version)
            content_hash = self._resolve_content_hash(snapshot, text)
            diff_text = self.diff_builder.run(
                text,
                updated_document,
                filename=self._resolve_document_label(snapshot),
                context=5,
            )
            patch_payload: dict[str, Any] = {
                "action": "patch",
                "diff": diff_text,
                "document_version": document_version,
                "content_hash": content_hash,
                "metadata": {
                    "matches": replacements,
                    "limited": limited,
                    "max_replacements": count_limit,
                    "scope": active_scope,
                    "target_range": range_hint.to_dict(),
                },
            }

            self.bridge.queue_edit(patch_payload)
            applied = True
            directive_version = getattr(self.bridge, "last_snapshot_version", None) or document_version

        return SearchReplaceResult(
            replacements=replacements,
            preview=preview,
            applied=applied,
            dry_run=dry_run,
            scope=active_scope,
            target_range=range_hint.to_tuple(),
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
    def _resolve_document_version(snapshot: Mapping[str, Any], fallback: str | None) -> str:
        candidate = snapshot.get("version") or fallback
        if not candidate:
            raise ValueError("Snapshot did not provide document_version for patch application")
        return str(candidate)

    @staticmethod
    def _resolve_content_hash(snapshot: Mapping[str, Any], text: str) -> str:
        token = snapshot.get("content_hash")
        if isinstance(token, str) and token.strip():
            return token.strip()
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _resolve_target_range(
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        length: int,
    ) -> tuple[int, int]:
        if target_range is None:
            return (0, length)
        if isinstance(target_range, Mapping):
            start = target_range.get("start")
            end = target_range.get("end")
        elif hasattr(target_range, "start") and hasattr(target_range, "end"):
            start = getattr(target_range, "start")
            end = getattr(target_range, "end")
        elif isinstance(target_range, Sequence) and len(target_range) == 2 and not isinstance(target_range, (str, bytes)):
            start, end = target_range
        else:
            raise ValueError("target_range must be a mapping or [start, end] sequence")
        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_range must provide numeric start/end offsets") from exc
        start_i = max(0, min(start_i, length))
        end_i = max(0, min(end_i, length))
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        return start_i, end_i

    @staticmethod
    def _compute_focus_index(match: re.Match[str] | None, offset: int) -> int | None:
        if match is None:
            return None
        return offset + match.start()

