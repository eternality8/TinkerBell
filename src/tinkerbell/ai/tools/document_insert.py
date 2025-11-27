"""Tool that inserts text at a specific line without overwriting existing content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Protocol, Sequence

from .document_apply_patch import DocumentApplyPatchTool
from .validation import parse_snapshot_token
from ...documents.ranges import LineRange
from ...services.telemetry import emit as telemetry_emit


class InsertBridge(Protocol):
    """Protocol describing the bridge required by DocumentInsertTool."""

    def generate_snapshot(
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_text: bool = True,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        ...


@dataclass(slots=True)
class DocumentInsertTool:
    """Insert text at a specific line position without overwriting existing content.

    This tool provides a safe API for text insertion that cannot accidentally
    overwrite existing paragraphs or content. Use this when you want to add
    new content between existing lines.

    For replacing existing content, use document_apply_patch instead.
    """

    patch_tool: DocumentApplyPatchTool
    insert_event: str = "document_insert"
    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        snapshot_token: str | None = None,
        after_line: int,
        content: str,
        rationale: str | None = None,
        tab_id: str | None = None,
    ) -> str:
        """Insert text after a specific line number.

        Args:
            snapshot_token: Compact version identifier in format 'tab_id:version_id'.
            after_line: The 0-based line number after which to insert content.
                       Use -1 to insert at the very beginning of the document.
            content: The text to insert. Will have a newline appended if not present.
            rationale: Optional explanation stored alongside the edit directive.
            tab_id: Optional tab identifier; defaults to value from snapshot_token.

        Returns:
            Status message indicating success or failure.

        Raises:
            InvalidSnapshotTokenError: If snapshot_token is malformed.
            ValueError: If required fields are missing or invalid.
        """
        # Parse snapshot_token to extract version info
        parsed_tab_id, parsed_version_id = parse_snapshot_token(snapshot_token, strict=True)

        if parsed_tab_id is not None and tab_id is None:
            tab_id = parsed_tab_id

        # Get the current snapshot to retrieve document text and version fields
        bridge = self.patch_tool.bridge
        try:
            snapshot = dict(bridge.generate_snapshot(
                delta_only=False,
                tab_id=tab_id,
                include_text=True,
            ))
        except Exception as exc:
            error_msg = str(exc)
            # Check if this looks like the AI used document_id instead of tab_id
            if "Unknown tab_id" in error_msg and parsed_tab_id:
                raise ValueError(
                    f"Invalid snapshot_token: '{parsed_tab_id}' is not a valid tab_id. "
                    "You may have used 'document_id' instead of 'tab_id'. "
                    "Use the 'snapshot_token' field directly from document_snapshot response, "
                    "do not construct it manually."
                ) from exc
            raise

        # Additional validation: check if the parsed tab_id matches the snapshot's tab_id
        snapshot_tab_id = snapshot.get("tab_id")
        if snapshot_tab_id and parsed_tab_id and parsed_tab_id != snapshot_tab_id:
            # The AI might have used document_id instead of tab_id
            document_id = snapshot.get("document_id")
            if parsed_tab_id == document_id:
                raise ValueError(
                    f"Invalid snapshot_token: You used 'document_id' ({document_id}) instead of 'tab_id' ({snapshot_tab_id}). "
                    "Copy the 'snapshot_token' field directly from document_snapshot response."
                )

        document_text = snapshot.get("text", "")
        if not isinstance(document_text, str):
            raise ValueError("Snapshot did not provide document text")

        # Build line offsets to convert line number to character offset
        offsets = self._resolve_line_start_offsets(snapshot, document_text)

        # Validate line number
        max_line = len(offsets) - 2  # -2 because offsets includes start (0) and end
        if max_line < 0:
            max_line = 0

        if after_line < -1:
            raise ValueError(f"after_line must be >= -1, got {after_line}")
        if after_line > max_line:
            raise ValueError(
                f"after_line {after_line} exceeds document line count "
                f"(max line index: {max_line}). Refresh snapshot and retry."
            )

        # Calculate insertion point
        if after_line == -1:
            # Insert at the very beginning
            insert_offset = 0
            insert_line = 0
        else:
            # Insert after the specified line (at the start of the next line)
            insert_line = after_line + 1
            if insert_line >= len(offsets) - 1:
                # Inserting at the end of document
                insert_offset = len(document_text)
            else:
                insert_offset = offsets[insert_line]

        # Ensure content ends with a newline for proper line separation
        insert_content = content
        if insert_content and not insert_content.endswith("\n"):
            # Check if we're inserting at end of document that doesn't end with newline
            if insert_offset == len(document_text) and document_text and not document_text.endswith("\n"):
                insert_content = "\n" + insert_content
            elif insert_offset < len(document_text):
                insert_content = insert_content + "\n"

        # Extract version fields from snapshot
        document_version = snapshot.get("version")
        version_id = parsed_version_id or snapshot.get("version_id")
        content_hash = snapshot.get("content_hash")

        if not document_version:
            raise ValueError("Could not retrieve document_version from snapshot")
        if version_id is None:
            raise ValueError("Could not retrieve version_id from snapshot")
        if not content_hash:
            raise ValueError("Could not retrieve content_hash from snapshot")

        # Emit telemetry for the insert operation
        telemetry_emit(
            self.insert_event,
            {
                "tab_id": tab_id,
                "snapshot_token": snapshot_token,
                "after_line": after_line,
                "insert_line": insert_line,
                "insert_offset": insert_offset,
                "content_length": len(insert_content),
                "document_length": len(document_text),
            },
        )

        # Create a zero-width span at the insertion point
        # We use a LineRange where start_line == end_line and the content
        # at that position is empty (zero-width replacement = insertion)
        target_span = LineRange(insert_line, insert_line - 1) if insert_line > 0 else LineRange(0, -1)

        # For the patch tool, we need to provide an actual range that results
        # in an insertion. We'll use target_range with identical start/end offsets.
        # The patch tool will then insert without replacing any content.

        # Delegate to DocumentApplyPatchTool
        # We bypass the line span conversion by using match_text to anchor
        # and providing the exact insertion point
        result = self._apply_insert(
            document_text=document_text,
            insert_offset=insert_offset,
            insert_content=insert_content,
            document_version=str(document_version),
            version_id=version_id,
            content_hash=content_hash,
            rationale=rationale,
            tab_id=tab_id,
            snapshot_token=snapshot_token,
            after_line=after_line,
        )

        return result

    def _apply_insert(
        self,
        *,
        document_text: str,
        insert_offset: int,
        insert_content: str,
        document_version: str,
        version_id: str | int,
        content_hash: str,
        rationale: str | None,
        tab_id: str | None,
        snapshot_token: str | None,
        after_line: int,
    ) -> str:
        """Apply the insertion using the patch tool's underlying edit mechanism."""
        # Build the new document content
        new_text = document_text[:insert_offset] + insert_content + document_text[insert_offset:]

        # Use the diff builder directly to create a proper diff
        diff = self.patch_tool.diff_builder.run(
            document_text,
            new_text,
            filename="document.txt",
            context=self.patch_tool.default_context_lines,
        )

        # Build scope details (required by edit tool)
        scope_details: dict[str, Any] = {
            "origin": "insert",
            "range": {"start": insert_offset, "end": insert_offset},
            "length": 0,  # Zero-width insertion
        }

        # Build the range entry for the edit tool
        range_entry: dict[str, Any] = {
            "start": insert_offset,
            "end": insert_offset,  # Zero-width range = insertion
            "replacement": insert_content,
            "match_text": "",  # Empty match text for insertion
            "scope": scope_details,
            "scope_origin": "insert",
            "scope_length": 0,
            "scope_range": {"start": insert_offset, "end": insert_offset},
        }

        payload: dict[str, Any] = {
            "action": "patch",
            "diff": diff,
            "document_version": document_version,
            "content_hash": content_hash,
            "ranges": [range_entry],
            "metadata": {
                "operation": "insert",
                "after_line": after_line,
                "insert_offset": insert_offset,
                "scope": scope_details,
                "scope_origin": "insert",
                "scope_length": 0,
                "scope_range": {"start": insert_offset, "end": insert_offset},
            },
        }
        if rationale is not None:
            payload["rationale"] = rationale

        # Call the edit tool directly
        status = self.patch_tool.edit_tool.run(tab_id=tab_id, **payload)

        # Enhance the status message to clarify it was an insertion
        if status.startswith("applied:"):
            lines_inserted = insert_content.count("\n") + (0 if insert_content.endswith("\n") else 1)
            return f"{status} (inserted {len(insert_content)} chars, ~{lines_inserted} lines after line {after_line})"

        return status

    def _resolve_line_start_offsets(
        self,
        snapshot: Mapping[str, Any],
        text: str,
    ) -> Sequence[int]:
        """Resolve line start offsets from snapshot or compute from text."""
        raw = snapshot.get("line_start_offsets")
        if raw is None:
            raw = snapshot.get("line_offsets")
        offsets: list[int] = []
        if isinstance(raw, Sequence):
            for value in raw:
                try:
                    cursor = int(value)
                except (TypeError, ValueError):
                    continue
                cursor = max(0, cursor)
                if offsets and cursor < offsets[-1]:
                    cursor = offsets[-1]
                offsets.append(cursor)
        if not offsets:
            offsets = self._build_line_start_offsets(text)
        if not offsets:
            offsets = [0]
        if offsets[0] != 0:
            offsets.insert(0, 0)
        length = len(text or "")
        if offsets[-1] < length:
            offsets.append(length)
        elif offsets[-1] > length:
            offsets[-1] = length
        return offsets

    @staticmethod
    def _build_line_start_offsets(text: str) -> list[int]:
        """Build line start offsets from document text."""
        offsets = [0]
        if not text:
            return offsets
        cursor = 0
        for segment in text.splitlines(keepends=True):
            cursor += len(segment)
            offsets.append(cursor)
        if offsets[-1] < len(text):
            offsets.append(len(text))
        return offsets


__all__ = ["DocumentInsertTool"]
