"""Tool that replaces the entire document content in a single call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Protocol

from .document_apply_patch import DocumentApplyPatchTool
from .validation import InvalidSnapshotTokenError, parse_snapshot_token
from ...services.telemetry import emit as telemetry_emit


class ReplaceAllBridge(Protocol):
    """Protocol describing the bridge required by DocumentReplaceAllTool."""

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
class DocumentReplaceAllTool:
    """Replace the entire document content using a simplified schema.
    
    This tool provides a minimal API for full-document replacements:
    - snapshot_token: Compact version identifier (tab_id:version_id)
    - content: The new document content
    
    It delegates to DocumentApplyPatchTool with replace_all=True.
    """

    patch_tool: DocumentApplyPatchTool
    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        snapshot_token: str | None = None,
        content: str,
        rationale: str | None = None,
        tab_id: str | None = None,
    ) -> str:
        """Replace the entire document with new content.
        
        Args:
            snapshot_token: Compact version identifier in format 'tab_id:version_id'.
            content: The new document content to replace the entire document.
            rationale: Optional explanation stored alongside the edit directive.
            tab_id: Optional tab identifier; defaults to value from snapshot_token.
        
        Returns:
            Status message indicating success or failure.
        
        Raises:
            InvalidSnapshotTokenError: If snapshot_token is malformed.
            ValueError: If required fields are missing or invalid.
        """
        # Parse snapshot_token to extract version info
        parsed_tab_id, parsed_version_id = self._parse_snapshot_token(snapshot_token)
        
        if parsed_tab_id is not None and tab_id is None:
            tab_id = parsed_tab_id
        
        # Get the current snapshot to retrieve required version fields
        bridge = self.patch_tool.bridge
        snapshot = dict(bridge.generate_snapshot(
            delta_only=False,
            tab_id=tab_id,
            include_text=False,
        ))
        
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
        
        # Emit telemetry for the replace_all operation
        telemetry_emit(
            "document_replace_all",
            {
                "tab_id": tab_id,
                "snapshot_token": snapshot_token,
                "content_length": len(content),
                "document_length": snapshot.get("length", 0),
            },
        )
        
        # Delegate to DocumentApplyPatchTool with replace_all=True
        return self.patch_tool.run(
            content=content,
            replace_all=True,
            scope="document",
            document_version=str(document_version),
            version_id=version_id,
            content_hash=content_hash,
            rationale=rationale,
            tab_id=tab_id,
            snapshot_token=snapshot_token,
        )

    def _parse_snapshot_token(self, token: str | None) -> tuple[str | None, str | None]:
        """Parse snapshot_token into (tab_id, version_id) components.

        Uses strict mode to raise InvalidSnapshotTokenError for malformed tokens.
        """
        return parse_snapshot_token(token, strict=True)


__all__ = ["DocumentReplaceAllTool"]
