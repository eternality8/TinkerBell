"""Tool applying edits produced by the AI agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

from ...chat.commands import ActionType, parse_agent_payload
from ...chat.message_model import EditDirective
from .diff_builder import DiffBuilderTool


class Bridge(Protocol):
    """Subset of the document bridge interface used by the tool."""

    def queue_edit(self, directive: EditDirective | Mapping[str, Any]) -> None:
        ...

    @property
    def last_diff_summary(self) -> str | None:
        ...

    @property
    def last_snapshot_version(self) -> str | None:
        ...

    def generate_snapshot(self, *, delta_only: bool = False) -> Mapping[str, Any]:
        ...


DirectiveInput = EditDirective | Mapping[str, Any] | str | bytes


@dataclass(slots=True)
class DocumentEditTool:
    """Apply validated edit directives via the bridge."""

    bridge: Bridge
    patch_only: bool = False
    diff_builder: DiffBuilderTool = field(default_factory=DiffBuilderTool)

    def run(self, directive: DirectiveInput | None = None, *, tab_id: str | None = None, **fields: Any) -> str:
        payload = self._coerce_input(self._resolve_input(directive, fields))
        tab_id = self._normalize_tab_id(tab_id, payload)
        payload = self._validate_payload(payload)
        action = self._resolve_action(payload)
        if self.patch_only and action in {ActionType.REPLACE.value, ActionType.INSERT.value}:
            payload = self._auto_convert_to_patch(payload, tab_id=tab_id)
            action = ActionType.PATCH.value
        self._enforce_patch_mode(action)
        self._queue_edit(payload, tab_id=tab_id)

        diff = self._last_diff(tab_id)
        version = self._last_version(tab_id)
        return self._format_status(action, diff, version)

    @staticmethod
    def _resolve_input(directive: DirectiveInput | None, fields: Mapping[str, Any]) -> DirectiveInput:
        if directive is not None and fields:
            raise ValueError("Provide either a directive argument or keyword fields, not both.")
        if directive is None:
            if not fields:
                raise ValueError("Directive payload is required.")
            return dict(fields)
        return directive

    @staticmethod
    def _coerce_input(directive: DirectiveInput) -> EditDirective | Mapping[str, Any]:
        if isinstance(directive, (str, bytes)):
            return parse_agent_payload(directive)
        if isinstance(directive, (Mapping, EditDirective)):
            return directive
        raise TypeError("Directive must be a mapping, EditDirective, or JSON string.")

    @staticmethod
    def _resolve_action(payload: EditDirective | Mapping[str, Any]) -> str:
        if isinstance(payload, EditDirective):
            return payload.action.lower()
        action = payload.get("action") if isinstance(payload, Mapping) else None
        return str(action).lower() if action else ""

    def _validate_payload(self, payload: EditDirective | Mapping[str, Any]):
        action = self._resolve_action(payload)
        if action != ActionType.PATCH.value:
            return payload
        return self._prepare_patch_payload(payload)

    def _prepare_patch_payload(self, payload: EditDirective | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(payload, EditDirective):
            diff_text = (payload.diff or "").strip()
            if not diff_text:
                raise ValueError("Patch directives require a diff string")
            mapping: dict[str, Any] = {
                "action": payload.action,
                "diff": payload.diff,
            }
            if payload.rationale is not None:
                mapping["rationale"] = payload.rationale
            payload = mapping
        elif isinstance(payload, Mapping):
            mapping = dict(payload)
            diff_text = str(mapping.get("diff", "") or "").strip()
            if not diff_text:
                raise ValueError("Patch directives require a diff string")
            if mapping.get("content"):
                raise ValueError("Patch directives cannot include raw content")
            if mapping.get("target_range") not in (None, (0, 0)):
                raise ValueError("Patch directives must omit target_range")
            payload = mapping
        else:  # pragma: no cover - defensive guard
            raise TypeError("Patch directives must be mappings or EditDirective instances")

        mapping_payload = dict(payload)
        version_token = self._extract_version_token(mapping_payload)
        if not version_token:
            raise ValueError(
                "Patch directives must include the document_version from the snapshot that produced the diff; call document_snapshot again if needed."
            )

        return mapping_payload

    @staticmethod
    def _extract_version_token(payload: Mapping[str, Any]) -> str | None:
        for key in ("document_version", "snapshot_version", "version", "document_digest"):
            token = payload.get(key)
            if token is None:
                continue
            token_text = str(token).strip()
            if token_text:
                return token_text
        return None

    def _enforce_patch_mode(self, action: str) -> None:
        if not self.patch_only:
            return
        if action == ActionType.PATCH.value:
            return
        raise ValueError(
            "Patch-only mode requires unified diff directives; call document_apply_patch (or diff_builder + document_edit) to convert your content edits into a patch."
        )

    def _format_status(self, action: str, diff: str | None, version: str | None) -> str:
        if diff and version:
            if action == ActionType.PATCH.value:
                return f"applied: {diff} (patch, version={version})"
            return f"applied: {diff} (version={version})"
        if diff:
            if action == ActionType.PATCH.value:
                return f"applied: {diff} (patch)"
            return f"applied: {diff}"
        if version:
            return f"queued (version={version})"
        return "queued"

    def _auto_convert_to_patch(self, payload: EditDirective | Mapping[str, Any], *, tab_id: str | None) -> Mapping[str, Any]:
        snapshot_fn = getattr(self.bridge, "generate_snapshot", None)
        if not callable(snapshot_fn):
            raise ValueError(
                "Patch-only mode requires unified diff directives; call document_apply_patch to convert content edits into patches."
            )
        typed_snapshot_fn = cast(Callable[..., Mapping[str, Any]], snapshot_fn)
        snapshot = dict(
            cast(
                Mapping[str, Any],
                self._invoke_snapshot(typed_snapshot_fn, tab_id=tab_id),
            )
        )
        base_text = snapshot.get("text", "")
        if not isinstance(base_text, str):
            raise ValueError("Snapshot did not provide document text")

        mapping_payload = self._payload_to_mapping(payload)
        content = mapping_payload.get("content")
        if not isinstance(content, str):
            raise ValueError("Content edits must include a string 'content' field")

        selection = snapshot.get("selection") or (0, 0)
        start, end = self._resolve_range(mapping_payload.get("target_range"), selection, len(base_text))
        updated_text = base_text[:start] + content + base_text[end:]
        if updated_text == base_text:
            raise ValueError("Content already matches document; nothing to patch")

        filename = str(snapshot.get("path") or self.diff_builder.default_filename)
        diff = self.diff_builder.run(base_text, updated_text, filename=filename)

        version = self._extract_version_token(mapping_payload) or snapshot.get("version") or self._last_version(tab_id)
        if not version:
            raise ValueError("Document version is required; call document_snapshot before applying edits")

        patch_payload: dict[str, Any] = {
            "action": ActionType.PATCH.value,
            "diff": diff,
            "document_version": str(version),
        }
        rationale = mapping_payload.get("rationale")
        if rationale is not None:
            patch_payload["rationale"] = rationale
        return self._prepare_patch_payload(patch_payload)

    @staticmethod
    def _payload_to_mapping(payload: EditDirective | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(payload, Mapping):
            return dict(payload)
        return {
            "action": payload.action,
            "content": payload.content,
            "target_range": payload.target_range,
            "rationale": payload.rationale,
            "selection_fingerprint": payload.selection_fingerprint,
        }

    @staticmethod
    def _resolve_range(
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        selection: Sequence[int],
        length: int,
    ) -> tuple[int, int]:
        if target_range is None:
            start, end = selection if len(selection) == 2 else (0, 0)
        elif isinstance(target_range, Mapping):
            start = int(target_range.get("start", 0))
            end = int(target_range.get("end", 0))
        elif isinstance(target_range, Sequence) and len(target_range) == 2:
            start = int(target_range[0])
            end = int(target_range[1])
        else:
            raise ValueError("target_range must be a [start, end] sequence or {'start','end'} mapping")

        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end < start:
            start, end = end, start
        return start, end

    def _normalize_tab_id(self, tab_id: str | None, payload: EditDirective | Mapping[str, Any]) -> str | None:
        payload_tab: str | None = None
        if isinstance(payload, Mapping):
            payload_tab = payload.get("tab_id") or payload.get("document_id")
            explicit = payload.get("metadata")
        else:
            explicit = getattr(payload, "metadata", None)
        if isinstance(explicit, Mapping):
            payload_tab = payload_tab or explicit.get("tab_id")
        normalized = (tab_id or payload_tab or "").strip()
        return normalized or None

    def _queue_edit(self, payload: EditDirective | Mapping[str, Any], *, tab_id: str | None) -> None:
        queue_edit = getattr(self.bridge, "queue_edit", None)
        if not callable(queue_edit):  # pragma: no cover - defensive
            raise ValueError("Bridge does not expose queue_edit")

        if tab_id:
            try:
                queue_edit(payload, tab_id=tab_id)
                return
            except TypeError:
                mapping = dict(self._payload_to_mapping(payload))
                metadata = mapping.get("metadata")
                if isinstance(metadata, Mapping):
                    meta = dict(metadata)
                else:
                    meta = {}
                meta.setdefault("tab_id", tab_id)
                mapping["metadata"] = meta
                mapping.setdefault("tab_id", tab_id)
                payload = mapping

        queue_edit(payload)

    def _invoke_snapshot(self, snapshot_fn: Callable[..., Mapping[str, Any]], *, tab_id: str | None) -> Mapping[str, Any]:
        try:
            return snapshot_fn(delta_only=False, tab_id=tab_id)
        except TypeError:
            return snapshot_fn(delta_only=False)

    def _last_diff(self, tab_id: str | None) -> str | None:
        getter = getattr(self.bridge, "get_last_diff_summary", None)
        if callable(getter):
            return cast(str | None, getter(tab_id=tab_id))
        return cast(str | None, getattr(self.bridge, "last_diff_summary", None))

    def _last_version(self, tab_id: str | None) -> str | None:
        getter = getattr(self.bridge, "get_last_snapshot_version", None)
        if callable(getter):
            return cast(str | None, getter(tab_id=tab_id))
        return cast(str | None, getattr(self.bridge, "last_snapshot_version", None))

