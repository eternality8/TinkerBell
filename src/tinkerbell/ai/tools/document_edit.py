"""Tool applying edits produced by the AI agent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterable, Mapping, Protocol, Sequence, cast

from ...chat.commands import ActionType, extract_tab_reference, parse_agent_payload, resolve_tab_reference
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
    allow_inline_edits: bool = False
    diff_builder: DiffBuilderTool = field(default_factory=DiffBuilderTool)
    diff_context_lines: int = 5
    summarizable: ClassVar[bool] = False

    def run(self, directive: DirectiveInput | None = None, *, tab_id: str | None = None, **fields: Any) -> str:
        payload = self._coerce_input(self._resolve_input(directive, fields))
        tab_id = self._normalize_tab_id(tab_id, payload)
        action = self._resolve_action(payload)
        payload = self._validate_payload(payload, action)
        if self.patch_only and action in {ActionType.REPLACE.value, ActionType.INSERT.value, ActionType.ANNOTATE.value}:
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

    def _validate_payload(self, payload: EditDirective | Mapping[str, Any], action: str):
        if action == ActionType.PATCH.value:
            return self._prepare_patch_payload(payload)
        if not self.allow_inline_edits and not self.patch_only:
            raise ValueError(
                "DocumentEditTool is configured for diff-only edits; call document_apply_patch to convert inline content edits into patches."
            )
        return payload

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
        content_hash = self._extract_content_hash(mapping_payload)
        if not content_hash:
            raise ValueError("Patch directives must include the content_hash from the originating snapshot.")

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

    @staticmethod
    def _extract_content_hash(payload: Mapping[str, Any]) -> str | None:
        token = payload.get("content_hash")
        if token is None:
            return None
        token_text = str(token).strip()
        return token_text or None

    def _enforce_patch_mode(self, action: str) -> None:
        if self.patch_only and action != ActionType.PATCH.value:
            raise ValueError(
                "Patch-only mode requires unified diff directives; call document_apply_patch (or diff_builder + document_edit) to convert your content edits into a patch."
            )
        if action != ActionType.PATCH.value and not self.allow_inline_edits:
            raise ValueError(
                "DocumentEditTool is configured for diff-only edits; convert content edits into patches before calling document_edit."
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

        filename = self._normalize_diff_filename(snapshot)
        diff = self.diff_builder.run(
            base_text,
            updated_text,
            filename=filename,
            context=self.diff_context_lines,
        )

        version = self._extract_version_token(mapping_payload) or snapshot.get("version") or self._last_version(tab_id)
        if not version:
            raise ValueError("Document version is required; call document_snapshot before applying edits")
        content_hash = self._extract_content_hash(mapping_payload) or snapshot.get("content_hash")
        if not content_hash:
            content_hash = self._hash_text(base_text)

        patch_payload: dict[str, Any] = {
            "action": ActionType.PATCH.value,
            "diff": diff,
            "document_version": str(version),
            "content_hash": str(content_hash),
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
        payload_metadata: Mapping[str, Any] | None = None
        if isinstance(payload, Mapping):
            payload_tab = payload.get("tab_id") or payload.get("document_id")
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                payload_metadata = metadata
        else:
            payload_tab = getattr(payload, "tab_id", None)
            payload_metadata = getattr(payload, "metadata", None)

        tabs = self._list_tabs()
        active_tab = self._active_tab_id()

        candidate = (tab_id or payload_tab or "").strip()
        resolved = self._resolve_tab_identifier(candidate, tabs, active_tab, allow_fallback=True)
        if resolved:
            return resolved

        reference = None
        mapping_payload = self._payload_to_mapping(payload)
        reference = extract_tab_reference(mapping_payload)
        if reference is None and isinstance(payload_metadata, Mapping):
            reference = extract_tab_reference(dict(payload_metadata))
        resolved = self._resolve_tab_identifier(reference or "", tabs, active_tab, allow_fallback=False)
        return resolved or None

    def _resolve_tab_identifier(
        self,
        candidate: str,
        tabs: Sequence[Mapping[str, Any]],
        active_tab_id: str | None,
        *,
        allow_fallback: bool,
    ) -> str | None:
        normalized = (candidate or "").strip()
        if not normalized:
            return None
        if not tabs:
            return normalized
        for entry in tabs:
            tab_id = str(entry.get("tab_id") or entry.get("id") or "").strip()
            if tab_id and tab_id.lower() == normalized.lower():
                return tab_id
        resolved = resolve_tab_reference(normalized, tabs, active_tab_id=active_tab_id)
        if resolved:
            return resolved
        return normalized if allow_fallback else None

    def _list_tabs(self) -> Sequence[Mapping[str, Any]]:
        provider = getattr(self.bridge, "list_tabs", None)
        if not callable(provider):
            return ()
        try:
            tabs = provider()
        except Exception:  # pragma: no cover - defensive guard
            return ()
        if isinstance(tabs, Sequence):
            return tabs
        if isinstance(tabs, Iterable):
            return list(tabs)
        return ()

    def _active_tab_id(self) -> str | None:
        getter = getattr(self.bridge, "active_tab_id", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # pragma: no cover - defensive guard
                return None
            if value is None:
                return None
            text = str(value).strip()
            return text or None
        return None

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

    @staticmethod
    def _normalize_diff_filename(snapshot: Mapping[str, Any]) -> str:
        path = snapshot.get("path")
        if isinstance(path, str) and path.strip():
            return path.strip()
        document_id = snapshot.get("document_id") or "document"
        return f"tab://{document_id}"

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

