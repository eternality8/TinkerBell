"""Helper for formatting manual outline/find-sections responses and traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..chat.message_model import ToolTrace

DocumentLabelResolver = Callable[[str | None, str | None], str]
WhitespaceNormalizer = Callable[[str], str]


@dataclass(slots=True)
class ManualToolController:
    """Encapsulates manual tool response formatting and trace recording."""

    chat_panel: Any
    document_label_resolver: DocumentLabelResolver
    whitespace_normalizer: WhitespaceNormalizer

    def render_outline_response(self, response: Mapping[str, Any], requested_label: str | None) -> str:
        status = str(response.get("status") or "unknown")
        doc_label = self.document_label_resolver(response.get("document_id"), requested_label)
        parts = [f"Document outline ({status}) for {doc_label}."]
        reason = response.get("reason")
        if reason:
            parts.append(f"Reason: {reason}.")

        nodes = response.get("nodes") or []
        if nodes:
            parts.append("Headings:")
            parts.extend(self.render_outline_tree_lines(nodes))
        else:
            outline_available = response.get("outline_available")
            if outline_available is False:
                parts.append("No outline is available for this document yet.")

        notes: list[str] = []
        if response.get("trimmed"):
            reason_text = response.get("trimmed_reason") or "request limits"
            notes.append(f"trimmed={reason_text}")
        if response.get("is_stale"):
            notes.append("stale compared to current document")
        if notes:
            parts.append("Notes: " + ", ".join(notes) + ".")

        generated = response.get("generated_at")
        if generated:
            parts.append(f"Generated at {generated}.")
        outline_digest = response.get("outline_digest")
        if outline_digest:
            parts.append(f"Digest: {outline_digest}.")

        return "\n".join(part for part in parts if part)

    def render_outline_tree_lines(
        self,
        nodes: Sequence[Mapping[str, Any]],
        *,
        limit: int = 24,
    ) -> list[str]:
        lines: list[str] = []
        truncated = False

        def visit(node: Mapping[str, Any], level_hint: int) -> None:
            nonlocal truncated
            if len(lines) >= limit:
                truncated = True
                return
            level = int(node.get("level") or level_hint or 1)
            indent = "  " * max(0, level - 1)
            text = str(node.get("text") or "Untitled").strip() or "Untitled"
            pointer = node.get("pointer_id") or node.get("id")
            suffix = f" ({pointer})" if pointer else ""
            lines.append(f"{indent}- {text}{suffix}")
            children = node.get("children") or []
            for child in children:
                if not isinstance(child, Mapping):
                    continue
                visit(child, level + 1)
                if truncated:
                    return

        for entry in nodes:
            if not isinstance(entry, Mapping):
                continue
            visit(entry, int(entry.get("level") or 1))
            if truncated:
                break

        if truncated:
            lines.append("  … additional headings omitted.")
        return lines

    def render_retrieval_response(
        self,
        response: Mapping[str, Any],
        requested_query: str | None,
        requested_document_label: str | None,
    ) -> str:
        status = str(response.get("status") or "unknown")
        doc_label = self.document_label_resolver(response.get("document_id"), requested_document_label)
        query_text = response.get("query") or (requested_query or "")
        if query_text:
            header = f"Find text ({status}) for {doc_label} — \"{query_text}\""
        else:
            header = f"Find text ({status}) for {doc_label}."
        parts = [header]

        details: list[str] = []
        strategy = response.get("strategy")
        if strategy:
            details.append(f"strategy={strategy}")
        fallback_reason = response.get("fallback_reason")
        if fallback_reason:
            details.append(f"fallback={fallback_reason}")
        latency = response.get("latency_ms")
        if isinstance(latency, (int, float)):
            details.append(f"latency={latency:.1f} ms")
        if details:
            parts.append("Details: " + ", ".join(details) + ".")

        pointers = response.get("pointers") or []
        if pointers:
            parts.append("Matches:")
            parts.extend(self.format_retrieval_pointers(pointers))
            extra = max(0, len(pointers) - 5)
            if extra:
                parts.append(f"… {extra} additional match(es).")
        else:
            parts.append("No matching spans were found.")

        return "\n".join(parts)

    def format_retrieval_pointers(
        self,
        pointers: Sequence[Mapping[str, Any]],
        *,
        limit: int = 5,
    ) -> list[str]:
        lines: list[str] = []
        for index, pointer in enumerate(pointers[:limit], start=1):
            if not isinstance(pointer, Mapping):
                continue
            pointer_id = pointer.get("pointer_id") or pointer.get("chunk_id") or f"chunk-{index}"
            outline_context = pointer.get("outline_context")
            heading = outline_context.get("heading") if isinstance(outline_context, Mapping) else None
            label_parts = [str(pointer_id)]
            if heading:
                label_parts.append(str(heading))
            score = pointer.get("score")
            score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "n/a"
            lines.append(f"{index}. {' · '.join(label_parts)} (score {score_text})")
            preview = pointer.get("preview")
            snippet = self.whitespace_normalizer(str(preview)) if isinstance(preview, str) else ""
            if snippet:
                if len(snippet) > 180:
                    snippet = f"{snippet[:177]}…"
                lines.append(f"    {snippet}")
        return lines

    @staticmethod
    def summarize_manual_input(label: str, args: Mapping[str, Any]) -> str:
        if not args:
            return label
        preferred = ("document_id", "query", "top_k", "desired_levels", "max_nodes", "min_confidence")
        parts: list[str] = []
        for key in preferred:
            if key in args:
                parts.append(f"{key}={args[key]}")
        if not parts:
            for key, value in list(args.items())[:4]:
                parts.append(f"{key}={value}")
        summary = ", ".join(parts[:4])
        return f"{label} ({summary})" if summary else label

    def record_manual_tool_trace(
        self,
        *,
        name: str,
        input_summary: str,
        output_summary: str,
        args: Mapping[str, Any],
        response: Mapping[str, Any] | Any,
    ) -> None:
        args_payload = dict(args) if isinstance(args, Mapping) else {"value": args}
        if isinstance(response, Mapping):
            response_payload: Mapping[str, Any] | Any = response
        else:
            response_payload = {"value": response}
        metadata = {
            "raw_input": json.dumps(args_payload, default=str),
            "raw_output": json.dumps(response_payload, default=str),
            "manual_command": True,
        }
        trace = ToolTrace(
            name=name,
            input_summary=input_summary,
            output_summary=output_summary,
            metadata=metadata,
        )
        show_trace = getattr(self.chat_panel, "show_tool_trace", None)
        if callable(show_trace):
            show_trace(trace)


__all__ = ["ManualToolController"]
