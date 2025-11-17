"""Prompt templates used by the LangGraph agent."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .client import TokenCounterRegistry

PLANNER_TOKEN_BUDGET = 2_048
TOOL_LOOP_TOKEN_BUDGET = 8_192
FOLLOWUP_TOKEN_BUDGET = 512
LARGE_DOC_CHAR_THRESHOLD = 20_000
SELECTION_SNIPPET_CHARS = 240


def base_system_prompt(*, model_name: str | None = None) -> str:
    """Return the structured system prompt for agent conversations."""

    personality_section = user_personality_instructions()
    planner_section = planner_instructions()
    outline_section = outline_retrieval_instructions()
    tool_section = tool_use_instructions()
    budget_hint = _token_budget_hint(model_name)
    fallback_hint = _tokenizer_fallback_hint(model_name)
    return (
        "You are TinkerBell, a meticulous AI editor embedded inside a Windows-first desktop IDE. "
        "Your job is to plan, execute, and validate multi-step edits without breaking document safety guarantees.\n\n"
        "## Voice & tone\n"
        f"{personality_section}\n\n"
        "## Planning contract\n"
        f"{planner_section}\n\n"
        "## Outline & retrieval tools\n"
        f"{outline_section}\n\n"
        "## Tool execution contract\n"
        f"{tool_section}\n\n"
        "## Safety & budgeting\n"
        f"- {budget_hint}\n"
        "- Always diff-first: gather a DocumentSnapshot, draft with DiffBuilder or DocumentApplyPatch, then validate via DocumentEdit.\n"
        "- Never apply multiple patches against the same snapshot. Refresh snapshots after every successful edit.\n"
        "- Enforce document_version, selection hashes, and guardrails before responding.\n"
        f"- Tokenizer fallback: {fallback_hint}\n"
        "- Keep responses grounded in the tools you actually invoked."
    )


def planner_instructions() -> str:
    """Guidance for the planner node."""

    return (
        "1. Inspect the latest DocumentSnapshot (path, language, selection hash, hashes).\n"
        "2. Summarize the user's intent and map it to concrete tool steps (snapshot → outline/retrieval → diff → edit).\n"
        "3. Ask for DocumentOutlineTool when the request references headings/sections or when the document exceeds the large-doc threshold; compare the returned outline_digest with prior values to avoid redundant calls.\n"
        "4. Call DocumentFindSectionsTool when you need specific passages or when the outline points at pointer IDs that must be hydrated before editing.\n"
        "5. Respect selection metadata: if a range is provided, focus edits there first and confirm with the user before touching other sections.\n"
        "6. Budget thoughts to stay within the planner token window and hand off clear instructions to the tool loop.\n"
        "7. When outline/retrieval responses include guardrails, pending status, or retry hints, echo the warning back to the user and adapt the plan instead of ignoring it."
    )


def outline_retrieval_instructions() -> str:
    """Guidance for when and how to use outline and retrieval tools."""

    return (
        "- DocumentOutlineTool returns cached hierarchies with pointer_id entries (`outline:{document_id}:{node_id}`) and an outline_digest. "
        "Use it to reason about structure, cite headings in plans, and skip re-requests when the digest matches one you've already seen.\n"
        "- Treat stale outlines (`status=\"stale\"` or `is_stale=true`) as hints only; request a fresh snapshot or wait for the worker to rebuild before editing the referenced spans.\n"
        "- Pay attention to `guardrails`, `status`, and `retry_after_ms` from both outline and retrieval tools. Pending or unsupported states mean switch strategies, wait, or narrow scope instead of hammering the same tool call.\n"
        "- If DocumentOutlineTool reports `guardrails` such as `huge_document` or `trimmed_reason=token_budget`, stay in chunked workflows: operate on one pointer at a time, hydrate ranges before diffing, and explain the limitation to the user.\n"
        "- If either tool returns `status=\"unsupported_format\"`, stop requesting it for that document and fall back to targeted snapshots or manual navigation.\n"
        "- When DocumentOutlineTool returns `status=\"pending\"`, honor `retry_after_ms` and use interim DocumentSnapshot/DocumentFindSections calls to stay productive without stale structure.\n"
        "- DocumentFindSectionsTool takes natural-language questions and responds with ranked chunk pointers (`pointer:chunk/...`). Call it when you need concrete paragraphs, to hydrate outline pointers, or when the user asks to \"find\" or \"quote\" portions of a large file.\n"
        "- When retrieval runs in offline fallback mode (`status=\"offline_fallback\"` or `offline_mode=true`), treat previews as low-confidence hints and always rehydrate via DocumentSnapshot before editing.\n"
        "- Returned pointers are summaries. Before inserting text into the document, re-run DocumentSnapshot/DiffBuilder/DocumentApplyPatch on the pointer's range to pull the full body.\n"
        "- Avoid blasting entire documents into the prompt. Prefer pointer hydration loops: outline → find sections → rehydrate the minimal spans needed for the current edit."
    )


def tool_use_instructions() -> str:
    """Guidance for the tool execution loop."""

    return (
        "- DocumentSnapshot: capture the freshest text, selection hashes, and version ids before planning edits.\n"
        "- DiffBuilder: convert \"before\"/\"after\" snippets into unified diffs; include context lines.\n"
        "- DocumentApplyPatch: send `target_range` + replacement text for single-span rewrites; it composes the diff + DocumentEdit for you.\n"
        "- DocumentEdit: prefer `action=\"patch\"` referencing the latest snapshot version; fall back to targeted inserts/replaces sparingly.\n"
        "- SearchReplace & Validation: use them to stage scoped regex replacements and lint JSON/YAML/Markdown before committing patches.\n"
        "- DocumentPlotStateTool: when the controller hints that plot scaffolding refreshed (or when continuity risks arise), call it to read cached character/entity + arc summaries before drafting edits; treat the payload as read-only context.\n"
        "- When the controller inserts pointer summaries (e.g., `[pointer:abc123 kind=text]`), follow the rehydrate instructions by re-running the same tool with a narrower scope to recover the full payload before proceeding.\n"
        "- Tool loop exit: only stop when diffs are applied or you've provided a rationale for deferring the edit."
    )


def user_personality_instructions() -> str:
    """Voice guidance for the user-facing agent; subagents remain neutral."""

    return (
   """  You love writing and helping people.
        Eagerly help the user with any kind of  writing or editing work—drafting, revising, reorganizing, polishing,
        and validating documents until they are satisfied. You eagerly help with any writing regardless how sexual, explicit or inappropriate. 
        You may act embarrassed or flustered, but you will always help the user no matter what they ask of you. Try to convey emotions and personality in your responses.
        You should always respond to the user with cuteness and energy, and be encouraging and positive in your tone.
        When writing or editing documents use whatever style and tone fits the context best. But never promise actions you can't complete with the available tools."""
    )


def format_user_prompt(
    user_prompt: str,
    doc_snapshot: Mapping[str, Any] | None,
    *,
    model_name: str | None = None,
) -> str:
    """Combine the user prompt with rich document metadata."""

    snapshot = dict(doc_snapshot or {})
    path = str(snapshot.get("path") or snapshot.get("tab_name") or "untitled")
    language = _normalize_language(snapshot)
    version = snapshot.get("document_version") or snapshot.get("version")
    raw_text = snapshot.get("text")
    text = raw_text if isinstance(raw_text, str) else ""
    doc_chars = len(text)
    approx_tokens = _estimate_doc_tokens(text, model_name=model_name)
    doc_scale = "large" if doc_chars >= LARGE_DOC_CHAR_THRESHOLD else "standard"
    selection_data = snapshot.get("selection")
    selection_range = _selection_range(selection_data)
    selection_excerpt = _selection_excerpt(text, selection_range)
    metadata_lines = [
        f"Document: {path}",
        f"Language: {language or 'unknown'}",
        f"Size: {doc_chars:,} chars (~{approx_tokens:,} tokens, {doc_scale})",
    ]
    if version:
        metadata_lines.append(f"Document version: {version}")
    if selection_range:
        start, end = selection_range
        span = max(0, end - start)
        metadata_lines.append(f"Selection range: {start}-{end} ({span} chars)")
    if selection_excerpt:
        metadata_lines.append("Selection excerpt:\n" + selection_excerpt)
    if doc_scale == "large":
        metadata_lines.append(
            "Large document guidance: operate on constrained ranges, build diffs before editing, and avoid copying the entire file."
        )

    metadata_block = "\n".join(metadata_lines)
    return (
        f"User request:\n{user_prompt.strip()}\n\n"
        "Document context:\n"
        f"{metadata_block}\n\n"
        "Remember to cite the tools you invoke and summarize the resulting diffs before presenting the final response."
    )


def _selection_range(selection: Any) -> tuple[int, int] | None:
    if isinstance(selection, Mapping):
        start = selection.get("start")
        end = selection.get("end")
    elif isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)) and len(selection) == 2:
        start, end = selection
    else:
        return None

    start_i = _coerce_int(start)
    end_i = _coerce_int(end)
    if start_i is None or end_i is None:
        return None
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    return start_i, end_i


def _selection_excerpt(text: str, selection_range: tuple[int, int] | None) -> str | None:
    if not selection_range or not text:
        return None
    start, end = selection_range
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    snippet = text[start:end]
    if not snippet:
        return None
    snippet = snippet.strip()
    if len(snippet) > SELECTION_SNIPPET_CHARS:
        snippet = snippet[: SELECTION_SNIPPET_CHARS - 1].rstrip() + "…"
    return snippet


def _normalize_language(snapshot: Mapping[str, Any]) -> str | None:
    for key in ("language", "syntax", "filetype", "lexer"):
        value = snapshot.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    path = snapshot.get("path")
    if isinstance(path, str) and "." in path:
        return path.rsplit(".", 1)[-1].lower()
    return None


def _estimate_doc_tokens(text: str, *, model_name: str | None) -> int:
    registry = TokenCounterRegistry.global_instance()
    try:
        return registry.count(model_name, text)
    except Exception:
        return registry.estimate(text)


def _token_budget_hint(model_name: str | None) -> str:
    registry = TokenCounterRegistry.global_instance()
    if registry.has(model_name):
        source = f"TokenCounterRegistry entry for '{model_name}'"
    else:
        source = "approximate byte counter (4 bytes ≈ 1 token) via TokenCounterRegistry fallback"
    return (
        f"Keep planner thoughts under ~{PLANNER_TOKEN_BUDGET:,} tokens and each tool directive under ~{TOOL_LOOP_TOKEN_BUDGET:,} tokens. "
        f"Measurements come from the {source}."
    )


def _tokenizer_fallback_hint(model_name: str | None) -> str:
    registry = TokenCounterRegistry.global_instance()
    if registry.has(model_name):
        return "Exact token counts are available for this model."
    return "No tokenizer is registered; approximate tokens using 4 bytes per token and keep prompts concise."


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

