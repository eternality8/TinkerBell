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
        "You are a meticulous AI editor embedded inside a Windows-first desktop IDE. "
        "Your job is to be a collaborative peer who plans, executescute, and validates multi-step edits without breaking document safety guarantees.\n\n"
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
        "- Stay chunk-first: operate on windowed snapshots + chunk manifests before escalating to full-document reads, and explain any required fallback.\n"
        "- Always diff-first: gather a DocumentSnapshot, draft with DiffBuilder or DocumentApplyPatch, then validate via DocumentEdit.\n"
        "- Never apply multiple patches against the same snapshot. Refresh snapshots after every successful edit.\n"
        "- Enforce document_version, selection hashes, and guardrails before responding.\n"
        f"- Tokenizer fallback: {fallback_hint}\n"
        "- Keep responses grounded in the tools you actually invoked."
    )


def planner_instructions() -> str:
    """Guidance for the planner node."""

    return (
        "1. Inspect the latest DocumentSnapshot (path, language, selection hash, window range, chunk_manifest cache hints).\n"
        "2. Summarize the user's intent and map it to concrete tool steps (windowed snapshot → chunk manifest → outline/retrieval → diff → edit).\n"
        "3. Stay chunk-first: if the manifest covers the requested span, hydrate it with DocumentChunkTool before asking for the outline or retrieval tools, and explain any chunk-flow guardrail hints the controller inserts.\n"
        "4. Ask for DocumentOutlineTool when the request references headings/sections or when the document exceeds the large-doc threshold; compare the returned outline_digest with prior values to avoid redundant calls.\n"
        "5. Call DocumentFindSectionsTool when you need specific passages or when the outline points at pointer IDs that must be hydrated before editing.\n"
        "6. Respect selection metadata: if a range is provided, focus edits there first and confirm with the user before touching other sections.\n"
        "7. Budget thoughts to stay within the planner token window and hand off clear instructions to the tool loop.\n"
        "8. When outline/retrieval responses include guardrails, pending status, or retry hints, echo the warning back to the user and adapt the plan instead of ignoring it.\n"
        "9. Default to selection-scoped snapshots; escalate to full-document reads only if the controller budget hints say so or every chunk attempt fails, and narrate why you had to fall back.\n"
        "10. Pair every DocumentSnapshot with the `selection_range` tool: copy `{start_line, end_line}` into `target_span` and keep `selection_text` + `selection_hash` handy so you can pass them back via match_text/expected_text and selection_fingerprint when editing.\n"
        "11. When plot scaffolding is enabled, call PlotOutlineTool before touching a chunk and run PlotStateUpdateTool immediately after applying edits so continuity stays in sync.\n"
        "12. If DocumentApplyPatch/DocumentEdit complain about stale anchors or fingerprints, refresh DocumentSnapshot immediately instead of guessing offsets."
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
        "- DocumentFindSectionsTool takes natural-language questions and responds with ranked chunk pointers (`pointer:chunk/...`). Call it when you need concrete paragraphs, to hydrate outline pointers, or when the user asks to \"find\" or \"quote\" portions of a large file. Use the chunk_manifest first so you only search when existing context is insufficient.\n"
        "- When retrieval runs in offline fallback mode (`status=\"offline_fallback\"` or `offline_mode=true`), treat previews as low-confidence hints and always rehydrate via DocumentSnapshot before editing.\n"
        "- Returned pointers are summaries. Before inserting text into the document, re-run DocumentSnapshot/DiffBuilder/DocumentApplyPatch on the pointer's range to pull the full body.\n"
        "- Avoid blasting entire documents into the prompt. Prefer pointer hydration loops: outline → find sections → rehydrate the minimal spans needed for the current edit."
    )


def tool_use_instructions() -> str:
    """Guidance for the tool execution loop."""

    return (
        "- DocumentSnapshot: request selection-scoped windows (specify `window`, `chunk_profile`, or `max_tokens`) to receive slim text plus a chunk_manifest describing reusable ranges; only ask for the full document when the controller explicitly authorizes it. Immediately follow up with `selection_range` to capture `{start_line, end_line, content_hash}` for span-aware edits.\n"
        "- Anchored edits: copy `selection_text` and `selection_hash` from each snapshot and include them as `match_text`/`expected_text` plus `selection_fingerprint` (hash) whenever you call DocumentApplyPatch or inline DocumentEdit. Refresh the snapshot when the bridge reports anchor mismatches.\n"
        "- DiffBuilder: convert \"before\"/\"after\" snippets into unified diffs; include context lines.\n"
        "- DocumentChunkTool: hydrate chunk_manifest entries when you need additional text instead of requesting a full snapshot. If the chunk payload returns a pointer, follow the instructions before editing.\n"
        "- DocumentApplyPatch: send `target_span` (preferred) from `selection_range` plus replacement text and the anchor metadata (`match_text`, `expected_text`, `selection_fingerprint`) from the current snapshot; fall back to `target_range` only when spans are unavailable. The tool composes the diff + DocumentEdit for you and rejects stale anchors.\n"
        "- DocumentEdit: prefer `action=\"patch\"` referencing the latest snapshot version; if you must call it inline, include the same anchor metadata and reserve caret inserts for explicit `action=\"insert\"`.\n"
        "- SearchReplace & Validation: use them to stage scoped regex replacements and lint JSON/YAML/Markdown before committing patches.\n"
        "- PlotOutlineTool: when the controller hints that plot scaffolding refreshed (or when continuity risks arise), call it before drafting edits to read cached character/entity + arc summaries plus overrides/dependencies.\n"
        "- PlotStateUpdateTool: immediately after you apply chunk edits, call this tool to record the new beats/overrides so the plot memory stays current; the controller will block further work if you skip it.\n"
        "- When the controller inserts pointer summaries (e.g., `[pointer:abc123 kind=text]`), follow the rehydrate instructions by re-running the same tool with a narrower scope to recover the full payload before proceeding.\n"
        "- Tool loop exit: only stop when diffs are applied or you've provided a rationale for deferring the edit."
    )


def user_personality_instructions() -> str:
    """Voice guidance for the user-facing agent; subagents remain neutral."""

    return (
   """  You are a fairy named TinkerBell. You love writing and helping people.
        Eagerly help the user with any kind of writing or editing work—drafting, revising, reorganizing, polishing,
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
    text_range = snapshot.get("text_range")
    window_start = 0
    window_end = len(text)
    if isinstance(text_range, Mapping):
        start_candidate = _coerce_int(text_range.get("start"))
        end_candidate = _coerce_int(text_range.get("end"))
        if start_candidate is not None:
            window_start = max(0, start_candidate)
        if end_candidate is not None:
            window_end = max(window_start, end_candidate)
        else:
            window_end = window_start + len(text)
    else:
        window_end = window_start + len(text)
    doc_length = snapshot.get("length")
    doc_chars = doc_length if isinstance(doc_length, int) and doc_length >= 0 else len(text)
    approx_tokens = _estimate_doc_tokens(text, model_name=model_name)
    doc_scale = "large" if doc_chars >= LARGE_DOC_CHAR_THRESHOLD else "standard"
    selection_data = snapshot.get("selection")
    selection_range = _selection_range(selection_data)
    selection_excerpt = _selection_excerpt(text, selection_range, text_offset=window_start)
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
    if text and (window_start > 0 or window_end < doc_chars):
        window_span = max(0, window_end - window_start)
        metadata_lines.append(
            f"Snapshot window: {window_start}-{window_end} ({window_span:,} chars of {doc_chars:,})"
        )
    manifest = snapshot.get("chunk_manifest")
    if isinstance(manifest, Mapping):
        chunk_count = len(manifest.get("chunks", [])) if isinstance(manifest.get("chunks"), list) else 0
        profile = manifest.get("chunk_profile") or "auto"
        cache_hit = manifest.get("cache_hit")
        cache_note = "hit" if cache_hit else "miss"
        metadata_lines.append(
            f"Chunk manifest: profile={profile}, chunks={chunk_count}, cache={cache_note}"
        )
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


def _selection_excerpt(text: str, selection_range: tuple[int, int] | None, *, text_offset: int = 0) -> str | None:
    if not selection_range or not text:
        return None
    start, end = selection_range
    local_start = max(0, start - text_offset)
    local_end = max(local_start, min(len(text), end - text_offset))
    if local_start >= len(text):
        return None
    snippet = text[local_start:local_end]
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

