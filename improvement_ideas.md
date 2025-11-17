# Large-File AI Handling Ideas

## Why this matters
Current agent flows rely on `DocumentSnapshotTool` returning the entire buffer plus metadata, so any tool call on a long file can dominate the context window and quickly stall the model. Below are concrete upgrades scoped to the `src/tinkerbell/ai` stack that make large documents manageable without downgrading editing fidelity.

## Proposed improvements

### 1. Chunk-aware `DocumentSnapshot`
- **Problem:** `DocumentBridge.generate_snapshot()` always returns the whole document text. Even a single snapshot dump can exceed the controller's `max_context_tokens`, producing runaway memory.
- **Proposal:** Add optional `window`/`max_tokens` parameters to `DocumentSnapshotTool.run` that ask the bridge for a bounded slice (selection ± N characters, nearby headings, etc.). The bridge can keep the current full-text path as default for backwards compatibility but support `include_text=False` or `delta_only="selection"` to deliver metadata-only snapshots. Pair this with a light chunk summary (start/end offsets, hashes) so the agent knows what portion it holds.
- **Key touches:** `DocumentSnapshotTool`, `DocumentBridge.generate_snapshot`, `chat.commands` schema for the tool. Requires updating the prompt contract (in `prompts.py`) so the agent knows how to request slim snapshots by default.

### 2. Secondary `DocumentChunkTool` backed by hashed segments
- **Problem:** After the snapshot windowing above, the agent still needs a way to fetch arbitrary regions on demand without re-sending the entire file.
- **Proposal:** Extend the bridge to maintain a chunk index (e.g., 2–4 KB slices with a SHA hash and line span). Add a `DocumentChunkTool` that accepts chunk ids or byte/line ranges and returns only that portion. Tool responses stay <2K tokens while enabling the agent to assemble the context it really needs.
- **Key touches:** new tool class under `src/tinkerbell/ai/tools/`, bridge helpers to cache chunk metadata, plus a light-weight cache inside `AIController` (per tab) so repeated chunk loads short-circuit.

#### Chunking strategy considerations
- **Workload-aware chunk size:** Analytical tasks (linting, structured data edits) can tolerate strict token ceilings (1–2 KB chunks) because boundaries rarely affect semantics. Creative-writing flows need larger, adaptive windows (e.g., whole-scene or multi-paragraph chunks) so voice and pacing survive. Provide presets ("code", "notes", "prose") in settings and let tools specify their desired profile when requesting chunks.
- **Semantic boundary detection:** Instead of chopping purely by bytes, derive chunk boundaries from structure: blank-line gaps, Markdown headings, YAML sections, screenplay scene headers, etc. For prose, avoid splitting inside paragraphs or dialogue exchanges by scanning for double newlines or punctuation patterns. The bridge can precompute boundary indices and store them with each chunk descriptor.
- **Dynamic resizing & overlap:** Allow the requester to expand a chunk by +/- N sentences when needed (e.g., when rewriting a character arc). Maintain a modest overlap (10–15% of chunk size) so the agent sees lead-in/lead-out context and avoids continuity breaks at borders.
- **Consistency tracking:** Tag each chunk with lineage metadata (chapter, POV character, timeline). This feeds character/storyline orchestration so edits in chunk A can flag dependent chunks B/C for follow-up without rereading the whole manuscript.
- **Streaming vs batch retrieval:** For summarization workloads, queue chunks in order and stream them to subagents; for precision edits, fetch on demand. The `DocumentChunkTool` API should expose both modes (single chunk vs iterator handle) so the controller picks the most efficient path.

### 3. Token-aware tool gating inside `AIController`
**Status:** ✅ Delivered in Phase 2 (context-budget policy, summarizer helper, pointer messages, and trace compactor hooks now live per `ai_v2_plan.md`).
- **Problem:** The controller currently only trims prior history, not tool payloads. If the agent naively adds multi-kilobyte tool outputs, the conversation still spills over the *response* reserve.
- **Proposal:** Before appending tool responses, measure their token count via `_estimate_text_tokens`. If a candidate blow past `prompt_budget`, summarize it (e.g., via a `summarize_tool_content` helper calling a tiny local heuristic) or replace it with a pointer (`"chunk:123"`). Also surface the current context usage in metadata so the agent can adapt its plan.
- **Key touches:** `_handle_tool_calls` / `_invoke_model_turn` to record token deltas, plus a new Controller-level policy object (configurable via settings) so advanced users can tune budgets per model.

### 4. Hierarchical document summaries & outlines
**Status:** ✅ Delivered in Phase 3 (outline worker, `DocumentOutlineTool`, and outline-aware snapshots shipped on the AI v2 branch).
- **Problem:** For extremely large files, even chunk sampling is noisy without a global map.
- **Proposal:** Leverage `DocumentSummaryMemory` to maintain multi-level summaries (whole file + per top-level heading). Expose a `DocumentOutlineTool` that returns only headings + short blurbs, and allow `DocumentSnapshotTool` to attach the latest outline digest. The agent can skim the outline and request precise chunks afterwards.
- **Key touches:** `memory/buffers.py` (extend `DocumentSummaryMemory` to store outlines), new tool wrapper referencing it, and a periodic job in `DocumentBridge` (or the editor widget) that refreshes summaries when the file crosses size thresholds.

### 5. Retrieval-augmented focusing via embeddings
**Status:** ✅ Delivered in Phase 3 (`DocumentEmbeddingIndex` + `DocumentFindSectionsTool` landed with guardrail-aware controller updates).
- **Problem:** The current toolset only supports regex search. Semantic lookups ("find the paragraph talking about quotas") force the model to read everything.
- **Proposal:** Index the document into embeddings (can reuse OpenAI embeddings via `AIClient` or a local model). Provide a `DocumentFindSectionsTool` that, given a query, returns top-k passages (chunk ids + text). Combine with the chunk-aware snapshot so the agent only sees relevant passages.
- **Key touches:** New service or extension to `services.importers` for indexing, a lightweight vector store (FAISS / sqlite w/ cosine), and a tool shim under `ai/tools`. Cache embeddings per version hash to avoid recompute for each prompt.

### 6. Automatic plan/summarize loop for long tool traces
**Status:** ✅ Delivered in Phase 2 (TraceCompactor GA plus controller logic to summarize oversized tool traces).
- **Problem:** Tool traces themselves accumulate; the controller appends every diff_builder or document_edit result verbatim.
- **Proposal:** After each tool round, if the accumulated tool trace exceeds X tokens, call a local summarizer (can be a minimal on-device model or heuristics) to compress earlier tool outputs into a "plan log" message. Keep the detailed trace separately in UI logs but feed only the summary back to the LLM.
- **Key touches:** `AIController.run_chat` loop (when `tool_iterations` increments), new helper to replace chunks of `conversation` with summaries while retaining the raw trace returned to the UI need (through `executed_tool_calls`).

### 7. Streaming diff construction for mega-edits
- **Problem:** `DocumentApplyPatchTool` reads the entire baseline text to build a diff even if the edit touches one span, which doubles memory use on very large files.
- **Proposal:** Introduce a streaming diff builder that reads only the targeted range (via chunk tool) and stitches changes without materializing the whole document. For multi-range edits, accept a list of disjoint ranges and merge them server-side before handing the diff to `DocumentEdit`.
- **Key touches:** new helper in `ai/tools/diff_builder.py` (or a sibling module) plus bridge support for extracting ranges without copying the full buffer.

### 8. Guardrail prompts encouraging selective reads
- **Problem:** The current `PATCH_PROMPT` instructs the agent to "Always fetch a fresh DocumentSnapshot" before editing, effectively guaranteeing a full-buffer dump.
- **Proposal:** Update the base system prompt to say "Prefer `DocumentSnapshot` in `delta_only` mode (selection + metadata). Use `DocumentChunk` / `DocumentOutline` tools to pull additional context." Combine this with telemetry (warn in chat when a tool output exceeded, say, 20K tokens) so the agent quickly retries with lighter data.
- **Key touches:** `prompts.py`, new controller telemetry hook, and UI messaging so users understand why the assistant may request multiple chunk pulls instead of one monolithic snapshot.

### 9. Manager + subagents for long-form tasks
**Status:** ✅ Delivered in Phase 4 (flagged for opt-in). `AIController` now coordinates scoped helper jobs via `SubagentManager`, caches summaries with `SubagentResultCache`, and emits `subagent.turn_summary` telemetry plus trace-compacted scouting reports.
- **Problem:** Even with chunk-aware tools, a single agent still has to ingest every chunk sequentially, and the conversation history accumulates summaries plus raw snippets, bloating the context window on multi-chapter documents.
- **Proposal:** Introduce a coordination pattern where the main agent delegates work to short-lived "subagents"—each subagent receives a specific chunk (via `DocumentChunkTool`), performs a focused task (summarize, extract entities, validate), and returns a compact report. The manager agent then synthesizes these partial results, keeping only the compressed summaries in its context. Cache per-chunk outputs keyed by version hash so repeated prompts can reuse prior work.
- **Key touches:** Extend `AIController` to spawn scoped worker instances (potentially with cheaper models) that have access only to chunk tools and validation. Define a `SubagentJobTool` contract for queueing chunk tasks and collecting results, store artifacts in `DocumentSummaryMemory`, and add safeguards so the manager re-runs subagents whenever a chunk's version hash changes.

### 10. Character/entity concordance toolchain
**Status:** 🟡 Partial Phase 4 delivery. Plot scaffolding + `DocumentPlotStateTool` expose cached entities/metadata, but alias tracking, concordance browsing, and edit planners remain future work.
- **Problem:** Creative-writing edits often target a specific character (e.g., "make Talia more sarcastic") which requires finding every chunk where the character appears and applying consistent tweaks. Manual regex search doesn’t capture aliases, pronouns, or indirect mentions.
- **Proposal:** Build a `CharacterMapTool` that maintains an entity index per document: names, aliases, pronouns, first/last appearances, and exemplar quotes. When invoked, it returns chunk ids plus context snippets where the entity shows up. Pair it with a `CharacterEditPlanner` subagent that walks the list, proposes edits, and tracks which chunks have been updated to guarantee coverage.
- **Key touches:** Extend the bridge/indexer to run a lightweight NER pipeline (spaCy or transformer-based) whenever the doc hash changes, storing results alongside chunk metadata. Add UI affordances to pick entities, and teach the agent prompt to call the concordance tool before making character-wide edits so it can apply diffs per chunk without rereading the entire manuscript.

### 11. Storyline consistency orchestration
**Status:** 🟡 Foundations in place (Phase 4 plot state store + controller hints). Need dependency graphs and planner loops before calling this done.
- **Problem:** When restructuring a long narrative, edits to chunk A (e.g., foreshadowing, character arc decisions) must ripple into later scenes. Without a stateful storyline model, the agent may produce contradictory changes across chunks.
- **Proposal:** Introduce a `PlotStateMemory` that records beats, timelines, and dependencies extracted from each chunk. During a rewrite, the manager agent consults this state before editing the next chunk, ensuring new instructions respect previous changes. After each edit, the chunk-level subagent updates the plot state (e.g., "Chapter 4 now ends with cliffhanger"), and the manager uses those notes when planning the following chunk.
- **Key touches:** New memory module (e.g., `memory/plot.py`) storing structured beat sheets, plus tools `PlotOutlineTool` and `PlotStateUpdateTool`. Update the prompt contract to require that storyline-wide tasks run in a loop: fetch plot state → edit chunk via chunk tool → apply patch → update plot state → repeat. Add guardrails so if the plot state indicates an unresolved dependency, the agent explicitly revisits affected chunks.

### 12. Preflight analysis + tool recommendation layer
- **Problem:** The agent currently decides which tools to call purely from conversation context, so it often pulls overly large snapshots or misses specialized tools (search, validation, character map) that would have improved the plan.
- **Proposal:** Add a lightweight "analysis pass" before the main run. An `AnalysisAgent` (or tool) inspects the prompt/document metadata, runs quick heuristics (document size, genre, pending diffs, entity index health), and emits structured recommendations: required chunk profile, suggested tools, caution flags. The main agent ingests this as an extra system message or metadata, letting it tailor its tool loop without burning context tokens on trial-and-error.
- **Key touches:** New module `ai/analysis` housing pluggable analyzers (rule-based + optional small LLM). Extend `AIController` to call the analyzer before `_build_messages`, injecting its output into the system prompt metadata. Provide tooling like `ToolUsageAdvisorTool` that the agent can invoke mid-turn to reassess if conditions change (e.g., doc size ballooned). Surface the analyzer’s verdict in the UI so users understand why the agent picked certain tools or chunk sizes.

## Next steps
1. Prototype the windowed snapshot + chunk tool (Ideas #1-2) since they deliver the biggest token savings and remain unsolved.
2. Build on the Phase 4 helper/plot-state bedrock by delivering full concordance + storyline orchestration (Ideas #10-11) and exposing that data in the UI.
3. Explore the preflight analysis layer (Idea #12) to help the agent choose between outline, retrieval, plot-state, and upcoming chunk tools without wasting tokens.
4. Once chunk tooling is live, iterate on helper orchestration (Idea #9 follow-ups) to stream chunk queues, reuse cached subagent results, and graduate the feature from flag-only to GA.
