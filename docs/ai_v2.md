# AI v2 – Phase 0 implementation notes

Phase 0 introduces the shared infrastructure that every subsequent AI milestone depends on. This document summarizes what shipped, how to exercise it, and where to hook follow-on work.

## Phase 3 – Outline & Retrieval Quickstart

Phase 3 layers guardrail-aware outline/retrieval tooling on top of the Phase 0/1/2 foundation. Use this section as the TL;DR before diving into the historical notes below.

### Feature highlights

- **DocumentOutlineTool vNext** – Returns nested headings, blurbs, pointer IDs, digest hashes, and guardrail metadata (`status`, `guardrails`, `trimmed_reason`, `retry_after_ms`).
- **DocumentFindTextTool** – Queries the embedding index (OpenAI or LangChain) and falls back to regex/outline heuristics when offline. Responses now flag `offline_mode`, `fallback_reason`, and per-pointer outline context.
- **Controller guardrail hints** – `AIController` injects `Guardrail hint (…)` system messages whenever a tool reports pending, unsupported, stale, trimmed, or offline states so the agent must acknowledge constraints before making edits.
- **Telemetry** – Outline/retrieval events record latency, cache hits, and `tokens_saved`, and the trace compactor continues to pointerize oversized payloads whenever the budget policy demands it.

### Setup checklist

1. **Enable the flag** – Toggle **Settings → Experimental → Phase 3 outline tools** (or launch with `--enable-phase3-outline-tools`).
2. **Pick an embedding mode/backend** – In **Settings → AI → Embeddings** choose a mode (`Same API`, `Custom API`, `Local`) and backend (`Auto/OpenAI`, `LangChain`, `SentenceTransformers`, or `Disabled`). CLI/env overrides remain available for the backend (`--embedding-backend`, `TINKERBELL_EMBEDDING_BACKEND`).
3. **Install the embeddings extra when needed** – Local `SentenceTransformers` mode depends on PyTorch. Run `uv sync --extra embeddings` (or `pip install -e '.[embeddings]'`) before selecting the mode so `sentence-transformers`, `torch`, and `numpy` are available.
4. **Seed samples** – Open any file under `test_data/phase3/` (see that folder’s README) to exercise the workflows without hunting for fixtures.
5. **Watch guardrails** – When the controller inserts a guardrail hint, restate it to the user and follow the suggested remediation (work in chunks, wait for pending outlines, disable offline fallbacks, etc.).

### Sample walkthrough (`test_data/phase3`)

| File | Purpose | Suggested prompt |
| --- | --- | --- |
| `stacked_outline_demo.md` | Deeply nested headings for outline/retrieval smoke tests. | “Summarize the Control Plane goals and cite pointer IDs.” |
Open a sample, ask the agent to outline or retrieve, and observe the tool payload plus guardrail hint that follows. The existing megabyte-scale fixtures (`5MB.json`, `War and Peace.txt`, `1MB.json`) remain useful for large-document depth limiting and pending rebuild tests.

### Guardrail & troubleshooting matrix

| Tool status / hint | What it means | Next step |
| --- | --- | --- |
| `guardrails[].type = "huge_document"`, `trimmed_reason = token_budget` | Outline capped depth/size. | Work pointer-by-pointer (DocumentFindTextTool + DocumentSnapshot) and mention the guardrail in your reasoning. |
| `status = "offline_fallback"`, `offline_mode = true` | Embedding provider is offline, retrieval switched to heuristics. | Treat previews as hints, rehydrate via DocumentSnapshot before editing, and restore connectivity/backend when possible. |
| `is_stale = true` | Cached outline digest no longer matches the current document version. | Trigger a rebuild (wait or poke the worker) and avoid committing edits that rely solely on stale headings. |

Additional reproduction ideas plus prompt language live in `test_data/phase3/README.md`.

## Caret-free AI tools (Workstream 1)

Workstream 1 removes every caret/selection mutation hook from the AI editing stack. Agents are now required to describe edits in terms of explicit ranges or `replace_all=true`, and the runtime refuses to infer intent from the live caret.

### API changes

- `document_apply_patch` and the inline `document_edit` auto-convert path now reject edits that omit `target_span` (or legacy `target_range`), `match_text`, or `replace_all=true`. Snapshot `selection_text` alone no longer counts as an anchor; callers must provide spans or match text from the captured window.
- Snapshot metadata (`text_range`) and chunk manifests now emit authoritative span hints (`snapshot_span`, chunk pointer ranges) for every window. The `selection_range` tool remains the fallback for inspecting the live caret when those hints are missing or the controller explicitly requests it; it still returns `{start_line, end_line, content_hash}` tied to the current snapshot so partners can persist their own fingerprints.
- DocumentBridge + EditorWidget now preserve the user’s caret/selection after every AI edit or diff application. The bridge always calls into the adapter with `preserve_selection=True`, eliminating the last caret mutation API.
- Caret-based schema fields (`insert_at_cursor`, `cursor_offset`, `selection_start`, `selection_end`) are rejected at normalization time. Deprecated callers trigger the `caret_call_blocked` telemetry event plus a validation error instructing them to refresh their snapshot.

### Migration checklist for partners

1. **Capture an authoritative snapshot** – Call `document_snapshot` (or the controller-provided stub) to grab `document_version`, `content_hash`, the surrounding text window, and the associated `snapshot_span`/chunk manifest ranges.
2. **Request live selection data only when spans are missing** – Use the spans baked into `text_range`, `snapshot_span`, or chunk manifests whenever they cover the requested window. Call `selection_range` only if the controller issues a fallback hint or the manifest lacks the required range; keep pairing its `{start_line, end_line, content_hash}` payload with the snapshot digest when you need to round-trip fingerprints.
3. **Build edits with explicit spans** – Convert model output into a diff by supplying either
  - `target_span` (`{start_line, end_line}`) copied from the snapshot/manifest span hints (preferred) or from the `selection_range` tool when hints are unavailable,
  - legacy `target_range` (`[start, end]` offsets) when spans are unavailable (the compatibility adapter still accepts them),
  - `match_text`/`expected_text` anchors extracted from the snapshot window, or
  - `replace_all=true` for full-document rewrites (requires hashing the entire snapshot).
4. **Attach deterministic anchors** – Provide the `target_span` derived from snapshot/manifest hints (or, as a fallback, returned by `selection_range`) or include `match_text`/`expected_text` snippets from the snapshot window so the bridge can relocate the slice deterministically.
5. **Handle guardrails** – When the runtime raises `caret_call_blocked` or `Edits must include target_span...`, refresh the snapshot/selection pair and rebuild the edit rather than retrying with caret metadata.
6. **Delete spans intentionally** – When the desired result is removal, send `content` as an empty string alongside the same `target_span`/scope metadata you would provide for replacements. Any other whitespace-only payloads are still rejected so accidental blanks do not sneak through.

### Partner changelog

- **Caret preservation** – `DocumentBridge` now applies patches with `preserve_selection=True`, so the UI never jumps to the AI edit location. Any downstream automation that previously looked for caret jumps must switch to telemetry (`patch.anchor`) or diff spans for auditing.
- **Span hint adoption** – Snapshot metadata and chunk manifests now provide the preferred `{start_line, end_line}` ranges. Store those spans alongside `document_snapshot` output and fall back to `selection_range` only when spans are missing or the controller explicitly asks for it (see the checklist above for ordering).
- **Telemetry signals** – Monitor `caret_call_blocked` vs. `span_snapshot_requested` to confirm all partners have migrated. Dashboards should alert if blocked calls remain once the feature flag is forced on.

#### Selection gateway + ownership guardrails

- **`SelectionGateway` facade** – Editor internals now expose a dedicated `editor.selection_gateway.SelectionGateway` that encapsulates the live caret/selection state plus line offsets. `SelectionRangeTool` depends on this facade, so AI callers never touch raw `SelectionRange` objects or document snapshots to determine spans.
- **Gateway-only consumers** – Any module that needs live selection data must request a `SelectionSnapshotProvider` (the gateway or a stub) and operate on its immutable payloads. Direct imports of `SelectionRange` outside `src/tinkerbell/editor/*` and `ai/tools/selection_range.py` are forbidden; use tuples/TextRange inputs when constructing `DocumentState` instead.
- **CI enforcement** – `tests/test_selection_guard.py` scans the tree and fails the suite if a new `SelectionRange` import sneaks into disallowed packages. Update or extend the gateway rather than bypassing it.
- **Testing guidance** – Tests that require spans should stub `SelectionSnapshotProvider` (see `_SelectionGatewayStub` in `tests/test_ai_tools.py`) or call `SelectionRangeTool` itself. Do not fabricate selection tuples from cached snapshots; instead, pass span dictionaries (`{"start": x, "end": y}`) into bridge stubs via helpers such as `_selection_span()` / `_document_span()` so fixtures mirror the span data that `window`/`text_range`/`snapshot_span` now carry.

##### Developer onboarding: requesting spans in tests

1. Prefer the spans already present in `DocumentSnapshot` (`text_range`, `snapshot_span`) and chunk manifest entries when assembling fixtures. Reserve `SelectionRangeTool` (or `SelectionGateway` stubs) for tests that explicitly validate fallback behavior.
2. When a test has to embed spans inside mock bridges, copy the snapshot/manifest values into helpers such as `_selection_span(start, end)` so `window`/`text_range` defaults (and `snapshot_span` for search/replace) stay faithful to real payloads.
3. Avoid importing `SelectionRange` outside editor-focused tests; controller/AI suites should rely on span helpers, chunk manifests, or (when unavoidable) gateway snapshots so SelectionRangeTool remains the single read-only fallback rather than the default span provider.

### Telemetry & monitoring

- `caret_call_blocked` surfaces every legacy caret attempt (name, tab, range metadata, reason) so dashboards can spot lagging clients.
- `span_snapshot_requested` fires when the new `selection_range` tool runs; compare it with `caret_call_blocked` to confirm migrations are complete.
- `patch.anchor` now reports `anchor_source = match_text | fingerprint | range_only`, making it obvious when clients omit fingerprints and forcing them through explicit anchors.

Rollouts should watch the `caret_call_blocked` → `span_snapshot_requested` ratio decline before flipping the feature flag to 100%.

## Workstream 11 – Scope provenance & telemetry enforcement

Workstream 11 closes the loop on “chunk or explicit range” guarantees. Every edit now carries forward the exact span/chunk data the agent captured so the bridge, controller, and telemetry can prove that edits were grounded in deterministic context.

### Runtime requirements

- **Scope metadata is mandatory** – `DocumentApplyPatch` and `DocumentEdit` requests must include a `scope` block for every streamed span: `{origin, range, length, chunk_id?, chunk_version?, snapshot_span?}`. `origin` is one of `chunk`, `explicit_span`, or `document`; `range` is the `{start_line, end_line}` window copied from the snapshot/chunk manifest; `length` is the total lines covered.
- **Bridge validation** – `DocumentBridge` rejects edits that omit scope metadata or send ranges outside the claimed snapshot. The bridge summarizes scopes per streamed edit (`scope_summary`) and emits it with success/failure telemetry so dashboards can audit provenance.
- **Controller propagation** – `AIController` extracts `scope_summary` from tool arguments, stamps it onto tool records, and feeds it into retry events (`document_edit.retry`) plus `needs_range` payloads. `span_hint` messages now prefer metadata-provided spans before issuing SelectionRangeTool fallbacks.
- **SelectionRangeTool remains last resort** – Agents (and downstream partners) may only call SelectionRangeTool when snapshots/chunk manifests cannot supply a span and the controller injects an explicit fallback hint. All other edits must reference the captured chunk/span.

### Agent + partner expectations

- **Capture + replay scope** – Whenever a snapshot or chunk manifest is hydrated, immediately copy the provided `text_range`/`chunk.range` into `target_span` and persist a matching `scope` entry. Send that same metadata with every downstream edit, even after multiple planning steps.
- **Never guess spans** – If an edit fails with `needs_range`, refresh the snapshot, rehydrate the manifest, and resend the explicit span. Do not fabricate offsets or ask SelectionRangeTool unless the controller tells you to.
- **Explain provenance in reasoning** – Prompts now instruct agents to mention whether an edit is chunk-backed (`scope_origin=chunk:<chunk_id>`) or explicit span-backed. This mirrors what telemetry captures and gives operators immediate visibility into compliance.
- **Tooling alignment** – Custom agents must echo the same behavior: persist spans from manifests, attach `scope` blocks to edit calls, and retry only after grabbing deterministic windows.

### Telemetry additions

- `patch.apply`, `patch.range_conflict`, `patch.hash_mismatch`, and `patch.rejected` now export `scope_summary` plus per-span fields (`scope_origin`, `scope_length`, `scope_range`). These show up in `docs/operations/telemetry.md` and downstream dashboards for auditing.
- `document_edit.retry` payloads include the failing scope metadata so incidents can be correlated with specific spans/chunks.
- `needs_range` telemetry reports the hinted span (`span_hint`, `span_hint_reason`) along with the previous scope summary, making it obvious when callers skipped chunk manifests.

### Troubleshooting quick reference

| Signal | Meaning | Recovery |
| --- | --- | --- |
| `scope_required` validation error | Edit lacked a `scope` block or claimed `scope_origin="chunk"` without a manifest ID. | Rehydrate the snapshot/chunk manifest, attach `{origin, range, length}` (and chunk IDs when relevant), then resend. |
| `needs_range` with `span_hint_reason=missing_scope` | Controller could not locate deterministic spans in the previous payload. | Grab a fresh snapshot, copy its `text_range` to `target_span`, include matching `scope`, and retry. |
| Telemetry shows `scope_origin=document` spikes | Agents are defaulting to whole-document spans. | Revisit prompts/configuration so chunk manifests drive most edits; SelectionRangeTool should remain a last resort. |
| `document_edit.retry` includes `scope_origin=document` then `scope_origin=chunk` | Automatic retry refreshed the scope successfully. | No action required; use this as proof the controller recovered by rehydrating the chunk manifest. |

These rules keep edits auditable and allow telemetry dashboards to trace every applied diff back to the exact window the agent inspected.

## Workstream 3 – Snapshot + plot-state validation

Workstream 3 hardens the diff pipeline by forcing every edit to reference a specific snapshot and by surfacing plot-outline drift the moment an edit drops tracked entities or beats.

### Schema & runtime changes

- `document_apply_patch` and `document_edit` now require the trio `{document_version, version_id, content_hash}` on every request. Missing fields raise `ValueError("…is required; call document_snapshot…")`; mismatches raise `DocumentVersionMismatchError` with `cause="hash_mismatch"` so callers know to refresh metadata.
- `DocumentSnapshotTool` always returns these fields (and computes `version_id`/`content_hash` when the bridge omits them) so partners can stash a single payload and replay it into subsequent edit calls.
- When plot scaffolding is enabled, `DocumentApplyPatchTool` asks `DocumentPlotStateStore` for the latest entity/beat roster before and after the edit. If a tracked entry disappears, the tool appends `warning: plot_outline_drift …` to the status string and emits `plot_state.warning` telemetry.
- The controller’s version-mismatch retry path now refreshes `document_snapshot`, injects the new metadata into the original call, and retries once automatically. If the second attempt still fails, the caller receives the explicit “stale even after an automatic retry” error.

### Client retry protocol

1. **Capture authoritative metadata** – Call `document_snapshot` immediately before a patch/edit. Persist `{document_version, version_id, content_hash, selection_hash}` alongside the planned `target_span` (preferred), fallback `target_range`, or `match_text` anchors.
2. **Send metadata back verbatim** – Pass those values into `document_apply_patch`/`document_edit`. Do not trim whitespace or coerce numbers into other types; hashes and version tokens must match exactly.
3. **Handle mismatch errors deterministically** – If you receive `DocumentVersionMismatchError` (or see the controller log a retry), drop the in-flight edit, refresh `document_snapshot`, rebuild your diff from the new payload, and resubmit it with the updated metadata. Never reuse hashes from a previous attempt.
4. **Honor controller retries** – When the controller reports “stale even after an automatic retry”, surface that guidance to the agent/operator. The controller already used a fresh snapshot; the only safe recovery is to re-run `document_snapshot` (optionally `selection_range`) and regenerate the edit instructions.

### Plot-state warning handling

- Watch for `status` strings ending in `warning: plot_outline_drift (entities=…, beats=…)`. Bubble these warnings into your UX (toast, inline banner, etc.) so authors understand which tracked characters/beats vanished.
- Offer a single-click remediation: fetch `document_plot_state` to review the latest outline and decide whether to restore the missing entity or intentionally accept the change. If the user confirms the removal, update the plot store via `plot_state_update` so future edits stop flagging it.
- Log `plot_state.warning` and `document_edit.retry` telemetry alongside your own audit trail so ops teams can trend drift frequency versus retries.

### Testing checklist

- `tests/test_document_apply_patch.py` exercises hash/ID enforcement and plot warning propagation. Extend these tests whenever you add new metadata fields.
- `tests/test_editor_widget.py::test_ai_rewrite_turn_retries_on_stale_snapshot` verifies the controller retry path end-to-end.
- For manual QA, run a turn that removes a tracked character from `DocumentPlotStateStore`; confirm the edit status includes the warning and that telemetry captures the `entities`/`beats` arrays.

## Phase 4 – Character & plot scaffolding (experimental)

Phase 4.3 adds an opt-in memory layer that captures lightweight character/entity and plot beat summaries from subagent jobs. The goal is to give the controller+agents continuity hints without persisting sensitive data or bloating prompts.

### Components


### Usage & limitations

1. Enable Phase 4 subagents and plot scaffolding in the settings dialog (or via CLI/env overrides) and restart the controller if it was already running.
2. Trigger a chat turn that causes the controller to spawn a subagent job (e.g., select >400 chars and ask for a scoped analysis). After the helper finishes, watch for the injected hint acknowledging that scaffolding refreshed for the document.
3. Within the same turn, the agent can call `document_plot_state` to retrieve the cached roster before drafting edits. The tool accepts `document_id`, `include_entities`, `include_arcs`, `max_entities`, and `max_beats` arguments to trim payloads when token budgets are tight.
4. In Phase 4 the store was transient—closing a document or emitting a `DocumentChanged`/`DocumentClosed` event wiped its entry. Phase 5 (see below) introduces optional persistence for operator overrides.

## Phase 5 – Storyline continuity orchestration (experimental)

Phase 5 builds on the same plot scaffolding flag to keep long-form edits narratively coherent. When the flag is enabled, the controller enforces a `plot_outline → document_edit/document_apply_patch → plot_state_update` loop before it allows the agent to wrap a turn.

### PlotStateMemory, overrides, and persistence

- `PlotStateMemory` replaces the transient store with dependency tracking, version metadata, and operator overrides. It subscribes to `DocumentCacheBus` events so stale documents are purged automatically.
- Manual overrides and dependency notes persist to `~/.tinkerbell/plot_overrides.json`. Each entry records `override_id`, summary text, optional `arc_id`/`beat_id`, author, and timestamps so human continuity decisions survive restarts.
- `PlotOutlineTool` (alias `DocumentPlotStateTool`) now hydrates enriched snapshots from `PlotStateMemory.snapshot_enriched()` including overrides, dependencies, and `version_metadata`.

### Enforced plot loop

1. **Read** – `_PlotLoopTracker` blocks `document_edit`/`document_apply_patch` if the agent has not called `plot_outline`/`document_plot_state` first, returning a guardrail hint that explains how to recover.
2. **Edit** – Successful edit calls mark the document as “pending update”. The very next planner request injects a system reminder instructing the agent to run `plot_state_update` before it can finish the turn.
3. **Update** – Once `plot_state_update` succeeds, the tracker clears the pending flag and emits a confirmation hint so operators know the continuity metadata is back in sync.

Disable the plot scaffolding flag to bypass this enforcement entirely (useful for smoketests or legacy sessions).

### Telemetry & observability

- `plot_state.read` fires whenever `PlotOutlineTool` runs, reporting `document_id`, entity/arc counts, override totals, and dependency totals.
- `plot_state.write` records every `PlotStateUpdateTool` call with deltas plus the persistence status for `plot_overrides.json`.
- Tool traces include `plot_loop_blocked` outcomes, and the chat panel mirrors the guardrail hints so operators can see when the agent skipped the enforced loop.

### Validation


## Phase 5 – Preflight analysis & tool recommendations

Phase 5.2 layers a rule-based analyzer (`tinkerbell.ai.analysis`) on top of the chunk/plot/concordance metadata so the controller can select the right tool mix before each turn.

### Analyzer + cache lifecycle

- `AnalysisAgent` ingests `AnalysisInput` snapshots generated by `AIController._build_analysis_input()` and emits structured `AnalysisAdvice` records (chunk profile, required/optional tools, outline refresh flag, warnings, trace metadata, cache state).
- Advice now travels with every chat turn (`AIController._analysis_hint_message`), manual `/analyze` command, and telemetry export via `TelemetryManager.emit_context_usage()` (`analysis_chunk_profile`, `analysis_required_tools`, `analysis_warning_codes`, etc.).
- The TTL cache subscribes to `DocumentCacheBus` events, so `DocumentChangedEvent`/`DocumentClosedEvent` immediately invalidate cached advice and drop stale snapshots; manual edits no longer reuse outdated recommendations.

### Operator interactions

- The status bar exposes a dedicated **Preflight** badge that summarizes the latest advice, while the chat panel mirrors it with hover text (chunk profile, required tools, warnings).
- Operators can run `/analyze [--doc ...] [--force-refresh] [--reason note]` to rerun the analyzer on demand. The helper leverages `AIController.request_analysis_advice()`, posts a formatted notice, refreshes badges, and records the run via `analysis.ui_override.*` telemetry.
- LangGraph planners call `ToolUsageAdvisorTool` when they need fresh guidance mid-turn; the tool bridges into `_advisor_tool_entrypoint()` and returns the serialized advice payload.

### Telemetry & exports

- `analysis.advisor_tool.invoked` fires whenever the tool surface runs, capturing `document_id`, selection overrides, force-refresh flag, and whether a cached snapshot was reused.
- `analysis.ui_override.requested/completed/failed` events trace manual `/analyze` executions along with reasons, cache states, and resulting tool lists so dashboards can flag repeated operator overrides.
- `scripts/export_context_usage.py` now prints the analysis columns alongside the existing outline/retrieval metrics, so CSV exports include chunk profile, tool lists, warnings, cache state, and rule traces for each turn.

### Phase 4.4 – Integration, telemetry, hardening

- **Registry + protocols** – `TokenCounterRegistry` and `TokenCounterProtocol` live in `tinkerbell.ai.client` / `tinkerbell.ai.ai_types`. Every `AIClient` registers a counter for its active model and falls back to a deterministic byte estimator when no backend is available.
- **Backends** – `TiktokenCounter` is used when the optional [`ai_tokenizers`](../pyproject.toml) extra (`tiktoken>=0.12,<0.13`) can be installed. Otherwise `ApproxByteCounter` provides a predictable bytes-per-token approximation so counts are still monotonic and reproducible.
- **CLI helper** – `scripts/inspect_tokens.py` consumes stdin or `--file` input and prints both precise (tiktoken) and estimated counts. Example:
  ```powershell
  uv run python -m tinkerbell.scripts.inspect_tokens --file "test_data/Carmilla.txt"
  ```
  Pass `--estimate-only` to skip `tiktoken` even when it is installed.
- **Windows / Python 3.13 heads-up** – `tiktoken` does not yet ship wheels for Python 3.13. Installing the optional extra on this interpreter requires a Rust toolchain (`rustup` on Windows). Until that is available, the registry logs a single warning and keeps using `ApproxByteCounter`.

## 2. Context usage instrumentation

- **Event schema** – `ContextUsageEvent` (in `tinkerbell.ai.services.telemetry`) captures `document_id`, `model`, `prompt_tokens`, `tool_tokens`, `response_reserve`, `conversation_length`, `tool_names`, a monotonic timestamp, and the active embedding backend/model/status tuple so downstream dashboards can segment runs.
- **Collection points** – `AIController` (`ai.orchestration.AIController`) emits an event per turn and aggregates tool invocations. Tool payload sizes are counted via the new token registry so metrics stay consistent.
- **Settings & UI** – The Settings dialog now exposes **Debug → Token logging enabled** and **Token log limit**. When enabled, the status bar shows running totals and the in-memory sink keeps the last *N* events for inspection/test assertions.
- **Programmatic access** – `AIController.get_recent_context_events()` exposes the rolling buffer for tests or external dashboards. Additional sinks can be registered via `TelemetrySink` to stream events elsewhere.
- **Export script** – `uv run python -m tinkerbell.scripts.export_context_usage --format csv --limit 50` dumps the persisted buffer (JSON/CSV) from `~/.tinkerbell/telemetry/context_usage.json` for audits or support bundles.
- **Outline/Retrieval telemetry** – Outline builder (`ai/services/outline_worker.py`), `DocumentOutlineTool`, `DocumentFindTextTool`, and `DocumentEmbeddingIndex` now emit structured events (`outline.build.start/end`, `outline.tool.hit/miss`, `outline.stale`, `retrieval.query`, `retrieval.provider.error`, `embedding.cache.hit/miss`) that include latency, cache hit counts, provider names, and `tokens_saved` deltas. Dashboards can subscribe via `TelemetrySink` or reuse `scripts/export_context_usage.py` to trend outline freshness, retrieval performance, and embedding cost spikes.
- **Scope provenance metrics** – Context usage events now include per-turn scope aggregates (`scope_origin_counts`, `scope_missing_count`, `scope_total_length`). Use them to confirm chunk-first editing is enforced (chunk/explicit spans should dominate, `scope_missing_count` should remain zero) and to spot suspicious full-document rewrites when `scope_total_length` spikes.

## 3. Document version IDs & optimistic patching

- **Document metadata** – `DocumentState` now tracks `document_id`, `version_id`, and `content_hash`; `document.snapshot()` and `DocumentBridge.generate_snapshot()` always include these fields along with a `version` token of the form `"{document_id}:{version_id}:{content_hash}"`.
- **Bridge enforcement** – `DocumentBridge.queue_edit()` rejects stale directives (raising `DocumentVersionMismatchError`) and annotates edits with selection hashes so callers must refresh snapshots when conflicts arise.
- **Tools** – `DocumentApplyPatchTool` insists on the current `document_version` before routing diffs through `DocumentEditTool`. Tests in `tests/test_bridge.py` and `tests/test_ai_tools.py` cover both the happy path and mismatch errors.

## 3b. Snapshot-anchored editing guardrails (Phase 5)

Phase 5 hardens diff tooling so every edit is anchored to an explicit snapshot slice. The goal is to stop legacy inline edits from duplicating paragraphs or inserting mid-word glitches when the document shifts between tool calls.

### Tool & schema changes

- `DocumentSnapshot` no longer exposes `selection_text` or `selection_hash`; callers must rely on the snapshot's `text_range`, `snapshot_span`, and chunk manifests to describe the intended span, falling back to `selection_range` only when those hints are missing.
- `DocumentApplyPatchTool` and `DocumentEditTool` accept `match_text` and `expected_text` anchors (plus the preferred `target_span`). Anchors are compared against the live document to relocate edits when offsets drift.
- The tool manifest plus system prompts (`src/tinkerbell/ai/prompts.py`) now instruct agents to copy spans and anchor text from `document_snapshot`/chunk manifests (and only reach for `selection_range` if spans are missing) before calling diff/edit tools.

### Recommended workflow

1. Call `document_snapshot` with `include_text=true` and capture `document_version`, the snapshot `text_range`/`snapshot_span`, and any chunk manifest ranges that cover the requested window (store offsets as a fallback `target_range`). Run `selection_range` only if those hints are missing and you need the live caret span.
2. When building a patch, include at least one of `target_span` or `match_text`. Prefer to send both (and optionally the legacy `target_range`) so the bridge can double-check offsets.
3. Copy the literal snippet from the snapshot window into `match_text` (and `expected_text` if the schema requires it for your caller) so the bridge can relocate the edit when offsets drift.
4. If the snapshot slice no longer matches the document, let the tools raise their guardrail errors and immediately refresh the snapshot instead of guessing offsets.
5. Reserve caret inserts (`target_range` start == end / `target_span` covering zero lines) for explicit `action="insert"` directives. Replace operations now require anchors or a non-empty span.

### Error handling quick reference

| Message fragment | Meaning | Next step |
| --- | --- | --- |
| `Edits must include target_span (preferred), target_range, match_text, or replace_all=true` | The edit lacked both a span and an anchor. | Re-run `document_snapshot` and send the captured `target_span` (or fallback `target_range`), include `match_text`, or set `replace_all=true` for full-document edits. |
| `Snapshot selection_text no longer matches document content` | The auto-anchor pulled from the snapshot was stale. | Provide explicit `match_text` from the user-visible context or fetch a new snapshot. |
| `match_text matched multiple ranges` | Anchoring text was ambiguous in the current document. | Narrow the target span (or provide exact offsets) and resend the edit with a more specific snippet. |
| `match_text did not match any content` | The document drifted beyond the provided snippet. | Refresh the snapshot before retrying the edit. |

### Telemetry & dashboards

- `patch.anchor` – emitted whenever `DocumentApplyPatchTool` or the inline `DocumentEdit` auto-convert path validates anchors. Payload highlights `status` (`success` or `reject`), `phase` (`requirements`, `alignment`, etc.), `anchor_source` (`match_text` or `range_only`), plus document/tab identifiers. Use this to trend anchor mismatch rates after rollout.
- `patch.apply` – emitted by `DocumentBridge` when a patch succeeds, conflicts, or arrives stale. Includes the duration in milliseconds, `range_count` (for streamed diffs), diff summary, and whether the bridge had to fall back because of a conflict. Dashboards can now plot success/conflict ratios directly from telemetry instead of scraping logs.

These events piggyback on the existing telemetry bus, so the status bar debug widgets and `scripts/export_context_usage.py` will surface them automatically once the sinks are subscribed.
Details for each payload (fields, status values, and sample diagnostics) now live in `docs/operations/telemetry.md` for downstream ingestion pipelines.

## 3c. Chunk-first selective read guardrails (Phase 5)

Phase 5 tightens the read flow so the agent stays on the "selection snapshot → chunk tool → outline/retrieval" path before touching edits.

- **Controller tracking** – `_ChunkFlowTracker` (inside `ai/orchestration/controller.py`) watches every `DocumentSnapshot`/`DocumentChunk` call. When the agent grabs a full snapshot without chunk hydration, it injects `Guardrail hint (Chunk Flow …)` system messages and emits telemetry.
- **Telemetry events** – New events `chunk_flow.requested`, `chunk_flow.escaped_full_snapshot`, and `chunk_flow.retry_success` ride the telemetry bus. Dashboards (or the status bar debug widgets) can subscribe to those names to trend how often the guardrail fires and how quickly agents recover.
- **UI badges** – The chat panel shows a `Chunk Flow` banner whenever a warning or recovery fires, mirroring the system hint. The status bar adds a matching badge so operators can monitor long-running sessions without scrolling through the transcript.
- **Recovery workflow** – When the badge reports `Chunk Flow Warning`, follow the guardrail hint: re-run `DocumentSnapshot` with a selection window or hydrate the manifest via `DocumentChunkTool`. Once the controller sees a chunk hydrate succeed, the badge flips to `Chunk Flow Recovered` until the next turn.
- **Testing** – `tests/test_chat_panel.py`, `tests/test_widgets_status_bar.py`, and `tests/test_telemetry_controller.py` cover the UI + telemetry plumbing, while the existing controller tests assert the hint injection logic.

## 4. Cache registry & invalidation bus

- **Pub/sub bus** – `tinkerbell.ai.memory.cache_bus` introduces `DocumentCacheBus` plus standard events (`DocumentChangedEvent`, `DocumentClosedEvent`). Subscribers can be strong or weak references, and helper classes (`ChunkCacheSubscriber`, `OutlineCacheSubscriber`, `EmbeddingCacheSubscriber`) log every notification.
- **Publishers** – `DocumentBridge` publishes change events after every directive/patch and a closed event when `DocumentWorkspace` shuts down a tab. Future caches (chunking, outline, embeddings) can subscribe without tight coupling to the editor.
- **Tests** – `tests/test_memory_buffers.py`, `tests/test_bridge.py`, and `tests/test_workspace.py` validate event ordering, weak-reference cleanup, and integration with the workspace router.

## 5. Validation & follow-up

- **End-to-end tests** – `uv run pytest` exercises the full suite (217 tests as of this phase).
- **Benchmarks & observations** – See `benchmarks/phase0_token_counts.md` for sample token measurements and environment notes. Re-run them whenever you add a new model entry or upgrade `tiktoken`.
- **Snippet validators** – `tinkerbell.ai.tools.validation.validate_snippet` now supports Markdown lint stubs (heading jumps, unclosed fences), optional JSON schema inputs, and `register_snippet_validator()` so downstream teams can plug in bespoke formats without touching the core registry.
- **Next steps** – Later AI v2 phases can now rely on deterministic budgets, telemetry hooks, and cache invalidation semantics. Typical extensions include persisting telemetry, wiring cache subscribers that build chunk/embedding stores, and surfacing version IDs in future API responses.

## 6. Settings, secrets, and overrides

- **Secret providers** – `SettingsStore` now delegates encryption to a pluggable provider: Windows DPAPI when available and Fernet everywhere else. The active backend is recorded inside `settings.json` (`secret_backend`) and can be forced with `TINKERBELL_SECRET_BACKEND=fernet|dpapi` for deterministic test runs.
- **Migration safeguards** – Legacy plaintext API keys are detected during load, re-encrypted with the current provider, and the file is rewritten at version `2` automatically so future CLI tooling can rely on consistent metadata.
- **Deterministic override order** – Settings are merged as **UI defaults → CLI `--set key=value` flags → environment variables**. CLI overrides accept ints/floats/bools plus JSON blobs for structured fields, letting you tweak `max_tool_iterations`, request timeouts, or metadata without touching disk.
- **Settings inspector** – Running `uv run tinkerbell --dump-settings` prints the fully merged configuration (API keys redacted), the resolved settings path, the secret backend, applied CLI overrides, and the `TINKERBELL_*` variables that influenced the run.
- **Custom locations** – Pass `--settings-path` when launching or dumping to point at alternate profiles (useful for smoke tests or portable builds). The CLI honors the same override order, so env vars still win if both are supplied.
- **Safe AI editing flag** – Guardrail inspections ship behind `safe_ai_edits` (default `False`). Toggle it in **Settings → Experimental → Safe AI edits**, or pass `--enable-safe-ai-edits` / `--disable-safe-ai-edits` on the CLI for one-off sessions. Environment overrides use `TINKERBELL_SAFE_AI_EDITS=1` along with `TINKERBELL_DUPLICATE_THRESHOLD` / `TINKERBELL_TOKEN_DRIFT` to tune the duplicate paragraph and token-drift thresholds. Every `DocumentBridge` instance picks up the flag at runtime (existing tabs reconfigure immediately), so QA can roll out guardrails gradually and disable them quickly if needed.

## 7. Desktop UX helpers (Phase 1)

- **Preview-first import/open dialogs** – `DocumentLoadDialog` replaces the native picker so every open/import run shows file size, inferred language, and token counts. The dialog highlights how much of the configured context budget a file would occupy and surfaces the first ~3k characters for sanity checks before loading a tab.
- **Document export dialog** – Saving a document now routes through `DocumentExportDialog`, which previews the start of the file, reports document-wide token totals, and keeps the token budget gauge visible so authors know when exports approach window limits.
- **Curated sample library** – A dropdown in the open dialog pulls Markdown/JSON/YAML fixtures from `test_data/` (and `assets/sample_docs/` when present) so smoke tests on large files are one click away. Selecting a sample immediately previews its contents and token footprint before creating the new tab.

## 8. Benchmarking + performance checkpoints

- **Token + diff baselines** – `benchmarks/phase0_token_counts.md` now tracks both the original tokenizer sanity checks and the new Phase 1 diff latency table. Run `uv run python benchmarks/measure_diff_latency.py` to reproduce the War and Peace / Large JSON runtimes cited there.
- **Automation hook** – The benchmark helper accepts `--case LABEL=PATH` (and `--json`) so CI jobs or future profiling scripts can extend the dataset without editing the file.
- **Pointer impact follow-up** – Sprint 3 enables pointer-driven compaction by default, and the benchmark harness now prints `diff_tokens`, `pointer_tokens`, and `tokens_saved` for every oversized tool payload. War and Peace’s 88K-token diff collapses to ~250 tokens, so the controller stays comfortably below the watchdog ceiling.
- **Patch pipeline timing** – `benchmarks/measure_diff_latency.py --mode pipeline` now spins up a `DocumentApplyPatchTool → DocumentBridge` flow (with and without safe edits) so we can track guardrail overhead. Use `--duplicate-threshold` and `--token-drift` to mirror QA settings, or `--mode diff` when you only need the legacy baseline.

## 9. Sprint 2 – Summaries & pointers

Phase 2 Sprint 2 builds on the dry-run budget policy by actually compacting oversized tool payloads. During this sprint the feature stayed behind the existing `context_policy.enabled` flag (defaulted to dry-run); Sprint 3 later flipped the default on for GA. Sprint 2 hinges on three pillars:

1. **Deterministic summarizer module** (`tinkerbell.ai.services.summarizer`)
  - Handles plaintext + diff payloads with conservative heuristics (line clamps, per-hunk stats, bullet synthesis) so results are reproducible in tests.
  - Emits `SummaryResult` records that report both estimated token savings and the pointer metadata needed downstream.
  - Tools can opt out by setting `summarizable = False` on their callable or dataclass; validators do this so lint findings never shrink.

2. **Pointer-aware chat schema** (`tinkerbell.chat.message_model.ToolPointerMessage`)
  - When `ContextBudgetPolicy.tokens_available()` returns `needs_summary`, the controller swaps the original tool response for a lightweight pointer message.
  - Pointer text explains why the payload shrank, includes key metadata (tool name, document/version IDs, diff stats), and always ends with explicit rehydration instructions for LangGraph agents.
  - Raw payloads are still preserved inside `executed_tool_calls` for UI expansion, export scripts, and audits.

3. **Controller + prompt integration** (`tinkerbell.ai.orchestration.AIController` & `ai/prompts.py`)
  - `_handle_tool_calls` records per-tool `summarizable` flags, feeds payloads to the summarizer, and caches pointer instructions alongside tool traces.
  - `_compact_tool_messages` retries the budget calculation after each summarization while suppressing duplicate telemetry so dashboards stay readable.
  - Prompt templates now brief the agent on pointer semantics (“If you see a pointer message, rerun the referenced tool with the provided parameters to fetch the full data”).

### Usage notes

- **Settings & telemetry** – No new toggles were added; reuse `context_policy.enabled` and the response reserve fields from Phase 1. Telemetry gains `summary_count` + `tokens_saved` deltas per turn, visible in the status bar debug counters.
- **Tests** – `tests/test_ai_tools.py` and `tests/test_agent.py` contain regression coverage for summarizer budgets, pointer serialization, and non-summarizable tools. Run `uv run pytest -k pointer` for a focused sweep.
- **Manual validation** – Load `test_data/War and Peace.txt`, ask the Workspace Agent for a snapshot, and confirm the tool entry shows a pointer badge. Clicking it in the chat panel or invoking “Show full payload” should rehydrate the original diff.

Sprint 3 completes the experience with a dedicated trace compactor service and UI badges everywhere pointers appear (see §10).

## 10. Sprint 3 – Trace compactor GA

Sprint 3 flips the context policy + trace compactor stack to General Availability. Oversized tool responses are now pointerized automatically, the UI surfaces compaction badges, and telemetry/status widgets expose savings without requiring debug flags.

1. **TraceCompactor service** (`tinkerbell.ai.services.trace_compactor`)
  - Maintains a rolling ledger of tool outputs, tracks token savings, and swaps entries for pointer summaries whenever the budget policy reports `needs_summary`.
  - Integrates with `AIController` so compaction happens off the main thread while raw payloads continue to flow to transcripts/export scripts.
2. **UI + transcript affordances** (`chat.chat_panel`, `main_window`, `services.bridge_router`)
  - Tool traces show “Compacted” badges plus pointer metadata, and the transcript/export paths retain the original payload for auditing.
  - Status bar text now appends `tokens_saved` / `total_compactions` counters so long-running sessions have immediate feedback.
3. **Telemetry + benchmarks** (`services.telemetry`, `benchmarks/measure_diff_latency.py`)
  - `trace_compaction` events capture `entries_tracked`, `total_compactions`, and `tokens_saved` for every turn. Tests assert the payloads so dashboards stay consistent.
  - The benchmark helper injects a massive diff per document, showing savings like “War and Peace: 88K diff tokens → 247 pointer tokens (88K saved) in 66 ms” to validate both latency and GA savings.
4. **Defaults flipped** (`services/settings.ContextPolicySettings`)
  - `context_policy.enabled` now defaults to `True` with `dry_run=False`, so fresh installs enforce the policy immediately. Users can still opt out or revert to dry-run via settings or environment overrides.

### Usage notes

- The Settings dialog highlights the active policy + compaction stats, and toggling “Dry run only” is now an explicit opt-in.
- Status bar and chat panel badges require no extra flags; pointer text always includes rehydrate instructions so LangGraph agents can fetch the raw data on demand.
- Telemetry exports (`scripts/export_context_usage.py`) include the latest `trace_compaction` snapshot alongside `context_budget_decision` events for auditors.

### Validation

- New tests `tests/test_trace_compactor.py`, `tests/test_chat_panel.py`, and `tests/test_main_window.py` cover ledger math, pointer metadata propagation, and status-bar stats.
- Full regression suites (`uv run pytest`) run green, and the refreshed benchmark numbers in `benchmarks/phase0_token_counts.md` document the <50 ms overhead requirement despite compaction.

## 11. Embedding runtime & LangChain backends (Phase 3 preview)

- **Feature flag** – Everything is gated behind `settings.phase3_outline_tools` (UI toggle or CLI `--enable-phase3-outline-tools`). When disabled, the embedding runtime tears down automatically and retrieval tools fall back to outline heuristics.
- **Mode + backend selection** – The settings dialog exposes an `embedding_mode` field (`same-api`, `custom-api`, `local`) plus the legacy backend dropdown (`auto`, `openai`, `langchain`, `sentence-transformers`, `disabled`). Remote modes reuse the OpenAI/LangChain stack, while `local` forces the backend to `sentence-transformers`. CLI/env overrides still target the backend (`--embedding-backend`, `TINKERBELL_EMBEDDING_BACKEND`, `TINKERBELL_EMBEDDING_MODEL`).
- **Optional embeddings extra** – Local mode requires PyTorch + SentenceTransformers. Run `uv sync --extra embeddings` (or `pip install -e '.[embeddings]'`) to pull in `sentence-transformers>=3.0`, `torch>=2.2`, and their transitive deps before flipping the flag.
- **Model routing** – `settings.embedding_model_name` feeds both OpenAI and LangChain adapters. LangChain mode defaults to `langchain_openai.OpenAIEmbeddings` but auto-detects known provider families (OpenAI, DeepSeek, GLM/Zhipu, Moonshot/Kimi) to preconfigure base URLs, tokenizer hints, and embedding dimensions. Provide family-specific keys via `settings.metadata["<family>_api_key"]` or env vars such as `DEEPSEEK_API_KEY`, `GLM_API_KEY`, and `MOONSHOT_API_KEY`; force a family with `settings.metadata.langchain_provider_family` or `TINKERBELL_LANGCHAIN_PROVIDER_FAMILY`. You can still point at any custom class by populating `settings.metadata.langchain_embeddings_class` and `settings.metadata.langchain_embeddings_kwargs` (or the mirrored env vars `TINKERBELL_LANGCHAIN_EMBEDDINGS_CLASS` / `..._KWARGS`).
- **Local SentenceTransformers workflow** – Selecting `embedding_mode="local"` unlocks fields for `st_model_path`, device (`cpu`, `cuda:0`, `mps`), dtype overrides, cache directory, and batch size. The **Test Embeddings** button now runs the same validator used in the controller so you can confirm the model loads before saving. Metadata is persisted under `settings.metadata` and redacted in `--dump-settings` so secrets/path hints stay private.
- **Credentials & packages** – Remote backends reuse your primary API key/base URL or the dedicated `metadata.embedding_api.*` bundle when `custom-api` is active. LangChain mode requires the provider-specific package (e.g., `langchain-openai`, `langchain-community`) plus whatever headers you list in the metadata. Local mode only touches the files/directories you supply and never ships third-party weights with the app.
- **Status + telemetry** – The status bar surfaces `Embeddings: OpenAI/LangChain/SentenceTransformers/Error` labels, and every `ContextUsageEvent` records `embedding_backend`, `embedding_model`, `embedding_status`, and `embedding_detail`. Export scripts and dashboards use those fields to separate OpenAI vs. LangChain vs. local runs and spot misconfigurations quickly.
- **Licensing & troubleshooting** – Runtime errors set `embedding_status="error"` and bubble the message into both the status bar tooltip and telemetry exports. Clearing `~/.tinkerbell/cache/embeddings/` plus toggling the backend forces a clean reinitialization for remote modes; local failures usually indicate a missing model path or GPU driver. Remember that every third-party model/provider ships under its own license—you must review and honor those terms (and any attribution requirements) before pointing TinkerBell at them.

## 12. Edge cases & resilience guardrails (Phase 3)

- **Document checks** – `ai/utils/document_checks.py` centralizes size + MIME heuristics (`document_size_bytes`, `is_huge_document`, `unsupported_format_reason`). Both the outline worker and retrieval tool reuse these helpers so every entry point agrees on what “unsupported” means.
- **Huge doc throttling** – `OutlineBuilderWorker` switches to top-level headings once `is_huge_document()` trips, tags the cache entry with `huge_document_guardrail`, and `DocumentOutlineTool` relays those guardrails + byte counts to the controller along with guidance to run targeted scans.
- **Unsupported / pending statuses** – Outline/retrieval tools now respond with `status="unsupported_format"` (reason string included) whenever binary files slip in, and they advertise `status="pending"` with a `retry_after_ms` when edits arrive faster than the worker debounce window.
- **Offline embeddings** – `DocumentFindTextTool` reports `offline_mode=True` and `status="offline_fallback"` whenever the embedding provider is unavailable so calling agents can treat results as heuristic-only.
- **Cache validation** – Outline cache entries persist the originating `content_hash`; mismatches trigger `_schedule_rebuild`, ensuring corrupted files never hydrate stale outlines.
- **Regression coverage** – Tests in `tests/test_outline_worker.py`, `tests/test_document_outline_tool.py`, `tests/test_retrieval_tool.py`, and `tests/test_memory_buffers.py` cover guardrails, unsupported flows, offline fallbacks, and metadata persistence.
