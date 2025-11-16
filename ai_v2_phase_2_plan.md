# AI v2 – Phase 2 Technical Plan

## 1. Mission & guardrails
- **Goal:** Enforce token-aware gating and trace summarization so 100K+ token documents stay within configured budgets without losing edit fidelity.
- **Definition of done:** Controller never appends prompt/tool payloads that exceed the configured reserve; oversized traces are summarized/pointerized automatically; regressions are caught by telemetry + automated tests.
- **Hard constraints:** Maintain backward-compatible tool schemas, keep heavy work off the UI thread, and reuse the instrumentation delivered in Phases 0–1.

## 2. Success criteria & telemetry
1. **Budget integrity:** `AIController` refuses to send a completion request when `ContextBudgetPolicy` reports insufficient headroom. Telemetry field `context_budget_violation` stays at 0 after rollout.
2. **Summaries stay lossless:** For any tool payload summarized/pointerized, the agent can rehydrate the original data via tool calls ≤2 hops. Tests must cover diff + plaintext payloads.
3. **Latency ceiling:** Additional policy checks/summarization add <50 ms median overhead per turn on `benchmarks/measure_diff_latency.py` against `test_data/1MB.json`.
4. **Observability:** Settings panel exposes live totals (prompt tokens, response reserve, pointer count) and trace compaction stats emit to `ai/services/telemetry.py`.

## 3. Architecture snapshot
```
┌──────────────────────────────────────────────────────────┐
│ AIController                                             │
│  ├─ ContextBudgetPolicy (new)                            │
│  │   • model profile registry                            │
│  │   • prompt/response reserve math                      │
│  ├─ Tool payload gate                                    │
│  │   • Token counting via tokenizer registry             │
│  │   • Summarize / pointer fallback                      │
│  ├─ TraceCompactor (new)                                 │
│  │   • Rolling token ledger                              │
│  │   • Summarizer + pointer swaps                        │
│  └─ Telemetry emitters                                   │
└──────────────────────────────────────────────────────────┘

Summarization helpers live under `ai/services` with hooks into tool schemas. Pointer metadata piggybacks on `chat/message_model.py` and `ai/agents/executor.py` so LangGraph steps see compact traces while the UI retains full payloads.
```

## 4. Workstreams & implementation details

### 4.1 Context budget policy layer
- **Files touched:** `src/tinkerbell/ai/client.py`, `ai/ai_types.py`, `ai/agents/executor.py`, `ai/services/settings.py`, `ai/controller` (inside `app.py` if controller defined there).
- **Contract:**
  - `ContextBudgetPolicy` class with methods:
    - `from_settings(model_profile)` → inject per-model prompt budget, response reserve, emergency floor.
    - `tokens_available(conversation, pending_tool_payloads)` → returns `BudgetDecision` {ok|needs_summary|reject, reason, deficit}.
    - `record_usage(turn_id, prompt_tokens, response_tokens_reserved)`.
  - Accept runtime overrides (e.g., `AIController(budget_policy=...)` for tests).
- **Settings UI:** Add `ai.context_policy.enabled`, `prompt_budget_override`, `response_reserve` fields with help text. Wire to status bar display.
- **Telemetry:** Emit budget checks + rejections via `telemetry.emit("context_budget_decision", …)`.
- **Tests:**
  - Unit tests in `tests/test_ai_client.py` or new `test_budget_policy.py` verifying policy math across models.
  - Integration test in `tests/test_agent.py` simulating oversize tool payload → expect summarization path triggered.

### 4.2 Summarizer + pointer helpers
- **Helper module:** `src/tinkerbell/ai/services/summarizer.py` (new).
- **APIs:**
  - `summarize_tool_content(payload: ToolPayload, schema_hint: str, budget: int) -> SummaryResult`
  - `build_pointer(payload_id, kind, metadata) -> ToolPointerMessage`
- **Input types:** plaintext, diff, bullet list, structured JSON (limit to shallow BFS to avoid recursion).
- **Implementation approach:**
  - Prefer heuristic summarizer (line truncation + bullet extraction) for deterministic behavior; optionally integrate tiny LLM later via setting `use_llm_summarizer`.
  - Diff summarization: capture change stats (insertions/deletions per hunk), first/last lines, and chunk IDs.
  - Plaintext summarization: sentence boundary detection (use `textwrap` + regex) up to `budget` tokens.
- **Pointer schema:**
  - Extend `chat/message_model.py` with `ToolPointerMessage(BaseModel)` containing `pointer_id`, `kind`, `display_text`, `rehydrate_instructions`.
  - `AIController` replaces original payload with pointer message while storing raw payload in `executed_tool_calls` for UI display.
- **Tool opt-out:** Add `summarizable: bool = True` to tool response dataclass; critical tools (e.g., validator) can skip summarization.
- **Tests:** `tests/test_ai_tools.py` verifying summarizer respects budgets and pointer rehydration instructions appear.

### 4.3 Trace compactor
- **Location:** `ai/agents/executor.py` or dedicated `ai/services/trace_compactor.py`.
- **Responsibilities:**
  1. Maintain running token ledger across user ↔ AI ↔ tool messages.
  2. When ledger > `policy.trace_threshold_tokens`, call summarizer on oldest tool messages while preserving user/system prompts.
  3. Persist compaction map so UI can expand collapsed entries.
- **Data model changes:**
  - `chat/chat_panel.py` needs to accept compaction metadata to render "summary" badges.
  - `services/bridge_router.py` ensures transcripts saved to disk include both pointer + raw payload for auditing.
- **Telemetry:** track `compactions_per_session`, `tokens_saved`.
- **Tests:** Add regression test in `tests/test_chat_panel.py` ensuring UI surfaces pointer toggle; conversation log tests verifying compactor preserves order.

### 4.4 Controller integration & flow changes
1. **Token accounting hook:**
   - Use tokenizer registry from Phase 0 inside `_build_messages` to count pending tokens before hitting the model.
   - If `BudgetDecision` returns `needs_summary`, call summarizer on the offending payload(s), update ledger, retry policy.
   - If `reject`, raise a user-facing event instructing them to reduce scope; log telemetry.
2. **LangGraph prompt tweaks:**
   - Update `prompts.py` to mention pointer messages + rehydrate instructions.
   - Ensure `ai/agents/graph.py` nodes don't attempt to parse pointers as real tool results.
3. **Backwards compatibility:**
   - Keep old behavior behind `context_policy.enabled = False`.
   - Add migration note in `docs/ai_v2.md` + `README` describing new settings and fallback.

## 5. Delivery timeline (3 sprints)
| Sprint | Focus | Key artifacts |
| --- | --- | --- |
| **S1 – Policy foundations (1 wk)** | Implement `ContextBudgetPolicy`, settings, and controller hooks for dry-run logging only. | Policy class, unit tests, telemetry events flagged as `dry_run`.
| **S2 – Summaries & pointers (1.5 wk)** | Add summarizer helper, pointer schema, tool opt-outs, and enable `needs_summary` path with feature flag. | `summarizer.py`, `ToolPointerMessage`, updated prompts/tests.
| **S3 – Trace compactor + GA (1 wk)** | Ship trace compaction, UI affordances, full enforcement of policy, docs/benchmarks update. | `trace_compactor.py`, chat panel badges, finalize docs & benchmarks.

### Sprint checklists

#### ✅/⬜ S1 – Policy foundations
- [x] `ContextBudgetPolicy` class implemented with `from_settings`, `tokens_available`, and `record_usage` APIs (`src/tinkerbell/ai/services/context_policy.py`).
- [x] Settings schema + UI for `context_policy.enabled`, `prompt_budget_override`, `response_reserve`, and status-bar surfacing (`services/settings.py`, `widgets/dialogs.py`, status bar text).
- [x] Controller wired for dry-run policy evaluation inside `_build_messages` (no enforcement yet) plus status reporting (`ai/agents/executor.py`, `main_window.py`).
- [x] Telemetry event `context_budget_decision` emitted with `dry_run=true` payload (`services/telemetry.py` + controller emit hook).
- [x] Unit tests (`tests/test_context_budget_policy.py`, `tests/test_agent.py`, `tests/test_widgets_dialogs.py`, `tests/test_services_telemetry.py`) cover multiple model profiles and telemetry paths.
- [x] Documentation stub (README note under AI v2 “Context budget policy (dry run)”) describing upcoming policy toggle for reviewers.

#### ✅/⬜ S2 – Summaries & pointers
- [x] `ai/services/summarizer.py` added with heuristics for plaintext + diff payloads. (`tinkerbell.ai.services.summarizer` now ships deterministic plaintext/diff summarizers plus pointer builders.)
- [x] `ToolPointerMessage` model introduced and serialized through chat pipeline. (`chat.message_model` exposes the pointer dataclass + helpers.)
- [x] Tools updated with `summarizable` flag plus opt-outs for validators/critical outputs. (`diff_builder` + `list_tabs` stayed summarizable while `document_edit`, `document_apply_patch`, `search_replace`, and validators now opt out.)
- [x] Controller path converts `needs_summary` decisions into summary + pointer swaps. (`AIController._compact_tool_messages` now produces pointers when `ContextBudgetPolicy` says `needs_summary`.)
- [x] Prompt/agent updates teach LangGraph nodes how to rehydrate pointers. (`ai/prompts.py` now briefs agents on re-running tools when pointer text appears.)
- [x] Expanded tests in `test_ai_tools.py` (summaries) and `test_agent.py` (pointer round-trip) pass under feature flag. (New cases verify tool flags plus pointer creation + non-summarizable behavior.)

#### ✅/⬜ S3 – Trace compactor + GA
- [x] `TraceCompactor` service integrated with controller token ledger.
- [x] Chat panel + transcript storage display compaction badges and provide raw payload access.
- [x] Telemetry metrics (`compactions_per_session`, `tokens_saved`) recorded and visualized.
- [x] Benchmarks refreshed (diff latency script) showing <50 ms overhead and documenting token savings.
- [x] Documentation/README + `docs/ai_v2.md` updated; migration note filed.
- [x] Feature flag flipped to enforce policy + compaction by default; rollout checklist signed off.

> Benchmark snapshot (Nov 2025): `benchmarks/measure_diff_latency.py` now reports `diff_tokens`, `pointer_tokens`, and `tokens_saved`, showing War and Peace’s 88K-token diff pointerized to 247 tokens (88K saved) and the 5 MB JSON case collapsing 134K tokens to 237 tokens while staying under 700 ms.

## 6. Validation strategy
- **Automated:**
  - Existing `uv run pytest` plus new suites: `test_budget_policy`, `test_trace_compactor`, pointer serialization tests.
  - Add benchmark case in `benchmarks/measure_diff_latency.py` invoking oversized tool payload to record compaction savings.
- **Manual:**
  - Scenario: open `test_data/War and Peace.txt`, request full snapshot; verify pointer created and UI shows expandable summary.
  - Scenario: simulate 5 sequential tool runs to force trace compaction and confirm transcripts remain readable.
- **Telemetry review:** Dashboards flagged for `context_budget_violation`, `summary_count`, `pointer_refetch_count`.

## 7. Risks & mitigations
| Risk | Mitigation |
| --- | --- |
| Summaries hide important details | Provide "rehydrate" affordance + allow tools to opt out. Add E2E test to ensure agent can refetch chunk via pointer instructions.
| Performance hit from token counting | Cache tokenizer results per message hash; run counts off main thread when possible.
| UI confusion about hidden data | Surface badges + tooltips in chat panel; add docs section "Why did my tool output shrink?".
| Policy misconfiguration blocking work | Offer safe defaults + "dry run" logging mode; add CLI command to inspect budgets (`scripts/inspect_tokens.py --policy`).

## 8. Documentation & rollout checklist
1. Update `docs/ai_v2.md` and new `docs/ai_v2_phase_2.md` excerpt summarizing implementation + settings.
2. README "What’s new" bullet under AI v2.
3. Migration note in `improvement_ideas.md` referencing completion status.
4. Telemetry dashboards screenshot stored in `assets/` for release notes.

## 9. Exit criteria
- All success metrics in §2 met across regression suite + manual smoke.
- Pointerized runs documented with before/after token totals.
- Feature flag defaulted to ON for internal builds; provide setting to disable while gathering feedback.
- Phase 3 backlog reviewed to ensure no blockers remain.
