# AI v2 – Phase 0 Technical Plan

## Scope recap
Phase 0 establishes shared infrastructure required by every later milestone:
1. **Tokenizer parity layer** – deterministic token counting per model.
2. **Context usage instrumentation** – turn-level telemetry with opt-in verbosity.
3. **Document version IDs** – bridge + tools report versioned snapshots.
4. **Cache registry & invalidation bus** – pub/sub for document change events.

The plan below details architecture, tasks, dependencies, validation, and rollout sequencing for each objective. Concerns about user education or staffing are out of scope per instructions.

## Architectural overview
```
┌───────────────────┐      ┌────────────────────┐      ┌──────────────────────┐
│ AI Client Token   │      │ AI Controller       │      │ Document Bridge       │
│ Registry           │────▶│ Context Telemetry   │────▶│ Versioned Snapshots    │
└───────────────────┘      │ + Budget Counters   │      └──────────────────────┘
            │              │          │                           │
            ▼              ▼          │                           ▼
    Tokenizer backends  Telemetry Bus │                Cache Bus / Subscribers
   (tiktoken, fallback)  (in-memory   │                (chunk cache, outline,
                         ring buffer) │                 embeddings later)
```

Core contracts introduced in this phase:
- `TokenCounterRegistry` (`ai/client.py`): `{model_name: TokenCounter}` map with methods `register`, `count(text)`, `estimate(tokens)`.
- `ContextUsageEvent` (`ai/ai_controller.py`): dataclass capturing per-turn token stats and build metadata.
- `DocumentVersion` (`tinkerbell/editor/document_model.py`): `{document_id, version_id, hash, edited_ranges}`.
- `DocumentCacheBus` (`ai/memory/cache_bus.py`): publish/subscribe interface broadcasting `DocumentChanged` events.

## Detailed work items
### 1. Tokenizer parity layer
**Objective:** Ensure token counts use model-authentic tokenizers with deterministic fallbacks.

#### Tasks
1. **Define registry abstraction**
   - Create `TokenCounterProtocol` in `ai/ai_types.py` with `count(text: str) -> int`.
   - Implement `TokenCounterRegistry` singleton in `ai/client.py` storing per-model counters.
2. **Add concrete counters**
   - `TiktokenCounter(model_name)` using OpenAI's tokenizer when available.
   - `ApproxByteCounter(charset="utf-8")` fallback estimating tokens via bytes/4.
   - Register defaults for existing models (read from `services/settings.AIModelSettings`).
3. **Wire into AI client + controller**
   - Replace `_estimate_text_tokens` with `_count_tokens(model, text)`; update `AIClient`, `AIController`, and any tool heuristics.
   - Ensure `AIClient` exposes `get_token_counter(model)` for reuse by later phases.
4. **Offline support**
   - Detect missing tiktoken package; log warning once and fall back automatically.
   - Add `pyproject.toml` optional dependency group `[project.optional-dependencies.ai_tokenizers]` referencing `tiktoken` so installs remain opt-in.

#### Technical notes
- Cache tokenizer instances per model to avoid repeated initialization.
- Provide small CLI helper (optional) `scripts/inspect_tokens.py` to inspect token counts during debugging.

### 2. Context usage instrumentation
**Objective:** Capture per-turn prompt/tool budget usage so regressions are visible.

#### Tasks
1. **Event schema**
   - Add `ContextUsageEvent` dataclass containing: `document_id`, `model`, `prompt_tokens`, `tool_tokens`, `response_reserve`, `timestamp`, `conversation_length`, `tool_names`, `run_id`.
2. **Collection points**
   - In `AIController._invoke_model_turn`, compute token counts before model call; after receiving response, emit event.
   - During `_handle_tool_calls`, log each tool payload size (request + response) and accumulate totals per turn.
3. **Telemetry bus**
   - Implement `TelemetrySink` interface with default `InMemoryTelemetrySink` (ring buffer of last N events, e.g., 200) under `ai/services/telemetry.py`.
   - Provide hooks to attach additional sinks later (file writer, stdout).
4. **Settings toggle & UI surfacing**
   - Add `settings.debug.token_logging_enabled: bool` default `False`.
   - When enabled, display rolling totals in the status bar widget (existing `widgets/status_bar.py`). For Phase 0, console logging is acceptable; UI hook can be stubbed behind feature flag.
5. **Persistence for tests**
   - Expose `AIController.get_recent_context_events()` for assertions.

#### Technical notes
- Use monotonic clock for timestamps to keep measurements stable.
- Ensure telemetry logging is non-blocking; use `asyncio.Queue` if `AIController` runs async contexts.

### 3. Document version IDs
**Objective:** Tag every document snapshot/edit with a monotonic version so caches know when to invalidate.

#### Tasks
1. **Document model changes**
   - Add `version_id: int` and `content_hash: str` to `editor/document_model.DocumentModel`.
   - Initialize `version_id=1` when the buffer loads; increment on each mutation commit.
2. **Bridge updates**
   - `services/bridge.py` and `ai/tools/document_snapshot.py` should include `version_id` + `content_hash` in returned payloads.
   - `DocumentBridge.generate_snapshot()` extends metadata schema accordingly.
3. **Patch tool integration**
   - `DocumentApplyPatchTool` validates that the incoming patch references the current `version_id` (optimistic concurrency). If mismatch, raise `DocumentVersionMismatchError` instructing caller to refresh.
   - After successfully applying a patch, bump `version_id` and recompute `content_hash` (fast hash like xxhash or SHA256 truncated).
4. **Test fixtures**
   - Update `tests/test_patches.py` and `tests/test_bridge.py` to expect version metadata.

#### Technical notes
- `content_hash` uses incremental hashing to avoid rehashing entire file each time: track dirty ranges and only rehash affected spans, or accept full hash until optimization needed.
- Provide helper `DocumentVersion.new_from_text(document_id, text, prev_version)` to centralize logic.

### 4. Cache registry + invalidation bus
**Objective:** Provide a single source of truth for document-change notifications.

#### Tasks
1. **Bus interface**
   - Create `DocumentCacheBus` in `ai/memory/cache_bus.py` with methods `subscribe(event_type, handler)` and `publish(event)`.
   - Event types initially: `DocumentChanged`, `DocumentClosed`.
   - `DocumentChanged` payload: `{document_id, version_id, changed_ranges: list[(start, end)], source}`.
2. **Integration points**
   - `DocumentApplyPatchTool` publishes `DocumentChanged` after version increment.
   - `DocumentBridge` publishes `DocumentClosed` when a tab is closed/unloaded.
3. **Subscribers (initial)**
   - Provide stub subscribers for: `ChunkCache`, `OutlineCache`, `EmbeddingCache` (to be implemented later). For Phase 0, create placeholder modules with logging to validate events propagate.
4. **Thread-safety**
   - Use `asyncio` event loop if available; otherwise, protect subscriber list with `threading.Lock` since editor events may come from Qt threads.
5. **Testing utilities**
   - Add `tests/test_memory_buffers.py` coverage verifying subscribers receive events in order and can unsubscribe.

#### Technical notes
- To avoid memory leaks, support weakref-based subscriber registration so caches tied to closed tabs auto-remove.
- Provide synchronous publish for deterministic unit tests; asynchronous mode can be added later.

## Validation strategy
| Work item | Tests | Benchmarks / Data | Acceptance criteria |
|-----------|-------|-------------------|---------------------|
| Tokenizer registry | `tests/test_ai_client.py::test_token_counter_registry`, `tests/test_ai_client.py::test_fallback_counter` | Script comparing counts between registry and known tokenizer on sample text | Token counts for supported models match reference tokenizer ±1 token; fallback used when package missing |
| Context telemetry | `tests/test_agent.py::test_context_usage_logging`, `tests/test_ai_client.py::test_tool_payload_accounting` | Use public-domain docs in `test_data/books/*.txt` to simulate long runs | Each model turn emits event with prompt/tool counts and respects logging toggle |
| Version IDs | `tests/test_patches.py::test_version_increment_on_patch`, `tests/test_bridge.py::test_snapshot_includes_version` | Replay edits on large books to ensure hashing cost acceptable (<50ms per 50k chars) | Snapshot + patch responses always include current version; mismatched edits raise deterministic error |
| Cache bus | `tests/test_memory_buffers.py::test_cache_bus_ordering`, `tests/test_ai_tools.py::test_document_changed_publishes` | Use synthetic subscribers to assert event fan-out latency <5ms within same thread | Publishing from patch tool notifies all subscribers exactly once |

## Use of `test_data` corpus
- Add pytest fixtures loading sample public domain books (e.g., `conftest.py` `@pytest.fixture(scope="session") def large_text(): ...`).
- Use these texts to:
  - Stress-test tokenizer performance and fallback accuracy.
  - Measure telemetry logging overhead at 50k–200k tokens.
  - Validate version hashing cost on large inputs.
- Store derived metrics in `benchmarks/phase0_token_counts.md` to monitor regressions.

## Rollout sequence
1. Implement tokenizer registry (behind feature flag `settings.ai.experimental_tokenizer=True`).
2. Once token counts verified, enable registry by default and land telemetry sink + settings toggle.
3. Introduce version metadata in bridge and patch tools (temporarily keep old fields for compatibility, e.g., `version_id` optional until all callers updated).
4. Add cache bus and wire publishing from patch tool; ship placeholder subscribers that simply log.
5. Remove feature guards after `uv run pytest` passes and metrics look stable on `test_data` corpus.

## Risks & mitigations
- **Performance hit from hashing large files:** start with fast non-cryptographic hash (xxhash64) and allow switching to SHA256 if future requirements demand cryptographic guarantees.
- **Telemetry noise clogging logs:** default buffer limited to 200 events; expose `settings.debug.token_log_limit` to adjust.
- **Tokenizer dependency drift:** pin `tiktoken` minor version and keep fallback estimator tested in CI.

## Deliverables checklist
- [ ] `ai/client.py`: registry + token counter implementations
- [ ] `ai/services/telemetry.py`: sink + event definitions
- [ ] `ai/memory/cache_bus.py`: bus implementation + placeholder subscribers
- [ ] `tinkerbell/editor/document_model.py`: version metadata
- [ ] `services/bridge.py`, `ai/tools/document_snapshot.py`, `ai/tools/document_apply_patch.py`: version-aware payloads
- [ ] Updated tests + new fixtures referencing `test_data`
- [ ] `benchmarks/phase0_token_counts.md` summarizing tokenizer accuracy/perf
- [ ] Docs section appended to `docs/ai_v2.md` describing Phase 0 features and toggles

Completing this plan ensures downstream phases can trust token math, observe budget usage, and rely on cache invalidation semantics when manipulating large documents.
