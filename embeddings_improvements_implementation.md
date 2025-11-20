# Embeddings Improvements – Implementation Plan

This document converts the high-level strategy from `embeddings_improvements.md` into actionable workstreams with checkbox-style trackers. Use the checklists to coordinate across contributors and gate releases.

## Workstream A – Settings Schema & Migration
- [x] **A1. Add `embedding_mode` metadata field**
  - Default to `"same-api"` when absent.
  - Wire CLI flag `--embedding-mode` and ensure unknown values fail early.
- [x] **A2. Couple mode ↔ backend logic**
  - Enforce `local` ⇒ `embedding_backend="sentence-transformers"`.
  - Restrict `same-api`/`custom-api` to remote backends (`auto`, `openai`, `langchain`).
- [x] **A3. Persist custom API credentials via `SecretVault`**
  - Store ciphertext + hint under `metadata.embedding_api.*`.
  - Redact in `--dump-settings`.
- [x] **A4. Implement migration helper**
  - On load, migrate legacy keys into the vault and set `embedding_backend="disabled"` until users choose a mode.
  - Add unit test covering old/new payloads.

- [x] **B1. Introduce `_build_sentence_transformer_provider`** in `src/tinkerbell/ui/embedding_controller.py` using `SentenceTransformer` + `LocalEmbeddingProvider`.
- [x] **B2. Mode-aware provider selection**
  - `same-api`: reuse existing `_build_openai_embedding_provider` or `LangChain` path with shared creds.
  - `custom-api`: build provider with dedicated `ClientSettings` derived from `metadata.embedding_api`.
  - `local`: call the new SentenceTransformers builder and populate `_embedding_resource`.
- [x] **B3. Validation hook**
  - Add an `EmbeddingValidator` helper that exercises the active backend per mode (remote ping vs local encode) and surfaces results in UI/logs.
- [x] **B4. Telemetry tagging**
  - Emit provider/mode labels in existing telemetry events for ingestion and retrieval.

## Workstream C – UI & UX Updates
- [x] **C1. Settings dialog mode selector**
  - Add radio or dropdown for the three modes and dynamically show relevant panels.
- [x] **C2. Field validation + error surfacing**
  - Block Apply/Save until required fields per mode are filled; show inline errors.
  - Bubble runtime errors (missing deps, invalid path) to the status bar via `EmbeddingRuntimeState`.
- [x] **C3. Test/Validate button**
  - Add “Test Embeddings” control that invokes Workstream B3 validator and reports success/failure.

## Workstream D – Dependencies, Packaging, and Docs
- [x] **D1. Optional dependency group**
  - Added `[project.optional-dependencies].embeddings` (SentenceTransformers + Torch + NumPy) and documented `uv sync --extra embeddings`.
- [x] **D2. Documentation updates**
  - Expanded `README.md` and `docs/ai_v2.md` with BYO-model steps, mode explanations, validation guidance, and installer commands.
- [x] **D3. Licensing notes**
  - Added explicit reminders that local/remote BYO models are governed by their upstream licenses and require user review/attribution.

## Workstream E – Testing 
- [ ] **E1. Unit tests**
  - Cover settings migration, backend gating, SecretVault serialization, and SentenceTransformers provider wiring using mocks.
- [ ] **E2. Integration tests**
  - Add retrieval ingestion tests with a dummy local encoder under `tests/test_document_chunk_tool.py` (or new suite).

## Delivery Sequencing
1. Complete Workstream A before others to stabilize config surface.
2. Workstreams B and C can proceed in parallel once schema/mode logic exists.
3. Land Workstream D updates alongside initial feature flagging so early adopters have guidance.
4. Finalize with Workstream E to ensure coverage before public release.
pip 