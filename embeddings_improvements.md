# Embeddings Improvements Plan

## 1. Objectives
- Support running SentenceTransformers-compatible embedding models entirely offline without bundling vendor weights.
- Allow users to point TinkerBell at any HF/SentenceTransformers model directory that fits their hardware (CPU, GPU, quantized variants, ONNX, etc.).
- Maintain parity with existing OpenAI/LangChain backends so the retrieval pipeline (outline worker, document search tools) keeps working with minimal toggles.

## 2. Architecture Overview
1. **New backend identifier**: extend `Settings.embedding_backend` plus CLI choices with `"sentence-transformers"` (alias `"st"`).
2. **Provider construction**: add `_build_sentence_transformer_provider` in `src/tinkerbell/ui/embedding_controller.py` that:
   - Imports `sentence_transformers` lazily.
   - Resolves model path/name, device, dtype, cache dir from settings metadata.
   - Instantiates `SentenceTransformer` and wraps it with the existing `LocalEmbeddingProvider` to satisfy the `DocumentEmbeddingIndex` API.
3. **Thread offloading**: rely on `LocalEmbeddingProvider` calling `asyncio.to_thread` so UI remains responsive even when SentenceTransformers is CPU-bound.
4. **Resource lifecycle**: store the instantiated model in `_embedding_resource` so shutdown logic can dispose it (torch `cpu` tensors, GPU memory, etc.).

## 3. Settings, CLI, and UI Changes
- **Settings fields** (all optional, stored inside `Settings.metadata` to avoid schema churn):
  - `st_model_path`: HF repo id (e.g., `Qwen/Qwen3-Embedding-0.6B`) or absolute directory.
  - `st_device`: `auto` (default), `cpu`, `cuda:0`, `mps`, etc.
  - `st_dtype`: `float16`, `bfloat16`, `int8`, etc., for torch autocasting.
  - `st_cache_dir`: path for downloaded weights.
  - `st_batch_size`: overrides provider batch size when users know their hardware limits.
- **CLI overrides**: `--embedding-backend sentence-transformers` plus `--set metadata.st_model_path=...` for quick testing.
- **Settings dialog** (later phase): expose a small panel listing backend, model path selector, and hardware dropdown. For now, document the JSON keys in `docs/ai_v2.md`.
- **Status bar messaging**: extend `EmbeddingRuntimeState` labels to show `SentenceTransformers/<model name>` and bubble up errors like “Model path not found” or “torch not installed.”

### 3.1 Mode-Based Settings Cleanup
To declutter the settings UI, define an `embedding_mode` metadata value with three mutually exclusive options that gate which controls appear:

1. **`same-api` (default)** – embeddings reuse the main LLM API credentials. Only show `embedding_backend` (auto/openai/langchain) plus the shared `model` dropdown. Hide custom base URL/API key fields.
2. **`custom-api`** – embeddings call a different remote provider. Show dedicated fields (base URL, API key/organization, timeout, retry knobs, custom headers). Persist them under `metadata.embedding_api.*` to keep the primary `Settings` schema stable.
3. **`local`** – activates SentenceTransformers/local runner settings. Expose only the local-specific inputs (model path picker, device/dtype selectors, batch size). Hide remote API toggles entirely.

Implementation tasks:
- Add `embedding_mode` to settings (default `"same-api"`) with CLI override `--embedding-mode`.
- Update the settings dialog to render dynamic sections based on the selected mode, so users see only relevant controls.
- Couple `embedding_mode` to acceptable backends: `same-api` inherits whatever backend the LLM uses (`auto`/`openai`/`langchain`), `custom-api` unlocks the remote-only backends (`openai`, future REST providers), and `local` forces `embedding_backend="sentence-transformers"` so we never end up with impossible combos.
- Ensure validation logic enforces required fields per mode (e.g., `custom-api` must have base URL + API key, `local` must have model path) and blocks Apply until satisfied.
- Extend telemetry/settings snapshots to include the current mode for support diagnostics.

### 3.2 Credential Storage
- Reuse the existing `SecretVault` plumbing instead of inventing a new cipher path. Persist alternate API credentials beside the primary ones as `metadata.embedding_api.api_key_ciphertext`, `metadata.embedding_api.api_key_hint`, etc., and decrypt via `SecretVault.decrypt()` during settings load.
- Mirror the helper methods already used for `Settings.api_key` so we get redaction + env override support for free.
- When exporting dumps (e.g., `--dump-settings`), redact the alternate API key the same way as the main one.

## 4. Dependency Strategy
- Add an optional dependency group in `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  embeddings = ["sentence-transformers>=3.0", "torch>=2.2", "numpy"]
  ```
- Detect missing packages at runtime and surface a friendly error telling users to install `uv pip install '.[embeddings]'`.
- Document alternative stacks (e.g., `onnxruntime` or `bitsandbytes`) but avoid forcing them so the base app stays slim.

## 5. Runtime Integration Details
- **Model resolution**: use Hugging Face’s local files first; if the path doesn’t exist, let HF download to `st_cache_dir` (or default `~/.cache/huggingface`).
- **Device selection**: allow `auto` to pick GPU when available; respect explicit overrides and fall back to CPU with a warning.
- **Batch sizing**: default to `min(8, model.max_seq_len // 1024)` to protect small GPUs; allow overrides through metadata.
- **Vector normalization**: ensure the provider outputs Python floats; `LocalEmbeddingProvider` already converts sequences to tuples for SQLite storage.
- **Telemetry**: tag ingest and retrieval events with `provider="sentence-transformers:<model>"` for future analytics.

## 6. Error Handling & UX
- If SentenceTransformers or torch is missing, set `EmbeddingRuntimeState.status="error"` with actionable guidance (e.g., “Install extras via `uv pip install '.[embeddings]'`).
- When the model path is invalid, keep fallback retrieval enabled and show `strategy="fallback"` in `DocumentFindSectionsTool` responses.
- Consider adding a “Test embeddings” button later that runs a short similarity query to confirm the local backend works.

## 7. Documentation & Licensing
- Update `README.md` / `docs/ai_v2.md` with a “Bring Your Own Embedding Model” section covering:
  1. Installing optional dependencies.
  2. Downloading a model via `huggingface-cli download` or `git lfs clone`.
  3. Setting metadata keys in `settings.json`.
  4. Verifying status in the TinkerBell status bar.
- Clarify that users are responsible for complying with the model’s license; the app only references files on disk.

## 8. Testing & Validation
- **Unit tests**: mock `SentenceTransformer` to assert `_build_sentence_transformer_provider` wires settings correctly and errors when missing deps.
- **Integration tests**: spin up a fake model returning deterministic vectors to ensure ingestion + retrieval (e.g., `tests/test_document_chunk_tool.py`) still pass.
- **Performance tests**: reuse `benchmarks/measure_diff_latency.py` (or create a new script) to compare ingestion times between OpenAI and local ST backend on representative documents.

## 9. Rollout Steps
1. Land settings + provider code with feature flag defaulting to existing backend.
2. Add docs + optional dependency instructions.
3. Collect feedback from internal testers using Qwen3-Embedding or other models.
4. Promote to public release once telemetry shows acceptable ingestion latency and no major UX regressions.

## 10. Migration & Validation
- **Migration defaults**: when upgrading existing `settings.json`, if `embedding_mode` is absent set it to `"same-api"` but leave the backend effectively **disabled** until the user picks a mode explicitly (e.g., treat missing/legacy configs as `embedding_backend="disabled"`). This avoids inadvertently switching users to OpenAI or local backends.
- **Secret reflow**: during migration, if `metadata.embedding_api.api_key_ciphertext` is missing yet legacy `metadata.embedding_api.api_key` exists, encrypt it via `SecretVault` the same way we migrate the primary `api_key` today.
- **Validation hook**: add a dedicated embedding validation routine (parallel to the existing LLM “Test Connection” flow) that pings the selected backend in each mode:
  - `same-api`: reuse the existing AI client test since it shares credentials.
  - `custom-api`: perform a lightweight embeddings request (e.g., `client.embeddings.create` with a short prompt) to confirm the alternate key/base URL works.
  - `local`: run a dry-run `sentence_transformer.encode(["ping"])` (or equivalent) to ensure the model path loads and vectors have the expected dimensionality.
- **Automated tests**: add unit tests for the validation helper covering success/error paths per mode so future regressions are caught without manual QA.
