# Guardrail Scenarios Cheat Sheet

Use these mini playbooks whenever you need to trigger a specific outline/retrieval guardrail during manual QA. The steps assume you already enabled **Settings → Experimental → Phase 3 outline tools**.

## 1. Huge document clamp
1. Open `test_data/5MB.json`.
2. Ask the agent: “Produce an outline for every key in this JSON file.”
3. Expected signals:
   - `DocumentOutlineTool` returns `guardrails[0].type = "huge_document"`.
   - `trimmed_reason = "token_budget"` and reduced `node_count`.
   - Controller injects `Guardrail hint (DocumentOutlineTool)` reminding you to work in chunks.
4. Follow-up: request targeted `DocumentFindSectionsTool` calls for the pointer IDs that remain.

## 2. Pending outline rebuild
1. Open `test_data/1MB.json`.
2. Issue several quick edits (rename headings, insert lorem ipsum) so the outline worker queues a rebuild.
3. Immediately ask for a fresh outline.
4. Expected signals:
   - Tool responds with `status = "pending"` and `retry_after_ms ≈ 750`.
   - Controller injects a guardrail hint advising you to wait or rely on DocumentSnapshot temporarily.

## 3. Unsupported format detection
1. Open `test_data/phase3/firmware_dump.bin`.
2. Ask for an outline or retrieval query.
3. Expected signals:
   - `DocumentOutlineTool` (and retrieval, if invoked) return `status = "unsupported_format"` with `reason = "binary_extension:.bin"`.
   - Guardrail hint instructs you to convert the file before retrying.

## 4. Offline retrieval fallback
1. Temporarily disable the embedding backend (Settings → AI → Embeddings → Disabled) **or** disconnect your network.
2. Open any Markdown/YAML doc (for example `test_data/phase3/stacked_outline_demo.md`).
3. Ask the agent: “Find the telemetry TODO paragraph.”
4. Expected signals:
   - `DocumentFindSectionsTool` returns `status = "offline_fallback"`, `strategy = "fallback"`, and `offline_mode = true`.
   - Controller guardrail hint explains previews are heuristic only.
5. Follow-up: run DocumentSnapshot on the pointer range before editing.

## 5. Outline cache integrity check (optional)
1. With any doc open, manually modify its text outside of TinkerBell (e.g., edit the file on disk) and reload.
2. Request an outline immediately.
3. Expected signals:
   - Tool detects a version mismatch, marks the cached outline as stale, and schedules a rebuild.
   - Guardrail hint reminds you to treat stale outlines as hints only.

Document these outcomes in your QA notes so future regressions have crisp reproduction steps.
