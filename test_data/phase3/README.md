# Phase 3 Samples

These fixtures live under `test_data/phase3/` so you can reproduce the new outline/retrieval workflows and guardrails without hunting for data. They pair with the larger canon files that already lived in `test_data/` (for example `War and Peace.txt`, `1MB.json`, and `5MB.json`). Open them from **File → Open…** or the sample dropdown and reference the table below when you need to trigger a specific controller hint.

## Outline walkthrough (fast)

- **`stacked_outline_demo.md`** – Deeply nested Markdown with pointer-friendly headings. Ideal for verifying that DocumentOutlineTool returns hierarchies, pointer IDs, and blurbs, and that DocumentFindTextTool can hydrate targeted sections before diffing.
- Suggested flow:
  1. Load the file, send “Summarize the architecture table by heading,” and confirm the agent calls DocumentOutlineTool.
  2. Ask the agent to "find the telemetry TODO"; it should follow the outline pointer, call DocumentFindTextTool, and hydrate the referenced chunk before editing.
  3. Approve the diff and note the guardrail-free trace.

## Guardrail + resilience scenarios

- **`guardrail_scenarios.md`** – Step-by-step prompts that reference other fixtures so you can issue the right commands quickly (huge docs, binary formats, offline fallbacks, and pending outlines).
- **`firmware_dump.bin`** – Tiny binary placeholder with a `.bin` extension. Opening it and requesting an outline forces `DocumentOutlineTool` to report `status="unsupported_format"`.
- **Existing large docs** – `5MB.json` and `War and Peace.txt` remain the quickest way to hit the huge-document guardrail while `1MB.json` is perfect for watching pending outlines after rapid edits.

## Quick reproduction table

| Scenario | File(s) | Prompt idea | Expected guardrail | Follow-up |
| --- | --- | --- | --- | --- |
| Huge document throttle | `test_data/5MB.json` | "Outline this entire JSON file" | `guardrails[0].type = "huge_document"`, `status="ok"` but trimmed | Ask the agent to work pointer-by-pointer and cite `trimmed_reason=token_budget` in replies.
| Pending rebuild | `test_data/1MB.json` (rapid consecutive edits) | Make several quick edits then request an outline | `status="pending"`, `retry_after_ms≈750` | Wait the suggested delay or continue with DocumentSnapshot until the outline finishes.
| Unsupported format | `test_data/phase3/firmware_dump.bin` | "Summarize the outline" | `status="unsupported_format"`, `reason="binary_extension:.bin"` | Switch to plain text files or convert content before retrying.
| Offline retrieval fallback | Any doc + temporarily disable embeddings/turn off network | "Find the security policy section" | `status="offline_fallback"`, `offline_mode=true` | Treat previews as heuristics and rehydrate via DocumentSnapshot before editing.

For additional troubleshooting tips see `docs/ai_v2.md` (Phase 3 quickstart) and the README section "Phase 3 outline + retrieval".
