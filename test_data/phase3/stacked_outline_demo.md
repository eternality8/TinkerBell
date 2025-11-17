# Telemetry Mesh Handbook

## 1. Control Plane

### 1.1 Planner goals
- describe how the controller routes outline vs. retrieval calls
- ensure `outline_digest` comparisons are logged
- note which settings toggles flip Phase 3 features

### 1.2 Budget handoffs
1. Read `ContextBudgetPolicy` status
2. Decide whether to compact pointer payloads
3. Emit trace compaction telemetry with `tokens_saved`

### 1.3 Guardrail hints
> Reminder for the agent: when you see `Guardrail hint (DocumentOutlineTool)`, restate it to the user before proposing edits.

## 2. Execution Plane

### 2.1 Outline hydration protocol
- call `DocumentOutlineTool` for structure
- inspect `guardrails`, `status`, and `outline_digest`
- rehydrate pointer IDs via `DocumentSnapshot`

### 2.2 Retrieval loop
1. Call `DocumentFindSectionsTool` with a natural-language query
2. If `offline_mode=true`, warn the user and rehydrate snippets manually
3. Track which outline node each pointer maps back to

### 2.3 Diff validation checklist
- confirm `document_version`
- apply patch via `DocumentEdit`
- cite resulting diffs before completion

## 3. Telemetry Appendix

### 3.1 Outline events
| Event | Payload | Why it matters |
| --- | --- | --- |
| `outline.tool.hit` | digest, node_count, `trimmed_reason` | Shows how much we saved vs. full doc |
| `outline.pending` | document_id, retry_after_ms | Confirms debounce backpressure |

### 3.2 Retrieval events
- `retrieval.query`
- `retrieval.provider.error`
- `embedding.cache.miss`

### 3.3 Sample prompt queue
```markdown
- Summarize the Control Plane goals.
- Hydrate pointer: outline:telemetry-mesh:planner/guardrails.
- Quote diff validation checklist verbatim.
```
