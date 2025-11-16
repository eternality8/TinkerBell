# Phase 0 token counting benchmarks

This note captures the initial token-count sanity checks that accompany the Phase 0 tokenizer work. The goal is to document the procedure, sample outputs, and current environment limitations so future runs are repeatable.

## Environment

- **OS**: Windows 11 (build 22635)
- **Python**: 3.13.3 (same interpreter used by `uv run pytest`)
- **Optional extras**: `tiktoken>=0.12,<0.13` (pulled in via `ai_tokenizers`) is *not* installed because wheels are not yet published for Python 3.13. Attempting `uv sync --extra ai_tokenizers` fails with `error: can't find Rust compiler`. Installing a Rust toolchain via [rustup](https://rustup.rs/) or running the command under Python 3.12 will enable the precise tokenizer backend as soon as upstream wheels are available.

## Sample measurements

All counts were generated with the built-in `ApproxByteCounter` (bytes ÷ 4) via `scripts/inspect_tokens.py`. Once `tiktoken` can be installed, re-run the commands without `--estimate-only` to capture precise values for comparison.

| Document | Characters | Tokens (precise column falling back to approximation) | Tokens (estimate) |
| --- | ---: | ---: | ---: |
| Alice's Adventures in Wonderland | 163,916 | 42,650 | 42,650 |
| Carmilla | 175,108 | 44,348 | 44,348 |
| Romeo and Juliet | 161,782 | 40,975 | 40,975 |

## How to reproduce

1. Run the helper script against any corpus file (falls back to stdin when `--file` is omitted):
   ```powershell
   uv run python -m tinkerbell.scripts.inspect_tokens --file "test_data/Alice's Adventures in Wonderland.txt"
   ```
2. To force the approximation path even when `tiktoken` is installed, append `--estimate-only`.
3. For deterministic/per-model counts, install the optional extra once wheels or a Rust toolchain are available:
   ```powershell
   uv sync --extra ai_tokenizers
   ```
   > On Python 3.13 the command currently fails unless a Rust compiler is on `PATH`. Installing [rustup](https://rustup.rs/) and re-running the sync resolves the issue; alternatively, use Python 3.12 where official wheels are published.

## Next steps

- Re-run the table whenever you add a new default model or adjust `_DEFAULT_BYTES_PER_TOKEN` so regressions are caught quickly.
- Once precise counts are available, record the delta between `tiktoken` and the byte estimator to validate that the fallback stays within the expected ±3% envelope.
- Consider extending this document with latency measurements (ms per 100k characters) for both the precise and approximate counters to guide future performance work.
