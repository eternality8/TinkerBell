"""Token estimation utilities for AI operations."""

from __future__ import annotations

import math

# Average characters per token for English prose (GPT-style tokenization)
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple byte-based heuristic of ~4 bytes per token.
    This provides a reasonable approximation for English prose
    with GPT-style tokenization.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count (minimum 1 for non-empty text, 0 for empty).
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8", errors="ignore")) / CHARS_PER_TOKEN))


__all__ = ["CHARS_PER_TOKEN", "estimate_tokens"]
