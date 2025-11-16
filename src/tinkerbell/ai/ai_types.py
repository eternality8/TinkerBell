"""Shared typing contracts for AI infrastructure."""

from __future__ import annotations

from typing import Protocol


class TokenCounterProtocol(Protocol):
    """Protocol describing tokenizer implementations."""

    model_name: str | None

    def count(self, text: str) -> int:
        """Return the precise token count for *text*."""

    def estimate(self, text: str) -> int:
        """Return a deterministic fallback estimate when precise counts fail."""


__all__ = ["TokenCounterProtocol"]
