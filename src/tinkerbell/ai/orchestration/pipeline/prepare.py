"""Prepare stage: Build messages and estimate token budget.

This module implements the first stage of the turn pipeline, which:
1. Builds the message sequence (system, history, user)
2. Estimates token usage
3. Evaluates budget constraints

All functions are stateless and operate on immutable data.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

from ... import prompts
from ..types import (
    BudgetEstimate,
    DocumentSnapshot,
    Message,
    PreparedTurn,
    TurnConfig,
    TurnInput,
)

LOGGER = logging.getLogger(__name__)

# Default token estimation when no counter is available
_DEFAULT_BYTES_PER_TOKEN = 4

# Minimum headroom to leave for prompt construction overhead
_PROMPT_HEADROOM = 6_000

# Maximum messages to include from history
_DEFAULT_HISTORY_LIMIT = 60


__all__ = [
    "build_messages",
    "estimate_budget",
    "prepare_turn",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "sanitize_history",
]


# -----------------------------------------------------------------------------
# Token Counter Protocol
# -----------------------------------------------------------------------------


class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def __call__(self, text: str) -> int:
        """Count tokens in the given text."""
        ...


# -----------------------------------------------------------------------------
# Token Estimation
# -----------------------------------------------------------------------------


def estimate_text_tokens(
    text: str,
    *,
    token_counter: TokenCounter | None = None,
) -> int:
    """Estimate token count for raw text.

    Uses the provided token counter if available, otherwise falls back
    to a byte-based heuristic (4 bytes ≈ 1 token).

    Args:
        text: Raw text to estimate.
        token_counter: Optional callable that counts tokens.

    Returns:
        Estimated token count (minimum 1 for non-empty text).
    """
    if not text:
        return 0

    if token_counter is not None:
        try:
            return int(token_counter(text))
        except Exception:
            LOGGER.debug("Token counter failed; using heuristic", exc_info=True)

    # Fallback: byte-based estimation
    byte_length = len(text.encode("utf-8", errors="ignore"))
    return max(1, math.ceil(byte_length / _DEFAULT_BYTES_PER_TOKEN))


def estimate_message_tokens(
    message: Message | Mapping[str, Any],
    *,
    token_counter: TokenCounter | None = None,
) -> int:
    """Estimate token count for a single message.

    Includes overhead for role and message boundary tokens.

    Args:
        message: Chat message (Message dataclass or dict).
        token_counter: Optional callable that counts tokens.

    Returns:
        Estimated token count including message overhead.
    """
    if isinstance(message, Message):
        content = message.content
        has_role = True
    else:
        content = str(message.get("content", "") or "")
        has_role = bool(message.get("role"))

    tokens = estimate_text_tokens(content, token_counter=token_counter)

    # Add overhead for role and message boundaries
    if has_role:
        tokens += 4

    return tokens


# -----------------------------------------------------------------------------
# History Sanitization
# -----------------------------------------------------------------------------


def sanitize_history(
    history: Sequence[Message | Mapping[str, Any]],
    *,
    limit: int = _DEFAULT_HISTORY_LIMIT,
    token_budget: int | None = None,
    token_counter: TokenCounter | None = None,
) -> tuple[Message, ...]:
    """Sanitize and trim conversation history to fit budget.

    Filters for valid roles, truncates to limit, and trims older
    messages to fit within token budget.

    Args:
        history: Raw conversation history.
        limit: Maximum number of messages to include.
        token_budget: Optional token limit for history.
        token_counter: Optional callable that counts tokens.

    Returns:
        Sanitized tuple of Message objects.
    """
    allowed_roles = {"user", "assistant", "system", "tool"}

    # Take the most recent messages up to limit
    window = list(history)[-limit:] if limit else list(history)

    # Normalize to Message objects and filter
    sanitized: list[Message] = []
    for entry in window:
        if isinstance(entry, Message):
            msg = entry
        else:
            role = str(entry.get("role", "user")).lower()
            if role not in allowed_roles:
                continue
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            msg = Message(
                role=role,  # type: ignore[arg-type]
                content=content,
                name=entry.get("name"),
                tool_call_id=entry.get("tool_call_id"),
            )
        sanitized.append(msg)

    # If no budget constraint, return all sanitized messages
    if token_budget is None:
        return tuple(sanitized)

    budget = max(0, token_budget)
    if budget == 0 or not sanitized:
        return ()

    # Trim from oldest to newest, keeping as many recent messages as fit
    trimmed: list[Message] = []
    remaining = budget

    for entry in reversed(sanitized):
        tokens = estimate_message_tokens(entry, token_counter=token_counter)

        # Always include at least one message even if over budget
        if trimmed and tokens > remaining:
            break

        # Include first message even if over budget
        if not trimmed and tokens > remaining:
            trimmed.append(entry)
            break

        trimmed.append(entry)
        remaining -= tokens

        if remaining <= 0:
            break

    trimmed.reverse()
    return tuple(trimmed)


# -----------------------------------------------------------------------------
# Message Building
# -----------------------------------------------------------------------------


def build_messages(
    prompt: str,
    snapshot: DocumentSnapshot,
    history: Sequence[Message | Mapping[str, Any]] = (),
    *,
    config: TurnConfig | None = None,
    token_counter: TokenCounter | None = None,
) -> tuple[Message, ...]:
    """Build the message sequence for a chat turn.

    Constructs system, history, and user messages while respecting
    token budget constraints.

    Args:
        prompt: User prompt text.
        snapshot: Document snapshot for context.
        history: Optional conversation history.
        config: Turn configuration (for budget limits).
        token_counter: Optional callable that counts tokens.

    Returns:
        Tuple of Message objects in order: [system, ...history, user].
    """
    cfg = config or TurnConfig()
    model_name = cfg.model_name

    # Build snapshot dict for prompt formatting
    snapshot_dict = _snapshot_to_dict(snapshot)

    # Format prompts using existing prompt templates
    system_content = prompts.base_system_prompt(model_name=model_name)
    user_content = prompts.format_user_prompt(prompt, snapshot_dict, model_name=model_name)

    system_message = Message.system(system_content)
    user_message = Message.user(user_content)

    # Calculate budget for history
    context_limit = cfg.max_context_tokens
    reserve = _effective_response_reserve(cfg.response_reserve, context_limit)
    prompt_budget = max(0, context_limit - reserve)

    system_tokens = estimate_message_tokens(system_message, token_counter=token_counter)
    user_tokens = estimate_message_tokens(user_message, token_counter=token_counter)
    required_prompt_tokens = system_tokens + user_tokens

    # Reclaim from reserve if needed
    if prompt_budget < required_prompt_tokens and reserve > 0:
        shortfall = required_prompt_tokens - prompt_budget
        reclaimed = min(reserve, shortfall)
        prompt_budget += reclaimed

    history_budget = max(0, prompt_budget - required_prompt_tokens)

    # Sanitize and trim history
    trimmed_history = sanitize_history(
        history,
        limit=_DEFAULT_HISTORY_LIMIT,
        token_budget=history_budget,
        token_counter=token_counter,
    )

    # Assemble final message list
    messages: list[Message] = [system_message]
    messages.extend(trimmed_history)
    messages.append(user_message)

    return tuple(messages)


# -----------------------------------------------------------------------------
# Budget Estimation
# -----------------------------------------------------------------------------


def estimate_budget(
    messages: Sequence[Message],
    config: TurnConfig,
    *,
    token_counter: TokenCounter | None = None,
) -> BudgetEstimate:
    """Estimate token budget for the prepared messages.

    Args:
        messages: Prepared message sequence.
        config: Turn configuration with budget limits.
        token_counter: Optional callable that counts tokens.

    Returns:
        BudgetEstimate with token counts and verdict.
    """
    # Calculate total prompt tokens
    prompt_tokens = sum(
        estimate_message_tokens(msg, token_counter=token_counter)
        for msg in messages
    )

    context_limit = config.max_context_tokens
    completion_budget = _effective_response_reserve(config.response_reserve, context_limit)
    total_budget = context_limit

    # Calculate headroom
    headroom = total_budget - prompt_tokens - completion_budget

    # Determine verdict
    if headroom >= 0:
        verdict = "ok"
        reason = "within-budget"
    elif headroom >= -_PROMPT_HEADROOM:
        # Within emergency buffer
        verdict = "needs_summary"
        reason = "exceeds-budget"
    else:
        verdict = "reject"
        reason = "exceeds-emergency"

    return BudgetEstimate(
        prompt_tokens=prompt_tokens,
        completion_budget=completion_budget,
        total_budget=total_budget,
        headroom=headroom,
        verdict=verdict,  # type: ignore[arg-type]
        reason=reason,
    )


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def prepare_turn(
    turn_input: TurnInput,
    *,
    token_counter: TokenCounter | None = None,
) -> PreparedTurn:
    """Prepare a turn for execution.

    This is the main entry point for the prepare stage. It builds
    messages, estimates budget, and returns a PreparedTurn ready
    for the analyze stage.

    Args:
        turn_input: The input to the turn.
        token_counter: Optional callable that counts tokens.

    Returns:
        PreparedTurn with messages and budget estimate.
    """
    # Build messages
    messages = build_messages(
        prompt=turn_input.prompt,
        snapshot=turn_input.snapshot,
        history=turn_input.history,
        config=turn_input.config,
        token_counter=token_counter,
    )

    # Estimate budget
    budget = estimate_budget(
        messages=messages,
        config=turn_input.config,
        token_counter=token_counter,
    )

    # Extract system prompt and document context for metadata
    system_prompt = ""
    document_context = ""
    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
            break

    return PreparedTurn(
        messages=messages,
        budget=budget,
        system_prompt=system_prompt,
        document_context=turn_input.snapshot.content,
    )


# -----------------------------------------------------------------------------
# Private Helpers
# -----------------------------------------------------------------------------


def _effective_response_reserve(response_reserve: int, context_limit: int) -> int:
    """Calculate effective response reserve given context limit.

    Ensures reserve doesn't exceed available space after prompt headroom.

    Args:
        response_reserve: Requested response reserve.
        context_limit: Total context window size.

    Returns:
        Adjusted response reserve value.
    """
    limit = max(0, int(context_limit))
    max_reserve = max(0, limit - _PROMPT_HEADROOM)
    return max(0, min(response_reserve, max_reserve))


def _snapshot_to_dict(snapshot: DocumentSnapshot) -> dict[str, Any]:
    """Convert DocumentSnapshot to dict for prompt formatting.

    Args:
        snapshot: Document snapshot.

    Returns:
        Dictionary with snapshot data for prompt templates.
    """
    return {
        "tab_id": snapshot.tab_id,
        "text": snapshot.content,
        "version": snapshot.version_token,
        "length": len(snapshot.content),
        "document_id": snapshot.tab_id,
    }
