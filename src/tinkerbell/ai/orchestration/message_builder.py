"""Message construction and token estimation for chat turns."""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from .. import prompts
from .model_types import MessagePlan

if TYPE_CHECKING:
    from ..client import AIClient

LOGGER = logging.getLogger(__name__)
_PROMPT_HEADROOM = 6_000


class MessageBuilder:
    """Builds and manages chat messages with token budget awareness.
    
    Handles message construction, history sanitization, and token estimation
    for chat turn preparation.
    """

    def __init__(
        self,
        client: "AIClient",
        max_context_tokens: int = 128_000,
        response_token_reserve: int = 16_000,
    ) -> None:
        """Initialize the message builder.
        
        Args:
            client: AI client for token counting.
            max_context_tokens: Maximum context window size in tokens.
            response_token_reserve: Token budget reserved for response.
        """
        self._client = client
        self._max_context_tokens = max_context_tokens
        self._response_token_reserve = response_token_reserve

    @property
    def max_context_tokens(self) -> int:
        """Return the current maximum context token limit."""
        return self._max_context_tokens

    @property
    def response_token_reserve(self) -> int:
        """Return the current response token reserve."""
        return self._response_token_reserve

    def update_limits(
        self,
        *,
        max_context_tokens: int | None = None,
        response_token_reserve: int | None = None,
    ) -> None:
        """Update context window limits.
        
        Args:
            max_context_tokens: New maximum context tokens (optional).
            response_token_reserve: New response reserve (optional).
        """
        if max_context_tokens is not None:
            self._max_context_tokens = _normalize_context_tokens(max_context_tokens)
        if response_token_reserve is not None:
            self._response_token_reserve = _normalize_response_reserve(response_token_reserve)

    def build_messages(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> MessagePlan:
        """Build the message sequence for a chat turn.
        
        Constructs system, history, and user messages while respecting
        token budget constraints.
        
        Args:
            prompt: User prompt text.
            snapshot: Document snapshot for context.
            history: Optional conversation history.
            
        Returns:
            MessagePlan with messages, completion budget, and token estimates.
        """
        model_name = self._get_model_name()
        user_prompt = prompts.format_user_prompt(prompt, dict(snapshot), model_name=model_name)
        system_message = {"role": "system", "content": prompts.base_system_prompt(model_name=model_name)}
        user_message = {"role": "user", "content": user_prompt}

        context_limit = self._max_context_tokens
        reserve = self.effective_response_reserve(context_limit)
        prompt_budget = max(0, context_limit - reserve)

        system_tokens = self.estimate_message_tokens(system_message)
        user_tokens = self.estimate_message_tokens(user_message)
        required_prompt_tokens = system_tokens + user_tokens

        if prompt_budget < required_prompt_tokens and reserve > 0:
            shortfall = required_prompt_tokens - prompt_budget
            reclaimed = min(reserve, shortfall)
            reserve -= reclaimed
            prompt_budget += reclaimed

        history_budget = max(0, prompt_budget - required_prompt_tokens)
        trimmed_history: list[dict[str, str]] = []
        if history:
            trimmed_history = self.sanitize_history(history, limit=60, token_budget=history_budget)

        messages: list[dict[str, str]] = [system_message]
        messages.extend(trimmed_history)
        messages.append(user_message)

        completion_budget = reserve if reserve > 0 else None
        prompt_tokens = sum(self.estimate_message_tokens(message) for message in messages)
        return MessagePlan(messages=messages, completion_budget=completion_budget, prompt_tokens=prompt_tokens)

    def sanitize_history(
        self,
        history: Sequence[Mapping[str, Any]],
        limit: int = 20,
        *,
        token_budget: int | None = None,
    ) -> list[dict[str, str]]:
        """Sanitize and trim conversation history to fit budget.
        
        Filters for valid roles, truncates to limit, and trims older
        messages to fit within token budget.
        
        Args:
            history: Raw conversation history.
            limit: Maximum number of messages to include.
            token_budget: Optional token limit for history.
            
        Returns:
            Sanitized list of history messages.
        """
        allowed_roles = {"user", "assistant", "system", "tool"}
        window = list(history)[-limit:] if limit else list(history)
        sanitized: list[dict[str, str]] = []
        for entry in window:
            role = str(entry.get("role", "user")).lower()
            if role not in allowed_roles:
                continue
            text = str(entry.get("content", "")).strip()
            if not text:
                continue
            sanitized.append({"role": role, "content": text})
        if token_budget is None:
            return sanitized
        budget = max(0, token_budget)
        if budget == 0 or not sanitized:
            return []
        trimmed: list[dict[str, str]] = []
        remaining = budget
        for entry in reversed(sanitized):
            tokens = self.estimate_message_tokens(entry)
            if trimmed and tokens > remaining:
                break
            if not trimmed and tokens > remaining:
                trimmed.append(entry)
                break
            trimmed.append(entry)
            remaining -= tokens
            if remaining <= 0:
                break
        trimmed.reverse()
        return trimmed

    def estimate_message_tokens(self, message: Mapping[str, Any]) -> int:
        """Estimate token count for a single message.
        
        Args:
            message: Chat message with 'content' and optionally 'role'.
            
        Returns:
            Estimated token count including message overhead.
        """
        content = str(message.get("content", "") or "")
        tokens = self.estimate_text_tokens(content)
        if message.get("role"):
            tokens += 4  # allowance for role + message boundary tokens
        return tokens

    def estimate_text_tokens(self, text: str) -> int:
        """Estimate token count for raw text.
        
        Uses the AI client's token counter if available, otherwise
        falls back to a byte-based heuristic.
        
        Args:
            text: Raw text to estimate.
            
        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        counter_fn = getattr(self._client, "count_tokens", None)
        if callable(counter_fn):
            try:
                value: Any = counter_fn(text)
                return int(value)
            except Exception:  # pragma: no cover - defensive fallback
                LOGGER.debug("AI client token counter failed; using heuristic", exc_info=True)
        byte_length = len(text.encode("utf-8", errors="ignore"))
        return max(1, math.ceil(byte_length / 4))

    def effective_response_reserve(self, context_limit: int) -> int:
        """Calculate effective response reserve given context limit.
        
        Ensures reserve doesn't exceed available space after prompt headroom.
        
        Args:
            context_limit: Total context window size.
            
        Returns:
            Adjusted response reserve value.
        """
        limit = max(0, int(context_limit))
        max_reserve = max(0, limit - _PROMPT_HEADROOM)
        return max(0, min(self._response_token_reserve, max_reserve))

    def outline_routing_hint(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        outline_digest_cache: dict[str, str] | None = None,
    ) -> str | None:
        """Generate routing hints for outline/retrieval tool usage.
        
        Analyzes the prompt and document context to suggest when to
        use outline or search tools.
        
        Args:
            prompt: User prompt text.
            snapshot: Document snapshot.
            outline_digest_cache: Cache mapping doc_id to outline digest.
            
        Returns:
            Hint message or None if no hints applicable.
        """
        prompt_text = (prompt or "").strip().lower()
        if not prompt_text and not snapshot:
            return None
        doc_id = _resolve_document_id(snapshot)
        raw_text = snapshot.get("text")
        doc_length = snapshot.get("length")
        if isinstance(doc_length, int):
            doc_chars = max(0, doc_length)
        elif isinstance(raw_text, str):
            doc_chars = len(raw_text)
        else:
            try:
                doc_chars = int(snapshot.get("char_count") or 0)
            except (TypeError, ValueError):
                doc_chars = 0

        outline_keywords = (
            "outline",
            "heading",
            "headings",
            "section",
            "sections",
            "toc",
            "table of contents",
            "chapter",
        )
        retrieval_keywords = (
            "find section",
            "find heading",
            "find chapter",
            "locate",
            "where is",
            "which section",
            "quote",
            "passage",
            "excerpt",
        )

        def _contains(text: str, keywords: tuple[str, ...]) -> bool:
            return any(keyword in text for keyword in keywords)

        mentions_outline = _contains(prompt_text, outline_keywords)
        mentions_retrieval = _contains(prompt_text, retrieval_keywords)
        large_doc = doc_chars >= prompts.LARGE_DOC_CHAR_THRESHOLD if doc_chars else False

        guidance: list[str] = []
        if mentions_outline or large_doc:
            reasons: list[str] = []
            if large_doc:
                reasons.append(f"the document is large (~{doc_chars:,} chars)")
            if mentions_outline:
                reasons.append("the user referenced headings/sections")
            reason_text = " and ".join(reasons) if reasons else "document context"
            guidance.append(
                f"Call get_outline first because {reason_text}. Compare outline_digest values to avoid redundant calls."
            )
        if mentions_retrieval:
            guidance.append(
                "After reviewing the outline, call search_document to pull the passages requested before drafting edits."
            )

        digest_hint: str | None = None
        digest = str(snapshot.get("outline_digest") or "").strip()
        cache = outline_digest_cache if outline_digest_cache is not None else {}
        if digest and doc_id:
            previous = cache.get(doc_id)
            cache[doc_id] = digest
            digest_prefix = digest[:8]
            if previous and previous == digest:
                digest_hint = (
                    f"Outline digest {digest_prefix}… matches your prior fetch this session; reuse the previous outline data unless it was marked stale."
                )
            else:
                digest_hint = (
                    f"Outline digest updated to {digest_prefix}…; use this version when reasoning about structure."
                )

        if not guidance and not digest_hint:
            return None

        lines = ["Controller hint:"]
        for entry in guidance:
            lines.append(f"- {entry}")
        if digest_hint:
            lines.append(f"- {digest_hint}")
        if not guidance and digest_hint:
            lines.append("- Only re-run get_outline if the digest changes or the tool reports stale data.")
        return "\n".join(lines)

    def _get_model_name(self) -> str | None:
        """Extract model name from the AI client settings."""
        return getattr(getattr(self._client, "settings", None), "model", None)


def _normalize_context_tokens(value: int | None) -> int:
    """Normalize context token limit to valid bounds.
    
    Args:
        value: Raw context token value.
        
    Returns:
        Clamped value between 32,000 and 512,000.
    """
    default = 128_000
    try:
        candidate = int(value) if value is not None else default
    except (TypeError, ValueError):
        candidate = default
    return max(32_000, min(candidate, 512_000))


def _normalize_response_reserve(value: int | None) -> int:
    """Normalize response reserve to valid bounds.
    
    Args:
        value: Raw response reserve value.
        
    Returns:
        Clamped value between 4,000 and 64,000.
    """
    default = 16_000
    try:
        candidate = int(value) if value is not None else default
    except (TypeError, ValueError):
        candidate = default
    return max(4_000, min(candidate, 64_000))


def _resolve_document_id(snapshot: Mapping[str, Any]) -> str | None:
    """Extract document ID from a snapshot.
    
    Args:
        snapshot: Document snapshot mapping.
        
    Returns:
        Document ID string or None if not found.
    """
    for key in ("document_id", "tab_id", "id"):
        value = snapshot.get(key)
        if value:
            return str(value)
    path = snapshot.get("path")
    if path:
        return str(path)
    version = snapshot.get("version")
    if version:
        return str(version)
    return None
