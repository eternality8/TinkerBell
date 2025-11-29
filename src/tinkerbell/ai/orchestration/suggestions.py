"""Follow-up suggestion generation for chat conversations."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AIClient

_SUGGESTION_SYSTEM_PROMPT = (
    "You are a helpful writing copilot asked to propose up to {max_suggestions} focused follow-up suggestions. "
    "Each suggestion should be a short imperative phrase (no numbering) tailored to the prior conversation transcript."
)


class SuggestionGenerator:
    """Generates contextual follow-up suggestions based on chat history."""

    def __init__(self, client: "AIClient", temperature: float = 0.2) -> None:
        self._client = client
        self._temperature = temperature

    async def generate(
        self,
        history: Sequence[Mapping[str, str]],
        *,
        max_suggestions: int = 4,
    ) -> list[str]:
        """Generate follow-up suggestions from conversation history.
        
        Args:
            history: Sequence of chat messages with 'role' and 'content' keys.
            max_suggestions: Maximum number of suggestions to return.
            
        Returns:
            List of suggestion strings (at most max_suggestions).
        """
        if not history:
            return []
        messages = self._build_messages(history, max_suggestions)
        response_text = await self._complete_chat(messages)
        return self._parse_response(response_text, max_suggestions)

    def _build_messages(
        self,
        history: Sequence[Mapping[str, str]],
        max_suggestions: int,
    ) -> list[dict[str, str]]:
        """Build the prompt messages for suggestion generation.
        
        Args:
            history: Recent conversation history.
            max_suggestions: Maximum suggestions requested.
            
        Returns:
            List of system and user messages for the completion.
        """
        transcript_lines: list[str] = []
        for entry in history[-10:]:
            role = (entry.get("role") or "user").lower()
            label = "User" if role == "user" else "Assistant"
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            if len(content) > 400:
                content = f"{content[:397].rstrip()}…"
            transcript_lines.append(f"{label}: {content}")

        transcript = "\n".join(transcript_lines) or "(no content)"
        system_prompt = _SUGGESTION_SYSTEM_PROMPT.format(max_suggestions=max(1, max_suggestions))
        user_prompt = (
            "Here is the recent conversation between the user and assistant:\n"
            f"{transcript}\n\n"
            "Suggest the next helpful questions or commands the user could ask to keep making progress."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _complete_chat(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Execute a simple chat completion for suggestions.
        
        Args:
            messages: Prompt messages for the AI client.
            
        Returns:
            Response text from the model.
        """
        chunks: list[str] = []
        final_chunk: str | None = None
        async for event in self._client.stream_chat(
            messages=messages,
            temperature=self._temperature,
            max_tokens=300,
        ):
            if event.type == "content.delta" and event.content:
                chunks.append(str(event.content))
            elif event.type == "content.done" and event.content:
                final_chunk = str(event.content)
        response = "".join(chunks)
        if final_chunk and final_chunk not in response:
            response += final_chunk
        return response.strip()

    def _parse_response(self, text: str, max_suggestions: int) -> list[str]:
        """Parse the model response into a list of suggestions.
        
        Args:
            text: Raw response text from the model.
            max_suggestions: Maximum suggestions to return.
            
        Returns:
            Parsed and sanitized suggestion list.
        """
        if not text:
            return []

        suggestions = self._try_parse_json(text, max_suggestions)
        if suggestions:
            return suggestions

        lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
        return sanitize_suggestions(lines, max_suggestions)

    def _try_parse_json(self, text: str, max_suggestions: int) -> list[str]:
        """Attempt to parse JSON-formatted suggestions.
        
        Args:
            text: Response text that may contain JSON.
            max_suggestions: Maximum suggestions to return.
            
        Returns:
            Parsed suggestions or empty list if not valid JSON.
        """
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, dict):
            if "suggestions" in parsed and isinstance(parsed["suggestions"], list):
                parsed = parsed["suggestions"]
            else:
                parsed = list(parsed.values())

        if isinstance(parsed, list):
            return sanitize_suggestions(parsed, max_suggestions)
        return []


def sanitize_suggestions(raw_items: Iterable[Any], max_suggestions: int) -> list[str]:
    """Deduplicate and limit suggestion items.
    
    Args:
        raw_items: Raw suggestion values from parsing.
        max_suggestions: Maximum suggestions to return.
        
    Returns:
        Sanitized list of unique suggestion strings.
    """
    sanitized: list[str] = []
    seen: set[str] = set()
    limit = max(1, max_suggestions)
    for item in raw_items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        sanitized.append(text)
        seen.add(text)
        if len(sanitized) >= limit:
            break
    return sanitized
