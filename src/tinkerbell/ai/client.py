"""Async AI client wrapper built around OpenAI-compatible endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterable, List


@dataclass(slots=True)
class ClientSettings:
    """Subset of settings required to configure the AI client."""

    base_url: str
    api_key: str
    model: str


class AIClient:
    """Async client placeholder providing streaming helpers."""

    def __init__(self, settings: ClientSettings) -> None:
        self._settings = settings

    async def stream_chat(self, messages: Iterable[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream chat completions for the provided messages."""

        del messages
        yield "[streaming not yet implemented]"

    async def list_models(self) -> List[str]:
        """Return a list of supported model identifiers."""

        return [self._settings.model]

