"""Agent executor façade wrapping LangChain/LangGraph interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class AIController:
    """High-level interface invoked by the chat panel."""

    graph: Dict[str, Any]

    async def run_chat(self, prompt: str, doc_snapshot: dict) -> dict:
        """Execute a chat turn against the compiled agent graph."""

        return {"prompt": prompt, "doc": doc_snapshot, "graph": self.graph}

