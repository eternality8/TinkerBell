"""LangGraph wiring stubs."""

from __future__ import annotations

from typing import Any, Dict


def build_agent_graph(*, tools: Dict[str, Any]) -> Dict[str, Any]:
    """Return a placeholder graph representation for later use."""

    return {"nodes": list(tools.keys())}

