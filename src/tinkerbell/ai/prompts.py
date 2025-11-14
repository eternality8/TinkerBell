"""Prompt templates used by the LangChain/LangGraph agent."""

from __future__ import annotations

BASE_PROMPT = """You are an assistant embedded inside a desktop text editor."""


def base_system_prompt() -> str:
    """Return the default system prompt for agent conversations."""

    return BASE_PROMPT


def format_user_prompt(user_prompt: str, doc_snapshot: dict) -> str:
    """Combine user prompt with document metadata."""

    return f"User request: {user_prompt}\nDocument info: {doc_snapshot.get('path', 'untitled')}"

