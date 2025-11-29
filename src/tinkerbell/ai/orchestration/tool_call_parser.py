"""Tool call parsing utilities for embedded tool call markers.

This module provides functions to parse tool calls embedded in model responses
using delimited text markers (e.g., <|tool_calls_begin|>...<|tool_calls_end|>).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Mapping, Sequence

__all__ = [
    "TOOL_MARKER_TRANSLATION",
    "TOOL_CALLS_BLOCK_RE",
    "TOOL_CALL_ENTRY_RE",
    "parse_embedded_tool_calls",
    "parse_tool_call_entries",
    "normalize_tool_marker_text",
    "parsed_tool_call_id",
    "try_parse_json_block",
]

# Normalizes stylized glyphs inside <|tool ...|> markers emitted by some models.
TOOL_MARKER_TRANSLATION = str.maketrans(
    {
        ord("＜"): "<",
        ord("﹤"): "<",
        ord("〈"): "<",
        ord("《"): "<",
        ord("＞"): ">",
        ord("﹥"): ">",
        ord("〉"): ">",
        ord("》"): ">",
        ord("｜"): "|",
        ord("￨"): "|",
        ord("│"): "|",
        ord("︱"): "|",
        ord("︲"): "|",
        ord("▁"): "_",
        ord("\u00a0"): " ",
        ord("\u1680"): " ",
        ord("\u2000"): " ",
        ord("\u2001"): " ",
        ord("\u2002"): " ",
        ord("\u2003"): " ",
        ord("\u2004"): " ",
        ord("\u2005"): " ",
        ord("\u2006"): " ",
        ord("\u2007"): " ",
        ord("\u2008"): " ",
        ord("\u2009"): " ",
        ord("\u200a"): " ",
        ord("\u200b"): " ",
        ord("\u200c"): " ",
        ord("\u200d"): " ",
        ord("\u202f"): " ",
        ord("\u205f"): " ",
        ord("\u3000"): " ",
        ord("\ufeff"): " ",
    }
)

TOOL_CALLS_BLOCK_RE = re.compile(
    r"<\s*\|?\s*tool[\s_]*calls[\s_]*begin\s*\|?\s*>(?P<body>.*?)<\s*\|?\s*tool[\s_]*calls[\s_]*end\s*\|?\s*>",
    re.IGNORECASE | re.DOTALL,
)

TOOL_CALL_ENTRY_RE = re.compile(
    r"<\s*\|?\s*tool[\s_]*call[\s_]*begin\s*\|?\s*>(?P<name>.*?)<\s*\|?\s*tool[\s_]*sep\s*\|?\s*>(?P<args>.*?)<\s*\|?\s*tool[\s_]*call[\s_]*end\s*\|?\s*>",
    re.IGNORECASE | re.DOTALL,
)


def normalize_tool_marker_text(text: str) -> str:
    """Normalize stylized Unicode glyphs to ASCII equivalents for tool parsing."""
    return text.translate(TOOL_MARKER_TRANSLATION)


def parse_embedded_tool_calls(
    text: str,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """Parse tool calls embedded in text using delimited markers.

    Looks for <|tool_calls_begin|>...<|tool_calls_end|> blocks and extracts
    individual tool calls from within.

    Args:
        text: The text to parse for embedded tool calls.
        start_index: Starting index for tool call numbering.

    Returns:
        List of tool call dictionaries with keys: id, name, arguments, index.
    """
    if not text or not isinstance(text, str):
        return []
    normalized = normalize_tool_marker_text(text)
    match = TOOL_CALLS_BLOCK_RE.search(normalized)
    if not match:
        return []
    body = match.group("body") or ""
    return parse_tool_call_entries(body, start_index)


def parse_tool_call_entries(
    body: str,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """Parse individual tool call entries from a tool calls block body.

    Args:
        body: The inner text of a tool_calls block.
        start_index: Starting index for tool call numbering.

    Returns:
        List of tool call dictionaries.
    """
    if not body or not isinstance(body, str):
        return []
    calls: list[dict[str, Any]] = []
    normalized = normalize_tool_marker_text(body)
    for idx, entry_match in enumerate(TOOL_CALL_ENTRY_RE.finditer(normalized)):
        name_raw = (entry_match.group("name") or "").strip()
        args_raw = (entry_match.group("args") or "").strip()
        name = name_raw.strip("\"' \t\n\r")
        parsed = try_parse_json_block(args_raw)
        call_id = parsed_tool_call_id(name, idx + start_index)
        calls.append({
            "id": call_id,
            "name": name,
            "arguments": args_raw,
            "index": idx + start_index,
        })
    return calls


def parsed_tool_call_id(name: str, index: int) -> str:
    """Generate a unique tool call ID for parsed tool calls."""
    return f"parsed_{name}_{index}_{uuid.uuid4().hex[:8]}"


def try_parse_json_block(text: str) -> dict[str, Any] | None:
    """Attempt to parse text as JSON, returning None on failure."""
    if not text:
        return None
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    return None
