"""Tool for extracting document outline/structure.

WS2.4: Implements Markdown heading detection, JSON/YAML structure detection,
and plain text heuristics with confidence scoring.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .base import ReadOnlyTool, ToolContext
from .errors import (
    InvalidTabIdError,
    TabNotFoundError,
    UnsupportedFileTypeError,
)
from .list_tabs import detect_file_type, is_supported_file_type
from .version import VersionManager, compute_content_hash, get_version_manager


# Try to import yaml parser
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None  # type: ignore


@dataclass(slots=True)
class OutlineNode:
    """Represents a node in the document outline."""

    title: str
    level: int
    line_start: int
    line_end: int | None = None  # None for last section
    children: list["OutlineNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "title": self.title,
            "level": self.level,
            "line_start": self.line_start,
        }
        if self.line_end is not None:
            result["line_end"] = self.line_end
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result


def split_lines(text: str) -> list[str]:
    """Split text into lines."""
    if not text:
        return []
    return text.split("\n")


# -----------------------------------------------------------------------------
# Markdown Outline Detection
# -----------------------------------------------------------------------------

# Pattern for ATX-style headings: # Heading
MARKDOWN_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#*)?$")

# Pattern for Setext-style headings (underlined)
MARKDOWN_SETEXT_H1 = re.compile(r"^=+\s*$")
MARKDOWN_SETEXT_H2 = re.compile(r"^-+\s*$")


def detect_markdown_outline(text: str) -> tuple[list[OutlineNode], str]:
    """Extract outline from Markdown document.

    Args:
        text: Markdown document content.

    Returns:
        Tuple of (outline nodes, detection_method).
    """
    lines = split_lines(text)
    nodes: list[OutlineNode] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check ATX-style heading
        match = MARKDOWN_ATX_HEADING.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            nodes.append(OutlineNode(
                title=title,
                level=level,
                line_start=i,
            ))
            i += 1
            continue

        # Check Setext-style heading (line followed by === or ---)
        if i + 1 < len(lines) and line.strip():
            next_line = lines[i + 1]
            if MARKDOWN_SETEXT_H1.match(next_line) and len(next_line.strip()) >= 2:
                nodes.append(OutlineNode(
                    title=line.strip(),
                    level=1,
                    line_start=i,
                ))
                i += 2
                continue
            elif MARKDOWN_SETEXT_H2.match(next_line) and len(next_line.strip()) >= 2:
                nodes.append(OutlineNode(
                    title=line.strip(),
                    level=2,
                    line_start=i,
                ))
                i += 2
                continue

        i += 1

    # Set line_end for each node (until next sibling or parent's end)
    _set_line_ends(nodes, len(lines) - 1)

    # Build hierarchy
    root = _build_hierarchy(nodes)

    return root, "markdown_headings"


# -----------------------------------------------------------------------------
# JSON Outline Detection
# -----------------------------------------------------------------------------

def detect_json_outline(text: str) -> tuple[list[OutlineNode], str]:
    """Extract outline from JSON document.

    Args:
        text: JSON document content.

    Returns:
        Tuple of (outline nodes, detection_method).
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], "json_parse_error"

    lines = split_lines(text)
    nodes: list[OutlineNode] = []

    if isinstance(data, dict):
        # Find line numbers for top-level keys
        for key in data.keys():
            line_num = _find_json_key_line(lines, key)
            nodes.append(OutlineNode(
                title=key,
                level=1,
                line_start=line_num,
            ))

            # Add nested keys if value is a dict
            value = data[key]
            if isinstance(value, dict):
                for subkey in value.keys():
                    subline = _find_json_key_line(lines, subkey, start=line_num)
                    nodes.append(OutlineNode(
                        title=f"{key}.{subkey}",
                        level=2,
                        line_start=subline,
                    ))

    elif isinstance(data, list) and data:
        # For arrays, show count and first few items
        nodes.append(OutlineNode(
            title=f"Array ({len(data)} items)",
            level=1,
            line_start=0,
        ))

    # Set line_end
    _set_line_ends(nodes, len(lines) - 1)

    return nodes, "json_structure"


def _find_json_key_line(lines: list[str], key: str, start: int = 0) -> int:
    """Find the line number where a JSON key appears."""
    pattern = re.compile(rf'^\s*["\']?{re.escape(key)}["\']?\s*:')
    for i, line in enumerate(lines[start:], start=start):
        if pattern.match(line):
            return i
    return start


# -----------------------------------------------------------------------------
# YAML Outline Detection
# -----------------------------------------------------------------------------

def detect_yaml_outline(text: str) -> tuple[list[OutlineNode], str]:
    """Extract outline from YAML document.

    Args:
        text: YAML document content.

    Returns:
        Tuple of (outline nodes, detection_method).
    """
    if not HAS_YAML:
        return [], "yaml_parser_unavailable"

    try:
        data = yaml.safe_load(text)
    except Exception:
        return [], "yaml_parse_error"

    if not isinstance(data, dict):
        return [], "yaml_not_mapping"

    lines = split_lines(text)
    nodes: list[OutlineNode] = []

    # Find top-level keys in YAML (no leading whitespace)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check for top-level key (no leading whitespace, ends with colon)
        if line and not line[0].isspace():
            match = re.match(r"^([^:]+):\s*", line)
            if match:
                key = match.group(1).strip()
                nodes.append(OutlineNode(
                    title=key,
                    level=1,
                    line_start=i,
                ))

    # Set line_end
    _set_line_ends(nodes, len(lines) - 1)

    return nodes, "yaml_structure"


# -----------------------------------------------------------------------------
# Plain Text Outline Detection (Heuristics)
# -----------------------------------------------------------------------------

# Common chapter patterns
CHAPTER_PATTERNS = [
    # "Chapter 1", "CHAPTER ONE", "Chapter I"
    re.compile(r"^(?:chapter|part|section|act|book)\s+[\divxlcm]+(?:\s*[:.]\s*.*)?$", re.IGNORECASE),
    # "Chapter 1: Title"
    re.compile(r"^(?:chapter|part|section|act|book)\s+[\divxlcm]+\s*:\s*.+$", re.IGNORECASE),
    # "1. Title" at start of line
    re.compile(r"^\d+\.\s+[A-Z]"),
    # "I. Title" (Roman numerals)
    re.compile(r"^[IVXLCDM]+\.\s+[A-Z]"),
]

# ALL CAPS line (potential heading)
ALL_CAPS_PATTERN = re.compile(r"^[A-Z][A-Z\s\-]+[A-Z]$")

# Separator patterns (scene breaks)
SEPARATOR_PATTERNS = [
    re.compile(r"^[\*\-=_]{3,}$"),  # ***, ---, ===, ___
    re.compile(r"^(?:\*\s*){3,}$"),  # * * *
]


def detect_plaintext_outline(text: str) -> tuple[list[OutlineNode], str, str]:
    """Extract outline from plain text using heuristics.

    Args:
        text: Plain text document content.

    Returns:
        Tuple of (outline nodes, detection_method, confidence).
    """
    lines = split_lines(text)
    nodes: list[OutlineNode] = []
    detection_method = "heuristic_unknown"
    confidence = "low"

    # Strategy 1: Look for chapter markers
    chapter_nodes = _detect_chapters(lines)
    if chapter_nodes:
        nodes = chapter_nodes
        detection_method = "chapter_markers"
        confidence = "high"
    else:
        # Strategy 2: Look for ALL CAPS headings
        caps_nodes = _detect_caps_headings(lines)
        if caps_nodes:
            nodes = caps_nodes
            detection_method = "caps_headings"
            confidence = "medium"
        else:
            # Strategy 3: Look for paragraph breaks (scene-based)
            para_nodes = _detect_paragraph_sections(lines)
            if para_nodes:
                nodes = para_nodes
                detection_method = "paragraph_breaks"
                confidence = "low"

    # Set line_end
    _set_line_ends(nodes, len(lines) - 1)

    return nodes, detection_method, confidence


def _detect_chapters(lines: list[str]) -> list[OutlineNode]:
    """Detect chapter-style headings."""
    nodes: list[OutlineNode] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        for pattern in CHAPTER_PATTERNS:
            if pattern.match(stripped):
                # Clean up the title
                title = stripped
                nodes.append(OutlineNode(
                    title=title,
                    level=1,
                    line_start=i,
                ))
                break

    return nodes


def _detect_caps_headings(lines: list[str]) -> list[OutlineNode]:
    """Detect ALL CAPS headings."""
    nodes: list[OutlineNode] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue

        # Check if line is ALL CAPS and looks like a heading
        if ALL_CAPS_PATTERN.match(stripped) and len(stripped) < 80:
            # Verify it's surrounded by blank lines or at start
            prev_blank = i == 0 or not lines[i - 1].strip()
            next_blank = i == len(lines) - 1 or not lines[i + 1].strip()

            if prev_blank or next_blank:
                nodes.append(OutlineNode(
                    title=stripped.title(),  # Convert to title case
                    level=1,
                    line_start=i,
                ))

    return nodes


def _detect_paragraph_sections(lines: list[str], min_gap: int = 2) -> list[OutlineNode]:
    """Detect sections based on paragraph breaks.

    This is a fallback for unstructured text.
    """
    nodes: list[OutlineNode] = []
    blank_count = 0
    section_start = 0
    section_num = 1

    for i, line in enumerate(lines):
        if not line.strip():
            blank_count += 1
        else:
            if blank_count >= min_gap and i > 0:
                # Found a section break
                # Get first line of new section as title
                title = line.strip()[:50]
                if len(line.strip()) > 50:
                    title += "..."

                nodes.append(OutlineNode(
                    title=f"Section {section_num}: {title}",
                    level=1,
                    line_start=i,
                ))
                section_num += 1
            blank_count = 0

    # Only return if we found meaningful sections
    if len(nodes) >= 2:
        return nodes
    return []


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def _set_line_ends(nodes: list[OutlineNode], last_line: int) -> None:
    """Set line_end for each node based on the next sibling."""
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            # End before next sibling starts
            node.line_end = nodes[i + 1].line_start - 1
        else:
            # Last node extends to end of document
            node.line_end = None  # Indicates "to end"


def _build_hierarchy(nodes: list[OutlineNode]) -> list[OutlineNode]:
    """Build a hierarchical structure from flat nodes based on level."""
    if not nodes:
        return []

    root: list[OutlineNode] = []
    stack: list[OutlineNode] = []

    for node in nodes:
        # Pop stack until we find a parent
        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            # Add as child of current stack top
            stack[-1].children.append(node)
        else:
            # Add to root
            root.append(node)

        stack.append(node)

    return root


# -----------------------------------------------------------------------------
# Main Tool Class
# -----------------------------------------------------------------------------

@dataclass
class GetOutlineTool(ReadOnlyTool):
    """Extract document outline/structure.

    This tool analyzes document structure and returns an outline with:
    - Section titles and levels
    - Line ranges for each section
    - Hierarchical nesting where applicable

    Supports different detection strategies based on file type:
    - Markdown: ATX and Setext headings
    - JSON: Top-level and nested keys
    - YAML: Top-level keys
    - Plain text: Chapter markers, ALL CAPS headings, paragraph heuristics

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab)

    Response includes:
        - outline: Hierarchical array of sections
        - detection_method: How the outline was detected
        - detection_confidence: high/medium/low
        - file_type: Detected file type
        - version: Version token for subsequent operations
    """

    name: ClassVar[str] = "get_outline"
    description: ClassVar[str] = "Extract document outline and structure"
    summarizable: ClassVar[bool] = True

    version_manager: VersionManager = field(default_factory=get_version_manager)

    def read(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the get_outline tool.

        Args:
            context: Tool execution context.
            params: Tool parameters.

        Returns:
            Document outline with metadata.
        """
        # Resolve tab ID
        tab_id = context.resolve_tab_id(params.get("tab_id"))
        if tab_id is None:
            raise InvalidTabIdError(
                message="No tab specified and no active tab available",
                tab_id=params.get("tab_id"),
            )

        # Get document content
        content = context.document_provider.get_document_content(tab_id)
        if content is None:
            raise TabNotFoundError(
                message=f"Tab '{tab_id}' not found",
                tab_id=tab_id,
            )

        # Get file metadata for type detection
        metadata = context.document_provider.get_document_metadata(tab_id)
        path = metadata.get("path") if metadata else None
        language = metadata.get("language") if metadata else None
        file_type = detect_file_type(path, language)

        if not is_supported_file_type(file_type):
            raise UnsupportedFileTypeError(
                message=f"Cannot analyze {file_type} files",
                file_type=file_type,
                file_path=path,
            )

        # Handle empty document
        if not content or not content.strip():
            return self._empty_response(tab_id, file_type, content or "")

        # Detect outline based on file type
        nodes: list[OutlineNode] = []
        detection_method = "none"
        confidence = "low"

        if file_type == "markdown":
            nodes, detection_method = detect_markdown_outline(content)
            confidence = "high" if nodes else "low"
        elif file_type == "json":
            nodes, detection_method = detect_json_outline(content)
            confidence = "high" if nodes else "low"
        elif file_type == "yaml":
            nodes, detection_method = detect_yaml_outline(content)
            confidence = "high" if nodes else "medium"
        else:
            # Plain text heuristics
            nodes, detection_method, confidence = detect_plaintext_outline(content)

        # Register version
        content_hash = compute_content_hash(content)
        if not self.version_manager.get_current_token(tab_id):
            doc_id = uuid.uuid4().hex
            self.version_manager.register_tab(tab_id, doc_id, content_hash)
        token = self.version_manager.get_current_token(tab_id)

        # Build response
        lines = split_lines(content)
        outline = [node.to_dict() for node in nodes]

        # Add suggestion for unstructured text
        suggestion = None
        if not nodes or confidence == "low":
            suggestion = "Document appears unstructured. Consider adding headings for better navigation."

        return {
            "outline": outline,
            "total_sections": len(nodes),
            "detection_method": detection_method,
            "detection_confidence": confidence,
            "file_type": file_type,
            "total_lines": len(lines),
            "tab_id": tab_id,
            "version": token.to_string() if token else None,
            "suggestion": suggestion,
        }

    def _empty_response(self, tab_id: str, file_type: str, content: str) -> dict[str, Any]:
        """Build response for empty document."""
        content_hash = compute_content_hash(content)
        if not self.version_manager.get_current_token(tab_id):
            doc_id = uuid.uuid4().hex
            self.version_manager.register_tab(tab_id, doc_id, content_hash)
        token = self.version_manager.get_current_token(tab_id)

        return {
            "outline": [],
            "total_sections": 0,
            "detection_method": "empty_document",
            "detection_confidence": "high",
            "file_type": file_type,
            "total_lines": 0,
            "tab_id": tab_id,
            "version": token.to_string() if token else None,
            "suggestion": "Document is empty.",
        }


# Factory function for creating the tool
def create_get_outline_tool(version_manager: VersionManager | None = None) -> GetOutlineTool:
    """Create a GetOutlineTool instance.

    Args:
        version_manager: Optional version manager.

    Returns:
        Configured GetOutlineTool instance.
    """
    if version_manager is None:
        version_manager = get_version_manager()
    return GetOutlineTool(version_manager=version_manager)
