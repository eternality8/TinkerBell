"""Unified Tool Registry for AI Agent Tools.

This module provides a clean, declarative registry for all AI tools.
It replaces the legacy registry.py with a simplified, schema-first approach.

WS6.1: New Tool Registry
- Clean parameter schemas
- Unified tool registration
- Category-based organization
- Schema validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tool Categories
# -----------------------------------------------------------------------------


class ToolCategory(Enum):
    """Categories of AI tools."""

    NAVIGATION = auto()  # Reading, browsing, searching
    WRITING = auto()  # Creating and editing documents
    ANALYSIS = auto()  # Analyzing document content
    TRANSFORMATION = auto()  # Large-scale document transformations
    UTILITY = auto()  # Helper tools (diff, validation, etc.)
    SYSTEM = auto()  # Internal/system tools


# -----------------------------------------------------------------------------
# Tool Schema Types
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ParameterSchema:
    """Schema for a single tool parameter.

    Attributes:
        name: Parameter name.
        type: JSON Schema type (string, integer, boolean, object, array).
        description: Human-readable description.
        required: Whether the parameter is required.
        default: Default value if not provided.
        enum: List of allowed values.
        minimum: Minimum value for numbers.
        maximum: Maximum value for numbers.
        min_length: Minimum string length.
        max_length: Maximum string length.
        properties: Nested properties for object types.
        items: Schema for array items.
    """

    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Sequence[Any] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    properties: Mapping[str, "ParameterSchema"] | None = None
    items: "ParameterSchema" | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.default is not None:
            schema["default"] = self.default
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.properties:
            schema["properties"] = {
                k: v.to_json_schema() for k, v in self.properties.items()
            }
            schema["additionalProperties"] = False
        if self.items:
            schema["items"] = self.items.to_json_schema()
        return schema


@dataclass(slots=True)
class ToolSchema:
    """Complete schema for a tool.

    Attributes:
        name: Tool name (identifier).
        description: Human-readable description shown to the model.
        parameters: List of parameters.
        category: Tool category for organization.
        requires_version: Whether tool requires version_token.
        writes_document: Whether tool modifies documents.
        summarizable: Whether output can be summarized.
    """

    name: str
    description: str
    parameters: Sequence[ParameterSchema] = field(default_factory=list)
    category: ToolCategory = ToolCategory.UTILITY
    requires_version: bool = False
    writes_document: bool = False
    summarizable: bool = True

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format for OpenAI function calling."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required

        return schema


# -----------------------------------------------------------------------------
# Tool Registration
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class ToolRegistration:
    """A registered tool with its implementation and schema.

    Attributes:
        schema: Tool schema.
        impl: Tool implementation (callable or tool object).
        enabled: Whether the tool is currently enabled.
        feature_flag: Optional feature flag that controls availability.
    """

    schema: ToolSchema
    impl: Any
    enabled: bool = True
    feature_flag: str | None = None

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.schema.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self.schema.description


@runtime_checkable
class ToolImplementation(Protocol):
    """Protocol for tool implementations."""

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the tool with the given parameters."""
        ...


# -----------------------------------------------------------------------------
# Registration Errors
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class RegistrationFailure:
    """Represents a tool registration failure."""

    name: str
    error: Exception
    recoverable: bool = True


class RegistrationError(RuntimeError):
    """Error raised when tool registration fails."""

    def __init__(self, failures: Sequence[RegistrationFailure]) -> None:
        names = ", ".join(f.name for f in failures)
        super().__init__(f"Failed to register tool(s): {names}")
        self.failures = list(failures)


# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Registry for AI agent tools.

    The registry manages tool registration, schema generation, and
    lookup. It provides a clean interface for the controller to
    access tools and their schemas.

    Example:
        registry = ToolRegistry()
        registry.register(ListTabsTool())
        schema = registry.get_schema("list_tabs")
        tool = registry.get_tool("list_tabs")
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._tools: dict[str, ToolRegistration] = {}
        self._by_category: dict[ToolCategory, list[str]] = {
            cat: [] for cat in ToolCategory
        }

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        tool: Any,
        *,
        schema: ToolSchema | None = None,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        category: ToolCategory | None = None,
        enabled: bool = True,
        feature_flag: str | None = None,
    ) -> None:
        """Register a tool.

        Args:
            tool: Tool implementation.
            schema: Full tool schema (if provided, overrides other args).
            name: Tool name (default: from tool.name or class name).
            description: Tool description.
            parameters: Parameter schema dict.
            category: Tool category.
            enabled: Whether the tool is enabled.
            feature_flag: Optional feature flag.
        """
        if schema is not None:
            tool_schema = schema
        else:
            tool_name = name or getattr(tool, "name", None) or tool.__class__.__name__
            tool_desc = description or getattr(tool, "description", "") or ""
            tool_params = self._parse_parameters(parameters or {})
            tool_cat = category or ToolCategory.UTILITY

            tool_schema = ToolSchema(
                name=tool_name,
                description=tool_desc,
                parameters=tool_params,
                category=tool_cat,
            )

        registration = ToolRegistration(
            schema=tool_schema,
            impl=tool,
            enabled=enabled,
            feature_flag=feature_flag,
        )

        self._tools[tool_schema.name] = registration
        if tool_schema.name not in self._by_category[tool_schema.category]:
            self._by_category[tool_schema.category].append(tool_schema.name)

        LOGGER.debug("Registered tool: %s (category=%s)", tool_schema.name, tool_schema.category.name)

    def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name.

        Returns:
            True if the tool was unregistered, False if not found.
        """
        registration = self._tools.pop(name, None)
        if registration:
            category_list = self._by_category.get(registration.schema.category, [])
            if name in category_list:
                category_list.remove(name)
            LOGGER.debug("Unregistered tool: %s", name)
            return True
        return False

    def _parse_parameters(self, params: Mapping[str, Any]) -> list[ParameterSchema]:
        """Parse parameter dict into ParameterSchema list."""
        result: list[ParameterSchema] = []
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        for name, spec in properties.items():
            param = ParameterSchema(
                name=name,
                type=spec.get("type", "string"),
                description=spec.get("description", ""),
                required=name in required,
                default=spec.get("default"),
                enum=spec.get("enum"),
                minimum=spec.get("minimum"),
                maximum=spec.get("maximum"),
                min_length=spec.get("minLength"),
                max_length=spec.get("maxLength"),
            )
            result.append(param)

        return result

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> Any | None:
        """Get a tool implementation by name."""
        reg = self._tools.get(name)
        return reg.impl if reg and reg.enabled else None

    def get_schema(self, name: str) -> ToolSchema | None:
        """Get a tool schema by name."""
        reg = self._tools.get(name)
        return reg.schema if reg else None

    def get_registration(self, name: str) -> ToolRegistration | None:
        """Get full registration by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered and enabled."""
        reg = self._tools.get(name)
        return reg is not None and reg.enabled

    def list_tools(
        self,
        *,
        category: ToolCategory | None = None,
        enabled_only: bool = True,
    ) -> list[str]:
        """List registered tool names.

        Args:
            category: Filter by category.
            enabled_only: Only include enabled tools.

        Returns:
            List of tool names.
        """
        if category:
            names = self._by_category.get(category, [])
        else:
            names = list(self._tools.keys())

        if enabled_only:
            names = [n for n in names if self._tools[n].enabled]

        return names

    def get_all_schemas(self, *, enabled_only: bool = True) -> list[ToolSchema]:
        """Get all tool schemas.

        Args:
            enabled_only: Only include enabled tools.

        Returns:
            List of tool schemas.
        """
        schemas: list[ToolSchema] = []
        for reg in self._tools.values():
            if not enabled_only or reg.enabled:
                schemas.append(reg.schema)
        return schemas

    def to_openai_tools(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI function calling format.

        Args:
            enabled_only: Only include enabled tools.

        Returns:
            List of tool definitions for OpenAI API.
        """
        tools: list[dict[str, Any]] = []
        for reg in self._tools.values():
            if not enabled_only or reg.enabled:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": reg.schema.name,
                        "description": reg.schema.description,
                        "parameters": reg.schema.to_json_schema(),
                        "strict": True,
                    },
                })
        return tools

    # ------------------------------------------------------------------
    # Enable/Disable
    # ------------------------------------------------------------------

    def enable_tool(self, name: str) -> bool:
        """Enable a tool."""
        reg = self._tools.get(name)
        if reg:
            reg.enabled = True
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """Disable a tool."""
        reg = self._tools.get(name)
        if reg:
            reg.enabled = False
            return True
        return False

    def set_enabled_by_flag(self, flag: str, enabled: bool) -> int:
        """Enable/disable all tools with a specific feature flag.

        Returns:
            Number of tools affected.
        """
        count = 0
        for reg in self._tools.values():
            if reg.feature_flag == flag:
                reg.enabled = enabled
                count += 1
        return count

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        for cat in self._by_category:
            self._by_category[cat] = []


# -----------------------------------------------------------------------------
# Global Registry
# -----------------------------------------------------------------------------

_GLOBAL_REGISTRY: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ToolRegistry()
    return _GLOBAL_REGISTRY


def reset_tool_registry() -> None:
    """Reset the global tool registry (for testing)."""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = None


# -----------------------------------------------------------------------------
# Schema Definitions for New Tools
# -----------------------------------------------------------------------------

# Common parameter schemas used across tools
VERSION_TOKEN_PARAM = ParameterSchema(
    name="version_token",
    type="string",
    description="Version token from read_document (format: 'tab_id:version_id:hash'). Required for write operations.",
    required=True,
)

TAB_ID_PARAM = ParameterSchema(
    name="tab_id",
    type="string",
    description="Target tab identifier. Optional if using version_token.",
    required=False,
)

OFFSET_PARAM = ParameterSchema(
    name="offset",
    type="integer",
    description="Line offset to start reading from (0-based).",
    required=False,
    default=0,
    minimum=0,
)

MAX_LINES_PARAM = ParameterSchema(
    name="max_lines",
    type="integer",
    description="Maximum number of lines to return.",
    required=False,
    minimum=1,
    maximum=2000,
)


# -----------------------------------------------------------------------------
# Tool Schema Definitions
# -----------------------------------------------------------------------------

# WS2: Navigation & Reading Tools

LIST_TABS_SCHEMA = ToolSchema(
    name="list_tabs",
    description="List all open document tabs with their IDs, titles, and status.",
    parameters=[],
    category=ToolCategory.NAVIGATION,
    requires_version=False,
    writes_document=False,
)

READ_DOCUMENT_SCHEMA = ToolSchema(
    name="read_document",
    description=(
        "Read document content and get a version_token. "
        "CALL THIS FIRST before any edit. Returns version_token (required for edits) "
        "and content. Use offset/max_lines for large documents."
    ),
    parameters=[
        TAB_ID_PARAM,
        OFFSET_PARAM,
        MAX_LINES_PARAM,
        ParameterSchema(
            name="include_metadata",
            type="boolean",
            description="Include document metadata (file type, line count, etc.).",
            required=False,
            default=True,
        ),
    ],
    category=ToolCategory.NAVIGATION,
    requires_version=False,
    writes_document=False,
)

SEARCH_DOCUMENT_SCHEMA = ToolSchema(
    name="search_document",
    description=(
        "Search for text in a document using exact match, regex, or semantic search. "
        "Returns line numbers and snippets for matches."
    ),
    parameters=[
        TAB_ID_PARAM,
        ParameterSchema(
            name="query",
            type="string",
            description="Search query string.",
            required=True,
            min_length=1,
        ),
        ParameterSchema(
            name="mode",
            type="string",
            description="Search mode: 'exact', 'regex', or 'semantic'.",
            required=False,
            default="exact",
            enum=["exact", "regex", "semantic"],
        ),
        ParameterSchema(
            name="case_sensitive",
            type="boolean",
            description="Whether to match case (for exact/regex modes).",
            required=False,
            default=False,
        ),
        ParameterSchema(
            name="max_results",
            type="integer",
            description="Maximum number of matches to return.",
            required=False,
            default=20,
            minimum=1,
            maximum=100,
        ),
    ],
    category=ToolCategory.NAVIGATION,
    requires_version=False,
    writes_document=False,
)

GET_OUTLINE_SCHEMA = ToolSchema(
    name="get_outline",
    description=(
        "Get document structure (headings, sections) for navigation. "
        "Returns hierarchical outline with line numbers."
    ),
    parameters=[
        TAB_ID_PARAM,
        ParameterSchema(
            name="max_depth",
            type="integer",
            description="Maximum heading depth to include.",
            required=False,
            minimum=1,
            maximum=6,
        ),
        ParameterSchema(
            name="include_line_numbers",
            type="boolean",
            description="Include line numbers for each heading.",
            required=False,
            default=True,
        ),
    ],
    category=ToolCategory.NAVIGATION,
    requires_version=False,
    writes_document=False,
)


# WS3: Writing Tools

CREATE_DOCUMENT_SCHEMA = ToolSchema(
    name="create_document",
    description="Create a new document tab with optional initial content.",
    parameters=[
        ParameterSchema(
            name="title",
            type="string",
            description="Document title (used as tab name).",
            required=True,
            min_length=1,
        ),
        ParameterSchema(
            name="content",
            type="string",
            description="Initial document content.",
            required=False,
            default="",
        ),
        ParameterSchema(
            name="file_type",
            type="string",
            description="Document type: 'markdown', 'text', 'json', 'yaml'.",
            required=False,
            enum=["markdown", "text", "json", "yaml"],
        ),
    ],
    category=ToolCategory.WRITING,
    requires_version=False,
    writes_document=True,
)

INSERT_LINES_SCHEMA = ToolSchema(
    name="insert_lines",
    description=(
        "Insert NEW lines at a specific position WITHOUT overwriting existing content. "
        "Use after_line=-1 to insert at the beginning."
    ),
    parameters=[
        VERSION_TOKEN_PARAM,
        ParameterSchema(
            name="after_line",
            type="integer",
            description="Insert after this line number (0-based). Use -1 for beginning.",
            required=True,
            minimum=-1,
        ),
        ParameterSchema(
            name="content",
            type="string",
            description="Content to insert.",
            required=True,
        ),
    ],
    category=ToolCategory.WRITING,
    requires_version=True,
    writes_document=True,
)

REPLACE_LINES_SCHEMA = ToolSchema(
    name="replace_lines",
    description=(
        "Replace a range of lines with new content. "
        "REQUIRES version_token from read_document."
    ),
    parameters=[
        VERSION_TOKEN_PARAM,
        ParameterSchema(
            name="start_line",
            type="integer",
            description="First line to replace (0-based, inclusive).",
            required=True,
            minimum=0,
        ),
        ParameterSchema(
            name="end_line",
            type="integer",
            description="Last line to replace (0-based, inclusive).",
            required=True,
            minimum=0,
        ),
        ParameterSchema(
            name="content",
            type="string",
            description="Replacement content.",
            required=True,
        ),
    ],
    category=ToolCategory.WRITING,
    requires_version=True,
    writes_document=True,
)

DELETE_LINES_SCHEMA = ToolSchema(
    name="delete_lines",
    description="Delete a range of lines from the document.",
    parameters=[
        VERSION_TOKEN_PARAM,
        ParameterSchema(
            name="start_line",
            type="integer",
            description="First line to delete (0-based, inclusive).",
            required=True,
            minimum=0,
        ),
        ParameterSchema(
            name="end_line",
            type="integer",
            description="Last line to delete (0-based, inclusive).",
            required=True,
            minimum=0,
        ),
    ],
    category=ToolCategory.WRITING,
    requires_version=True,
    writes_document=True,
)

WRITE_DOCUMENT_SCHEMA = ToolSchema(
    name="write_document",
    description="Replace the ENTIRE document content. Use for full rewrites.",
    parameters=[
        VERSION_TOKEN_PARAM,
        ParameterSchema(
            name="content",
            type="string",
            description="Complete new document content.",
            required=True,
        ),
    ],
    category=ToolCategory.WRITING,
    requires_version=True,
    writes_document=True,
)

FIND_AND_REPLACE_SCHEMA = ToolSchema(
    name="find_and_replace",
    description="Find and replace text throughout the document.",
    parameters=[
        VERSION_TOKEN_PARAM,
        ParameterSchema(
            name="find",
            type="string",
            description="Text or pattern to find.",
            required=True,
            min_length=1,
        ),
        ParameterSchema(
            name="replace",
            type="string",
            description="Replacement text.",
            required=True,
        ),
        ParameterSchema(
            name="is_regex",
            type="boolean",
            description="Interpret 'find' as a regular expression.",
            required=False,
            default=False,
        ),
        ParameterSchema(
            name="case_sensitive",
            type="boolean",
            description="Match case when finding.",
            required=False,
            default=True,
        ),
        ParameterSchema(
            name="max_replacements",
            type="integer",
            description="Maximum number of replacements. Omit for all.",
            required=False,
            minimum=1,
        ),
        ParameterSchema(
            name="dry_run",
            type="boolean",
            description="Preview replacements without applying them.",
            required=False,
            default=False,
        ),
    ],
    category=ToolCategory.WRITING,
    requires_version=True,
    writes_document=True,
)


# WS5: Analysis & Transformation Tools

ANALYZE_DOCUMENT_SCHEMA = ToolSchema(
    name="analyze_document",
    description=(
        "Analyze document content for characters, plot, style, or custom analysis. "
        "Auto-chunks large documents for parallel processing."
    ),
    parameters=[
        TAB_ID_PARAM,
        ParameterSchema(
            name="analysis_type",
            type="string",
            description="Type of analysis to perform.",
            required=True,
            enum=["characters", "plot", "style", "summary", "custom"],
        ),
        ParameterSchema(
            name="custom_prompt",
            type="string",
            description="Custom analysis prompt (required for 'custom' type).",
            required=False,
        ),
        ParameterSchema(
            name="output_format",
            type="string",
            description="Output format: 'markdown', 'json', or 'plain'.",
            required=False,
            default="markdown",
            enum=["markdown", "json", "plain"],
        ),
        ParameterSchema(
            name="max_chunks",
            type="integer",
            description="Maximum number of chunks to analyze in parallel.",
            required=False,
            default=8,
            minimum=1,
            maximum=16,
        ),
    ],
    category=ToolCategory.ANALYSIS,
    requires_version=False,
    writes_document=False,
)

TRANSFORM_DOCUMENT_SCHEMA = ToolSchema(
    name="transform_document",
    description=(
        "Apply document-wide transformations like character renames, "
        "setting changes, or style rewrites. Creates new tab by default."
    ),
    parameters=[
        TAB_ID_PARAM,
        ParameterSchema(
            name="transformation_type",
            type="string",
            description="Type of transformation to apply.",
            required=True,
            enum=["character_rename", "setting_change", "style_rewrite", "tense_change", "pov_change", "custom"],
        ),
        ParameterSchema(
            name="old_name",
            type="string",
            description="Current character name (for character_rename).",
            required=False,
        ),
        ParameterSchema(
            name="new_name",
            type="string",
            description="New character name (for character_rename).",
            required=False,
        ),
        ParameterSchema(
            name="aliases",
            type="array",
            description="Character name aliases to also rename.",
            required=False,
        ),
        ParameterSchema(
            name="setting_description",
            type="string",
            description="New setting description (for setting_change).",
            required=False,
        ),
        ParameterSchema(
            name="target_style",
            type="string",
            description="Target writing style (for style_rewrite).",
            required=False,
        ),
        ParameterSchema(
            name="target_tense",
            type="string",
            description="Target tense: 'past' or 'present' (for tense_change).",
            required=False,
            enum=["past", "present"],
        ),
        ParameterSchema(
            name="target_pov",
            type="string",
            description="Target POV: 'first', 'second', 'third' (for pov_change).",
            required=False,
            enum=["first", "second", "third"],
        ),
        ParameterSchema(
            name="custom_prompt",
            type="string",
            description="Custom transformation prompt (for 'custom' type).",
            required=False,
        ),
        ParameterSchema(
            name="output_mode",
            type="string",
            description="Output mode: 'new_tab' (default) or 'in_place' (requires version_token).",
            required=False,
            default="new_tab",
            enum=["new_tab", "in_place"],
        ),
        ParameterSchema(
            name="version_token",
            type="string",
            description="Version token (required for 'in_place' output_mode).",
            required=False,
        ),
    ],
    category=ToolCategory.TRANSFORMATION,
    requires_version=False,  # Only required for in_place mode
    writes_document=True,
)


# Utility Tools

DIFF_BUILDER_SCHEMA = ToolSchema(
    name="diff_builder",
    description="Generate a unified diff between two text snippets.",
    parameters=[
        ParameterSchema(
            name="original",
            type="string",
            description="Original text.",
            required=True,
        ),
        ParameterSchema(
            name="updated",
            type="string",
            description="Updated text.",
            required=True,
        ),
        ParameterSchema(
            name="context_lines",
            type="integer",
            description="Number of context lines to include.",
            required=False,
            default=3,
            minimum=0,
            maximum=10,
        ),
    ],
    category=ToolCategory.UTILITY,
    requires_version=False,
    writes_document=False,
)

VALIDATE_SNIPPET_SCHEMA = ToolSchema(
    name="validate_snippet",
    description="Validate YAML/JSON snippets before inserting.",
    parameters=[
        ParameterSchema(
            name="text",
            type="string",
            description="Snippet content to validate.",
            required=True,
        ),
        ParameterSchema(
            name="format",
            type="string",
            description="Expected format.",
            required=True,
            enum=["yaml", "json", "markdown"],
        ),
    ],
    category=ToolCategory.UTILITY,
    requires_version=False,
    writes_document=False,
)


# All schema definitions
ALL_TOOL_SCHEMAS: dict[str, ToolSchema] = {
    # Navigation
    "list_tabs": LIST_TABS_SCHEMA,
    "read_document": READ_DOCUMENT_SCHEMA,
    "search_document": SEARCH_DOCUMENT_SCHEMA,
    "get_outline": GET_OUTLINE_SCHEMA,
    # Writing
    "create_document": CREATE_DOCUMENT_SCHEMA,
    "insert_lines": INSERT_LINES_SCHEMA,
    "replace_lines": REPLACE_LINES_SCHEMA,
    "delete_lines": DELETE_LINES_SCHEMA,
    "write_document": WRITE_DOCUMENT_SCHEMA,
    "find_and_replace": FIND_AND_REPLACE_SCHEMA,
    # Analysis & Transformation
    "analyze_document": ANALYZE_DOCUMENT_SCHEMA,
    "transform_document": TRANSFORM_DOCUMENT_SCHEMA,
    # Utility
    "diff_builder": DIFF_BUILDER_SCHEMA,
    "validate_snippet": VALIDATE_SNIPPET_SCHEMA,
}


__all__ = [
    # Enums
    "ToolCategory",
    # Schema types
    "ParameterSchema",
    "ToolSchema",
    "ToolRegistration",
    "ToolImplementation",
    # Errors
    "RegistrationFailure",
    "RegistrationError",
    # Registry
    "ToolRegistry",
    "get_tool_registry",
    "reset_tool_registry",
    # Common parameters
    "VERSION_TOKEN_PARAM",
    "TAB_ID_PARAM",
    "OFFSET_PARAM",
    "MAX_LINES_PARAM",
    # Tool schemas
    "LIST_TABS_SCHEMA",
    "READ_DOCUMENT_SCHEMA",
    "SEARCH_DOCUMENT_SCHEMA",
    "GET_OUTLINE_SCHEMA",
    "CREATE_DOCUMENT_SCHEMA",
    "INSERT_LINES_SCHEMA",
    "REPLACE_LINES_SCHEMA",
    "DELETE_LINES_SCHEMA",
    "WRITE_DOCUMENT_SCHEMA",
    "FIND_AND_REPLACE_SCHEMA",
    "ANALYZE_DOCUMENT_SCHEMA",
    "TRANSFORM_DOCUMENT_SCHEMA",
    "DIFF_BUILDER_SCHEMA",
    "VALIDATE_SNIPPET_SCHEMA",
    "ALL_TOOL_SCHEMAS",
]
