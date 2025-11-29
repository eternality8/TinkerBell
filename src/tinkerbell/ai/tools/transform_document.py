"""Transform Document Tool for AI operations.

Provides document transformation capabilities including character renaming,
setting changes, style rewrites, and custom transformations.
Uses subagents for parallel chunk processing of large documents.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, Coroutine, Protocol

from .base import SubagentTool, ToolContext
from .errors import (
    ContentRequiredError,
    InvalidParameterError,
    TabNotFoundError,
    ToolError,
)
from .subagent import (
    ChunkCoordinator,
    ChunkSpec,
    SubagentOrchestrator,
    SubagentResult,
    SubagentTask,
    SubagentType,
    TaskPriority,
    TransformAggregator,
    estimate_tokens,
)
from .version import VersionToken

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Transformation Types
# =============================================================================


class TransformationType(Enum):
    """Type of document transformation to perform."""

    CHARACTER_RENAME = "character_rename"  # Rename a character
    SETTING_CHANGE = "setting_change"  # Change setting/location
    STYLE_REWRITE = "style_rewrite"  # Rewrite in different style
    TENSE_CHANGE = "tense_change"  # Change narrative tense
    POV_CHANGE = "pov_change"  # Change point of view
    CUSTOM = "custom"  # Custom transformation


class OutputMode(Enum):
    """How to output transformation results."""

    NEW_TAB = "new_tab"  # Create a new tab with transformed content
    IN_PLACE = "in_place"  # Replace content in current tab (requires version)
    PREVIEW = "preview"  # Return preview without applying


# Transformation instructions
TRANSFORM_INSTRUCTIONS = {
    TransformationType.CHARACTER_RENAME: """Transform this text chunk by renaming a character.

Instructions:
- Replace all occurrences of "{old_name}" with "{new_name}"
- Update pronouns if specified: {pronoun_update}
- Maintain possessive forms (e.g., "{old_name}'s" → "{new_name}'s")
- Handle name variations (nicknames, titles) if provided
- Preserve formatting and surrounding context

Return as JSON:
{{
    "transformed_content": "string with renamed character",
    "replacements": number,
    "locations": [line_numbers]
}}""",

    TransformationType.SETTING_CHANGE: """Transform this text chunk by changing the setting/location.

Instructions:
- Change references from "{old_setting}" to "{new_setting}"
- Update related cultural details: {cultural_details}
- Adapt descriptions, landmarks, customs as appropriate
- Maintain narrative consistency

Return as JSON:
{{
    "transformed_content": "string with changed setting",
    "replacements": number,
    "adaptations": ["list of cultural adaptations made"]
}}""",

    TransformationType.STYLE_REWRITE: """Rewrite this text chunk in a different style.

Target style: {target_style}
Style guidelines: {style_guidelines}

Instructions:
- Maintain the same plot/events/information
- Adapt vocabulary, sentence structure, and tone to match target style
- Preserve character voices where appropriate
- Keep the same overall length (within 20%)

Return as JSON:
{{
    "transformed_content": "string in new style",
    "style_changes": ["list of key style changes made"]
}}""",

    TransformationType.TENSE_CHANGE: """Transform this text chunk by changing the narrative tense.

From: {from_tense}
To: {to_tense}

Instructions:
- Convert all narrative verbs from {from_tense} to {to_tense} tense
- Keep dialogue unchanged (only change narrative/prose)
- Maintain temporal relationships (before/after events)
- Handle irregular verbs correctly

Return as JSON:
{{
    "transformed_content": "string in new tense",
    "verbs_changed": number
}}""",

    TransformationType.POV_CHANGE: """Transform this text chunk by changing the point of view.

From: {from_pov}
To: {to_pov}
Focal character: {focal_character}

Instructions:
- Change pronouns and perspective accordingly
- Adjust internal thoughts/feelings to match new POV
- Handle scene information appropriately (limited POV can't know others' thoughts)
- Maintain narrative consistency

Return as JSON:
{{
    "transformed_content": "string in new POV",
    "pov_adjustments": ["list of key POV changes made"]
}}""",
}


# =============================================================================
# Document Editor Protocol
# =============================================================================


class TransformDocumentEditor(Protocol):
    """Protocol for editing document content."""

    def set_document_text(self, tab_id: str, new_text: str) -> None:
        """Set the complete text content of a document."""
        ...

    def create_document(self, title: str, content: str) -> str | None:
        """Create a new document and return its tab ID."""
        ...


# =============================================================================
# Character Rename Helper
# =============================================================================


@dataclass
class CharacterRenameSpec:
    """Specification for character renaming."""

    old_name: str
    new_name: str
    aliases: list[str] = field(default_factory=list)
    update_pronouns: bool = False
    old_pronouns: dict[str, str] = field(default_factory=dict)  # he->she, his->her
    new_pronouns: dict[str, str] = field(default_factory=dict)


def find_character_mentions(
    text: str,
    name: str,
    aliases: list[str] | None = None,
) -> list[tuple[int, int, str]]:
    """Find all mentions of a character in text.

    Args:
        text: The text to search.
        name: Primary character name.
        aliases: Alternative names/nicknames.

    Returns:
        List of (start, end, matched_text) tuples.
    """
    mentions = []
    all_names = [name] + (aliases or [])

    for search_name in all_names:
        # Match whole words, including possessive forms
        pattern = re.compile(
            rf"\b{re.escape(search_name)}(?:'s?)?\b",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            mentions.append((match.start(), match.end(), match.group()))

    # Sort by position and deduplicate overlapping
    mentions.sort(key=lambda x: x[0])
    return mentions


def apply_character_rename(
    text: str,
    spec: CharacterRenameSpec,
) -> tuple[str, int]:
    """Apply character renaming to text.

    Args:
        text: The text to transform.
        spec: Rename specification.

    Returns:
        Tuple of (transformed_text, replacement_count).
    """
    replacements = 0
    result = text

    # Build replacement patterns
    patterns = [(spec.old_name, spec.new_name)]
    for alias in spec.aliases:
        # Map alias to new name (simple approach)
        patterns.append((alias, spec.new_name))

    for old, new in patterns:
        # Handle possessive
        old_poss = f"{old}'s"
        new_poss = f"{new}'s"

        # Count replacements
        count = len(re.findall(rf"\b{re.escape(old)}\b", result, re.IGNORECASE))
        count += len(re.findall(rf"\b{re.escape(old_poss)}\b", result, re.IGNORECASE))

        # Apply replacements
        result = re.sub(
            rf"\b{re.escape(old_poss)}\b",
            new_poss,
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            result,
            flags=re.IGNORECASE,
        )

        replacements += count

    # Handle pronoun updates if requested
    if spec.update_pronouns:
        for old_pron, new_pron in spec.old_pronouns.items():
            new_val = spec.new_pronouns.get(old_pron, new_pron)
            count = len(re.findall(rf"\b{re.escape(old_pron)}\b", result, re.IGNORECASE))
            result = re.sub(
                rf"\b{re.escape(old_pron)}\b",
                new_val,
                result,
                flags=re.IGNORECASE,
            )
            replacements += count

    return result, replacements


# =============================================================================
# Transform Document Tool
# =============================================================================


@dataclass
class TransformDocumentTool(SubagentTool):
    """Tool for transforming document content using subagents.

    Performs document transformations with automatic chunking for large documents.
    Supports multiple transformation types that process chunks in parallel.

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab).
        transformation_type: Type of transformation to perform.
            - "character_rename": Rename a character throughout the document
            - "setting_change": Change the setting/location
            - "style_rewrite": Rewrite in a different style
            - "tense_change": Change narrative tense
            - "pov_change": Change point of view
            - "custom": Custom transformation (requires custom_prompt)

        For character_rename:
            - old_name: Current character name (required)
            - new_name: New character name (required)
            - aliases: List of alternative names/nicknames to also rename
            - update_pronouns: Whether to update pronouns (default False)
            - pronoun_map: Mapping of old pronouns to new pronouns

        For setting_change:
            - old_setting: Current setting name (required)
            - new_setting: New setting name (required)
            - cultural_details: Specific cultural adaptations to make

        For style_rewrite:
            - target_style: Target writing style (e.g., "formal", "casual", "noir")
            - style_guidelines: Additional style guidelines

        For tense_change:
            - from_tense: Current tense (e.g., "past", "present")
            - to_tense: Target tense

        For pov_change:
            - from_pov: Current POV (e.g., "first", "third-limited", "third-omniscient")
            - to_pov: Target POV
            - focal_character: Character for limited POV focus

        For custom:
            - custom_prompt: Custom transformation instructions (required)

        output_mode: How to output results.
            - "new_tab": Create new tab with transformed content (default)
            - "in_place": Replace content in current tab (requires version)
            - "preview": Return preview without applying

        version: Version token (required if output_mode="in_place")
        output_tab_title: Title for new output tab (optional)

    Returns:
        status: "complete" or "partial"
        chunks_processed: Number of chunks transformed
        total_replacements: Total number of replacements made
        output_tab_id: ID of output tab (if new_tab mode)
        version: New version token (if in_place mode)
        preview: Preview of changes (if preview mode)
    """

    name: ClassVar[str] = "transform_document"
    summarizable: ClassVar[bool] = False

    # Dependencies (injected)
    orchestrator: SubagentOrchestrator | None = None
    document_editor: TransformDocumentEditor | None = None
    chunk_coordinator: ChunkCoordinator | None = None

    def validate(self, params: dict[str, Any]) -> None:
        """Validate parameters."""
        super().validate(params)

        transform_type = params.get("transformation_type", "custom")
        valid_types = [t.value for t in TransformationType]

        if transform_type not in valid_types:
            raise InvalidParameterError(
                message=f"Invalid transformation_type: {transform_type}",
                parameter="transformation_type",
                value=transform_type,
                expected=f"one of {valid_types}",
            )

        # Type-specific validation
        if transform_type == "character_rename":
            if not params.get("old_name"):
                raise ContentRequiredError(
                    message="old_name is required for character_rename",
                    field_name="old_name",
                )
            if not params.get("new_name"):
                raise ContentRequiredError(
                    message="new_name is required for character_rename",
                    field_name="new_name",
                )

        elif transform_type == "setting_change":
            if not params.get("old_setting"):
                raise ContentRequiredError(
                    message="old_setting is required for setting_change",
                    field_name="old_setting",
                )
            if not params.get("new_setting"):
                raise ContentRequiredError(
                    message="new_setting is required for setting_change",
                    field_name="new_setting",
                )

        elif transform_type == "style_rewrite":
            if not params.get("target_style"):
                raise ContentRequiredError(
                    message="target_style is required for style_rewrite",
                    field_name="target_style",
                )

        elif transform_type == "tense_change":
            if not params.get("from_tense") or not params.get("to_tense"):
                raise ContentRequiredError(
                    message="from_tense and to_tense are required for tense_change",
                    field_name="from_tense/to_tense",
                )

        elif transform_type == "pov_change":
            if not params.get("from_pov") or not params.get("to_pov"):
                raise ContentRequiredError(
                    message="from_pov and to_pov are required for pov_change",
                    field_name="from_pov/to_pov",
                )

        elif transform_type == "custom":
            if not params.get("custom_prompt"):
                raise ContentRequiredError(
                    message="custom_prompt is required for custom transformation",
                    field_name="custom_prompt",
                )

        # Validate output mode
        output_mode = params.get("output_mode", "new_tab")
        valid_modes = [m.value for m in OutputMode]
        if output_mode not in valid_modes:
            raise InvalidParameterError(
                message=f"Invalid output_mode: {output_mode}",
                parameter="output_mode",
                value=output_mode,
                expected=f"one of {valid_modes}",
            )

        if output_mode == "in_place" and not params.get("version"):
            raise ContentRequiredError(
                message="version token is required for in_place output_mode",
                field_name="version",
            )

    def execute(
        self,
        context: ToolContext,
        params: dict[str, Any],
    ) -> dict[str, Any] | Coroutine[Any, Any, dict[str, Any]]:
        """Execute document transformation.

        Overrides the base SubagentTool.execute to use async orchestrator
        when available for LLM-based transformations. Character rename
        transformations are handled locally without LLM.
        """
        # Plan the work
        tasks = self.plan(context, params)
        if not tasks:
            return {"status": "no_tasks", "results": []}

        transform_type_str = params.get("transformation_type", "custom")
        transform_type = TransformationType(transform_type_str)

        # Character rename can be done locally without LLM
        if transform_type == TransformationType.CHARACTER_RENAME:
            return self._execute_sync(context, tasks)

        # For LLM-based transformations, use async execution if orchestrator available
        if self.orchestrator and self.orchestrator._executor is not None:
            return self._execute_async(context, params, tasks)

        # Fall back to sync execution (which will return errors without executor)
        return self._execute_sync(context, tasks)

    async def _execute_async(
        self,
        context: ToolContext,
        params: dict[str, Any],
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute transformation asynchronously using the orchestrator."""
        transform_type_str = params.get("transformation_type", "custom")

        # Convert task dicts to SubagentTask objects
        subagent_tasks = []
        for task in tasks:
            chunk: ChunkSpec = task["chunk"]
            subagent_task = SubagentTask(
                task_id=f"transform-{uuid.uuid4().hex[:8]}",
                subagent_type=SubagentType.TRANSFORMER,
                chunk=chunk,
                instructions=task["instructions"],
                priority=TaskPriority.NORMAL,
                metadata={
                    "transformation_type": transform_type_str,
                    "params": {
                        k: v for k, v in params.items()
                        if k not in ("tab_id", "version")  # Exclude non-serializable
                    },
                },
            )
            subagent_tasks.append(subagent_task)

        # Run all tasks through orchestrator
        results = await self.orchestrator.run_tasks(subagent_tasks, parallel=True)

        # Aggregate results
        aggregated = self._aggregate_results(results, transform_type_str)

        return aggregated

    def _execute_sync(
        self,
        context: ToolContext,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute transformation synchronously (for local transformations)."""
        results = []
        errors = []
        for task in tasks:
            try:
                result = self.execute_subagent(context, task)
                results.append(result)
            except Exception as exc:
                LOGGER.warning("Subagent task failed: %s", exc)
                errors.append({"task": task, "error": str(exc)})

        # Aggregate results
        aggregated = self.aggregate(results)

        # Include error info if any
        if errors:
            aggregated["partial_errors"] = errors
            aggregated["completed"] = len(results)
            aggregated["total"] = len(tasks)

        return aggregated

    def _aggregate_results(
        self,
        results: list[SubagentResult],
        transformation_type: str,
    ) -> dict[str, Any]:
        """Aggregate SubagentResult objects into final output."""
        aggregator = TransformAggregator()
        aggregated = aggregator.aggregate(results)

        # Add transformation-specific summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        aggregated["status"] = "complete" if not failed else "partial"
        aggregated["chunks_processed"] = len(successful)
        aggregated["chunks_failed"] = len(failed)

        # Include errors if any
        if failed:
            aggregated["errors"] = [
                {"chunk_id": r.chunk_id, "error": r.error}
                for r in failed
            ]

        return aggregated

    def plan(
        self,
        context: ToolContext,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Plan transformation tasks for document chunks."""
        # Resolve tab ID
        tab_id = context.require_tab_id(params.get("tab_id"))

        # Get document content
        content = context.document_provider.get_document_content(tab_id)
        if content is None:
            raise TabNotFoundError(
                message=f"Document not found: {tab_id}",
                tab_id=tab_id,
            )

        # Parse transformation type
        transform_type_str = params.get("transformation_type", "custom")
        transform_type = TransformationType(transform_type_str)

        # Build instructions based on type
        instructions = self._build_instructions(transform_type, params)

        # Create chunk coordinator
        coordinator = self.chunk_coordinator or ChunkCoordinator()

        # Create chunks
        if coordinator.should_chunk(content):
            chunks = coordinator.create_chunks(content, tab_id)
            LOGGER.info(
                "Document chunked into %d pieces for transformation (type=%s)",
                len(chunks),
                transform_type.value,
            )
        else:
            # Single chunk for small documents
            chunks = [ChunkSpec(
                chunk_id=f"{tab_id}-chunk-0",
                document_id=tab_id,
                content=content,
                start_char=0,
                end_char=len(content),
                token_estimate=estimate_tokens(content),
            )]

        # Convert to task specs
        tasks = []
        for chunk in chunks:
            tasks.append({
                "chunk": chunk,
                "transformation_type": transform_type,
                "instructions": instructions,
                "params": params,  # Pass original params for type-specific handling
            })

        return tasks

    def aggregate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate transformation results from all chunks."""
        # Convert dict results to SubagentResult objects
        subagent_results = []
        for r in results:
            if isinstance(r, SubagentResult):
                subagent_results.append(r)
            else:
                subagent_results.append(SubagentResult(
                    task_id=r.get("task_id", "unknown"),
                    success=r.get("success", False),
                    output=r.get("output", {}),
                    error=r.get("error"),
                    chunk_id=r.get("chunk_id", ""),
                    tokens_used=r.get("tokens_used", 0),
                ))

        # Use the transform aggregator
        aggregator = TransformAggregator()
        return aggregator.aggregate(subagent_results)

    def execute_subagent(
        self,
        context: ToolContext,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute transformation on a single chunk."""
        chunk: ChunkSpec = task["chunk"]
        transform_type: TransformationType = task["transformation_type"]
        params: dict[str, Any] = task["params"]

        # For character_rename, we can do local transformation
        if transform_type == TransformationType.CHARACTER_RENAME:
            return self._transform_character_rename(chunk, params)

        # For other types, create subagent task
        subagent_task = SubagentTask(
            task_id=f"transform-{uuid.uuid4().hex[:8]}",
            subagent_type=SubagentType.TRANSFORMER,
            chunk=chunk,
            instructions=task["instructions"],
            priority=TaskPriority.NORMAL,
        )

        # If we have an orchestrator, queue the task
        # Note: The orchestrator handles async execution; this tool returns
        # a task reference that can be checked later for results
        if self.orchestrator:
            # TODO: When async execution is implemented, this should queue the task
            # For now, we return an error indicating the limitation
            LOGGER.warning(
                "transform_document: orchestrator is configured but sync execution is not yet supported. "
                "Task %s for chunk %s with type '%s' cannot be executed.",
                subagent_task.task_id,
                chunk.chunk_id,
                transform_type.value,
            )
            return {
                "task_id": subagent_task.task_id,
                "success": False,
                "chunk_id": chunk.chunk_id,
                "error": f"Transformation type '{transform_type.value}' is queued but sync execution is not yet implemented. "
                         "Only 'character_rename' transformations are currently fully supported.",
                "output": {},
                "tokens_used": 0,
            }

        # No orchestrator - return error indicating the feature isn't ready
        return {
            "task_id": subagent_task.task_id,
            "success": False,
            "chunk_id": chunk.chunk_id,
            "error": f"Transformation type '{transform_type.value}' requires subagent execution which is not yet configured. "
                     "Only 'character_rename' transformations are currently supported.",
            "output": {},
            "tokens_used": 0,
        }

    def _transform_character_rename(
        self,
        chunk: ChunkSpec,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform character rename transformation locally."""
        spec = CharacterRenameSpec(
            old_name=params["old_name"],
            new_name=params["new_name"],
            aliases=params.get("aliases", []),
            update_pronouns=params.get("update_pronouns", False),
            old_pronouns=params.get("pronoun_map", {}).get("old", {}),
            new_pronouns=params.get("pronoun_map", {}).get("new", {}),
        )

        transformed, count = apply_character_rename(chunk.content, spec)

        return {
            "task_id": f"rename-{uuid.uuid4().hex[:8]}",
            "success": True,
            "chunk_id": chunk.chunk_id,
            "output": {
                "transformation_type": "character_rename",
                "transformed_content": transformed,
                "replacements": count,
            },
            "tokens_used": 0,  # Local transformation, no LLM tokens
        }

    def _build_instructions(
        self,
        transform_type: TransformationType,
        params: dict[str, Any],
    ) -> str:
        """Build instructions for the transformation."""
        if transform_type == TransformationType.CUSTOM:
            return params["custom_prompt"]

        template = TRANSFORM_INSTRUCTIONS.get(transform_type, "")
        if not template:
            return ""

        # Format template with params
        format_args = {}

        if transform_type == TransformationType.CHARACTER_RENAME:
            format_args = {
                "old_name": params.get("old_name", ""),
                "new_name": params.get("new_name", ""),
                "pronoun_update": "Update pronouns accordingly" if params.get("update_pronouns") else "Keep pronouns unchanged",
            }

        elif transform_type == TransformationType.SETTING_CHANGE:
            format_args = {
                "old_setting": params.get("old_setting", ""),
                "new_setting": params.get("new_setting", ""),
                "cultural_details": params.get("cultural_details", "Adapt as appropriate"),
            }

        elif transform_type == TransformationType.STYLE_REWRITE:
            format_args = {
                "target_style": params.get("target_style", ""),
                "style_guidelines": params.get("style_guidelines", ""),
            }

        elif transform_type == TransformationType.TENSE_CHANGE:
            format_args = {
                "from_tense": params.get("from_tense", ""),
                "to_tense": params.get("to_tense", ""),
            }

        elif transform_type == TransformationType.POV_CHANGE:
            format_args = {
                "from_pov": params.get("from_pov", ""),
                "to_pov": params.get("to_pov", ""),
                "focal_character": params.get("focal_character", "the protagonist"),
            }

        try:
            return template.format(**format_args)
        except KeyError:
            return template


# =============================================================================
# Consistency Checking
# =============================================================================


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue found during transformation."""

    issue_type: str  # "name_conflict", "pronoun_mismatch", "context_break"
    description: str
    location: tuple[int, int]  # (start_char, end_char)
    severity: str  # "warning", "error"
    suggestion: str = ""


def check_transformation_consistency(
    original: str,
    transformed: str,
    transform_type: TransformationType,
) -> list[ConsistencyIssue]:
    """Check for consistency issues in transformed content.

    Args:
        original: Original document content.
        transformed: Transformed content.
        transform_type: Type of transformation performed.

    Returns:
        List of consistency issues found.
    """
    issues: list[ConsistencyIssue] = []

    # Check for significant length changes
    original_len = len(original)
    transformed_len = len(transformed)
    length_ratio = transformed_len / max(1, original_len)

    if length_ratio < 0.5 or length_ratio > 2.0:
        issues.append(ConsistencyIssue(
            issue_type="length_change",
            description=f"Significant length change: {length_ratio:.0%} of original",
            location=(0, transformed_len),
            severity="warning",
            suggestion="Review transformation for unintended content loss or addition",
        ))

    # Check for orphaned quotes (basic check)
    original_quotes = original.count('"') + original.count("'")
    transformed_quotes = transformed.count('"') + transformed.count("'")

    if abs(original_quotes - transformed_quotes) > original_quotes * 0.2:
        issues.append(ConsistencyIssue(
            issue_type="quote_imbalance",
            description="Significant change in quote count may indicate dialogue issues",
            location=(0, transformed_len),
            severity="warning",
            suggestion="Review dialogue sections for completeness",
        ))

    return issues


__all__ = [
    "TransformDocumentTool",
    "TransformationType",
    "OutputMode",
    "CharacterRenameSpec",
    "ConsistencyIssue",
    "find_character_mentions",
    "apply_character_rename",
    "check_transformation_consistency",
]
