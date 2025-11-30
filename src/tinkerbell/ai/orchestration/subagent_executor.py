"""Subagent Executor - LLM Integration for Document Analysis and Transformation.

WS9: Subagent Execution (LLM Integration)
WS9.1: SubagentExecutor Implementation

This module provides the SubagentExecutor class that makes actual LLM calls
to perform document analysis and transformation tasks. It integrates with
the existing AIClient infrastructure and SubagentOrchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..client import AIClient
from ..tools.subagent import (
    SubagentTask,
    SubagentResult,
    SubagentType,
    SubagentOrchestrator,
    SubagentExecutorProtocol,
    ChunkCoordinator,
    ProgressTracker,
)
from .subagent_prompts import (
    get_analysis_prompt,
    get_transform_prompt,
    ANALYSIS_SCHEMAS,
    TRANSFORM_SCHEMAS,
)

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(slots=True)
class SubagentExecutorConfig:
    """Configuration for the SubagentExecutor.
    
    Attributes:
        model: Model to use for subagent tasks (defaults to client's model)
        temperature: Temperature for generation (default 0.3 for consistency)
        max_tokens: Maximum tokens per response (default 4000)
        max_retries: Maximum retry attempts for transient failures
        retry_delay_base: Base delay for exponential backoff (seconds)
        retry_delay_max: Maximum delay between retries (seconds)
        timeout_seconds: Timeout per task (default 120s)
    """
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4000
    max_retries: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 30.0
    timeout_seconds: float = 120.0


# =============================================================================
# Response Parsing (WS9.5)
# =============================================================================


class ResponseParseError(Exception):
    """Error parsing LLM response."""
    pass


def extract_json_from_response(response: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling various formats.
    
    Handles:
    - Pure JSON responses
    - JSON wrapped in markdown code blocks
    - JSON with surrounding text
    
    Args:
        response: Raw response text from LLM
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ResponseParseError: If JSON cannot be extracted
    """
    if not response or not response.strip():
        raise ResponseParseError("Empty response")
    
    text = response.strip()
    
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    code_block_patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    
    # Try finding JSON object in the text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        potential_json = text[brace_start:brace_end + 1]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass
    
    raise ResponseParseError(f"Could not extract valid JSON from response: {text[:200]}...")


def validate_response_schema(
    data: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate response data against a schema.
    
    Performs basic schema validation without external dependencies.
    
    Args:
        data: Parsed response data
        schema: JSON schema to validate against
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []
    
    def validate_value(value: Any, type_schema: dict[str, Any], path: str) -> None:
        expected_type = type_schema.get("type")
        
        if expected_type == "object":
            if not isinstance(value, dict):
                errors.append(f"{path}: expected object, got {type(value).__name__}")
                return
            
            # Check required fields
            required = type_schema.get("required", [])
            for field in required:
                if field not in value:
                    errors.append(f"{path}: missing required field '{field}'")
            
            # Validate properties
            properties = type_schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_name in value:
                    validate_value(value[prop_name], prop_schema, f"{path}.{prop_name}")
                    
        elif expected_type == "array":
            if not isinstance(value, list):
                errors.append(f"{path}: expected array, got {type(value).__name__}")
                return
            
            items_schema = type_schema.get("items", {})
            for i, item in enumerate(value):
                validate_value(item, items_schema, f"{path}[{i}]")
                
        elif expected_type == "string":
            if not isinstance(value, str):
                errors.append(f"{path}: expected string, got {type(value).__name__}")
            # Check enum if present
            enum_values = type_schema.get("enum")
            if enum_values and value not in enum_values:
                errors.append(f"{path}: value '{value}' not in enum {enum_values}")
                
        elif expected_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{path}: expected integer, got {type(value).__name__}")
                
        elif expected_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{path}: expected number, got {type(value).__name__}")
                
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                errors.append(f"{path}: expected boolean, got {type(value).__name__}")
                
        elif isinstance(expected_type, list):
            # Union type like ["string", "null"]
            valid = False
            for t in expected_type:
                if t == "null" and value is None:
                    valid = True
                    break
                elif t == "string" and isinstance(value, str):
                    valid = True
                    break
                elif t == "integer" and isinstance(value, int) and not isinstance(value, bool):
                    valid = True
                    break
            if not valid:
                errors.append(f"{path}: value does not match any of {expected_type}")
    
    validate_value(data, schema, "root")
    return len(errors) == 0, errors


def normalize_analysis_result(
    data: dict[str, Any],
    analysis_type: str,
) -> dict[str, Any]:
    """Normalize analysis result data with defaults.
    
    Ensures all expected fields are present with sensible defaults.
    
    Args:
        data: Parsed response data
        analysis_type: Type of analysis performed
        
    Returns:
        Normalized data dictionary
    """
    result = dict(data)
    
    if analysis_type == "characters":
        characters = result.get("characters", [])
        for char in characters:
            char.setdefault("aliases", [])
            char.setdefault("role", "mentioned")
            char.setdefault("traits", [])
            char.setdefault("relationships", [])
            char.setdefault("mentions", [])
        result["characters"] = characters
        
    elif analysis_type == "plot":
        result.setdefault("plot_points", [])
        result.setdefault("tension_level", "unknown")
        result.setdefault("setting", None)
        
    elif analysis_type == "style":
        style = result.get("style", {})
        style.setdefault("voice", {"person": "unknown", "tense": "unknown"})
        style.setdefault("tone", [])
        style.setdefault("sentence_complexity", "unknown")
        style.setdefault("dialogue_frequency", "unknown")
        style.setdefault("description_density", "unknown")
        style.setdefault("pacing", "unknown")
        style.setdefault("techniques", [])
        result["style"] = style
        
    elif analysis_type == "summary":
        result.setdefault("summary", "")
        
    elif analysis_type == "themes":
        result.setdefault("themes", [])
        result.setdefault("motifs", [])
        result.setdefault("subtext", None)
    
    elif analysis_type == "custom":
        # For custom analysis, if there's no "custom" key, wrap the entire response
        if "custom" not in result:
            result = {"custom": result}
    
    return result


def validate_transformation_output(
    original: str,
    transformed: str,
) -> list[str]:
    """Validate transformation output for common issues.
    
    Args:
        original: Original text
        transformed: Transformed text
        
    Returns:
        List of warning messages (empty if no issues)
    """
    warnings: list[str] = []
    
    if not transformed or not transformed.strip():
        warnings.append("Transformation produced empty output")
        return warnings
    
    # Check length change
    orig_len = len(original)
    trans_len = len(transformed)
    
    if orig_len > 0:
        length_ratio = trans_len / orig_len
        if length_ratio < 0.5:
            warnings.append(f"Significant content reduction: {length_ratio:.0%} of original")
        elif length_ratio > 2.0:
            warnings.append(f"Significant content expansion: {length_ratio:.0%} of original")
    
    # Check quote balance
    orig_quotes = original.count('"')
    trans_quotes = transformed.count('"')
    if orig_quotes > 0:
        quote_diff = abs(orig_quotes - trans_quotes)
        if quote_diff > max(2, orig_quotes * 0.3):
            warnings.append(f"Quote count changed significantly: {orig_quotes} → {trans_quotes}")
    
    # Check for incomplete sentences (ending with incomplete punctuation)
    if transformed.rstrip()[-1:] in (",", "-", "—"):
        warnings.append("Output may be truncated (ends with incomplete punctuation)")
    
    return warnings


# =============================================================================
# SubagentExecutor (WS9.1)
# =============================================================================


class SubagentExecutor(SubagentExecutorProtocol):
    """Executor that performs subagent tasks via LLM calls.
    
    This class implements the SubagentExecutorProtocol and provides
    actual LLM integration for document analysis and transformation.
    
    Attributes:
        client: The AIClient for making LLM calls
        config: Executor configuration
    """
    
    def __init__(
        self,
        client: AIClient,
        config: SubagentExecutorConfig | None = None,
    ) -> None:
        """Initialize the executor.
        
        Args:
            client: AIClient for LLM calls
            config: Optional configuration (uses defaults if not provided)
        """
        self._client = client
        self._config = config or SubagentExecutorConfig()
    
    @property
    def client(self) -> AIClient:
        """Get the underlying AI client."""
        return self._client
    
    @property
    def config(self) -> SubagentExecutorConfig:
        """Get the executor configuration."""
        return self._config
    
    async def execute(self, task: SubagentTask) -> SubagentResult:
        """Execute a single subagent task.
        
        Routes to appropriate handler based on subagent type.
        
        Args:
            task: The subagent task to execute
            
        Returns:
            SubagentResult with success/failure and output data
        """
        start_time = time.perf_counter()
        
        try:
            if task.subagent_type == SubagentType.CHUNK_ANALYZER:
                result = await self._execute_analysis(task)
            elif task.subagent_type == SubagentType.TRANSFORMER:
                result = await self._execute_transformation(task)
            else:
                result = await self._execute_generic(task)
            
            result.latency_ms = (time.perf_counter() - start_time) * 1000.0
            return result
            
        except asyncio.CancelledError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            LOGGER.warning(
                "Subagent task %s was cancelled after %.1fms (chunk=%s, type=%s)",
                task.task_id,
                elapsed_ms,
                task.chunk.chunk_id,
                task.subagent_type.value,
            )
            return SubagentResult(
                task_id=task.task_id,
                success=False,
                error=f"Task cancelled after {elapsed_ms:.0f}ms",
                latency_ms=elapsed_ms,
                chunk_id=task.chunk.chunk_id,
            )
        except Exception as exc:
            LOGGER.exception("Subagent task %s failed", task.task_id)
            return SubagentResult(
                task_id=task.task_id,
                success=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                chunk_id=task.chunk.chunk_id,
            )
    
    async def _execute_analysis(self, task: SubagentTask) -> SubagentResult:
        """Execute an analysis task.
        
        Args:
            task: Analysis task with chunk content and instructions
            
        Returns:
            SubagentResult with analysis output
        """
        # Get analysis type from task metadata
        analysis_type = task.metadata.get("analysis_type", "summary")
        custom_prompt = task.metadata.get("custom_prompt")
        
        # Build the prompt
        system_prompt = get_analysis_prompt(analysis_type, custom_prompt)
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this text:\n\n{task.chunk.content}"},
        ]
        
        # Make LLM call with retries
        response_text = await self._call_llm_with_retries(messages, task.task_id)
        
        # Parse response
        try:
            data = extract_json_from_response(response_text)
        except ResponseParseError as exc:
            return SubagentResult(
                task_id=task.task_id,
                success=False,
                error=f"Failed to parse response: {exc}",
                chunk_id=task.chunk.chunk_id,
            )
        
        # Validate against schema (optional, just log warnings)
        schema = ANALYSIS_SCHEMAS.get(analysis_type)
        if schema:
            is_valid, errors = validate_response_schema(data, schema)
            if not is_valid:
                LOGGER.warning(
                    "Analysis response validation warnings for %s: %s",
                    task.task_id,
                    errors,
                )
        
        # Normalize the result
        normalized = normalize_analysis_result(data, analysis_type)
        
        return SubagentResult(
            task_id=task.task_id,
            success=True,
            output=normalized,
            chunk_id=task.chunk.chunk_id,
            tokens_used=self._estimate_tokens_used(messages, response_text),
        )
    
    async def _execute_transformation(self, task: SubagentTask) -> SubagentResult:
        """Execute a transformation task.
        
        Args:
            task: Transformation task with chunk content and instructions
            
        Returns:
            SubagentResult with transformed content
        """
        # Get transformation type from task metadata
        transform_type = task.metadata.get("transformation_type", "custom")
        params = task.metadata.get("params", {})
        
        # Build the prompt
        system_prompt = get_transform_prompt(transform_type, params)
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transform this text:\n\n{task.chunk.content}"},
        ]
        
        # Make LLM call with retries
        response_text = await self._call_llm_with_retries(messages, task.task_id)
        
        # Parse response
        try:
            data = extract_json_from_response(response_text)
        except ResponseParseError as exc:
            return SubagentResult(
                task_id=task.task_id,
                success=False,
                error=f"Failed to parse response: {exc}",
                chunk_id=task.chunk.chunk_id,
            )
        
        # Validate transformed content exists
        transformed_content = data.get("transformed_content")
        if not transformed_content:
            return SubagentResult(
                task_id=task.task_id,
                success=False,
                error="Transformation did not produce transformed_content",
                chunk_id=task.chunk.chunk_id,
            )
        
        # Validate transformation output
        warnings = validate_transformation_output(task.chunk.content, transformed_content)
        if warnings:
            data["_warnings"] = warnings
            LOGGER.warning(
                "Transformation warnings for %s: %s",
                task.task_id,
                warnings,
            )
        
        return SubagentResult(
            task_id=task.task_id,
            success=True,
            output=data,
            chunk_id=task.chunk.chunk_id,
            tokens_used=self._estimate_tokens_used(messages, response_text),
        )
    
    async def _execute_generic(self, task: SubagentTask) -> SubagentResult:
        """Execute a generic subagent task.
        
        Uses the task's instructions directly as the prompt.
        
        Args:
            task: Generic task with instructions
            
        Returns:
            SubagentResult
        """
        messages = [
            {"role": "system", "content": task.instructions},
            {"role": "user", "content": task.chunk.content},
        ]
        
        response_text = await self._call_llm_with_retries(messages, task.task_id)
        
        try:
            data = extract_json_from_response(response_text)
        except ResponseParseError:
            # For generic tasks, allow non-JSON responses
            data = {"response": response_text}
        
        return SubagentResult(
            task_id=task.task_id,
            success=True,
            output=data,
            chunk_id=task.chunk.chunk_id,
            tokens_used=self._estimate_tokens_used(messages, response_text),
        )
    
    async def _call_llm_with_retries(
        self,
        messages: list[dict[str, str]],
        task_id: str,
    ) -> str:
        """Call the LLM with retry logic.
        
        Implements exponential backoff for transient failures.
        
        Args:
            messages: Chat messages to send
            task_id: Task ID for logging
            
        Returns:
            Response text from LLM
            
        Raises:
            Exception: If all retries fail
        """
        last_error: Exception | None = None
        
        for attempt in range(self._config.max_retries):
            try:
                return await self._call_llm(messages)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                delay = min(
                    self._config.retry_delay_base * (2 ** attempt),
                    self._config.retry_delay_max,
                )
                LOGGER.warning(
                    "LLM call for task %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                    task_id,
                    attempt + 1,
                    self._config.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        
        raise last_error or RuntimeError("All retry attempts failed")
    
    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Make a single LLM call.
        
        Args:
            messages: Chat messages
            
        Returns:
            Response text
        """
        full_response: list[str] = []
        final_content: str | None = None
        event_types_seen: list[str] = []
        refusal_content: str | None = None
        
        async for event in self._client.stream_chat(
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        ):
            event_types_seen.append(event.type)
            if event.type == "content.delta" and event.content:
                full_response.append(event.content)
            elif event.type == "content.done" and event.content:
                # Capture final content but let the generator complete naturally
                final_content = event.content
            elif event.type in ("refusal.delta", "refusal.done") and event.content:
                refusal_content = event.content
        
        # Prefer final content if available, otherwise join deltas
        result = final_content if final_content is not None else "".join(full_response)
        
        # Log diagnostic info if response is empty
        if not result:
            unique_event_types = sorted(set(event_types_seen))
            LOGGER.warning(
                "LLM returned empty response. Events received: %d total, types: %s. "
                "Refusal content: %s. Final content was: %s. Delta count: %d.",
                len(event_types_seen),
                unique_event_types,
                refusal_content[:200] if refusal_content else None,
                repr(final_content),
                len(full_response),
            )
        
        return result
    
    def _estimate_tokens_used(
        self,
        messages: list[dict[str, str]],
        response: str,
    ) -> int:
        """Estimate tokens used for a request/response pair.
        
        Args:
            messages: Input messages
            response: Response text
            
        Returns:
            Estimated token count
        """
        total_text = "".join(m.get("content", "") for m in messages) + response
        # Use client's token counter if available
        return self._client.count_tokens(total_text, estimate_only=True)


# =============================================================================
# Orchestrator Factory (WS9.2)
# =============================================================================


def create_subagent_orchestrator(
    client: AIClient,
    config: SubagentExecutorConfig | None = None,
    *,
    max_parallel: int = 4,
    progress_tracker: ProgressTracker | None = None,
) -> SubagentOrchestrator:
    """Create a SubagentOrchestrator with LLM integration.
    
    Factory function that creates an orchestrator configured with
    a SubagentExecutor for actual LLM calls.
    
    Args:
        client: AIClient for LLM calls
        config: Optional executor configuration
        max_parallel: Maximum parallel subagents (default 4)
        progress_tracker: Optional progress tracker for UI
        
    Returns:
        Configured SubagentOrchestrator
    """
    executor = SubagentExecutor(client, config)
    orchestrator = SubagentOrchestrator(
        executor=executor,
        max_parallel=max_parallel,
        progress_tracker=progress_tracker,
    )
    return orchestrator


# =============================================================================
# Tool Integration Helpers (WS9.2.2-3)
# =============================================================================


def configure_analyze_tool_executor(
    tool: Any,
    orchestrator: SubagentOrchestrator,
) -> None:
    """Configure an AnalyzeDocumentTool with an orchestrator.
    
    Args:
        tool: AnalyzeDocumentTool instance
        orchestrator: Configured orchestrator
    """
    tool.orchestrator = orchestrator


def configure_transform_tool_executor(
    tool: Any,
    orchestrator: SubagentOrchestrator,
) -> None:
    """Configure a TransformDocumentTool with an orchestrator.
    
    Args:
        tool: TransformDocumentTool instance
        orchestrator: Configured orchestrator
    """
    tool.orchestrator = orchestrator


__all__ = [
    # Configuration
    "SubagentExecutorConfig",
    # Executor
    "SubagentExecutor",
    # Response parsing
    "ResponseParseError",
    "extract_json_from_response",
    "validate_response_schema",
    "normalize_analysis_result",
    "validate_transformation_output",
    # Factory
    "create_subagent_orchestrator",
    # Tool integration
    "configure_analyze_tool_executor",
    "configure_transform_tool_executor",
]
