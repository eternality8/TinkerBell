"""Standardized error types for AI tools.

This module provides a hierarchy of error classes with consistent
JSON serialization for tool responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .version import VersionMismatchError as VME, VersionToken


# -----------------------------------------------------------------------------
# Error Code Constants
# -----------------------------------------------------------------------------

class ErrorCode:
    """Constants for error codes used in tool responses."""

    # Version/state errors
    VERSION_MISMATCH = "version_mismatch"
    INVALID_VERSION_TOKEN = "invalid_version_token"

    # Tab/document errors
    INVALID_TAB_ID = "invalid_tab_id"
    TAB_NOT_FOUND = "tab_not_found"
    DOCUMENT_NOT_FOUND = "document_not_found"

    # Range/content errors
    INVALID_LINE_RANGE = "invalid_line_range"
    LINE_OUT_OF_BOUNDS = "line_out_of_bounds"
    CONTENT_REQUIRED = "content_required"
    INVALID_CONTENT = "invalid_content"

    # File type errors
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    BINARY_FILE = "binary_file"

    # Search errors
    PATTERN_INVALID = "pattern_invalid"
    NO_MATCHES = "no_matches"
    TOO_MANY_MATCHES = "too_many_matches"

    # Permission/state errors
    DOCUMENT_LOCKED = "document_locked"
    OPERATION_CANCELLED = "operation_cancelled"
    TIMEOUT = "timeout"

    # General errors
    INTERNAL_ERROR = "internal_error"
    INVALID_PARAMETER = "invalid_parameter"
    MISSING_PARAMETER = "missing_parameter"


# -----------------------------------------------------------------------------
# Base Error Class
# -----------------------------------------------------------------------------

@dataclass
class ToolError(Exception):
    """Base exception class for all tool errors.

    Provides consistent JSON serialization and error categorization.

    Attributes:
        error_code: Machine-readable error identifier.
        message: Human-readable error description.
        details: Additional structured error information.
        suggestion: Actionable guidance for recovery.
    """

    error_code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""

    # Class-level default for error severity
    severity: ClassVar[str] = "error"

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON tool responses."""
        result: dict[str, Any] = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = dict(self.details)
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


# -----------------------------------------------------------------------------
# Version/State Errors
# -----------------------------------------------------------------------------

@dataclass
class VersionMismatchToolError(ToolError):
    """Error raised when a write operation references a stale document version.

    This wraps the core VersionMismatchError for tool response compatibility.
    """

    error_code: str = field(default=ErrorCode.VERSION_MISMATCH)
    message: str = field(default="Document has changed since the provided version token was issued")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Fetch a new snapshot with read_document and retry")

    your_version: dict[str, Any] | None = field(default=None)
    current_version: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.your_version:
            result["your_version"] = self.your_version
        if self.current_version:
            result["current_version"] = self.current_version
        return result

    @classmethod
    def from_version_error(cls, exc: "VME") -> "VersionMismatchToolError":
        """Create from a core VersionMismatchError."""
        return cls(
            message=exc.message,
            suggestion=exc.suggestion or "Fetch a new snapshot with read_document and retry",
            your_version=exc.your_version.to_dict() if exc.your_version else None,
            current_version=exc.current_version.to_dict() if exc.current_version else None,
        )


@dataclass
class InvalidVersionTokenError(ToolError):
    """Error raised when a version token is malformed or invalid."""

    error_code: str = field(default=ErrorCode.INVALID_VERSION_TOKEN)
    message: str = field(default="The provided version token is malformed")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Ensure the version token matches the format from read_document")

    token: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.token is not None:
            result["token"] = self.token
        return result


# -----------------------------------------------------------------------------
# Tab/Document Errors
# -----------------------------------------------------------------------------

@dataclass
class InvalidTabIdError(ToolError):
    """Error raised when a tab ID is invalid or cannot be resolved."""

    error_code: str = field(default=ErrorCode.INVALID_TAB_ID)
    message: str = field(default="Invalid or missing tab_id")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Use list_tabs to get valid tab IDs")

    tab_id: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.tab_id is not None:
            result["tab_id"] = self.tab_id
        return result


@dataclass
class TabNotFoundError(ToolError):
    """Error raised when a specified tab does not exist."""

    error_code: str = field(default=ErrorCode.TAB_NOT_FOUND)
    message: str = field(default="The specified tab was not found")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="The tab may have been closed. Use list_tabs to see available tabs")

    tab_id: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.tab_id is not None:
            result["tab_id"] = self.tab_id
        return result


@dataclass
class DocumentNotFoundError(ToolError):
    """Error raised when a document cannot be found."""

    error_code: str = field(default=ErrorCode.DOCUMENT_NOT_FOUND)
    message: str = field(default="Document not found")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Ensure the document is open in the editor")

    document_id: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.document_id is not None:
            result["document_id"] = self.document_id
        return result


# -----------------------------------------------------------------------------
# Range/Content Errors
# -----------------------------------------------------------------------------

@dataclass
class NeedsRangeError(ToolError):
    """Error raised when a large insert/edit requires explicit range bounds.

    This is typically thrown when content exceeds a threshold (e.g., 1KB)
    and no explicit target_range is provided.
    """

    error_code: str = field(default="needs_range")
    message: str = field(default="Large content requires explicit range bounds")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(
        default="Call document_snapshot to capture the intended span and retry with target_range or replace_all=true"
    )

    content_length: int | None = field(default=None)
    threshold: int | None = field(default=None)
    tab_id: str | None = field(default=None)

    code: str = field(default="needs_range")  # Alias for compatibility

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["needs_range"] = True
        if self.content_length is not None:
            result["content_length"] = self.content_length
        if self.threshold is not None:
            result["threshold"] = self.threshold
        if self.tab_id is not None:
            result["tab_id"] = self.tab_id
        return result


@dataclass
class InvalidLineRangeError(ToolError):
    """Error raised when line range parameters are invalid."""

    error_code: str = field(default=ErrorCode.INVALID_LINE_RANGE)
    message: str = field(default="Invalid line range specified")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Ensure start_line <= end_line and both are non-negative")

    start_line: int | None = field(default=None)
    end_line: int | None = field(default=None)
    total_lines: int | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.start_line is not None:
            result["start_line"] = self.start_line
        if self.end_line is not None:
            result["end_line"] = self.end_line
        if self.total_lines is not None:
            result["total_lines"] = self.total_lines
        return result


@dataclass
class LineOutOfBoundsError(ToolError):
    """Error raised when a line number is outside the document bounds."""

    error_code: str = field(default=ErrorCode.LINE_OUT_OF_BOUNDS)
    message: str = field(default="Line number is out of bounds")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Use read_document to check the total line count")

    line: int | None = field(default=None)
    total_lines: int | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.line is not None:
            result["line"] = self.line
        if self.total_lines is not None:
            result["total_lines"] = self.total_lines
        return result


@dataclass
class ContentRequiredError(ToolError):
    """Error raised when required content is missing."""

    error_code: str = field(default=ErrorCode.CONTENT_REQUIRED)
    message: str = field(default="Content is required for this operation")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Provide the required content parameter")

    field_name: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.field_name is not None:
            result["field"] = self.field_name
        return result


@dataclass
class InvalidContentError(ToolError):
    """Error raised when content is invalid or malformed."""

    error_code: str = field(default=ErrorCode.INVALID_CONTENT)
    message: str = field(default="The provided content is invalid")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Check the content format and try again")


# -----------------------------------------------------------------------------
# File Type Errors
# -----------------------------------------------------------------------------

@dataclass
class UnsupportedFileTypeError(ToolError):
    """Error raised when a file type is not supported."""

    error_code: str = field(default=ErrorCode.UNSUPPORTED_FILE_TYPE)
    message: str = field(default="This file type is not supported")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Supported types: .md, .txt, .json, .yaml, .yml")

    file_type: str | None = field(default=None)
    file_path: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.file_type is not None:
            result["file_type"] = self.file_type
        if self.file_path is not None:
            result["file_path"] = self.file_path
        return result


@dataclass
class BinaryFileError(ToolError):
    """Error raised when attempting to process a binary file."""

    error_code: str = field(default=ErrorCode.BINARY_FILE)
    message: str = field(default="Cannot process binary files")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="This tool only works with text files")

    file_path: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.file_path is not None:
            result["file_path"] = self.file_path
        return result


# -----------------------------------------------------------------------------
# Search Errors
# -----------------------------------------------------------------------------

@dataclass
class PatternInvalidError(ToolError):
    """Error raised when a search pattern is invalid."""

    error_code: str = field(default=ErrorCode.PATTERN_INVALID)
    message: str = field(default="Invalid search pattern")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Check the regex syntax and try again")

    pattern: str | None = field(default=None)
    reason: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.pattern is not None:
            result["pattern"] = self.pattern
        if self.reason is not None:
            result["reason"] = self.reason
        return result


@dataclass
class NoMatchesError(ToolError):
    """Error/info when a search returns no matches."""

    error_code: str = field(default=ErrorCode.NO_MATCHES)
    message: str = field(default="No matches found")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Try a different search term or pattern")

    severity: ClassVar[str] = "info"  # This is informational, not an error

    pattern: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.pattern is not None:
            result["pattern"] = self.pattern
        return result


@dataclass
class TooManyMatchesError(ToolError):
    """Error when a search returns too many matches to process."""

    error_code: str = field(default=ErrorCode.TOO_MANY_MATCHES)
    message: str = field(default="Too many matches found")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Narrow your search with a more specific pattern")

    match_count: int | None = field(default=None)
    max_allowed: int | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.match_count is not None:
            result["match_count"] = self.match_count
        if self.max_allowed is not None:
            result["max_allowed"] = self.max_allowed
        return result


# -----------------------------------------------------------------------------
# Permission/State Errors
# -----------------------------------------------------------------------------

@dataclass
class DocumentLockedError(ToolError):
    """Error raised when the document is locked by another operation."""

    error_code: str = field(default=ErrorCode.DOCUMENT_LOCKED)
    message: str = field(default="Document is currently locked")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Wait for the current operation to complete")

    locked_by: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.locked_by is not None:
            result["locked_by"] = self.locked_by
        return result


@dataclass
class OperationCancelledError(ToolError):
    """Error raised when an operation was cancelled by the user."""

    error_code: str = field(default=ErrorCode.OPERATION_CANCELLED)
    message: str = field(default="Operation was cancelled")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="")

    reason: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.reason is not None:
            result["reason"] = self.reason
        return result


@dataclass
class TimeoutError(ToolError):
    """Error raised when an operation times out."""

    error_code: str = field(default=ErrorCode.TIMEOUT)
    message: str = field(default="Operation timed out")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Try again or use a smaller scope")

    timeout_seconds: float | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.timeout_seconds is not None:
            result["timeout_seconds"] = self.timeout_seconds
        return result


# -----------------------------------------------------------------------------
# General Errors
# -----------------------------------------------------------------------------

@dataclass
class InvalidParameterError(ToolError):
    """Error raised when a parameter value is invalid."""

    error_code: str = field(default=ErrorCode.INVALID_PARAMETER)
    message: str = field(default="Invalid parameter value")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Check the parameter requirements")

    parameter: str | None = field(default=None)
    value: Any = field(default=None)
    expected: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.parameter is not None:
            result["parameter"] = self.parameter
        if self.value is not None:
            result["value"] = repr(self.value)
        if self.expected is not None:
            result["expected"] = self.expected
        return result


@dataclass
class MissingParameterError(ToolError):
    """Error raised when a required parameter is missing."""

    error_code: str = field(default=ErrorCode.MISSING_PARAMETER)
    message: str = field(default="Required parameter is missing")
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = field(default="Provide the required parameter")

    parameter: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.parameter is not None:
            result["parameter"] = self.parameter
        return result


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def error_from_dict(data: Mapping[str, Any]) -> ToolError:
    """Reconstruct a ToolError from its dictionary representation.

    Args:
        data: Dictionary with 'error' (code) and 'message' keys.

    Returns:
        ToolError instance (base class, not specific subclass).
    """
    return ToolError(
        error_code=data.get("error", ErrorCode.INTERNAL_ERROR),
        message=data.get("message", "Unknown error"),
        details=dict(data.get("details", {})),
        suggestion=data.get("suggestion", ""),
    )


__all__ = [
    # Error codes
    "ErrorCode",
    # Base error
    "ToolError",
    # Version errors
    "VersionMismatchToolError",
    "InvalidVersionTokenError",
    # Tab/document errors
    "InvalidTabIdError",
    "TabNotFoundError",
    "DocumentNotFoundError",
    # Range/content errors
    "NeedsRangeError",
    "InvalidLineRangeError",
    "LineOutOfBoundsError",
    "ContentRequiredError",
    "InvalidContentError",
    # File type errors
    "UnsupportedFileTypeError",
    "BinaryFileError",
    # Search errors
    "PatternInvalidError",
    "NoMatchesError",
    "TooManyMatchesError",
    # Permission/state errors
    "DocumentLockedError",
    "OperationCancelledError",
    "TimeoutError",
    # General errors
    "InvalidParameterError",
    "MissingParameterError",
    # Utilities
    "error_from_dict",
]
