"""Tests for the error response system."""

from __future__ import annotations

import pytest

from tinkerbell.ai.tools.errors import (
    ErrorCode,
    ToolError,
    VersionMismatchToolError,
    InvalidVersionTokenError,
    InvalidTabIdError,
    TabNotFoundError,
    DocumentNotFoundError,
    InvalidLineRangeError,
    LineOutOfBoundsError,
    ContentRequiredError,
    InvalidContentError,
    UnsupportedFileTypeError,
    BinaryFileError,
    PatternInvalidError,
    NoMatchesError,
    TooManyMatchesError,
    DocumentLockedError,
    OperationCancelledError,
    TimeoutError,
    InvalidParameterError,
    MissingParameterError,
    error_from_dict,
)


class TestErrorCode:
    """Tests for ErrorCode constants."""

    def test_version_errors(self) -> None:
        """Test version-related error codes."""
        assert ErrorCode.VERSION_MISMATCH == "version_mismatch"
        assert ErrorCode.INVALID_VERSION_TOKEN == "invalid_version_token"

    def test_tab_errors(self) -> None:
        """Test tab-related error codes."""
        assert ErrorCode.INVALID_TAB_ID == "invalid_tab_id"
        assert ErrorCode.TAB_NOT_FOUND == "tab_not_found"

    def test_range_errors(self) -> None:
        """Test range-related error codes."""
        assert ErrorCode.INVALID_LINE_RANGE == "invalid_line_range"
        assert ErrorCode.LINE_OUT_OF_BOUNDS == "line_out_of_bounds"


class TestToolError:
    """Tests for base ToolError class."""

    def test_create_basic_error(self) -> None:
        """Test creating a basic tool error."""
        error = ToolError(
            error_code="test_error",
            message="Something went wrong",
        )

        assert error.error_code == "test_error"
        assert error.message == "Something went wrong"
        assert str(error) == "[test_error] Something went wrong"

    def test_error_with_details(self) -> None:
        """Test creating error with details."""
        error = ToolError(
            error_code="test_error",
            message="Error",
            details={"key": "value", "count": 42},
        )

        assert error.details["key"] == "value"
        assert error.details["count"] == 42

    def test_error_with_suggestion(self) -> None:
        """Test creating error with suggestion."""
        error = ToolError(
            error_code="test_error",
            message="Error",
            suggestion="Try doing X instead",
        )

        assert error.suggestion == "Try doing X instead"

    def test_to_dict_basic(self) -> None:
        """Test basic serialization."""
        error = ToolError(
            error_code="test_error",
            message="Error message",
        )

        result = error.to_dict()

        assert result == {
            "error": "test_error",
            "message": "Error message",
        }

    def test_to_dict_full(self) -> None:
        """Test serialization with all fields."""
        error = ToolError(
            error_code="test_error",
            message="Error message",
            details={"key": "value"},
            suggestion="Try X",
        )

        result = error.to_dict()

        assert result["error"] == "test_error"
        assert result["message"] == "Error message"
        assert result["details"] == {"key": "value"}
        assert result["suggestion"] == "Try X"

    def test_error_is_exception(self) -> None:
        """Test that ToolError is a proper exception."""
        error = ToolError(error_code="test", message="Test message")

        with pytest.raises(ToolError) as exc_info:
            raise error

        assert exc_info.value.message == "Test message"


class TestVersionMismatchToolError:
    """Tests for VersionMismatchToolError."""

    def test_default_values(self) -> None:
        """Test default error values."""
        error = VersionMismatchToolError()

        assert error.error_code == "version_mismatch"
        assert "version token" in error.message.lower()
        assert "read_document" in error.suggestion

    def test_with_version_info(self) -> None:
        """Test error with version information."""
        error = VersionMismatchToolError(
            message="Document changed",
            your_version={"tab_id": "tab-1", "version_id": 1},
            current_version={"tab_id": "tab-1", "version_id": 3},
        )

        result = error.to_dict()

        assert result["your_version"]["version_id"] == 1
        assert result["current_version"]["version_id"] == 3

    def test_from_version_error(self) -> None:
        """Test creating from core VersionMismatchError."""
        from tinkerbell.ai.tools.version import VersionMismatchError, VersionToken

        your_token = VersionToken("tab-1", "doc", 1, "old")
        current_token = VersionToken("tab-1", "doc", 3, "new")

        core_error = VersionMismatchError(
            message="Version mismatch",
            your_version=your_token,
            current_version=current_token,
            suggestion="Refresh",
        )

        tool_error = VersionMismatchToolError.from_version_error(core_error)

        assert tool_error.message == "Version mismatch"
        assert tool_error.your_version["version_id"] == 1
        assert tool_error.current_version["version_id"] == 3


class TestInvalidVersionTokenError:
    """Tests for InvalidVersionTokenError."""

    def test_basic_error(self) -> None:
        """Test basic error."""
        error = InvalidVersionTokenError(
            message="Bad token format",
            token="invalid-token",
        )

        assert error.error_code == "invalid_version_token"
        assert error.token == "invalid-token"

    def test_to_dict(self) -> None:
        """Test serialization."""
        error = InvalidVersionTokenError(
            message="Malformed",
            token="bad:token",
        )

        result = error.to_dict()

        assert result["error"] == "invalid_version_token"
        assert result["token"] == "bad:token"


class TestTabErrors:
    """Tests for tab-related errors."""

    def test_invalid_tab_id(self) -> None:
        """Test InvalidTabIdError."""
        error = InvalidTabIdError(
            message="Tab ID required",
            tab_id="bad-tab",
        )

        result = error.to_dict()

        assert result["error"] == "invalid_tab_id"
        assert result["tab_id"] == "bad-tab"
        assert "list_tabs" in error.suggestion

    def test_tab_not_found(self) -> None:
        """Test TabNotFoundError."""
        error = TabNotFoundError(
            message="Tab closed",
            tab_id="old-tab",
        )

        result = error.to_dict()

        assert result["error"] == "tab_not_found"
        assert result["tab_id"] == "old-tab"

    def test_document_not_found(self) -> None:
        """Test DocumentNotFoundError."""
        error = DocumentNotFoundError(
            message="Document missing",
            document_id="doc-123",
        )

        result = error.to_dict()

        assert result["error"] == "document_not_found"
        assert result["document_id"] == "doc-123"


class TestRangeErrors:
    """Tests for range-related errors."""

    def test_invalid_line_range(self) -> None:
        """Test InvalidLineRangeError."""
        error = InvalidLineRangeError(
            message="Invalid range",
            start_line=10,
            end_line=5,
            total_lines=100,
        )

        result = error.to_dict()

        assert result["error"] == "invalid_line_range"
        assert result["start_line"] == 10
        assert result["end_line"] == 5
        assert result["total_lines"] == 100

    def test_line_out_of_bounds(self) -> None:
        """Test LineOutOfBoundsError."""
        error = LineOutOfBoundsError(
            message="Line 150 doesn't exist",
            line=150,
            total_lines=100,
        )

        result = error.to_dict()

        assert result["error"] == "line_out_of_bounds"
        assert result["line"] == 150
        assert result["total_lines"] == 100

    def test_content_required(self) -> None:
        """Test ContentRequiredError."""
        error = ContentRequiredError(
            message="Content is required",
            field_name="content",
        )

        result = error.to_dict()

        assert result["error"] == "content_required"
        assert result["field"] == "content"


class TestFileTypeErrors:
    """Tests for file type errors."""

    def test_unsupported_file_type(self) -> None:
        """Test UnsupportedFileTypeError."""
        error = UnsupportedFileTypeError(
            message="Cannot process .exe files",
            file_type=".exe",
            file_path="/path/to/file.exe",
        )

        result = error.to_dict()

        assert result["error"] == "unsupported_file_type"
        assert result["file_type"] == ".exe"
        assert result["file_path"] == "/path/to/file.exe"

    def test_binary_file(self) -> None:
        """Test BinaryFileError."""
        error = BinaryFileError(
            message="Binary file detected",
            file_path="/path/to/binary",
        )

        result = error.to_dict()

        assert result["error"] == "binary_file"
        assert result["file_path"] == "/path/to/binary"


class TestSearchErrors:
    """Tests for search-related errors."""

    def test_pattern_invalid(self) -> None:
        """Test PatternInvalidError."""
        error = PatternInvalidError(
            message="Invalid regex",
            pattern="[invalid",
            reason="Unclosed bracket",
        )

        result = error.to_dict()

        assert result["error"] == "pattern_invalid"
        assert result["pattern"] == "[invalid"
        assert result["reason"] == "Unclosed bracket"

    def test_no_matches(self) -> None:
        """Test NoMatchesError."""
        error = NoMatchesError(
            message="No results",
            pattern="nonexistent",
        )

        result = error.to_dict()

        assert result["error"] == "no_matches"
        assert result["pattern"] == "nonexistent"
        # NoMatchesError is informational
        assert NoMatchesError.severity == "info"

    def test_too_many_matches(self) -> None:
        """Test TooManyMatchesError."""
        error = TooManyMatchesError(
            message="Too many results",
            match_count=1500,
            max_allowed=100,
        )

        result = error.to_dict()

        assert result["error"] == "too_many_matches"
        assert result["match_count"] == 1500
        assert result["max_allowed"] == 100


class TestStateErrors:
    """Tests for state/permission errors."""

    def test_document_locked(self) -> None:
        """Test DocumentLockedError."""
        error = DocumentLockedError(
            message="Document is locked",
            locked_by="another_operation",
        )

        result = error.to_dict()

        assert result["error"] == "document_locked"
        assert result["locked_by"] == "another_operation"

    def test_operation_cancelled(self) -> None:
        """Test OperationCancelledError."""
        error = OperationCancelledError(
            message="Cancelled",
            reason="User requested",
        )

        result = error.to_dict()

        assert result["error"] == "operation_cancelled"
        assert result["reason"] == "User requested"

    def test_timeout(self) -> None:
        """Test TimeoutError."""
        error = TimeoutError(
            message="Operation timed out",
            timeout_seconds=30.0,
        )

        result = error.to_dict()

        assert result["error"] == "timeout"
        assert result["timeout_seconds"] == 30.0


class TestParameterErrors:
    """Tests for parameter-related errors."""

    def test_invalid_parameter(self) -> None:
        """Test InvalidParameterError."""
        error = InvalidParameterError(
            message="Invalid value",
            parameter="count",
            value=-5,
            expected="positive integer",
        )

        result = error.to_dict()

        assert result["error"] == "invalid_parameter"
        assert result["parameter"] == "count"
        assert "-5" in result["value"]
        assert result["expected"] == "positive integer"

    def test_missing_parameter(self) -> None:
        """Test MissingParameterError."""
        error = MissingParameterError(
            message="Required param missing",
            parameter="content",
        )

        result = error.to_dict()

        assert result["error"] == "missing_parameter"
        assert result["parameter"] == "content"


class TestErrorFromDict:
    """Tests for error_from_dict utility."""

    def test_reconstruct_basic_error(self) -> None:
        """Test reconstructing a basic error."""
        data = {
            "error": "test_error",
            "message": "Test message",
        }

        error = error_from_dict(data)

        assert isinstance(error, ToolError)
        assert error.error_code == "test_error"
        assert error.message == "Test message"

    def test_reconstruct_with_details(self) -> None:
        """Test reconstructing error with details."""
        data = {
            "error": "test_error",
            "message": "Test",
            "details": {"key": "value"},
            "suggestion": "Try again",
        }

        error = error_from_dict(data)

        assert error.details == {"key": "value"}
        assert error.suggestion == "Try again"

    def test_reconstruct_missing_fields(self) -> None:
        """Test reconstruction with missing fields uses defaults."""
        data = {}

        error = error_from_dict(data)

        assert error.error_code == "internal_error"
        assert error.message == "Unknown error"


class TestErrorInheritance:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_tool_error(self) -> None:
        """Test all custom errors inherit from ToolError."""
        error_classes = [
            VersionMismatchToolError,
            InvalidVersionTokenError,
            InvalidTabIdError,
            TabNotFoundError,
            DocumentNotFoundError,
            InvalidLineRangeError,
            LineOutOfBoundsError,
            ContentRequiredError,
            InvalidContentError,
            UnsupportedFileTypeError,
            BinaryFileError,
            PatternInvalidError,
            NoMatchesError,
            TooManyMatchesError,
            DocumentLockedError,
            OperationCancelledError,
            TimeoutError,
            InvalidParameterError,
            MissingParameterError,
        ]

        for cls in error_classes:
            error = cls()
            assert isinstance(error, ToolError)
            assert isinstance(error, Exception)

    def test_all_errors_have_to_dict(self) -> None:
        """Test all errors implement to_dict."""
        error_classes = [
            VersionMismatchToolError,
            InvalidVersionTokenError,
            InvalidTabIdError,
            TabNotFoundError,
            DocumentNotFoundError,
        ]

        for cls in error_classes:
            error = cls()
            result = error.to_dict()
            assert isinstance(result, dict)
            assert "error" in result
            assert "message" in result
