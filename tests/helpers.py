"""Shared test helpers and stub classes.

This module contains reusable test stubs that are used across multiple test files.
Import from here instead of duplicating these classes in individual test files.
"""

from __future__ import annotations


class DummyAIClient:
    """Minimal AI client stub for tests that need a client-like object.
    
    Provides just the `settings.model` attribute that many components expect.
    Use this when you need a client-like object but don't need full AI functionality.
    
    Example:
        from tests.helpers import DummyAIClient
        from typing import cast
        from tinkerbell.ai.client import AIClient
        
        client = cast(AIClient, DummyAIClient())
    """
    settings = type("S", (), {"model": "test-model"})()
