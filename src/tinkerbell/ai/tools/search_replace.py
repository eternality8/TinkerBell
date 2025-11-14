"""Search/replace helper tool."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class SearchReplaceResult:
    """Outcome summary for a search/replace request."""

    replacements: int
    preview: str


def search_and_replace(text: str, pattern: str, replacement: str, *, is_regex: bool = False) -> SearchReplaceResult:
    """Perform search/replace with optional regex support."""

    if is_regex:
        compiled = re.compile(pattern)
        new_text, count = compiled.subn(replacement, text)
    else:
        new_text = text.replace(pattern, replacement)
        count = text.count(pattern)
    preview = new_text[:200]
    return SearchReplaceResult(replacements=count, preview=preview)

