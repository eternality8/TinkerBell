"""Tests for streaming diff builder helpers."""

from __future__ import annotations

import pytest

from tinkerbell.ai.tools.diff_builder import StreamedDiffBuilder, StreamedEditRequest


def _window_loader_factory(text: str):
    def _loader(start: int, end: int) -> dict[str, object]:
        return {
            "text": text[start:end],
            "window": {"start": start, "end": end},
            "chunk_manifest": {
                "chunks": [
                    {"start": 0, "end": len(text), "id": "chunk-0", "hash": "hash-0"},
                ]
            },
        }

    return _loader


def test_streamed_diff_builder_populates_match_text_and_chunk_metadata():
    text = "alpha beta gamma"
    builder = StreamedDiffBuilder()
    requests = [StreamedEditRequest(start=0, end=5, replacement="ALPHA")]

    result = builder.build(requests, window_loader=_window_loader_factory(text), manifest=None)

    assert result.ranges[0].match_text == "alpha"
    assert result.ranges[0].chunk_id == "chunk-0"
    assert result.stats.range_count == 1
    assert result.stats.replaced_chars == 5


def test_streamed_diff_builder_rejects_overlapping_ranges():
    text = "alpha beta"
    builder = StreamedDiffBuilder()
    loader = _window_loader_factory(text)
    requests = [
        StreamedEditRequest(start=0, end=5, replacement="ALPHA"),
        StreamedEditRequest(start=4, end=8, replacement="BETA"),
    ]

    with pytest.raises(ValueError):
        builder.build(requests, window_loader=loader)
