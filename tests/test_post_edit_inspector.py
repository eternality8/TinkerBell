from tinkerbell.editor.post_edit_inspector import PostEditInspector


def test_post_edit_inspector_detects_duplicate_windows() -> None:
    inspector = PostEditInspector()
    before = "".join(f"line {idx}\n" for idx in range(10))
    after = before + before
    spans = [(0, len(after))]

    result = inspector.inspect(
        before_text=before,
        after_text=after,
        spans=spans,
        range_hint=None,
    )

    assert not result.ok
    assert result.reason == "duplicate_windows"
    duplicate = result.diagnostics.get("duplicate")
    assert isinstance(duplicate, dict)
    assert duplicate.get("count", 0) >= 2


def test_post_edit_inspector_detects_split_token_regex() -> None:
    inspector = PostEditInspector()
    before = "hello world\n#Heading\n"
    after = "hello world\nlo\n#Heading\n"
    spans = [(0, len(after))]

    result = inspector.inspect(
        before_text=before,
        after_text=after,
        spans=spans,
        range_hint=None,
    )

    assert not result.ok
    assert result.reason == "split_token_regex"
    pattern = result.diagnostics.get("split_pattern")
    assert isinstance(pattern, dict)
    assert pattern.get("pattern")
