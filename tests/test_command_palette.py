"""Command palette helper tests."""

from __future__ import annotations

from tinkerbell.ui.models.actions import WindowAction
from tinkerbell.ui.widgets.command_palette import PaletteCommand, build_palette_commands


def test_build_palette_commands_excludes_actions() -> None:
    actions = {
        "command_palette": WindowAction(name="command_palette", text="Command Palette", shortcut="Ctrl+Shift+P"),
        "view_document_status": WindowAction(
            name="view_document_status",
            text="Document Status...",
            shortcut="Ctrl+Shift+D",
            status_tip="Inspect document readiness",
        ),
        "file_open": WindowAction(name="file_open", text="Open...", shortcut="Ctrl+O"),
    }

    entries = build_palette_commands(actions, exclude=("command_palette",))

    ids = [entry.command_id for entry in entries]
    assert "command_palette" not in ids
    assert set(ids) == {"file_open", "view_document_status"}
    # ensure entries are sorted case-insensitively by label
    labels = [entry.label for entry in entries]
    assert labels == sorted(labels, key=str.casefold)


def test_palette_command_matches_all_tokens() -> None:
    command = PaletteCommand(
        command_id="view_document_status",
        label="Document Status...",
        detail="Chunks + outline + telemetry",
        shortcut="Ctrl+Shift+D",
        callback=lambda: None,
    )

    assert command.matches("document status") is True
    assert command.matches("chunks telemetry") is True
    assert command.matches("status ctrl") is True
    assert command.matches("unknown token") is False
