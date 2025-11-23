"""Document Status dialog surfaced via Workstream 8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..document_status import DocumentDescriptor, format_document_status_summary

try:  # pragma: no cover - optional Qt dependency
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QTabWidget,
        QMessageBox,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except Exception:  # pragma: no cover - headless fallback
    QApplication = None  # type: ignore[assignment]
    QComboBox = None  # type: ignore[assignment]
    QDialog = None  # type: ignore[assignment]
    QFileDialog = None  # type: ignore[assignment]
    QGridLayout = None  # type: ignore[assignment]
    QHBoxLayout = None  # type: ignore[assignment]
    QLabel = None  # type: ignore[assignment]
    QPlainTextEdit = None  # type: ignore[assignment]
    QPushButton = None  # type: ignore[assignment]
    QTabWidget = None  # type: ignore[assignment]
    QMessageBox = None  # type: ignore[assignment]
    QVBoxLayout = None  # type: ignore[assignment]
    QWidget = None  # type: ignore[assignment]
    Qt = None  # type: ignore[assignment]
    QCloseEvent = None  # type: ignore[assignment]
    _QT_AVAILABLE = False

_STATUS_DIALOG_STYLESHEET = """
#tb-status-dialog-header {
    font-size: 16px;
    font-weight: 600;
    padding: 4px 0px;
}

#tb-status-dialog-badge {
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    background-color: #2f323a;
    color: #f4f6fa;
}

#tb-status-dialog-badge[data-severity="info"] {
    background-color: #2a4b7c;
    color: #d8edff;
}

#tb-status-dialog-badge[data-severity="warning"] {
    background-color: #7a3b00;
    color: #ffe1b0;
}

#tb-status-dialog-badge[data-severity="danger"],
#tb-status-dialog-badge[data-severity="error"] {
    background-color: #641b1b;
    color: #ffd7d7;
}

#tb-status-dialog-badge[data-severity="success"] {
    background-color: #1f4d2c;
    color: #c5f7d2;
}

#tb-status-dialog-summary {
    font-size: 12px;
    color: #cdd2dd;
}

#tb-status-dialog-docinfo {
    color: #a9adb7;
    font-size: 11px;
}

#tb-status-dialog-metadata QLabel {
    font-size: 11px;
}

#tb-status-dialog-metadata .tb-status-meta-label {
    color: #7c8191;
}

#tb-status-dialog-metadata .tb-status-meta-value {
    color: #dfe3ec;
}

QPlainTextEdit#tb-status-tab-telemetry {
    border: 1px solid #3e424a;
    border-radius: 4px;
}

QPlainTextEdit#tb-status-tab-telemetry[data-severity="info"] {
    border-color: #337ec7;
    background-color: rgba(51, 126, 199, 0.08);
}

QPlainTextEdit#tb-status-tab-telemetry[data-severity="warning"] {
    border-color: #d9822b;
    background-color: rgba(217, 130, 43, 0.1);
}

QPlainTextEdit#tb-status-tab-telemetry[data-severity="success"] {
    border-color: #3a8f5a;
    background-color: rgba(58, 143, 90, 0.1);
}

QPlainTextEdit#tb-status-tab-telemetry[data-severity="normal"],
QPlainTextEdit#tb-status-tab-telemetry[data-severity=""] {
    border-color: #3e424a;
}
"""


class DocumentStatusWindow:
    """Qt dialog (with headless fallback) for inspecting document readiness."""

    def __init__(
        self,
        *,
        documents: Sequence[DocumentDescriptor],
        status_loader: Callable[[str | None], Mapping[str, Any]],
        parent: Any | None = None,
        enable_qt: bool | None = None,
    ) -> None:
        self._documents = list(documents)
        self._loader = status_loader
        self._parent = parent
        self._last_payload: Mapping[str, Any] | None = None
        self._current_document_id: str | None = None
        if enable_qt is None:
            self._qt_enabled = bool(_QT_AVAILABLE)
        else:
            self._qt_enabled = bool(enable_qt and _QT_AVAILABLE)

        # Qt widgets (optional)
        self._dialog: Any | None = None
        self._header_label: Any | None = None
        self._document_combo: Any | None = None
        self._badge_label: Any | None = None
        self._summary_label: Any | None = None
        self._document_info_label: Any | None = None
        self._metadata_widget: Any | None = None
        self._metadata_labels: dict[str, Any] = {}
        self._tab_widget: Any | None = None
        self._chunks_view: Any | None = None
        self._outline_view: Any | None = None
        self._plot_view: Any | None = None
        self._telemetry_view: Any | None = None
        self._save_button: Any | None = None
        self._last_export_path: Path | None = None

        if self._qt_enabled:
            self._build_dialog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def show(self, *, document_id: str | None = None) -> Mapping[str, Any] | None:
        """Display the dialog (or simply return the payload when headless)."""

        if document_id is not None:
            document_id = self._select_document(document_id) or document_id
        payload = self._load_document(document_id)
        if self._dialog is not None:
            try:
                self._dialog.show()
                self._dialog.raise_()  # type: ignore[attr-defined]
                self._dialog.activateWindow()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive Qt guard
                pass
        return payload

    def refresh(self, document_id: str | None = None) -> Mapping[str, Any] | None:
        """Refresh the current payload without forcing the dialog visible."""

        if document_id is not None:
            document_id = self._select_document(document_id) or document_id
        return self._load_document(document_id)

    def update_documents(self, documents: Sequence[DocumentDescriptor]) -> None:
        self._documents = list(documents)
        if self._document_combo is not None:
            try:
                self._document_combo.blockSignals(True)
                self._document_combo.clear()
                for descriptor in self._documents:
                    self._document_combo.addItem(descriptor.label, descriptor.document_id)
            finally:
                self._document_combo.blockSignals(False)
        if not self._documents:
            self._current_document_id = None
            if self._document_combo is not None:
                self._document_combo.setCurrentIndex(-1)
            return
        ids = {descriptor.document_id for descriptor in self._documents}
        if self._current_document_id not in ids:
            self._current_document_id = self._documents[0].document_id if self._documents else None
        selected = self._select_document(self._current_document_id)
        if selected is not None:
            self._current_document_id = selected

    def export_payload(self, path: str | Path | None = None) -> str:
        """Return (and optionally persist) the last payload as JSON."""

        if self._last_payload is None:
            raise RuntimeError("Document status payload unavailable")
        body = json.dumps(self._last_payload, indent=2, ensure_ascii=False)
        if path is not None:
            target = Path(path)
            target.write_text(body, encoding="utf-8")
        return body

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_dialog(self) -> None:
        if not self._qt_enabled or not _QT_AVAILABLE or QDialog is None:
            return
        assert QVBoxLayout is not None
        assert QLabel is not None
        assert QHBoxLayout is not None
        assert QComboBox is not None
        assert QPushButton is not None
        assert QTabWidget is not None
        dialog = QDialog(self._parent)
        dialog.setObjectName("tb-status-dialog")
        dialog.setWindowTitle("Document Status")
        if _STATUS_DIALOG_STYLESHEET:
            try:
                dialog.setStyleSheet(_STATUS_DIALOG_STYLESHEET)
            except Exception:  # pragma: no cover - defensive Qt guard
                pass
        layout = QVBoxLayout(dialog)

        self._header_label = QLabel("Document status not loaded")
        self._header_label.setObjectName("tb-status-dialog-header")
        layout.addWidget(self._header_label)

        badge_row = QHBoxLayout()
        self._badge_label = QLabel("Badge unavailable")
        self._badge_label.setObjectName("tb-status-dialog-badge")
        badge_row.addWidget(self._badge_label)
        self._summary_label = QLabel("Summary not loaded yet")
        self._summary_label.setObjectName("tb-status-dialog-summary")
        self._summary_label.setWordWrap(True)
        badge_row.addWidget(self._summary_label, 1)
        layout.addLayout(badge_row)

        self._document_info_label = QLabel("Document metadata unavailable")
        self._document_info_label.setObjectName("tb-status-dialog-docinfo")
        self._document_info_label.setWordWrap(True)
        layout.addWidget(self._document_info_label)
        self._build_metadata_panel(layout)

        controls = QHBoxLayout()
        self._document_combo = QComboBox()
        self._document_combo.setObjectName("tb-status-dialog-combo")
        for descriptor in self._documents:
            self._document_combo.addItem(descriptor.label, descriptor.document_id)
        self._document_combo.currentIndexChanged.connect(self._handle_doc_changed)  # type: ignore[attr-defined]
        controls.addWidget(self._document_combo, 1)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._handle_refresh_clicked)  # type: ignore[attr-defined]
        controls.addWidget(refresh_button)

        copy_button = QPushButton("Copy JSON")
        copy_button.clicked.connect(self._handle_copy_clicked)  # type: ignore[attr-defined]
        controls.addWidget(copy_button)

        save_button = QPushButton("Save JSON…")
        save_button.clicked.connect(self._handle_save_clicked)  # type: ignore[attr-defined]
        controls.addWidget(save_button)
        self._save_button = save_button

        layout.addLayout(controls)

        self._tab_widget = QTabWidget()
        self._chunks_view = self._add_tab("Chunks")
        self._outline_view = self._add_tab("Outline")
        self._plot_view = self._add_tab("Plot & Concordance")
        self._telemetry_view = self._add_tab("Telemetry")
        layout.addWidget(self._tab_widget, 1)

        self._dialog = dialog

    def _add_tab(self, title: str) -> Any | None:
        if self._tab_widget is None or QPlainTextEdit is None:
            return None
        editor = QPlainTextEdit()
        editor.setObjectName(f"tb-status-tab-{title.lower().replace(' ', '-')}")
        editor.setReadOnly(True)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        editor.setPlaceholderText("Status data not loaded yet")
        self._tab_widget.addTab(editor, title)
        return editor

    def _handle_doc_changed(self, index: int) -> None:
        if self._document_combo is None:
            return
        document_id = self._document_combo.itemData(index)
        if isinstance(document_id, str):
            self._load_document(document_id)

    def _handle_refresh_clicked(self) -> None:
        self._load_document(self._current_document_id)

    def _handle_copy_clicked(self) -> None:
        if self._last_payload is None:
            return
        body = self.export_payload()
        if not self._qt_enabled or QApplication is None:
            return
        try:
            clipboard = QApplication.clipboard()
            clipboard.setText(body)
        except Exception:  # pragma: no cover - clipboard may be unavailable
            pass

    def _handle_save_clicked(self) -> None:
        if self._last_payload is None or not self._qt_enabled or not _QT_AVAILABLE or QFileDialog is None:
            return
        directory_hint = self._last_export_path.parent if self._last_export_path else Path.cwd()
        filename_hint = self._last_export_path.name if self._last_export_path else "document_status.json"
        initial_path = str((directory_hint / filename_hint).resolve())
        try:
            path_str, _ = QFileDialog.getSaveFileName(
                self._dialog,
                "Export Document Status",
                initial_path,
                "JSON Files (*.json);;All Files (*)",
            )
        except Exception:  # pragma: no cover - file dialog unavailable
            return
        if not path_str:
            return
        target = Path(path_str)
        try:
            self.export_payload(target)
        except Exception as exc:  # pragma: no cover - show message for failures
            self._show_message_box("Unable to export status", str(exc))
            return
        self._last_export_path = target
        self._show_message_box("Document Status", f"Saved to {target}")

    def _load_document(self, document_id: str | None) -> Mapping[str, Any] | None:
        target_id = document_id or self._current_document_id
        if target_id is None and self._documents:
            target_id = self._documents[0].document_id
        if target_id is not None:
            self._current_document_id = target_id
        try:
            payload = self._loader(target_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._last_payload = None
            if self._header_label is not None:
                self._header_label.setText(f"Unable to load status: {exc}")
            return None
        self._last_payload = payload
        self._current_document_id = target_id
        self._update_views()
        return payload

    def _update_views(self) -> None:
        payload = self._last_payload or {}
        if self._header_label is not None:
            summary = format_document_status_summary(payload)
            self._header_label.setText(summary)
        self._update_badge_summary(payload)
        self._update_document_info(payload)
        self._render_chunks(payload.get("chunks"))
        document_payload = payload.get("document")
        self._render_outline(payload.get("outline"), document_payload)
        self._render_plot_sections(payload.get("plot"), payload.get("concordance"), payload.get("planner"))
        self._render_telemetry(payload.get("telemetry"))

    def _render_json(self, widget: Any | None, data: Any) -> None:
        if widget is None:
            return
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except TypeError:
            text = "{}" if data is None else str(data)
        try:
            widget.setPlainText(text)
        except Exception:  # pragma: no cover - make unit tests robust
            pass

    def _render_chunks(self, chunk_payload: Any) -> None:
        if self._chunks_view is None:
            return
        lines = self._format_chunk_lines(chunk_payload)
        if lines:
            text = "\n".join(lines)
            try:
                self._chunks_view.setPlainText(text)
            except Exception:  # pragma: no cover - Qt safeguards
                pass
            return
        self._render_json(self._chunks_view, chunk_payload)

    def _render_outline(self, outline_payload: Any, document_payload: Any) -> None:
        if self._outline_view is None:
            return
        lines = self._format_outline_lines(outline_payload, document_payload)
        if lines:
            text = "\n".join(lines)
            try:
                self._outline_view.setPlainText(text)
            except Exception:  # pragma: no cover - Qt safeguards
                pass
            return
        fallback = {"outline": outline_payload, "document": document_payload}
        self._render_json(self._outline_view, fallback)

    def _render_plot_sections(
        self,
        plot_payload: Any,
        concordance_payload: Any,
        planner_payload: Any,
    ) -> None:
        if self._plot_view is None:
            return
        lines: list[str] = []
        for block in (
            self._format_plot_lines(plot_payload),
            self._format_concordance_lines(concordance_payload),
            self._format_planner_lines(planner_payload),
        ):
            if block:
                if lines:
                    lines.append("")
                lines.extend(block)
        if lines:
            text = "\n".join(lines)
            try:
                self._plot_view.setPlainText(text)
            except Exception:  # pragma: no cover - Qt safeguards
                pass
            return
        fallback = {
            "plot": plot_payload,
            "concordance": concordance_payload,
            "planner": planner_payload,
        }
        self._render_json(self._plot_view, fallback)

    def _render_telemetry(self, telemetry_payload: Any) -> None:
        if self._telemetry_view is None:
            return
        lines = self._format_telemetry_lines(telemetry_payload)
        if lines:
            text = "\n".join(lines)
            try:
                self._telemetry_view.setPlainText(text)
            except Exception:  # pragma: no cover - Qt safeguards
                pass
            severity = self._determine_telemetry_severity(telemetry_payload)
            self._apply_tab_severity(self._telemetry_view, severity)
            return
        self._render_json(self._telemetry_view, telemetry_payload)
        severity = self._determine_telemetry_severity(telemetry_payload)
        self._apply_tab_severity(self._telemetry_view, severity)

    def _update_badge_summary(self, payload: Mapping[str, Any]) -> None:
        if not self._qt_enabled:
            return
        badge_data = payload.get("badge") if isinstance(payload, Mapping) else None
        summary_text = str(payload.get("summary") or "").strip()
        if self._summary_label is not None:
            self._summary_label.setText(summary_text or "Summary unavailable")
        if self._badge_label is None:
            return
        if not isinstance(badge_data, Mapping):
            self._badge_label.setText("Doc Status")
            self._badge_label.setToolTip("")
            self._badge_label.setProperty("data-severity", "")
            self._refresh_widget_style(self._badge_label)
            return
        status = str(badge_data.get("status") or "Doc Status").strip()
        detail = str(badge_data.get("detail") or "").strip()
        severity = str(badge_data.get("severity") or "").strip()
        self._badge_label.setText(status)
        self._badge_label.setToolTip(detail)
        self._badge_label.setProperty("data-severity", severity)
        self._refresh_widget_style(self._badge_label)

    def _update_document_info(self, payload: Mapping[str, Any]) -> None:
        if not self._qt_enabled or self._document_info_label is None:
            return
        document_payload = payload.get("document") if isinstance(payload, Mapping) else None
        if not isinstance(document_payload, Mapping):
            self._document_info_label.setText("Document metadata unavailable")
            for key in self._metadata_labels:
                self._set_metadata_field(key, "—")
            return
        info_lines = self._format_document_info(document_payload)
        self._document_info_label.setText("\n".join(info_lines))
        metadata = self._build_document_meta_fields(document_payload)
        for key, value in metadata.items():
            self._set_metadata_field(key, value)

    @staticmethod
    def _format_document_info(document_payload: Mapping[str, Any]) -> list[str]:
        label = str(document_payload.get("label") or document_payload.get("document_id") or "document").strip()
        language = str(document_payload.get("language") or "").strip()
        dirty = " (unsaved changes)" if document_payload.get("dirty") else ""
        header = f"Document: {label}{dirty}"
        if language:
            header = f"{header} — {language}"
        details = [header]
        path = document_payload.get("path")
        if path:
            details.append(f"Path: {path}")
        length = document_payload.get("length")
        if isinstance(length, int):
            details.append(f"Length: {length:,} chars")
        version_id = document_payload.get("version_id")
        content_hash = document_payload.get("content_hash")
        version_bits = []
        if version_id:
            version_bits.append(f"version={version_id}")
        if content_hash:
            version_bits.append(f"hash={content_hash}")
        if version_bits:
            details.append("Version: " + ", ".join(version_bits))
        return details

    @staticmethod
    def _build_document_meta_fields(document_payload: Mapping[str, Any]) -> dict[str, str]:
        length = document_payload.get("length")
        length_text = f"{length:,} chars" if isinstance(length, int) else "—"
        version_id = document_payload.get("version_id")
        content_hash = document_payload.get("content_hash")
        version_bits: list[str] = []
        if version_id:
            version_bits.append(str(version_id))
        if content_hash:
            version_bits.append(str(content_hash))
        version_text = " / ".join(version_bits) if version_bits else "—"
        path = document_payload.get("path")
        label = document_payload.get("label") or document_payload.get("document_id") or "document"
        language_value = document_payload.get("language")
        if isinstance(language_value, str) and language_value.strip():
            language_text = language_value.strip()
        else:
            language_text = "—"
        return {
            "document": str(label),
            "path": str(path) if path else "—",
            "length": length_text,
            "version": version_text,
            "language": language_text,
        }

    @staticmethod
    def _refresh_widget_style(widget: Any) -> None:
        style = getattr(widget, "style", None)
        if style is None:
            return
        unpolish = getattr(style, "unpolish", None)
        polish = getattr(style, "polish", None)
        if callable(unpolish) and callable(polish):
            try:
                unpolish(widget)
                polish(widget)
            except Exception:  # pragma: no cover - best effort only
                pass

    @staticmethod
    def _format_chunk_lines(chunk_payload: Any) -> list[str]:
        if not isinstance(chunk_payload, Mapping):
            return []
        chunk_profile = str(chunk_payload.get("chunk_profile") or "auto").strip()
        chunk_count = DocumentStatusWindow._coerce_chunk_count(chunk_payload)
        window_line = DocumentStatusWindow._format_window_line(chunk_payload)
        generated_line = DocumentStatusWindow._format_generated_line(chunk_payload)
        version_line = DocumentStatusWindow._format_version_line(chunk_payload.get("document_version"))
        has_summary = any(
            [
                chunk_count is not None,
                bool(window_line),
                bool(generated_line),
                bool(version_line),
            ]
        )
        if not has_summary:
            return []

        if chunk_count is not None:
            header = f"Chunks: {chunk_count} (profile {chunk_profile})"
        else:
            header = f"Chunks: profile {chunk_profile}"

        lines = [header]
        if window_line:
            lines.append(window_line)
        if generated_line:
            lines.append(generated_line)
        if version_line:
            lines.append(version_line)

        return lines

    @staticmethod
    def _coerce_chunk_count(chunk_payload: Mapping[str, Any]) -> int | None:
        manifest = chunk_payload.get("chunk_manifest")
        if isinstance(manifest, Mapping):
            chunks = manifest.get("chunks")
            if isinstance(chunks, Sequence):
                return len(chunks)
        stats = chunk_payload.get("stats")
        if isinstance(stats, Mapping):
            count = stats.get("chunk_count")
            if isinstance(count, int):
                return count
        return None

    @staticmethod
    def _format_window_line(chunk_payload: Mapping[str, Any]) -> str:
        window = chunk_payload.get("window")
        if not isinstance(window, Mapping):
            return ""
        start = window.get("start")
        end = window.get("end")
        if isinstance(start, int) and isinstance(end, int):
            return f"Window: {start}–{end}"
        return ""

    @staticmethod
    def _format_generated_line(chunk_payload: Mapping[str, Any]) -> str:
        manifest = chunk_payload.get("chunk_manifest")
        generated_at = None
        if isinstance(manifest, Mapping):
            generated_at = manifest.get("generated_at")
        stats = chunk_payload.get("stats")
        if generated_at is None and isinstance(stats, Mapping):
            generated_at = stats.get("generated_at")
        if generated_at:
            return f"Manifest generated at {generated_at}"
        return ""

    @staticmethod
    def _format_version_line(version_payload: Any) -> str:
        if not isinstance(version_payload, Mapping):
            return ""
        version_id = version_payload.get("version") or version_payload.get("version_id")
        content_hash = version_payload.get("content_hash")
        bits: list[str] = []
        if version_id:
            bits.append(f"version={version_id}")
        if content_hash:
            bits.append(f"hash={content_hash}")
        return f"Document version: {', '.join(bits)}" if bits else ""

    @staticmethod
    def _format_outline_lines(outline_payload: Any, document_payload: Any) -> list[str]:
        if not isinstance(outline_payload, Mapping):
            return []
        lines: list[str] = []
        status = str(outline_payload.get("status") or "").strip()
        node_count = outline_payload.get("node_count")
        header_bits: list[str] = []
        if status:
            header_bits.append(f"Outline status: {status}")
        if isinstance(node_count, int):
            header_bits.append(f"Nodes: {node_count}")
        if header_bits:
            lines.append(" · ".join(header_bits))

        updated_at = outline_payload.get("updated_at")
        if updated_at:
            lines.append(f"Last updated {updated_at}")

        doc_version = document_payload.get("version_id") if isinstance(document_payload, Mapping) else None
        outline_version = outline_payload.get("version_id")
        if outline_version:
            if doc_version and doc_version != outline_version:
                lines.append(f"Version mismatch: outline={outline_version} document={doc_version}")
            else:
                lines.append(f"Outline version: {outline_version}")
        elif doc_version:
            lines.append(f"Document version: {doc_version}")

        outline_hash = outline_payload.get("outline_hash")
        if outline_hash:
            lines.append(f"Outline hash: {outline_hash}")

        summary = str(outline_payload.get("summary") or "").strip()
        if summary:
            if lines:
                lines.append("")
            lines.append("Summary:")
            lines.append(summary)

        highlights = outline_payload.get("highlights")
        highlight_lines: list[str] = []
        if isinstance(highlights, Sequence):
            for value in highlights:
                if isinstance(value, Mapping):
                    text = str(value.get("text") or value.get("label") or "").strip()
                else:
                    text = str(value).strip()
                if text:
                    highlight_lines.append(text)
        if highlight_lines:
            if lines:
                lines.append("")
            lines.append("Highlights:")
            max_highlights = 5
            for entry in highlight_lines[:max_highlights]:
                lines.append(f"- {entry}")
            remaining = len(highlight_lines) - max_highlights
            if remaining > 0:
                lines.append(f"- … {remaining} more highlight(s)")

        return lines

    @staticmethod
    def _format_plot_lines(plot_payload: Any) -> list[str]:
        if not isinstance(plot_payload, Mapping):
            return []
        entity_count = plot_payload.get("entity_count")
        arc_count = plot_payload.get("arc_count")
        version_id = plot_payload.get("version_id")
        generated_at = plot_payload.get("generated_at")
        total_beats = DocumentStatusWindow._count_plot_beats(plot_payload)
        header_bits: list[str] = []
        if isinstance(entity_count, int):
            header_bits.append(f"{entity_count} entities")
        if isinstance(arc_count, int):
            header_bits.append(f"{arc_count} arcs")
        if total_beats:
            header_bits.append(f"{total_beats} beats")
        header = "Plot state"
        if header_bits:
            header = f"{header}: {', '.join(header_bits)}"
        lines = [header]
        if version_id:
            lines.append(f"Version: {version_id}")
        if generated_at:
            lines.append(f"Updated at {generated_at}")

        arcs = plot_payload.get("arcs")
        arc_lines: list[str] = []
        if isinstance(arcs, Sequence) and not isinstance(arcs, (str, bytes)):
            for arc in arcs[:3]:
                if not isinstance(arc, Mapping):
                    continue
                name = str(arc.get("name") or arc.get("arc_id") or "Arc").strip()
                beats = arc.get("beats")
                beat_count = len(beats) if isinstance(beats, Sequence) and not isinstance(beats, (str, bytes)) else 0
                summary = str(arc.get("summary") or "").strip()
                descriptor = f"{name} ({beat_count} beats)"
                if summary:
                    arc_lines.append(f"- {descriptor}: {summary}")
                else:
                    arc_lines.append(f"- {descriptor}")
        if arc_lines:
            lines.append("")
            lines.append("Arcs:")
            lines.extend(arc_lines)

        entities = plot_payload.get("entities")
        entity_lines: list[str] = []
        if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
            for entity in entities[:5]:
                if not isinstance(entity, Mapping):
                    continue
                name = str(entity.get("name") or entity.get("entity_id") or "Entity").strip()
                if not name:
                    continue
                summary = str(entity.get("summary") or "").strip()
                salience = entity.get("salience")
                descriptor = name
                if isinstance(salience, (int, float)) and salience > 0:
                    descriptor = f"{descriptor} (salience {salience:.2f})"
                if summary:
                    entity_lines.append(f"- {descriptor}: {summary}")
                else:
                    entity_lines.append(f"- {descriptor}")
        if entity_lines:
            lines.append("")
            lines.append("Entities:")
            lines.extend(entity_lines)

        metadata = plot_payload.get("metadata")
        if isinstance(metadata, Mapping):
            stats = metadata.get("stats")
            if isinstance(stats, Mapping):
                ingested = stats.get("ingested_chunks")
                if isinstance(ingested, int):
                    lines.append("")
                    lines.append(f"Chunks ingested: {ingested}")

        overrides = plot_payload.get("overrides")
        override_lines = DocumentStatusWindow._format_override_lines(overrides)
        if override_lines:
            lines.append("")
            lines.append("Overrides:")
            lines.extend(override_lines)

        return lines

    @staticmethod
    def _format_concordance_lines(concordance_payload: Any) -> list[str]:
        if not isinstance(concordance_payload, Mapping):
            return []
        entity_count = concordance_payload.get("entity_count")
        generated_at = concordance_payload.get("generated_at")
        version_id = concordance_payload.get("version_id")
        stats = concordance_payload.get("stats")
        lines: list[str] = []
        header_bits: list[str] = []
        if isinstance(entity_count, int):
            header_bits.append(f"{entity_count} entities")
        if isinstance(stats, Mapping):
            ingested = stats.get("ingested_chunks")
            if isinstance(ingested, int):
                header_bits.append(f"{ingested} chunk ingests")
        header = "Concordance"
        if header_bits:
            header = f"{header}: {', '.join(header_bits)}"
        lines.append(header)
        if version_id:
            lines.append(f"Version: {version_id}")
        if generated_at:
            lines.append(f"Updated at {generated_at}")

        entities = concordance_payload.get("entities")
        entity_lines: list[str] = []
        if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
            for entry in entities[:5]:
                if not isinstance(entry, Mapping):
                    continue
                name = str(entry.get("name") or entry.get("entity_id") or "Entity").strip()
                if not name:
                    continue
                pronouns = entry.get("pronouns")
                pronoun_text = ""
                if isinstance(pronouns, Sequence) and not isinstance(pronouns, (str, bytes)):
                    pronoun_list = [str(item).strip() for item in pronouns if str(item).strip()]
                    if pronoun_list:
                        pronoun_text = f" ({', '.join(pronoun_list)})"
                mention_count = entry.get("mention_count")
                mention_text = f" — {mention_count} mentions" if isinstance(mention_count, int) else ""
                entity_lines.append(f"- {name}{pronoun_text}{mention_text}")
        if entity_lines:
            lines.append("")
            lines.append("Entities:")
            lines.extend(entity_lines)

        return lines

    @staticmethod
    def _format_planner_lines(planner_payload: Any) -> list[str]:
        if not isinstance(planner_payload, Mapping):
            return []
        pending = planner_payload.get("pending")
        completed = planner_payload.get("completed")
        header_bits: list[str] = []
        if isinstance(completed, int):
            header_bits.append(f"{completed} completed")
        if isinstance(pending, int):
            header_bits.append(f"{pending} pending")
        if not header_bits:
            return []
        lines = ["Planner: " + " / ".join(header_bits)]

        tasks = planner_payload.get("tasks")
        task_lines: list[str] = []
        if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes)):
            for task in tasks[:5]:
                if not isinstance(task, Mapping):
                    continue
                task_id = str(task.get("task_id") or "task").strip()
                status = str(task.get("status") or "pending").strip() or "pending"
                note = str(task.get("note") or "").strip()
                descriptor = f"- {task_id}: {status}"
                if note:
                    descriptor = f"{descriptor} — {note}"
                task_lines.append(descriptor)
        if task_lines:
            lines.append("")
            lines.append("Tasks:")
            lines.extend(task_lines)

        return lines

    @staticmethod
    def _format_telemetry_lines(telemetry_payload: Any) -> list[str]:
        if not isinstance(telemetry_payload, Mapping):
            return []
        lines: list[str] = []
        chunk_flow = telemetry_payload.get("chunk_flow")
        if isinstance(chunk_flow, Mapping):
            status = str(chunk_flow.get("status") or "Chunk flow").strip()
            detail = str(chunk_flow.get("detail") or "").strip()
            lines.append(f"Chunk flow: {status}")
            if detail:
                lines.append(f"  {detail}")

        analysis = telemetry_payload.get("analysis")
        if isinstance(analysis, Mapping):
            badge = str(analysis.get("badge") or analysis.get("status") or "Analysis").strip()
            detail = str(analysis.get("detail") or "").strip()
            if lines:
                lines.append("")
            lines.append(badge)
            if detail:
                lines.append(detail)

        return lines

    @staticmethod
    def _format_override_lines(overrides: Any) -> list[str]:
        if not isinstance(overrides, Sequence) or isinstance(overrides, (str, bytes)):
            return []
        lines: list[str] = []
        max_items = 5
        for entry in overrides[:max_items]:
            if not isinstance(entry, Mapping):
                continue
            override_id = str(entry.get("override_id") or "override").strip()
            summary = str(entry.get("summary") or "").strip()
            author = str(entry.get("author") or "operator").strip()
            descriptor = f"- {override_id}"
            if author:
                descriptor = f"{descriptor} ({author})"
            if summary:
                descriptor = f"{descriptor}: {summary}"
            lines.append(descriptor)
        remaining = len(overrides) - min(len(overrides), max_items)
        if remaining > 0:
            lines.append(f"- … {remaining} more override(s)")
        return lines

    @staticmethod
    def _count_plot_beats(plot_payload: Mapping[str, Any]) -> int:
        arcs = plot_payload.get("arcs")
        if not isinstance(arcs, Sequence) or isinstance(arcs, (str, bytes)):
            return 0
        total = 0
        for arc in arcs:
            beats = arc.get("beats") if isinstance(arc, Mapping) else None
            if isinstance(beats, Sequence) and not isinstance(beats, (str, bytes)):
                total += len(beats)
        return total

    def _apply_tab_severity(self, widget: Any | None, severity: str) -> None:
        if widget is None or not self._qt_enabled:
            return
        widget.setProperty("data-severity", severity or "")
        self._refresh_widget_style(widget)

    @staticmethod
    def _determine_telemetry_severity(payload: Any) -> str:
        if not isinstance(payload, Mapping):
            return ""
        chunk_flow = payload.get("chunk_flow")
        if isinstance(chunk_flow, Mapping):
            status_text = str(chunk_flow.get("status") or "").lower()
            if "warn" in status_text:
                return "warning"
            if "recover" in status_text:
                return "success"
            if status_text:
                return "info"
        analysis = payload.get("analysis")
        if isinstance(analysis, Mapping):
            badge = str(analysis.get("badge") or "").lower()
            status = str(analysis.get("status") or "").lower()
            combined = f"{badge} {status}".strip()
            if "refresh" in combined or "warning" in combined:
                return "info"
            if combined:
                return "normal"
        return ""

    def _show_message_box(self, title: str, text: str) -> None:
        if not self._qt_enabled or not _QT_AVAILABLE or QMessageBox is None:
            return
        parent = self._dialog
        try:
            QMessageBox.information(parent, title, text)
        except Exception:  # pragma: no cover - guard for headless tests
            pass

    def _build_metadata_panel(self, layout: Any) -> None:
        if not self._qt_enabled or QWidget is None or QGridLayout is None or QLabel is None:
            return
        container = QWidget()
        container.setObjectName("tb-status-dialog-metadata")
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 4, 0, 6)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)
        fields = (
            ("Document", "document"),
            ("Path", "path"),
            ("Length", "length"),
            ("Version", "version"),
            ("Language", "language"),
        )
        for row, (label_text, key) in enumerate(fields):
            label = QLabel(f"{label_text}:")
            label.setProperty("class", "tb-status-meta-label")
            value = QLabel("—")
            value.setProperty("class", "tb-status-meta-value")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1)
            self._metadata_labels[key] = value
        layout.addWidget(container)
        self._metadata_widget = container

    def _set_metadata_field(self, key: str, value: str) -> None:
        widget = self._metadata_labels.get(key)
        if widget is None:
            return
        widget.setText(value or "—")

    def _select_document(self, document_id: str | None) -> str | None:
        combo = self._document_combo
        if combo is None:
            return None
        total = combo.count()
        if total <= 0:
            return None
        match_index = -1
        selected_id: str | None = None
        for index in range(total):
            data = combo.itemData(index)
            if not isinstance(data, str):
                continue
            if document_id is None:
                match_index = index
                selected_id = data
                break
            if data == document_id:
                match_index = index
                selected_id = data
                break
        if match_index == -1:
            data = combo.itemData(0)
            if isinstance(data, str):
                match_index = 0
                selected_id = data
        try:
            combo.blockSignals(True)
            if match_index >= 0:
                combo.setCurrentIndex(match_index)
            else:
                combo.setCurrentIndex(-1)
        finally:
            combo.blockSignals(False)
        return selected_id


__all__ = ["DocumentStatusWindow"]
