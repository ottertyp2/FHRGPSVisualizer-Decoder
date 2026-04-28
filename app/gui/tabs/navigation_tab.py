"""Bit extraction and navigation display tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS, bits_to_str
from app.models import AcquisitionResult, BitDecisionResult, NavigationDecodeResult, TrackingState


class NavigationTab(QtWidgets.QWidget):
    """Display 1 ms prompt values, bit sums, and LNAV words."""

    decode_requested = QtCore.Signal()
    selection_changed = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        root_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_splitter.setChildrenCollapsible(False)
        layout.addWidget(root_splitter)

        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setMinimumWidth(300)
        sidebar_scroll.setMaximumWidth(430)
        sidebar = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 8, 0)
        sidebar_layout.setSpacing(8)
        sidebar_scroll.setWidget(sidebar)

        help_label = QtWidgets.QLabel(
            f"{TOOLTIPS['nav_bits']} {TOOLTIPS['ca_code']}"
        )
        help_label.setWordWrap(True)
        sidebar_layout.addWidget(help_label)

        decode_group = QtWidgets.QGroupBox("Decode setup")
        decode_layout = QtWidgets.QVBoxLayout(decode_group)
        self.prn_combo = QtWidgets.QComboBox()
        self.prn_combo.addItem("No PRN")
        self.bit_source_combo = QtWidgets.QComboBox()
        self.bit_source_combo.addItem("Auto source", "auto")
        self.bit_source_combo.addItem("Carrier-aligned prompt", "carrier_aligned")
        self.bit_source_combo.addItem("Prompt I only", "prompt_i")
        self.bit_source_combo.addItem("Prompt Q only", "prompt_q")
        self.bit_source_combo.setToolTip(
            "Choose which tracked prompt component is converted into 20 ms navigation bits."
        )
        self.decode_button = QtWidgets.QPushButton("Decode Selected PRN")
        decode_layout.addWidget(QtWidgets.QLabel("Satellite / PRN"))
        decode_layout.addWidget(self.prn_combo)
        decode_layout.addWidget(QtWidgets.QLabel("Bit source"))
        decode_layout.addWidget(self.bit_source_combo)
        decode_layout.addWidget(self.decode_button)
        sidebar_layout.addWidget(decode_group)

        self.summary_label = QtWidgets.QLabel("Navigation decoding not started.")
        self.summary_label.setWordWrap(True)
        self.task_status_label = QtWidgets.QLabel("Navigation decoder idle.")
        self.task_status_label.setWordWrap(True)
        self.task_progress_bar = QtWidgets.QProgressBar()
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setValue(0)
        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        status_layout.addWidget(self.task_status_label)
        status_layout.addWidget(self.task_progress_bar)
        status_layout.addWidget(self.summary_label)
        sidebar_layout.addWidget(status_group)

        guide_group = QtWidgets.QGroupBox("Quick guide")
        guide_layout = QtWidgets.QVBoxLayout(guide_group)
        self.stage_hint_label = QtWidgets.QLabel(
            "Navigation decoding uses the tracked PRN only: 1 ms prompt values are summed into 20 ms LNAV bit decisions."
        )
        self.stage_hint_label.setWordWrap(True)
        guide_layout.addWidget(self.stage_hint_label)
        sidebar_layout.addWidget(guide_group)
        sidebar_layout.addStretch()
        root_splitter.addWidget(sidebar_scroll)

        workspace = QtWidgets.QWidget()
        workspace_layout = QtWidgets.QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(8)

        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.prompt_plot = pg.PlotWidget(title="1 ms prompt values")
        self.prompt_plot.setMinimumHeight(300)
        self.prompt_plot.setToolTip("These are the per-millisecond despread prompt values before 20 ms bit summing.")
        self.prompt_curve = self.prompt_plot.plot(pen="c")
        self.bit_plot = pg.PlotWidget(title="20 ms bit accumulations")
        self.bit_plot.setMinimumHeight(300)
        self.bit_plot.setToolTip("Each point is the sum of 20 prompt values; its sign becomes one navigation bit.")
        self.bit_curve = self.bit_plot.plot(pen=None, symbol="o", symbolBrush="y")
        plot_splitter.addWidget(self.prompt_plot)
        plot_splitter.addWidget(self.bit_plot)
        plot_splitter.setStretchFactor(0, 1)
        plot_splitter.setStretchFactor(1, 1)
        workspace_layout.addWidget(plot_splitter, stretch=2)

        self.navigation_tabs = QtWidgets.QTabWidget()
        self.navigation_tabs.setDocumentMode(True)
        workspace_layout.addWidget(self.navigation_tabs, stretch=2)

        self.bits_text = QtWidgets.QPlainTextEdit()
        self.bits_text.setReadOnly(True)
        self.navigation_tabs.addTab(self.bits_text, "Bit Stream")

        self._current_nav_result: NavigationDecodeResult | None = None

        self.word_table = QtWidgets.QTableWidget(0, 7)
        self.word_table.setHorizontalHeaderLabels(
            ["Start bit", "Label", "Parity", "Corrected", "Corrected bit", "Hex", "Bits"]
        )
        self._configure_table(self.word_table)
        self.word_table.horizontalHeader().setStretchLastSection(True)
        self.navigation_tabs.addTab(self.word_table, "LNAV Words")

        self.subframe_table = QtWidgets.QTableWidget(0, 7)
        self.subframe_table.setHorizontalHeaderLabels(
            ["Start bit", "Subframe ID", "TOW [s]", "Category", "Parity OK", "Corrected", "Status"]
        )
        self._configure_table(self.subframe_table)
        self.subframe_table.horizontalHeader().setStretchLastSection(True)
        self.subframe_table.itemSelectionChanged.connect(self._refresh_decoded_fields_table)
        self.navigation_tabs.addTab(self.subframe_table, "Subframes")

        self.fields_table = QtWidgets.QTableWidget(0, 5)
        self.fields_table.setHorizontalHeaderLabels(["Subframe", "Field", "Value", "Bit range", "Description"])
        self._configure_table(self.fields_table)
        self.fields_table.horizontalHeader().setStretchLastSection(True)
        self.navigation_tabs.addTab(self.fields_table, "Decoded Fields")

        self.almanac_ephemeris_text = QtWidgets.QPlainTextEdit()
        self.almanac_ephemeris_text.setReadOnly(True)
        self.navigation_tabs.addTab(self.almanac_ephemeris_text, "Almanac / Ephemeris")

        self.evidence_text = QtWidgets.QPlainTextEdit()
        self.evidence_text.setReadOnly(True)
        self.navigation_tabs.addTab(self.evidence_text, "Evidence")

        guide_page = QtWidgets.QWidget()
        what_layout = QtWidgets.QVBoxLayout(guide_page)
        what_label = QtWidgets.QLabel(
            "Navigation works on tracked Prompt values, not on raw IQ. "
            "Each prompt point is already carrier-wiped and despread for the selected PRN. "
            "The PRN correlation separates satellites; the 20 ms sum improves the 50 bps LNAV bit decision."
        )
        what_label.setWordWrap(True)
        what_layout.addWidget(what_label)
        bit_help_label = QtWidgets.QLabel(
            "GPS LNAV is 50 bps, so one data bit lasts 20 ms. "
            "The decoder tries the 20 possible millisecond offsets, sums twenty 1 ms prompt values, and uses the sign as the hard bit."
        )
        bit_help_label.setWordWrap(True)
        what_layout.addWidget(bit_help_label)
        what_layout.addStretch()
        self.navigation_tabs.addTab(guide_page, "Guide")

        root_splitter.addWidget(workspace)
        root_splitter.setStretchFactor(0, 0)
        root_splitter.setStretchFactor(1, 1)
        root_splitter.setSizes([340, 1060])

        self.decode_button.clicked.connect(self.decode_requested.emit)
        self.prn_combo.currentIndexChanged.connect(self._emit_selection_changed)

    @staticmethod
    def _configure_table(table: QtWidgets.QTableWidget) -> None:
        """Make navigation word tables readable at narrow and wide sizes."""

        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setMinimumSectionSize(70)

    @staticmethod
    def _subframe_status(parity_ok_words: int, total_words: int, corrected_words: int) -> str:
        """Return a compact status label for the subframe table."""

        if total_words and parity_ok_words == total_words:
            return "valid" if corrected_words == 0 else "valid after correction"
        if parity_ok_words:
            return "partial / uncertain"
        return "unverified"

    def _refresh_decoded_fields_table(self) -> None:
        """Show decoded fields for selected subframes, or all if none is selected."""

        nav_result = self._current_nav_result
        if nav_result is None:
            self.fields_table.setRowCount(0)
            return

        selected_rows = sorted(index.row() for index in self.subframe_table.selectionModel().selectedRows())
        if selected_rows:
            subframes = [nav_result.subframes[row] for row in selected_rows if row < len(nav_result.subframes)]
        else:
            subframes = nav_result.subframes

        rows = [(subframe, field) for subframe in subframes for field in subframe.fields]
        self.fields_table.setRowCount(len(rows))
        for row, (subframe, field) in enumerate(rows):
            subframe_label = (
                f"{subframe.subframe_id} @ {subframe.start_bit}"
                if subframe.subframe_id is not None
                else f"? @ {subframe.start_bit}"
            )
            self.fields_table.setItem(row, 0, QtWidgets.QTableWidgetItem(subframe_label))
            self.fields_table.setItem(row, 1, QtWidgets.QTableWidgetItem(field.name))
            self.fields_table.setItem(row, 2, QtWidgets.QTableWidgetItem(field.value))
            self.fields_table.setItem(row, 3, QtWidgets.QTableWidgetItem(field.bit_range))
            self.fields_table.setItem(row, 4, QtWidgets.QTableWidgetItem(field.description))
        self.fields_table.resizeColumnsToContents()
        self.fields_table.horizontalHeader().setStretchLastSection(True)

    def _format_almanac_ephemeris(self, nav_result: NavigationDecodeResult) -> str:
        """Build a grouped raw payload overview for clock, ephemeris, and almanac areas."""

        if not nav_result.subframes:
            return "No complete 10-word subframes decoded yet."

        groups = [
            ("Clock/Health", [subframe for subframe in nav_result.subframes if subframe.subframe_id == 1]),
            ("Ephemeris part 1", [subframe for subframe in nav_result.subframes if subframe.subframe_id == 2]),
            ("Ephemeris part 2", [subframe for subframe in nav_result.subframes if subframe.subframe_id == 3]),
            (
                "Almanac pages",
                [subframe for subframe in nav_result.subframes if subframe.subframe_id in (4, 5)],
            ),
        ]
        lines: list[str] = []
        for title, subframes in groups:
            lines.append(title)
            if not subframes:
                lines.append("  raw / not fully decoded yet: no matching subframe decoded.")
                lines.append("")
                continue
            for subframe in subframes:
                tow = subframe.tow_seconds if subframe.tow_seconds is not None else "?"
                page = f", {subframe.page_label}" if subframe.page_label else ""
                lines.append(
                    f"  start bit {subframe.start_bit}, TOW {tow}, parity "
                    f"{subframe.parity_ok_words}/{len(subframe.words)}, corrected {subframe.corrected_words}{page}"
                )
                lines.append("  raw / not fully decoded yet")
            lines.append("")
        return "\n".join(lines).strip()

    def _emit_selection_changed(self) -> None:
        data = self.prn_combo.currentData()
        if data is None:
            return
        self.selection_changed.emit(int(data))

    def set_available_prns(self, prns: list[int], selected_prn: int | None = None) -> None:
        """Refresh the per-satellite navigation selector."""

        self.prn_combo.blockSignals(True)
        self.prn_combo.clear()
        if not prns:
            self.prn_combo.addItem("No PRN", None)
        for prn in sorted(prns):
            self.prn_combo.addItem(f"PRN {prn}", prn)
        if selected_prn is not None:
            index = self.prn_combo.findData(selected_prn)
            if index >= 0:
                self.prn_combo.setCurrentIndex(index)
        self.prn_combo.blockSignals(False)

    def set_task_message(self, message: str) -> None:
        """Show the current navigation-side job message."""

        self.task_status_label.setText(message)

    def set_task_progress(self, value: int) -> None:
        """Show navigation-side worker progress."""

        self.task_progress_bar.setValue(max(0, min(100, int(value))))

    def bit_source_mode(self) -> str:
        """Return the selected 20 ms bit source mode."""

        return str(self.bit_source_combo.currentData())

    def update_results(
        self,
        bit_result: BitDecisionResult,
        nav_result: NavigationDecodeResult,
        prn: int,
        acquisition: AcquisitionResult | None = None,
        tracking: TrackingState | None = None,
    ) -> None:
        """Refresh the bit and LNAV views."""

        self._current_nav_result = nav_result
        self.summary_label.setText(
            f"PRN {prn}: bit offset {bit_result.best_offset_ms} ms, "
            f"{bit_result.bit_values.size} hard decisions, "
            f"{len(nav_result.preamble_indices)} preambles, "
            f"{len(nav_result.subframes)} subframes, "
            f"{nav_result.parity_ok_count} parity-valid words, "
            f"{nav_result.corrected_word_count} corrected words, "
            f"{nav_result.failed_word_count} failed/unchecked words."
        )
        self.prompt_curve.setData(np.arange(bit_result.prompt_ms.size), bit_result.prompt_ms)
        self.bit_curve.setData(np.arange(bit_result.bit_sums.size), bit_result.bit_sums)
        evidence_lines = []
        if acquisition is not None:
            evidence_lines.append(
                f"Started from acquisition peak at search frequency {acquisition.best_candidate.carrier_frequency_hz:.1f} Hz "
                f"(relative Doppler {acquisition.best_candidate.doppler_hz:+.1f} Hz) and code phase {acquisition.best_candidate.code_phase_samples}."
            )
        if tracking is not None:
            evidence_lines.append(
                f"Tracking {'locked' if tracking.lock_detected else 'did not fully lock'} with average lock metric {float(tracking.lock_metric.mean()):.2f}."
            )
        evidence_lines.extend(
            [
                f"20 ms integration offset: {bit_result.best_offset_ms} ms",
                f"Detected preambles: {len(nav_result.preamble_indices)}",
                f"Decoded subframes: {len(nav_result.subframes)}",
                f"Parity-valid words: {nav_result.parity_ok_count}",
                f"Corrected words: {nav_result.corrected_word_count}",
                f"Failed/unchecked words: {nav_result.failed_word_count}",
                "Interpretation: this view shows the actual bit stream that was formed after despreading and tracking for the selected PRN only.",
            ]
        )
        self.evidence_text.setPlainText("\n".join(evidence_lines))
        preview_lines = [
            f"Bit decisions: {bits_to_str(bit_result.bit_values[:200])}",
            "",
            *nav_result.summary_lines,
        ]
        self.bits_text.setPlainText("\n".join(preview_lines))
        self.word_table.setRowCount(len(nav_result.words))
        for row, word in enumerate(nav_result.words):
            self.word_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(word.start_bit)))
            self.word_table.setItem(row, 1, QtWidgets.QTableWidgetItem(word.label or "-"))
            self.word_table.setItem(row, 2, QtWidgets.QTableWidgetItem("OK" if word.parity_ok else "Fail"))
            self.word_table.setItem(row, 3, QtWidgets.QTableWidgetItem("yes" if word.corrected else "no"))
            self.word_table.setItem(
                row,
                4,
                QtWidgets.QTableWidgetItem(
                    str(word.corrected_bit_index) if word.corrected_bit_index is not None else "-"
                ),
            )
            self.word_table.setItem(row, 5, QtWidgets.QTableWidgetItem(word.hex_word))
            self.word_table.setItem(row, 6, QtWidgets.QTableWidgetItem(word.bits))
        self.word_table.resizeColumnsToContents()
        self.word_table.horizontalHeader().setStretchLastSection(True)

        self.subframe_table.blockSignals(True)
        self.subframe_table.setRowCount(len(nav_result.subframes))
        for row, subframe in enumerate(nav_result.subframes):
            self.subframe_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(subframe.start_bit)))
            self.subframe_table.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(str(subframe.subframe_id) if subframe.subframe_id is not None else "-"),
            )
            self.subframe_table.setItem(
                row,
                2,
                QtWidgets.QTableWidgetItem(str(subframe.tow_seconds) if subframe.tow_seconds is not None else "-"),
            )
            self.subframe_table.setItem(row, 3, QtWidgets.QTableWidgetItem(subframe.category or "Unknown"))
            self.subframe_table.setItem(
                row,
                4,
                QtWidgets.QTableWidgetItem(f"{subframe.parity_ok_words}/{len(subframe.words)}"),
            )
            self.subframe_table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(subframe.corrected_words)))
            self.subframe_table.setItem(
                row,
                6,
                QtWidgets.QTableWidgetItem(
                    self._subframe_status(subframe.parity_ok_words, len(subframe.words), subframe.corrected_words)
                ),
            )
        self.subframe_table.blockSignals(False)
        self.subframe_table.resizeColumnsToContents()
        self.subframe_table.horizontalHeader().setStretchLastSection(True)
        self._refresh_decoded_fields_table()
        self.almanac_ephemeris_text.setPlainText(self._format_almanac_ephemeris(nav_result))
        self.navigation_tabs.setCurrentWidget(self.word_table)
