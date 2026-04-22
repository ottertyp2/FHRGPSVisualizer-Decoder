"""Bit extraction and navigation display tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS, bits_to_str
from app.models import BitDecisionResult, NavigationDecodeResult


class NavigationTab(QtWidgets.QWidget):
    """Display 1 ms prompt values, bit sums, and LNAV words."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        help_label = QtWidgets.QLabel(
            f"{TOOLTIPS['nav_bits']} {TOOLTIPS['ca_code']}"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.summary_label = QtWidgets.QLabel("Navigation decoding not started.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.prompt_plot = pg.PlotWidget(title="1 ms prompt values")
        self.prompt_curve = self.prompt_plot.plot(pen="c")
        layout.addWidget(self.prompt_plot, stretch=1)

        self.bit_plot = pg.PlotWidget(title="20 ms bit accumulations")
        self.bit_curve = self.bit_plot.plot(pen=None, symbol="o", symbolBrush="y")
        layout.addWidget(self.bit_plot, stretch=1)

        self.bits_text = QtWidgets.QPlainTextEdit()
        self.bits_text.setReadOnly(True)
        layout.addWidget(self.bits_text, stretch=1)

        self.word_table = QtWidgets.QTableWidget(0, 5)
        self.word_table.setHorizontalHeaderLabels(["Start bit", "Label", "Parity", "Hex", "Bits"])
        self.word_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.word_table, stretch=1)

    def update_results(self, bit_result: BitDecisionResult, nav_result: NavigationDecodeResult) -> None:
        """Refresh the bit and LNAV views."""

        self.summary_label.setText(
            f"Bit offset: {bit_result.best_offset_ms} ms, "
            f"{bit_result.bit_values.size} hard decisions, "
            f"{len(nav_result.preamble_indices)} preambles, "
            f"{nav_result.parity_ok_count} words with valid parity."
        )
        self.prompt_curve.setData(np.arange(bit_result.prompt_ms.size), bit_result.prompt_ms)
        self.bit_curve.setData(np.arange(bit_result.bit_sums.size), bit_result.bit_sums)
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
            self.word_table.setItem(row, 3, QtWidgets.QTableWidgetItem(word.hex_word))
            self.word_table.setItem(row, 4, QtWidgets.QTableWidgetItem(word.bits))
