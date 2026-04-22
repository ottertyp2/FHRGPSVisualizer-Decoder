"""Acquisition result visualization tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS
from app.models import AcquisitionResult


class AcquisitionTab(QtWidgets.QWidget):
    """PRN selection and acquisition heatmap tab."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        help_label = QtWidgets.QLabel(
            f"{TOOLTIPS['ca_code']} {TOOLTIPS['despreading']}"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        controls = QtWidgets.QFormLayout()
        self.prn_spin = QtWidgets.QSpinBox()
        self.prn_spin.setRange(1, 32)
        self.prn_spin.setValue(1)
        self.doppler_min_spin = QtWidgets.QSpinBox()
        self.doppler_min_spin.setRange(-20_000, 0)
        self.doppler_min_spin.setValue(-6_000)
        self.doppler_max_spin = QtWidgets.QSpinBox()
        self.doppler_max_spin.setRange(0, 20_000)
        self.doppler_max_spin.setValue(6_000)
        self.doppler_step_spin = QtWidgets.QSpinBox()
        self.doppler_step_spin.setRange(50, 2_000)
        self.doppler_step_spin.setValue(250)
        controls.addRow("PRN", self.prn_spin)
        controls.addRow("Doppler min [Hz]", self.doppler_min_spin)
        controls.addRow("Doppler max [Hz]", self.doppler_max_spin)
        controls.addRow("Doppler step [Hz]", self.doppler_step_spin)
        layout.addLayout(controls)

        self.summary_label = QtWidgets.QLabel("No acquisition result.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.heatmap_plot = pg.PlotWidget(title="Code phase vs Doppler")
        self.heatmap_image = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_image)
        layout.addWidget(self.heatmap_plot, stretch=1)

        self.candidate_table = QtWidgets.QTableWidget(0, 3)
        self.candidate_table.setHorizontalHeaderLabels(["PRN", "Doppler [Hz]", "Metric"])
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.candidate_table, stretch=1)

    def update_result(self, result: AcquisitionResult) -> None:
        """Populate the heatmap and candidate table."""

        self.summary_label.setText(
            f"Best peak: PRN {result.best_candidate.prn}, "
            f"Doppler {result.best_candidate.doppler_hz:.1f} Hz, "
            f"code phase {result.best_candidate.code_phase_samples} samples, "
            f"metric {result.best_candidate.metric:.2f}"
        )
        self.heatmap_image.setImage(result.heatmap.T, autoLevels=True)
        rect = pg.QtCore.QRectF(
            float(result.doppler_bins_hz[0]),
            float(result.code_phases_samples[0]),
            float(result.doppler_bins_hz[-1] - result.doppler_bins_hz[0]) if result.doppler_bins_hz.size > 1 else 1.0,
            float(result.code_phases_samples[-1] - result.code_phases_samples[0]) if result.code_phases_samples.size > 1 else 1.0,
        )
        self.heatmap_image.setRect(rect)
        self.candidate_table.setRowCount(len(result.candidates))
        for row, candidate in enumerate(result.candidates):
            self.candidate_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(candidate.prn)))
            self.candidate_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{candidate.doppler_hz:.1f}"))
            self.candidate_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{candidate.metric:.2f}"))
