"""Spectrum and waterfall tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS, compute_spectrum, compute_waterfall


class SpectrumTab(QtWidgets.QWidget):
    """FFT and waterfall diagnostic plots."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        help_label = QtWidgets.QLabel(
            f"{TOOLTIPS['spectrum']} {TOOLTIPS['waterfall']}"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        controls = QtWidgets.QHBoxLayout()
        self.fft_size_combo = QtWidgets.QComboBox()
        self.fft_size_combo.addItems(["1024", "2048", "4096", "8192"])
        self.fft_size_combo.setCurrentText("4096")
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman", "rectangular"])
        self.average_spin = QtWidgets.QSpinBox()
        self.average_spin.setRange(1, 64)
        self.average_spin.setValue(4)
        controls.addWidget(QtWidgets.QLabel("FFT size"))
        controls.addWidget(self.fft_size_combo)
        controls.addWidget(QtWidgets.QLabel("Window"))
        controls.addWidget(self.window_combo)
        controls.addWidget(QtWidgets.QLabel("Averages"))
        controls.addWidget(self.average_spin)
        controls.addStretch()
        layout.addLayout(controls)

        self.spectrum_plot = pg.PlotWidget(title="Spectrum")
        self.spectrum_curve = self.spectrum_plot.plot(pen="y")
        layout.addWidget(self.spectrum_plot, stretch=1)

        self.waterfall_plot = pg.PlotWidget(title="Waterfall")
        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_image)
        layout.addWidget(self.waterfall_plot, stretch=1)

    def update_signal(self, samples: np.ndarray, sample_rate: float) -> None:
        """Refresh spectrum and waterfall from a selected window."""

        if samples.size == 0:
            return
        fft_size = int(self.fft_size_combo.currentText())
        window_name = self.window_combo.currentText()
        average_count = int(self.average_spin.value())
        freqs, spectrum = compute_spectrum(samples, sample_rate, fft_size, window_name, average_count)
        self.spectrum_curve.setData(freqs, spectrum)

        wf_freqs, wf_times, waterfall = compute_waterfall(samples, sample_rate, fft_size=max(512, fft_size // 2), window_name=window_name)
        if waterfall.size:
            self.waterfall_image.setImage(waterfall.T, autoLevels=True)
            rect = pg.QtCore.QRectF(
                float(wf_times[0]) if wf_times.size else 0.0,
                float(wf_freqs[0]) if wf_freqs.size else 0.0,
                float(wf_times[-1] - wf_times[0]) if wf_times.size > 1 else 1.0,
                float(wf_freqs[-1] - wf_freqs[0]) if wf_freqs.size > 1 else 1.0,
            )
            self.waterfall_image.setRect(rect)
