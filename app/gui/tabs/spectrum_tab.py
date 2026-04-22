"""Spectrum and waterfall tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS, compute_spectrum, compute_waterfall
from app.models import AcquisitionResult, SessionConfig


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
        self.search_region = pg.LinearRegionItem(values=(-6_000, 6_000), orientation="vertical", movable=False)
        self.search_region.setBrush(pg.mkBrush(255, 180, 0, 35))
        for line in self.search_region.lines:
            line.setPen(pg.mkPen((255, 180, 0, 140), width=2))
        self.center_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((100, 180, 255, 180), width=1.5))
        self.best_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((255, 80, 80, 220), width=2))
        self.spectrum_plot.addItem(self.search_region)
        self.spectrum_plot.addItem(self.center_line)
        self.spectrum_plot.addItem(self.best_line)
        layout.addWidget(self.spectrum_plot, stretch=1)

        self.search_label = QtWidgets.QLabel(
            "Search overlay: blue = nominal carrier after wipeoff, orange band = Doppler search range, red = best acquisition Doppler."
        )
        self.search_label.setWordWrap(True)
        layout.addWidget(self.search_label)

        self.waterfall_plot = pg.PlotWidget(title="Waterfall")
        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_image)
        layout.addWidget(self.waterfall_plot, stretch=1)

    def update_signal(
        self,
        samples: np.ndarray,
        sample_rate: float,
        session: SessionConfig | None = None,
        acquisition: AcquisitionResult | None = None,
    ) -> None:
        """Refresh spectrum and waterfall from a selected window."""

        if samples.size == 0:
            return
        fft_size = int(self.fft_size_combo.currentText())
        window_name = self.window_combo.currentText()
        average_count = int(self.average_spin.value())
        freqs, spectrum = compute_spectrum(samples, sample_rate, fft_size, window_name, average_count)
        self.spectrum_curve.setData(freqs, spectrum)
        self._update_search_overlay(session, acquisition)

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

    def _update_search_overlay(
        self,
        session: SessionConfig | None,
        acquisition: AcquisitionResult | None,
    ) -> None:
        """Show which frequency region the acquisition is currently searching."""

        session = session or SessionConfig()
        doppler_min = float(session.doppler_min)
        doppler_max = float(session.doppler_max)
        center_hz = 0.0 if session.is_baseband else float(session.if_frequency_hz)
        self.search_region.setRegion((center_hz + doppler_min, center_hz + doppler_max))
        self.center_line.setPos(center_hz)

        if acquisition is not None:
            self.best_line.setVisible(True)
            self.best_line.setPos(float(acquisition.best_candidate.carrier_frequency_hz))
            self.search_label.setText(
                "Search overlay: blue = IF / search center, "
                f"orange = searched band [{center_hz + doppler_min:.0f}, {center_hz + doppler_max:.0f}] Hz, "
                f"red = best PRN {acquisition.prn} peak at {acquisition.best_candidate.carrier_frequency_hz:.1f} Hz "
                f"(relative Doppler {acquisition.best_candidate.doppler_hz:+.1f} Hz)."
            )
        else:
            self.best_line.setVisible(False)
            self.search_label.setText(
                "Search overlay: blue = IF / search center, "
                f"orange = searched band [{center_hz + doppler_min:.0f}, {center_hz + doppler_max:.0f}] Hz."
            )
