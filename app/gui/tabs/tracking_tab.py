"""Tracking loop visualization tab."""

from __future__ import annotations

from PySide6 import QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS
from app.models import TrackingState


class TrackingTab(QtWidgets.QWidget):
    """Plots for tracking loop states."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        help_label = QtWidgets.QLabel(
            f"{TOOLTIPS['early_late']} {TOOLTIPS['nav_bits']}"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.status_label = QtWidgets.QLabel("Tracking not started.")
        layout.addWidget(self.status_label)

        self.prompt_plot = pg.PlotWidget(title="Prompt I/Q")
        self.prompt_i_curve = self.prompt_plot.plot(pen="c", name="I")
        self.prompt_q_curve = self.prompt_plot.plot(pen="m", name="Q")
        layout.addWidget(self.prompt_plot, stretch=1)

        self.mag_plot = pg.PlotWidget(title="Early / Prompt / Late magnitude")
        self.early_curve = self.mag_plot.plot(pen="r")
        self.prompt_mag_curve = self.mag_plot.plot(pen="y")
        self.late_curve = self.mag_plot.plot(pen="g")
        layout.addWidget(self.mag_plot, stretch=1)

        self.error_plot = pg.PlotWidget(title="Code and carrier error")
        self.code_error_curve = self.error_plot.plot(pen="w")
        self.carrier_error_curve = self.error_plot.plot(pen="c")
        layout.addWidget(self.error_plot, stretch=1)

        self.freq_plot = pg.PlotWidget(title="Estimated Doppler and code frequency")
        self.doppler_curve = self.freq_plot.plot(pen="y")
        self.code_freq_curve = self.freq_plot.plot(pen="m")
        layout.addWidget(self.freq_plot, stretch=1)

        self.lock_plot = pg.PlotWidget(title="Lock metric")
        self.lock_curve = self.lock_plot.plot(pen="g")
        layout.addWidget(self.lock_plot, stretch=1)

    def update_state(self, state: TrackingState) -> None:
        """Refresh all tracking plots."""

        time_ms = state.times_s * 1_000.0
        self.status_label.setText(
            f"PRN {state.prn} tracking {'locked' if state.lock_detected else 'not locked'} "
            f"after {state.times_s.size} ms."
        )
        self.prompt_i_curve.setData(time_ms, state.prompt_i)
        self.prompt_q_curve.setData(time_ms, state.prompt_q)
        self.early_curve.setData(time_ms, state.early_mag)
        self.prompt_mag_curve.setData(time_ms, state.prompt_mag)
        self.late_curve.setData(time_ms, state.late_mag)
        self.code_error_curve.setData(time_ms, state.code_error)
        self.carrier_error_curve.setData(time_ms, state.carrier_error)
        self.doppler_curve.setData(time_ms, state.doppler_est_hz)
        self.code_freq_curve.setData(time_ms, state.code_freq_est_hz)
        self.lock_curve.setData(time_ms, state.lock_metric)
