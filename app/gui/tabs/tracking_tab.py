"""Tracking loop visualization tab."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS
from app.models import AcquisitionResult, BitDecisionResult, NavigationDecodeResult, TrackingState


class TrackingTab(QtWidgets.QWidget):
    """Plots for tracking loop states."""

    track_requested = QtCore.Signal()
    decode_requested = QtCore.Signal()
    selection_changed = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        help_label = QtWidgets.QLabel(
            f"{TOOLTIPS['early_late']} {TOOLTIPS['nav_bits']}"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        control_row = QtWidgets.QHBoxLayout()
        self.prn_combo = QtWidgets.QComboBox()
        self.prn_combo.addItem("No PRN")
        self.track_button = QtWidgets.QPushButton("Start Tracking For Selected PRN")
        self.decode_button = QtWidgets.QPushButton("Decode Selected PRN")
        control_row.addWidget(QtWidgets.QLabel("Satellite / PRN"))
        control_row.addWidget(self.prn_combo)
        control_row.addWidget(self.track_button)
        control_row.addWidget(self.decode_button)
        control_row.addStretch()
        layout.addLayout(control_row)

        self.status_label = QtWidgets.QLabel("Tracking not started.")
        layout.addWidget(self.status_label)

        self.evidence_text = QtWidgets.QPlainTextEdit()
        self.evidence_text.setReadOnly(True)
        self.evidence_text.setMaximumHeight(140)
        layout.addWidget(self.evidence_text)

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

        self.track_button.clicked.connect(self.track_requested.emit)
        self.decode_button.clicked.connect(self.decode_requested.emit)
        self.prn_combo.currentIndexChanged.connect(self._emit_selection_changed)

    def _emit_selection_changed(self) -> None:
        data = self.prn_combo.currentData()
        if data is None:
            return
        self.selection_changed.emit(int(data))

    def set_available_prns(self, prns: list[int], selected_prn: int | None = None) -> None:
        """Refresh the PRN selector for tracked or detected channels."""

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

    def update_state(
        self,
        state: TrackingState,
        acquisition: AcquisitionResult | None = None,
        bit_result: BitDecisionResult | None = None,
        nav_result: NavigationDecodeResult | None = None,
    ) -> None:
        """Refresh all tracking plots."""

        time_ms = state.times_s * 1_000.0
        self.set_available_prns([state.prn], state.prn)
        self.status_label.setText(
            f"PRN {state.prn} tracking {'locked' if state.lock_detected else 'not locked'} "
            f"after {state.times_s.size} ms."
        )
        evidence_lines = []
        if acquisition is not None:
            evidence_lines.extend(
                [
                    f"Acquisition start point: search frequency {acquisition.best_candidate.carrier_frequency_hz:.1f} Hz, "
                    f"relative Doppler {acquisition.best_candidate.doppler_hz:+.1f} Hz, "
                    f"code phase {acquisition.best_candidate.code_phase_samples} samples, metric {acquisition.best_candidate.metric:.2f}.",
                ]
            )
        evidence_lines.extend(
            [
                f"Median prompt magnitude: {float(state.prompt_mag.mean()):.4f}",
                f"Median lock metric: {float(state.lock_metric.mean()):.2f}",
                f"Tracking interpretation: {'carrier/code alignment looks stable' if state.lock_detected else 'alignment is still weak or noisy'}.",
            ]
        )
        if bit_result is not None:
            evidence_lines.append(f"Bit extraction already produced {bit_result.bit_values.size} bits.")
        if nav_result is not None:
            evidence_lines.append(
                f"Navigation view already found {len(nav_result.preamble_indices)} preambles and {nav_result.parity_ok_count} valid words."
            )
        self.evidence_text.setPlainText("\n".join(evidence_lines))
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
