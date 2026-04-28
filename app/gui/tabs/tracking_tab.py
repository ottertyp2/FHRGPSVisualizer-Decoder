"""Tracking loop visualization tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS, decimate_for_display
from app.models import AcquisitionResult, BitDecisionResult, NavigationDecodeResult, SessionConfig, TrackingState


class TrackingTab(QtWidgets.QWidget):
    """Plots for tracking loop states."""

    track_requested = QtCore.Signal()
    decode_requested = QtCore.Signal()
    selection_changed = QtCore.Signal(int)
    settings_changed = QtCore.Signal()

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
            f"{TOOLTIPS['early_late']} {TOOLTIPS['nav_bits']}"
        )
        help_label.setWordWrap(True)
        sidebar_layout.addWidget(help_label)

        channel_group = QtWidgets.QGroupBox("Channel")
        channel_layout = QtWidgets.QVBoxLayout(channel_group)
        self.prn_combo = QtWidgets.QComboBox()
        self.prn_combo.addItem("No PRN")
        self.track_button = QtWidgets.QPushButton("Start Tracking For Selected PRN")
        self.decode_button = QtWidgets.QPushButton("Decode Selected PRN")
        channel_layout.addWidget(QtWidgets.QLabel("Satellite / PRN"))
        channel_layout.addWidget(self.prn_combo)
        channel_layout.addWidget(self.track_button)
        channel_layout.addWidget(self.decode_button)
        sidebar_layout.addWidget(channel_group)

        defaults = SessionConfig()
        loop_group = QtWidgets.QGroupBox("Tracking loop controls")
        loop_layout = QtWidgets.QFormLayout(loop_group)
        loop_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.early_late_spin = QtWidgets.QDoubleSpinBox()
        self.early_late_spin.setRange(0.05, 1.5)
        self.early_late_spin.setDecimals(2)
        self.early_late_spin.setSingleStep(0.05)
        self.early_late_spin.setValue(defaults.early_late_spacing_chips)
        self.early_late_spin.setSuffix(" chips")
        self.early_late_spin.setToolTip("Distance between Early and Late code replicas. Smaller is sharper but noisier.")
        self.dll_gain_spin = QtWidgets.QDoubleSpinBox()
        self.dll_gain_spin.setRange(0.0, 1.0)
        self.dll_gain_spin.setDecimals(3)
        self.dll_gain_spin.setSingleStep(0.01)
        self.dll_gain_spin.setValue(defaults.dll_gain)
        self.dll_gain_spin.setToolTip("Code tracking loop gain. Higher follows code timing faster but can become noisy.")
        self.pll_gain_spin = QtWidgets.QDoubleSpinBox()
        self.pll_gain_spin.setRange(0.0, 50.0)
        self.pll_gain_spin.setDecimals(2)
        self.pll_gain_spin.setSingleStep(0.5)
        self.pll_gain_spin.setValue(defaults.pll_gain)
        self.pll_gain_spin.setToolTip("Carrier phase loop gain. Higher follows phase faster but can overreact.")
        self.fll_gain_spin = QtWidgets.QDoubleSpinBox()
        self.fll_gain_spin.setRange(0.0, 2.0)
        self.fll_gain_spin.setDecimals(3)
        self.fll_gain_spin.setSingleStep(0.01)
        self.fll_gain_spin.setValue(defaults.fll_gain)
        self.fll_gain_spin.setToolTip("Early frequency pull-in gain used while the carrier is still settling.")
        self.reset_loop_button = QtWidgets.QPushButton("Reset Loop Defaults")
        loop_layout.addRow("Early/Late spacing", self.early_late_spin)
        loop_layout.addRow("DLL gain", self.dll_gain_spin)
        loop_layout.addRow("PLL gain", self.pll_gain_spin)
        loop_layout.addRow("FLL gain", self.fll_gain_spin)
        loop_layout.addRow(self.reset_loop_button)
        sidebar_layout.addWidget(loop_group)

        self.status_label = QtWidgets.QLabel("Tracking not started.")
        self.status_label.setWordWrap(True)
        self.task_status_label = QtWidgets.QLabel("Tracking idle.")
        self.task_status_label.setWordWrap(True)
        self.task_progress_bar = QtWidgets.QProgressBar()
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setValue(0)
        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        status_layout.addWidget(self.task_status_label)
        status_layout.addWidget(self.task_progress_bar)
        status_layout.addWidget(self.status_label)
        sidebar_layout.addWidget(status_group)

        guide_group = QtWidgets.QGroupBox("Quick guide")
        guide_layout = QtWidgets.QVBoxLayout(guide_group)
        self.stage_hint_label = QtWidgets.QLabel(
            "Tracking starts from the selected acquisition peak and then updates Doppler and code phase every millisecond."
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

        self.tracking_tabs = QtWidgets.QTabWidget()
        self.tracking_tabs.setDocumentMode(True)
        workspace_layout.addWidget(self.tracking_tabs, stretch=1)

        self.stage_iq_scatters: dict[str, pg.ScatterPlotItem] = {}
        iq_page = QtWidgets.QWidget()
        iq_layout = QtWidgets.QGridLayout(iq_page)
        iq_layout.setSpacing(8)
        stage_explanations = {
            "Raw IQ": "Raw IQ is the sum of all received signals plus front-end effects and noise.",
            "Carrier wiped": "Carrier wiped IQ has the selected Doppler hypothesis removed.",
            "Despread": "Despreading multiplies by the selected local PRN.",
            "Integrated prompt": "One complex value per 1 ms after carrier wipeoff and PRN despreading.",
        }
        for index, (name, explanation) in enumerate(stage_explanations.items()):
            panel = QtWidgets.QWidget()
            panel_layout = QtWidgets.QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            plot = pg.PlotWidget(title=name)
            plot.setMinimumHeight(250)
            plot.setToolTip(explanation)
            plot.setLabel("bottom", "I")
            plot.setLabel("left", "Q")
            plot.showGrid(x=True, y=True, alpha=0.2)
            scatter = pg.ScatterPlotItem(size=4, pen=pg.mkPen(None), brush=pg.mkBrush(80, 190, 255, 150))
            plot.addItem(scatter)
            text = QtWidgets.QLabel(explanation)
            text.setWordWrap(True)
            text.setMaximumHeight(46)
            self.stage_iq_scatters[name] = scatter
            panel_layout.addWidget(plot, stretch=1)
            panel_layout.addWidget(text)
            iq_layout.addWidget(panel, index // 2, index % 2)
        self.tracking_tabs.addTab(iq_page, "IQ Stages")

        loop_page = QtWidgets.QWidget()
        loop_layout_grid = QtWidgets.QGridLayout(loop_page)
        loop_layout_grid.setSpacing(8)

        self.prompt_plot = pg.PlotWidget(title="Prompt I/Q")
        self.prompt_plot.setMinimumHeight(210)
        self.prompt_plot.setToolTip("Prompt I/Q is one complex value per 1 ms after carrier wipeoff and PRN despreading.")
        self.prompt_i_curve = self.prompt_plot.plot(pen="c", name="I")
        self.prompt_q_curve = self.prompt_plot.plot(pen="m", name="Q")

        self.mag_plot = pg.PlotWidget(title="Early / Prompt / Late magnitude")
        self.mag_plot.setMinimumHeight(210)
        self.mag_plot.setToolTip("Prompt should usually sit above Early/Late when the code phase is aligned.")
        self.early_curve = self.mag_plot.plot(pen="r")
        self.prompt_mag_curve = self.mag_plot.plot(pen="y")
        self.late_curve = self.mag_plot.plot(pen="g")

        self.error_plot = pg.PlotWidget(title="Code and carrier error")
        self.error_plot.setMinimumHeight(210)
        self.error_plot.setToolTip("Code error steers PRN timing; carrier error steers residual Doppler/phase.")
        self.code_error_curve = self.error_plot.plot(pen="w")
        self.carrier_error_curve = self.error_plot.plot(pen="c")

        self.freq_plot = pg.PlotWidget(title="Estimated Doppler and code frequency")
        self.freq_plot.setMinimumHeight(210)
        self.doppler_curve = self.freq_plot.plot(pen="y")
        self.code_freq_curve = self.freq_plot.plot(pen="m")

        self.lock_plot = pg.PlotWidget(title="Lock metric")
        self.lock_plot.setMinimumHeight(180)
        self.lock_curve = self.lock_plot.plot(pen="g")

        loop_layout_grid.addWidget(self.prompt_plot, 0, 0)
        loop_layout_grid.addWidget(self.mag_plot, 0, 1)
        loop_layout_grid.addWidget(self.error_plot, 1, 0)
        loop_layout_grid.addWidget(self.freq_plot, 1, 1)
        loop_layout_grid.addWidget(self.lock_plot, 2, 0, 1, 2)
        self.tracking_tabs.addTab(loop_page, "Loop Trends")

        self.evidence_text = QtWidgets.QPlainTextEdit()
        self.evidence_text.setReadOnly(True)
        self.tracking_tabs.addTab(self.evidence_text, "Evidence")

        guide_page = QtWidgets.QWidget()
        what_layout = QtWidgets.QVBoxLayout(guide_page)
        what_label = QtWidgets.QLabel(
            "Tracking takes the acquisition peak as a starting point. "
            "It updates carrier frequency and code phase every millisecond so the selected PRN stays aligned. "
            "Early/Prompt/Late tell whether the local code is too early, on time, or too late."
        )
        what_label.setWordWrap(True)
        what_layout.addWidget(what_label)
        carrier_label = QtWidgets.QLabel(
            "Doppler compensation stops rotation for the selected channel hypothesis. "
            "The absolute residual IQ phase can still be arbitrary; PLL/tracking only tries to keep the prompt energy stable, usually near I."
        )
        carrier_label.setWordWrap(True)
        what_layout.addWidget(carrier_label)
        what_layout.addStretch()
        self.tracking_tabs.addTab(guide_page, "Guide")

        root_splitter.addWidget(workspace)
        root_splitter.setStretchFactor(0, 0)
        root_splitter.setStretchFactor(1, 1)
        root_splitter.setSizes([340, 1060])

        self.track_button.clicked.connect(self.track_requested.emit)
        self.decode_button.clicked.connect(self.decode_requested.emit)
        self.reset_loop_button.clicked.connect(self.reset_loop_controls)
        for spin in (self.early_late_spin, self.dll_gain_spin, self.pll_gain_spin, self.fll_gain_spin):
            spin.valueChanged.connect(lambda *_: self.settings_changed.emit())
        self.prn_combo.currentIndexChanged.connect(self._emit_selection_changed)

    def reset_loop_controls(self) -> None:
        """Restore the default teaching-oriented loop parameters."""

        defaults = SessionConfig()
        self.early_late_spin.setValue(defaults.early_late_spacing_chips)
        self.dll_gain_spin.setValue(defaults.dll_gain)
        self.pll_gain_spin.setValue(defaults.pll_gain)
        self.fll_gain_spin.setValue(defaults.fll_gain)
        self.settings_changed.emit()

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

    def set_task_message(self, message: str) -> None:
        """Show the current tracking-side job message."""

        self.task_status_label.setText(message)

    def set_task_progress(self, value: int) -> None:
        """Show tracking-side worker progress."""

        self.task_progress_bar.setValue(max(0, min(100, int(value))))

    def update_state(
        self,
        state: TrackingState,
        acquisition: AcquisitionResult | None = None,
        bit_result: BitDecisionResult | None = None,
        nav_result: NavigationDecodeResult | None = None,
    ) -> None:
        """Refresh all tracking plots."""

        time_ms = state.times_s * 1_000.0
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
                "Median prompt magnitude: "
                f"{float(np.median(state.prompt_mag)) if state.prompt_mag.size else 0.0:.4f}",
                "Prompt vs early/late ratio: "
                f"{float(np.median(state.prompt_mag) / max(np.median((state.early_mag + state.late_mag) * 0.5), 1e-12)) if state.prompt_mag.size else 0.0:.2f}",
                "Median lock metric: "
                f"{float(np.median(state.lock_metric)) if state.lock_metric.size else 0.0:.2f}",
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
        self._update_iq_stage_views(state)

    def _update_iq_stage_views(self, state: TrackingState) -> None:
        """Show compact IQ-plane views for each tracking processing stage."""

        for name, scatter in self.stage_iq_scatters.items():
            values = state.iq_views.get(name, np.empty(0, dtype=np.complex64))
            if values.size == 0:
                scatter.setData([], [])
                continue
            display, _step = decimate_for_display(values.astype(np.complex64, copy=False), max_points=1_500)
            scatter.setData(display.real, display.imag)
