"""Interactive GPS signal intuition tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from app.dsp.concept_lab import ConceptLabConfig, ConceptLabResult, generate_concept_lab_signal
from app.dsp.utils import decimate_for_display


class ConceptLabTab(QtWidgets.QWidget):
    """Small synthetic lab for the main GPS signal-flow concepts."""

    def __init__(self) -> None:
        super().__init__()
        self.current_result: ConceptLabResult | None = None
        self.step_index = -1

        outer_layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Signal Intuition uses a tiny synthetic GPS-like signal, not a file. "
            "It is meant for intuition: BPSK line, Doppler rotation, noise cloud, PRN correlation, and 1 ms prompts."
        )
        intro.setWordWrap(True)
        outer_layout.addWidget(intro)

        controls = QtWidgets.QGroupBox("Synthetic mini-signal controls")
        controls_layout = QtWidgets.QGridLayout(controls)
        self.prn_spin = QtWidgets.QSpinBox()
        self.prn_spin.setRange(1, 32)
        self.prn_spin.setValue(1)
        self.prn_spin.setToolTip("PRN / C/A code: the repeating 1023-chip pattern used to separate satellites.")
        self.doppler_spin = QtWidgets.QDoubleSpinBox()
        self.doppler_spin.setRange(-5_000.0, 5_000.0)
        self.doppler_spin.setValue(700.0)
        self.doppler_spin.setSuffix(" Hz")
        self.doppler_spin.setToolTip("Doppler bin: a frequency hypothesis used to stop IQ rotation, not a satellite ID.")
        self.code_phase_spin = QtWidgets.QSpinBox()
        self.code_phase_spin.setRange(0, 1022)
        self.code_phase_spin.setValue(120)
        self.code_phase_spin.setToolTip("Code phase: sample offset inside the repeated 1 ms C/A code, not an IQ angle.")
        self.noise_spin = QtWidgets.QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 2.0)
        self.noise_spin.setDecimals(2)
        self.noise_spin.setSingleStep(0.05)
        self.noise_spin.setValue(0.15)
        self.noise_spin.setToolTip("Noise level: turns clean BPSK points into a cloud like a real SDR capture.")
        self.second_checkbox = QtWidgets.QCheckBox("Enable second PRN")
        self.second_checkbox.setToolTip("A second PRN can share the same Doppler but still separate by code correlation.")
        self.second_prn_spin = QtWidgets.QSpinBox()
        self.second_prn_spin.setRange(1, 32)
        self.second_prn_spin.setValue(7)
        self.generate_button = QtWidgets.QPushButton("Generate Demo")
        self.step_button = QtWidgets.QPushButton("Step Through Explanation")

        controls_layout.addWidget(QtWidgets.QLabel("PRN"), 0, 0)
        controls_layout.addWidget(self.prn_spin, 0, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Doppler"), 0, 2)
        controls_layout.addWidget(self.doppler_spin, 0, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Code phase"), 0, 4)
        controls_layout.addWidget(self.code_phase_spin, 0, 5)
        controls_layout.addWidget(QtWidgets.QLabel("Noise"), 1, 0)
        controls_layout.addWidget(self.noise_spin, 1, 1)
        controls_layout.addWidget(self.second_checkbox, 1, 2)
        controls_layout.addWidget(QtWidgets.QLabel("Second PRN"), 1, 3)
        controls_layout.addWidget(self.second_prn_spin, 1, 4)
        controls_layout.addWidget(self.generate_button, 1, 5)
        controls_layout.addWidget(self.step_button, 1, 6)
        outer_layout.addWidget(controls)

        self.scenario_label = QtWidgets.QLabel("Custom demo. Press Step Through Explanation for guided presets.")
        self.scenario_label.setWordWrap(True)
        outer_layout.addWidget(self.scenario_label)

        help_group = QtWidgets.QGroupBox("Concept help")
        help_layout = QtWidgets.QVBoxLayout(help_group)
        help_label = QtWidgets.QLabel(
            "IQ phase is the angle of the complex sample; after carrier wipeoff it may become stable but not necessarily 0 degrees. "
            "Code phase is a time shift of the repeating 1 ms C/A code. "
            "Doppler is a frequency hypothesis; satellites are primarily separated by PRN code, so two PRNs can share one Doppler bin."
        )
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)
        outer_layout.addWidget(help_group)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll, stretch=1)

        self.time_plot, self.time_text = self._add_plot_row(
            "Time-domain I/Q",
            "The I and Q traces are the two coordinates of the complex SDR samples. "
            "Clean BPSK flips sign, Doppler makes those signs rotate over time, and noise adds random motion. "
            "This view shows why raw samples alone are hard to read by eye.",
        )
        self.time_i_curve = self.time_plot.plot(pen="c")
        self.time_q_curve = self.time_plot.plot(pen="m")
        self.time_plot.setLabel("bottom", "Time", units="ms")
        self.time_plot.setLabel("left", "Amplitude")

        self.iq_plot, self.iq_text = self._add_plot_row(
            "IQ Plane",
            "A single BPSK channel without frequency error lies on a line with two opposite states. "
            "Doppler rotates that line, and noise plus other PRNs turn it into a cloud. "
            "A cloud is normal for raw GPS IQ because many weak signals are overlaid.",
        )
        self.iq_ideal_scatter = pg.ScatterPlotItem(size=4, pen=pg.mkPen(None), brush=pg.mkBrush(120, 120, 120, 80))
        self.iq_raw_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(70, 180, 255, 160))
        self.iq_plot.addItem(self.iq_ideal_scatter)
        self.iq_plot.addItem(self.iq_raw_scatter)
        self.iq_plot.setLabel("bottom", "I")
        self.iq_plot.setLabel("left", "Q")

        self.wiped_plot, self.wiped_text = self._add_plot_row(
            "Carrier wiped IQ",
            "Carrier wipeoff multiplies by the opposite Doppler frequency to stop the rotation for this channel hypothesis. "
            "It does not force the absolute IQ phase to 0 degrees. "
            "PLL/tracking later tries to keep energy mostly on I, but a constant residual phase can remain.",
        )
        self.wiped_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 210, 80, 160))
        self.wiped_plot.addItem(self.wiped_scatter)
        self.wiped_plot.setLabel("bottom", "I")
        self.wiped_plot.setLabel("left", "Q")

        self.correlation_plot, self.correlation_text = self._add_plot_row(
            "PRN correlation vs code phase",
            "The local PRN code is shifted through the 1 ms code period and multiplied with the wiped signal. "
            "The peak is a time offset: where the selected PRN code lines up. "
            "Code phase is not an IQ angle; it is sample timing inside the repeated C/A code.",
        )
        self.correlation_curve = self.correlation_plot.plot(pen="y")
        self.correlation_peak = pg.ScatterPlotItem(size=9, pen=pg.mkPen("#111111"), brush=pg.mkBrush("#ffd43b"))
        self.correlation_plot.addItem(self.correlation_peak)
        self.correlation_plot.setLabel("bottom", "Code phase", units="samples")
        self.correlation_plot.setLabel("left", "Correlation")

        self.acq_plot, self.acq_text = self._add_plot_row(
            "Mini acquisition heatmap",
            "Acquisition searches Doppler and code phase together. "
            "For each Doppler bin, the assumed carrier rotation is removed and then PRN correlation is tested over code phase. "
            "A bright peak says this PRN, this Doppler hypothesis, and this code timing fit best.",
        )
        self.acq_image = pg.ImageItem(axisOrder="row-major")
        self.acq_plot.addItem(self.acq_image)
        self.acq_plot.setLabel("bottom", "Code phase", units="samples")
        self.acq_plot.setLabel("left", "Doppler", units="Hz")

        self.prompt_plot, self.prompt_text = self._add_plot_row(
            "Integrated prompt points",
            "After carrier wipeoff and despreading, one complex prompt value is formed per millisecond. "
            "These 1 ms prompt values are where a tracked BPSK channel most clearly becomes a line or two clusters. "
            "LNAV bits are later formed by summing 20 of these 1 ms values.",
        )
        self.prompt_scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(120, 255, 160, 180))
        self.prompt_plot.addItem(self.prompt_scatter)
        self.prompt_plot.setLabel("bottom", "I")
        self.prompt_plot.setLabel("left", "Q")

        self.generate_button.clicked.connect(self.generate_demo)
        self.step_button.clicked.connect(self.step_through)
        self.generate_demo()

    def _add_plot_row(self, title: str, explanation: str) -> tuple[pg.PlotWidget, QtWidgets.QLabel]:
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        plot = pg.PlotWidget(title=title)
        plot.setMinimumHeight(180)
        plot.showGrid(x=True, y=True, alpha=0.25)
        text = QtWidgets.QLabel(explanation)
        text.setWordWrap(True)
        text.setMinimumWidth(280)
        text.setMaximumWidth(420)
        row_layout.addWidget(plot, stretch=3)
        row_layout.addWidget(text, stretch=1)
        self.plot_layout.addWidget(row)
        return plot, text

    def _config_from_controls(self) -> ConceptLabConfig:
        second_prn = int(self.second_prn_spin.value())
        prn = int(self.prn_spin.value())
        if second_prn == prn:
            second_prn = 2 if prn != 2 else 3
        return ConceptLabConfig(
            prn=prn,
            doppler_hz=float(self.doppler_spin.value()),
            code_phase_samples=int(self.code_phase_spin.value()),
            noise_std=float(self.noise_spin.value()),
            second_prn_enabled=self.second_checkbox.isChecked(),
            second_prn=second_prn,
        )

    def generate_demo(self) -> None:
        """Generate a synthetic signal from the current controls."""

        self.current_result = generate_concept_lab_signal(self._config_from_controls())
        self._update_plots(self.current_result)

    def step_through(self) -> None:
        """Cycle through a small set of didactic presets."""

        scenarios = [
            (
                "1. Clean BPSK: no Doppler, no noise, one PRN. The IQ plane should be a simple line with two states.",
                {"doppler": 0.0, "noise": 0.0, "second": False, "code_phase": 120},
            ),
            (
                "2. Add Doppler: the same BPSK states now rotate in the IQ plane because the carrier frequency is offset.",
                {"doppler": 900.0, "noise": 0.0, "second": False, "code_phase": 120},
            ),
            (
                "3. Add noise: the rotating points become a cloud, like a real weak GPS recording.",
                {"doppler": 900.0, "noise": 0.35, "second": False, "code_phase": 120},
            ),
            (
                "4. Add a second PRN at the same Doppler: Doppler alone does not identify a satellite; PRN correlation separates them.",
                {"doppler": 900.0, "noise": 0.20, "second": True, "code_phase": 120},
            ),
            (
                "5. Wipe carrier and correlate PRN: rotation stops, codephase peak appears, and 1 ms prompts become the bit source.",
                {"doppler": 900.0, "noise": 0.15, "second": True, "code_phase": 320},
            ),
        ]
        self.step_index = (self.step_index + 1) % len(scenarios)
        label, values = scenarios[self.step_index]
        self.doppler_spin.setValue(values["doppler"])
        self.noise_spin.setValue(values["noise"])
        self.second_checkbox.setChecked(values["second"])
        self.code_phase_spin.setValue(values["code_phase"])
        self.scenario_label.setText(label)
        self.generate_demo()

    def _update_plots(self, result: ConceptLabResult) -> None:
        display_count = min(result.raw_iq.size, 4_000)
        time_ms = result.time_s[:display_count] * 1_000.0
        raw = result.raw_iq[:display_count]
        self.time_i_curve.setData(time_ms, raw.real)
        self.time_q_curve.setData(time_ms, raw.imag)

        ideal_display, _ideal_step = decimate_for_display(result.ideal_bpsk[:display_count], max_points=2_500)
        raw_display, _raw_step = decimate_for_display(raw, max_points=2_500)
        self.iq_ideal_scatter.setData(ideal_display.real, ideal_display.imag)
        self.iq_raw_scatter.setData(raw_display.real, raw_display.imag)

        wiped_display, _wiped_step = decimate_for_display(result.carrier_wiped_iq[:display_count], max_points=2_500)
        self.wiped_scatter.setData(wiped_display.real, wiped_display.imag)

        self.correlation_curve.setData(result.correlation_code_phases, result.correlation_values)
        peak_index = int(np.argmax(result.correlation_values))
        self.correlation_peak.setData(
            [float(result.correlation_code_phases[peak_index])],
            [float(result.correlation_values[peak_index])],
        )

        self.acq_image.setImage(result.acquisition_heatmap, autoLevels=True)
        doppler_min = float(result.acquisition_doppler_bins_hz[0])
        doppler_max = float(result.acquisition_doppler_bins_hz[-1])
        self.acq_image.setRect(
            QtCore.QRectF(
                0.0,
                doppler_min,
                float(max(1, result.acquisition_heatmap.shape[1] - 1)),
                float(max(1.0, doppler_max - doppler_min)),
            )
        )
        self.acq_plot.setXRange(0.0, float(result.acquisition_heatmap.shape[1] - 1), padding=0.02)
        self.acq_plot.setYRange(doppler_min, doppler_max, padding=0.02)

        prompt_display, _prompt_step = decimate_for_display(result.prompt_points, max_points=500)
        self.prompt_scatter.setData(prompt_display.real, prompt_display.imag)

        for plot in (self.iq_plot, self.wiped_plot, self.prompt_plot):
            plot.enableAutoRange()
