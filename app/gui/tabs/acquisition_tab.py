"""Acquisition result visualization tab."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from app.dsp.acquisition import acquisition_interpretation
from app.dsp.utils import TOOLTIPS
from app.models import AcquisitionResult


class AcquisitionTab(QtWidgets.QWidget):
    """PRN selection and acquisition heatmap tab."""

    run_requested = QtCore.Signal()
    scan_requested = QtCore.Signal()
    sweep_requested = QtCore.Signal()
    auto_rate_survey_requested = QtCore.Signal()
    track_selected_requested = QtCore.Signal()
    selection_changed = QtCore.Signal(int)
    sweep_selection_changed = QtCore.Signal(float, int)
    sample_rate_selection_changed = QtCore.Signal(float, int)

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
        self.integration_spin = QtWidgets.QSpinBox()
        self.integration_spin.setRange(1, 500)
        self.integration_spin.setValue(80)
        self.integration_spin.setSuffix(" ms blocks")
        self.segment_count_spin = QtWidgets.QSpinBox()
        self.segment_count_spin.setRange(1, 32)
        self.segment_count_spin.setValue(8)
        self.segment_count_spin.setSuffix(" segments")
        self.spread_blocks_checkbox = QtWidgets.QCheckBox("Spread 1 ms blocks across loaded source")
        self.spread_blocks_checkbox.setChecked(False)
        controls.addRow("PRN", self.prn_spin)
        controls.addRow("Doppler min [Hz]", self.doppler_min_spin)
        controls.addRow("Doppler max [Hz]", self.doppler_max_spin)
        controls.addRow("Doppler step [Hz]", self.doppler_step_spin)
        controls.addRow("Accumulation", self.integration_spin)
        controls.addRow("Deep search", self.segment_count_spin)
        controls.addRow("Weak-signal mode", self.spread_blocks_checkbox)
        layout.addLayout(controls)

        sweep_group = QtWidgets.QGroupBox("Auto-search IF / center sweep")
        sweep_layout = QtWidgets.QFormLayout(sweep_group)
        self.center_min_spin = QtWidgets.QSpinBox()
        self.center_min_spin.setRange(-100_000, 100_000)
        self.center_min_spin.setValue(-10_000)
        self.center_max_spin = QtWidgets.QSpinBox()
        self.center_max_spin.setRange(-100_000, 100_000)
        self.center_max_spin.setValue(10_000)
        self.center_step_spin = QtWidgets.QSpinBox()
        self.center_step_spin.setRange(100, 20_000)
        self.center_step_spin.setValue(500)
        self.center_sweep_button = QtWidgets.QPushButton("Sweep Search Center")
        self.auto_detect_button = QtWidgets.QPushButton("Auto Detect Capture")
        sweep_layout.addRow("Center min [Hz]", self.center_min_spin)
        sweep_layout.addRow("Center max [Hz]", self.center_max_spin)
        sweep_layout.addRow("Center step [Hz]", self.center_step_spin)
        sweep_layout.addRow(self.center_sweep_button)
        sweep_layout.addRow(self.auto_detect_button)
        layout.addWidget(sweep_group)

        action_row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run Acquisition For PRN")
        self.scan_button = QtWidgets.QPushButton("Scan PRNs 1-32")
        self.track_button = QtWidgets.QPushButton("Track Highlighted PRN")
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.scan_button)
        action_row.addWidget(self.track_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.task_status_label = QtWidgets.QLabel("Acquisition idle.")
        self.task_status_label.setWordWrap(True)
        layout.addWidget(self.task_status_label)

        self.task_progress_bar = QtWidgets.QProgressBar()
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setValue(0)
        layout.addWidget(self.task_progress_bar)

        self.summary_label = QtWidgets.QLabel("No acquisition result yet. Run one PRN or scan multiple PRNs.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        self.heatmap_plot = pg.PlotWidget(title="Code phase vs Doppler")
        self.heatmap_image = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_image)
        left_layout.addWidget(self.heatmap_plot, stretch=1)

        self.candidate_table = QtWidgets.QTableWidget(0, 5)
        self.candidate_table.setHorizontalHeaderLabels(["PRN", "Search freq [Hz]", "Rel. Doppler [Hz]", "Code phase", "Metric"])
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.candidate_table, stretch=1)
        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.addWidget(QtWidgets.QLabel("Best sample-rate hypotheses"))
        self.rate_table = QtWidgets.QTableWidget(0, 6)
        self.rate_table.setHorizontalHeaderLabels(["Sample rate [Sa/s]", "Best PRN", "Consistent segs", "Score", "Metric", "Search freq [Hz]"])
        self.rate_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.rate_table, stretch=1)

        right_layout.addWidget(QtWidgets.QLabel("Best IF / search-center hypotheses"))
        self.center_table = QtWidgets.QTableWidget(0, 5)
        self.center_table.setHorizontalHeaderLabels(["Center [Hz]", "Best PRN", "Metric", "Search freq [Hz]", "Rel. Doppler [Hz]"])
        self.center_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.center_table, stretch=1)

        right_layout.addWidget(QtWidgets.QLabel("Detected / inspected satellites"))
        self.satellite_table = QtWidgets.QTableWidget(0, 7)
        self.satellite_table.setHorizontalHeaderLabels(["PRN", "Metric", "Consistent segs", "Search freq [Hz]", "Rel. Doppler [Hz]", "Code phase", "Interpretation"])
        self.satellite_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.satellite_table, stretch=1)

        right_layout.addWidget(QtWidgets.QLabel("Why this looks like one satellite"))
        self.evidence_text = QtWidgets.QPlainTextEdit()
        self.evidence_text.setReadOnly(True)
        right_layout.addWidget(self.evidence_text, stretch=1)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, stretch=1)

        self.run_button.clicked.connect(self.run_requested.emit)
        self.scan_button.clicked.connect(self.scan_requested.emit)
        self.center_sweep_button.clicked.connect(self.sweep_requested.emit)
        self.auto_detect_button.clicked.connect(self.auto_rate_survey_requested.emit)
        self.track_button.clicked.connect(self.track_selected_requested.emit)
        self.satellite_table.itemSelectionChanged.connect(self._emit_selection_changed)
        self.center_table.itemSelectionChanged.connect(self._emit_sweep_selection_changed)
        self.rate_table.itemSelectionChanged.connect(self._emit_sample_rate_selection_changed)

    def _emit_selection_changed(self) -> None:
        items = self.satellite_table.selectedItems()
        if not items:
            return
        try:
            prn = int(self.satellite_table.item(items[0].row(), 0).text())
        except (TypeError, ValueError, AttributeError):
            return
        self.selection_changed.emit(prn)

    def set_task_message(self, message: str) -> None:
        """Show the current acquisition-side job message."""

        self.task_status_label.setText(message)

    def set_task_progress(self, value: int) -> None:
        """Show acquisition-side worker progress."""

        self.task_progress_bar.setValue(max(0, min(100, int(value))))

    def update_result(self, result: AcquisitionResult, total_results: list[AcquisitionResult] | None = None) -> None:
        """Populate the heatmap and candidate table."""

        self.summary_label.setText(
            f"Best peak: PRN {result.best_candidate.prn}, "
            f"search frequency {result.best_candidate.carrier_frequency_hz:.1f} Hz, "
            f"relative Doppler {result.best_candidate.doppler_hz:+.1f} Hz, "
            f"code phase {result.best_candidate.code_phase_samples} samples, "
            f"segment start {result.best_candidate.segment_start_sample / max(result.sample_rate_hz, 1.0):.3f} s, "
            f"metric {result.best_candidate.metric:.2f}, "
            f"consistent segments {result.consistent_segments}, "
            f"interpretation {acquisition_interpretation(result)}. "
            "A strong, repeated peak across segments is more trustworthy than one isolated hit."
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
            self.candidate_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{candidate.carrier_frequency_hz:.1f}"))
            self.candidate_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{candidate.doppler_hz:+.1f}"))
            self.candidate_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(candidate.code_phase_samples)))
            self.candidate_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{candidate.metric:.2f}"))

        results = total_results or [result]
        self.satellite_table.setRowCount(len(results))
        for row, sat_result in enumerate(results):
            candidate = sat_result.best_candidate
            interpretation = acquisition_interpretation(sat_result)
            self.satellite_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(candidate.prn)))
            self.satellite_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{candidate.metric:.2f}"))
            self.satellite_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(sat_result.consistent_segments)))
            self.satellite_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{candidate.carrier_frequency_hz:.1f}"))
            self.satellite_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{candidate.doppler_hz:+.1f}"))
            self.satellite_table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(candidate.code_phase_samples)))
            self.satellite_table.setItem(row, 6, QtWidgets.QTableWidgetItem(interpretation))

        self.evidence_text.setPlainText(
            "\n".join(
                [
                    f"Selected PRN: {result.prn}",
                    f"Sample-rate hypothesis: {result.sample_rate_hz / 1e6:.3f} MSa/s",
                    f"Search center / IF: {result.search_center_hz:.1f} Hz",
                    f"Best searched frequency: {result.best_candidate.carrier_frequency_hz:.1f} Hz",
                    f"Relative Doppler bin: {result.best_candidate.doppler_hz:+.1f} Hz",
                    f"Best code phase: {result.best_candidate.code_phase_samples} samples",
                    f"Peak metric: {result.best_candidate.metric:.2f}",
                    f"Consistent segment count: {result.consistent_segments}",
                    f"Consistency score: {result.consistency_score:.2f}",
                    f"Interpretation label: {acquisition_interpretation(result)}",
                    "",
                    "How to read this:",
                    "- The brighter the heatmap peak, the better the local PRN matches the signal.",
                    "- Search frequency is the actual tone removed in the sample domain.",
                    "- Relative Doppler is the offset around the chosen IF / search center.",
                    "- Code phase tells you where the PRN alignment happens within the 1 ms code.",
                    "- Repeated hits across different file segments are much more convincing than one isolated maximum.",
                    "- Repetition alone is not enough when the raw metric stays weak across every segment.",
                    "",
                    "Segment evidence:",
                ]
                + [
                    f"- t={candidate.segment_start_sample / max(result.sample_rate_hz, 1.0):.3f} s, "
                    f"freq={candidate.carrier_frequency_hz:.1f} Hz, "
                    f"doppler={candidate.doppler_hz:+.1f} Hz, "
                    f"code={candidate.code_phase_samples}, metric={candidate.metric:.2f}"
                    for candidate in sorted(result.segment_candidates, key=lambda item: item.metric, reverse=True)[:8]
                ]
            )
        )

        if results:
            self.satellite_table.selectRow(next((i for i, item in enumerate(results) if item.prn == result.prn), 0))

    def update_sweep_results(self, sweep_entries) -> None:
        """Show ranked IF / search-center hypotheses."""

        self.center_table.setRowCount(len(sweep_entries))
        for row, entry in enumerate(sweep_entries):
            best = entry.best_result.best_candidate
            self.center_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{entry.search_center_hz:.1f}"))
            self.center_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(entry.best_result.prn)))
            self.center_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{best.metric:.2f}"))
            self.center_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{best.carrier_frequency_hz:.1f}"))
            self.center_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{best.doppler_hz:+.1f}"))
        if sweep_entries:
            self.center_table.selectRow(0)

    def update_sample_rate_survey(self, survey_entries) -> None:
        """Show ranked sample-rate hypotheses."""

        self.rate_table.setRowCount(len(survey_entries))
        for row, entry in enumerate(survey_entries):
            best = entry.best_result.best_candidate
            self.rate_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{entry.sample_rate_hz:.1f}"))
            self.rate_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(entry.best_result.prn)))
            self.rate_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(entry.best_result.consistent_segments)))
            self.rate_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{entry.best_result.consistency_score:.2f}"))
            self.rate_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{best.metric:.2f}"))
            self.rate_table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{best.carrier_frequency_hz:.1f}"))
        if survey_entries:
            self.rate_table.selectRow(0)

    def build_search_centers(self) -> list[float]:
        """Return the configured sweep centers."""

        start = int(self.center_min_spin.value())
        stop = int(self.center_max_spin.value())
        step = max(1, int(self.center_step_spin.value()))
        if stop < start:
            start, stop = stop, start
        return [float(value) for value in range(start, stop + step, step)]

    def _emit_sweep_selection_changed(self) -> None:
        items = self.center_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        try:
            center_hz = float(self.center_table.item(row, 0).text())
            prn = int(self.center_table.item(row, 1).text())
        except (TypeError, ValueError, AttributeError):
            return
        self.sweep_selection_changed.emit(center_hz, prn)

    def _emit_sample_rate_selection_changed(self) -> None:
        items = self.rate_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        try:
            sample_rate_hz = float(self.rate_table.item(row, 0).text())
            prn = int(self.rate_table.item(row, 1).text())
        except (TypeError, ValueError, AttributeError):
            return
        self.sample_rate_selection_changed.emit(sample_rate_hz, prn)
