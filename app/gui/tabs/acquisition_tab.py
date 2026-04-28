"""Acquisition result visualization tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from app.dsp.acquisition import STRONG_ACQUISITION_METRIC_THRESHOLD
from app.dsp.acquisition import acquisition_interpretation
from app.dsp.acquisition import acquisition_result_is_plausible
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
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        root_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_splitter.setChildrenCollapsible(False)
        layout.addWidget(root_splitter)

        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        sidebar_scroll.setMinimumWidth(300)
        sidebar_scroll.setMaximumWidth(430)
        sidebar = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 8, 0)
        sidebar_layout.setSpacing(8)
        sidebar_scroll.setWidget(sidebar)

        help_label = QtWidgets.QLabel(f"{TOOLTIPS['ca_code']} {TOOLTIPS['despreading']}")
        help_label.setWordWrap(True)
        sidebar_layout.addWidget(help_label)

        setup_group = QtWidgets.QGroupBox("Search setup")
        controls = QtWidgets.QFormLayout(setup_group)
        controls.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.prn_spin = QtWidgets.QSpinBox()
        self.prn_spin.setRange(1, 32)
        self.prn_spin.setValue(1)
        self.scan_prns_edit = QtWidgets.QLineEdit("1-32")
        self.scan_prns_edit.setToolTip(
            "PRNs for the scan. Examples: 1-32, 1,3,8-10, or 32."
        )
        self.doppler_min_spin = QtWidgets.QSpinBox()
        self.doppler_min_spin.setRange(-20_000, 0)
        self.doppler_min_spin.setValue(-6_000)
        self.doppler_min_spin.setToolTip(TOOLTIPS["doppler_bin"])
        self.doppler_max_spin = QtWidgets.QSpinBox()
        self.doppler_max_spin.setRange(0, 20_000)
        self.doppler_max_spin.setValue(6_000)
        self.doppler_max_spin.setToolTip(TOOLTIPS["doppler_bin"])
        self.doppler_step_spin = QtWidgets.QSpinBox()
        self.doppler_step_spin.setRange(50, 2_000)
        self.doppler_step_spin.setValue(500)
        self.doppler_step_spin.setToolTip("Spacing between Doppler hypotheses. Smaller steps cost more but can reveal narrower peaks.")
        self.integration_spin = QtWidgets.QSpinBox()
        self.integration_spin.setRange(1, 500)
        self.integration_spin.setValue(20)
        self.integration_spin.setSuffix(" ms blocks")
        self.integration_spin.setToolTip(TOOLTIPS["integration_1ms"])
        self.segment_count_spin = QtWidgets.QSpinBox()
        self.segment_count_spin.setRange(1, 32)
        self.segment_count_spin.setValue(4)
        self.segment_count_spin.setSuffix(" segments")
        self.spread_blocks_checkbox = QtWidgets.QCheckBox("Spread 1 ms blocks across loaded source")
        self.spread_blocks_checkbox.setChecked(False)
        controls.addRow("PRN", self.prn_spin)
        controls.addRow("Scan PRNs", self.scan_prns_edit)
        controls.addRow("Doppler min [Hz]", self.doppler_min_spin)
        controls.addRow("Doppler max [Hz]", self.doppler_max_spin)
        controls.addRow("Doppler step [Hz]", self.doppler_step_spin)
        controls.addRow("Accumulation", self.integration_spin)
        controls.addRow("Deep search", self.segment_count_spin)
        controls.addRow("Weak-signal mode", self.spread_blocks_checkbox)
        sidebar_layout.addWidget(setup_group)

        sweep_group = QtWidgets.QGroupBox("IF / sample-rate search")
        sweep_layout = QtWidgets.QFormLayout(sweep_group)
        sweep_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
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
        sidebar_layout.addWidget(sweep_group)

        action_group = QtWidgets.QGroupBox("Run")
        action_layout = QtWidgets.QVBoxLayout(action_group)
        self.run_button = QtWidgets.QPushButton("Run Acquisition For PRN")
        self.scan_button = QtWidgets.QPushButton("Scan PRN List")
        self.track_button = QtWidgets.QPushButton("Track Highlighted PRN")
        action_layout.addWidget(self.run_button)
        action_layout.addWidget(self.scan_button)
        action_layout.addWidget(self.track_button)
        sidebar_layout.addWidget(action_group)

        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        self.task_status_label = QtWidgets.QLabel("Acquisition idle.")
        self.task_status_label.setWordWrap(True)
        self.task_progress_bar = QtWidgets.QProgressBar()
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setValue(0)
        status_layout.addWidget(self.task_status_label)
        status_layout.addWidget(self.task_progress_bar)
        sidebar_layout.addWidget(status_group)

        guide_group = QtWidgets.QGroupBox("Quick guide")
        guide_layout = QtWidgets.QVBoxLayout(guide_group)
        self.stage_hint_label = QtWidgets.QLabel(
            "Acquisition tries one PRN code at many Doppler bins and code phases. "
            "A bright heatmap peak means this PRN, this Doppler, and this code timing match the recording best."
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

        self.summary_label = QtWidgets.QLabel("No acquisition result yet. Run one PRN or scan multiple PRNs.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setMaximumHeight(72)
        workspace_layout.addWidget(self.summary_label)

        self.selected_prn_label = QtWidgets.QLabel("Selected PRN: none")
        self.selected_prn_label.setWordWrap(True)
        self.selected_prn_label.setMaximumHeight(56)
        workspace_layout.addWidget(self.selected_prn_label)

        workspace_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        workspace_splitter.setChildrenCollapsible(False)
        workspace_layout.addWidget(workspace_splitter, stretch=1)

        self.heatmap_plot = pg.PlotWidget(title="PRN / code phase vs Doppler")
        self.heatmap_plot.setMinimumHeight(360)
        self.heatmap_plot.setLabel("bottom", "Relative Doppler", units="Hz")
        self.heatmap_plot.setLabel("left", "Code phase", units="samples")
        self.heatmap_image = pg.ImageItem(axisOrder="row-major")
        self.heatmap_plot.addItem(self.heatmap_image)
        self.peak_scatter = pg.ScatterPlotItem()
        self.heatmap_plot.addItem(self.peak_scatter)
        self.selected_prn_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#ffd43b", width=2),
        )
        self.selected_prn_line.setVisible(False)
        self.heatmap_plot.addItem(self.selected_prn_line)
        workspace_splitter.addWidget(self.heatmap_plot)

        self.detail_tabs = QtWidgets.QTabWidget()
        self.detail_tabs.setDocumentMode(True)
        workspace_splitter.addWidget(self.detail_tabs)

        self.satellite_page = QtWidgets.QWidget()
        satellite_layout = QtWidgets.QVBoxLayout(self.satellite_page)
        self.satellite_hint_label = QtWidgets.QLabel(
            "Click a PRN row to inspect that satellite candidate. Then use Track Highlighted PRN to follow its code and carrier over time."
        )
        self.satellite_hint_label.setWordWrap(True)
        satellite_layout.addWidget(self.satellite_hint_label)
        self.satellite_table = QtWidgets.QTableWidget(0, 9)
        self._configure_table(self.satellite_table, minimum_height=260)
        self.satellite_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.satellite_table.setHorizontalHeaderLabels(
            [
                "PRN",
                "Metric",
                "Consistent segs",
                "Search freq [Hz]",
                "Rel. Doppler [Hz]",
                "Code phase",
                "Tracking",
                "Navigation",
                "Interpretation",
            ]
        )
        self.satellite_table.horizontalHeader().setStretchLastSection(True)
        satellite_layout.addWidget(self.satellite_table, stretch=1)
        self.detail_tabs.addTab(self.satellite_page, "Satellites")

        slice_page = QtWidgets.QWidget()
        slice_layout = QtWidgets.QHBoxLayout(slice_page)
        self.codephase_slice_plot = pg.PlotWidget(title="Codephase slice at best Doppler")
        self.codephase_slice_plot.setMinimumHeight(240)
        self.codephase_slice_plot.setToolTip(
            "Shows correlation over codephase while Doppler is fixed at the selected peak."
        )
        self.codephase_slice_plot.setLabel("bottom", "Code phase", units="samples")
        self.codephase_slice_plot.setLabel("left", "Correlation")
        self.codephase_slice_curve = self.codephase_slice_plot.plot(pen="y")
        self.codephase_slice_peak = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#111111"), brush=pg.mkBrush("#ffd43b"))
        self.codephase_slice_plot.addItem(self.codephase_slice_peak)
        self.doppler_slice_plot = pg.PlotWidget(title="Doppler slice at best codephase")
        self.doppler_slice_plot.setMinimumHeight(240)
        self.doppler_slice_plot.setToolTip(
            "Shows correlation over Doppler while codephase is fixed at the selected peak."
        )
        self.doppler_slice_plot.setLabel("bottom", "Relative Doppler", units="Hz")
        self.doppler_slice_plot.setLabel("left", "Correlation")
        self.doppler_slice_curve = self.doppler_slice_plot.plot(pen="c")
        self.doppler_slice_peak = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#111111"), brush=pg.mkBrush("#ffd43b"))
        self.doppler_slice_plot.addItem(self.doppler_slice_peak)
        slice_layout.addWidget(self.codephase_slice_plot)
        slice_layout.addWidget(self.doppler_slice_plot)
        self.detail_tabs.addTab(slice_page, "Peak Slices")

        candidates_page = QtWidgets.QWidget()
        candidates_layout = QtWidgets.QVBoxLayout(candidates_page)
        self.peak_hint_label = QtWidgets.QLabel(
            "When multiple PRNs were scanned, the heatmap has exactly one row per scanned PRN. "
            "Brightness is the best code-phase match at that Doppler; weak best-noise peaks are darkened. "
            "Yellow marks the selected PRN, green marks plausible repeated candidates."
        )
        self.peak_hint_label.setWordWrap(True)
        candidates_layout.addWidget(self.peak_hint_label)
        self.candidate_table = QtWidgets.QTableWidget(0, 5)
        self._configure_table(self.candidate_table, minimum_height=220)
        self.candidate_table.setHorizontalHeaderLabels(["PRN", "Search freq [Hz]", "Rel. Doppler [Hz]", "Code phase", "Metric"])
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        candidates_layout.addWidget(self.candidate_table, stretch=1)
        self.detail_tabs.addTab(candidates_page, "Selected PRN Peaks")

        self.survey_page = QtWidgets.QWidget()
        survey_layout = QtWidgets.QHBoxLayout(self.survey_page)
        rate_group = QtWidgets.QGroupBox("Best sample-rate hypotheses")
        rate_layout = QtWidgets.QVBoxLayout(rate_group)
        self.rate_table = QtWidgets.QTableWidget(0, 6)
        self._configure_table(self.rate_table, minimum_height=220)
        self.rate_table.setHorizontalHeaderLabels(["Sample rate [Sa/s]", "Best PRN", "Consistent segs", "Score", "Metric", "Search freq [Hz]"])
        self.rate_table.horizontalHeader().setStretchLastSection(True)
        rate_layout.addWidget(self.rate_table)
        center_group = QtWidgets.QGroupBox("Best IF / search-center hypotheses")
        center_layout = QtWidgets.QVBoxLayout(center_group)
        self.center_table = QtWidgets.QTableWidget(0, 5)
        self._configure_table(self.center_table, minimum_height=220)
        self.center_table.setHorizontalHeaderLabels(["Center [Hz]", "Best PRN", "Metric", "Search freq [Hz]", "Rel. Doppler [Hz]"])
        self.center_table.horizontalHeader().setStretchLastSection(True)
        center_layout.addWidget(self.center_table)
        survey_layout.addWidget(rate_group)
        survey_layout.addWidget(center_group)
        self.detail_tabs.addTab(self.survey_page, "Sample / IF Survey")

        evidence_page = QtWidgets.QWidget()
        evidence_layout = QtWidgets.QVBoxLayout(evidence_page)
        self.evidence_text = QtWidgets.QPlainTextEdit()
        self.evidence_text.setReadOnly(True)
        evidence_layout.addWidget(self.evidence_text)
        self.detail_tabs.addTab(evidence_page, "Evidence")

        guide_page = QtWidgets.QWidget()
        what_layout = QtWidgets.QVBoxLayout(guide_page)
        self.what_label = QtWidgets.QLabel(
            "This view searches one PRN over Doppler and code phase. "
            "For each Doppler bin, the assumed carrier rotation is removed; then the local PRN code is shifted and correlated. "
            "A bright peak means this PRN, this frequency hypothesis, and this 1 ms code timing fit best."
        )
        self.what_label.setWordWrap(True)
        what_layout.addWidget(self.what_label)
        self.axis_help_label = QtWidgets.QLabel(
            "X axis: Doppler / search frequency hypothesis. Y axis: codephase / sample offset inside the repeated 1 ms C/A code. "
            "Brightness: correlation energy. Doppler bin is not a satellite ID; satellites are primarily separated by PRN code. "
            "Codephase is not IQ phase; correct Doppler stops rotation but does not force codephase to 0."
        )
        self.axis_help_label.setToolTip(
            f"{TOOLTIPS['code_phase']} {TOOLTIPS['doppler_bin']} {TOOLTIPS['iq_phase']}"
        )
        self.axis_help_label.setWordWrap(True)
        what_layout.addWidget(self.axis_help_label)
        what_layout.addStretch()
        self.detail_tabs.addTab(guide_page, "Guide")

        workspace_splitter.setStretchFactor(0, 3)
        workspace_splitter.setStretchFactor(1, 2)
        workspace_splitter.setSizes([560, 360])

        root_splitter.addWidget(workspace)
        root_splitter.setStretchFactor(0, 0)
        root_splitter.setStretchFactor(1, 1)
        root_splitter.setSizes([340, 1060])

        self.run_button.clicked.connect(self.run_requested.emit)
        self.scan_button.clicked.connect(self.scan_requested.emit)
        self.center_sweep_button.clicked.connect(self.sweep_requested.emit)
        self.auto_detect_button.clicked.connect(self.auto_rate_survey_requested.emit)
        self.track_button.clicked.connect(self.track_selected_requested.emit)
        self.satellite_table.itemSelectionChanged.connect(self._emit_selection_changed)
        self.satellite_table.cellDoubleClicked.connect(lambda *_: self.track_selected_requested.emit())
        self.center_table.itemSelectionChanged.connect(self._emit_sweep_selection_changed)
        self.rate_table.itemSelectionChanged.connect(self._emit_sample_rate_selection_changed)

    @staticmethod
    def _configure_table(table: QtWidgets.QTableWidget, minimum_height: int) -> None:
        """Make dense evidence tables readable and horizontally scrollable."""

        table.setMinimumHeight(minimum_height)
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
    def parse_prn_list(text: str) -> list[int]:
        """Parse comma-separated PRNs and ranges such as 1,3,8-10."""

        values: set[int] = set()
        for token in text.replace(";", ",").split(","):
            part = token.strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                start = int(left.strip())
                stop = int(right.strip())
                if stop < start:
                    start, stop = stop, start
                values.update(range(start, stop + 1))
            else:
                values.add(int(part))
        if not values:
            raise ValueError("Enter at least one PRN, for example 1-32 or 1,3,8.")
        invalid = sorted(value for value in values if value < 1 or value > 32)
        if invalid:
            raise ValueError(f"PRNs must be in the GPS C/A range 1..32; invalid: {invalid}.")
        return sorted(values)

    def selected_scan_prns(self) -> list[int]:
        """Return the user-selected PRNs for a scan."""

        return self.parse_prn_list(self.scan_prns_edit.text())

    @staticmethod
    def build_prn_doppler_overview(
        results: list[AcquisitionResult],
    ) -> tuple[list[AcquisitionResult], np.ndarray, np.ndarray]:
        """Collapse per-PRN code-phase heatmaps into one PRN-vs-Doppler overview."""

        ordered_results = sorted(results, key=lambda item: item.prn)
        if not ordered_results:
            return [], np.empty(0, dtype=np.float32), np.empty((0, 0), dtype=np.float32)

        doppler_values = sorted(
            {
                round(float(doppler), 6)
                for result in ordered_results
                for doppler in result.doppler_bins_hz
            }
        )
        doppler_bins = np.asarray(doppler_values, dtype=np.float32)
        overview = np.zeros((len(ordered_results), doppler_bins.size), dtype=np.float32)
        doppler_index = {value: index for index, value in enumerate(doppler_values)}

        for row, result in enumerate(ordered_results):
            heatmap = np.asarray(result.heatmap, dtype=np.float32)
            if heatmap.ndim != 2 or heatmap.size == 0 or result.doppler_bins_hz.size == 0:
                continue
            row_count = min(heatmap.shape[0], result.doppler_bins_hz.size)
            if row_count <= 0:
                continue
            usable_heatmap = heatmap[:row_count]
            noise_floor = float(np.mean(usable_heatmap) + 1e-12)
            doppler_profile = np.max(usable_heatmap, axis=1) / noise_floor
            for doppler, metric in zip(result.doppler_bins_hz[:row_count], doppler_profile):
                overview[row, doppler_index[round(float(doppler), 6)]] = float(metric)

        return ordered_results, doppler_bins, overview

    @staticmethod
    def threshold_prn_doppler_overview(overview: np.ndarray) -> np.ndarray:
        """Darken bins below the strong acquisition threshold for the overview plot."""

        visible = np.asarray(overview, dtype=np.float32).copy()
        visible[visible < STRONG_ACQUISITION_METRIC_THRESHOLD] = 0.0
        return visible

    @staticmethod
    def sparse_prn_axis_ticks(
        ordered_results: list[AcquisitionResult],
        selected_prn: int,
    ) -> list[tuple[float, str]]:
        """Return readable PRN ticks without labeling all 32 rows on top of each other."""

        if not ordered_results:
            return []
        row_count = len(ordered_results)
        if row_count <= 16:
            rows = set(range(row_count))
        else:
            step = max(1, int(np.ceil(row_count / 8.0)))
            rows = set(range(0, row_count, step))
            rows.add(row_count - 1)
        selected_row = next(
            (row for row, item in enumerate(ordered_results) if item.prn == selected_prn),
            None,
        )
        if selected_row is not None:
            rows.add(selected_row)
        return [(row + 0.5, str(ordered_results[row].prn)) for row in sorted(rows)]

    @staticmethod
    def overview_marker_rows(
        ordered_results: list[AcquisitionResult],
        selected_prn: int,
    ) -> list[int]:
        """Return row indices that should get candidate markers in the overview plot."""

        return [
            row
            for row, result in enumerate(ordered_results)
            if result.prn == selected_prn or acquisition_result_is_plausible(result)
        ]

    @staticmethod
    def best_heatmap_indices(result: AcquisitionResult) -> tuple[int, int]:
        """Return heatmap row/column for the selected best candidate."""

        heatmap = np.asarray(result.heatmap)
        if heatmap.ndim != 2 or heatmap.size == 0:
            return 0, 0
        row_count, column_count = heatmap.shape
        if row_count <= 0 or column_count <= 0:
            return 0, 0
        if result.doppler_bins_hz.size:
            row = int(np.argmin(np.abs(result.doppler_bins_hz[:row_count] - result.best_candidate.doppler_hz)))
        else:
            row = 0
        column = int((-int(result.best_candidate.code_phase_samples)) % column_count)
        return max(0, min(row, row_count - 1)), max(0, min(column, column_count - 1))

    @classmethod
    def codephase_slice(
        cls,
        result: AcquisitionResult,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Return correlation over codephase at the best Doppler row."""

        heatmap = np.asarray(result.heatmap, dtype=np.float32)
        if heatmap.ndim != 2 or heatmap.size == 0:
            return np.empty(0), np.empty(0), 0.0, 0.0
        row, column = cls.best_heatmap_indices(result)
        column_count = heatmap.shape[1]
        codephase_axis = (column_count - np.arange(column_count, dtype=np.int32)) % column_count
        order = np.argsort(codephase_axis)
        values = heatmap[row, order]
        peak_phase = float(result.best_candidate.code_phase_samples % column_count)
        peak_value = float(heatmap[row, column])
        return codephase_axis[order].astype(np.float32), values.astype(np.float32), peak_phase, peak_value

    @classmethod
    def doppler_slice(
        cls,
        result: AcquisitionResult,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Return correlation over Doppler at the best codephase column."""

        heatmap = np.asarray(result.heatmap, dtype=np.float32)
        if heatmap.ndim != 2 or heatmap.size == 0:
            return np.empty(0), np.empty(0), 0.0, 0.0
        row, column = cls.best_heatmap_indices(result)
        row_count = min(heatmap.shape[0], result.doppler_bins_hz.size)
        doppler_axis = result.doppler_bins_hz[:row_count].astype(np.float32)
        values = heatmap[:row_count, column].astype(np.float32)
        peak_doppler = float(result.doppler_bins_hz[row]) if result.doppler_bins_hz.size else 0.0
        peak_value = float(heatmap[row, column])
        return doppler_axis, values, peak_doppler, peak_value

    def clear_result_view(self) -> None:
        """Clear acquisition plots and tables when the source context changes."""

        self.heatmap_plot.setTitle("PRN / code phase vs Doppler")
        self.heatmap_plot.setLabel("bottom", "Relative Doppler", units="Hz")
        self.heatmap_plot.setLabel("left", "Code phase", units="samples")
        self.heatmap_plot.getAxis("left").setTicks(None)
        self.heatmap_image.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=False)
        self.peak_scatter.setData(spots=[])
        self.selected_prn_line.setVisible(False)
        self.codephase_slice_curve.setData([], [])
        self.codephase_slice_peak.setData([], [])
        self.doppler_slice_curve.setData([], [])
        self.doppler_slice_peak.setData([], [])

    def _update_slice_plots(self, result: AcquisitionResult) -> None:
        """Refresh codephase and Doppler cuts through the selected acquisition peak."""

        code_axis, code_values, peak_phase, peak_value = self.codephase_slice(result)
        self.codephase_slice_curve.setData(code_axis, code_values)
        self.codephase_slice_peak.setData([peak_phase], [peak_value])
        doppler_axis, doppler_values, peak_doppler, doppler_peak_value = self.doppler_slice(result)
        self.doppler_slice_curve.setData(doppler_axis, doppler_values)
        self.doppler_slice_peak.setData([peak_doppler], [doppler_peak_value])

    def _update_single_prn_heatmap(self, result: AcquisitionResult) -> None:
        """Show the full code-phase search surface for one selected PRN."""

        self.heatmap_plot.setTitle(
            f"PRN {result.prn} acquisition heatmap: code phase vs relative Doppler"
        )
        self.heatmap_plot.setLabel("bottom", "Relative Doppler", units="Hz")
        self.heatmap_plot.setLabel("left", "Code phase", units="samples")
        self.heatmap_plot.getAxis("left").setTicks(None)
        self.heatmap_image.setImage(result.heatmap.T, autoLevels=True)
        rect = pg.QtCore.QRectF(
            float(result.doppler_bins_hz[0]),
            float(result.code_phases_samples[0]),
            float(result.doppler_bins_hz[-1] - result.doppler_bins_hz[0]) if result.doppler_bins_hz.size > 1 else 1.0,
            float(result.code_phases_samples[-1] - result.code_phases_samples[0]) if result.code_phases_samples.size > 1 else 1.0,
        )
        self.heatmap_image.setRect(rect)
        self.peak_scatter.setData(
            spots=[
                {
                    "pos": (result.best_candidate.doppler_hz, result.best_candidate.code_phase_samples),
                    "size": 10,
                    "pen": pg.mkPen("#111111", width=1),
                    "brush": pg.mkBrush("#ffd43b"),
                }
            ]
        )
        self.selected_prn_line.setVisible(False)

    def _update_prn_doppler_overview(
        self,
        selected_result: AcquisitionResult,
        results: list[AcquisitionResult],
    ) -> None:
        """Show one overview heatmap for all scanned PRNs."""

        ordered_results, doppler_bins, overview = self.build_prn_doppler_overview(results)
        plausible_count = sum(1 for item in ordered_results if acquisition_result_is_plausible(item))
        self.heatmap_plot.setTitle(
            f"PRN/Doppler overview: {len(ordered_results)} scanned PRN rows, {plausible_count} plausible candidates"
        )
        self.heatmap_plot.setLabel("bottom", "Relative Doppler", units="Hz")
        self.heatmap_plot.setLabel("left", "Scanned PRN")
        self.heatmap_plot.getAxis("left").setTicks(
            [self.sparse_prn_axis_ticks(ordered_results, selected_result.prn)]
        )

        if overview.size == 0 or doppler_bins.size == 0:
            self.heatmap_image.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=False)
            self.peak_scatter.setData(spots=[])
            self.selected_prn_line.setVisible(False)
            return

        self.heatmap_image.setImage(self.threshold_prn_doppler_overview(overview), autoLevels=True)
        doppler_width = float(doppler_bins[-1] - doppler_bins[0]) if doppler_bins.size > 1 else 1.0
        self.heatmap_image.setRect(
            pg.QtCore.QRectF(
                float(doppler_bins[0]),
                0.0,
                doppler_width,
                float(len(ordered_results)),
            )
        )

        selected_row = next(
            (row for row, item in enumerate(ordered_results) if item.prn == selected_result.prn),
            None,
        )
        marker_rows = set(self.overview_marker_rows(ordered_results, selected_result.prn))
        spots = []
        for row, item in enumerate(ordered_results):
            if row not in marker_rows:
                continue
            is_selected = item.prn == selected_result.prn
            spots.append(
                {
                    "pos": (item.best_candidate.doppler_hz, row + 0.5),
                    "size": 12 if is_selected else 8,
                    "pen": pg.mkPen("#111111", width=1),
                    "brush": pg.mkBrush("#ffd43b" if is_selected else "#51cf66"),
                }
            )
        self.peak_scatter.setData(spots=spots)
        if selected_row is None:
            self.selected_prn_line.setVisible(False)
        else:
            self.selected_prn_line.setValue(selected_row + 0.5)
            self.selected_prn_line.setVisible(True)
        self.heatmap_plot.setYRange(0.0, float(len(ordered_results)), padding=0.02)
        if doppler_bins.size > 1:
            self.heatmap_plot.setXRange(float(doppler_bins[0]), float(doppler_bins[-1]), padding=0.02)

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

    def update_result(
        self,
        result: AcquisitionResult,
        total_results: list[AcquisitionResult] | None = None,
        tracked_prns: set[int] | None = None,
        decoded_prns: set[int] | None = None,
    ) -> None:
        """Populate the heatmap and candidate table."""

        tracked_prns = tracked_prns or set()
        decoded_prns = decoded_prns or set()
        results = total_results or [result]
        if len(results) > 1:
            self.selected_prn_label.setText(
                f"Selected PRN {result.prn}: yellow marks this candidate; green marks other plausible repeated candidates."
            )
        else:
            self.selected_prn_label.setText(
                f"Selected PRN {result.prn}: heatmap, peaks, slices, and evidence all belong to this candidate."
            )
        overview_note = (
            f" Overview heatmap compares {len(results)} scanned PRN rows."
            if len(results) > 1
            else ""
        )
        self.summary_label.setText(
            f"Best peak: PRN {result.best_candidate.prn}, "
            f"freq {result.best_candidate.carrier_frequency_hz:.1f} Hz, "
            f"Doppler {result.best_candidate.doppler_hz:+.1f} Hz, "
            f"code {result.best_candidate.code_phase_samples}, "
            f"metric {result.best_candidate.metric:.2f}, "
            f"consistent segments {result.consistent_segments}, "
            f"interpretation {acquisition_interpretation(result)}."
            f"{overview_note} "
            "Confirm with tracking and navigation before treating it as a satellite."
        )
        if len(results) > 1:
            self._update_prn_doppler_overview(result, results)
        else:
            self._update_single_prn_heatmap(result)
        self._update_slice_plots(result)
        self.candidate_table.setRowCount(len(result.candidates))
        for row, candidate in enumerate(result.candidates):
            self.candidate_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(candidate.prn)))
            self.candidate_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{candidate.carrier_frequency_hz:.1f}"))
            self.candidate_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{candidate.doppler_hz:+.1f}"))
            self.candidate_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(candidate.code_phase_samples)))
            self.candidate_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{candidate.metric:.2f}"))
        self.candidate_table.resizeColumnsToContents()
        self.candidate_table.horizontalHeader().setStretchLastSection(True)

        self.satellite_table.blockSignals(True)
        self.satellite_table.setRowCount(len(results))
        for row, sat_result in enumerate(results):
            candidate = sat_result.best_candidate
            interpretation = acquisition_interpretation(sat_result)
            tracking_status = "tracked" if sat_result.prn in tracked_prns else "-"
            navigation_status = "decoded" if sat_result.prn in decoded_prns else "-"
            self.satellite_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(candidate.prn)))
            self.satellite_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{candidate.metric:.2f}"))
            self.satellite_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(sat_result.consistent_segments)))
            self.satellite_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{candidate.carrier_frequency_hz:.1f}"))
            self.satellite_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{candidate.doppler_hz:+.1f}"))
            self.satellite_table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(candidate.code_phase_samples)))
            self.satellite_table.setItem(row, 6, QtWidgets.QTableWidgetItem(tracking_status))
            self.satellite_table.setItem(row, 7, QtWidgets.QTableWidgetItem(navigation_status))
            self.satellite_table.setItem(row, 8, QtWidgets.QTableWidgetItem(interpretation))

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
                    "- In the all-PRN overview, each row is one scanned PRN; the plot does not create more rows than the scan list.",
                    f"- Bins below the strong metric threshold ({STRONG_ACQUISITION_METRIC_THRESHOLD:.1f}) are darkened so noise maxima do not look like targets.",
                    "- Yellow marks the selected PRN. Green marks other repeated, strong acquisition candidates.",
                    "- Best peaks from weak rows are not drawn as target markers.",
                    "- The local peak table gives the actual code-phase alternatives for the selected PRN.",
                    "- The brighter the heatmap peak, the better the local PRN matches the signal.",
                    "- Search frequency is the actual tone removed in the sample domain.",
                    "- Relative Doppler is the offset around the chosen IF / search center.",
                    "- Code phase tells you where the PRN alignment happens within the 1 ms code.",
                    "- The codephase slice fixes Doppler and shows why the best codephase is a timing offset.",
                    "- The Doppler slice fixes codephase and shows why carrier frequency and PRN timing must be searched together.",
                    "- Repeated hits across different file segments are more convincing than one isolated maximum.",
                    "- Repetition alone is not enough when the raw metric stays weak across every segment.",
                    "- Acquisition alone does not prove that a real satellite is present; verify the highlighted candidate with tracking and navigation decoding.",
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
        self.satellite_table.blockSignals(False)
        self.satellite_table.resizeColumnsToContents()
        self.satellite_table.horizontalHeader().setStretchLastSection(True)
        if len(results) > 1:
            self.detail_tabs.setCurrentWidget(self.satellite_page)

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
        self.center_table.resizeColumnsToContents()
        self.center_table.horizontalHeader().setStretchLastSection(True)
        if sweep_entries:
            self.center_table.selectRow(0)
            self.detail_tabs.setCurrentWidget(self.survey_page)

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
        self.rate_table.resizeColumnsToContents()
        self.rate_table.horizontalHeader().setStretchLastSection(True)
        if survey_entries:
            self.rate_table.selectRow(0)
            self.detail_tabs.setCurrentWidget(self.survey_page)

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
