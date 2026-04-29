"""PVT / time solution tab."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from app.dsp.pvt import PVTComputationResult


class PVTTab(QtWidgets.QWidget):
    """Show how decoded LNAV evidence becomes receiver time and position."""

    solve_requested = QtCore.Signal()
    pipeline_requested = QtCore.Signal()

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

        solve_group = QtWidgets.QGroupBox("PVT solve")
        solve_layout = QtWidgets.QVBoxLayout(solve_group)
        self.solve_button = QtWidgets.QPushButton("Solve From Decoded PRNs")
        self.pipeline_button = QtWidgets.QPushButton("Run Auto PVT Pipeline")
        solve_layout.addWidget(self.solve_button)
        solve_layout.addWidget(self.pipeline_button)
        sidebar_layout.addWidget(solve_group)

        pipeline_group = QtWidgets.QGroupBox("Auto pipeline window")
        pipeline_layout = QtWidgets.QFormLayout(pipeline_group)
        pipeline_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.start_time_spin = QtWidgets.QDoubleSpinBox()
        self.start_time_spin.setRange(0.0, 24 * 3600.0)
        self.start_time_spin.setDecimals(3)
        self.start_time_spin.setValue(60.0)
        self.start_time_spin.setSuffix(" s")
        self.acquisition_window_spin = QtWidgets.QDoubleSpinBox()
        self.acquisition_window_spin.setRange(1.0, 30.0)
        self.acquisition_window_spin.setDecimals(1)
        self.acquisition_window_spin.setValue(3.0)
        self.acquisition_window_spin.setSuffix(" s")
        self.tracking_seconds_spin = QtWidgets.QDoubleSpinBox()
        self.tracking_seconds_spin.setRange(20.0, 180.0)
        self.tracking_seconds_spin.setDecimals(1)
        self.tracking_seconds_spin.setValue(60.0)
        self.tracking_seconds_spin.setSuffix(" s")
        self.max_satellites_spin = QtWidgets.QSpinBox()
        self.max_satellites_spin.setRange(4, 12)
        self.max_satellites_spin.setValue(8)
        pipeline_layout.addRow("Start", self.start_time_spin)
        pipeline_layout.addRow("Acquire", self.acquisition_window_spin)
        pipeline_layout.addRow("Track", self.tracking_seconds_spin)
        pipeline_layout.addRow("Max PRNs", self.max_satellites_spin)
        sidebar_layout.addWidget(pipeline_group)

        self.task_status_label = QtWidgets.QLabel("PVT idle.")
        self.task_status_label.setWordWrap(True)
        self.task_progress_bar = QtWidgets.QProgressBar()
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setValue(0)
        self.summary_label = QtWidgets.QLabel("No PVT solution yet.")
        self.summary_label.setWordWrap(True)
        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        status_layout.addWidget(self.task_status_label)
        status_layout.addWidget(self.task_progress_bar)
        status_layout.addWidget(self.summary_label)
        sidebar_layout.addWidget(status_group)
        sidebar_layout.addStretch()
        root_splitter.addWidget(sidebar_scroll)

        workspace = QtWidgets.QWidget()
        workspace_layout = QtWidgets.QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(8)
        self.result_tabs = QtWidgets.QTabWidget()
        self.result_tabs.setDocumentMode(True)
        workspace_layout.addWidget(self.result_tabs)

        self.solution_text = QtWidgets.QPlainTextEdit()
        self.solution_text.setReadOnly(True)
        self.result_tabs.addTab(self.solution_text, "Solution")

        self.ephemeris_table = QtWidgets.QTableWidget(0, 8)
        self.ephemeris_table.setHorizontalHeaderLabels(
            ["PRN", "Week", "IODE", "Health", "Toe [s]", "Toc [s]", "sqrt(A)", "e"]
        )
        self._configure_table(self.ephemeris_table)
        self.result_tabs.addTab(self.ephemeris_table, "Ephemerides")

        self.observation_table = QtWidgets.QTableWidget(0, 8)
        self.observation_table.setHorizontalHeaderLabels(
            ["PRN", "SF", "TX TOW [s]", "RX file [s]", "Pseudorange [km]", "Sat clock [us]", "Start bit", "Bit ms"]
        )
        self._configure_table(self.observation_table)
        self.result_tabs.addTab(self.observation_table, "Pseudoranges")

        self.evidence_text = QtWidgets.QPlainTextEdit()
        self.evidence_text.setReadOnly(True)
        self.result_tabs.addTab(self.evidence_text, "Evidence")

        root_splitter.addWidget(workspace)
        root_splitter.setStretchFactor(0, 0)
        root_splitter.setStretchFactor(1, 1)
        root_splitter.setSizes([340, 1060])

        self.solve_button.clicked.connect(self.solve_requested.emit)
        self.pipeline_button.clicked.connect(self.pipeline_requested.emit)

    @staticmethod
    def _configure_table(table: QtWidgets.QTableWidget) -> None:
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setStretchLastSection(True)

    def pipeline_settings(self) -> dict[str, float | int]:
        """Return auto-pipeline settings from the controls."""

        return {
            "start_time_s": float(self.start_time_spin.value()),
            "acquisition_window_s": float(self.acquisition_window_spin.value()),
            "tracking_s": float(self.tracking_seconds_spin.value()),
            "max_satellites": int(self.max_satellites_spin.value()),
        }

    def set_task_message(self, message: str) -> None:
        self.task_status_label.setText(message)

    def set_task_progress(self, value: int) -> None:
        self.task_progress_bar.setValue(max(0, min(100, int(value))))

    def clear_result(self) -> None:
        self.summary_label.setText("No PVT solution yet.")
        self.solution_text.clear()
        self.ephemeris_table.setRowCount(0)
        self.observation_table.setRowCount(0)
        self.evidence_text.clear()

    def update_result(self, result: PVTComputationResult) -> None:
        """Refresh all PVT result views."""

        self._update_ephemeris_table(result)
        self._update_observation_table(result)
        evidence = list(result.summary_lines)
        if result.solution is None:
            self.summary_label.setText("PVT needs at least four PRNs with ephemeris and timing evidence.")
            self.solution_text.setPlainText("\n".join(evidence))
            self.evidence_text.setPlainText("\n".join(evidence))
            return

        solution = result.solution
        self.summary_label.setText(
            f"Position {solution.latitude_deg:.6f}, {solution.longitude_deg:.6f}, "
            f"alt {solution.altitude_m:.1f} m; {solution.used_satellites} PRNs."
        )
        lines = [
            f"Latitude: {solution.latitude_deg:.8f} deg",
            f"Longitude: {solution.longitude_deg:.8f} deg",
            f"Altitude: {solution.altitude_m:.1f} m",
            f"ECEF: X {solution.ecef_m[0]:.1f} m, Y {solution.ecef_m[1]:.1f} m, Z {solution.ecef_m[2]:.1f} m",
            f"Receiver clock bias: {solution.receiver_clock_bias_m:.1f} m",
            f"Used PRNs: {', '.join(str(obs.prn) for obs in result.observations)}",
            f"Residual RMS: {result.residual_rms_m:.1f} m" if result.residual_rms_m is not None else "Residual RMS: n/a",
            f"GPS week/TOW: {result.gps_week}, {result.gps_time_of_week_s:.3f} s"
            if result.gps_week is not None and result.gps_time_of_week_s is not None
            else "GPS week/TOW: n/a",
            f"UTC estimate: {result.utc_datetime.isoformat()}" if result.utc_datetime else "UTC estimate: n/a",
        ]
        self.solution_text.setPlainText("\n".join(line for line in lines if line))
        self.evidence_text.setPlainText("\n".join(evidence))
        self.result_tabs.setCurrentWidget(self.solution_text)

    def _update_ephemeris_table(self, result: PVTComputationResult) -> None:
        ephemerides = [result.ephemerides[prn] for prn in sorted(result.ephemerides)]
        self.ephemeris_table.setRowCount(len(ephemerides))
        for row, ephemeris in enumerate(ephemerides):
            values = [
                str(ephemeris.prn),
                str(ephemeris.week_number_mod1024),
                str(ephemeris.iode),
                str(ephemeris.health),
                f"{ephemeris.toe_s:.0f}",
                f"{ephemeris.toc_s:.0f}",
                f"{ephemeris.sqrt_a_sqrt_m:.3f}",
                f"{ephemeris.eccentricity:.8f}",
            ]
            for column, value in enumerate(values):
                self.ephemeris_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self.ephemeris_table.resizeColumnsToContents()
        self.ephemeris_table.horizontalHeader().setStretchLastSection(True)

    def _update_observation_table(self, result: PVTComputationResult) -> None:
        self.observation_table.setRowCount(len(result.observations))
        for row, observation in enumerate(result.observations):
            values = [
                str(observation.prn),
                str(observation.subframe_id),
                f"{observation.transmit_time_s:.0f}",
                f"{observation.receive_file_time_s:.6f}",
                f"{observation.pseudorange_m / 1_000.0:.3f}",
                f"{observation.satellite_clock_correction_s * 1e6:.3f}",
                str(observation.subframe_start_bit),
                str(observation.bit_start_ms),
            ]
            for column, value in enumerate(values):
                self.observation_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self.observation_table.resizeColumnsToContents()
        self.observation_table.horizontalHeader().setStretchLastSection(True)
