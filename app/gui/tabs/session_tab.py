"""Session configuration and logging tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS
from app.models import FileMetadata, SessionConfig


class SessionTab(QtWidgets.QWidget):
    """File loading, metadata, logs, and analysis control tab."""

    load_file_requested = QtCore.Signal()
    preview_requested = QtCore.Signal()
    acquisition_requested = QtCore.Signal()
    tracking_requested = QtCore.Signal()
    decode_requested = QtCore.Signal()
    demo_requested = QtCore.Signal()
    benchmark_requested = QtCore.Signal()
    settings_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        info_label = QtWidgets.QLabel(
            "Load a complex64 IQ file or generate a demo signal, then inspect a sample window."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form = QtWidgets.QFormLayout()
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setReadOnly(True)
        self.sample_rate_spin = QtWidgets.QDoubleSpinBox()
        self.sample_rate_spin.setRange(1_000.0, 100_000_000.0)
        self.sample_rate_spin.setDecimals(1)
        self.sample_rate_spin.setValue(4_092_000.0)
        self.sample_rate_spin.setSuffix(" Sa/s")
        self.center_frequency_spin = QtWidgets.QDoubleSpinBox()
        self.center_frequency_spin.setRange(0.0, 10_000_000_000.0)
        self.center_frequency_spin.setDecimals(1)
        self.center_frequency_spin.setValue(1_575_420_000.0)
        self.center_frequency_spin.setSuffix(" Hz")
        self.if_frequency_spin = QtWidgets.QDoubleSpinBox()
        self.if_frequency_spin.setRange(-50_000_000.0, 50_000_000.0)
        self.if_frequency_spin.setDecimals(1)
        self.if_frequency_spin.setValue(0.0)
        self.if_frequency_spin.setSuffix(" Hz")
        self.if_frequency_spin.setToolTip(
            "Intermediate frequency / search center inside the sampled spectrum. "
            "Use 0 Hz for true baseband data."
        )
        self.baseband_checkbox = QtWidgets.QCheckBox("Baseband / residual Doppler search")
        self.baseband_checkbox.setChecked(True)
        self.start_sample_spin = QtWidgets.QSpinBox()
        self.start_sample_spin.setRange(0, 2_000_000_000)
        self.sample_count_spin = QtWidgets.QSpinBox()
        self.sample_count_spin.setRange(1_000, 2_000_000_000)
        self.sample_count_spin.setValue(4_092_000)
        self.preload_checkbox = QtWidgets.QCheckBox("Preload full window to RAM")
        self.preload_checkbox.setChecked(True)
        self.preload_checkbox.setToolTip(
            "If enabled, the tool loads the complete selected source into RAM before analysis. "
            "If disabled, only the active analysis window is loaded on demand."
        )

        form.addRow("File", self.file_edit)
        form.addRow("Sample rate", self.sample_rate_spin)
        form.addRow("Center frequency", self.center_frequency_spin)
        form.addRow("Signal mode", self.baseband_checkbox)
        form.addRow("IF / search center", self.if_frequency_spin)
        form.addRow("Start sample", self.start_sample_spin)
        form.addRow("Window samples", self.sample_count_spin)
        form.addRow("RAM preload", self.preload_checkbox)
        layout.addLayout(form)

        button_row = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("Load File")
        self.preview_button = QtWidgets.QPushButton("Preview")
        self.acq_button = QtWidgets.QPushButton("Start Acquisition")
        self.track_button = QtWidgets.QPushButton("Start Tracking")
        self.decode_button = QtWidgets.QPushButton("Decode Bits / Nav")
        self.demo_button = QtWidgets.QPushButton("Generate Demo Signal")
        self.benchmark_button = QtWidgets.QPushButton("Run Benchmark")
        for button in (
            self.load_button,
            self.preview_button,
            self.acq_button,
            self.track_button,
            self.decode_button,
            self.demo_button,
            self.benchmark_button,
        ):
            button_row.addWidget(button)
        layout.addLayout(button_row)

        self.metadata_label = QtWidgets.QLabel("No file loaded.")
        self.metadata_label.setWordWrap(True)
        layout.addWidget(self.metadata_label)

        self.large_file_label = QtWidgets.QLabel(
            "RAM mode: after you start analysis, the complete IQ file is loaded into RAM once and all views work from that in-memory copy."
        )
        self.large_file_label.setWordWrap(True)
        layout.addWidget(self.large_file_label)

        self.ram_status_label = QtWidgets.QLabel("RAM status: no source selected.")
        self.ram_status_label.setWordWrap(True)
        layout.addWidget(self.ram_status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.preview_plot = pg.PlotWidget(title="Preview magnitude")
        self.preview_plot.setToolTip(TOOLTIPS["spectrum"])
        self.preview_plot.setLabel("bottom", "Time", units="s")
        self.preview_plot.setLabel("left", "Magnitude")
        self.preview_curve = self.preview_plot.plot(pen="y")
        layout.addWidget(self.preview_plot, stretch=1)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(500)
        layout.addWidget(self.log_edit, stretch=1)

        self.load_button.clicked.connect(self.load_file_requested.emit)
        self.preview_button.clicked.connect(self.preview_requested.emit)
        self.acq_button.clicked.connect(self.acquisition_requested.emit)
        self.track_button.clicked.connect(self.tracking_requested.emit)
        self.decode_button.clicked.connect(self.decode_requested.emit)
        self.demo_button.clicked.connect(self.demo_requested.emit)
        self.benchmark_button.clicked.connect(self.benchmark_requested.emit)

        self.sample_rate_spin.valueChanged.connect(lambda *_: self.settings_changed.emit())
        self.if_frequency_spin.valueChanged.connect(lambda *_: self.settings_changed.emit())
        self.start_sample_spin.valueChanged.connect(lambda *_: self.settings_changed.emit())
        self.sample_count_spin.valueChanged.connect(lambda *_: self.settings_changed.emit())
        self.preload_checkbox.toggled.connect(lambda *_: self.settings_changed.emit())
        self.baseband_checkbox.toggled.connect(self._on_baseband_toggled)
        self._on_baseband_toggled(self.baseband_checkbox.isChecked())

    def _on_baseband_toggled(self, checked: bool) -> None:
        """Enable or disable the IF hint depending on the signal mode."""

        self.if_frequency_spin.setEnabled(not checked)
        if checked:
            self.if_frequency_spin.setValue(0.0)
        self.settings_changed.emit()

    def preload_enabled(self) -> bool:
        """Return whether preload mode is enabled."""

        return self.preload_checkbox.isChecked()

    def set_ram_status(self, text: str) -> None:
        """Show RAM usage and load-policy information."""

        self.ram_status_label.setText(text)

    def get_session_config(self) -> SessionConfig:
        """Build a session config from the current UI state."""

        return SessionConfig(
            file_path=self.file_edit.text() or None,
            sample_rate=float(self.sample_rate_spin.value()),
            center_frequency=float(self.center_frequency_spin.value()),
            is_baseband=self.baseband_checkbox.isChecked(),
            if_frequency_hz=float(self.if_frequency_spin.value()),
            start_sample=int(self.start_sample_spin.value()),
            sample_count=int(self.sample_count_spin.value()),
        )

    def set_file_path(self, file_path: str) -> None:
        self.file_edit.setText(file_path)

    def set_metadata(self, metadata: FileMetadata) -> None:
        self.metadata_label.setText(
            f"File: {metadata.file_name}\n"
            f"Type: {metadata.data_type} ({metadata.endianness}-endian)\n"
            f"Samples: {metadata.total_samples:,}\n"
            f"Estimated duration: {metadata.estimated_duration_s:.3f} s\n"
            f"Preview RMS: {metadata.preview_stats.get('rms', 0.0):.6f}"
        )
        if metadata.total_samples > 0:
            self.start_sample_spin.setMaximum(metadata.total_samples - 1)
            self.sample_count_spin.setMaximum(metadata.total_samples)
        if metadata.preview_samples.size:
            time_axis = np.arange(metadata.preview_samples.size, dtype=float) / max(metadata.sample_rate_hz, 1.0)
            self.preview_curve.setData(time_axis, abs(metadata.preview_samples))

    def append_log(self, message: str) -> None:
        self.log_edit.appendPlainText(message)

    def set_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)
