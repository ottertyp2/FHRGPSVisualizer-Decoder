"""Basic GUI smoke test."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from app.gui.main_window import MainWindow
from app.models import DEFAULT_SAMPLE_RATE_HZ, FileMetadata


def test_main_window_smoke() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.tabs.count() == 8
    assert window.session_tab.sample_rate_spin.value() == round(DEFAULT_SAMPLE_RATE_HZ)
    assert window.session_tab.compute_backend_combo.currentData() == "auto"
    assert window.session_tab.worker_spin.value() == 0
    assert "Runtime status:" in window.session_tab.compute_status_label.text()
    assert window.acquisition_tab.center_sweep_button.text() == "Sweep Search Center"
    assert window.acquisition_tab.auto_detect_button.text() == "Auto Detect Capture"
    assert window.acquisition_tab.task_status_label.text() == "Acquisition idle."
    assert window.tracking_tab.task_status_label.text() == "Tracking idle."
    assert window.navigation_tab.task_status_label.text() == "Navigation decoder idle."


def test_session_tab_accepts_large_file_sample_ranges() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    metadata = FileMetadata(
        file_path="huge.bin",
        file_name="huge.bin",
        file_size_bytes=28 * 1024 ** 3,
        data_type="complex64",
        endianness="little",
        sample_rate_hz=6_061_000.0,
        total_samples=3_599_999_999,
        estimated_duration_s=593.9600725952813,
        common_rate_duration_hints={},
        preview_stats={"rms": 0.0},
        preview_samples=np.zeros(16, dtype=np.complex64),
    )

    window.session_tab.set_metadata(metadata)

    assert int(window.session_tab.start_sample_spin.maximum()) == metadata.total_samples - 1
    assert int(window.session_tab.sample_count_spin.maximum()) == metadata.total_samples


def test_inspect_file_disables_preload_for_oversized_sources(monkeypatch) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    metadata = FileMetadata(
        file_path="huge.bin",
        file_name="huge.bin",
        file_size_bytes=28 * 1024 ** 3,
        data_type="complex64",
        endianness="little",
        sample_rate_hz=6_061_000.0,
        total_samples=3_599_999_999,
        estimated_duration_s=593.9600725952813,
        common_rate_duration_hints={},
        preview_stats={"rms": 0.0},
        preview_samples=np.zeros(16, dtype=np.complex64),
    )

    monkeypatch.setattr("app.gui.main_window.inspect_complex64_file", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(window, "_memory_status", lambda: (32 * 1024 ** 3, 20 * 1024 ** 3))

    window.session_tab.preload_checkbox.setChecked(True)
    window.inspect_file("huge.bin")

    assert not window.session_tab.preload_enabled()
    assert "turned off automatically" in window.session_tab.log_edit.toPlainText()


def test_windowed_deep_acquisition_uses_the_selected_window() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    selected_window = (np.arange(50, dtype=np.float32) + 1j * np.arange(50, dtype=np.float32)).astype(np.complex64)
    window.current_samples = selected_window
    window.current_samples_signature = ("memory", False, 10, selected_window.size)
    window.session.file_path = "memory"
    window.session_tab.sample_rate_spin.setValue(1_000.0)
    window.session_tab.start_sample_spin.setValue(10)
    window.session_tab.sample_count_spin.setValue(selected_window.size)
    window.session_tab.preload_checkbox.setChecked(True)
    window.session_tab.preload_checkbox.setChecked(False)
    window.acquisition_tab.integration_spin.setValue(20)
    window.acquisition_tab.segment_count_spin.setValue(4)

    acquisition_samples = window.load_acquisition_samples()

    np.testing.assert_allclose(acquisition_samples, selected_window)
