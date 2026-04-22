"""Basic GUI smoke test."""

from __future__ import annotations

from PySide6 import QtWidgets

from app.gui.main_window import MainWindow


def test_main_window_smoke() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.tabs.count() == 8
    assert window.session_tab.sample_rate_spin.value() == 6_061_000.0
    assert window.session_tab.compute_backend_combo.currentData() == "auto"
    assert window.session_tab.worker_spin.value() == 0
    assert "Runtime status:" in window.session_tab.compute_status_label.text()
    assert window.acquisition_tab.center_sweep_button.text() == "Sweep Search Center"
    assert window.acquisition_tab.auto_detect_button.text() == "Auto Detect Capture"
