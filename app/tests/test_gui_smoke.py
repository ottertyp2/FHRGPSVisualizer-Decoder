"""Basic GUI smoke test."""

from __future__ import annotations

from PySide6 import QtWidgets

from app.gui.main_window import MainWindow


def test_main_window_smoke() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.tabs.count() == 8
    assert window.acquisition_tab.center_sweep_button.text() == "Sweep Search Center"
