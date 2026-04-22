"""Application entry point."""

from __future__ import annotations

import sys

from PySide6 import QtWidgets
import pyqtgraph as pg

from app.gui.main_window import MainWindow


def main() -> int:
    """Run the Qt application."""

    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("GPS L1 C/A Offline Decoder")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
