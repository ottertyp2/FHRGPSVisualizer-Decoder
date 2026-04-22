"""IQ-plane and phasor visualization tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS, decimate_for_display


class IQPlaneTab(QtWidgets.QWidget):
    """Interactive IQ scatter and phasor view."""

    def __init__(self) -> None:
        super().__init__()
        self.sources: dict[str, np.ndarray] = {}
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self._advance_slider)

        layout = QtWidgets.QVBoxLayout(self)
        help_label = QtWidgets.QLabel(
            f"{TOOLTIPS['bpsk']} {TOOLTIPS['despreading']} {TOOLTIPS['early_late']}"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        controls = QtWidgets.QHBoxLayout()
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Raw IQ"])
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(10, 100_000)
        self.window_spin.setValue(2_000)
        self.play_button = QtWidgets.QPushButton("Play")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.connect_checkbox = QtWidgets.QCheckBox("Show trajectory")
        self.connect_checkbox.setChecked(True)
        controls.addWidget(QtWidgets.QLabel("View"))
        controls.addWidget(self.source_combo)
        controls.addWidget(QtWidgets.QLabel("Window"))
        controls.addWidget(self.window_spin)
        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.connect_checkbox)
        controls.addStretch()
        layout.addLayout(controls)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        layout.addWidget(self.slider)

        self.plot = pg.PlotWidget(title="IQ Plane")
        self.scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(50, 180, 255, 170))
        self.path_curve = self.plot.plot(pen=pg.mkPen((255, 255, 255, 100), width=1))
        self.arrow_curve = self.plot.plot(pen=pg.mkPen("y", width=2))
        self.plot.addItem(self.scatter)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Imag")
        self.plot.setLabel("bottom", "Real")
        layout.addWidget(self.plot, stretch=1)

        self.value_label = QtWidgets.QLabel("I=0, Q=0, |z|=0, phase=0 rad / 0 deg")
        layout.addWidget(self.value_label)
        self.decimation_label = QtWidgets.QLabel("")
        layout.addWidget(self.decimation_label)

        self.source_combo.currentTextChanged.connect(self.refresh_plot)
        self.slider.valueChanged.connect(self.refresh_plot)
        self.window_spin.valueChanged.connect(self.refresh_plot)
        self.connect_checkbox.toggled.connect(self.refresh_plot)
        self.play_button.clicked.connect(self.timer.start)
        self.pause_button.clicked.connect(self.timer.stop)

    def set_sources(self, sources: dict[str, np.ndarray]) -> None:
        """Update the available IQ views."""

        self.sources = {name: values for name, values in sources.items() if values.size}
        current = self.source_combo.currentText()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItems(list(self.sources.keys()) or ["Raw IQ"])
        if current in self.sources:
            self.source_combo.setCurrentText(current)
        self.source_combo.blockSignals(False)
        current_values = self.current_values()
        size = max(0, current_values.size - 1)
        self.slider.setRange(0, size)
        self.refresh_plot()

    def current_values(self) -> np.ndarray:
        """Return the active signal view."""

        return self.sources.get(self.source_combo.currentText(), np.empty(0, dtype=np.complex64))

    def _advance_slider(self) -> None:
        maximum = self.slider.maximum()
        if maximum <= 0:
            return
        new_value = self.slider.value() + 1
        if new_value > maximum:
            new_value = 0
        self.slider.setValue(new_value)

    def refresh_plot(self) -> None:
        """Redraw the scatter, trajectory, and current phasor."""

        values = self.current_values()
        if values.size == 0:
            return
        current_index = min(self.slider.value(), values.size - 1)
        half_window = self.window_spin.value() // 2
        start = max(0, current_index - half_window)
        stop = min(values.size, current_index + half_window)
        view = values[start:stop]
        display, step = decimate_for_display(view, max_points=2_500)
        self.scatter.setData(display.real, display.imag)
        if self.connect_checkbox.isChecked():
            self.path_curve.setData(display.real, display.imag)
        else:
            self.path_curve.setData([], [])
        current = values[current_index]
        self.arrow_curve.setData([0.0, current.real], [0.0, current.imag])
        self.value_label.setText(
            f"I={current.real:.6f}, Q={current.imag:.6f}, |z|={abs(current):.6f}, "
            f"phase={np.angle(current):.3f} rad / {np.degrees(np.angle(current)):.2f} deg"
        )
        self.decimation_label.setText(
            "Display decimated for readability." if step > 1 else "Showing all points in the selected window."
        )
