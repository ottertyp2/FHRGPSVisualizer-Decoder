"""Raw signal visualization tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg

from app.dsp.utils import TOOLTIPS, decimate_for_display


class RawSignalTab(QtWidgets.QWidget):
    """Time-domain raw signal views."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        help_label = QtWidgets.QLabel(
            f"{TOOLTIPS['bpsk']} {TOOLTIPS['despreading']}"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.i_plot = pg.PlotWidget(title="I(t)")
        self.q_plot = pg.PlotWidget(title="Q(t)")
        self.mag_plot = pg.PlotWidget(title="Magnitude")
        self.phase_plot = pg.PlotWidget(title="Phase")
        self.i_curve = self.i_plot.plot(pen="c")
        self.q_curve = self.q_plot.plot(pen="m")
        self.mag_curve = self.mag_plot.plot(pen="y")
        self.phase_curve = self.phase_plot.plot(pen="w")

        for widget in (self.i_plot, self.q_plot, self.mag_plot, self.phase_plot):
            layout.addWidget(widget, stretch=1)

        self.decimation_label = QtWidgets.QLabel("")
        layout.addWidget(self.decimation_label)

    def update_signal(self, samples: np.ndarray) -> None:
        """Refresh the raw signal plots."""

        if samples.size == 0:
            return
        display, step = decimate_for_display(samples)
        x = np.arange(display.size) * step
        self.i_curve.setData(x, display.real)
        self.q_curve.setData(x, display.imag)
        self.mag_curve.setData(x, np.abs(display))
        self.phase_curve.setData(x, np.angle(display))
        self.decimation_label.setText(
            "Display decimated for readability." if step > 1 else "Showing full selected window."
        )
