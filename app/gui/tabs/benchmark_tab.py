"""Performance benchmark visualization tab."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg

from app.models import BenchmarkResult


class BenchmarkTab(QtWidgets.QWidget):
    """Show laptop suitability, throughput, and bottleneck information."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        info_label = QtWidgets.QLabel(
            "This benchmark estimates whether the laptop can keep up with large IQ recordings and highlights the slowest subsystem."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.summary_label = QtWidgets.QLabel("Benchmark not run yet.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.system_label = QtWidgets.QLabel("")
        self.system_label.setWordWrap(True)
        layout.addWidget(self.system_label)

        self.plot = pg.PlotWidget(title="Realtime factor at 6 MSa/s")
        self.bar_item = pg.BarGraphItem(x=[], height=[], width=0.6, brush="y")
        self.plot.addItem(self.bar_item)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot, stretch=1)

        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Component",
                "Elapsed [s]",
                "MSa/s",
                "MB/s",
                "x realtime\ncurrent",
                "x realtime\n6 MSa/s",
                "10 GiB est. [s]",
                "Detail",
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=2)

    def update_result(self, result: BenchmarkResult) -> None:
        """Refresh the benchmark summary, chart, and table."""

        self.summary_label.setText(result.suitability_summary)
        self.system_label.setText(
            " | ".join(
                [
                    f"CPU: {result.system_info.get('cpu', 'unknown')}",
                    f"Cores: {result.system_info.get('logical_cores', 'unknown')}",
                    f"RAM: {result.system_info.get('ram_gib', 'unknown')} GiB",
                    f"Python: {result.system_info.get('python', 'unknown')}",
                ]
            )
        )

        self.table.setRowCount(len(result.components))
        names: list[str] = []
        heights: list[float] = []
        brushes: list[str] = []
        for row, component in enumerate(result.components):
            names.append(component.name)
            heights.append(component.realtime_factor_target)
            brushes.append("r" if component.name == result.bottleneck_name else "y")
            values = [
                component.name,
                f"{component.elapsed_s:.3f}",
                f"{component.throughput_samples_s / 1e6:.2f}",
                f"{component.throughput_mbytes_s:.1f}",
                f"{component.realtime_factor_current:.2f}",
                f"{component.realtime_factor_target:.2f}",
                f"{component.estimated_time_for_10gb_s:.1f}",
                component.detail,
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

        self.bar_item.setOpts(x=np.arange(len(heights)), height=heights, width=0.6, brushes=brushes)
        axis = self.plot.getAxis("bottom")
        axis.setTicks([list(enumerate(names))])
        self.plot.setYRange(0.0, max(max(heights, default=1.0) * 1.15, 1.0))
