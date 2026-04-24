"""Guided overview of the GPS decoding pipeline."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from app.models import AcquisitionResult, BitDecisionResult, NavigationDecodeResult, TrackingState


class LearningTab(QtWidgets.QWidget):
    """Compact, PRN-specific explanation of the current receiver pipeline state."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        self.title_label = QtWidgets.QLabel("GPS L1 C/A decoding flow")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.selected_label = QtWidgets.QLabel("Selected PRN: none")
        self.selected_label.setWordWrap(True)
        layout.addWidget(self.selected_label)

        self.pipeline_table = QtWidgets.QTableWidget(5, 4)
        self.pipeline_table.setHorizontalHeaderLabels(["Stage", "What it does", "Current evidence", "Next useful action"])
        self.pipeline_table.verticalHeader().setVisible(False)
        self.pipeline_table.horizontalHeader().setStretchLastSection(True)
        self.pipeline_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.pipeline_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout.addWidget(self.pipeline_table, stretch=2)

        self.layer_text = QtWidgets.QPlainTextEdit()
        self.layer_text.setReadOnly(True)
        self.layer_text.setPlainText(
            "\n".join(
                [
                    "Layer separation to keep in mind:",
                    "1. Carrier/Doppler: the complex IQ phasor rotates until you wipe off the right frequency.",
                    "2. PRN code: the selected satellite becomes narrow only when the local C/A code and code phase match.",
                    "3. 1 ms prompt: after despreading, each millisecond becomes one complex prompt value.",
                    "4. 20 ms bits: GPS LNAV data changes slowly, so twenty prompt values form one navigation bit.",
                    "5. Navigation words: preambles and parity tell you whether the bit stream is believable.",
                ]
            )
        )
        layout.addWidget(self.layer_text, stretch=1)

        self.update_pipeline(None, None, None, None, None)

    def update_pipeline(
        self,
        selected_prn: int | None,
        acquisition: AcquisitionResult | None,
        tracking: TrackingState | None,
        bit_result: BitDecisionResult | None,
        nav_result: NavigationDecodeResult | None,
    ) -> None:
        """Refresh the pipeline table for the currently selected PRN."""

        self.selected_label.setText(
            f"Selected PRN: {selected_prn}" if selected_prn is not None else "Selected PRN: none"
        )

        acquisition_evidence = "Not run for this PRN yet."
        acquisition_action = "Run acquisition or click a PRN row after a scan."
        if acquisition is not None:
            best = acquisition.best_candidate
            acquisition_evidence = (
                f"Metric {best.metric:.2f}, Doppler {best.doppler_hz:+.1f} Hz, "
                f"code phase {best.code_phase_samples}, {acquisition.consistent_segments} repeated segments."
            )
            acquisition_action = "Track this PRN to see whether the peak survives over time."

        tracking_evidence = "Not tracked yet."
        tracking_action = "Start tracking once acquisition has a repeated candidate."
        if tracking is not None:
            prompt = float(np.median(tracking.prompt_mag)) if tracking.prompt_mag.size else 0.0
            side = float(np.median((tracking.early_mag + tracking.late_mag) * 0.5)) if tracking.early_mag.size else 0.0
            ratio = prompt / max(side, 1e-12)
            tracking_evidence = (
                f"{'Locked' if tracking.lock_detected else 'Not locked'}, "
                f"prompt/early-late ratio {ratio:.2f}, {tracking.times_s.size} ms processed."
            )
            tracking_action = "Decode bits when tracking is stable enough."

        bit_evidence = "No 20 ms bit decisions yet."
        bit_action = "Decode the tracked PRN."
        if bit_result is not None:
            bit_evidence = (
                f"{bit_result.bit_values.size} hard decisions, best 20 ms offset {bit_result.best_offset_ms} ms."
            )
            bit_action = "Inspect preambles and parity-valid words."

        nav_evidence = "No LNAV words decoded yet."
        nav_action = "Need parity-valid words before trusting navigation data."
        if nav_result is not None:
            nav_evidence = (
                f"{len(nav_result.preamble_indices)} preambles, {nav_result.parity_ok_count} parity-valid words."
            )
            nav_action = "Use several confirmed PRNs before attempting a position solution."

        rows = [
            (
                "Raw IQ",
                "The sampled complex voltage from the SDR.",
                "Loaded window is shared by all later stages.",
                "Preview spectrum/IQ before heavy processing.",
            ),
            (
                "Acquisition",
                "Tries PRN, Doppler, and code phase until correlation peaks.",
                acquisition_evidence,
                acquisition_action,
            ),
            (
                "Tracking",
                "Keeps carrier and code phase aligned millisecond by millisecond.",
                tracking_evidence,
                tracking_action,
            ),
            (
                "20 ms Bits",
                "Sums prompt values into slow 50 bps LNAV decisions.",
                bit_evidence,
                bit_action,
            ),
            (
                "LNAV Words",
                "Searches preambles and checks parity to prove the bit stream.",
                nav_evidence,
                nav_action,
            ),
        ]
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(value)
                self.pipeline_table.setItem(row, column, item)
        self.pipeline_table.resizeRowsToContents()
