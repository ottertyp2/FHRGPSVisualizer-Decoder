"""GUI source-switching regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtWidgets

from app.gui.main_window import MainWindow
from app.models import AcquisitionCandidate, AcquisitionResult, DemoSignalResult


def _make_result() -> AcquisitionResult:
    candidate = AcquisitionCandidate(
        prn=3,
        doppler_hz=0.0,
        carrier_frequency_hz=0.0,
        code_phase_samples=0,
        metric=9.0,
    )
    return AcquisitionResult(
        prn=3,
        sample_rate_hz=1_000_000.0,
        search_center_hz=0.0,
        doppler_bins_hz=np.asarray([0.0], dtype=np.float32),
        code_phases_samples=np.asarray([0], dtype=np.int32),
        heatmap=np.ones((1, 1), dtype=np.float32),
        best_candidate=candidate,
    )


def test_real_file_selection_after_demo_clears_demo_source(tmp_path: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    file_samples = np.asarray([10 + 1j, 20 + 2j, 30 + 3j, 40 + 4j], dtype=np.complex64)
    file_path = tmp_path / "iq.bin"
    file_samples.tofile(file_path)

    demo_samples = np.asarray([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex64)
    window.demo_signal = DemoSignalResult(
        samples=demo_samples,
        prn=3,
        sample_rate=1_000_000.0,
        doppler_hz=0.0,
        code_phase_samples=0,
        nav_bits=np.zeros(1, dtype=np.int8),
    )
    window.current_samples = demo_samples
    window.current_display_samples = demo_samples
    window.current_samples_signature = ("<demo>", True, 0, demo_samples.size)
    window.acquisition_result = _make_result()
    window.acquisition_results_by_prn[3] = window.acquisition_result

    window.session_tab.set_file_path(str(file_path))
    window.session_tab.sample_rate_spin.setValue(1_000_000.0)
    window.session_tab.sample_count_spin.setValue(float(file_samples.size))
    window.inspect_file(str(file_path))

    assert window.demo_signal is None
    assert window.acquisition_result is None
    assert window.acquisition_results_by_prn == {}
    np.testing.assert_allclose(window.ensure_samples(), file_samples)
