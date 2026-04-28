"""Basic GUI smoke test."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

from app.gui.tabs.acquisition_tab import AcquisitionTab
from app.gui.main_window import MainWindow
from app.models import (
    DEFAULT_SAMPLE_RATE_HZ,
    AcquisitionCandidate,
    AcquisitionResult,
    FileMetadata,
    SampleRateSurveyEntry,
    SampleRateSurveyResult,
    SearchCenterSweepEntry,
    SearchCenterSweepResult,
    TrackingState,
)


def _make_acquisition_result(
    prn: int,
    metric: float = 9.0,
    *,
    sample_rate_hz: float = 1_000_000.0,
    search_center_hz: float = 0.0,
    doppler_hz: float | None = None,
) -> AcquisitionResult:
    doppler = 500.0 * prn if doppler_hz is None else float(doppler_hz)
    candidate = AcquisitionCandidate(
        prn=prn,
        doppler_hz=doppler,
        carrier_frequency_hz=search_center_hz + doppler,
        code_phase_samples=100 * prn,
        metric=metric,
    )
    return AcquisitionResult(
        prn=prn,
        sample_rate_hz=sample_rate_hz,
        search_center_hz=search_center_hz,
        doppler_bins_hz=np.asarray([-500.0, 0.0, 500.0], dtype=np.float32),
        code_phases_samples=np.asarray([0, 1, 2], dtype=np.int32),
        heatmap=np.ones((3, 3), dtype=np.float32) * metric,
        best_candidate=candidate,
        consistent_segments=4,
        consistency_score=metric * 4.0,
    )


def _make_tracking_state(prn: int) -> TrackingState:
    values = np.ones(8, dtype=np.float32)
    return TrackingState(
        prn=prn,
        times_s=np.arange(8, dtype=np.float32) * 1e-3,
        prompt_i=values.copy(),
        prompt_q=np.zeros(8, dtype=np.float32),
        early_mag=values * 0.8,
        prompt_mag=values,
        late_mag=values * 0.8,
        code_error=np.zeros(8, dtype=np.float32),
        carrier_error=np.zeros(8, dtype=np.float32),
        doppler_est_hz=values * 500.0,
        code_freq_est_hz=values * 1_023_000.0,
        lock_metric=values * 2.0,
        lock_detected=True,
    )


def test_main_window_smoke() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.tabs.count() == 10
    assert window.tabs.tabText(1) == "Learning Flow"
    assert window.tabs.tabText(2) == "Signal Intuition"
    assert window.concept_lab_tab.current_result is not None
    assert window.session_tab.sample_rate_spin.value() == round(DEFAULT_SAMPLE_RATE_HZ)
    assert window.session_tab.compute_backend_combo.currentData() == "auto"
    assert window.session_tab.worker_spin.value() == 0
    assert "Runtime status:" in window.session_tab.compute_status_label.text()
    assert window.acquisition_tab.center_sweep_button.text() == "Sweep Search Center"
    assert window.acquisition_tab.auto_detect_button.text() == "Auto Detect Capture"
    assert window.acquisition_tab.scan_button.text() == "Scan PRN List"
    assert window.acquisition_tab.detail_tabs.count() == 6
    assert window.acquisition_tab.detail_tabs.tabText(0) == "Satellites"
    assert window.acquisition_tab.heatmap_plot.minimumHeight() >= 360
    assert window.tracking_tab.tracking_tabs.count() == 4
    assert window.tracking_tab.tracking_tabs.tabText(0) == "IQ Stages"
    assert window.tracking_tab.prompt_plot.plotItem.legend is not None
    assert window.tracking_tab.freq_plot.plotItem.legend is not None
    assert window.tracking_tab.prompt_i_curve.opts["name"] == "Prompt I"
    assert window.tracking_tab.doppler_curve.opts["name"] == "Doppler estimate"
    assert window.navigation_tab.navigation_tabs.count() == 7
    assert window.navigation_tab.navigation_tabs.tabText(1) == "LNAV Words"
    assert window.navigation_tab.navigation_tabs.tabText(2) == "Subframes"
    assert window.navigation_tab.navigation_tabs.tabText(4) == "Almanac / Ephemeris"
    assert window.raw_tab.i_plot.minimumHeight() >= 260
    assert window.navigation_tab.bit_source_mode() == "auto"
    assert window.acquisition_tab.task_status_label.text() == "Acquisition idle."
    assert window.tracking_tab.task_status_label.text() == "Tracking idle."
    assert window.navigation_tab.task_status_label.text() == "Navigation decoder idle."


def test_acquisition_scan_prn_list_parser() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    window.acquisition_tab.scan_prns_edit.setText("1, 3, 8-10")
    assert window.acquisition_tab.selected_scan_prns() == [1, 3, 8, 9, 10]

    window.acquisition_tab.scan_prns_edit.setText("33")
    with pytest.raises(ValueError):
        window.acquisition_tab.selected_scan_prns()


def test_prn_doppler_overview_collapses_each_prn_to_best_code_phase() -> None:
    first = _make_acquisition_result(1)
    first.heatmap = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0],
            [9.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    second = _make_acquisition_result(2)
    second.heatmap = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [7.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )

    ordered, dopplers, overview = AcquisitionTab.build_prn_doppler_overview([second, first])

    assert [result.prn for result in ordered] == [1, 2]
    np.testing.assert_allclose(dopplers, [-500.0, 0.0, 500.0])
    np.testing.assert_allclose(
        overview[0],
        np.max(first.heatmap, axis=1) / np.mean(first.heatmap),
    )
    np.testing.assert_allclose(
        overview[1],
        np.max(second.heatmap, axis=1) / np.mean(second.heatmap),
    )


def test_prn_doppler_overview_does_not_mark_every_noise_peak() -> None:
    weak = _make_acquisition_result(1, metric=2.0)
    strong = _make_acquisition_result(2, metric=10.0)
    selected_weak = _make_acquisition_result(3, metric=2.0)
    ordered = [weak, strong, selected_weak]

    visible = AcquisitionTab.threshold_prn_doppler_overview(
        np.asarray([[1.0, 5.9, 6.0, 10.0]], dtype=np.float32)
    )
    marker_rows = AcquisitionTab.overview_marker_rows(ordered, selected_prn=3)

    np.testing.assert_allclose(visible, [[0.0, 0.0, 6.0, 10.0]])
    assert marker_rows == [1, 2]


def test_prn_doppler_overview_uses_sparse_axis_labels_for_full_scan() -> None:
    results = [_make_acquisition_result(prn) for prn in range(1, 33)]

    ticks = AcquisitionTab.sparse_prn_axis_ticks(results, selected_prn=32)

    assert len(ticks) < 16
    assert ticks[-1] == (31.5, "32")


def test_acquisition_peak_slices_use_codephase_and_doppler_axes() -> None:
    result = _make_acquisition_result(1)
    result.heatmap = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 20.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    result.doppler_bins_hz = np.asarray([-500.0, 0.0, 500.0], dtype=np.float32)
    result.best_candidate.doppler_hz = 0.0
    result.best_candidate.code_phase_samples = 1

    code_axis, code_values, peak_phase, peak_value = AcquisitionTab.codephase_slice(result)
    doppler_axis, doppler_values, peak_doppler, doppler_peak_value = AcquisitionTab.doppler_slice(result)

    np.testing.assert_allclose(code_axis, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(code_values, [5.0, 20.0, 7.0, 6.0])
    assert peak_phase == 1.0
    assert peak_value == 20.0
    np.testing.assert_allclose(doppler_axis, [-500.0, 0.0, 500.0])
    np.testing.assert_allclose(doppler_values, [4.0, 20.0, 12.0])
    assert peak_doppler == 0.0
    assert doppler_peak_value == 20.0


def test_acquisition_table_labels_rows_from_result_prn() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = AcquisitionTab()
    result = _make_acquisition_result(3)
    result.best_candidate.prn = 8

    tab.update_result(result)

    assert "Best peak: PRN 3" in tab.summary_label.text()
    assert tab.satellite_table.item(0, 0).text() == "3"


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


def test_per_prn_views_keep_all_acquisition_candidates_selectable() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.acquisition_results_by_prn = {
        1: _make_acquisition_result(1, metric=10.0),
        2: _make_acquisition_result(2, metric=8.0),
    }
    window.tracking_results_by_prn[1] = _make_tracking_state(1)
    window.selected_prn = 1

    window.refresh_satellite_views()

    assert window.tracking_tab.prn_combo.findData(1) >= 0
    assert window.tracking_tab.prn_combo.findData(2) >= 0
    assert window.acquisition_tab.satellite_table.item(0, 6).text() == "tracked"
    assert "Overview heatmap compares 2 scanned PRN rows" in window.acquisition_tab.summary_label.text()
    assert "Selected PRN: 1" in window.learning_tab.selected_label.text()

    window.set_selected_prn(2)

    assert "Selected PRN 2" in window.acquisition_tab.selected_prn_label.text()
    assert window.acquisition_tab.prn_spin.value() == 2
    assert "Selected PRN: 2" in window.learning_tab.selected_label.text()


def test_tracking_requires_acquisition_for_selected_prn(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **_kwargs: warnings.append(args))
    window.acquisition_result = _make_acquisition_result(3)
    window.selected_prn = 8

    window.start_tracking()

    assert warnings
    assert window.tracking_tab.task_status_label.text() == "Run acquisition for PRN 8 before tracking."


def test_tracking_rejects_stale_acquisition_context(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **_kwargs: warnings.append(args))
    result = _make_acquisition_result(3, sample_rate_hz=float(window.session_tab.sample_rate_spin.value()))
    window.acquisition_result = result
    window.acquisition_results_by_prn[3] = result
    window._remember_acquisition_contexts([result])
    window.selected_prn = 3

    window.session_tab.sample_rate_spin.setValue(window.session_tab.sample_rate_spin.value() + 10_000.0)
    window.start_tracking()

    assert warnings
    assert "Run acquisition again for PRN 3" in window.tracking_tab.task_status_label.text()


def test_search_center_selection_swaps_to_matching_sweep_result() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.session_tab.set_file_path("")
    first = _make_acquisition_result(3, search_center_hz=0.0, doppler_hz=500.0)
    second = _make_acquisition_result(8, search_center_hz=1_000.0, doppler_hz=-250.0)
    window.search_center_sweep_result = SearchCenterSweepResult(
        [
            SearchCenterSweepEntry(search_center_hz=0.0, best_result=first),
            SearchCenterSweepEntry(search_center_hz=1_000.0, best_result=second),
        ]
    )
    window.acquisition_results_by_prn = {3: first}
    window.acquisition_result = first

    window.apply_search_center_selection(1_000.0, 8)

    assert not window.session.is_baseband
    assert window.session.if_frequency_hz == 1_000.0
    assert window.acquisition_result is second
    assert window.acquisition_results_by_prn == {8: second}
    assert window.selected_prn == 8


def test_sample_rate_selection_swaps_to_matching_survey_results() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.session_tab.set_file_path("")
    first = _make_acquisition_result(3, sample_rate_hz=1_000_000.0, doppler_hz=500.0)
    second = _make_acquisition_result(8, sample_rate_hz=2_000_000.0, doppler_hz=-250.0)
    window.sample_rate_survey_result = SampleRateSurveyResult(
        [
            SampleRateSurveyEntry(sample_rate_hz=1_000_000.0, best_result=first, all_results=[first]),
            SampleRateSurveyEntry(sample_rate_hz=2_000_000.0, best_result=second, all_results=[second]),
        ]
    )
    window.acquisition_results_by_prn = {3: first}
    window.acquisition_result = first

    window.apply_sample_rate_selection(2_000_000.0, 8)

    assert window.session.sample_rate == 2_000_000.0
    assert window.acquisition_result is second
    assert window.acquisition_results_by_prn == {8: second}
    assert window.selected_prn == 8


def test_tracking_loop_controls_sync_into_session() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.tracking_tab.early_late_spin.setValue(0.35)
    window.tracking_tab.dll_gain_spin.setValue(0.12)
    window.tracking_tab.pll_gain_spin.setValue(8.5)
    window.tracking_tab.fll_gain_spin.setValue(0.22)

    window.sync_session_from_ui()

    assert window.session.early_late_spacing_chips == 0.35
    assert window.session.dll_gain == 0.12
    assert window.session.pll_gain == 8.5
    assert window.session.fll_gain == 0.22
