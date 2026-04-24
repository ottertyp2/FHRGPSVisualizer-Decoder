"""Main application window."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

from app.dsp.acquisition import acquisition_from_session
from app.dsp.acquisition import acquisition_interpretation
from app.dsp.acquisition import acquisition_metric_is_strong
from app.dsp.acquisition import acquisition_rank_key
from app.dsp.acquisition import acquisition_result_is_plausible
from app.dsp.acquisition import scan_prns_from_session
from app.dsp.acquisition import survey_sample_rates
from app.dsp.acquisition import sweep_search_centers_from_session
from app.dsp.benchmark import run_benchmark
from app.dsp.compute import resolve_compute_plan
from app.dsp.demo import generate_demo_signal
from app.dsp.io import (
    COMMON_GNSS_SAMPLE_RATES,
    common_sample_rate_hints,
    inspect_complex64_file,
    load_complex64_file_with_progress,
    load_complex64_samples_with_progress,
)
from app.dsp.navdecode import decode_navigation_from_tracking
from app.dsp.tracking import track_file
from app.dsp.tracking import track_signal
from app.gui.tabs.acquisition_tab import AcquisitionTab
from app.gui.tabs.benchmark_tab import BenchmarkTab
from app.gui.tabs.iq_tab import IQPlaneTab
from app.gui.tabs.learning_tab import LearningTab
from app.gui.tabs.navigation_tab import NavigationTab
from app.gui.tabs.raw_signal_tab import RawSignalTab
from app.gui.tabs.session_tab import SessionTab
from app.gui.tabs.spectrum_tab import SpectrumTab
from app.gui.tabs.tracking_tab import TrackingTab
from app.gui.workers import Worker
from app.models import (
    AcquisitionResult,
    BenchmarkResult,
    BitDecisionResult,
    DemoSignalResult,
    FileMetadata,
    NavigationDecodeResult,
    SearchCenterSweepResult,
    SampleRateSurveyResult,
    SessionConfig,
    TrackingState,
)


class MainWindow(QtWidgets.QMainWindow):
    """Top-level GUI container and application controller."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GPS L1 C/A Offline Decoder")
        self.resize(1400, 1000)

        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.session = SessionConfig()
        self.file_metadata: FileMetadata | None = None
        self.current_samples = np.empty(0, dtype=np.complex64)
        self.current_display_samples = np.empty(0, dtype=np.complex64)
        self.current_samples_signature: tuple[str | None, bool, int, int] | None = None
        self.demo_signal: DemoSignalResult | None = None
        self.acquisition_result: AcquisitionResult | None = None
        self.tracking_state: TrackingState | None = None
        self.bit_result: BitDecisionResult | None = None
        self.nav_result: NavigationDecodeResult | None = None
        self.acquisition_results_by_prn: dict[int, AcquisitionResult] = {}
        self.tracking_results_by_prn: dict[int, TrackingState] = {}
        self.bit_results_by_prn: dict[int, BitDecisionResult] = {}
        self.nav_results_by_prn: dict[int, NavigationDecodeResult] = {}
        self.search_center_sweep_result: SearchCenterSweepResult | None = None
        self.sample_rate_survey_result: SampleRateSurveyResult | None = None
        self.selected_prn: int | None = None
        self.pending_navigation_prn: int | None = None

        self.tabs = QtWidgets.QTabWidget()
        self.session_tab = SessionTab()
        self.learning_tab = LearningTab()
        self.raw_tab = RawSignalTab()
        self.spectrum_tab = SpectrumTab()
        self.iq_tab = IQPlaneTab()
        self.acquisition_tab = AcquisitionTab()
        self.tracking_tab = TrackingTab()
        self.navigation_tab = NavigationTab()
        self.benchmark_tab = BenchmarkTab()

        self.tabs.addTab(self.session_tab, "File / Session")
        self.tabs.addTab(self.learning_tab, "Learning Flow")
        self.tabs.addTab(self.raw_tab, "Raw Signal")
        self.tabs.addTab(self.spectrum_tab, "Spectrum / Waterfall")
        self.tabs.addTab(self.iq_tab, "IQ Plane")
        self.tabs.addTab(self.acquisition_tab, "Acquisition")
        self.tabs.addTab(self.tracking_tab, "Tracking")
        self.tabs.addTab(self.navigation_tab, "Bits / Navigation")
        self.tabs.addTab(self.benchmark_tab, "Benchmark")
        self.setCentralWidget(self.tabs)

        self._connect_signals()
        self._load_default_file_if_present()

    def _clear_processing_results(self) -> None:
        """Drop results that belong to a previous source or analysis context."""

        self.acquisition_result = None
        self.tracking_state = None
        self.bit_result = None
        self.nav_result = None
        self.acquisition_results_by_prn.clear()
        self.tracking_results_by_prn.clear()
        self.bit_results_by_prn.clear()
        self.nav_results_by_prn.clear()
        self.search_center_sweep_result = None
        self.sample_rate_survey_result = None
        self.selected_prn = None
        self.pending_navigation_prn = None
        self.acquisition_tab.summary_label.setText("No acquisition result yet. Run one PRN or scan multiple PRNs.")
        self.acquisition_tab.selected_prn_label.setText("Selected PRN: none")
        self.acquisition_tab.candidate_table.setRowCount(0)
        self.acquisition_tab.rate_table.setRowCount(0)
        self.acquisition_tab.center_table.setRowCount(0)
        self.acquisition_tab.satellite_table.setRowCount(0)
        self.acquisition_tab.evidence_text.clear()
        self.acquisition_tab.clear_result_view()
        self.acquisition_tab.set_task_message("Acquisition idle.")
        self.acquisition_tab.set_task_progress(0)
        self.tracking_tab.set_available_prns([])
        self.tracking_tab.status_label.setText("Tracking not started.")
        self.tracking_tab.evidence_text.clear()
        self.tracking_tab.set_task_message("Tracking idle.")
        self.tracking_tab.set_task_progress(0)
        self.navigation_tab.set_available_prns([])
        self.navigation_tab.summary_label.setText("Navigation decoding not started.")
        self.navigation_tab.evidence_text.clear()
        self.navigation_tab.bits_text.clear()
        self.navigation_tab.word_table.setRowCount(0)
        self.navigation_tab.set_task_message("Navigation decoder idle.")
        self.navigation_tab.set_task_progress(0)
        self.learning_tab.update_pipeline(None, None, None, None, None)

    def _clear_loaded_samples(self) -> None:
        """Drop cached sample arrays so the next action reads the active source."""

        self.current_samples = np.empty(0, dtype=np.complex64)
        self.current_display_samples = np.empty(0, dtype=np.complex64)
        self.current_samples_signature = None

    def _connect_signals(self) -> None:
        self.session_tab.load_file_requested.connect(self.load_file_dialog)
        self.session_tab.preview_requested.connect(self.preview_selected_window)
        self.session_tab.acquisition_requested.connect(self.start_acquisition)
        self.session_tab.tracking_requested.connect(self.start_tracking)
        self.session_tab.decode_requested.connect(self.decode_navigation)
        self.session_tab.demo_requested.connect(self.generate_demo)
        self.session_tab.benchmark_requested.connect(self.start_benchmark)
        self.session_tab.settings_changed.connect(self.update_ram_status)
        self.acquisition_tab.run_requested.connect(self.start_acquisition)
        self.acquisition_tab.scan_requested.connect(self.scan_all_prns)
        self.acquisition_tab.sweep_requested.connect(self.start_search_center_sweep)
        self.acquisition_tab.auto_rate_survey_requested.connect(self.start_sample_rate_survey)
        self.acquisition_tab.track_selected_requested.connect(self.start_tracking)
        self.acquisition_tab.selection_changed.connect(self.set_selected_prn)
        self.acquisition_tab.sweep_selection_changed.connect(self.apply_search_center_selection)
        self.acquisition_tab.sample_rate_selection_changed.connect(self.apply_sample_rate_selection)
        self.tracking_tab.track_requested.connect(self.start_tracking)
        self.tracking_tab.decode_requested.connect(self.decode_navigation)
        self.tracking_tab.selection_changed.connect(self.set_selected_prn)
        self.tracking_tab.settings_changed.connect(self.update_ram_status)
        self.navigation_tab.decode_requested.connect(self.decode_navigation)
        self.navigation_tab.selection_changed.connect(self.set_selected_prn)

    def _load_default_file_if_present(self) -> None:
        default_file = Path.cwd() / "test3min.bin"
        if not default_file.exists():
            default_file = Path.cwd() / "test1.bin"
        if default_file.exists():
            self.session_tab.set_file_path(str(default_file))
            self.append_log(f"Found default IQ file: {default_file}")
            self.update_ram_status()

    def append_log(self, message: str) -> None:
        self.session_tab.append_log(message)
        self.statusBar().showMessage(message, 5_000)

    def _memory_status(self) -> tuple[int, int]:
        """Return total and available physical RAM in bytes on Windows."""

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        return 0, 0

    def _planned_ram_bytes(self) -> int:
        """Estimate how much data the next load would bring into RAM."""

        if self.demo_signal is not None:
            if self.session_tab.preload_enabled():
                return int(self.demo_signal.samples.nbytes)
            available = max(0, self.demo_signal.samples.size - int(self.session.start_sample))
            return int(min(available, self.session.sample_count)) * np.dtype(np.complex64).itemsize

        if self.file_metadata is None:
            return int(self.session.sample_count) * np.dtype(np.complex64).itemsize
        if self.session_tab.preload_enabled():
            return int(self.file_metadata.file_size_bytes)
        return int(self.session.sample_count) * np.dtype(np.complex64).itemsize

    def update_ram_status(self) -> None:
        """Refresh the RAM status label in the session tab."""

        self.sync_session_from_ui()
        total_ram, avail_ram = self._memory_status()
        planned = self._planned_ram_bytes()
        mode = "full source preload" if self.session_tab.preload_enabled() else "selected window only"
        signal_mode = "baseband" if self.session.is_baseband else f"IF center {self.session.if_frequency_hz:.1f} Hz"
        total_text = f"{total_ram / (1024 ** 3):.1f} GiB" if total_ram else "unknown"
        avail_text = f"{avail_ram / (1024 ** 3):.1f} GiB" if avail_ram else "unknown"
        planned_text = f"{planned / (1024 ** 2):.1f} MiB"
        compute_plan = resolve_compute_plan(
            self.session.compute_backend,
            self.session.max_workers,
            gpu_enabled=self.session.gpu_enabled,
        )
        self.session_tab.set_ram_status(
            f"RAM status: mode={mode}, signal={signal_mode}, planned load={planned_text}, available RAM={avail_text}, total RAM={total_text}."
        )
        self.session_tab.set_compute_status(compute_plan.status_text())

    def _should_prefer_windowed_loading(self, bytes_to_load: int) -> bool:
        """Return whether the source is large enough that windowed loading is safer."""

        total_ram, avail_ram = self._memory_status()
        if avail_ram and bytes_to_load > int(avail_ram * 0.7):
            return True
        if total_ram and bytes_to_load > int(total_ram * 0.5):
            return True
        return bytes_to_load >= 8 * 1024 ** 3

    def _confirm_large_ram_load(self, bytes_to_load: int) -> bool:
        """Warn before very large RAM loads."""

        total_ram, avail_ram = self._memory_status()
        warning = False
        if avail_ram and bytes_to_load > int(avail_ram * 0.7):
            warning = True
        if total_ram and bytes_to_load > int(total_ram * 0.5):
            warning = True
        if bytes_to_load >= 4 * 1024 ** 3:
            warning = True
        if not warning:
            return True

        reply = QtWidgets.QMessageBox.warning(
            self,
            "Large RAM load",
            "The requested RAM load is large.\n\n"
            f"Planned load: {bytes_to_load / (1024 ** 3):.2f} GiB\n"
            f"Available RAM: {avail_ram / (1024 ** 3):.2f} GiB\n"
            f"Total RAM: {total_ram / (1024 ** 3):.2f} GiB\n\n"
            "Continue anyway?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return reply == QtWidgets.QMessageBox.Yes

    def _load_with_progress_dialog(self, label: str, load_fn) -> np.ndarray:
        """Run a blocking load with a progress dialog."""

        dialog = QtWidgets.QProgressDialog(label, "Cancel", 0, 100, self)
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        def progress(value: int) -> None:
            dialog.setValue(max(0, min(100, value)))
            QtWidgets.QApplication.processEvents()
            if dialog.wasCanceled():
                raise RuntimeError("RAM load canceled by user.")

        try:
            data = load_fn(progress)
        finally:
            dialog.setValue(100)
            dialog.close()
        return data

    def set_selected_prn(self, prn: int) -> None:
        """Select one PRN across the GUI and refresh per-satellite views."""

        self.selected_prn = prn
        self.acquisition_tab.prn_spin.setValue(prn)
        self.tracking_tab.set_available_prns(self.available_prns(), prn)
        self.navigation_tab.set_available_prns(self.available_prns(with_tracking=True), prn)
        self.refresh_satellite_views()

    def apply_search_center_selection(self, center_hz: float, prn: int) -> None:
        """Apply one IF / search-center hypothesis from the sweep table."""

        if abs(center_hz) < 1e-9:
            self.session_tab.baseband_checkbox.setChecked(True)
        else:
            self.session_tab.baseband_checkbox.setChecked(False)
            self.session_tab.if_frequency_spin.setValue(center_hz)
        self.set_selected_prn(prn)
        self.append_log(
            f"Applied search-center hypothesis: center {center_hz:.1f} Hz, PRN {prn}."
        )

    def apply_sample_rate_selection(self, sample_rate_hz: float, prn: int) -> None:
        """Apply one sample-rate hypothesis from the survey table."""

        self.session_tab.sample_rate_spin.setValue(float(sample_rate_hz))
        self.sync_session_from_ui()
        if self.session.file_path and self.session.file_path != "<demo>":
            self.inspect_file(self.session.file_path)
        self.set_selected_prn(prn)
        self.append_log(
            f"Applied sample-rate hypothesis: {sample_rate_hz / 1e6:.3f} MSa/s, PRN {prn}."
        )

    def available_prns(self, with_tracking: bool = False) -> list[int]:
        """Return the currently known PRNs."""

        if with_tracking:
            return sorted(self.tracking_results_by_prn.keys())
        return sorted(set(self.acquisition_results_by_prn.keys()) | set(self.tracking_results_by_prn.keys()))

    def sync_session_from_ui(self) -> None:
        config = self.session_tab.get_session_config()
        config.prn = self.acquisition_tab.prn_spin.value()
        config.doppler_min = self.acquisition_tab.doppler_min_spin.value()
        config.doppler_max = self.acquisition_tab.doppler_max_spin.value()
        config.doppler_step = self.acquisition_tab.doppler_step_spin.value()
        config.integration_ms = self.acquisition_tab.integration_spin.value()
        config.spread_acquisition_blocks = self.acquisition_tab.spread_blocks_checkbox.isChecked()
        config.acquisition_segment_count = self.acquisition_tab.segment_count_spin.value()
        config.early_late_spacing_chips = self.tracking_tab.early_late_spin.value()
        config.dll_gain = self.tracking_tab.dll_gain_spin.value()
        config.pll_gain = self.tracking_tab.pll_gain_spin.value()
        config.fll_gain = self.tracking_tab.fll_gain_spin.value()
        self.session = config

    def load_file_dialog(self) -> None:
        """Open a file dialog and inspect the selected file."""

        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open complex64 IQ file",
            str(Path.cwd()),
            "Binary files (*.bin *.dat);;All files (*)",
        )
        if not selected:
            return
        self.session_tab.set_file_path(selected)
        self.inspect_file(selected)

    def inspect_file(self, file_path: str) -> None:
        """Inspect the current file and update metadata widgets."""

        previous_metadata_path = self.file_metadata.file_path if self.file_metadata is not None else None
        source_changed = self.demo_signal is not None or (
            previous_metadata_path is not None and previous_metadata_path != file_path
        )
        if source_changed:
            self.demo_signal = None
            self._clear_loaded_samples()
            self._clear_processing_results()

        self.sync_session_from_ui()
        metadata = inspect_complex64_file(file_path, self.session.sample_rate)
        self.file_metadata = metadata
        self.session.file_path = file_path
        self.session_tab.set_metadata(metadata)
        if self.session_tab.preload_enabled() and self._should_prefer_windowed_loading(metadata.file_size_bytes):
            self.session_tab.preload_checkbox.setChecked(False)
            self.append_log(
                f"{metadata.file_name} is {metadata.file_size_bytes / (1024 ** 3):.2f} GiB, "
                "so RAM preload was turned off automatically. The app will use the selected window instead."
            )
        self.update_ram_status()
        self.append_log(
            f"Loaded metadata for {metadata.file_name}: {metadata.total_samples:,} complex64 samples."
        )
        if metadata.common_rate_duration_hints:
            preferred_hints = []
            for label in ("2.000 MSa/s", "2.046 MSa/s", "4.092 MSa/s", "6.000 MSa/s", "6.061 MSa/s"):
                if label in metadata.common_rate_duration_hints:
                    preferred_hints.append(f"{label} -> {metadata.common_rate_duration_hints[label]:.2f} s")
            if preferred_hints:
                self.append_log("Common GNSS duration hints: " + ", ".join(preferred_hints) + ".")

    def _current_signature(self) -> tuple[str | None, bool, int, int]:
        """Return a signature for the currently loaded source or window policy."""

        return (
            self.session.file_path,
            self.session_tab.preload_enabled(),
            int(self.session.start_sample),
            int(self.session.sample_count),
        )

    def current_window_samples(self) -> np.ndarray:
        """Return the selected analysis window from the in-memory source buffer."""

        if self.current_samples.size == 0:
            return np.empty(0, dtype=np.complex64)
        if not self.session_tab.preload_enabled():
            return self.current_samples
        start = max(0, int(self.session.start_sample))
        stop = min(self.current_samples.size, start + int(self.session.sample_count))
        if stop <= start:
            return np.empty(0, dtype=np.complex64)
        return self.current_samples[start:stop]

    def ensure_samples(self) -> np.ndarray:
        """Ensure the full source is loaded into RAM and return the selected window."""

        self.sync_session_from_ui()
        self.update_ram_status()
        signature = self._current_signature()
        if self.demo_signal is not None:
            if self.current_samples_signature != signature:
                if self.session_tab.preload_enabled():
                    self.current_samples = self.demo_signal.samples
                else:
                    self.current_samples = self.demo_signal.samples[
                        self.session.start_sample : self.session.start_sample + self.session.sample_count
                    ]
                self.current_samples_signature = signature
            return self.current_window_samples()

        if not self.session.file_path:
            raise ValueError("No file selected.")

        if self.current_samples_signature != signature or self.current_samples.size == 0:
            bytes_to_load = self._planned_ram_bytes()
            if not self._confirm_large_ram_load(bytes_to_load):
                self.append_log("RAM load canceled by user.")
                return np.empty(0, dtype=np.complex64)
            self.append_log(
                f"Loading IQ data into RAM: "
                f"({bytes_to_load / (1024 ** 2):.1f} MiB)."
            )
            if self.session_tab.preload_enabled():
                self.current_samples = self._load_with_progress_dialog(
                    "Loading complete IQ file into RAM...",
                    lambda progress: load_complex64_file_with_progress(
                        self.session.file_path,
                        progress_callback=progress,
                    ),
                )
            else:
                self.current_samples = self._load_with_progress_dialog(
                    "Loading selected window into RAM...",
                    lambda progress: load_complex64_samples_with_progress(
                        self.session.file_path,
                        start_sample=self.session.start_sample,
                        sample_count=self.session.sample_count,
                        progress_callback=progress,
                    ),
                )
            self.current_samples_signature = signature
            self.append_log(
                f"RAM load finished. {self.current_samples.size:,} complex samples are now available in memory."
            )
        return self.current_window_samples()

    def load_display_samples(self, max_samples: int = 500_000) -> np.ndarray:
        """Return a bounded display window from the in-memory source buffer."""

        self.sync_session_from_ui()
        window = self.current_window_samples()
        if window.size == 0:
            window = self.ensure_samples()
        count = min(window.size, int(max_samples))
        self.current_display_samples = window[:count]
        if self.session.sample_count > count:
            source_text = (
                "The full file is still resident in RAM."
                if self.session_tab.preload_enabled()
                else "Only the selected window was loaded."
            )
            self.append_log(
                f"Display window capped to {count:,} samples for responsiveness. "
                f"{source_text}"
            )
        return self.current_display_samples

    def load_acquisition_samples(self) -> np.ndarray:
        """Return only the short segment needed for acquisition from RAM."""

        self.sync_session_from_ui()
        samples_per_ms = int(round(self.session.sample_rate * 1e-3))
        count = max(samples_per_ms, samples_per_ms * int(self.session.integration_ms))
        window = self.current_window_samples()
        if window.size == 0:
            window = self.ensure_samples()
        if self.session.spread_acquisition_blocks or self.session.acquisition_segment_count > 1:
            return window
        return window[:count]

    def preview_selected_window(self) -> None:
        """Load the selected file window and refresh passive plots."""

        self.sync_session_from_ui()
        if self.session.file_path and (self.file_metadata is None or self.file_metadata.file_path != self.session.file_path):
            self.inspect_file(self.session.file_path)
        samples_loaded = self.ensure_samples()
        if samples_loaded.size == 0:
            return
        samples = self.load_display_samples()
        self.update_passive_views(samples)
        self.append_log(f"Preview loaded: {samples.size:,} samples from the selected window.")

    def update_passive_views(self, samples: np.ndarray) -> None:
        """Refresh plots that only depend on the selected sample window."""

        self.raw_tab.update_signal(samples, self.session.sample_rate)
        acquisition = self.acquisition_results_by_prn.get(self.selected_prn) if self.selected_prn is not None else self.acquisition_result
        self.spectrum_tab.update_signal(samples, self.session.sample_rate, session=self.session, acquisition=acquisition)
        self.iq_tab.set_sources({"Raw IQ": samples})

    def refresh_satellite_views(self) -> None:
        """Refresh tracking and navigation tabs for the currently selected PRN."""

        if self.selected_prn is None:
            return
        acquisition = self.acquisition_results_by_prn.get(self.selected_prn)
        tracking = self.tracking_results_by_prn.get(self.selected_prn)
        bit_result = self.bit_results_by_prn.get(self.selected_prn)
        nav_result = self.nav_results_by_prn.get(self.selected_prn)

        if acquisition is not None:
            all_results = sorted(
                self.acquisition_results_by_prn.values(),
                key=acquisition_rank_key,
                reverse=True,
            )
            self.acquisition_tab.update_result(
                acquisition,
                all_results,
                tracked_prns=set(self.tracking_results_by_prn),
                decoded_prns=set(self.nav_results_by_prn),
            )
        self.tracking_tab.set_available_prns(self.available_prns(), self.selected_prn)
        self.navigation_tab.set_available_prns(self.available_prns(with_tracking=True), self.selected_prn)
        if tracking is not None:
            self.tracking_tab.update_state(tracking, acquisition=acquisition, bit_result=bit_result, nav_result=nav_result)
        if bit_result is not None and nav_result is not None:
            self.navigation_tab.update_results(bit_result, nav_result, self.selected_prn, acquisition=acquisition, tracking=tracking)
        self.learning_tab.update_pipeline(self.selected_prn, acquisition, tracking, bit_result, nav_result)

    def _set_tab_task_failed(self, tab, message: str) -> None:
        """Show a failed task state on one processing tab."""

        tab.set_task_message(message)
        tab.set_task_progress(0)

    def _set_tab_task_finished(self, tab, message: str) -> None:
        """Show a completed task state on one processing tab."""

        tab.set_task_message(message)
        tab.set_task_progress(100)

    def _start_worker(
        self,
        worker: Worker,
        finished_handler,
        *,
        progress_handlers: list | None = None,
        log_handlers: list | None = None,
        error_handlers: list | None = None,
    ) -> None:
        self.session_tab.set_progress(0)
        worker.signals.progress.connect(self.session_tab.set_progress)
        for handler in progress_handlers or []:
            worker.signals.progress.connect(handler)
        worker.signals.log.connect(self.append_log)
        for handler in log_handlers or []:
            worker.signals.log.connect(handler)
        worker.signals.error.connect(self._handle_worker_error)
        for handler in error_handlers or []:
            worker.signals.error.connect(handler)
        worker.signals.finished.connect(finished_handler)
        self.thread_pool.start(worker)

    def _handle_worker_error(self, traceback_text: str) -> None:
        self.append_log("Worker failed. See traceback below.")
        self.session_tab.append_log(traceback_text)
        self.session_tab.set_progress(0)

    def start_acquisition(self) -> None:
        """Launch acquisition in a background worker."""

        self.sync_session_from_ui()
        self.acquisition_tab.set_task_message(f"Running acquisition for PRN {self.session.prn}.")
        self.acquisition_tab.set_task_progress(0)
        full_samples = self.ensure_samples()
        if full_samples.size == 0:
            return
        display_samples = full_samples[: min(full_samples.size, 500_000)]
        self.current_display_samples = display_samples
        self.update_passive_views(display_samples)
        samples = self.load_acquisition_samples()
        self.tabs.setCurrentWidget(self.acquisition_tab)
        worker = Worker(acquisition_from_session, samples, self.session)
        self._start_worker(
            worker,
            self._on_acquisition_finished,
            progress_handlers=[self.acquisition_tab.set_task_progress],
            log_handlers=[self.acquisition_tab.set_task_message],
            error_handlers=[lambda _trace: self._set_tab_task_failed(self.acquisition_tab, "Acquisition failed. Check File / Session log.")],
        )

    def start_search_center_sweep(self) -> None:
        """Try multiple IF / search-center hypotheses automatically."""

        self.sync_session_from_ui()
        self.acquisition_tab.set_task_message("Sweeping IF / search-center hypotheses.")
        self.acquisition_tab.set_task_progress(0)
        full_samples = self.ensure_samples()
        if full_samples.size == 0:
            return
        display_samples = full_samples[: min(full_samples.size, 500_000)]
        self.current_display_samples = display_samples
        self.update_passive_views(display_samples)
        samples = self.load_acquisition_samples()
        centers = self.acquisition_tab.build_search_centers()
        worker = Worker(
            sweep_search_centers_from_session,
            samples,
            self.session,
            centers,
            list(range(1, 33)),
        )
        self.tabs.setCurrentWidget(self.acquisition_tab)
        self._start_worker(
            worker,
            self._on_search_center_sweep_finished,
            progress_handlers=[self.acquisition_tab.set_task_progress],
            log_handlers=[self.acquisition_tab.set_task_message],
            error_handlers=[lambda _trace: self._set_tab_task_failed(self.acquisition_tab, "Search-center sweep failed. Check File / Session log.")],
        )

    def start_sample_rate_survey(self) -> None:
        """Try several common GNSS sample-rate hypotheses automatically."""

        self.sync_session_from_ui()
        self.acquisition_tab.set_task_message("Surveying common GNSS sample-rate hypotheses.")
        self.acquisition_tab.set_task_progress(0)
        full_samples = self.ensure_samples()
        if full_samples.size == 0:
            return
        display_samples = full_samples[: min(full_samples.size, 500_000)]
        self.current_display_samples = display_samples
        self.update_passive_views(display_samples)
        samples = self.current_samples if self.current_samples.size else self.ensure_samples()
        common_rates = list(COMMON_GNSS_SAMPLE_RATES)
        if self.session.sample_rate not in common_rates:
            common_rates.append(float(self.session.sample_rate))
        refinement_offsets = (-1_000.0, -750.0, -600.0, -500.0, -400.0, -250.0, 250.0, 400.0, 500.0, 600.0, 750.0, 1_000.0)
        refinement_bases = {float(self.session.sample_rate), 6_061_000.0}
        for base_rate in refinement_bases:
            if base_rate > 1_000_000.0:
                common_rates.extend(base_rate + offset for offset in refinement_offsets)
        common_rates = sorted(set(common_rates))
        worker = Worker(
            survey_sample_rates,
            samples,
            self.session,
            common_rates,
            list(range(1, 33)),
        )
        self.tabs.setCurrentWidget(self.acquisition_tab)
        self._start_worker(
            worker,
            self._on_sample_rate_survey_finished,
            progress_handlers=[self.acquisition_tab.set_task_progress],
            log_handlers=[self.acquisition_tab.set_task_message],
            error_handlers=[lambda _trace: self._set_tab_task_failed(self.acquisition_tab, "Sample-rate survey failed. Check File / Session log.")],
        )

    def scan_all_prns(self) -> None:
        """Run acquisition over PRNs 1..32 for a more intuitive satellite overview."""

        self.sync_session_from_ui()
        try:
            prns = self.acquisition_tab.selected_scan_prns()
        except ValueError as exc:
            message = f"Invalid PRN scan list: {exc}"
            self.acquisition_tab.set_task_message(message)
            self.append_log(message)
            QtWidgets.QMessageBox.warning(self, "PRN scan list", message)
            return
        prn_text = ",".join(str(prn) for prn in prns)
        self.acquisition_tab.set_task_message(f"Scanning PRNs {prn_text}.")
        self.acquisition_tab.set_task_progress(0)
        full_samples = self.ensure_samples()
        if full_samples.size == 0:
            return
        display_samples = full_samples[: min(full_samples.size, 500_000)]
        self.current_display_samples = display_samples
        self.update_passive_views(display_samples)
        samples = self.load_acquisition_samples()
        worker = Worker(scan_prns_from_session, samples, self.session, prns)
        self.tabs.setCurrentWidget(self.acquisition_tab)
        self._start_worker(
            worker,
            self._on_acquisition_scan_finished,
            progress_handlers=[self.acquisition_tab.set_task_progress],
            log_handlers=[self.acquisition_tab.set_task_message],
            error_handlers=[lambda _trace: self._set_tab_task_failed(self.acquisition_tab, "PRN scan failed. Check File / Session log.")],
        )

    def _on_acquisition_finished(self, result: AcquisitionResult) -> None:
        self.acquisition_result = result
        self.acquisition_results_by_prn[result.prn] = result
        self.selected_prn = result.prn
        self.refresh_satellite_views()
        if self.current_display_samples.size:
            self.spectrum_tab.update_signal(
                self.current_display_samples,
                self.session.sample_rate,
                session=self.session,
                acquisition=result,
            )
        self.session_tab.set_progress(100)
        near_edge = abs(result.best_candidate.doppler_hz) >= (max(abs(self.session.doppler_min), abs(self.session.doppler_max)) - self.session.doppler_step)
        if acquisition_result_is_plausible(result):
            self.append_log(
                f"Acquisition finished. PRN {result.prn} repeats across {result.consistent_segments} segments, "
                f"so it is a stronger acquisition candidate than a one-off peak. "
                f"The raw metric is {result.best_candidate.metric:.2f}; run tracking before treating it as a satellite."
            )
        elif result.consistent_segments >= 3:
            self.append_log(
                f"Acquisition finished. PRN {result.prn} repeats across {result.consistent_segments} segments, "
                f"but the raw metric is still only {result.best_candidate.metric:.2f}, so this remains {acquisition_interpretation(result)}."
            )
        elif not acquisition_metric_is_strong(result.best_candidate.metric):
            self.append_log(
                f"Acquisition finished, but the best metric is only {result.best_candidate.metric:.2f} at "
                f"{result.best_candidate.carrier_frequency_hz:.1f} Hz "
                f"(relative Doppler {result.best_candidate.doppler_hz:+.1f} Hz). "
                "That is weak, so the sample rate / IF assumptions may still be wrong or the signal may be faint."
            )
        elif near_edge:
            self.append_log(
                "Acquisition finished, but the best peak sits near the search-band edge. "
                "That often means the IF / search center should be adjusted."
            )
        else:
            self.append_log("Acquisition finished.")
        self._set_tab_task_finished(self.acquisition_tab, f"Acquisition finished for PRN {result.prn}.")
        self.tabs.setCurrentWidget(self.acquisition_tab)

    def _on_acquisition_scan_finished(self, results: list[AcquisitionResult]) -> None:
        self.acquisition_results_by_prn.update({item.prn: item for item in results})
        if results:
            self.acquisition_result = results[0]
            self.selected_prn = results[0].prn
            self.refresh_satellite_views()
            if self.current_display_samples.size:
                self.spectrum_tab.update_signal(
                    self.current_display_samples,
                    self.session.sample_rate,
                    session=self.session,
                    acquisition=results[0],
                )
        self.session_tab.set_progress(100)
        if results and acquisition_result_is_plausible(results[0]):
            self.append_log(
                f"PRN scan finished. Best hypothesis is PRN {results[0].prn} with repeated hits in "
                f"{results[0].consistent_segments} segments at {results[0].best_candidate.carrier_frequency_hz:.1f} Hz. "
                "That kind of repetition is useful acquisition evidence, but tracking is still required before calling it a satellite."
            )
        elif results and results[0].consistent_segments >= 3:
            self.append_log(
                f"PRN scan finished. The strongest repeated pattern is PRN {results[0].prn}, "
                f"but its metric is still only {results[0].best_candidate.metric:.2f}, so it remains {acquisition_interpretation(results[0])}. "
                "This looks more like weak structure than a trustworthy lock candidate."
            )
        elif results and not acquisition_metric_is_strong(results[0].best_candidate.metric):
            self.append_log(
                f"PRN scan finished. Best hypothesis is PRN {results[0].prn} at {results[0].best_candidate.carrier_frequency_hz:.1f} Hz "
                f"(relative Doppler {results[0].best_candidate.doppler_hz:+.1f} Hz) with metric {results[0].best_candidate.metric:.2f}, "
                "which is still weak. The file duration is probably not the main issue; check sample rate, IF/baseband assumption, and signal quality."
            )
        elif results and abs(results[0].best_candidate.doppler_hz) >= (max(abs(self.session.doppler_min), abs(self.session.doppler_max)) - self.session.doppler_step):
            self.append_log(
                "PRN scan finished, but the best hypothesis is near the edge of the Doppler band. "
                "That suggests the IF / search center may need to be moved."
            )
        else:
            self.append_log(f"PRN scan finished. Ranked {len(results)} satellite hypotheses.")
        if results:
            self._set_tab_task_finished(
                self.acquisition_tab,
                f"PRN scan finished. Best current hypothesis is PRN {results[0].prn}.",
            )
        else:
            self._set_tab_task_finished(self.acquisition_tab, "PRN scan finished with no ranked hypotheses.")
        self.tabs.setCurrentWidget(self.acquisition_tab)

    def _on_search_center_sweep_finished(self, result: SearchCenterSweepResult) -> None:
        self.search_center_sweep_result = result
        self.acquisition_tab.update_sweep_results(result.entries)
        if result.entries:
            best_entry = result.entries[0]
            self.apply_search_center_selection(best_entry.search_center_hz, best_entry.best_result.prn)
            self.acquisition_result = best_entry.best_result
            self.acquisition_results_by_prn[best_entry.best_result.prn] = best_entry.best_result
            self.refresh_satellite_views()
            if acquisition_metric_is_strong(best_entry.best_result.best_candidate.metric):
                self.append_log(
                    f"Search-center sweep finished. Best center is {best_entry.search_center_hz:.1f} Hz "
                    f"with PRN {best_entry.best_result.prn} and metric {best_entry.best_result.best_candidate.metric:.2f}."
                )
            else:
                self.append_log(
                    f"Search-center sweep finished. Best center is {best_entry.search_center_hz:.1f} Hz "
                    f"with PRN {best_entry.best_result.prn}, but the metric is still only "
                    f"{best_entry.best_result.best_candidate.metric:.2f}. "
                    "That means the IF / center sweep did not uncover a strong acquisition."
                )
        else:
            self.append_log("Search-center sweep finished, but no candidates were produced.")
        self.session_tab.set_progress(100)
        if result.entries:
            self._set_tab_task_finished(
                self.acquisition_tab,
                f"Search-center sweep finished. Best center {result.entries[0].search_center_hz:.1f} Hz.",
            )
        else:
            self._set_tab_task_finished(self.acquisition_tab, "Search-center sweep finished with no candidates.")
        self.tabs.setCurrentWidget(self.acquisition_tab)

    def _on_sample_rate_survey_finished(self, result: SampleRateSurveyResult) -> None:
        self.sample_rate_survey_result = result
        self.acquisition_tab.update_sample_rate_survey(result.entries)
        if result.entries:
            best_entry = result.entries[0]
            self.session_tab.sample_rate_spin.setValue(best_entry.sample_rate_hz)
            self.sync_session_from_ui()
            if self.session.file_path and self.session.file_path != "<demo>":
                self.inspect_file(self.session.file_path)
            self.acquisition_results_by_prn = {item.prn: item for item in best_entry.all_results}
            self.acquisition_result = best_entry.best_result
            self.selected_prn = best_entry.best_result.prn
            self.refresh_satellite_views()
            if self.current_display_samples.size:
                self.spectrum_tab.update_signal(
                    self.current_display_samples,
                    self.session.sample_rate,
                    session=self.session,
                    acquisition=best_entry.best_result,
                )
            if acquisition_metric_is_strong(best_entry.best_result.best_candidate.metric):
                self.append_log(
                    f"Sample-rate survey finished. Best hypothesis is {best_entry.sample_rate_hz / 1e6:.3f} MSa/s "
                    f"with PRN {best_entry.best_result.prn}, metric {best_entry.best_result.best_candidate.metric:.2f}, "
                    f"consistent segments {best_entry.best_result.consistent_segments}."
                )
            else:
                self.append_log(
                    f"Sample-rate survey finished. The least-bad hypothesis is {best_entry.sample_rate_hz / 1e6:.3f} MSa/s "
                    f"with PRN {best_entry.best_result.prn}, metric {best_entry.best_result.best_candidate.metric:.2f}, "
                    f"consistent segments {best_entry.best_result.consistent_segments}. "
                    "That is still weak, so the sample-rate survey did not find a convincing lock."
                )
        else:
            self.append_log("Sample-rate survey finished, but no candidates were produced.")
        self.session_tab.set_progress(100)
        if result.entries:
            self._set_tab_task_finished(
                self.acquisition_tab,
                f"Sample-rate survey finished. Best rate {result.entries[0].sample_rate_hz / 1e6:.3f} MSa/s.",
            )
        else:
            self._set_tab_task_finished(self.acquisition_tab, "Sample-rate survey finished with no candidates.")
        self.tabs.setCurrentWidget(self.acquisition_tab)

    def start_tracking(self) -> None:
        """Launch tracking from the current acquisition result."""

        selected_prn = self.selected_prn or self.acquisition_tab.prn_spin.value()
        acquisition = self.acquisition_results_by_prn.get(selected_prn) or self.acquisition_result
        if acquisition is None:
            self.tracking_tab.set_task_message("Run acquisition before tracking.")
            self.tracking_tab.set_task_progress(0)
            QtWidgets.QMessageBox.warning(self, "Tracking", "Run acquisition before tracking.")
            return
        self.sync_session_from_ui()
        self.session.prn = selected_prn
        self.tracking_tab.set_task_message(f"Tracking PRN {selected_prn}.")
        self.tracking_tab.set_task_progress(0)
        start_offset = int(acquisition.best_candidate.segment_start_sample)
        self.tabs.setCurrentWidget(self.tracking_tab)
        if (
            self.session.file_path
            and self.session.file_path != "<demo>"
            and not self.session_tab.preload_enabled()
        ):
            absolute_start = int(self.session.start_sample) + start_offset
            worker = Worker(track_file, self.session.file_path, absolute_start, self.session, acquisition)
        else:
            source_samples = self.ensure_samples()
            samples = source_samples[start_offset:] if start_offset < source_samples.size else source_samples
            if samples.size == 0:
                return
            worker = Worker(track_signal, samples, self.session, acquisition)
        self._start_worker(
            worker,
            self._on_tracking_finished,
            progress_handlers=[self.tracking_tab.set_task_progress],
            log_handlers=[self.tracking_tab.set_task_message],
            error_handlers=[lambda _trace: self._set_tab_task_failed(self.tracking_tab, "Tracking failed. Check File / Session log.")],
        )

    def _on_tracking_finished(self, result: TrackingState) -> None:
        self.tracking_state = result
        self.tracking_results_by_prn[result.prn] = result
        self.selected_prn = result.prn
        raw_source = self.current_display_samples if self.current_display_samples.size else self.current_samples
        sources = {"Raw IQ": raw_source}
        sources.update(result.iq_views)
        self.iq_tab.set_sources(sources)
        self.refresh_satellite_views()
        self.session_tab.set_progress(100)
        if result.lock_detected:
            self.append_log(f"Tracking finished. PRN {result.prn} produced a lock indication.")
        else:
            self.append_log(
                f"Tracking finished. PRN {result.prn} did not lock, so the acquisition candidate is not confirmed."
            )
        self._set_tab_task_finished(self.tracking_tab, f"Tracking finished for PRN {result.prn}.")
        self.tabs.setCurrentWidget(self.tracking_tab)

    def decode_navigation(self) -> None:
        """Decode bits and LNAV framing from the current tracking state."""

        selected_prn = self.selected_prn
        tracking = self.tracking_results_by_prn.get(selected_prn) if selected_prn is not None else self.tracking_state
        if tracking is None:
            self.navigation_tab.set_task_message("Run tracking before decoding.")
            self.navigation_tab.set_task_progress(0)
            QtWidgets.QMessageBox.warning(self, "Navigation", "Run tracking before bit decoding.")
            return
        self.navigation_tab.set_task_message(f"Decoding PRN {tracking.prn}.")
        self.navigation_tab.set_task_progress(0)
        self.pending_navigation_prn = tracking.prn
        self.tabs.setCurrentWidget(self.navigation_tab)
        worker = Worker(
            decode_navigation_from_tracking,
            tracking,
            bit_source=self.navigation_tab.bit_source_mode(),
        )
        self._start_worker(
            worker,
            self._on_navigation_finished,
            progress_handlers=[self.navigation_tab.set_task_progress],
            log_handlers=[self.navigation_tab.set_task_message],
            error_handlers=[lambda _trace: self._set_tab_task_failed(self.navigation_tab, "Navigation decode failed. Check File / Session log.")],
        )

    def _on_navigation_finished(self, result: tuple[BitDecisionResult, NavigationDecodeResult]) -> None:
        bit_result, nav_result = result
        tracking_prn = self.pending_navigation_prn
        self.pending_navigation_prn = None
        tracking = self.tracking_results_by_prn.get(tracking_prn) if tracking_prn is not None else self.tracking_state
        if tracking is None:
            self._set_tab_task_failed(self.navigation_tab, "Navigation decode finished, but tracking state is no longer available.")
            self.session_tab.set_progress(0)
            return
        self.bit_result = bit_result
        self.nav_result = nav_result
        self.bit_results_by_prn[tracking.prn] = self.bit_result
        self.nav_results_by_prn[tracking.prn] = self.nav_result
        self.selected_prn = tracking.prn
        self.refresh_satellite_views()
        self.session_tab.set_progress(100)
        self.append_log("Bit extraction and LNAV framing finished.")
        self._set_tab_task_finished(
            self.navigation_tab,
            f"Navigation decode finished for PRN {tracking.prn}: "
            f"{len(nav_result.preamble_indices)} preambles, {nav_result.parity_ok_count} parity-valid words.",
        )
        self.tabs.setCurrentWidget(self.navigation_tab)

    def generate_demo(self) -> None:
        """Generate a synthetic GPS-like signal and refresh the preview."""

        self.sync_session_from_ui()
        demo = generate_demo_signal(
            sample_rate=self.session.sample_rate,
            duration_s=max(0.2, self.session.sample_count / self.session.sample_rate),
            prn=self.session.prn,
        )
        self._clear_processing_results()
        self.demo_signal = demo
        self.current_samples = demo.samples
        self.current_display_samples = demo.samples[: min(500_000, demo.samples.size)]
        self.current_samples_signature = self._current_signature()
        self.selected_prn = demo.prn
        self.file_metadata = FileMetadata(
            file_path="<demo>",
            file_name="synthetic_demo",
            file_size_bytes=int(demo.samples.nbytes),
            data_type="complex64",
            endianness="little",
            sample_rate_hz=float(demo.sample_rate),
            total_samples=demo.samples.size,
            estimated_duration_s=demo.samples.size / demo.sample_rate,
            common_rate_duration_hints=common_sample_rate_hints(demo.samples.size),
            preview_stats={
                "rms": float(np.sqrt(np.mean(np.abs(demo.samples) ** 2))),
                "mean_mag": float(np.mean(np.abs(demo.samples))),
            },
            preview_samples=demo.samples[:8_192],
        )
        self.session_tab.set_file_path("<demo>")
        self.session_tab.set_metadata(self.file_metadata)
        self.update_ram_status()
        self.update_passive_views(self.current_display_samples)
        self.append_log(
            f"Generated demo signal for PRN {demo.prn}, Doppler {demo.doppler_hz:.1f} Hz."
        )

    def start_benchmark(self) -> None:
        """Run the local suitability benchmark in a background worker."""

        self.sync_session_from_ui()
        if self.session.file_path and self.session.file_path != "<demo>":
            if self.file_metadata is None or self.file_metadata.file_path != self.session.file_path:
                self.inspect_file(self.session.file_path)
        worker = Worker(run_benchmark, self.session)
        self._start_worker(worker, self._on_benchmark_finished)

    def _on_benchmark_finished(self, result: BenchmarkResult) -> None:
        self.benchmark_tab.update_result(result)
        self.session_tab.set_progress(100)
        self.append_log("Benchmark finished.")
        self.tabs.setCurrentWidget(self.benchmark_tab)
