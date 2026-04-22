"""Main application window."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

from app.dsp.acquisition import acquisition_from_session
from app.dsp.acquisition import scan_prns_from_session
from app.dsp.acquisition import sweep_search_centers_from_session
from app.dsp.benchmark import run_benchmark
from app.dsp.bitsync import extract_navigation_bits
from app.dsp.demo import generate_demo_signal
from app.dsp.io import (
    inspect_complex64_file,
    load_complex64_file_with_progress,
    load_complex64_samples_with_progress,
)
from app.dsp.navdecode import decode_navigation_bits
from app.dsp.tracking import track_signal
from app.gui.tabs.acquisition_tab import AcquisitionTab
from app.gui.tabs.benchmark_tab import BenchmarkTab
from app.gui.tabs.iq_tab import IQPlaneTab
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
        self.current_samples_signature: str | None = None
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
        self.selected_prn: int | None = None

        self.tabs = QtWidgets.QTabWidget()
        self.session_tab = SessionTab()
        self.raw_tab = RawSignalTab()
        self.spectrum_tab = SpectrumTab()
        self.iq_tab = IQPlaneTab()
        self.acquisition_tab = AcquisitionTab()
        self.tracking_tab = TrackingTab()
        self.navigation_tab = NavigationTab()
        self.benchmark_tab = BenchmarkTab()

        self.tabs.addTab(self.session_tab, "File / Session")
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
        self.acquisition_tab.track_selected_requested.connect(self.start_tracking)
        self.acquisition_tab.selection_changed.connect(self.set_selected_prn)
        self.acquisition_tab.sweep_selection_changed.connect(self.apply_search_center_selection)
        self.tracking_tab.track_requested.connect(self.start_tracking)
        self.tracking_tab.decode_requested.connect(self.decode_navigation)
        self.tracking_tab.selection_changed.connect(self.set_selected_prn)
        self.navigation_tab.decode_requested.connect(self.decode_navigation)
        self.navigation_tab.selection_changed.connect(self.set_selected_prn)

    def _load_default_file_if_present(self) -> None:
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
            return int(min(self.demo_signal.samples.size - self.session.start_sample, self.session.sample_count)) * np.dtype(np.complex64).itemsize

        if self.file_metadata is None:
            return int(self.session.sample_count) * np.dtype(np.complex64).itemsize
        if self.session_tab.preload_enabled():
            return int(self.file_metadata.file_size_bytes)
        return int(self.session.sample_count) * np.dtype(np.complex64).itemsize

    def update_ram_status(self) -> None:
        """Refresh the RAM status label in the session tab."""

        total_ram, avail_ram = self._memory_status()
        planned = self._planned_ram_bytes()
        mode = "full source preload" if self.session_tab.preload_enabled() else "selected window only"
        signal_mode = "baseband" if self.session.is_baseband else f"IF center {self.session.if_frequency_hz:.1f} Hz"
        total_text = f"{total_ram / (1024 ** 3):.1f} GiB" if total_ram else "unknown"
        avail_text = f"{avail_ram / (1024 ** 3):.1f} GiB" if avail_ram else "unknown"
        planned_text = f"{planned / (1024 ** 2):.1f} MiB"
        self.session_tab.set_ram_status(
            f"RAM status: mode={mode}, signal={signal_mode}, planned load={planned_text}, available RAM={avail_text}, total RAM={total_text}."
        )

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

        self.sync_session_from_ui()
        metadata = inspect_complex64_file(file_path, self.session.sample_rate)
        self.file_metadata = metadata
        self.session.file_path = file_path
        self.session_tab.set_metadata(metadata)
        self.update_ram_status()
        self.append_log(
            f"Loaded metadata for {metadata.file_name}: {metadata.total_samples:,} complex64 samples."
        )

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
            self.append_log(
                f"Display window capped to {count:,} samples for responsiveness. "
                "The full file is still resident in RAM."
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
            all_results = sorted(self.acquisition_results_by_prn.values(), key=lambda item: item.best_candidate.metric, reverse=True)
            self.acquisition_tab.update_result(acquisition, all_results)
        self.tracking_tab.set_available_prns(self.available_prns(), self.selected_prn)
        self.navigation_tab.set_available_prns(self.available_prns(with_tracking=True), self.selected_prn)
        if tracking is not None:
            self.tracking_tab.update_state(tracking, acquisition=acquisition, bit_result=bit_result, nav_result=nav_result)
        if bit_result is not None and nav_result is not None:
            self.navigation_tab.update_results(bit_result, nav_result, self.selected_prn, acquisition=acquisition, tracking=tracking)

    def _start_worker(self, worker: Worker, finished_handler) -> None:
        self.session_tab.set_progress(0)
        worker.signals.progress.connect(self.session_tab.set_progress)
        worker.signals.log.connect(self.append_log)
        worker.signals.error.connect(self._handle_worker_error)
        worker.signals.finished.connect(finished_handler)
        self.thread_pool.start(worker)

    def _handle_worker_error(self, traceback_text: str) -> None:
        self.append_log("Worker failed. See traceback below.")
        self.session_tab.append_log(traceback_text)
        self.session_tab.set_progress(0)

    def start_acquisition(self) -> None:
        """Launch acquisition in a background worker."""

        self.sync_session_from_ui()
        full_samples = self.ensure_samples()
        if full_samples.size == 0:
            return
        display_samples = full_samples[: min(full_samples.size, 500_000)]
        self.current_display_samples = display_samples
        self.update_passive_views(display_samples)
        samples = self.load_acquisition_samples()
        worker = Worker(acquisition_from_session, samples, self.session)
        self._start_worker(worker, self._on_acquisition_finished)

    def start_search_center_sweep(self) -> None:
        """Try multiple IF / search-center hypotheses automatically."""

        self.sync_session_from_ui()
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
        self._start_worker(worker, self._on_search_center_sweep_finished)

    def scan_all_prns(self) -> None:
        """Run acquisition over PRNs 1..32 for a more intuitive satellite overview."""

        self.sync_session_from_ui()
        full_samples = self.ensure_samples()
        if full_samples.size == 0:
            return
        display_samples = full_samples[: min(full_samples.size, 500_000)]
        self.current_display_samples = display_samples
        self.update_passive_views(display_samples)
        samples = self.load_acquisition_samples()
        worker = Worker(scan_prns_from_session, samples, self.session, list(range(1, 33)))
        self._start_worker(worker, self._on_acquisition_scan_finished)

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
        if result.best_candidate.metric < 6.0:
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
        if results and results[0].best_candidate.metric < 6.0:
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
            self.append_log(
                f"Search-center sweep finished. Best center is {best_entry.search_center_hz:.1f} Hz "
                f"with PRN {best_entry.best_result.prn} and metric {best_entry.best_result.best_candidate.metric:.2f}."
            )
        else:
            self.append_log("Search-center sweep finished, but no candidates were produced.")
        self.session_tab.set_progress(100)
        self.tabs.setCurrentWidget(self.acquisition_tab)

    def start_tracking(self) -> None:
        """Launch tracking from the current acquisition result."""

        selected_prn = self.selected_prn or self.acquisition_tab.prn_spin.value()
        acquisition = self.acquisition_results_by_prn.get(selected_prn) or self.acquisition_result
        if acquisition is None:
            QtWidgets.QMessageBox.warning(self, "Tracking", "Run acquisition before tracking.")
            return
        self.sync_session_from_ui()
        self.session.prn = selected_prn
        samples = self.ensure_samples()
        if samples.size == 0:
            return
        worker = Worker(track_signal, samples, self.session, acquisition)
        self._start_worker(worker, self._on_tracking_finished)

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
        self.append_log("Tracking finished.")
        self.tabs.setCurrentWidget(self.tracking_tab)

    def decode_navigation(self) -> None:
        """Decode bits and LNAV framing from the current tracking state."""

        selected_prn = self.selected_prn
        tracking = self.tracking_results_by_prn.get(selected_prn) if selected_prn is not None else self.tracking_state
        if tracking is None:
            QtWidgets.QMessageBox.warning(self, "Navigation", "Run tracking before bit decoding.")
            return
        self.bit_result = extract_navigation_bits(tracking)
        self.nav_result = decode_navigation_bits(self.bit_result)
        self.bit_results_by_prn[tracking.prn] = self.bit_result
        self.nav_results_by_prn[tracking.prn] = self.nav_result
        self.selected_prn = tracking.prn
        self.refresh_satellite_views()
        self.append_log("Bit extraction and LNAV framing finished.")
        self.tabs.setCurrentWidget(self.navigation_tab)

    def generate_demo(self) -> None:
        """Generate a synthetic GPS-like signal and refresh the preview."""

        self.sync_session_from_ui()
        demo = generate_demo_signal(
            sample_rate=self.session.sample_rate,
            duration_s=max(0.2, self.session.sample_count / self.session.sample_rate),
            prn=self.session.prn,
        )
        self.demo_signal = demo
        self.current_samples = demo.samples
        self.current_display_samples = demo.samples[: min(500_000, demo.samples.size)]
        self.current_samples_signature = self._current_signature()
        self.acquisition_results_by_prn.clear()
        self.tracking_results_by_prn.clear()
        self.bit_results_by_prn.clear()
        self.nav_results_by_prn.clear()
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
