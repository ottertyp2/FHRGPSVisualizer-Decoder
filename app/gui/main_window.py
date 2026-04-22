"""Main application window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

from app.dsp.acquisition import acquisition_from_session
from app.dsp.benchmark import run_benchmark
from app.dsp.bitsync import extract_navigation_bits
from app.dsp.demo import generate_demo_signal
from app.dsp.io import Complex64FileSource, inspect_complex64_file, load_complex64_samples
from app.dsp.navdecode import decode_navigation_bits
from app.dsp.tracking import track_file, track_signal
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
        self.demo_signal: DemoSignalResult | None = None
        self.acquisition_result: AcquisitionResult | None = None
        self.tracking_state: TrackingState | None = None
        self.bit_result: BitDecisionResult | None = None
        self.nav_result: NavigationDecodeResult | None = None

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

    def _load_default_file_if_present(self) -> None:
        default_file = Path.cwd() / "test1.bin"
        if default_file.exists():
            self.session_tab.set_file_path(str(default_file))
            self.append_log(f"Found default IQ file: {default_file}")

    def append_log(self, message: str) -> None:
        self.session_tab.append_log(message)
        self.statusBar().showMessage(message, 5_000)

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
        self.append_log(
            f"Loaded metadata for {metadata.file_name}: {metadata.total_samples:,} complex64 samples."
        )

    def ensure_samples(self) -> np.ndarray:
        """Return the currently selected analysis samples, loading them if needed."""

        self.sync_session_from_ui()
        if self.demo_signal is not None:
            self.current_samples = self.demo_signal.samples[
                self.session.start_sample : self.session.start_sample + self.session.sample_count
            ]
            return self.current_samples

        if not self.session.file_path:
            raise ValueError("No file selected.")

        self.current_samples = load_complex64_samples(
            self.session.file_path,
            self.session.start_sample,
            self.session.sample_count,
        )
        return self.current_samples

    def load_display_samples(self, max_samples: int = 500_000) -> np.ndarray:
        """Load a bounded sample window for plotting so large files stay responsive."""

        self.sync_session_from_ui()
        count = min(int(self.session.sample_count), int(max_samples))
        if self.demo_signal is not None:
            self.current_display_samples = self.demo_signal.samples[
                self.session.start_sample : self.session.start_sample + count
            ]
        else:
            if not self.session.file_path:
                raise ValueError("No file selected.")
            source = Complex64FileSource(self.session.file_path)
            self.current_display_samples = source.read_window(self.session.start_sample, count)
        if self.session.sample_count > count:
            self.append_log(
                f"Display window capped to {count:,} samples for responsiveness. "
                "Processing can still stream over a longer range."
            )
        return self.current_display_samples

    def load_acquisition_samples(self) -> np.ndarray:
        """Load only the short segment needed for acquisition."""

        self.sync_session_from_ui()
        samples_per_ms = int(round(self.session.sample_rate * 1e-3))
        count = max(samples_per_ms, samples_per_ms * int(self.session.integration_ms))
        if self.demo_signal is not None:
            return self.demo_signal.samples[self.session.start_sample : self.session.start_sample + count]
        if not self.session.file_path:
            raise ValueError("No file selected.")
        return load_complex64_samples(self.session.file_path, self.session.start_sample, count)

    def preview_selected_window(self) -> None:
        """Load the selected file window and refresh passive plots."""

        self.sync_session_from_ui()
        if self.session.file_path and (self.file_metadata is None or self.file_metadata.file_path != self.session.file_path):
            self.inspect_file(self.session.file_path)
        samples = self.load_display_samples()
        self.update_passive_views(samples)
        self.append_log(f"Preview loaded: {samples.size:,} samples from the selected window.")

    def update_passive_views(self, samples: np.ndarray) -> None:
        """Refresh plots that only depend on the selected sample window."""

        self.raw_tab.update_signal(samples)
        self.spectrum_tab.update_signal(samples, self.session.sample_rate)
        self.iq_tab.set_sources({"Raw IQ": samples})

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
        display_samples = self.load_display_samples()
        self.update_passive_views(display_samples)
        samples = self.load_acquisition_samples()
        worker = Worker(acquisition_from_session, samples, self.session)
        self._start_worker(worker, self._on_acquisition_finished)

    def _on_acquisition_finished(self, result: AcquisitionResult) -> None:
        self.acquisition_result = result
        self.acquisition_tab.update_result(result)
        self.session_tab.set_progress(100)
        self.append_log("Acquisition finished.")
        self.tabs.setCurrentWidget(self.acquisition_tab)

    def start_tracking(self) -> None:
        """Launch tracking from the current acquisition result."""

        if self.acquisition_result is None:
            QtWidgets.QMessageBox.warning(self, "Tracking", "Run acquisition before tracking.")
            return
        self.sync_session_from_ui()
        if self.demo_signal is not None:
            samples = self.ensure_samples()
            worker = Worker(track_signal, samples, self.session, self.acquisition_result)
        else:
            if not self.session.file_path:
                raise ValueError("No file selected.")
            worker = Worker(
                track_file,
                self.session.file_path,
                self.session.start_sample,
                self.session,
                self.acquisition_result,
            )
        self._start_worker(worker, self._on_tracking_finished)

    def _on_tracking_finished(self, result: TrackingState) -> None:
        self.tracking_state = result
        self.tracking_tab.update_state(result)
        raw_source = self.current_display_samples if self.current_display_samples.size else self.current_samples
        sources = {"Raw IQ": raw_source}
        sources.update(result.iq_views)
        self.iq_tab.set_sources(sources)
        self.session_tab.set_progress(100)
        self.append_log("Tracking finished.")
        self.tabs.setCurrentWidget(self.tracking_tab)

    def decode_navigation(self) -> None:
        """Decode bits and LNAV framing from the current tracking state."""

        if self.tracking_state is None:
            QtWidgets.QMessageBox.warning(self, "Navigation", "Run tracking before bit decoding.")
            return
        self.bit_result = extract_navigation_bits(self.tracking_state)
        self.nav_result = decode_navigation_bits(self.bit_result)
        self.navigation_tab.update_results(self.bit_result, self.nav_result)
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
        self.file_metadata = FileMetadata(
            file_path="<demo>",
            file_name="synthetic_demo",
            file_size_bytes=int(demo.samples.nbytes),
            data_type="complex64",
            endianness="little",
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
