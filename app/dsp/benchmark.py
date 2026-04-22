"""Benchmark helpers for large-file suitability and bottleneck analysis."""

from __future__ import annotations

import ctypes
from dataclasses import replace
import multiprocessing
import platform
import time
from pathlib import Path

import numpy as np

from app.dsp.acquisition import acquisition_from_session
from app.dsp.demo import generate_demo_signal
from app.dsp.io import Complex64FileSource
from app.dsp.tracking import track_file, track_signal
from app.models import BenchmarkComponentResult, BenchmarkResult, SessionConfig

BYTES_PER_COMPLEX64_SAMPLE = np.dtype(np.complex64).itemsize
TEN_GIB_BYTES = 10 * 1024 * 1024 * 1024


def _get_total_memory_bytes() -> int:
    """Return total physical memory on Windows when available."""

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
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
    except AttributeError:
        return 0
    return 0


def _system_info() -> dict[str, str]:
    """Collect lightweight system information."""

    total_mem = _get_total_memory_bytes()
    return {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": str(multiprocessing.cpu_count()),
        "python": platform.python_version(),
        "ram_gib": f"{total_mem / (1024 ** 3):.1f}" if total_mem else "unknown",
    }


def _component_result(
    name: str,
    elapsed_s: float,
    samples_processed: int,
    bytes_processed: int,
    current_rate: float,
    target_rate: float,
    detail: str,
) -> BenchmarkComponentResult:
    """Build a normalized benchmark result row."""

    elapsed_s = max(elapsed_s, 1e-9)
    throughput_samples_s = float(samples_processed / elapsed_s)
    throughput_mbytes_s = float(bytes_processed / elapsed_s / 1e6)
    byte_rate = float(bytes_processed / elapsed_s)
    estimated_time_for_10gb_s = float(TEN_GIB_BYTES / max(byte_rate, 1e-9))
    return BenchmarkComponentResult(
        name=name,
        elapsed_s=float(elapsed_s),
        samples_processed=int(samples_processed),
        bytes_processed=int(bytes_processed),
        throughput_samples_s=throughput_samples_s,
        throughput_mbytes_s=throughput_mbytes_s,
        realtime_factor_current=float(throughput_samples_s / max(current_rate, 1e-9)),
        realtime_factor_target=float(throughput_samples_s / max(target_rate, 1e-9)),
        estimated_time_for_10gb_s=estimated_time_for_10gb_s,
        detail=detail,
    )


def run_benchmark(
    session: SessionConfig,
    progress_callback=None,
    log_callback=None,
) -> BenchmarkResult:
    """Benchmark file I/O and key DSP stages to estimate laptop suitability."""

    current_rate = float(session.sample_rate)
    target_rate = current_rate
    target_rate_label = f"{target_rate / 1e6:.3f} MSa/s"
    samples_per_ms = int(round(current_rate * 1e-3))
    benchmark_ms = 200
    compute_samples = max(samples_per_ms * benchmark_ms, samples_per_ms * 40)
    target_path = session.file_path if session.file_path and session.file_path != "<demo>" else None
    source = Complex64FileSource(target_path) if target_path else None

    if source is not None:
        available = max(0, source.total_samples - int(session.start_sample))
        compute_samples = min(compute_samples, max(samples_per_ms * 40, available))
        loaded = source.read_window(session.start_sample, compute_samples)
        benchmark_samples = loaded if loaded.size else generate_demo_signal(sample_rate=current_rate, duration_s=0.25, prn=session.prn).samples
    else:
        benchmark_samples = generate_demo_signal(
            sample_rate=current_rate,
            duration_s=max(0.25, compute_samples / current_rate),
            prn=session.prn,
        ).samples
        compute_samples = int(benchmark_samples.size)

    compute_samples = min(compute_samples, int(benchmark_samples.size))
    benchmark_samples = benchmark_samples[:compute_samples]
    components: list[BenchmarkComponentResult] = []

    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    log("Benchmark started. Measuring I/O and DSP throughput.")
    if progress_callback:
        progress_callback(5)

    preview_count = min(compute_samples, max(250_000, samples_per_ms * 20))
    if source is not None:
        start = time.perf_counter()
        preview = source.read_window(session.start_sample, preview_count)
        elapsed = time.perf_counter() - start
        components.append(
            _component_result(
                "Windowed file read",
                elapsed,
                preview.size,
                preview.nbytes,
                current_rate,
                target_rate,
                "Bounded memmap window copy used for plots and previews.",
            )
        )
    else:
        start = time.perf_counter()
        preview = np.array(benchmark_samples[:preview_count], copy=True)
        elapsed = time.perf_counter() - start
        components.append(
            _component_result(
                "Window copy in memory",
                elapsed,
                preview.size,
                preview.nbytes,
                current_rate,
                target_rate,
                "Copy cost for preview windows when data is already in memory.",
            )
        )
    log(f"{components[-1].name}: {components[-1].throughput_mbytes_s:.1f} MB/s.")

    if progress_callback:
        progress_callback(20)

    if source is not None:
        block_count = max(20, min(200, compute_samples // max(samples_per_ms, 1)))
        start = time.perf_counter()
        streamed_samples = 0
        for block in source.iter_blocks(session.start_sample, samples_per_ms, block_count):
            streamed_samples += int(block.size)
            _ = float(np.abs(block).mean())
        elapsed = time.perf_counter() - start
        components.append(
            _component_result(
                "Sequential file streaming",
                elapsed,
                streamed_samples,
                streamed_samples * BYTES_PER_COMPLEX64_SAMPLE,
                current_rate,
                target_rate,
                "1 ms block streaming similar to long-recording tracking reads.",
            )
        )
        log(f"Sequential file streaming: {components[-1].throughput_mbytes_s:.1f} MB/s.")

    if progress_callback:
        progress_callback(35)

    fft_samples = benchmark_samples[: min(benchmark_samples.size, max(65_536, samples_per_ms * 20))]
    fft_size = 4096 if fft_samples.size >= 4096 else max(256, int(2 ** np.floor(np.log2(max(fft_samples.size, 256)))))
    fft_iterations = max(8, min(64, fft_samples.size // max(fft_size, 1)))
    start = time.perf_counter()
    fft_processed = 0
    for idx in range(fft_iterations):
        offset = (idx * fft_size) % max(fft_samples.size - fft_size + 1, 1)
        segment = fft_samples[offset : offset + fft_size]
        _ = np.fft.fft(segment)
        fft_processed += int(segment.size)
    elapsed = time.perf_counter() - start
    components.append(
        _component_result(
            "FFT core",
            elapsed,
            fft_processed,
            fft_processed * BYTES_PER_COMPLEX64_SAMPLE,
            current_rate,
            target_rate,
            f"{fft_iterations} x {fft_size}-point FFTs for spectrum and waterfall style work.",
        )
    )
    log(f"FFT core: {components[-1].throughput_samples_s / 1e6:.2f} MSa/s equivalent.")

    if progress_callback:
        progress_callback(50)

    acquisition_samples = benchmark_samples[: min(benchmark_samples.size, samples_per_ms * max(4, session.integration_ms))]
    acq_session = replace(session)
    start = time.perf_counter()
    acquisition = acquisition_from_session(acquisition_samples, acq_session)
    elapsed = time.perf_counter() - start
    doppler_bins = int((session.doppler_max - session.doppler_min) / max(session.doppler_step, 1)) + 1
    effective_samples = int(acquisition_samples.size * doppler_bins)
    components.append(
        _component_result(
            "Acquisition search",
            elapsed,
            effective_samples,
            acquisition_samples.nbytes,
            current_rate,
            target_rate,
            f"Search over {doppler_bins} Doppler bins with {session.integration_ms} ms coherent accumulation.",
        )
    )
    log(f"Acquisition search: {components[-1].realtime_factor_target:.2f}x realtime at {target_rate_label}.")

    if progress_callback:
        progress_callback(70)

    tracking_samples = benchmark_samples[: min(benchmark_samples.size, samples_per_ms * min(session.tracking_ms, benchmark_ms))]
    track_session = replace(session)
    track_session.tracking_ms = min(session.tracking_ms, int(tracking_samples.size // max(samples_per_ms, 1)))
    start = time.perf_counter()
    _ = track_signal(tracking_samples, track_session, acquisition)
    elapsed = time.perf_counter() - start
    components.append(
        _component_result(
            "Tracking compute",
            elapsed,
            tracking_samples.size,
            tracking_samples.nbytes,
            current_rate,
            target_rate,
            f"In-memory Early/Prompt/Late tracking over {track_session.tracking_ms} ms.",
        )
    )
    log(f"Tracking compute: {components[-1].realtime_factor_target:.2f}x realtime at {target_rate_label}.")

    if progress_callback:
        progress_callback(85)

    if source is not None:
        stream_tracking_ms = min(track_session.tracking_ms, max(40, compute_samples // max(samples_per_ms, 1)))
        stream_session = replace(session)
        stream_session.tracking_ms = int(stream_tracking_ms)
        start = time.perf_counter()
        _ = track_file(target_path, session.start_sample, stream_session, acquisition)
        elapsed = time.perf_counter() - start
        streamed_tracking_samples = int(stream_tracking_ms * samples_per_ms)
        components.append(
            _component_result(
                "Tracking with file streaming",
                elapsed,
                streamed_tracking_samples,
                streamed_tracking_samples * BYTES_PER_COMPLEX64_SAMPLE,
                current_rate,
                target_rate,
                "Tracking loop reading 1 ms blocks from the selected IQ file.",
            )
        )
        log(f"Tracking with file streaming: {components[-1].realtime_factor_target:.2f}x realtime at {target_rate_label}.")

    bottleneck = min(components, key=lambda item: item.realtime_factor_target) if components else None
    suitability = (
        f"Bottleneck: {bottleneck.name}. Estimated pipeline speed is "
        f"{bottleneck.realtime_factor_target:.2f}x realtime at {target_rate_label}."
        if bottleneck
        else "No benchmark components were measured."
    )
    if bottleneck:
        if bottleneck.realtime_factor_target >= 1.5:
            suitability += " This laptop looks comfortable for offline work and likely near or above realtime at the target rate."
        elif bottleneck.realtime_factor_target >= 1.0:
            suitability += f" This laptop looks usable for {target_rate_label}, but with limited headroom."
        else:
            suitability += f" This laptop will likely be slower than realtime at {target_rate_label} unless the workflow is narrowed or optimized."
    if source is not None:
        file_size_gib = Path(target_path).stat().st_size / (1024 ** 3)
        suitability += f" Current file size is {file_size_gib:.2f} GiB."

    if progress_callback:
        progress_callback(100)
    log("Benchmark finished.")

    return BenchmarkResult(
        target_sample_rate=target_rate,
        current_sample_rate=current_rate,
        components=components,
        bottleneck_name=bottleneck.name if bottleneck else "",
        suitability_summary=suitability,
        system_info=_system_info(),
    )
