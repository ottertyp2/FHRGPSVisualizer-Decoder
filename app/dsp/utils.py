"""Utility helpers for display-oriented DSP tasks."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from app.dsp.compute import get_cupy_module, parallel_ordered_map, resolve_compute_plan
from app.models import SessionConfig


TOOLTIPS: dict[str, str] = {
    "ca_code": "The C/A code is the 1023-chip spreading code that identifies one GPS satellite PRN.",
    "bpsk": "GPS L1 C/A uses BPSK: the carrier phase flips by 180 degrees to carry chips and data bits.",
    "spectrum": "Spread-spectrum signals look wide and weak in a normal FFT because their energy is distributed.",
    "despreading": "Despreading multiplies the received signal by the local PRN so the desired satellite adds coherently.",
    "waterfall": "A waterfall helps spot changing interference or Doppler trends, even when the GPS signal is weak.",
    "early_late": "Early, Prompt, and Late correlators compare alignment around the code phase to steer the DLL.",
    "nav_bits": "GPS LNAV carries one navigation bit every 20 ms, so 20 prompt integrations form one bit decision.",
    "iq_phase": "IQ phase is the angle of a complex sample. Doppler wipeoff can make that angle stop rotating, but it does not have to become 0 degrees.",
    "code_phase": "Code phase is a time offset inside the repeating 1 ms C/A code. It is not an angle in the IQ plane.",
    "doppler_bin": "A Doppler bin is one frequency hypothesis tested during acquisition. It is not a satellite identity.",
    "prn_code": "PRN / C/A code is the repeating spreading pattern that primarily separates GPS satellites on L1 C/A.",
    "carrier_wipeoff": "Carrier wipeoff multiplies by an opposite complex tone to remove an assumed Doppler rotation.",
    "integration_1ms": "1 ms integration sums one full C/A-code period into a prompt value after carrier wipeoff and despreading.",
    "lnav_20ms": "20 ms LNAV accumulation sums twenty 1 ms prompt values to decide one 50 bps navigation bit.",
}


def decimate_for_display(values: np.ndarray, max_points: int = 5_000) -> tuple[np.ndarray, int]:
    """Return a lightly decimated copy of an array for plotting."""

    if values.size <= max_points:
        return values, 1
    step = int(math.ceil(values.size / max_points))
    return values[::step], step


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average used for smooth display traces."""

    if window <= 1 or values.size == 0:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def get_window(window_name: str, length: int) -> np.ndarray:
    """Return a standard FFT window."""

    name = window_name.lower()
    if name == "hann":
        return np.hanning(length)
    if name == "hamming":
        return np.hamming(length)
    if name == "blackman":
        return np.blackman(length)
    return np.ones(length, dtype=float)


def _segment_power_db(segment: np.ndarray, fft_size: int) -> np.ndarray:
    """Return one FFT power row in dB."""

    fft_values = np.fft.fftshift(np.fft.fft(segment, n=fft_size))
    return 20.0 * np.log10(np.abs(fft_values) + 1e-12)


def _compute_power_rows(
    segments: list[np.ndarray],
    fft_size: int,
    session: SessionConfig | None,
) -> tuple[np.ndarray, str]:
    """Return FFT power rows plus the backend that produced them."""

    if not segments:
        return np.empty((0, fft_size)), "cpu"

    requested_backend = session.compute_backend if session is not None else "auto"
    requested_workers = session.max_workers if session is not None else 0
    gpu_enabled = session.gpu_enabled if session is not None else True
    plan = resolve_compute_plan(
        requested_backend,
        requested_workers,
        gpu_enabled=gpu_enabled,
        max_tasks=len(segments),
        prefer_gpu=True,
    )

    if plan.active_backend == "gpu":
        cupy = get_cupy_module()
        if cupy is not None:
            try:
                gpu_segments = cupy.asarray(np.asarray(segments, dtype=np.complex64))
                fft_values = cupy.fft.fft(gpu_segments, n=fft_size, axis=1)
                shifted = cupy.fft.fftshift(fft_values, axes=1)
                power = 20.0 * cupy.log10(cupy.abs(shifted) + 1e-12)
                return np.asarray(cupy.asnumpy(power)), "gpu"
            except Exception:
                pass

    cpu_workers = min(plan.selected_workers, len(segments))
    if cpu_workers <= 1:
        rows = [_segment_power_db(segment, fft_size) for segment in segments]
    else:
        rows = parallel_ordered_map(
            segments,
            lambda _index, segment: _segment_power_db(segment, fft_size),
            max_workers=cpu_workers,
        )
    return np.vstack(rows), "cpu"


def compute_spectrum(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
    window_name: str = "hann",
    average_count: int = 1,
    session: SessionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple averaged power spectrum."""

    if samples.size == 0:
        return np.empty(0), np.empty(0)

    fft_size = max(256, min(int(fft_size), samples.size))
    average_count = max(1, int(average_count))
    hop = fft_size
    window = get_window(window_name, fft_size)
    segments: list[np.ndarray] = []
    for idx in range(average_count):
        start = idx * hop
        stop = start + fft_size
        if stop > samples.size:
            break
        segments.append(samples[start:stop] * window)

    if not segments:
        segments.append(samples[:fft_size] * window)

    spectra, _backend = _compute_power_rows(segments, fft_size, session)
    spectrum = np.mean(spectra, axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    return freqs, spectrum


def compute_waterfall(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 1024,
    step: int | None = None,
    window_name: str = "hann",
    max_rows: int = 200,
    session: SessionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a compact waterfall image."""

    if samples.size == 0:
        return np.empty(0), np.empty(0), np.empty((0, 0))

    fft_size = max(256, min(int(fft_size), samples.size))
    step = step or fft_size // 2
    step = max(1, step)
    window = get_window(window_name, fft_size)
    segments: list[np.ndarray] = []
    times: list[float] = []
    for start in range(0, max(samples.size - fft_size, 1), step):
        stop = start + fft_size
        if stop > samples.size:
            break
        segments.append(samples[start:stop] * window)
        times.append(start / sample_rate)
        if len(segments) >= max_rows:
            break

    if not segments:
        segments.append(np.zeros(fft_size, dtype=np.complex64))
        times.append(0.0)

    rows, _backend = _compute_power_rows(segments, fft_size, session)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    return freqs, np.asarray(times), rows


def bits_to_str(bits: Iterable[int | bool]) -> str:
    """Return a compact 0/1 string."""

    return "".join("1" if int(bit) else "0" for bit in bits)
