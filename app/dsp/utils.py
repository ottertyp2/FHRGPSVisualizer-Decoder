"""Utility helpers for display-oriented DSP tasks."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


TOOLTIPS: dict[str, str] = {
    "ca_code": "The C/A code is the 1023-chip spreading code that identifies one GPS satellite PRN.",
    "bpsk": "GPS L1 C/A uses BPSK: the carrier phase flips by 180 degrees to carry chips and data bits.",
    "spectrum": "Spread-spectrum signals look wide and weak in a normal FFT because their energy is distributed.",
    "despreading": "Despreading multiplies the received signal by the local PRN so the desired satellite adds coherently.",
    "waterfall": "A waterfall helps spot changing interference or Doppler trends, even when the GPS signal is weak.",
    "early_late": "Early, Prompt, and Late correlators compare alignment around the code phase to steer the DLL.",
    "nav_bits": "GPS LNAV carries one navigation bit every 20 ms, so 20 prompt integrations form one bit decision.",
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


def compute_spectrum(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
    window_name: str = "hann",
    average_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple averaged power spectrum."""

    if samples.size == 0:
        return np.empty(0), np.empty(0)

    fft_size = max(256, min(int(fft_size), samples.size))
    average_count = max(1, int(average_count))
    hop = fft_size
    window = get_window(window_name, fft_size)
    spectra: list[np.ndarray] = []
    for idx in range(average_count):
        start = idx * hop
        stop = start + fft_size
        if stop > samples.size:
            break
        segment = samples[start:stop] * window
        fft_values = np.fft.fftshift(np.fft.fft(segment, n=fft_size))
        power_db = 20.0 * np.log10(np.abs(fft_values) + 1e-12)
        spectra.append(power_db)

    if not spectra:
        spectra.append(20.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[:fft_size] * window))) + 1e-12))

    spectrum = np.mean(np.vstack(spectra), axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    return freqs, spectrum


def compute_waterfall(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 1024,
    step: int | None = None,
    window_name: str = "hann",
    max_rows: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a compact waterfall image."""

    if samples.size == 0:
        return np.empty(0), np.empty(0), np.empty((0, 0))

    fft_size = max(256, min(int(fft_size), samples.size))
    step = step or fft_size // 2
    step = max(1, step)
    window = get_window(window_name, fft_size)
    rows: list[np.ndarray] = []
    times: list[float] = []
    for start in range(0, max(samples.size - fft_size, 1), step):
        stop = start + fft_size
        if stop > samples.size:
            break
        segment = samples[start:stop] * window
        fft_values = np.fft.fftshift(np.fft.fft(segment))
        rows.append(20.0 * np.log10(np.abs(fft_values) + 1e-12))
        times.append(start / sample_rate)
        if len(rows) >= max_rows:
            break

    if not rows:
        rows.append(np.zeros(fft_size))
        times.append(0.0)

    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    return freqs, np.asarray(times), np.vstack(rows)


def bits_to_str(bits: Iterable[int | bool]) -> str:
    """Return a compact 0/1 string."""

    return "".join("1" if int(bit) else "0" for bit in bits)
