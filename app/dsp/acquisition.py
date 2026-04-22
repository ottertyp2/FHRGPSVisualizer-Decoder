"""Simple acquisition search over Doppler and code phase."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.dsp.gps_ca import sample_ca_code
from app.models import AcquisitionCandidate, AcquisitionResult, SessionConfig


@dataclass(slots=True)
class AcquisitionConfig:
    """Compact acquisition settings."""

    sample_rate: float
    prn: int
    doppler_min: int
    doppler_max: int
    doppler_step: int
    integration_ms: int = 4


def acquire_signal(
    samples: np.ndarray,
    config: AcquisitionConfig,
    progress_callback=None,
    log_callback=None,
) -> AcquisitionResult:
    """Run a clear but not heavily optimized acquisition search."""

    if samples.size == 0:
        raise ValueError("No samples available for acquisition.")

    sample_rate = config.sample_rate
    samples_per_ms = int(round(sample_rate * 1e-3))
    integration_ms = max(1, int(config.integration_ms))
    usable = min(samples.size // samples_per_ms, integration_ms)
    if usable == 0:
        raise ValueError("Not enough data for one 1 ms acquisition block.")

    total = usable * samples_per_ms
    signal = samples[:total]
    local_code = sample_ca_code(config.prn, sample_rate, samples_per_ms)
    code_fft = np.conj(np.fft.fft(local_code))
    doppler_bins = np.arange(config.doppler_min, config.doppler_max + config.doppler_step, config.doppler_step)
    heatmap = np.zeros((doppler_bins.size, samples_per_ms), dtype=np.float32)
    time_vector = np.arange(samples_per_ms, dtype=np.float64) / sample_rate

    for row, doppler in enumerate(doppler_bins):
        metrics = np.zeros(samples_per_ms, dtype=np.float64)
        for block_index in range(usable):
            start = block_index * samples_per_ms
            stop = start + samples_per_ms
            block = signal[start:stop]
            wiped = block * np.exp(-1j * 2.0 * np.pi * doppler * time_vector)
            correlation = np.fft.ifft(np.fft.fft(wiped) * code_fft)
            metrics += np.abs(correlation) ** 2
        heatmap[row, :] = metrics.astype(np.float32)
        if progress_callback:
            progress_callback(int(100 * (row + 1) / doppler_bins.size))

    flat = heatmap.ravel()
    best_flat_index = int(np.argmax(flat))
    best_row, best_col = np.unravel_index(best_flat_index, heatmap.shape)
    noise_floor = float(np.mean(heatmap) + 1e-12)
    best_metric = float(heatmap[best_row, best_col] / noise_floor)
    best_code_phase = int((samples_per_ms - best_col) % samples_per_ms)
    best = AcquisitionCandidate(
        prn=config.prn,
        doppler_hz=float(doppler_bins[best_row]),
        code_phase_samples=best_code_phase,
        metric=best_metric,
    )

    top_indices = np.argsort(flat)[-8:][::-1]
    candidates: list[AcquisitionCandidate] = []
    used_locations: list[tuple[int, int]] = []
    for flat_index in top_indices:
        row, col = np.unravel_index(int(flat_index), heatmap.shape)
        if any(abs(row - prev_row) <= 1 and abs(col - prev_col) <= 2 for prev_row, prev_col in used_locations):
            continue
        used_locations.append((row, col))
        code_phase = int((samples_per_ms - col) % samples_per_ms)
        candidates.append(
            AcquisitionCandidate(
                prn=config.prn,
                doppler_hz=float(doppler_bins[row]),
                code_phase_samples=code_phase,
                metric=float(heatmap[row, col] / noise_floor),
            )
        )

    if log_callback:
        log_callback(
            f"Acquisition PRN {config.prn}: best peak metric {best.metric:.2f} at "
            f"{best.doppler_hz:.1f} Hz, code phase {best.code_phase_samples} samples."
        )

    return AcquisitionResult(
        prn=config.prn,
        doppler_bins_hz=doppler_bins.astype(np.float32),
        code_phases_samples=np.arange(samples_per_ms, dtype=np.int32),
        heatmap=heatmap,
        best_candidate=best,
        candidates=candidates,
    )


def acquisition_from_session(
    samples: np.ndarray,
    session: SessionConfig,
    progress_callback=None,
    log_callback=None,
) -> AcquisitionResult:
    """Build the acquisition config from the session settings."""

    config = AcquisitionConfig(
        sample_rate=session.sample_rate,
        prn=session.prn,
        doppler_min=session.doppler_min,
        doppler_max=session.doppler_max,
        doppler_step=session.doppler_step,
        integration_ms=session.integration_ms,
    )
    return acquire_signal(samples, config, progress_callback=progress_callback, log_callback=log_callback)
