"""Simple educational GPS tracking loops."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from app.dsp.gps_ca import CA_CODE_RATE_HZ, code_phase_samples_to_chips, sample_ca_code
from app.dsp.io import Complex64FileSource
from app.models import AcquisitionResult, SessionConfig, TrackingState


def _track_blocks(
    blocks: Iterable[np.ndarray],
    session: SessionConfig,
    acquisition: AcquisitionResult,
    max_ms: int,
    raw_preview_source: np.ndarray | None = None,
    progress_callback=None,
    log_callback=None,
) -> TrackingState:
    """Track one PRN using a sequence of 1 ms blocks."""

    samples_per_ms = int(round(session.sample_rate * 1e-3))
    if max_ms <= 0:
        raise ValueError("Not enough samples for 1 ms tracking updates.")

    best = acquisition.best_candidate
    spacing = float(session.early_late_spacing_chips)
    code_phase_chips = code_phase_samples_to_chips(best.code_phase_samples, session.sample_rate)
    code_freq = CA_CODE_RATE_HZ
    carrier_freq = float(best.doppler_hz)
    carrier_phase = 0.0
    prev_prompt = 0.0j

    prompt_i = np.zeros(max_ms, dtype=np.float32)
    prompt_q = np.zeros(max_ms, dtype=np.float32)
    early_mag = np.zeros(max_ms, dtype=np.float32)
    prompt_mag = np.zeros(max_ms, dtype=np.float32)
    late_mag = np.zeros(max_ms, dtype=np.float32)
    code_error = np.zeros(max_ms, dtype=np.float32)
    carrier_error = np.zeros(max_ms, dtype=np.float32)
    doppler_est = np.zeros(max_ms, dtype=np.float32)
    code_freq_est = np.zeros(max_ms, dtype=np.float32)
    lock_metric = np.zeros(max_ms, dtype=np.float32)
    prompt_history = np.zeros(max_ms, dtype=np.complex64)
    raw_preview = np.empty(0, dtype=np.complex64)
    wiped_preview = np.empty(0, dtype=np.complex64)
    despread_preview = np.empty(0, dtype=np.complex64)

    valid_count = 0

    for ms_index, block in enumerate(blocks):
        if ms_index >= max_ms:
            break
        if block.size < samples_per_ms:
            break

        block_time = np.arange(samples_per_ms, dtype=np.float64) / session.sample_rate
        carrier = np.exp(-1j * (carrier_phase + 2.0 * np.pi * carrier_freq * block_time))
        wiped = block * carrier

        prompt_code = sample_ca_code(session.prn, session.sample_rate, samples_per_ms, code_phase_chips, code_freq)
        early_code = sample_ca_code(
            session.prn,
            session.sample_rate,
            samples_per_ms,
            code_phase_chips - spacing / 2.0,
            code_freq,
        )
        late_code = sample_ca_code(
            session.prn,
            session.sample_rate,
            samples_per_ms,
            code_phase_chips + spacing / 2.0,
            code_freq,
        )

        early = np.vdot(early_code, wiped) / samples_per_ms
        prompt = np.vdot(prompt_code, wiped) / samples_per_ms
        late = np.vdot(late_code, wiped) / samples_per_ms

        prompt_i[ms_index] = float(prompt.real)
        prompt_q[ms_index] = float(prompt.imag)
        early_mag[ms_index] = float(np.abs(early))
        prompt_mag[ms_index] = float(np.abs(prompt))
        late_mag[ms_index] = float(np.abs(late))
        prompt_history[ms_index] = np.complex64(prompt)

        dll_disc = (np.abs(early) - np.abs(late)) / (np.abs(early) + np.abs(late) + 1e-12)
        pll_disc = np.arctan2(prompt.imag, prompt.real)
        code_error[ms_index] = float(dll_disc)
        carrier_error[ms_index] = float(pll_disc)

        if ms_index > 0:
            phase_step = np.angle(prompt * np.conj(prev_prompt))
            freq_error_hz = phase_step / (2.0 * np.pi * 1e-3)
        else:
            freq_error_hz = 0.0

        if ms_index < 40:
            carrier_freq += session.fll_gain * freq_error_hz
        carrier_freq += session.pll_gain * pll_disc / (2.0 * np.pi)
        code_freq = CA_CODE_RATE_HZ - session.dll_gain * dll_disc * 400.0

        doppler_est[ms_index] = float(carrier_freq)
        code_freq_est[ms_index] = float(code_freq)
        lock_metric[ms_index] = float(np.abs(prompt.real) / (np.abs(prompt.imag) + 1e-6))

        code_phase_chips = (code_phase_chips + (code_freq * 1e-3)) % 1023.0
        carrier_phase = (carrier_phase + 2.0 * np.pi * carrier_freq * 1e-3) % (2.0 * np.pi)
        prev_prompt = prompt

        if ms_index == 0:
            preview_source = raw_preview_source if raw_preview_source is not None and raw_preview_source.size else block
            raw_preview = preview_source[: min(4_000, preview_source.size)].astype(np.complex64, copy=False)
            wiped_preview = wiped[: min(4_000, wiped.size)].astype(np.complex64, copy=False)
            despread_preview = (wiped * prompt_code)[: min(4_000, wiped.size)].astype(np.complex64, copy=False)

        if progress_callback:
            progress_callback(int(100 * (ms_index + 1) / max_ms))
        valid_count = ms_index + 1

    valid = valid_count
    times_s = np.arange(valid, dtype=np.float32) * 1e-3
    lock_detected = bool(np.median(lock_metric[max(0, valid - 50) : valid]) > 1.5) if valid else False

    if log_callback:
        state = "locked" if lock_detected else "not locked"
        log_callback(
            f"Tracking PRN {session.prn}: processed {valid} ms, final Doppler {doppler_est[valid - 1]:.1f} Hz, {state}."
        )

    return TrackingState(
        prn=session.prn,
        times_s=times_s,
        prompt_i=prompt_i[:valid],
        prompt_q=prompt_q[:valid],
        early_mag=early_mag[:valid],
        prompt_mag=prompt_mag[:valid],
        late_mag=late_mag[:valid],
        code_error=code_error[:valid],
        carrier_error=carrier_error[:valid],
        doppler_est_hz=doppler_est[:valid],
        code_freq_est_hz=code_freq_est[:valid],
        lock_metric=lock_metric[:valid],
        lock_detected=lock_detected,
        iq_views={
            "Raw IQ": raw_preview,
            "Carrier wiped": wiped_preview,
            "Despread": despread_preview,
            "Integrated prompt": prompt_history[:valid].astype(np.complex64, copy=False),
        },
        loop_states={
            "pll_disc_rad": carrier_error[:valid],
            "dll_disc": code_error[:valid],
        },
    )


def track_signal(
    samples: np.ndarray,
    session: SessionConfig,
    acquisition: AcquisitionResult,
    progress_callback=None,
    log_callback=None,
) -> TrackingState:
    """Track one PRN from an in-memory sample array."""

    if samples.size == 0:
        raise ValueError("No samples available for tracking.")

    samples_per_ms = int(round(session.sample_rate * 1e-3))
    max_ms = min(int(session.tracking_ms), samples.size // samples_per_ms)
    blocks = (
        samples[start : start + samples_per_ms]
        for start in range(0, max_ms * samples_per_ms, samples_per_ms)
    )
    return _track_blocks(
        blocks,
        session,
        acquisition,
        max_ms=max_ms,
        raw_preview_source=samples,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )


def track_file(
    file_path: str,
    start_sample: int,
    session: SessionConfig,
    acquisition: AcquisitionResult,
    progress_callback=None,
    log_callback=None,
) -> TrackingState:
    """Track one PRN directly from a large file without loading it all into memory."""

    source = Complex64FileSource(file_path)
    samples_per_ms = int(round(session.sample_rate * 1e-3))
    available_ms = max(0, (source.total_samples - int(start_sample)) // samples_per_ms)
    max_ms = min(int(session.tracking_ms), int(available_ms))
    raw_preview = source.read_window(start_sample, min(4_000, samples_per_ms))
    return _track_blocks(
        source.iter_blocks(start_sample, samples_per_ms, max_ms),
        session,
        acquisition,
        max_ms=max_ms,
        raw_preview_source=raw_preview,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )
