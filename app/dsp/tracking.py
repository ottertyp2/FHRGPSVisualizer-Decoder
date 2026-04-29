"""Simple educational GPS tracking loops."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from app.dsp.gps_ca import CA_CODE_RATE_HZ, code_phase_samples_to_chips, sample_ca_code
from app.dsp.io import Complex64FileSource
from app.models import AcquisitionResult, SessionConfig, TrackingState


def _session_search_center_hz(session: SessionConfig) -> float:
    """Return the active acquisition/tracking search center from the session."""

    return 0.0 if session.is_baseband else float(session.if_frequency_hz)


def _sample_index_for_ms(ms_index: int, sample_rate: float) -> int:
    """Return the nearest sample index for an integer millisecond boundary."""

    return int(round(int(ms_index) * float(sample_rate) * 1e-3))


@dataclass(slots=True)
class _TrackingLoop:
    """Mutable carrier/code state for the teaching tracking loop."""

    code_phase_chips: float
    code_freq_hz: float
    carrier_freq_hz: float
    carrier_phase_rad: float = 0.0
    previous_prompt: complex = 0.0j


@dataclass(slots=True)
class _CorrelatorOutput:
    """Prompt, early, and late correlator outputs for one millisecond."""

    prompt_code: np.ndarray
    early: complex
    prompt: complex
    late: complex


@dataclass(slots=True)
class _TrackingHistory:
    """Preallocated arrays filled by the tracking loop."""

    prompt_i: np.ndarray
    prompt_q: np.ndarray
    early_mag: np.ndarray
    prompt_mag: np.ndarray
    late_mag: np.ndarray
    code_error: np.ndarray
    carrier_error: np.ndarray
    doppler_est: np.ndarray
    code_freq_est: np.ndarray
    lock_metric: np.ndarray
    prompt_history: np.ndarray
    raw_preview: np.ndarray
    wiped_preview: np.ndarray
    despread_preview: np.ndarray


def _validate_tracking_inputs(
    session: SessionConfig,
    acquisition: AcquisitionResult,
    max_ms: int,
) -> tuple[int, float]:
    """Validate tracking inputs and return samples/ms plus search center."""

    if not np.isfinite(session.sample_rate) or session.sample_rate <= 0:
        raise ValueError("Sample rate must be positive for tracking.")
    if not np.isclose(
        float(session.sample_rate),
        float(acquisition.sample_rate_hz),
        rtol=1e-9,
        atol=1e-3,
    ):
        raise ValueError(
            f"Tracking sample-rate mismatch: session {session.sample_rate:.6f} Sa/s "
            f"does not match acquisition {acquisition.sample_rate_hz:.6f} Sa/s."
        )

    search_center_hz = _session_search_center_hz(session)
    if not np.isclose(
        search_center_hz,
        float(acquisition.search_center_hz),
        rtol=1e-9,
        atol=1e-3,
    ):
        raise ValueError(
            f"Tracking search-center mismatch: session {search_center_hz:.3f} Hz "
            f"does not match acquisition {acquisition.search_center_hz:.3f} Hz."
        )

    if max_ms <= 0:
        raise ValueError("Not enough samples for 1 ms tracking updates.")

    best = acquisition.best_candidate
    if int(best.prn) != int(acquisition.prn):
        raise ValueError(
            f"Acquisition PRN mismatch: result PRN {acquisition.prn} does not match candidate PRN {best.prn}."
        )
    if int(session.prn) != int(acquisition.prn):
        raise ValueError(
            f"Tracking PRN mismatch: session PRN {session.prn} does not match acquisition PRN {acquisition.prn}."
        )

    return int(round(session.sample_rate * 1e-3)), search_center_hz


def _initial_tracking_loop(session: SessionConfig, acquisition: AcquisitionResult) -> _TrackingLoop:
    """Initialize carrier and code state from the acquisition peak."""

    best = acquisition.best_candidate
    return _TrackingLoop(
        code_phase_chips=code_phase_samples_to_chips(best.code_phase_samples, session.sample_rate),
        code_freq_hz=CA_CODE_RATE_HZ,
        carrier_freq_hz=float(best.carrier_frequency_hz),
    )


def _empty_tracking_history(max_ms: int) -> _TrackingHistory:
    """Allocate the per-millisecond tracking arrays."""

    return _TrackingHistory(
        prompt_i=np.zeros(max_ms, dtype=np.float32),
        prompt_q=np.zeros(max_ms, dtype=np.float32),
        early_mag=np.zeros(max_ms, dtype=np.float32),
        prompt_mag=np.zeros(max_ms, dtype=np.float32),
        late_mag=np.zeros(max_ms, dtype=np.float32),
        code_error=np.zeros(max_ms, dtype=np.float32),
        carrier_error=np.zeros(max_ms, dtype=np.float32),
        doppler_est=np.zeros(max_ms, dtype=np.float32),
        code_freq_est=np.zeros(max_ms, dtype=np.float32),
        lock_metric=np.zeros(max_ms, dtype=np.float32),
        prompt_history=np.zeros(max_ms, dtype=np.complex64),
        raw_preview=np.empty(0, dtype=np.complex64),
        wiped_preview=np.empty(0, dtype=np.complex64),
        despread_preview=np.empty(0, dtype=np.complex64),
    )


def _wipe_carrier(block: np.ndarray, loop: _TrackingLoop, block_time: np.ndarray) -> np.ndarray:
    """Remove the current carrier estimate from one 1 ms block."""

    carrier = np.exp(-1j * (loop.carrier_phase_rad + 2.0 * np.pi * loop.carrier_freq_hz * block_time))
    return block * carrier


def _correlate_prompt_early_late(
    wiped: np.ndarray,
    prn: int,
    sample_rate: float,
    samples_per_ms: int,
    loop: _TrackingLoop,
    early_late_spacing_chips: float,
) -> _CorrelatorOutput:
    """Correlate wiped samples against early, prompt, and late C/A replicas."""

    prompt_code = sample_ca_code(
        prn,
        sample_rate,
        samples_per_ms,
        loop.code_phase_chips,
        loop.code_freq_hz,
    )
    early_code = sample_ca_code(
        prn,
        sample_rate,
        samples_per_ms,
        loop.code_phase_chips - early_late_spacing_chips / 2.0,
        loop.code_freq_hz,
    )
    late_code = sample_ca_code(
        prn,
        sample_rate,
        samples_per_ms,
        loop.code_phase_chips + early_late_spacing_chips / 2.0,
        loop.code_freq_hz,
    )
    return _CorrelatorOutput(
        prompt_code=prompt_code,
        early=np.vdot(early_code, wiped) / samples_per_ms,
        prompt=np.vdot(prompt_code, wiped) / samples_per_ms,
        late=np.vdot(late_code, wiped) / samples_per_ms,
    )


def _loop_discriminators(output: _CorrelatorOutput) -> tuple[float, float]:
    """Return the simple DLL and PLL discriminator values."""

    early_abs = np.abs(output.early)
    late_abs = np.abs(output.late)
    dll_disc = (early_abs - late_abs) / (early_abs + late_abs + 1e-12)
    pll_disc = np.arctan2(output.prompt.imag, np.abs(output.prompt.real) + 1e-12)
    return float(dll_disc), float(pll_disc)


def _advance_tracking_loop(
    loop: _TrackingLoop,
    prompt: complex,
    dll_disc: float,
    pll_disc: float,
    ms_index: int,
    session: SessionConfig,
) -> None:
    """Update carrier and code estimates after one millisecond."""

    if ms_index > 0:
        phase_step = np.angle(prompt * np.conj(loop.previous_prompt))
        freq_error_hz = phase_step / (2.0 * np.pi * 1e-3)
    else:
        freq_error_hz = 0.0

    if ms_index < 40:
        loop.carrier_freq_hz += session.fll_gain * freq_error_hz
    loop.carrier_freq_hz += session.pll_gain * pll_disc / (2.0 * np.pi)
    loop.code_freq_hz = CA_CODE_RATE_HZ - session.dll_gain * dll_disc * 400.0
    loop.code_phase_chips = (loop.code_phase_chips + loop.code_freq_hz * 1e-3) % 1023.0
    loop.carrier_phase_rad = (
        loop.carrier_phase_rad + 2.0 * np.pi * loop.carrier_freq_hz * 1e-3
    ) % (2.0 * np.pi)
    loop.previous_prompt = prompt


def _store_tracking_outputs(
    history: _TrackingHistory,
    ms_index: int,
    output: _CorrelatorOutput,
    dll_disc: float,
    pll_disc: float,
    loop: _TrackingLoop,
    search_center_hz: float,
) -> None:
    """Store one millisecond of loop outputs."""

    prompt = output.prompt
    history.prompt_i[ms_index] = float(prompt.real)
    history.prompt_q[ms_index] = float(prompt.imag)
    history.early_mag[ms_index] = float(np.abs(output.early))
    history.prompt_mag[ms_index] = float(np.abs(prompt))
    history.late_mag[ms_index] = float(np.abs(output.late))
    history.code_error[ms_index] = float(dll_disc)
    history.carrier_error[ms_index] = float(pll_disc)
    history.doppler_est[ms_index] = float(loop.carrier_freq_hz - search_center_hz)
    history.code_freq_est[ms_index] = float(loop.code_freq_hz)
    history.lock_metric[ms_index] = float(np.abs(prompt.real) / (np.abs(prompt.imag) + 1e-6))
    history.prompt_history[ms_index] = np.complex64(prompt)


def _store_iq_previews(
    history: _TrackingHistory,
    raw_preview_source: np.ndarray | None,
    block: np.ndarray,
    wiped: np.ndarray,
    prompt_code: np.ndarray,
) -> None:
    """Keep small IQ previews for the GUI diagnostic plots."""

    preview_source = raw_preview_source if raw_preview_source is not None and raw_preview_source.size else block
    history.raw_preview = preview_source[: min(4_000, preview_source.size)].astype(np.complex64, copy=False)
    history.wiped_preview = wiped[: min(4_000, wiped.size)].astype(np.complex64, copy=False)
    history.despread_preview = (wiped * prompt_code)[: min(4_000, wiped.size)].astype(np.complex64, copy=False)


def _detect_lock(history: _TrackingHistory, valid: int) -> bool:
    """Return the simple lock decision used by the educational tracker."""

    phase_lock = float(np.median(history.lock_metric[max(0, valid - 50) : valid])) > 1.5
    prompt_median = float(np.median(history.prompt_mag[:valid]))
    side_median = float(np.median((history.early_mag[:valid] + history.late_mag[:valid]) * 0.5))
    code_peak_lock = valid >= 200 and prompt_median > side_median * 1.15
    return bool(phase_lock or code_peak_lock)


def _build_tracking_state(
    acquisition: AcquisitionResult,
    history: _TrackingHistory,
    valid: int,
    lock_detected: bool,
) -> TrackingState:
    """Trim preallocated arrays and build the public tracking model."""

    times_s = np.arange(valid, dtype=np.float32) * 1e-3
    return TrackingState(
        prn=acquisition.prn,
        times_s=times_s,
        prompt_i=history.prompt_i[:valid],
        prompt_q=history.prompt_q[:valid],
        early_mag=history.early_mag[:valid],
        prompt_mag=history.prompt_mag[:valid],
        late_mag=history.late_mag[:valid],
        code_error=history.code_error[:valid],
        carrier_error=history.carrier_error[:valid],
        doppler_est_hz=history.doppler_est[:valid],
        code_freq_est_hz=history.code_freq_est[:valid],
        lock_metric=history.lock_metric[:valid],
        lock_detected=lock_detected,
        iq_views={
            "Raw IQ": history.raw_preview,
            "Carrier wiped": history.wiped_preview,
            "Despread": history.despread_preview,
            "Integrated prompt": history.prompt_history[:valid].astype(
                np.complex64,
                copy=False,
            ),
        },
        loop_states={
            "pll_disc_rad": history.carrier_error[:valid],
            "dll_disc": history.code_error[:valid],
        },
    )


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

    samples_per_ms, search_center_hz = _validate_tracking_inputs(
        session,
        acquisition,
        max_ms,
    )
    loop = _initial_tracking_loop(session, acquisition)
    history = _empty_tracking_history(max_ms)
    valid_count = 0
    block_time = np.arange(samples_per_ms, dtype=np.float64) / session.sample_rate

    for ms_index, block in enumerate(blocks):
        if ms_index >= max_ms:
            break
        if block.size < samples_per_ms:
            break

        wiped = _wipe_carrier(block, loop, block_time)
        correlators = _correlate_prompt_early_late(
            wiped,
            acquisition.prn,
            session.sample_rate,
            samples_per_ms,
            loop,
            float(session.early_late_spacing_chips),
        )
        dll_disc, pll_disc = _loop_discriminators(correlators)
        _advance_tracking_loop(
            loop,
            correlators.prompt,
            dll_disc,
            pll_disc,
            ms_index,
            session,
        )
        _store_tracking_outputs(
            history,
            ms_index,
            correlators,
            dll_disc,
            pll_disc,
            loop,
            search_center_hz,
        )

        if ms_index == 0:
            _store_iq_previews(
                history,
                raw_preview_source,
                block,
                wiped,
                correlators.prompt_code,
            )

        if progress_callback:
            progress_callback(int(100 * (ms_index + 1) / max_ms))
        valid_count = ms_index + 1

    valid = valid_count
    if valid == 0:
        raise ValueError("No complete 1 ms blocks were available for tracking.")

    lock_detected = _detect_lock(history, valid)

    if log_callback:
        state = "locked" if lock_detected else "not locked"
        log_callback(
            f"Tracking PRN {acquisition.prn}: processed {valid} ms, final carrier {loop.carrier_freq_hz:.1f} Hz "
            f"(relative Doppler {history.doppler_est[valid - 1]:+.1f} Hz), {state}."
        )

    return _build_tracking_state(acquisition, history, valid, lock_detected)


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
    available_ms = int(np.floor(samples.size / max(session.sample_rate * 1e-3, 1e-9)))
    max_ms = min(int(session.tracking_ms), available_ms)
    blocks = (
        samples[start : start + samples_per_ms]
        for start in (_sample_index_for_ms(ms_index, session.sample_rate) for ms_index in range(max_ms))
        if start + samples_per_ms <= samples.size
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
    available_ms = int(np.floor(max(0, source.total_samples - int(start_sample)) / max(session.sample_rate * 1e-3, 1e-9)))
    max_ms = min(int(session.tracking_ms), int(available_ms))
    raw_preview = source.read_window(start_sample, min(4_000, samples_per_ms))

    def iter_ms_blocks():
        for ms_index in range(max_ms):
            block_start = int(start_sample) + _sample_index_for_ms(ms_index, session.sample_rate)
            yield source.read_window(block_start, samples_per_ms)

    return _track_blocks(
        iter_ms_blocks(),
        session,
        acquisition,
        max_ms=max_ms,
        raw_preview_source=raw_preview,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )
