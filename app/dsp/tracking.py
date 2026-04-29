"""Simple educational GPS tracking loops."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.dsp.compute import get_cupy_module, resolve_compute_plan
from app.dsp.gps_ca import CA_CODE_LENGTH, CA_CODE_RATE_HZ, code_phase_samples_to_chips, generate_ca_code
from app.dsp.io import Complex64FileSource
from app.dsp.tracking_gpu import get_tracking_correlator_kernel
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

    prompt_code: Any | None
    early: complex
    prompt: complex
    late: complex


@dataclass(slots=True)
class _TrackingMath:
    """Small NumPy/CuPy adapter for per-millisecond vector work."""

    backend: str
    array_module: Any
    base_code: Any
    block_time: Any
    sample_indices: Any
    cupy: Any | None = None
    gpu_correlator: Any | None = None
    gpu_output: Any | None = None
    gpu_threads: int = 256


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


def _tracking_math_for_backend(
    backend: str,
    prn: int,
    sample_rate: float,
    samples_per_ms: int,
) -> _TrackingMath:
    """Build NumPy or CuPy arrays shared by all 1 ms tracking updates."""

    if backend == "gpu":
        cupy = get_cupy_module()
        if cupy is None:
            raise RuntimeError("CuPy is not available for GPU tracking.")
        return _TrackingMath(
            backend="gpu",
            array_module=cupy,
            base_code=cupy.asarray(generate_ca_code(prn), dtype=cupy.float32),
            block_time=cupy.arange(samples_per_ms, dtype=cupy.float64) / sample_rate,
            sample_indices=cupy.arange(samples_per_ms, dtype=cupy.float64),
            cupy=cupy,
            gpu_correlator=get_tracking_correlator_kernel(),
            gpu_output=cupy.empty(6, dtype=cupy.float64),
        )

    return _TrackingMath(
        backend="cpu",
        array_module=np,
        base_code=generate_ca_code(prn),
        block_time=np.arange(samples_per_ms, dtype=np.float64) / sample_rate,
        sample_indices=np.arange(samples_per_ms, dtype=np.float64),
    )


def _scalars_to_complex(values: tuple[Any, ...], math: _TrackingMath) -> tuple[complex, ...]:
    """Copy backend scalars to Python complex values."""

    if math.backend == "gpu" and math.cupy is not None:
        gpu_values = math.cupy.stack([math.cupy.asarray(value) for value in values])
        values = tuple(math.cupy.asnumpy(gpu_values).ravel())
    return tuple(complex(value) for value in values)


def _as_numpy_complex64(values: Any, math: _TrackingMath, limit: int | None = None) -> np.ndarray:
    """Copy a CPU/GPU vector into a compact complex64 NumPy preview."""

    if limit is not None:
        values = values[: min(int(limit), int(values.size))]
    if math.backend == "gpu" and math.cupy is not None and not isinstance(values, np.ndarray):
        values = math.cupy.asnumpy(values)
    return np.asarray(values, dtype=np.complex64)


def _sample_ca_code_for_phase(
    math: _TrackingMath,
    code_phase_chips: float,
    code_freq_hz: float,
    sample_rate: float,
) -> Any:
    """Sample the local C/A code with the active array backend."""

    chip_positions = code_phase_chips + math.sample_indices * code_freq_hz / sample_rate
    chip_indices = (
        math.array_module.floor(chip_positions).astype(math.array_module.int64)
        % CA_CODE_LENGTH
    )
    return math.base_code[chip_indices]


def _sample_ca_code_for_loop(math: _TrackingMath, loop: _TrackingLoop, sample_rate: float) -> Any:
    """Sample the prompt C/A code for the current tracking loop state."""

    return _sample_ca_code_for_phase(
        math,
        loop.code_phase_chips,
        loop.code_freq_hz,
        sample_rate,
    )


def _wipe_carrier(block: Any, loop: _TrackingLoop, math: _TrackingMath) -> Any:
    """Remove the current carrier estimate from one 1 ms block."""

    xp = math.array_module
    carrier_phase = loop.carrier_phase_rad + 2.0 * xp.pi * loop.carrier_freq_hz * math.block_time
    carrier = xp.exp(-1j * carrier_phase)
    return block * carrier


def _correlate_prompt_early_late(
    wiped: Any,
    sample_rate: float,
    samples_per_ms: int,
    loop: _TrackingLoop,
    early_late_spacing_chips: float,
    math: _TrackingMath,
) -> _CorrelatorOutput:
    """Correlate wiped samples against early, prompt, and late C/A replicas."""

    xp = math.array_module
    prompt_code = _sample_ca_code_for_loop(math, loop, sample_rate)
    early_code = _sample_ca_code_for_phase(
        math,
        loop.code_phase_chips - early_late_spacing_chips / 2.0,
        loop.code_freq_hz,
        sample_rate,
    )
    late_code = _sample_ca_code_for_phase(
        math,
        loop.code_phase_chips + early_late_spacing_chips / 2.0,
        loop.code_freq_hz,
        sample_rate,
    )

    early = xp.vdot(early_code, wiped) / samples_per_ms
    prompt = xp.vdot(prompt_code, wiped) / samples_per_ms
    late = xp.vdot(late_code, wiped) / samples_per_ms
    early_value, prompt_value, late_value = _scalars_to_complex((early, prompt, late), math)
    return _CorrelatorOutput(
        prompt_code=prompt_code,
        early=early_value,
        prompt=prompt_value,
        late=late_value,
    )


def _correlate_block_gpu(
    block: Any,
    session: SessionConfig,
    samples_per_ms: int,
    loop: _TrackingLoop,
    math: _TrackingMath,
) -> _CorrelatorOutput:
    """Run carrier wipe and E/P/L correlations in one compact GPU kernel."""

    if math.gpu_correlator is None or math.gpu_output is None:
        raise RuntimeError("GPU tracking correlator is not initialized.")

    threads = min(math.gpu_threads, max(32, int(2 ** np.ceil(np.log2(samples_per_ms)))))
    threads = max(32, min(1024, int(threads)))
    shared_bytes = 6 * threads * np.dtype(np.float64).itemsize
    math.gpu_correlator(
        (1,),
        (threads,),
        (
            block,
            math.base_code,
            np.int32(samples_per_ms),
            np.float64(session.sample_rate),
            np.float64(loop.code_phase_chips),
            np.float64(loop.code_freq_hz),
            np.float64(session.early_late_spacing_chips),
            np.float64(loop.carrier_phase_rad),
            np.float64(loop.carrier_freq_hz),
            math.gpu_output,
        ),
        shared_mem=shared_bytes,
    )
    values = math.cupy.asnumpy(math.gpu_output) if math.cupy is not None else np.zeros(6)
    return _CorrelatorOutput(
        prompt_code=None,
        early=complex(values[0], values[1]),
        prompt=complex(values[2], values[3]),
        late=complex(values[4], values[5]),
    )


def _correlate_block(
    block: Any,
    session: SessionConfig,
    samples_per_ms: int,
    loop: _TrackingLoop,
    math: _TrackingMath,
) -> tuple[_CorrelatorOutput, Any | None]:
    """Correlate one block and return the wiped block when it was materialized."""

    if math.backend == "gpu" and math.gpu_correlator is not None:
        return _correlate_block_gpu(block, session, samples_per_ms, loop, math), None

    wiped = _wipe_carrier(block, loop, math)
    return (
        _correlate_prompt_early_late(
            wiped,
            session.sample_rate,
            samples_per_ms,
            loop,
            float(session.early_late_spacing_chips),
            math,
        ),
        wiped,
    )


def _loop_discriminators(output: _CorrelatorOutput) -> tuple[float, float]:
    """Return the simple DLL and PLL discriminator values."""

    early_abs = np.abs(output.early)
    late_abs = np.abs(output.late)
    dll_disc = (early_abs - late_abs) / (early_abs + late_abs + 1e-12)
    data_sign = 1.0 if output.prompt.real >= 0.0 else -1.0
    pll_disc = np.arctan2(data_sign * output.prompt.imag, np.abs(output.prompt.real) + 1e-12)
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
        # Squaring removes the 180 degree BPSK navigation-bit ambiguity before
        # the early FLL estimates the residual carrier frequency.
        phase_step = 0.5 * np.angle((prompt * prompt) * np.conj(loop.previous_prompt * loop.previous_prompt))
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
    block: Any,
    wiped: Any,
    prompt_code: Any,
    math: _TrackingMath,
) -> None:
    """Keep small IQ previews for the GUI diagnostic plots."""

    preview_source = raw_preview_source if raw_preview_source is not None and raw_preview_source.size else block
    history.raw_preview = _as_numpy_complex64(preview_source, math, 4_000)
    history.wiped_preview = _as_numpy_complex64(wiped, math, 4_000)
    history.despread_preview = _as_numpy_complex64(wiped * prompt_code, math, 4_000)


def _detect_lock(history: _TrackingHistory, valid: int) -> bool:
    """Return the simple lock decision used by the educational tracker."""

    phase_lock = float(np.median(history.lock_metric[max(0, valid - 50) : valid])) > 1.5
    prompt_median = float(np.median(history.prompt_mag[:valid]))
    side_median = float(np.median((history.early_mag[:valid] + history.late_mag[:valid]) * 0.5))
    code_peak_lock = valid >= 200 and prompt_median > side_median * 1.15
    return bool(phase_lock or code_peak_lock)


def _build_tracking_state(
    acquisition: AcquisitionResult,
    session: SessionConfig,
    history: _TrackingHistory,
    valid: int,
    lock_detected: bool,
    source_start_sample: int = 0,
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
        source_start_sample=int(source_start_sample),
        sample_rate_hz=float(session.sample_rate),
        code_phase_samples=int(acquisition.best_candidate.code_phase_samples),
    )


def _run_tracking_loop(
    blocks: Iterable[Any],
    session: SessionConfig,
    acquisition: AcquisitionResult,
    max_ms: int,
    samples_per_ms: int,
    search_center_hz: float,
    math: _TrackingMath,
    raw_preview_source: np.ndarray | None = None,
    source_start_sample: int = 0,
    progress_callback=None,
    log_callback=None,
) -> TrackingState:
    """Run the serial 1 ms tracking loop with CPU or GPU vector math."""

    loop = _initial_tracking_loop(session, acquisition)
    history = _empty_tracking_history(max_ms)
    valid_count = 0

    for ms_index, block in enumerate(blocks):
        if ms_index >= max_ms:
            break
        if block.size < samples_per_ms:
            break

        correlators, wiped = _correlate_block(
            block,
            session,
            samples_per_ms,
            loop,
            math,
        )
        dll_disc, pll_disc = _loop_discriminators(correlators)

        if ms_index == 0:
            if wiped is None:
                wiped = _wipe_carrier(block, loop, math)
            prompt_code = correlators.prompt_code
            if prompt_code is None:
                prompt_code = _sample_ca_code_for_loop(math, loop, session.sample_rate)
            _store_iq_previews(
                history,
                raw_preview_source,
                block,
                wiped,
                prompt_code,
                math,
            )

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

    return _build_tracking_state(
        acquisition,
        session,
        history,
        valid,
        lock_detected,
        source_start_sample=source_start_sample,
    )


def _log_tracking_backend(plan, log_callback=None) -> None:
    """Log the selected tracking compute backend."""

    if not log_callback:
        return
    gpu_text = plan.gpu_name if plan.gpu_available else "unavailable"
    log_callback(
        f"Tracking runtime: backend {plan.active_backend}, "
        f"workers {plan.selected_workers}/{plan.logical_cores}, GPU {gpu_text}."
    )


def _log_tracking_fallback(exc: Exception, log_callback=None) -> None:
    """Log one compact GPU-to-CPU fallback reason."""

    if not log_callback:
        return
    reason = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    log_callback(f"GPU tracking path failed; falling back to CPU. Reason: {reason}.")


def _iter_sample_blocks(
    samples: Any,
    sample_rate: float,
    samples_per_ms: int,
    max_ms: int,
) -> Iterable[Any]:
    """Yield exact 1 ms blocks from an in-memory CPU or GPU sample array."""

    for ms_index in range(max_ms):
        start = _sample_index_for_ms(ms_index, sample_rate)
        stop = start + samples_per_ms
        if stop > samples.size:
            break
        yield samples[start:stop]


def _track_sample_array_with_backend(
    samples: np.ndarray,
    session: SessionConfig,
    acquisition: AcquisitionResult,
    max_ms: int,
    backend: str,
    source_start_sample: int = 0,
    progress_callback=None,
    log_callback=None,
) -> TrackingState:
    """Track an in-memory sample array with one selected backend."""

    samples_per_ms, search_center_hz = _validate_tracking_inputs(
        session,
        acquisition,
        max_ms,
    )
    math = _tracking_math_for_backend(backend, acquisition.prn, session.sample_rate, samples_per_ms)
    backend_samples = (
        math.cupy.asarray(samples)
        if math.backend == "gpu" and math.cupy is not None
        else samples
    )
    blocks = _iter_sample_blocks(backend_samples, session.sample_rate, samples_per_ms, max_ms)
    return _run_tracking_loop(
        blocks,
        session,
        acquisition,
        max_ms=max_ms,
        samples_per_ms=samples_per_ms,
        search_center_hz=search_center_hz,
        math=math,
        raw_preview_source=samples,
        source_start_sample=source_start_sample,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )


def _track_stream_with_backend(
    source: Complex64FileSource,
    start_sample: int,
    raw_preview: np.ndarray,
    session: SessionConfig,
    acquisition: AcquisitionResult,
    max_ms: int,
    backend: str,
    progress_callback=None,
    log_callback=None,
) -> TrackingState:
    """Track a file stream with one selected backend."""

    samples_per_ms, search_center_hz = _validate_tracking_inputs(
        session,
        acquisition,
        max_ms,
    )
    math = _tracking_math_for_backend(backend, acquisition.prn, session.sample_rate, samples_per_ms)

    def iter_blocks() -> Iterable[Any]:
        for ms_index in range(max_ms):
            block_start = int(start_sample) + _sample_index_for_ms(ms_index, session.sample_rate)
            block = source.read_window(block_start, samples_per_ms)
            if math.backend == "gpu" and math.cupy is not None:
                yield math.cupy.asarray(block)
            else:
                yield block

    return _run_tracking_loop(
        iter_blocks(),
        session,
        acquisition,
        max_ms=max_ms,
        samples_per_ms=samples_per_ms,
        search_center_hz=search_center_hz,
        math=math,
        raw_preview_source=raw_preview,
        source_start_sample=int(start_sample),
        progress_callback=progress_callback,
        log_callback=log_callback,
    )


def track_signal(
    samples: np.ndarray,
    session: SessionConfig,
    acquisition: AcquisitionResult,
    source_start_sample: int = 0,
    progress_callback=None,
    log_callback=None,
) -> TrackingState:
    """Track one PRN from an in-memory sample array."""

    if samples.size == 0:
        raise ValueError("No samples available for tracking.")

    available_ms = int(np.floor(samples.size / max(session.sample_rate * 1e-3, 1e-9)))
    max_ms = min(int(session.tracking_ms), available_ms)
    plan = resolve_compute_plan(
        session.compute_backend,
        session.max_workers,
        gpu_enabled=session.gpu_enabled,
        max_tasks=max_ms,
        prefer_gpu=True,
    )
    _log_tracking_backend(plan, log_callback)

    if plan.active_backend == "gpu":
        try:
            return _track_sample_array_with_backend(
                samples,
                session,
                acquisition,
                max_ms,
                backend="gpu",
                source_start_sample=source_start_sample,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        except Exception as exc:
            _log_tracking_fallback(exc, log_callback)

    return _track_sample_array_with_backend(
        samples,
        session,
        acquisition,
        max_ms=max_ms,
        backend="cpu",
        source_start_sample=source_start_sample,
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
    available_ms = int(
        np.floor(
            max(0, source.total_samples - int(start_sample))
            / max(session.sample_rate * 1e-3, 1e-9)
        )
    )
    max_ms = min(int(session.tracking_ms), int(available_ms))
    raw_preview = source.read_window(start_sample, min(4_000, samples_per_ms))
    plan = resolve_compute_plan(
        session.compute_backend,
        session.max_workers,
        gpu_enabled=session.gpu_enabled,
        max_tasks=max_ms,
        prefer_gpu=True,
    )
    _log_tracking_backend(plan, log_callback)

    if plan.active_backend == "gpu":
        try:
            return _track_stream_with_backend(
                source,
                start_sample,
                raw_preview,
                session,
                acquisition,
                max_ms,
                backend="gpu",
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        except Exception as exc:
            _log_tracking_fallback(exc, log_callback)

    return _track_stream_with_backend(
        source,
        start_sample,
        raw_preview,
        session,
        acquisition,
        max_ms=max_ms,
        backend="cpu",
        progress_callback=progress_callback,
        log_callback=log_callback,
    )
