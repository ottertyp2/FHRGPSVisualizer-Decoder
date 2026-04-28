"""Simple acquisition search over Doppler and code phase."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from app.dsp.compute import (
    ProgressTracker,
    get_cupy_module,
    parallel_ordered_map,
    resolve_compute_plan,
    split_nested_worker_budget,
)
from app.dsp.gps_ca import sample_ca_code
from app.models import (
    AcquisitionCandidate,
    AcquisitionResult,
    SearchCenterSweepEntry,
    SearchCenterSweepResult,
    SampleRateSurveyEntry,
    SampleRateSurveyResult,
    SessionConfig,
)


STRONG_ACQUISITION_METRIC_THRESHOLD = 6.0


def _sample_index_for_ms(ms_index: int, sample_rate: float) -> int:
    """Return the nearest sample index for an integer millisecond boundary."""

    return int(round(int(ms_index) * float(sample_rate) * 1e-3))


@dataclass(slots=True)
class AcquisitionConfig:
    """Compact acquisition settings."""

    sample_rate: float
    prn: int
    doppler_min: int
    doppler_max: int
    doppler_step: int
    search_center_hz: float = 0.0
    integration_ms: int = 4
    spread_acquisition_blocks: bool = True
    acquisition_segment_count: int = 1
    compute_backend: str = "auto"
    max_workers: int = 0
    gpu_enabled: bool = True


def _select_ms_blocks(
    samples: np.ndarray,
    sample_rate: float,
    block_count: int,
    spread_blocks: bool,
) -> np.ndarray:
    """Select 1 ms blocks either contiguously or spread across the full source."""

    samples_per_ms = int(round(sample_rate * 1e-3))
    total_ms = samples.size // samples_per_ms
    usable = min(max(1, int(block_count)), total_ms)
    if usable <= 0:
        return np.empty((0, samples_per_ms), dtype=np.complex64)

    if spread_blocks and total_ms > usable:
        starts_ms = np.linspace(0, total_ms - 1, num=usable, dtype=int)
    else:
        starts_ms = np.arange(usable, dtype=int)

    blocks = np.empty((usable, samples_per_ms), dtype=np.complex64)
    for row, start_ms in enumerate(starts_ms):
        start = _sample_index_for_ms(int(start_ms), sample_rate)
        stop = start + samples_per_ms
        if stop > samples.size:
            return blocks[:row]
        blocks[row, :] = samples[start:stop]
    return blocks


def _select_segment_starts_ms(total_ms: int, block_count: int, segment_count: int) -> np.ndarray:
    """Return evenly spaced segment starts for deeper acquisition."""

    usable_segment_count = max(1, int(segment_count))
    max_start = max(0, total_ms - max(1, int(block_count)))
    if usable_segment_count == 1 or max_start == 0:
        return np.asarray([0], dtype=int)
    return np.linspace(0, max_start, num=usable_segment_count, dtype=int)


def _cluster_segment_candidates(
    candidates: list[AcquisitionCandidate],
    doppler_tolerance_hz: float,
    code_phase_tolerance_samples: int,
    code_period_samples: int,
) -> tuple[list[AcquisitionCandidate], float]:
    """Find the dominant cluster of repeated PRN hits across searched segments."""

    if not candidates:
        return [], 0.0

    def circular_distance(left: int, right: int) -> int:
        period = max(1, int(code_period_samples))
        delta = abs(int(left) - int(right)) % period
        return min(delta, period - delta)

    def unwrap_code_phases(cluster: list[AcquisitionCandidate]) -> np.ndarray:
        period = max(1, int(code_period_samples))
        phases: list[float] = []
        for candidate in sorted(cluster, key=lambda item: item.segment_start_sample):
            phase = float(candidate.code_phase_samples % period)
            if phases:
                previous = phases[-1]
                wraps = round((previous - phase) / period)
                phase += wraps * period
                if phase - previous > period / 2.0:
                    phase -= period
                elif previous - phase > period / 2.0:
                    phase += period
            phases.append(phase)
        return np.asarray(phases, dtype=np.float64)

    def smooth_code_drift_score(cluster: list[AcquisitionCandidate]) -> float:
        if len(cluster) < 3:
            return 0.0
        ordered = sorted(cluster, key=lambda item: item.segment_start_sample)
        times = np.asarray([item.segment_start_sample for item in ordered], dtype=np.float64)
        times = (times - times[0]) / max(float(code_period_samples), 1.0)
        phases = unwrap_code_phases(ordered)
        if np.ptp(times) <= 0.0:
            return 0.0
        slope, intercept = np.polyfit(times, phases, deg=1)
        residual = phases - (slope * times + intercept)
        rms = float(np.sqrt(np.mean(residual**2)))
        tolerance = max(float(code_phase_tolerance_samples), float(code_period_samples) * 0.03)
        if rms > tolerance:
            return 0.0
        return max(0.1, 1.0 - rms / max(tolerance, 1.0))

    best_cluster: list[AcquisitionCandidate] = []
    best_score = -np.inf
    for seed in candidates:
        cluster = [
            candidate
            for candidate in candidates
            if abs(candidate.doppler_hz - seed.doppler_hz) <= doppler_tolerance_hz
        ]
        drift_score = smooth_code_drift_score(cluster)
        if drift_score <= 0.0:
            cluster = [
                candidate
                for candidate in cluster
                if circular_distance(candidate.code_phase_samples, seed.code_phase_samples) <= code_phase_tolerance_samples
            ]
            drift_score = 1.0 if cluster else 0.0
        score = float(len(cluster)) * float(np.mean([candidate.metric for candidate in cluster])) * drift_score
        if score > best_score:
            best_cluster = cluster
            best_score = score
    return best_cluster, float(max(best_score, 0.0))


def acquisition_metric_is_strong(metric: float) -> bool:
    """Return whether a raw acquisition metric is strong enough to trust."""

    return float(metric) >= STRONG_ACQUISITION_METRIC_THRESHOLD


def acquisition_result_is_plausible(result: AcquisitionResult) -> bool:
    """Return whether repetition and raw metric together look believable."""

    return acquisition_metric_is_strong(result.best_candidate.metric) and result.consistent_segments >= 3


def acquisition_interpretation(result: AcquisitionResult) -> str:
    """Provide a short user-facing interpretation label for one result."""

    if acquisition_result_is_plausible(result):
        return "repeated candidate; verify with tracking"
    if result.consistent_segments >= 3:
        return "repeated but still weak"
    return "weak / uncertain"


def acquisition_rank_key(result: AcquisitionResult) -> tuple[float, ...]:
    """Rank strong acquisitions by consistency and weak ones by raw metric first."""

    if acquisition_metric_is_strong(result.best_candidate.metric):
        return (
            1.0,
            float(result.consistent_segments),
            float(result.consistency_score),
            float(result.best_candidate.metric),
        )
    return (
        0.0,
        float(result.best_candidate.metric),
        float(result.consistent_segments),
        float(result.consistency_score),
    )


def _segment_blocks_for_start(
    samples: np.ndarray,
    sample_rate: float,
    config: AcquisitionConfig,
    segment_start_ms: int,
) -> np.ndarray:
    """Return normalized 1 ms acquisition blocks for one segment start."""

    samples_per_ms = int(round(sample_rate * 1e-3))
    segment_start = _sample_index_for_ms(int(segment_start_ms), sample_rate)
    if config.acquisition_segment_count > 1:
        segment_stop = min(samples.size, segment_start + max(1, int(config.integration_ms)) * samples_per_ms)
        segment_samples = np.asarray(samples[segment_start:segment_stop], dtype=np.complex64)
        spread_blocks = False
    else:
        segment_samples = np.asarray(samples[segment_start:], dtype=np.complex64)
        spread_blocks = bool(config.spread_acquisition_blocks)

    if segment_samples.size:
        segment_samples = segment_samples - np.mean(segment_samples)
    return _select_ms_blocks(
        segment_samples,
        sample_rate,
        block_count=max(1, int(config.integration_ms)),
        spread_blocks=spread_blocks,
    )


def _compute_heatmap_row(
    doppler: float,
    selected_blocks: np.ndarray,
    search_center_hz: float,
    time_vector: np.ndarray,
    code_fft: np.ndarray,
) -> np.ndarray:
    """Return one Doppler row of the acquisition heatmap."""

    search_frequency_hz = search_center_hz + float(doppler)
    carrier = np.exp(-1j * 2.0 * np.pi * search_frequency_hz * time_vector).astype(np.complex64)
    wiped = selected_blocks * carrier[np.newaxis, :]
    correlation = np.fft.ifft(np.fft.fft(wiped, axis=1) * code_fft[np.newaxis, :], axis=1)
    metrics = np.sum(np.abs(correlation) ** 2, axis=0)
    return metrics.astype(np.float32)


def _build_heatmap_cpu(
    selected_blocks: np.ndarray,
    doppler_bins: np.ndarray,
    search_center_hz: float,
    time_vector: np.ndarray,
    code_fft: np.ndarray,
    worker_count: int,
) -> np.ndarray:
    """Build the acquisition heatmap on the CPU."""

    effective_workers = min(max(1, int(worker_count)), int(doppler_bins.size))
    rows = parallel_ordered_map(
        list(doppler_bins),
        lambda _index, doppler: _compute_heatmap_row(
            float(doppler),
            selected_blocks,
            search_center_hz,
            time_vector,
            code_fft,
        ),
        max_workers=effective_workers,
    )
    return np.vstack(rows).astype(np.float32, copy=False)


def _build_heatmap_gpu(
    selected_blocks: np.ndarray,
    doppler_bins: np.ndarray,
    search_center_hz: float,
    time_vector: np.ndarray,
    code_fft: np.ndarray,
) -> np.ndarray:
    """Build the acquisition heatmap on an optional CuPy backend."""

    cupy = get_cupy_module()
    if cupy is None:
        return _build_heatmap_cpu(selected_blocks, doppler_bins, search_center_hz, time_vector, code_fft, 1)

    gpu_blocks = cupy.asarray(selected_blocks)
    gpu_time = cupy.asarray(time_vector)
    gpu_code_fft = cupy.asarray(code_fft)
    heatmap = cupy.zeros((doppler_bins.size, selected_blocks.shape[1]), dtype=cupy.float32)
    for row, doppler in enumerate(doppler_bins):
        search_frequency_hz = search_center_hz + float(doppler)
        carrier = cupy.exp(-1j * 2.0 * cupy.pi * search_frequency_hz * gpu_time)
        wiped = gpu_blocks * carrier[None, :]
        correlation = cupy.fft.ifft(cupy.fft.fft(wiped, axis=1) * gpu_code_fft[None, :], axis=1)
        metrics = cupy.sum(cupy.abs(correlation) ** 2, axis=0)
        heatmap[row, :] = metrics.astype(cupy.float32)
    cupy.cuda.Stream.null.synchronize()
    return np.asarray(cupy.asnumpy(heatmap), dtype=np.float32)


def _build_heatmap(
    selected_blocks: np.ndarray,
    doppler_bins: np.ndarray,
    config: AcquisitionConfig,
    time_vector: np.ndarray,
    code_fft: np.ndarray,
    log_callback=None,
) -> tuple[np.ndarray, str, int]:
    """Build one heatmap and report the active backend and worker count."""

    plan = resolve_compute_plan(
        config.compute_backend,
        config.max_workers,
        gpu_enabled=config.gpu_enabled,
        max_tasks=int(doppler_bins.size),
        prefer_gpu=True,
    )
    if plan.active_backend == "gpu":
        try:
            return (
                _build_heatmap_gpu(
                    selected_blocks,
                    doppler_bins,
                    config.search_center_hz,
                    time_vector,
                    code_fft,
                ),
                "gpu",
                plan.selected_workers,
            )
        except Exception as exc:
            if log_callback:
                log_callback(
                    f"GPU acquisition path failed for PRN {config.prn}; falling back to CPU. "
                    f"Reason: {str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__}."
                )
    return (
        _build_heatmap_cpu(
            selected_blocks,
            doppler_bins,
            config.search_center_hz,
            time_vector,
            code_fft,
            plan.selected_workers,
        ),
        "cpu",
        plan.selected_workers,
    )


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
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive for acquisition.")
    if not np.isfinite(config.search_center_hz):
        raise ValueError("Search center must be finite for acquisition.")
    if not np.isfinite(config.doppler_min) or not np.isfinite(config.doppler_max):
        raise ValueError("Doppler range must be finite for acquisition.")
    if not np.isfinite(config.doppler_step) or config.doppler_step <= 0:
        raise ValueError("Doppler step must be positive for acquisition.")
    if config.doppler_min > config.doppler_max:
        raise ValueError("Doppler minimum must not exceed Doppler maximum for acquisition.")

    samples_per_ms = int(round(sample_rate * 1e-3))
    if samples_per_ms <= 0:
        raise ValueError("Sample rate is too low for 1 ms acquisition blocks.")
    total_ms = int(np.floor(samples.size / max(sample_rate * 1e-3, 1e-9)))
    if total_ms <= 0:
        raise ValueError("Selected sample window is shorter than one 1 ms acquisition block.")

    segment_starts_ms = _select_segment_starts_ms(total_ms, config.integration_ms, config.acquisition_segment_count)
    if segment_starts_ms.size == 0:
        raise ValueError("Not enough data for one 1 ms acquisition block.")

    local_code = sample_ca_code(config.prn, sample_rate, samples_per_ms)
    code_fft = np.conj(np.fft.fft(local_code))
    doppler_bins = np.arange(config.doppler_min, config.doppler_max + config.doppler_step, config.doppler_step)
    if doppler_bins.size == 0:
        raise ValueError("Doppler search produced no bins.")
    time_vector = np.arange(samples_per_ms, dtype=np.float64) / sample_rate
    segment_plan = resolve_compute_plan(
        config.compute_backend,
        config.max_workers,
        gpu_enabled=config.gpu_enabled,
        max_tasks=int(doppler_bins.size),
        prefer_gpu=True,
    )

    best_heatmap = np.zeros((doppler_bins.size, samples_per_ms), dtype=np.float32)
    best_metric = -np.inf
    best_segment_start_ms = 0
    best_row = 0
    best_col = 0
    best_usable = 0
    segment_candidates: list[AcquisitionCandidate] = []

    if log_callback:
        gpu_text = segment_plan.gpu_name if segment_plan.gpu_available else "unavailable"
        log_callback(
            f"Acquisition PRN {config.prn}: backend {segment_plan.active_backend}, "
            f"workers {segment_plan.selected_workers}/{segment_plan.logical_cores}, GPU {gpu_text}."
        )

    for segment_index, segment_start_ms in enumerate(segment_starts_ms):
        selected_blocks = _segment_blocks_for_start(samples, sample_rate, config, int(segment_start_ms))
        usable = int(selected_blocks.shape[0])
        if usable == 0:
            continue

        heatmap, _backend, _workers = _build_heatmap(
            selected_blocks,
            doppler_bins,
            config,
            time_vector,
            code_fft,
            log_callback=log_callback,
        )
        flat = heatmap.ravel()
        flat_index = int(np.argmax(flat))
        row, col = np.unravel_index(flat_index, heatmap.shape)
        noise_floor = float(np.mean(heatmap) + 1e-12)
        metric = float(heatmap[row, col] / noise_floor)
        code_phase = int((samples_per_ms - col) % samples_per_ms)
        segment_best = AcquisitionCandidate(
            prn=config.prn,
            doppler_hz=float(doppler_bins[row]),
            carrier_frequency_hz=float(config.search_center_hz + doppler_bins[row]),
            code_phase_samples=code_phase,
            metric=float(metric),
            segment_start_sample=_sample_index_for_ms(int(segment_start_ms), sample_rate),
        )
        segment_candidates.append(segment_best)
        if metric > best_metric:
            best_metric = metric
            best_heatmap = heatmap
            best_segment_start_ms = int(segment_start_ms)
            best_row = int(row)
            best_col = int(col)
            best_usable = usable

        if progress_callback:
            progress_callback(int(100 * (segment_index + 1) / max(1, segment_starts_ms.size)))

    heatmap = best_heatmap
    noise_floor = float(np.mean(heatmap) + 1e-12)
    best = AcquisitionCandidate(
        prn=config.prn,
        doppler_hz=float(doppler_bins[best_row]),
        carrier_frequency_hz=float(config.search_center_hz + doppler_bins[best_row]),
        code_phase_samples=int((samples_per_ms - best_col) % samples_per_ms),
        metric=float(best_metric),
        segment_start_sample=_sample_index_for_ms(int(best_segment_start_ms), sample_rate),
    )

    dominant_cluster, consistency_score = _cluster_segment_candidates(
        segment_candidates,
        doppler_tolerance_hz=max(float(config.doppler_step) * 2.0, 750.0),
        code_phase_tolerance_samples=max(25, samples_per_ms // 32),
        code_period_samples=samples_per_ms,
    )
    consistent_segments = len(dominant_cluster)
    if dominant_cluster:
        dominant_cluster = sorted(dominant_cluster, key=lambda candidate: candidate.metric, reverse=True)
        best = dominant_cluster[0]
        best_segment_start_ms = int(round(best.segment_start_sample / max(sample_rate * 1e-3, 1e-9)))
        selected_blocks = _segment_blocks_for_start(samples, sample_rate, config, best_segment_start_ms)
        if selected_blocks.size:
            best_usable = int(selected_blocks.shape[0])
            heatmap, _backend, _workers = _build_heatmap(
                selected_blocks,
                doppler_bins,
                config,
                time_vector,
                code_fft,
                log_callback=log_callback,
            )

    flat = heatmap.ravel()
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
                carrier_frequency_hz=float(config.search_center_hz + doppler_bins[row]),
                code_phase_samples=code_phase,
                segment_start_sample=_sample_index_for_ms(int(best_segment_start_ms), sample_rate),
                metric=float(heatmap[row, col] / noise_floor),
            )
        )

    if log_callback:
        log_callback(
            f"Acquisition PRN {config.prn}: best peak metric {best.metric:.2f} at "
            f"search frequency {best.carrier_frequency_hz:.1f} Hz "
            f"(relative Doppler {best.doppler_hz:+.1f} Hz), code phase {best.code_phase_samples} samples, "
            f"using {best_usable} x 1 ms blocks starting at {best.segment_start_sample / sample_rate:.3f} s"
            f"{' spread across the source' if config.acquisition_segment_count == 1 and config.spread_acquisition_blocks else ''}"
            f"{' from a segmented deep search' if config.acquisition_segment_count > 1 else ''}"
            f"{', consistent in ' + str(consistent_segments) + ' segment(s)' if consistent_segments > 1 else ''}."
        )

    return AcquisitionResult(
        prn=config.prn,
        sample_rate_hz=float(sample_rate),
        search_center_hz=float(config.search_center_hz),
        doppler_bins_hz=doppler_bins.astype(np.float32),
        code_phases_samples=np.arange(samples_per_ms, dtype=np.int32),
        heatmap=heatmap,
        best_candidate=best,
        candidates=candidates,
        segment_candidates=segment_candidates,
        consistent_segments=consistent_segments,
        consistency_score=consistency_score,
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
        search_center_hz=0.0 if session.is_baseband else session.if_frequency_hz,
        doppler_min=session.doppler_min,
        doppler_max=session.doppler_max,
        doppler_step=session.doppler_step,
        integration_ms=session.integration_ms,
        spread_acquisition_blocks=session.spread_acquisition_blocks,
        acquisition_segment_count=session.acquisition_segment_count,
        compute_backend=session.compute_backend,
        max_workers=session.max_workers,
        gpu_enabled=session.gpu_enabled,
    )
    return acquire_signal(samples, config, progress_callback=progress_callback, log_callback=log_callback)


def scan_prns_from_session(
    samples: np.ndarray,
    session: SessionConfig,
    prns: list[int] | None = None,
    progress_callback=None,
    log_callback=None,
) -> list[AcquisitionResult]:
    """Run acquisition for multiple PRNs and return the per-satellite results."""

    prn_list = prns or list(range(1, 33))
    total = max(len(prn_list), 1)
    outer_plan = resolve_compute_plan(
        session.compute_backend,
        session.max_workers,
        gpu_enabled=session.gpu_enabled,
        max_tasks=total,
        prefer_gpu=True,
    )
    parallel_outer = outer_plan.active_backend == "cpu" and outer_plan.selected_workers > 1 and total > 1
    results: list[AcquisitionResult] = []

    if log_callback:
        gpu_text = outer_plan.gpu_name if outer_plan.gpu_available else "unavailable"
        log_callback(
            f"PRN scan: backend {outer_plan.active_backend}, workers {outer_plan.selected_workers}/{outer_plan.logical_cores}, GPU {gpu_text}."
        )

    if parallel_outer:
        tracker = ProgressTracker(total, progress_callback)
        worker_session = replace(session, compute_backend="cpu", max_workers=1, gpu_enabled=False)

        def run_one(index: int, prn: int) -> AcquisitionResult:
            local_session = replace(worker_session, prn=prn)
            return acquisition_from_session(
                samples,
                local_session,
                progress_callback=lambda value, idx=index: tracker.update(idx, value),
                log_callback=log_callback,
            )

        results = parallel_ordered_map(
            prn_list,
            run_one,
            max_workers=outer_plan.selected_workers,
        )
    else:
        for index, prn in enumerate(prn_list):
            local_session = replace(session, prn=prn)

            def nested_progress(local_progress: int, *, base=index) -> None:
                if progress_callback:
                    combined = int(((base + local_progress / 100.0) / total) * 100)
                    progress_callback(combined)

            result = acquisition_from_session(samples, local_session, progress_callback=nested_progress, log_callback=log_callback)
            results.append(result)

    results.sort(
        key=acquisition_rank_key,
        reverse=True,
    )
    if progress_callback:
        progress_callback(100)
    return results


def sweep_search_centers_from_session(
    samples: np.ndarray,
    session: SessionConfig,
    search_centers_hz: list[float],
    prns: list[int] | None = None,
    progress_callback=None,
    log_callback=None,
) -> SearchCenterSweepResult:
    """Sweep several IF / search-center hypotheses and rank the best one."""

    centers = search_centers_hz or [0.0]
    prn_list = prns or list(range(1, 33))
    total = max(len(centers), 1)
    entries: list[SearchCenterSweepEntry] = []
    outer_plan = resolve_compute_plan(
        session.compute_backend,
        session.max_workers,
        gpu_enabled=session.gpu_enabled,
        max_tasks=total * max(1, len(prn_list)),
        prefer_gpu=True,
    )
    outer_workers, inner_workers = split_nested_worker_budget(
        outer_plan.selected_workers,
        outer_tasks=total,
        inner_tasks=max(1, len(prn_list)),
    )
    parallel_outer = outer_plan.active_backend == "cpu" and outer_workers > 1 and total > 1

    if log_callback:
        gpu_text = outer_plan.gpu_name if outer_plan.gpu_available else "unavailable"
        log_callback(
            f"Search-center sweep: backend {outer_plan.active_backend}, workers {outer_plan.selected_workers}/{outer_plan.logical_cores}, "
            f"outer centers {outer_workers}, inner PRNs {inner_workers}, GPU {gpu_text}."
        )

    if parallel_outer:
        tracker = ProgressTracker(total, progress_callback)
        worker_session = replace(session, compute_backend="cpu", max_workers=inner_workers, gpu_enabled=False)

        def run_one(index: int, center_hz: float) -> SearchCenterSweepEntry:
            local_session = replace(worker_session)
            local_session.is_baseband = abs(center_hz) < 1e-9
            local_session.if_frequency_hz = float(center_hz)
            results = scan_prns_from_session(
                samples,
                local_session,
                prns=prn_list,
                progress_callback=lambda value, idx=index: tracker.update(idx, value),
                log_callback=None,
            )
            best = results[0]
            if log_callback:
                log_callback(
                    f"Search-center sweep {center_hz:.1f} Hz: best PRN {best.prn}, "
                    f"metric {best.best_candidate.metric:.2f}, search frequency {best.best_candidate.carrier_frequency_hz:.1f} Hz."
                )
            return SearchCenterSweepEntry(search_center_hz=float(center_hz), best_result=best)

        entries = parallel_ordered_map(
            centers,
            run_one,
            max_workers=outer_workers,
        )
    else:
        for center_index, center_hz in enumerate(centers):
            local_session = replace(session)
            local_session.is_baseband = abs(center_hz) < 1e-9
            local_session.if_frequency_hz = float(center_hz)
            if outer_plan.active_backend == "cpu":
                local_session.compute_backend = "cpu"
                local_session.max_workers = inner_workers
                local_session.gpu_enabled = False

            def nested_progress(local_progress: int, *, base=center_index) -> None:
                if progress_callback:
                    combined = int(((base + local_progress / 100.0) / total) * 100)
                    progress_callback(combined)

            results = scan_prns_from_session(
                samples,
                local_session,
                prns=prn_list,
                progress_callback=nested_progress,
                log_callback=None,
            )
            best = results[0]
            entries.append(SearchCenterSweepEntry(search_center_hz=float(center_hz), best_result=best))
            if log_callback:
                log_callback(
                    f"Search-center sweep {center_hz:.1f} Hz: best PRN {best.prn}, "
                    f"metric {best.best_candidate.metric:.2f}, search frequency {best.best_candidate.carrier_frequency_hz:.1f} Hz."
                )

    entries.sort(
        key=lambda item: acquisition_rank_key(item.best_result),
        reverse=True,
    )
    if progress_callback:
        progress_callback(100)
    return SearchCenterSweepResult(entries=entries)


def survey_sample_rates(
    samples: np.ndarray,
    session: SessionConfig,
    sample_rates_hz: list[float],
    prns: list[int] | None = None,
    progress_callback=None,
    log_callback=None,
) -> SampleRateSurveyResult:
    """Rank several sample-rate hypotheses for one recording."""

    rates = sample_rates_hz or [session.sample_rate]
    total = max(len(rates), 1)
    entries: list[SampleRateSurveyEntry] = []
    prn_list = prns or list(range(1, 33))
    outer_plan = resolve_compute_plan(
        session.compute_backend,
        session.max_workers,
        gpu_enabled=session.gpu_enabled,
        max_tasks=total * max(1, len(prn_list)),
        prefer_gpu=True,
    )
    outer_workers, inner_workers = split_nested_worker_budget(
        outer_plan.selected_workers,
        outer_tasks=total,
        inner_tasks=max(1, len(prn_list)),
    )
    parallel_outer = outer_plan.active_backend == "cpu" and outer_workers > 1 and total > 1

    if log_callback:
        gpu_text = outer_plan.gpu_name if outer_plan.gpu_available else "unavailable"
        log_callback(
            f"Sample-rate survey: backend {outer_plan.active_backend}, workers {outer_plan.selected_workers}/{outer_plan.logical_cores}, "
            f"outer rates {outer_workers}, inner PRNs {inner_workers}, GPU {gpu_text}."
        )

    def build_local_session(base_session: SessionConfig, sample_rate_hz: float) -> SessionConfig:
        local_session = replace(base_session)
        local_session.sample_rate = float(sample_rate_hz)
        local_session.doppler_min = min(int(base_session.doppler_min), -12_000)
        local_session.doppler_max = max(int(base_session.doppler_max), 12_000)
        local_session.doppler_step = max(int(base_session.doppler_step), 500)
        local_session.integration_ms = min(max(int(base_session.integration_ms), 20), 40)
        local_session.acquisition_segment_count = max(int(base_session.acquisition_segment_count), 4)
        local_session.spread_acquisition_blocks = False
        return local_session

    if parallel_outer:
        tracker = ProgressTracker(total, progress_callback)
        worker_session = replace(session, compute_backend="cpu", max_workers=inner_workers, gpu_enabled=False)

        def run_one(index: int, sample_rate_hz: float) -> SampleRateSurveyEntry:
            local_session = build_local_session(worker_session, sample_rate_hz)
            results = scan_prns_from_session(
                samples,
                local_session,
                prns=prn_list,
                progress_callback=lambda value, idx=index: tracker.update(idx, value),
                log_callback=None,
            )
            best = results[0]
            if log_callback:
                log_callback(
                    f"Sample-rate survey {sample_rate_hz / 1e6:.3f} MSa/s: best PRN {best.prn}, "
                    f"metric {best.best_candidate.metric:.2f}, consistent segments {best.consistent_segments}, "
                    f"search frequency {best.best_candidate.carrier_frequency_hz:.1f} Hz."
                )
            return SampleRateSurveyEntry(
                sample_rate_hz=float(sample_rate_hz),
                best_result=best,
                all_results=results,
            )

        entries = parallel_ordered_map(
            rates,
            run_one,
            max_workers=outer_workers,
        )
    else:
        for rate_index, sample_rate_hz in enumerate(rates):
            local_session = build_local_session(session, sample_rate_hz)
            if outer_plan.active_backend == "cpu":
                local_session.compute_backend = "cpu"
                local_session.max_workers = inner_workers
                local_session.gpu_enabled = False

            def nested_progress(local_progress: int, *, base=rate_index) -> None:
                if progress_callback:
                    combined = int(((base + local_progress / 100.0) / total) * 100)
                    progress_callback(combined)

            results = scan_prns_from_session(
                samples,
                local_session,
                prns=prn_list,
                progress_callback=nested_progress,
                log_callback=None,
            )
            best = results[0]
            entries.append(
                SampleRateSurveyEntry(
                    sample_rate_hz=float(sample_rate_hz),
                    best_result=best,
                    all_results=results,
                )
            )
            if log_callback:
                log_callback(
                    f"Sample-rate survey {sample_rate_hz / 1e6:.3f} MSa/s: best PRN {best.prn}, "
                    f"metric {best.best_candidate.metric:.2f}, consistent segments {best.consistent_segments}, "
                    f"search frequency {best.best_candidate.carrier_frequency_hz:.1f} Hz."
                )

    entries.sort(
        key=lambda entry: acquisition_rank_key(entry.best_result),
        reverse=True,
    )
    if progress_callback:
        progress_callback(100)
    return SampleRateSurveyResult(entries=entries)
