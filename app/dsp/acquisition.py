"""Simple acquisition search over Doppler and code phase."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

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
        start = int(start_ms) * samples_per_ms
        stop = start + samples_per_ms
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
) -> tuple[list[AcquisitionCandidate], float]:
    """Find the dominant cluster of repeated PRN hits across searched segments."""

    if not candidates:
        return [], 0.0

    best_cluster: list[AcquisitionCandidate] = []
    best_score = -np.inf
    for seed in candidates:
        cluster = [
            candidate
            for candidate in candidates
            if abs(candidate.doppler_hz - seed.doppler_hz) <= doppler_tolerance_hz
            and abs(candidate.code_phase_samples - seed.code_phase_samples) <= code_phase_tolerance_samples
        ]
        score = float(len(cluster)) * float(np.mean([candidate.metric for candidate in cluster]))
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
        return "repeated / plausible"
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
    total_ms = samples.size // samples_per_ms
    segment_starts_ms = _select_segment_starts_ms(total_ms, config.integration_ms, config.acquisition_segment_count)
    if segment_starts_ms.size == 0:
        raise ValueError("Not enough data for one 1 ms acquisition block.")

    local_code = sample_ca_code(config.prn, sample_rate, samples_per_ms)
    code_fft = np.conj(np.fft.fft(local_code))
    doppler_bins = np.arange(config.doppler_min, config.doppler_max + config.doppler_step, config.doppler_step)
    time_vector = np.arange(samples_per_ms, dtype=np.float64) / sample_rate

    best_heatmap = np.zeros((doppler_bins.size, samples_per_ms), dtype=np.float32)
    best_metric = -np.inf
    best_segment_start_ms = 0
    best_row = 0
    best_col = 0
    best_usable = 0
    segment_candidates: list[AcquisitionCandidate] = []

    for segment_index, segment_start_ms in enumerate(segment_starts_ms):
        segment_start = int(segment_start_ms) * samples_per_ms
        if config.acquisition_segment_count > 1:
            segment_stop = min(samples.size, segment_start + max(1, int(config.integration_ms)) * samples_per_ms)
            segment_samples = np.asarray(samples[segment_start:segment_stop], dtype=np.complex64)
            segment_spread = False
        else:
            segment_samples = np.asarray(samples[segment_start:], dtype=np.complex64)
            segment_spread = bool(config.spread_acquisition_blocks)

        if segment_samples.size:
            segment_samples = segment_samples - np.mean(segment_samples)

        selected_blocks = _select_ms_blocks(
            segment_samples,
            sample_rate,
            block_count=max(1, int(config.integration_ms)),
            spread_blocks=segment_spread,
        )
        usable = int(selected_blocks.shape[0])
        if usable == 0:
            continue

        heatmap = np.zeros((doppler_bins.size, samples_per_ms), dtype=np.float32)
        for row, doppler in enumerate(doppler_bins):
            search_frequency_hz = config.search_center_hz + float(doppler)
            metrics = np.zeros(samples_per_ms, dtype=np.float64)
            for block_index in range(usable):
                block = selected_blocks[block_index]
                wiped = block * np.exp(-1j * 2.0 * np.pi * search_frequency_hz * time_vector)
                correlation = np.fft.ifft(np.fft.fft(wiped) * code_fft)
                metrics += np.abs(correlation) ** 2
            heatmap[row, :] = metrics.astype(np.float32)

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
            segment_start_sample=int(segment_start_ms * samples_per_ms),
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
        segment_start_sample=int(best_segment_start_ms * samples_per_ms),
    )

    dominant_cluster, consistency_score = _cluster_segment_candidates(
        segment_candidates,
        doppler_tolerance_hz=max(float(config.doppler_step) * 2.0, 750.0),
        code_phase_tolerance_samples=max(25, samples_per_ms // 32),
    )
    consistent_segments = len(dominant_cluster)
    if dominant_cluster:
        dominant_cluster = sorted(dominant_cluster, key=lambda candidate: candidate.metric, reverse=True)
        best = dominant_cluster[0]
        dominant_segment_start = best.segment_start_sample // samples_per_ms
        best_segment_start_ms = int(dominant_segment_start)
        for segment_index, segment_start_ms in enumerate(segment_starts_ms):
            if int(segment_start_ms) == dominant_segment_start:
                # The heatmap should reflect the segment used for the chosen best candidate.
                segment_start = int(segment_start_ms) * samples_per_ms
                if config.acquisition_segment_count > 1:
                    segment_stop = min(samples.size, segment_start + max(1, int(config.integration_ms)) * samples_per_ms)
                    segment_samples = np.asarray(samples[segment_start:segment_stop], dtype=np.complex64)
                    segment_spread = False
                else:
                    segment_samples = np.asarray(samples[segment_start:], dtype=np.complex64)
                    segment_spread = bool(config.spread_acquisition_blocks)
                if segment_samples.size:
                    segment_samples = segment_samples - np.mean(segment_samples)
                selected_blocks = _select_ms_blocks(
                    segment_samples,
                    sample_rate,
                    block_count=max(1, int(config.integration_ms)),
                    spread_blocks=segment_spread,
                )
                if selected_blocks.size:
                    cluster_heatmap = np.zeros((doppler_bins.size, samples_per_ms), dtype=np.float32)
                    for row, doppler in enumerate(doppler_bins):
                        search_frequency_hz = config.search_center_hz + float(doppler)
                        metrics = np.zeros(samples_per_ms, dtype=np.float64)
                        for block_index in range(int(selected_blocks.shape[0])):
                            block = selected_blocks[block_index]
                            wiped = block * np.exp(-1j * 2.0 * np.pi * search_frequency_hz * time_vector)
                            correlation = np.fft.ifft(np.fft.fft(wiped) * code_fft)
                            metrics += np.abs(correlation) ** 2
                        cluster_heatmap[row, :] = metrics.astype(np.float32)
                    heatmap = cluster_heatmap
                break

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
                segment_start_sample=int(best_segment_start_ms * samples_per_ms),
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
    results: list[AcquisitionResult] = []
    total = max(len(prn_list), 1)

    for index, prn in enumerate(prn_list):
        config = AcquisitionConfig(
            sample_rate=session.sample_rate,
            prn=prn,
            search_center_hz=0.0 if session.is_baseband else session.if_frequency_hz,
            doppler_min=session.doppler_min,
            doppler_max=session.doppler_max,
            doppler_step=session.doppler_step,
            integration_ms=session.integration_ms,
            spread_acquisition_blocks=session.spread_acquisition_blocks,
            acquisition_segment_count=session.acquisition_segment_count,
        )

        def nested_progress(local_progress: int, *, base=index) -> None:
            if progress_callback:
                combined = int(((base + local_progress / 100.0) / total) * 100)
                progress_callback(combined)

        result = acquire_signal(samples, config, progress_callback=nested_progress, log_callback=log_callback)
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
    entries: list[SearchCenterSweepEntry] = []
    total = max(len(centers), 1)

    for center_index, center_hz in enumerate(centers):
        local_session = replace(session)
        local_session.is_baseband = abs(center_hz) < 1e-9
        local_session.if_frequency_hz = float(center_hz)

        def nested_progress(local_progress: int, *, base=center_index) -> None:
            if progress_callback:
                combined = int(((base + local_progress / 100.0) / total) * 100)
                progress_callback(combined)

        results = scan_prns_from_session(
            samples,
            local_session,
            prns=prns,
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
    entries: list[SampleRateSurveyEntry] = []
    total = max(len(rates), 1)
    prn_list = prns or list(range(1, 33))

    for rate_index, sample_rate_hz in enumerate(rates):
        local_session = replace(session)
        local_session.sample_rate = float(sample_rate_hz)
        local_session.doppler_min = min(int(session.doppler_min), -20_000)
        local_session.doppler_max = max(int(session.doppler_max), 20_000)
        local_session.doppler_step = max(int(session.doppler_step), 500)
        local_session.integration_ms = min(max(int(session.integration_ms), 20), 80)
        local_session.acquisition_segment_count = max(int(session.acquisition_segment_count), 6)
        local_session.spread_acquisition_blocks = False

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
