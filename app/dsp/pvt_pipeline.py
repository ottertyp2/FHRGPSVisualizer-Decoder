"""Small end-to-end helper for the PVT tab.

This module wires together existing DSP stages without hiding them: acquire
several PRNs, track each channel, decode LNAV, then call the PVT builder.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.dsp.acquisition import acquisition_result_is_plausible, scan_prns_from_session
from app.dsp.io import Complex64FileSource
from app.dsp.navdecode import decode_navigation_from_tracking
from app.dsp.pvt import PVTComputationResult, compute_pvt_from_navigation
from app.dsp.tracking import track_file
from app.models import (
    AcquisitionResult,
    BitDecisionResult,
    NavigationDecodeResult,
    SessionConfig,
    TrackingState,
)


@dataclass(slots=True)
class PVTPipelineResult:
    """Outputs from the automated PVT teaching pipeline."""

    pvt_result: PVTComputationResult
    acquisition_results: list[AcquisitionResult]
    tracking_results_by_prn: dict[int, TrackingState]
    bit_results_by_prn: dict[int, BitDecisionResult]
    nav_results_by_prn: dict[int, NavigationDecodeResult]


def _pvt_session(
    base_session: SessionConfig,
    *,
    start_sample: int,
    sample_count: int,
    tracking_ms: int,
) -> SessionConfig:
    session = replace(base_session)
    session.start_sample = int(start_sample)
    session.sample_count = int(sample_count)
    session.tracking_ms = int(tracking_ms)
    session.doppler_min = min(int(session.doppler_min), -12_000)
    session.doppler_max = max(int(session.doppler_max), 12_000)
    session.doppler_step = max(int(session.doppler_step), 500)
    session.integration_ms = min(max(int(session.integration_ms), 40), 80)
    session.acquisition_segment_count = max(int(session.acquisition_segment_count), 4)
    session.spread_acquisition_blocks = False
    return session


def run_pvt_pipeline(
    file_path: str,
    session: SessionConfig,
    *,
    start_time_s: float = 60.0,
    acquisition_window_s: float = 3.0,
    tracking_s: float = 60.0,
    max_satellites: int = 8,
    progress_callback=None,
    log_callback=None,
) -> PVTPipelineResult:
    """Acquire, track, decode, and solve PVT from one file window."""

    if session.sample_rate <= 0.0:
        raise ValueError("Sample rate must be positive for PVT.")

    source = Complex64FileSource(file_path)
    start_sample = int(round(max(0.0, float(start_time_s)) * float(session.sample_rate)))
    sample_count = int(round(max(1.0, float(acquisition_window_s)) * float(session.sample_rate)))
    sample_count = min(sample_count, max(0, source.total_samples - start_sample))
    if sample_count <= 0:
        raise ValueError("The requested PVT acquisition window is outside the file.")

    if log_callback:
        log_callback(
            f"PVT acquisition window: start {start_sample / session.sample_rate:.3f} s, "
            f"{sample_count / session.sample_rate:.3f} s long."
        )
    samples = source.read_window(start_sample, sample_count)
    pvt_session = _pvt_session(
        session,
        start_sample=start_sample,
        sample_count=sample_count,
        tracking_ms=int(round(max(1.0, float(tracking_s)) * 1_000.0)),
    )

    def acquisition_progress(value: int) -> None:
        if progress_callback:
            progress_callback(int(value * 0.25))

    acquisition_results = scan_prns_from_session(
        samples,
        pvt_session,
        prns=list(range(1, 33)),
        progress_callback=acquisition_progress,
        log_callback=log_callback,
    )
    candidates = [result for result in acquisition_results if acquisition_result_is_plausible(result)]
    candidates = candidates[: max(1, int(max_satellites))]
    if log_callback:
        if candidates:
            log_callback(
                "PVT candidates: "
                + ", ".join(
                    f"PRN {result.prn} metric {result.best_candidate.metric:.1f}"
                    for result in candidates
                )
                + "."
            )
        else:
            log_callback("PVT did not find enough repeated strong acquisition candidates.")

    tracking_results_by_prn: dict[int, TrackingState] = {}
    bit_results_by_prn: dict[int, BitDecisionResult] = {}
    nav_results_by_prn: dict[int, NavigationDecodeResult] = {}
    total_candidates = max(1, len(candidates))
    for index, acquisition in enumerate(candidates):
        local_session = replace(pvt_session, prn=acquisition.prn)
        absolute_start = start_sample + int(acquisition.best_candidate.segment_start_sample)
        if log_callback:
            log_callback(
                f"PVT tracking PRN {acquisition.prn}: file time "
                f"{absolute_start / session.sample_rate:.3f} s."
            )

        def channel_progress(value: int, *, base=index) -> None:
            if progress_callback:
                progress_callback(25 + int(((base + value / 100.0) / total_candidates) * 65))

        tracking = track_file(
            file_path,
            absolute_start,
            local_session,
            acquisition,
            progress_callback=channel_progress,
            log_callback=log_callback,
        )
        bit_result, nav_result = decode_navigation_from_tracking(
            tracking,
            progress_callback=None,
            log_callback=log_callback,
        )
        tracking_results_by_prn[acquisition.prn] = tracking
        bit_results_by_prn[acquisition.prn] = bit_result
        nav_results_by_prn[acquisition.prn] = nav_result

    if progress_callback:
        progress_callback(92)
    pvt_result = compute_pvt_from_navigation(
        tracking_results_by_prn,
        bit_results_by_prn,
        nav_results_by_prn,
    )
    if progress_callback:
        progress_callback(100)
    return PVTPipelineResult(
        pvt_result=pvt_result,
        acquisition_results=acquisition_results,
        tracking_results_by_prn=tracking_results_by_prn,
        bit_results_by_prn=bit_results_by_prn,
        nav_results_by_prn=nav_results_by_prn,
    )
