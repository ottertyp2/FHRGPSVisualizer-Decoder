"""Confidence and ranking helpers for noisy acquisition results."""

from __future__ import annotations

import numpy as np

from app.dsp.acquisition import _cluster_segment_candidates
from app.dsp.acquisition import acquisition_interpretation, acquisition_rank_key
from app.models import AcquisitionCandidate, AcquisitionResult


def _make_result(metric: float, consistent_segments: int, consistency_score: float) -> AcquisitionResult:
    candidate = AcquisitionCandidate(
        prn=1,
        doppler_hz=1500.0,
        carrier_frequency_hz=1500.0,
        code_phase_samples=200,
        metric=metric,
        segment_start_sample=0,
    )
    return AcquisitionResult(
        prn=1,
        sample_rate_hz=2_000_000.0,
        search_center_hz=0.0,
        doppler_bins_hz=np.asarray([-1500.0, 0.0, 1500.0], dtype=np.float32),
        code_phases_samples=np.arange(3, dtype=np.int32),
        heatmap=np.zeros((3, 3), dtype=np.float32),
        best_candidate=candidate,
        consistent_segments=consistent_segments,
        consistency_score=consistency_score,
    )


def test_weak_results_rank_by_metric_before_consistency() -> None:
    weak_repeated = _make_result(metric=1.90, consistent_segments=4, consistency_score=7.60)
    weak_stronger = _make_result(metric=2.10, consistent_segments=3, consistency_score=6.30)

    ranked = sorted([weak_repeated, weak_stronger], key=acquisition_rank_key, reverse=True)

    assert ranked[0] is weak_stronger


def test_strong_results_still_rank_by_consistency() -> None:
    strong_single = _make_result(metric=8.50, consistent_segments=1, consistency_score=8.50)
    strong_repeated = _make_result(metric=6.50, consistent_segments=4, consistency_score=26.00)

    ranked = sorted([strong_single, strong_repeated], key=acquisition_rank_key, reverse=True)

    assert ranked[0] is strong_repeated


def test_interpretation_requires_strong_metric_for_plausible_label() -> None:
    weak_repeated = _make_result(metric=1.95, consistent_segments=4, consistency_score=7.80)
    strong_repeated = _make_result(metric=6.20, consistent_segments=4, consistency_score=24.80)

    assert acquisition_interpretation(weak_repeated) == "repeated but still weak"
    assert acquisition_interpretation(strong_repeated) == "repeated / plausible"


def test_segment_consistency_allows_smooth_code_phase_drift() -> None:
    candidates = [
        AcquisitionCandidate(
            prn=30,
            doppler_hz=250.0,
            carrier_frequency_hz=250.0,
            code_phase_samples=(900 + index * 280) % 6061,
            metric=20.0,
            segment_start_sample=index * 4_294_000,
        )
        for index in range(6)
    ]

    cluster, score = _cluster_segment_candidates(
        candidates,
        doppler_tolerance_hz=750.0,
        code_phase_tolerance_samples=190,
        code_period_samples=6061,
    )

    assert len(cluster) == len(candidates)
    assert score > 0.0
