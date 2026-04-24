"""Concept lab synthetic signal helpers."""

from __future__ import annotations

import numpy as np

from app.dsp.concept_lab import ConceptLabConfig, generate_concept_lab_signal


def test_concept_lab_correlation_peaks_at_selected_code_phase() -> None:
    config = ConceptLabConfig(
        duration_ms=4,
        prn=3,
        doppler_hz=1_000.0,
        code_phase_samples=120,
        noise_std=0.0,
        second_prn_enabled=False,
    )

    result = generate_concept_lab_signal(config)

    peak_phase = int(result.correlation_code_phases[np.argmax(result.correlation_values)])
    best_doppler = float(
        result.acquisition_doppler_bins_hz[np.argmax(np.max(result.acquisition_heatmap, axis=1))]
    )
    assert peak_phase == config.code_phase_samples
    assert abs(best_doppler - config.doppler_hz) <= 1e-6
    assert result.prompt_points.size == config.duration_ms


def test_concept_lab_same_doppler_prns_remain_code_separable() -> None:
    config = ConceptLabConfig(
        duration_ms=4,
        prn=1,
        doppler_hz=500.0,
        code_phase_samples=80,
        noise_std=0.0,
        second_prn_enabled=True,
        second_prn=9,
        second_code_phase_samples=330,
    )

    result = generate_concept_lab_signal(config)

    selected_peak = float(np.max(result.correlation_values))
    median_floor = float(np.median(result.correlation_values))
    assert selected_peak > median_floor * 8.0
    assert result.acquisition_heatmap.shape[0] == result.acquisition_doppler_bins_hz.size
    assert result.acquisition_heatmap.shape[1] == result.correlation_code_phases.size
