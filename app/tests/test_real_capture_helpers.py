"""Helpers that improve robustness on real recorded captures."""

from __future__ import annotations

from app.dsp.acquisition import AcquisitionConfig, acquire_signal, survey_sample_rates
from app.dsp.demo import generate_demo_signal
from app.models import SessionConfig


def test_segment_consistency_is_reported_for_demo_signal() -> None:
    demo = generate_demo_signal(
        sample_rate=2_046_000.0,
        duration_s=0.25,
        doppler_hz=1750.0,
        code_phase_samples=220,
    )
    result = acquire_signal(
        demo.samples,
        AcquisitionConfig(
            sample_rate=demo.sample_rate,
            prn=demo.prn,
            doppler_min=-4000,
            doppler_max=4000,
            doppler_step=250,
            integration_ms=20,
            acquisition_segment_count=4,
            spread_acquisition_blocks=False,
        ),
    )
    assert result.consistent_segments >= 2
    assert result.consistency_score > 0.0


def test_sample_rate_survey_prefers_the_true_rate() -> None:
    demo = generate_demo_signal(
        sample_rate=2_046_000.0,
        duration_s=0.25,
        doppler_hz=1500.0,
        code_phase_samples=180,
        prn=7,
    )
    session = SessionConfig(
        sample_rate=6_000_000.0,
        prn=demo.prn,
        doppler_min=-4000,
        doppler_max=4000,
        doppler_step=250,
        integration_ms=20,
        acquisition_segment_count=4,
    )
    survey = survey_sample_rates(
        demo.samples,
        session,
        [2_000_000.0, 2_046_000.0, 6_000_000.0],
        prns=[7, 8, 9],
    )
    assert survey.entries
    assert survey.entries[0].sample_rate_hz == demo.sample_rate
    assert survey.entries[0].best_result.prn == demo.prn
