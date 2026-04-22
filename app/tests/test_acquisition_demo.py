"""Acquisition should find the demo signal near its truth values."""

from __future__ import annotations

from app.dsp.acquisition import AcquisitionConfig, acquire_signal
from app.dsp.demo import generate_demo_signal


def test_acquisition_finds_demo_signal() -> None:
    demo = generate_demo_signal(duration_s=0.1, doppler_hz=1750.0, code_phase_samples=250)
    result = acquire_signal(
        demo.samples,
        AcquisitionConfig(
            sample_rate=demo.sample_rate,
            prn=demo.prn,
            doppler_min=-3000,
            doppler_max=3000,
            doppler_step=250,
            integration_ms=4,
        ),
    )
    assert abs(result.best_candidate.doppler_hz - demo.doppler_hz) <= 500.0
    assert abs(result.best_candidate.code_phase_samples - demo.code_phase_samples) <= 40
