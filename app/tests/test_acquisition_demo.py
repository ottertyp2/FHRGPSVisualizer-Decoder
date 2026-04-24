"""Acquisition should find the demo signal near its truth values."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_acquisition_rejects_windows_shorter_than_one_ms() -> None:
    samples = np.zeros(999, dtype=np.complex64)
    with pytest.raises(ValueError, match="shorter than one 1 ms acquisition block"):
        acquire_signal(
            samples,
            AcquisitionConfig(
                sample_rate=1_000_000.0,
                prn=1,
                doppler_min=-500,
                doppler_max=500,
                doppler_step=500,
            ),
        )
