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


@pytest.mark.parametrize(
    "prn, doppler_hz, code_phase_samples",
    [
        (1, 1500.0, 300),
        (2, 500.0, 500),
        (3, 2250.0, 700),
        (8, -1750.0, 1200),
        (10, -2000.0, 800),
        (22, 500.0, 1800),
        (32, -2500.0, 450),
    ],
)
def test_acquisition_preserves_prn_doppler_and_codephase_identity(
    prn: int,
    doppler_hz: float,
    code_phase_samples: int,
) -> None:
    demo = generate_demo_signal(
        sample_rate=2_046_000.0,
        duration_s=0.08,
        prn=prn,
        doppler_hz=doppler_hz,
        code_phase_samples=code_phase_samples,
        cn0_like_amplitude=1.0,
        noise_std=0.08,
    )

    result = acquire_signal(
        demo.samples,
        AcquisitionConfig(
            sample_rate=demo.sample_rate,
            prn=prn,
            doppler_min=-3000,
            doppler_max=3000,
            doppler_step=250,
            integration_ms=8,
            acquisition_segment_count=3,
            spread_acquisition_blocks=False,
            compute_backend="cpu",
            max_workers=2,
            gpu_enabled=False,
        ),
    )

    assert result.prn == prn
    assert result.best_candidate.prn == prn
    assert result.best_candidate.doppler_hz == doppler_hz
    assert result.best_candidate.code_phase_samples == code_phase_samples
    assert result.consistent_segments == 3


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


def test_acquisition_rejects_empty_doppler_range() -> None:
    demo = generate_demo_signal(duration_s=0.01)

    with pytest.raises(ValueError, match="Doppler minimum"):
        acquire_signal(
            demo.samples,
            AcquisitionConfig(
                sample_rate=demo.sample_rate,
                prn=demo.prn,
                doppler_min=1000,
                doppler_max=-1000,
                doppler_step=250,
            ),
        )
