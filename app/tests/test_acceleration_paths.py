"""Equivalence tests for threaded and optional accelerated DSP paths."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

import app.dsp.tracking as tracking_module
from app.dsp.acquisition import AcquisitionConfig, acquisition_interpretation, acquire_signal
from app.dsp.compute import ComputePlan, detect_gpu_info
from app.dsp.demo import generate_demo_signal
from app.dsp.utils import compute_spectrum, compute_waterfall
from app.models import SessionConfig


def test_acquisition_matches_between_single_and_multi_worker_cpu() -> None:
    demo = generate_demo_signal(duration_s=0.12, doppler_hz=1750.0, code_phase_samples=250, prn=5)
    base_config = AcquisitionConfig(
        sample_rate=demo.sample_rate,
        prn=demo.prn,
        doppler_min=-3000,
        doppler_max=3000,
        doppler_step=250,
        integration_ms=4,
        compute_backend="cpu",
        gpu_enabled=False,
    )

    single = acquire_signal(demo.samples, base_config)
    multi = acquire_signal(demo.samples, replace(base_config, max_workers=4))

    assert multi.best_candidate.prn == single.best_candidate.prn
    assert multi.best_candidate.doppler_hz == single.best_candidate.doppler_hz
    assert multi.best_candidate.code_phase_samples == single.best_candidate.code_phase_samples
    assert acquisition_interpretation(multi) == acquisition_interpretation(single)
    np.testing.assert_allclose(multi.heatmap, single.heatmap, rtol=1e-6, atol=1e-6)


def test_fft_helpers_match_between_single_and_multi_worker_cpu() -> None:
    demo = generate_demo_signal(duration_s=0.2, prn=3)
    samples = demo.samples[:32_768]
    single_session = SessionConfig(sample_rate=demo.sample_rate, compute_backend="cpu", max_workers=1, gpu_enabled=False)
    multi_session = SessionConfig(sample_rate=demo.sample_rate, compute_backend="cpu", max_workers=4, gpu_enabled=False)

    freqs_single, spectrum_single = compute_spectrum(samples, demo.sample_rate, 4096, "hann", 4, session=single_session)
    freqs_multi, spectrum_multi = compute_spectrum(samples, demo.sample_rate, 4096, "hann", 4, session=multi_session)
    np.testing.assert_allclose(freqs_multi, freqs_single)
    np.testing.assert_allclose(spectrum_multi, spectrum_single, rtol=1e-6, atol=1e-6)

    wf_freqs_single, wf_times_single, waterfall_single = compute_waterfall(
        samples,
        demo.sample_rate,
        fft_size=2048,
        window_name="hann",
        max_rows=12,
        session=single_session,
    )
    wf_freqs_multi, wf_times_multi, waterfall_multi = compute_waterfall(
        samples,
        demo.sample_rate,
        fft_size=2048,
        window_name="hann",
        max_rows=12,
        session=multi_session,
    )
    np.testing.assert_allclose(wf_freqs_multi, wf_freqs_single)
    np.testing.assert_allclose(wf_times_multi, wf_times_single)
    np.testing.assert_allclose(waterfall_multi, waterfall_single, rtol=1e-6, atol=1e-6)


def test_acquisition_falls_back_to_cpu_when_gpu_build_fails(monkeypatch) -> None:
    demo = generate_demo_signal(duration_s=0.08, doppler_hz=1500.0, code_phase_samples=180, prn=7)
    config = AcquisitionConfig(
        sample_rate=demo.sample_rate,
        prn=demo.prn,
        doppler_min=-3000,
        doppler_max=3000,
        doppler_step=250,
        integration_ms=4,
        compute_backend="gpu",
        max_workers=1,
        gpu_enabled=True,
    )

    monkeypatch.setattr(
        "app.dsp.acquisition.resolve_compute_plan",
        lambda *args, **kwargs: type(
            "Plan",
            (),
            {
                "active_backend": "gpu",
                "selected_workers": 1,
                "logical_cores": 8,
                "gpu_available": True,
                "gpu_name": "Fake GPU",
            },
        )(),
    )
    monkeypatch.setattr("app.dsp.acquisition._build_heatmap_gpu", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("GPU blew up")))

    result = acquire_signal(demo.samples, config)

    assert result.best_candidate.prn == demo.prn
    assert result.heatmap.size > 0


def test_tracking_falls_back_to_cpu_when_gpu_build_fails(monkeypatch) -> None:
    demo = generate_demo_signal(duration_s=0.12, doppler_hz=1500.0, code_phase_samples=180, prn=7)
    acquisition = acquire_signal(
        demo.samples,
        AcquisitionConfig(
            sample_rate=demo.sample_rate,
            prn=demo.prn,
            doppler_min=-3000,
            doppler_max=3000,
            doppler_step=250,
            integration_ms=4,
            compute_backend="cpu",
            gpu_enabled=False,
        ),
    )
    session = SessionConfig(
        sample_rate=demo.sample_rate,
        sample_count=demo.samples.size,
        tracking_ms=80,
        prn=demo.prn,
        compute_backend="gpu",
        gpu_enabled=True,
    )

    monkeypatch.setattr(
        "app.dsp.tracking.resolve_compute_plan",
        lambda *args, **kwargs: ComputePlan(
            requested_backend="gpu",
            active_backend="gpu",
            logical_cores=8,
            selected_workers=1,
            gpu_available=True,
            gpu_name="Fake GPU",
            gpu_library="cupy",
            gpu_enabled=True,
            reason="test",
        ),
    )
    original_math_builder = tracking_module._tracking_math_for_backend

    def fail_gpu_math(backend, *args, **kwargs):
        if backend == "gpu":
            raise RuntimeError("GPU blew up")
        return original_math_builder(backend, *args, **kwargs)

    monkeypatch.setattr(tracking_module, "_tracking_math_for_backend", fail_gpu_math)
    log_messages: list[str] = []

    tracking = tracking_module.track_signal(demo.samples, session, acquisition, log_callback=log_messages.append)

    assert tracking.prompt_i.size == session.tracking_ms
    assert tracking.prompt_mag.mean() > 0.01
    assert any("falling back to CPU" in message for message in log_messages)


@pytest.mark.skipif(not detect_gpu_info().available, reason="Optional CuPy/CUDA backend not available")
def test_acquisition_gpu_matches_cpu_when_available() -> None:
    demo = generate_demo_signal(duration_s=0.08, doppler_hz=1500.0, code_phase_samples=180, prn=7)
    cpu = acquire_signal(
        demo.samples,
        AcquisitionConfig(
            sample_rate=demo.sample_rate,
            prn=demo.prn,
            doppler_min=-3000,
            doppler_max=3000,
            doppler_step=250,
            integration_ms=4,
            compute_backend="cpu",
            max_workers=1,
            gpu_enabled=False,
        ),
    )
    gpu = acquire_signal(
        demo.samples,
        AcquisitionConfig(
            sample_rate=demo.sample_rate,
            prn=demo.prn,
            doppler_min=-3000,
            doppler_max=3000,
            doppler_step=250,
            integration_ms=4,
            compute_backend="gpu",
            max_workers=1,
            gpu_enabled=True,
        ),
    )

    assert gpu.best_candidate.prn == cpu.best_candidate.prn
    assert abs(gpu.best_candidate.doppler_hz - cpu.best_candidate.doppler_hz) <= 1e-3
    assert abs(gpu.best_candidate.code_phase_samples - cpu.best_candidate.code_phase_samples) <= 1
    assert abs(gpu.best_candidate.metric - cpu.best_candidate.metric) <= 1e-3


@pytest.mark.skipif(not detect_gpu_info().available, reason="Optional CuPy/CUDA backend not available")
def test_tracking_gpu_matches_cpu_when_available() -> None:
    demo = generate_demo_signal(duration_s=0.2, doppler_hz=1500.0, code_phase_samples=180, prn=7)
    acquisition = acquire_signal(
        demo.samples,
        AcquisitionConfig(
            sample_rate=demo.sample_rate,
            prn=demo.prn,
            doppler_min=-3000,
            doppler_max=3000,
            doppler_step=250,
            integration_ms=4,
            compute_backend="cpu",
            gpu_enabled=False,
        ),
    )
    cpu_session = SessionConfig(
        sample_rate=demo.sample_rate,
        sample_count=demo.samples.size,
        tracking_ms=120,
        prn=demo.prn,
        compute_backend="cpu",
        gpu_enabled=False,
    )
    gpu_session = replace(cpu_session, compute_backend="gpu", gpu_enabled=True)

    cpu = tracking_module.track_signal(demo.samples, cpu_session, acquisition)
    gpu = tracking_module.track_signal(demo.samples, gpu_session, acquisition)

    assert gpu.prompt_i.size == cpu.prompt_i.size
    assert gpu.lock_detected == cpu.lock_detected
    np.testing.assert_allclose(gpu.prompt_mag, cpu.prompt_mag, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gpu.doppler_est_hz, cpu.doppler_est_hz, rtol=1e-4, atol=1e-3)
