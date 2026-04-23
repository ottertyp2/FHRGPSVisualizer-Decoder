"""Search-center sweep sanity checks."""

from __future__ import annotations

import numpy as np

from app.dsp.acquisition import sweep_search_centers_from_session
from app.dsp.compute import ComputePlan
from app.dsp.demo import generate_demo_signal
from app.models import AcquisitionCandidate, AcquisitionResult
from app.models import SessionConfig


def test_search_center_sweep_returns_ranked_entries() -> None:
    demo = generate_demo_signal(duration_s=0.1, doppler_hz=1750.0, code_phase_samples=250)
    session = SessionConfig(sample_rate=demo.sample_rate, prn=demo.prn, doppler_min=-3000, doppler_max=3000, doppler_step=250)
    result = sweep_search_centers_from_session(demo.samples[: int(demo.sample_rate * 0.004)], session, [0.0, 1000.0, 2000.0], prns=[1, 2, 3])
    assert result.entries
    assert result.entries[0].best_result.best_candidate.metric >= result.entries[-1].best_result.best_candidate.metric


def test_search_center_sweep_splits_cpu_budget_between_centers_and_prns(monkeypatch) -> None:
    recorded_sessions: list[SessionConfig] = []
    recorded_outer_workers: list[int] = []

    def fake_resolve_compute_plan(*_args, **_kwargs) -> ComputePlan:
        return ComputePlan(
            requested_backend="cpu",
            active_backend="cpu",
            logical_cores=12,
            selected_workers=8,
            gpu_available=False,
            gpu_name="unavailable",
            gpu_library="none",
            gpu_enabled=False,
            reason="test",
        )

    def fake_parallel_ordered_map(items, fn, *, max_workers):
        recorded_outer_workers.append(max_workers)
        return [fn(index, item) for index, item in enumerate(items)]

    def fake_scan_prns(samples, session, prns=None, progress_callback=None, log_callback=None):
        recorded_sessions.append(session)
        if progress_callback:
            progress_callback(100)
        best_prn = (prns or [1])[0]
        best = AcquisitionCandidate(
            prn=best_prn,
            doppler_hz=0.0,
            carrier_frequency_hz=float(session.if_frequency_hz),
            code_phase_samples=0,
            metric=10.0,
        )
        return [
            AcquisitionResult(
                prn=best_prn,
                sample_rate_hz=float(session.sample_rate),
                search_center_hz=float(session.if_frequency_hz),
                doppler_bins_hz=np.asarray([0.0], dtype=np.float32),
                code_phases_samples=np.asarray([0], dtype=np.int32),
                heatmap=np.ones((1, 1), dtype=np.float32),
                best_candidate=best,
                candidates=[best],
            )
        ]

    monkeypatch.setattr("app.dsp.acquisition.resolve_compute_plan", fake_resolve_compute_plan)
    monkeypatch.setattr("app.dsp.acquisition.parallel_ordered_map", fake_parallel_ordered_map)
    monkeypatch.setattr("app.dsp.acquisition.scan_prns_from_session", fake_scan_prns)

    session = SessionConfig(compute_backend="cpu", max_workers=8, gpu_enabled=False)
    result = sweep_search_centers_from_session(
        np.zeros(8_192, dtype=np.complex64),
        session,
        [0.0, 1000.0, 2000.0],
        prns=[1, 2, 3, 4],
    )

    assert result.entries
    assert recorded_outer_workers == [2]
    assert recorded_sessions
    assert all(item.compute_backend == "cpu" for item in recorded_sessions)
    assert all(item.max_workers == 4 for item in recorded_sessions)
