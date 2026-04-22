"""Search-center sweep sanity checks."""

from __future__ import annotations

from app.dsp.acquisition import sweep_search_centers_from_session
from app.dsp.demo import generate_demo_signal
from app.models import SessionConfig


def test_search_center_sweep_returns_ranked_entries() -> None:
    demo = generate_demo_signal(duration_s=0.1, doppler_hz=1750.0, code_phase_samples=250)
    session = SessionConfig(sample_rate=demo.sample_rate, prn=demo.prn, doppler_min=-3000, doppler_max=3000, doppler_step=250)
    result = sweep_search_centers_from_session(demo.samples[: int(demo.sample_rate * 0.004)], session, [0.0, 1000.0, 2000.0], prns=[1, 2, 3])
    assert result.entries
    assert result.entries[0].best_result.best_candidate.metric >= result.entries[-1].best_result.best_candidate.metric
