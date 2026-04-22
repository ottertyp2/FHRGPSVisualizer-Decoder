"""Benchmark output sanity tests."""

from __future__ import annotations

from app.dsp.benchmark import run_benchmark
from app.models import SessionConfig


def test_benchmark_returns_components() -> None:
    session = SessionConfig(file_path=None, sample_rate=4_092_000.0, sample_count=500_000, tracking_ms=80)
    result = run_benchmark(session)
    assert result.components
    assert result.bottleneck_name
    assert any(component.name == "Tracking compute" for component in result.components)
    assert "logical_cores" in result.system_info
    assert "selected_workers" in result.system_info
    assert "active_backend" in result.system_info
    assert "gpu_name" in result.system_info
