"""Tests for compute backend selection and worker policies."""

from __future__ import annotations

from app.dsp import compute


def test_auto_worker_count_uses_all_but_one_core() -> None:
    assert compute.resolve_worker_count(0, logical_cores=8) == 7


def test_manual_worker_count_respects_task_cap() -> None:
    assert compute.resolve_worker_count(6, logical_cores=12, max_tasks=3) == 3


def test_gpu_request_falls_back_to_cpu_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(compute, "detect_logical_cores", lambda: 8)
    monkeypatch.setattr(
        compute,
        "detect_gpu_info",
        lambda: compute.GPUInfo(available=False, library="cupy", reason="CuPy not installed"),
    )

    plan = compute.resolve_compute_plan("gpu", 0, gpu_enabled=True, prefer_gpu=True)

    assert plan.active_backend == "cpu"
    assert plan.selected_workers == 7
    assert plan.gpu_available is False
