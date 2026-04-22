"""Compute backend detection and shared acceleration helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
import importlib
import multiprocessing
import os
import threading
from typing import Any, Callable, Sequence, TypeVar


T = TypeVar("T")
U = TypeVar("U")


@dataclass(slots=True)
class GPUInfo:
    """Runtime information about optional GPU support."""

    available: bool
    name: str = "not detected"
    library: str = "none"
    reason: str = ""


@dataclass(slots=True)
class ComputePlan:
    """Resolved compute settings for one operation."""

    requested_backend: str
    active_backend: str
    logical_cores: int
    selected_workers: int
    gpu_available: bool
    gpu_name: str
    gpu_library: str
    gpu_enabled: bool
    reason: str

    def status_text(self) -> str:
        """Return one user-facing status line."""

        gpu_text = self.gpu_name if self.gpu_available else f"unavailable ({self.reason})"
        return (
            f"Runtime status: backend={self.active_backend}, requested={self.requested_backend}, "
            f"workers={self.selected_workers}/{self.logical_cores}, GPU={gpu_text}."
        )


def detect_logical_cores() -> int:
    """Return the number of logical CPU cores."""

    logical_cores = os.cpu_count()
    if logical_cores is None:
        logical_cores = multiprocessing.cpu_count()
    return max(1, int(logical_cores))


@lru_cache(maxsize=1)
def get_cupy_module() -> Any | None:
    """Return the optional CuPy module when installed."""

    try:
        return importlib.import_module("cupy")
    except Exception:
        return None


@lru_cache(maxsize=1)
def detect_gpu_info() -> GPUInfo:
    """Detect whether a usable CuPy/CUDA GPU backend is available."""

    cupy = get_cupy_module()
    if cupy is None:
        return GPUInfo(available=False, reason="CuPy not installed")

    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return GPUInfo(available=False, library="cupy", reason=f"CUDA runtime unavailable: {exc}")

    if device_count <= 0:
        return GPUInfo(available=False, library="cupy", reason="no CUDA device found")

    try:
        properties = cupy.cuda.runtime.getDeviceProperties(0)
        raw_name = properties.get("name", "CUDA GPU")
        if isinstance(raw_name, bytes):
            gpu_name = raw_name.decode("utf-8", errors="replace")
        else:
            gpu_name = str(raw_name)
    except Exception:
        gpu_name = "CUDA GPU"
    return GPUInfo(available=True, name=gpu_name, library="cupy", reason="CuPy/CUDA ready")


def resolve_worker_count(
    requested_workers: int = 0,
    *,
    logical_cores: int | None = None,
    max_tasks: int | None = None,
) -> int:
    """Resolve one worker count from user preference and task count."""

    cores = logical_cores or detect_logical_cores()
    if int(requested_workers) > 0:
        selected = int(requested_workers)
    else:
        selected = max(1, int(cores) - 1)
    if max_tasks is not None:
        selected = min(selected, max(1, int(max_tasks)))
    return max(1, min(selected, int(cores)))


def resolve_compute_plan(
    requested_backend: str = "auto",
    requested_workers: int = 0,
    *,
    gpu_enabled: bool = True,
    max_tasks: int | None = None,
    prefer_gpu: bool = True,
) -> ComputePlan:
    """Resolve one compute policy into an executable plan."""

    backend = (requested_backend or "auto").strip().lower()
    if backend not in {"auto", "cpu", "gpu"}:
        backend = "auto"

    logical_cores = detect_logical_cores()
    selected_workers = resolve_worker_count(
        requested_workers,
        logical_cores=logical_cores,
        max_tasks=max_tasks,
    )
    gpu_info = detect_gpu_info()

    if not gpu_enabled:
        active_backend = "cpu"
        reason = "GPU disabled in session"
    elif backend == "cpu":
        active_backend = "cpu"
        reason = "CPU forced in session"
    elif backend == "gpu":
        if gpu_info.available:
            active_backend = "gpu"
            reason = "GPU forced in session"
        else:
            active_backend = "cpu"
            reason = "GPU requested but unavailable"
    elif prefer_gpu and gpu_info.available:
        active_backend = "gpu"
        reason = "Auto selected available GPU"
    else:
        active_backend = "cpu"
        reason = "Auto selected CPU"

    return ComputePlan(
        requested_backend=backend,
        active_backend=active_backend,
        logical_cores=logical_cores,
        selected_workers=selected_workers,
        gpu_available=gpu_info.available,
        gpu_name=gpu_info.name,
        gpu_library=gpu_info.library,
        gpu_enabled=bool(gpu_enabled),
        reason=reason if gpu_info.available or active_backend == "cpu" else gpu_info.reason,
    )


class ProgressTracker:
    """Aggregate subtask progress into one monotonic percentage."""

    def __init__(self, task_count: int, progress_callback: Callable[[int], None] | None) -> None:
        self.task_count = max(1, int(task_count))
        self.progress_callback = progress_callback
        self.values = [0 for _ in range(self.task_count)]
        self.last_reported = -1
        self.lock = threading.Lock()

    def update(self, index: int, value: int) -> None:
        """Update one subtask and emit aggregate progress."""

        if self.progress_callback is None:
            return
        clamped = max(0, min(100, int(value)))
        with self.lock:
            self.values[index] = max(self.values[index], clamped)
            combined = int(sum(self.values) / self.task_count)
            combined = max(combined, self.last_reported)
            if combined == self.last_reported:
                return
            self.last_reported = combined
        self.progress_callback(combined)


def parallel_ordered_map(
    items: Sequence[T],
    fn: Callable[[int, T], U],
    *,
    max_workers: int,
) -> list[U]:
    """Run independent items in parallel and preserve input order."""

    if len(items) <= 1 or max_workers <= 1:
        return [fn(index, item) for index, item in enumerate(items)]

    results: list[U | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(fn, index, item): index
            for index, item in enumerate(items)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return [result for result in results if result is not None]
