"""Compute backend detection and shared acceleration helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
import importlib
import multiprocessing
import os
from pathlib import Path
import site
import threading
from typing import Any, Callable, Sequence, TypeVar


T = TypeVar("T")
U = TypeVar("U")

CUDA_NAMESPACE_PACKAGES = (
    "cuda_nvrtc",
    "cuda_runtime",
    "cufft",
    "nvjitlink",
)


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


def _site_roots() -> list[Path]:
    """Return candidate site-packages roots for optional CUDA wheels."""

    roots: list[Path] = []
    for candidate in [*site.getsitepackages(), site.getusersitepackages()]:
        path = Path(candidate)
        if path.exists():
            roots.append(path)
    return roots


def _prepend_env_path(path: Path) -> None:
    """Prepend one directory to PATH when it is not already present."""

    resolved = str(path)
    existing = os.environ.get("PATH", "").split(os.pathsep)
    normalized = {item.lower() for item in existing if item}
    if resolved.lower() in normalized:
        return
    os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")


@lru_cache(maxsize=1)
def bootstrap_cuda_runtime_paths() -> dict[str, str]:
    """Expose pip-installed NVIDIA CUDA DLL paths to the current process."""

    discovered: dict[str, str] = {}
    for root in _site_roots():
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        for package_name in CUDA_NAMESPACE_PACKAGES:
            package_root = nvidia_root / package_name
            if not package_root.exists():
                continue
            discovered[package_name] = str(package_root)
            bin_dir = package_root / "bin"
            if bin_dir.exists():
                _prepend_env_path(bin_dir)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(bin_dir))
                    except (FileNotFoundError, OSError):
                        pass

    if "cuda_nvrtc" in discovered and not os.environ.get("CUDA_PATH"):
        os.environ["CUDA_PATH"] = discovered["cuda_nvrtc"]
    return discovered


def _cupy_runtime_probe(cupy: Any) -> None:
    """Run a minimal CuPy + cuFFT smoke test."""

    vector = cupy.arange(8, dtype=cupy.float32)
    _ = cupy.asnumpy(vector * vector)
    _ = cupy.asnumpy(cupy.fft.fft(vector.astype(cupy.complex64)))


@lru_cache(maxsize=1)
def get_cupy_module() -> Any | None:
    """Return the optional CuPy module when installed."""

    bootstrap_cuda_runtime_paths()
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

    try:
        _cupy_runtime_probe(cupy)
    except Exception as exc:
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
        return GPUInfo(
            available=False,
            name=gpu_name,
            library="cupy",
            reason=f"CuPy runtime incomplete: {first_line}",
        )
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


def split_nested_worker_budget(
    total_workers: int,
    *,
    outer_tasks: int,
    inner_tasks: int,
) -> tuple[int, int]:
    """Split one CPU worker budget across outer and inner parallel loops."""

    total = max(1, int(total_workers))
    outer = max(1, int(outer_tasks))
    inner = max(1, int(inner_tasks))

    if outer <= 1:
        return 1, min(total, inner)
    if inner <= 1:
        return min(total, outer), 1

    best_outer = 1
    best_inner = min(total, inner)
    best_score = (-1, -1, -1, -1)

    for outer_workers in range(1, min(total, outer) + 1):
        inner_workers = min(inner, max(1, total // outer_workers))
        dimensions = int(outer_workers > 1) + int(inner_workers > 1)
        utilized = outer_workers * inner_workers
        balance = min(outer_workers, inner_workers)
        score = (dimensions, utilized, balance, outer_workers)
        if score > best_score:
            best_score = score
            best_outer = outer_workers
            best_inner = inner_workers

    return best_outer, best_inner


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
