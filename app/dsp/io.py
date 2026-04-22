"""Offline file loading and preview helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from app.models import FileMetadata


def inspect_complex64_file(
    file_path: str,
    sample_rate: float,
    preview_samples: int = 8_192,
) -> FileMetadata:
    """Inspect a little-endian complex64 raw IQ file."""

    path = Path(file_path)
    size_bytes = path.stat().st_size
    total_samples = size_bytes // np.dtype(np.complex64).itemsize
    preview = np.fromfile(path, dtype=np.complex64, count=min(preview_samples, total_samples))
    magnitude = np.abs(preview)
    preview_stats = {
        "min_i": float(preview.real.min()) if preview.size else 0.0,
        "max_i": float(preview.real.max()) if preview.size else 0.0,
        "min_q": float(preview.imag.min()) if preview.size else 0.0,
        "max_q": float(preview.imag.max()) if preview.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.abs(preview) ** 2))) if preview.size else 0.0,
        "mean_mag": float(magnitude.mean()) if magnitude.size else 0.0,
    }
    return FileMetadata(
        file_path=str(path),
        file_name=path.name,
        file_size_bytes=size_bytes,
        data_type="complex64",
        endianness="little",
        sample_rate_hz=float(sample_rate),
        total_samples=total_samples,
        estimated_duration_s=(total_samples / sample_rate) if sample_rate > 0 else 0.0,
        preview_stats=preview_stats,
        preview_samples=preview.astype(np.complex64, copy=False),
    )


def load_complex64_samples(file_path: str, start_sample: int, sample_count: int) -> np.ndarray:
    """Load a sample window from a complex64 raw file."""

    dtype = np.dtype(np.complex64)
    offset = max(0, int(start_sample)) * dtype.itemsize
    count = max(0, int(sample_count))
    with Path(file_path).open("rb") as handle:
        handle.seek(offset)
        return np.fromfile(handle, dtype=dtype, count=count)


def load_complex64_samples_with_progress(
    file_path: str,
    start_sample: int,
    sample_count: int,
    progress_callback=None,
    chunk_samples: int = 2_000_000,
) -> np.ndarray:
    """Load a sample window in chunks and emit progress updates."""

    dtype = np.dtype(np.complex64)
    count = max(0, int(sample_count))
    if count == 0:
        if progress_callback:
            progress_callback(100)
        return np.empty(0, dtype=dtype)

    if progress_callback is None or count <= chunk_samples:
        result = load_complex64_samples(file_path, start_sample, count)
        if progress_callback:
            progress_callback(100)
        return result

    result = np.empty(count, dtype=dtype)
    offset = max(0, int(start_sample)) * dtype.itemsize
    loaded = 0
    with Path(file_path).open("rb") as handle:
        handle.seek(offset)
        while loaded < count:
            chunk = min(chunk_samples, count - loaded)
            data = np.fromfile(handle, dtype=dtype, count=chunk)
            if data.size == 0:
                break
            result[loaded : loaded + data.size] = data
            loaded += data.size
            if progress_callback:
                progress_callback(int(100 * loaded / count))
    return result[:loaded]


def load_complex64_file(file_path: str) -> np.ndarray:
    """Load an entire complex64 raw file into RAM."""

    return np.fromfile(Path(file_path), dtype=np.complex64)


def load_complex64_file_with_progress(
    file_path: str,
    progress_callback=None,
    chunk_samples: int = 2_000_000,
) -> np.ndarray:
    """Load an entire complex64 raw file into RAM with progress updates."""

    total_samples = Path(file_path).stat().st_size // np.dtype(np.complex64).itemsize
    return load_complex64_samples_with_progress(
        file_path,
        start_sample=0,
        sample_count=total_samples,
        progress_callback=progress_callback,
        chunk_samples=chunk_samples,
    )


class Complex64FileSource:
    """Memory-mapped windowed reader for very large complex64 IQ files."""

    def __init__(self, file_path: str) -> None:
        self.path = Path(file_path)
        self._memmap = np.memmap(self.path, dtype=np.complex64, mode="r")

    @property
    def total_samples(self) -> int:
        """Total number of complex samples in the file."""

        return int(self._memmap.size)

    def read_window(self, start_sample: int, sample_count: int) -> np.ndarray:
        """Return a copy of one selected sample window."""

        start = max(0, int(start_sample))
        stop = min(self.total_samples, start + max(0, int(sample_count)))
        if stop <= start:
            return np.empty(0, dtype=np.complex64)
        return np.asarray(self._memmap[start:stop], dtype=np.complex64)

    def iter_blocks(
        self,
        start_sample: int,
        block_size: int,
        block_count: int,
    ) -> Iterator[np.ndarray]:
        """Yield consecutive fixed-size blocks without loading the whole file."""

        if block_size <= 0 or block_count <= 0:
            return
        for block_index in range(int(block_count)):
            start = int(start_sample) + block_index * int(block_size)
            stop = start + int(block_size)
            if stop > self.total_samples:
                break
            yield np.asarray(self._memmap[start:stop], dtype=np.complex64)
