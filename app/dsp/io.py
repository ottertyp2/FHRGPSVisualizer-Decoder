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
