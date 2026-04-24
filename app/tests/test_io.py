"""Tests for complex64 file loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.dsp.io import common_sample_rate_hints, inspect_complex64_file, load_complex64_samples


def test_complex64_inspection_and_load(tmp_path: Path) -> None:
    data = (np.arange(32, dtype=np.float32) + 1j * np.arange(32, dtype=np.float32)).astype(np.complex64)
    file_path = tmp_path / "iq.bin"
    data.tofile(file_path)
    metadata = inspect_complex64_file(str(file_path), sample_rate=4.0)
    assert metadata.total_samples == 32
    assert metadata.estimated_duration_s == 8.0
    assert "6.061 MSa/s" in metadata.common_rate_duration_hints
    loaded = load_complex64_samples(str(file_path), start_sample=4, sample_count=8)
    np.testing.assert_allclose(loaded, data[4:12])


def test_x310_six_megahertz_hint_uses_rational_rate() -> None:
    hints = common_sample_rate_hints(3_600_000_000)

    assert hints["6.061 MSa/s"] == 594.0


def test_complex64_window_load_is_capped_to_file_size(tmp_path: Path) -> None:
    data = (np.arange(8, dtype=np.float32) + 1j * np.arange(8, dtype=np.float32)).astype(np.complex64)
    file_path = tmp_path / "iq.bin"
    data.tofile(file_path)

    loaded = load_complex64_samples(str(file_path), start_sample=6, sample_count=100)

    np.testing.assert_allclose(loaded, data[6:])
