"""Large-file oriented block reader tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.dsp.io import Complex64FileSource


def test_complex64_file_source_blocks(tmp_path: Path) -> None:
    data = (np.arange(128, dtype=np.float32) + 1j * np.arange(128, dtype=np.float32)).astype(np.complex64)
    file_path = tmp_path / "big_iq.bin"
    data.tofile(file_path)
    source = Complex64FileSource(str(file_path))
    window = source.read_window(10, 20)
    np.testing.assert_allclose(window, data[10:30])
    blocks = list(source.iter_blocks(0, 16, 3))
    assert len(blocks) == 3
    np.testing.assert_allclose(blocks[1], data[16:32])
