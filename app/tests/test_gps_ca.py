"""Tests for GPS C/A code generation."""

from __future__ import annotations

import numpy as np

from app.dsp.gps_ca import generate_ca_code, sample_ca_code


def test_generate_ca_code_length_and_values() -> None:
    code = generate_ca_code(1)
    assert code.shape == (1023,)
    assert set(np.unique(code)).issubset({-1.0, 1.0})


def test_sample_ca_code_size() -> None:
    sampled = sample_ca_code(3, 4_092_000.0, 4_092)
    assert sampled.shape == (4_092,)
