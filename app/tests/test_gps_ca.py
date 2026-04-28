"""Tests for GPS C/A code generation."""

from __future__ import annotations

import pytest
import numpy as np

from app.dsp.gps_ca import G2_TAPS, generate_ca_code, sample_ca_code


REFERENCE_FIRST_10_CHIPS_OCTAL = {
    1: "1440",
    2: "1620",
    3: "1710",
    4: "1744",
    5: "1133",
    6: "1455",
    7: "1131",
    8: "1454",
    9: "1626",
    10: "1504",
    11: "1642",
    12: "1750",
    13: "1764",
    14: "1772",
    15: "1775",
    16: "1776",
    17: "1156",
    18: "1467",
    19: "1633",
    20: "1715",
    21: "1746",
    22: "1763",
    23: "1063",
    24: "1706",
    25: "1743",
    26: "1761",
    27: "1770",
    28: "1774",
    29: "1127",
    30: "1453",
    31: "1625",
    32: "1712",
}


def _first_10_chips_octal(prn: int) -> str:
    """Return IS-GPS-200-style octal for the first 10 C/A chips."""

    bits = "".join("1" if chip < 0 else "0" for chip in generate_ca_code(prn)[:10])
    return bits[0] + format(int(bits[1:], 2), "03o")


def test_generate_ca_code_length_and_values() -> None:
    code = generate_ca_code(1)
    assert code.shape == (1023,)
    assert set(np.unique(code)).issubset({-1.0, 1.0})


def test_generate_ca_code_matches_reference_start() -> None:
    # PRN 1 uses G2 stages 2 and 6, counted from the register input side.
    expected_start = np.asarray(
        [-1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(generate_ca_code(1)[: expected_start.size], expected_start)


@pytest.mark.parametrize("prn, expected_octal", REFERENCE_FIRST_10_CHIPS_OCTAL.items())
def test_generate_ca_code_matches_reference_first_10_chips(prn: int, expected_octal: str) -> None:
    # IS-GPS-200N Table 3-Ia gives these first-10-chip octal values.
    assert _first_10_chips_octal(prn) == expected_octal


def test_prn_3_and_8_use_distinct_reference_taps() -> None:
    assert G2_TAPS[3] == (4, 8)
    assert _first_10_chips_octal(3) == "1710"
    assert G2_TAPS[8] == (2, 9)
    assert _first_10_chips_octal(8) == "1454"


def test_generate_ca_code_is_cached() -> None:
    assert generate_ca_code(1) is generate_ca_code(1)


def test_sample_ca_code_size() -> None:
    sampled = sample_ca_code(3, 4_092_000.0, 4_092)
    assert sampled.shape == (4_092,)


def test_sample_ca_code_rejects_invalid_sampling_parameters() -> None:
    with pytest.raises(ValueError, match="Sample rate"):
        sample_ca_code(1, 0.0, 10)
    with pytest.raises(ValueError, match="Number"):
        sample_ca_code(1, 1_000_000.0, -1)
