"""PVT solver tests."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.dsp.pvt import expand_gps_week, gps_utc_datetime, lla_to_ecef, solve_position_from_pseudoranges


def test_position_solver_recovers_wachtberg_like_receiver() -> None:
    receiver = lla_to_ecef(50.616, 7.128, 250.0)
    satellites = np.asarray(
        [
            [15_600_000.0, 7_540_000.0, 20_140_000.0],
            [18_760_000.0, -13_400_000.0, 13_480_000.0],
            [17_610_000.0, 14_630_000.0, -13_480_000.0],
            [19_170_000.0, 610_000.0, 18_390_000.0],
            [-13_400_000.0, 18_760_000.0, 13_480_000.0],
        ],
        dtype=np.float64,
    )
    clock_bias_m = 72_000.0
    pseudoranges = np.linalg.norm(satellites - receiver[np.newaxis, :], axis=1) + clock_bias_m

    solution = solve_position_from_pseudoranges(satellites, pseudoranges)

    np.testing.assert_allclose(solution.ecef_m, receiver, atol=1e-3)
    assert abs(solution.receiver_clock_bias_m - clock_bias_m) < 1e-3
    assert abs(solution.latitude_deg - 50.616) < 1e-8
    assert abs(solution.longitude_deg - 7.128) < 1e-8
    np.testing.assert_allclose(solution.residuals_m, np.zeros(satellites.shape[0]), atol=1e-6)


def test_position_solver_requires_four_satellites() -> None:
    satellites = np.zeros((3, 3), dtype=np.float64)
    pseudoranges = np.ones(3, dtype=np.float64)

    try:
        solve_position_from_pseudoranges(satellites, pseudoranges)
    except ValueError as exc:
        assert "At least four satellites" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for underdetermined PVT.")


def test_position_solver_requires_positive_iteration_count() -> None:
    satellites = np.zeros((4, 3), dtype=np.float64)
    pseudoranges = np.ones(4, dtype=np.float64)

    try:
        solve_position_from_pseudoranges(satellites, pseudoranges, max_iterations=0)
    except ValueError as exc:
        assert "max_iterations" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for max_iterations=0.")


def test_gps_week_expansion_and_utc_conversion() -> None:
    full_week = expand_gps_week(367, reference=datetime(2026, 4, 29, tzinfo=UTC))

    assert full_week == 2415
    assert gps_utc_datetime(full_week, 312_222.0).isoformat() == "2026-04-22T14:43:24+00:00"
