"""Position, velocity, time helper routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0
WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(slots=True)
class PositionSolution:
    """One least-squares receiver position solution."""

    ecef_m: np.ndarray
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    receiver_clock_bias_m: float
    residuals_m: np.ndarray
    used_satellites: int
    iterations: int


def lla_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    """Convert geodetic WGS-84 latitude, longitude, altitude to ECEF metres."""

    lat = np.deg2rad(float(latitude_deg))
    lon = np.deg2rad(float(longitude_deg))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    normal = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (normal + altitude_m) * cos_lat * np.cos(lon)
    y = (normal + altitude_m) * cos_lat * np.sin(lon)
    z = (normal * (1.0 - WGS84_E2) + altitude_m) * sin_lat
    return np.asarray([x, y, z], dtype=np.float64)


def ecef_to_lla(ecef_m: np.ndarray) -> tuple[float, float, float]:
    """Convert ECEF metres to geodetic WGS-84 latitude, longitude, altitude."""

    x, y, z = np.asarray(ecef_m, dtype=np.float64)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = np.sin(lat)
        normal = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        lat = np.arctan2(z + WGS84_E2 * normal * sin_lat, p)
    sin_lat = np.sin(lat)
    normal = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / max(np.cos(lat), 1e-12) - normal
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


def solve_position_from_pseudoranges(
    satellite_positions_m: np.ndarray,
    pseudoranges_m: np.ndarray,
    *,
    initial_ecef_m: np.ndarray | None = None,
    max_iterations: int = 12,
) -> PositionSolution:
    """Solve receiver ECEF and clock bias from satellite positions and pseudoranges."""

    sat_positions = np.asarray(satellite_positions_m, dtype=np.float64)
    pseudoranges = np.asarray(pseudoranges_m, dtype=np.float64)
    if sat_positions.ndim != 2 or sat_positions.shape[1] != 3:
        raise ValueError("satellite_positions_m must have shape (N, 3).")
    if pseudoranges.shape != (sat_positions.shape[0],):
        raise ValueError("pseudoranges_m must have shape (N,).")
    if sat_positions.shape[0] < 4:
        raise ValueError("At least four satellites are required for a 3D position fix.")

    receiver = np.zeros(3, dtype=np.float64) if initial_ecef_m is None else np.asarray(initial_ecef_m, dtype=np.float64).copy()
    clock_bias_m = 0.0
    residuals = np.zeros(sat_positions.shape[0], dtype=np.float64)
    iteration = 0

    for iteration in range(1, int(max_iterations) + 1):
        delta = sat_positions - receiver[np.newaxis, :]
        ranges = np.linalg.norm(delta, axis=1)
        ranges = np.maximum(ranges, 1e-9)
        predicted = ranges + clock_bias_m
        residuals = pseudoranges - predicted
        geometry = np.empty((sat_positions.shape[0], 4), dtype=np.float64)
        geometry[:, :3] = -delta / ranges[:, np.newaxis]
        geometry[:, 3] = 1.0
        update, *_ = np.linalg.lstsq(geometry, residuals, rcond=None)
        receiver += update[:3]
        clock_bias_m += float(update[3])
        if float(np.linalg.norm(update)) < 1e-4:
            break

    latitude_deg, longitude_deg, altitude_m = ecef_to_lla(receiver)
    return PositionSolution(
        ecef_m=receiver,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        altitude_m=altitude_m,
        receiver_clock_bias_m=float(clock_bias_m),
        residuals_m=residuals,
        used_satellites=int(sat_positions.shape[0]),
        iterations=iteration,
    )
