"""Position, velocity, time helper routines."""

from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from app.dsp.ephemeris import (
    GpsEphemeris,
    decode_ephemeris,
    rotate_ecef_for_transit,
    satellite_clock_correction_s,
    satellite_position_ecef_m,
)
from app.models import BitDecisionResult, NavigationDecodeResult, NavigationSubframe, TrackingState

SPEED_OF_LIGHT_M_S = 299_792_458.0
WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
GPS_WEEK_SECONDS = 604_800.0
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=UTC)
GPS_UTC_LEAP_SECONDS = 18


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


@dataclass(slots=True)
class PseudorangeObservation:
    """One didactic PVT observation formed from a valid LNAV subframe."""

    prn: int
    transmit_time_s: float
    corrected_transmit_time_s: float
    receive_file_time_s: float
    pseudorange_m: float
    satellite_clock_correction_s: float
    satellite_position_m: np.ndarray
    subframe_id: int
    subframe_start_bit: int
    bit_start_ms: int


@dataclass(slots=True)
class PVTComputationResult:
    """End-to-end PVT result plus the evidence used to produce it."""

    solution: PositionSolution | None
    observations: list[PseudorangeObservation]
    ephemerides: dict[int, GpsEphemeris]
    gps_week: int | None
    gps_time_of_week_s: float | None
    utc_datetime: datetime | None
    receiver_time_offset_s: float | None
    residual_rms_m: float | None
    summary_lines: list[str]


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
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive for a position fix.")

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

    final_delta = sat_positions - receiver[np.newaxis, :]
    final_ranges = np.maximum(np.linalg.norm(final_delta, axis=1), 1e-9)
    residuals = pseudoranges - (final_ranges + clock_bias_m)

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


def expand_gps_week(week_number_mod1024: int, reference: datetime | None = None) -> int:
    """Expand a 10-bit LNAV week number to the full week nearest a reference date."""

    reference_dt = reference or datetime.now(UTC)
    if reference_dt.tzinfo is None:
        reference_dt = reference_dt.replace(tzinfo=UTC)
    gps_reference = reference_dt + timedelta(seconds=GPS_UTC_LEAP_SECONDS)
    reference_week = int((gps_reference - GPS_EPOCH).total_seconds() // GPS_WEEK_SECONDS)
    base = int(week_number_mod1024)
    candidates = [base + 1024 * offset for offset in range(max(0, reference_week // 1024 - 2), reference_week // 1024 + 3)]
    return min(candidates, key=lambda week: abs(week - reference_week))


def gps_utc_datetime(
    gps_week: int,
    time_of_week_s: float,
    leap_seconds: int = GPS_UTC_LEAP_SECONDS,
) -> datetime:
    """Convert GPS week/TOW to UTC using the configured leap-second offset."""

    gps_time = GPS_EPOCH + timedelta(weeks=int(gps_week), seconds=float(time_of_week_s))
    return gps_time - timedelta(seconds=int(leap_seconds))


def _tracking_start_file_time_s(tracking: TrackingState) -> float:
    sample_rate = float(tracking.sample_rate_hz) if tracking.sample_rate_hz > 0.0 else 0.0
    if sample_rate <= 0.0:
        return 0.0
    return float(tracking.source_start_sample) / sample_rate


def _subframe_receive_time_s(
    tracking: TrackingState,
    bit_result: BitDecisionResult,
    subframe: NavigationSubframe,
) -> tuple[float, int] | None:
    if subframe.start_bit < 0 or subframe.start_bit >= bit_result.bit_start_ms.size:
        return None
    bit_start_ms = int(bit_result.bit_start_ms[subframe.start_bit])
    code_phase_s = 0.0
    if tracking.sample_rate_hz > 0.0:
        code_phase_s = float(tracking.code_phase_samples) / float(tracking.sample_rate_hz)
    return _tracking_start_file_time_s(tracking) + bit_start_ms * 1e-3 - code_phase_s, bit_start_ms


def _candidate_observation_groups(
    tracking_by_prn: dict[int, TrackingState],
    bit_results_by_prn: dict[int, BitDecisionResult],
    nav_results_by_prn: dict[int, NavigationDecodeResult],
    ephemerides: dict[int, GpsEphemeris],
) -> dict[float, list[tuple[int, NavigationSubframe, float, int]]]:
    """Group valid subframe arrivals by their GPS transmit epoch."""

    groups: dict[float, list[tuple[int, NavigationSubframe, float, int]]] = {}
    for prn, ephemeris in ephemerides.items():
        tracking = tracking_by_prn.get(prn)
        bit_result = bit_results_by_prn.get(prn)
        nav_result = nav_results_by_prn.get(prn)
        if tracking is None or bit_result is None or nav_result is None or ephemeris is None:
            continue
        for subframe in nav_result.subframes:
            if not subframe.valid or subframe.tow_seconds is None or subframe.subframe_id is None:
                continue
            timing = _subframe_receive_time_s(tracking, bit_result, subframe)
            if timing is None:
                continue
            receive_file_time_s, bit_start_ms = timing
            transmit_time_s = float((int(subframe.tow_seconds) - 6) % int(GPS_WEEK_SECONDS))
            groups.setdefault(transmit_time_s, []).append((prn, subframe, receive_file_time_s, bit_start_ms))
    return groups


def _build_observations_for_group(
    group: list[tuple[int, NavigationSubframe, float, int]],
    ephemerides: dict[int, GpsEphemeris],
    target_range_s: float = 0.075,
) -> list[PseudorangeObservation]:
    """Turn one equal-TOW group into pseudoranges and satellite positions."""

    raw_rows: list[tuple[int, NavigationSubframe, float, int, float, float]] = []
    for prn, subframe, receive_file_time_s, bit_start_ms in group:
        ephemeris = ephemerides[prn]
        transmit_time_s = float((int(subframe.tow_seconds or 0) - 6) % int(GPS_WEEK_SECONDS))
        clock_s = satellite_clock_correction_s(ephemeris, transmit_time_s)
        corrected_transmit_s = transmit_time_s - clock_s
        raw_rows.append((prn, subframe, receive_file_time_s, bit_start_ms, corrected_transmit_s, clock_s))

    if len(raw_rows) < 4:
        return []

    receiver_time_offset_s = (
        float(np.median([corrected_tx_s - receive_s for _prn, _sf, receive_s, _ms, corrected_tx_s, _clock in raw_rows]))
        + float(target_range_s)
    )
    observations: list[PseudorangeObservation] = []
    for prn, subframe, receive_file_time_s, bit_start_ms, corrected_transmit_s, clock_s in raw_rows:
        pseudorange_m = SPEED_OF_LIGHT_M_S * (receive_file_time_s + receiver_time_offset_s - corrected_transmit_s)
        transmit_for_orbit_s = corrected_transmit_s
        satellite_position = satellite_position_ecef_m(ephemerides[prn], transmit_for_orbit_s)
        satellite_position = rotate_ecef_for_transit(satellite_position, pseudorange_m / SPEED_OF_LIGHT_M_S)
        observations.append(
            PseudorangeObservation(
                prn=prn,
                transmit_time_s=float((int(subframe.tow_seconds or 0) - 6) % int(GPS_WEEK_SECONDS)),
                corrected_transmit_time_s=float(corrected_transmit_s),
                receive_file_time_s=float(receive_file_time_s),
                pseudorange_m=float(pseudorange_m),
                satellite_clock_correction_s=float(clock_s),
                satellite_position_m=satellite_position,
                subframe_id=int(subframe.subframe_id or 0),
                subframe_start_bit=int(subframe.start_bit),
                bit_start_ms=int(bit_start_ms),
            )
        )
    return observations


def _solve_observation_subset(
    observations: list[PseudorangeObservation],
) -> tuple[PositionSolution, list[PseudorangeObservation]]:
    """Choose a consistent satellite subset without hiding the simple logic."""

    best_solution: PositionSolution | None = None
    best_indices: tuple[int, ...] = ()
    best_score = np.inf
    total = len(observations)

    for subset_size in range(total, 3, -1):
        for indices in combinations(range(total), subset_size):
            satellites = np.vstack([observations[index].satellite_position_m for index in indices])
            pseudoranges = np.asarray([observations[index].pseudorange_m for index in indices], dtype=np.float64)
            solution = solve_position_from_pseudoranges(satellites, pseudoranges)
            residual_rms = float(np.sqrt(np.mean(solution.residuals_m**2))) if solution.residuals_m.size else 0.0
            altitude_penalty = 0.0
            if solution.altitude_m < -1_000.0:
                altitude_penalty = abs(solution.altitude_m + 1_000.0)
            elif solution.altitude_m > 10_000.0:
                altitude_penalty = solution.altitude_m - 10_000.0
            dropped_penalty = float(total - subset_size) * 1_000.0
            score = residual_rms + altitude_penalty + dropped_penalty
            if score < best_score:
                best_score = score
                best_solution = solution
                best_indices = indices

    if best_solution is None:
        raise ValueError("At least four observations are required for PVT.")
    return best_solution, [observations[index] for index in best_indices]


def compute_pvt_from_navigation(
    tracking_by_prn: dict[int, TrackingState],
    bit_results_by_prn: dict[int, BitDecisionResult],
    nav_results_by_prn: dict[int, NavigationDecodeResult],
    progress_callback=None,
    log_callback=None,
) -> PVTComputationResult:
    """Build ephemerides, pseudoranges, receiver position, and GPS/UTC time."""

    if log_callback:
        log_callback("PVT: decoding broadcast ephemerides from parity-valid subframes.")
    if progress_callback:
        progress_callback(15)
    ephemerides = {
        prn: ephemeris
        for prn, nav_result in nav_results_by_prn.items()
        if (ephemeris := decode_ephemeris(prn, nav_result.subframes)) is not None
    }
    summary_lines: list[str] = []
    if len(ephemerides) < 4:
        summary_lines.append(
            f"Need at least 4 decoded ephemerides; currently have {len(ephemerides)}."
        )
        return PVTComputationResult(
            solution=None,
            observations=[],
            ephemerides=ephemerides,
            gps_week=None,
            gps_time_of_week_s=None,
            utc_datetime=None,
            receiver_time_offset_s=None,
            residual_rms_m=None,
            summary_lines=summary_lines,
        )

    if progress_callback:
        progress_callback(45)
    if log_callback:
        log_callback(f"PVT: found {len(ephemerides)} usable ephemerides.")
    groups = _candidate_observation_groups(
        tracking_by_prn,
        bit_results_by_prn,
        nav_results_by_prn,
        ephemerides,
    )
    ranked_groups = sorted(groups.items(), key=lambda item: (len(item[1]), -item[0]), reverse=True)
    best_result: PVTComputationResult | None = None
    best_score = np.inf

    for transmit_time_s, group in ranked_groups:
        unique_prns = sorted({prn for prn, _sf, _rx, _ms in group})
        if len(unique_prns) < 4:
            continue
        unique_group = []
        seen: set[int] = set()
        for row in group:
            if row[0] in seen:
                continue
            seen.add(row[0])
            unique_group.append(row)
        observations = _build_observations_for_group(unique_group, ephemerides)
        if len(observations) < 4:
            continue
        solution, used_observations = _solve_observation_subset(observations)
        residual_rms = float(np.sqrt(np.mean(solution.residuals_m**2))) if solution.residuals_m.size else 0.0
        plausible_altitude_penalty = max(0.0, abs(solution.altitude_m) - 20_000.0)
        score = residual_rms + plausible_altitude_penalty + float(len(observations) - len(used_observations)) * 1_000.0
        gps_week = expand_gps_week(next(iter(ephemerides.values())).week_number_mod1024)
        receiver_time_offset_s = float(
            np.median([obs.corrected_transmit_time_s - obs.receive_file_time_s for obs in used_observations]) + 0.075
        )
        gps_time_of_week_s = float((np.median([obs.receive_file_time_s for obs in used_observations]) + receiver_time_offset_s) % GPS_WEEK_SECONDS)
        utc_dt = gps_utc_datetime(gps_week, gps_time_of_week_s)
        candidate = PVTComputationResult(
            solution=solution,
            observations=used_observations,
            ephemerides=ephemerides,
            gps_week=gps_week,
            gps_time_of_week_s=gps_time_of_week_s,
            utc_datetime=utc_dt,
            receiver_time_offset_s=receiver_time_offset_s,
            residual_rms_m=residual_rms,
            summary_lines=[
                f"Decoded ephemerides for PRNs {', '.join(str(prn) for prn in sorted(ephemerides))}.",
                f"Used GPS transmit epoch TOW {transmit_time_s:.0f} s with PRNs {', '.join(str(obs.prn) for obs in used_observations)}.",
                f"Least-squares residual RMS {residual_rms:.1f} m.",
            ],
        )
        if score < best_score:
            best_score = score
            best_result = candidate

    if best_result is not None:
        if progress_callback:
            progress_callback(100)
        if log_callback and best_result.solution is not None:
            log_callback(
                f"PVT: solved {best_result.solution.used_satellites} satellites at "
                f"{best_result.solution.latitude_deg:.5f}, {best_result.solution.longitude_deg:.5f}."
            )
        return best_result

    summary_lines.append(
        f"Decoded {len(ephemerides)} ephemerides, but no common valid subframe epoch had at least 4 PRNs."
    )
    return PVTComputationResult(
        solution=None,
        observations=[],
        ephemerides=ephemerides,
        gps_week=None,
        gps_time_of_week_s=None,
        utc_datetime=None,
        receiver_time_offset_s=None,
        residual_rms_m=None,
        summary_lines=summary_lines,
    )
