"""Decode GPS LNAV ephemerides and compute satellite positions.

The functions in this module deliberately stay close to the broadcast
navigation equations: subframes 1, 2, and 3 provide the satellite clock and
orbit parameters, then the user algorithm turns those fields into ECEF
coordinates for one transmit time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.dsp.navdecode import extract_data_bits
from app.dsp.utils import bits_to_str
from app.models import NavigationSubframe, NavigationWord

GPS_WEEK_SECONDS = 604_800.0
GPS_HALF_WEEK_SECONDS = GPS_WEEK_SECONDS / 2.0
GPS_MU_M3_S2 = 3.986005e14
GPS_EARTH_ROTATION_RAD_S = 7.2921151467e-5
GPS_RELATIVISTIC_F = -4.442807633e-10


@dataclass(slots=True)
class GpsEphemeris:
    """Broadcast LNAV ephemeris for one GPS PRN."""

    prn: int
    week_number_mod1024: int
    ura_index: int
    health: int
    iodc: int
    iode: int
    toc_s: float
    toe_s: float
    tgd_s: float
    af0_s: float
    af1_s_s: float
    af2_s_s2: float
    crs_m: float
    delta_n_rad_s: float
    m0_rad: float
    cuc_rad: float
    eccentricity: float
    cus_rad: float
    sqrt_a_sqrt_m: float
    cic_rad: float
    omega0_rad: float
    cis_rad: float
    i0_rad: float
    crc_m: float
    omega_rad: float
    omega_dot_rad_s: float
    idot_rad_s: float
    fit_interval_flag: int
    subframe_start_bits: tuple[int, int, int]


def _uint(bits: list[int]) -> int:
    if not bits:
        return 0
    return int(bits_to_str([int(bit) for bit in bits]), 2)


def _sint(bits: list[int]) -> int:
    """Return a two's-complement integer from an MSB-first bit list."""

    value = _uint(bits)
    if bits and bits[0]:
        value -= 1 << len(bits)
    return value


def _word_data(words: list[NavigationWord], word_number: int) -> list[int]:
    """Return the 24 de-whitened data bits for one 1-based LNAV word."""

    index = int(word_number) - 1
    previous = words[index - 1] if index > 0 else None
    return extract_data_bits(words[index], previous)


def _field(word_bits: list[int], first_bit: int, last_bit: int) -> list[int]:
    """Slice 1-based inclusive D-bit ranges from one LNAV data word."""

    return word_bits[first_bit - 1 : last_bit]


def _join(*parts: list[int]) -> list[int]:
    joined: list[int] = []
    for part in parts:
        joined.extend(part)
    return joined


def _valid_subframes(subframes: list[NavigationSubframe], subframe_id: int) -> list[NavigationSubframe]:
    return [subframe for subframe in subframes if subframe.valid and subframe.subframe_id == subframe_id]


@dataclass(slots=True)
class _ClockFields:
    week_number_mod1024: int
    ura_index: int
    health: int
    iodc: int
    toc_s: float
    tgd_s: float
    af0_s: float
    af1_s_s: float
    af2_s_s2: float
    start_bit: int


@dataclass(slots=True)
class _OrbitPart2:
    iode: int
    crs_m: float
    delta_n_rad_s: float
    m0_rad: float
    cuc_rad: float
    eccentricity: float
    cus_rad: float
    sqrt_a_sqrt_m: float
    toe_s: float
    fit_interval_flag: int
    start_bit: int


@dataclass(slots=True)
class _OrbitPart3:
    iode: int
    cic_rad: float
    omega0_rad: float
    cis_rad: float
    i0_rad: float
    crc_m: float
    omega_rad: float
    omega_dot_rad_s: float
    idot_rad_s: float
    start_bit: int


def _parse_clock(subframe: NavigationSubframe) -> _ClockFields:
    words = subframe.words
    w3 = _word_data(words, 3)
    w7 = _word_data(words, 7)
    w8 = _word_data(words, 8)
    w9 = _word_data(words, 9)
    w10 = _word_data(words, 10)

    iodc = (_uint(_field(w3, 23, 24)) << 8) | _uint(_field(w8, 1, 8))
    return _ClockFields(
        week_number_mod1024=_uint(_field(w3, 1, 10)),
        ura_index=_uint(_field(w3, 13, 16)),
        health=_uint(_field(w3, 17, 22)),
        iodc=iodc,
        toc_s=float(_uint(_field(w8, 9, 24)) * 2**4),
        tgd_s=float(_sint(_field(w7, 17, 24)) * 2**-31),
        af2_s_s2=float(_sint(_field(w9, 1, 8)) * 2**-55),
        af1_s_s=float(_sint(_field(w9, 9, 24)) * 2**-43),
        af0_s=float(_sint(_field(w10, 1, 22)) * 2**-31),
        start_bit=subframe.start_bit,
    )


def _parse_orbit_part2(subframe: NavigationSubframe) -> _OrbitPart2:
    words = subframe.words
    w3 = _word_data(words, 3)
    w4 = _word_data(words, 4)
    w5 = _word_data(words, 5)
    w6 = _word_data(words, 6)
    w7 = _word_data(words, 7)
    w8 = _word_data(words, 8)
    w9 = _word_data(words, 9)
    w10 = _word_data(words, 10)

    return _OrbitPart2(
        iode=_uint(_field(w3, 1, 8)),
        crs_m=float(_sint(_field(w3, 9, 24)) * 2**-5),
        delta_n_rad_s=float(_sint(_field(w4, 1, 16)) * 2**-43 * np.pi),
        m0_rad=float(_sint(_join(_field(w4, 17, 24), w5)) * 2**-31 * np.pi),
        cuc_rad=float(_sint(_field(w6, 1, 16)) * 2**-29),
        eccentricity=float(_uint(_join(_field(w6, 17, 24), w7)) * 2**-33),
        cus_rad=float(_sint(_field(w8, 1, 16)) * 2**-29),
        sqrt_a_sqrt_m=float(_uint(_join(_field(w8, 17, 24), w9)) * 2**-19),
        toe_s=float(_uint(_field(w10, 1, 16)) * 2**4),
        fit_interval_flag=_uint(_field(w10, 17, 17)),
        start_bit=subframe.start_bit,
    )


def _parse_orbit_part3(subframe: NavigationSubframe) -> _OrbitPart3:
    words = subframe.words
    w3 = _word_data(words, 3)
    w4 = _word_data(words, 4)
    w5 = _word_data(words, 5)
    w6 = _word_data(words, 6)
    w7 = _word_data(words, 7)
    w8 = _word_data(words, 8)
    w9 = _word_data(words, 9)
    w10 = _word_data(words, 10)

    return _OrbitPart3(
        iode=_uint(_field(w10, 1, 8)),
        cic_rad=float(_sint(_field(w3, 1, 16)) * 2**-29),
        omega0_rad=float(_sint(_join(_field(w3, 17, 24), w4)) * 2**-31 * np.pi),
        cis_rad=float(_sint(_field(w5, 1, 16)) * 2**-29),
        i0_rad=float(_sint(_join(_field(w5, 17, 24), w6)) * 2**-31 * np.pi),
        crc_m=float(_sint(_field(w7, 1, 16)) * 2**-5),
        omega_rad=float(_sint(_join(_field(w7, 17, 24), w8)) * 2**-31 * np.pi),
        omega_dot_rad_s=float(_sint(w9) * 2**-43 * np.pi),
        idot_rad_s=float(_sint(_field(w10, 9, 22)) * 2**-43 * np.pi),
        start_bit=subframe.start_bit,
    )


def decode_ephemeris(prn: int, subframes: list[NavigationSubframe]) -> GpsEphemeris | None:
    """Decode one complete LNAV ephemeris from valid subframes 1, 2, and 3."""

    clocks = [_parse_clock(subframe) for subframe in _valid_subframes(subframes, 1)]
    part2_items = [_parse_orbit_part2(subframe) for subframe in _valid_subframes(subframes, 2)]
    part3_items = [_parse_orbit_part3(subframe) for subframe in _valid_subframes(subframes, 3)]
    if not clocks or not part2_items or not part3_items:
        return None

    for part2 in part2_items:
        for part3 in part3_items:
            if part2.iode != part3.iode:
                continue
            matching_clocks = [clock for clock in clocks if (clock.iodc & 0xFF) == part2.iode]
            clock = matching_clocks[0] if matching_clocks else clocks[0]
            return GpsEphemeris(
                prn=int(prn),
                week_number_mod1024=clock.week_number_mod1024,
                ura_index=clock.ura_index,
                health=clock.health,
                iodc=clock.iodc,
                iode=part2.iode,
                toc_s=clock.toc_s,
                toe_s=part2.toe_s,
                tgd_s=clock.tgd_s,
                af0_s=clock.af0_s,
                af1_s_s=clock.af1_s_s,
                af2_s_s2=clock.af2_s_s2,
                crs_m=part2.crs_m,
                delta_n_rad_s=part2.delta_n_rad_s,
                m0_rad=part2.m0_rad,
                cuc_rad=part2.cuc_rad,
                eccentricity=part2.eccentricity,
                cus_rad=part2.cus_rad,
                sqrt_a_sqrt_m=part2.sqrt_a_sqrt_m,
                cic_rad=part3.cic_rad,
                omega0_rad=part3.omega0_rad,
                cis_rad=part3.cis_rad,
                i0_rad=part3.i0_rad,
                crc_m=part3.crc_m,
                omega_rad=part3.omega_rad,
                omega_dot_rad_s=part3.omega_dot_rad_s,
                idot_rad_s=part3.idot_rad_s,
                fit_interval_flag=part2.fit_interval_flag,
                subframe_start_bits=(clock.start_bit, part2.start_bit, part3.start_bit),
            )
    return None


def gps_time_delta_seconds(delta_s: float) -> float:
    """Wrap a GPS time difference into +/- half a week."""

    value = float(delta_s)
    while value > GPS_HALF_WEEK_SECONDS:
        value -= GPS_WEEK_SECONDS
    while value < -GPS_HALF_WEEK_SECONDS:
        value += GPS_WEEK_SECONDS
    return value


def _eccentric_anomaly(ephemeris: GpsEphemeris, transmit_time_s: float) -> tuple[float, float]:
    semi_major_axis_m = ephemeris.sqrt_a_sqrt_m * ephemeris.sqrt_a_sqrt_m
    computed_mean_motion = np.sqrt(GPS_MU_M3_S2 / (semi_major_axis_m**3))
    corrected_mean_motion = computed_mean_motion + ephemeris.delta_n_rad_s
    tk = gps_time_delta_seconds(float(transmit_time_s) - ephemeris.toe_s)
    mean_anomaly = ephemeris.m0_rad + corrected_mean_motion * tk
    eccentric_anomaly = mean_anomaly
    for _ in range(12):
        eccentric_anomaly = mean_anomaly + ephemeris.eccentricity * np.sin(eccentric_anomaly)
    return float(eccentric_anomaly), float(tk)


def satellite_clock_correction_s(ephemeris: GpsEphemeris, transmit_time_s: float) -> float:
    """Return the L1 C/A satellite clock correction in seconds."""

    eccentric_anomaly, _tk = _eccentric_anomaly(ephemeris, transmit_time_s)
    clock_dt = gps_time_delta_seconds(float(transmit_time_s) - ephemeris.toc_s)
    relativistic_s = (
        GPS_RELATIVISTIC_F
        * ephemeris.eccentricity
        * ephemeris.sqrt_a_sqrt_m
        * np.sin(eccentric_anomaly)
    )
    return float(
        ephemeris.af0_s
        + ephemeris.af1_s_s * clock_dt
        + ephemeris.af2_s_s2 * clock_dt * clock_dt
        + relativistic_s
        - ephemeris.tgd_s
    )


def satellite_position_ecef_m(ephemeris: GpsEphemeris, transmit_time_s: float) -> np.ndarray:
    """Compute the satellite antenna phase-center ECEF position in metres."""

    eccentric_anomaly, tk = _eccentric_anomaly(ephemeris, transmit_time_s)
    eccentricity = ephemeris.eccentricity
    semi_major_axis_m = ephemeris.sqrt_a_sqrt_m * ephemeris.sqrt_a_sqrt_m

    true_anomaly = np.arctan2(
        np.sqrt(1.0 - eccentricity * eccentricity) * np.sin(eccentric_anomaly),
        np.cos(eccentric_anomaly) - eccentricity,
    )
    argument_of_latitude = true_anomaly + ephemeris.omega_rad
    two_phi = 2.0 * argument_of_latitude

    corrected_u = (
        argument_of_latitude
        + ephemeris.cus_rad * np.sin(two_phi)
        + ephemeris.cuc_rad * np.cos(two_phi)
    )
    corrected_r = (
        semi_major_axis_m * (1.0 - eccentricity * np.cos(eccentric_anomaly))
        + ephemeris.crs_m * np.sin(two_phi)
        + ephemeris.crc_m * np.cos(two_phi)
    )
    corrected_i = (
        ephemeris.i0_rad
        + ephemeris.idot_rad_s * tk
        + ephemeris.cis_rad * np.sin(two_phi)
        + ephemeris.cic_rad * np.cos(two_phi)
    )

    x_orbital = corrected_r * np.cos(corrected_u)
    y_orbital = corrected_r * np.sin(corrected_u)
    omega = (
        ephemeris.omega0_rad
        + (ephemeris.omega_dot_rad_s - GPS_EARTH_ROTATION_RAD_S) * tk
        - GPS_EARTH_ROTATION_RAD_S * ephemeris.toe_s
    )

    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(corrected_i)
    sin_i = np.sin(corrected_i)
    return np.asarray(
        [
            x_orbital * cos_omega - y_orbital * cos_i * sin_omega,
            x_orbital * sin_omega + y_orbital * cos_i * cos_omega,
            y_orbital * sin_i,
        ],
        dtype=np.float64,
    )


def rotate_ecef_for_transit(ecef_m: np.ndarray, travel_time_s: float) -> np.ndarray:
    """Rotate a transmit-time satellite position into the receive-time frame."""

    angle = GPS_EARTH_ROTATION_RAD_S * float(travel_time_s)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x, y, z = np.asarray(ecef_m, dtype=np.float64)
    return np.asarray([cos_a * x + sin_a * y, -sin_a * x + cos_a * y, z], dtype=np.float64)
