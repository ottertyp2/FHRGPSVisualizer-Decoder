"""Small synthetic signals for the GUI concept lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.dsp.gps_ca import CA_CODE_RATE_HZ, CA_CODE_LENGTH, sample_ca_code


@dataclass(slots=True)
class ConceptLabConfig:
    """Parameters for a tiny didactic GPS-like signal."""

    sample_rate_hz: float = CA_CODE_RATE_HZ
    duration_ms: int = 40
    prn: int = 1
    doppler_hz: float = 700.0
    code_phase_samples: int = 120
    noise_std: float = 0.15
    second_prn_enabled: bool = False
    second_prn: int = 7
    second_code_phase_samples: int = 410
    selected_amplitude: float = 1.0
    second_amplitude: float = 0.65
    carrier_phase_rad: float = 0.65
    seed: int = 1234


@dataclass(slots=True)
class ConceptLabResult:
    """Derived arrays shown in the concept lab plots."""

    config: ConceptLabConfig
    time_s: np.ndarray
    ideal_bpsk: np.ndarray
    raw_iq: np.ndarray
    carrier_wiped_iq: np.ndarray
    despread_iq: np.ndarray
    correlation_code_phases: np.ndarray
    correlation_values: np.ndarray
    acquisition_doppler_bins_hz: np.ndarray
    acquisition_heatmap: np.ndarray
    prompt_times_ms: np.ndarray
    prompt_points: np.ndarray


def _samples_per_ms(sample_rate_hz: float) -> int:
    """Return the sample count for one C/A-code period."""

    return max(1, int(round(float(sample_rate_hz) * 1e-3)))


def _phase_samples_to_chips(code_phase_samples: int, sample_rate_hz: float) -> float:
    """Convert a sample offset inside 1 ms into C/A-code chips."""

    samples_per_ms = _samples_per_ms(sample_rate_hz)
    return float(int(code_phase_samples) % samples_per_ms) * CA_CODE_LENGTH / float(samples_per_ms)


def _code_for_phase(prn: int, sample_rate_hz: float, sample_count: int, code_phase_samples: int) -> np.ndarray:
    """Sample one local PRN replica from a code-phase offset in samples."""

    return sample_ca_code(
        int(prn),
        float(sample_rate_hz),
        int(sample_count),
        _phase_samples_to_chips(code_phase_samples, sample_rate_hz),
    )


def _correlation_profile(
    block: np.ndarray,
    prn: int,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return correlation magnitude indexed by code-phase sample offset."""

    samples_per_ms = block.size
    local_code = sample_ca_code(int(prn), float(sample_rate_hz), samples_per_ms)
    correlation = np.fft.ifft(np.fft.fft(block) * np.conj(np.fft.fft(local_code)))
    code_phase_axis = (samples_per_ms - np.arange(samples_per_ms, dtype=np.int32)) % samples_per_ms
    order = np.argsort(code_phase_axis)
    return code_phase_axis[order].astype(np.int32), np.abs(correlation[order]).astype(np.float32)


def _component(
    prn: int,
    sample_rate_hz: float,
    time_s: np.ndarray,
    doppler_hz: float,
    code_phase_samples: int,
    amplitude: float,
    carrier_phase_rad: float,
) -> np.ndarray:
    """Build one simplified BPSK PRN component."""

    code = _code_for_phase(prn, sample_rate_hz, time_s.size, code_phase_samples)
    carrier = np.exp(1j * (2.0 * np.pi * float(doppler_hz) * time_s + float(carrier_phase_rad)))
    return (float(amplitude) * code * carrier).astype(np.complex64)


def generate_concept_lab_signal(config: ConceptLabConfig | None = None) -> ConceptLabResult:
    """Generate a fast, deterministic GPS-like teaching signal."""

    config = config or ConceptLabConfig()
    sample_rate_hz = float(config.sample_rate_hz)
    samples_per_ms = _samples_per_ms(sample_rate_hz)
    sample_count = max(samples_per_ms, samples_per_ms * max(1, int(config.duration_ms)))
    time_s = np.arange(sample_count, dtype=np.float64) / sample_rate_hz

    selected_code = _code_for_phase(config.prn, sample_rate_hz, sample_count, config.code_phase_samples)
    ideal_bpsk = (
        float(config.selected_amplitude)
        * selected_code
        * np.exp(1j * float(config.carrier_phase_rad))
    ).astype(np.complex64)
    signal = _component(
        config.prn,
        sample_rate_hz,
        time_s,
        config.doppler_hz,
        config.code_phase_samples,
        config.selected_amplitude,
        config.carrier_phase_rad,
    )
    if config.second_prn_enabled:
        signal = signal + _component(
            config.second_prn,
            sample_rate_hz,
            time_s,
            config.doppler_hz,
            config.second_code_phase_samples,
            config.second_amplitude,
            -0.35,
        )

    rng = np.random.default_rng(int(config.seed))
    noise = (
        rng.normal(0.0, float(config.noise_std), sample_count)
        + 1j * rng.normal(0.0, float(config.noise_std), sample_count)
    ) / np.sqrt(2.0)
    raw_iq = (signal + noise).astype(np.complex64)

    wipe = np.exp(-1j * 2.0 * np.pi * float(config.doppler_hz) * time_s)
    carrier_wiped = (raw_iq * wipe).astype(np.complex64)
    despread = (carrier_wiped * selected_code).astype(np.complex64)

    first_ms = carrier_wiped[:samples_per_ms]
    code_phase_axis, correlation_values = _correlation_profile(first_ms, config.prn, sample_rate_hz)

    doppler_bins = np.arange(
        float(config.doppler_hz) - 2_000.0,
        float(config.doppler_hz) + 2_000.0 + 250.0,
        250.0,
        dtype=np.float32,
    )
    acquisition_rows: list[np.ndarray] = []
    raw_first_ms = raw_iq[:samples_per_ms]
    first_ms_time = time_s[:samples_per_ms]
    for doppler in doppler_bins:
        wiped_block = raw_first_ms * np.exp(-1j * 2.0 * np.pi * float(doppler) * first_ms_time)
        _axis, values = _correlation_profile(wiped_block, config.prn, sample_rate_hz)
        acquisition_rows.append(values)
    acquisition_heatmap = np.vstack(acquisition_rows).astype(np.float32)

    prompt_count = sample_count // samples_per_ms
    local_prompt_code = _code_for_phase(config.prn, sample_rate_hz, samples_per_ms, config.code_phase_samples)
    prompt_points = np.empty(prompt_count, dtype=np.complex64)
    for ms_index in range(prompt_count):
        start = ms_index * samples_per_ms
        stop = start + samples_per_ms
        block = carrier_wiped[start:stop]
        prompt_points[ms_index] = np.complex64(np.vdot(local_prompt_code, block) / samples_per_ms)
    prompt_times_ms = np.arange(prompt_count, dtype=np.float32)

    return ConceptLabResult(
        config=config,
        time_s=time_s.astype(np.float32),
        ideal_bpsk=ideal_bpsk,
        raw_iq=raw_iq,
        carrier_wiped_iq=carrier_wiped,
        despread_iq=despread,
        correlation_code_phases=code_phase_axis,
        correlation_values=correlation_values,
        acquisition_doppler_bins_hz=doppler_bins,
        acquisition_heatmap=acquisition_heatmap,
        prompt_times_ms=prompt_times_ms,
        prompt_points=prompt_points,
    )
