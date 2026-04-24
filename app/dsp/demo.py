"""Synthetic GPS-like signal generation for demos and tests."""

from __future__ import annotations

import numpy as np

from app.dsp.gps_ca import CA_CODE_RATE_HZ, sample_ca_code
from app.dsp.navdecode import PREAMBLE, compute_lnav_parity
from app.models import DEFAULT_SAMPLE_RATE_HZ, DemoSignalResult


def _build_demo_nav_bits(num_bits: int) -> np.ndarray:
    """Build a deterministic bit stream with an LNAV-like preamble and valid parity words."""

    rng = np.random.default_rng(42)
    stream: list[int] = []
    previous_word = [0] * 30
    while len(stream) < num_bits:
        data = rng.integers(0, 2, size=24, endpoint=False).tolist()
        if len(stream) == 0:
            preamble_bits = [int(char) for char in PREAMBLE]
            data[:8] = preamble_bits
        parity = compute_lnav_parity(data, previous_word[28], previous_word[29])
        word = data + parity
        stream.extend(word)
        previous_word = word
    return np.asarray(stream[:num_bits], dtype=np.int8)


def generate_demo_signal(
    sample_rate: float = DEFAULT_SAMPLE_RATE_HZ,
    duration_s: float = 0.5,
    prn: int = 1,
    doppler_hz: float = 2_000.0,
    code_phase_samples: int = 350,
    cn0_like_amplitude: float = 0.7,
    noise_std: float = 0.35,
) -> DemoSignalResult:
    """Generate a simplified GPS-like baseband signal."""

    sample_count = int(sample_rate * duration_s)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    bit_samples = int(round(sample_rate * 0.02))
    nav_bit_count = max(1, int(np.ceil(sample_count / max(bit_samples, 1))))
    nav_bits = _build_demo_nav_bits(nav_bit_count)
    nav_stream = np.repeat(1 - 2 * nav_bits, bit_samples)[:sample_count].astype(np.float32)
    code_phase_chips = code_phase_samples * 1023.0 / int(round(sample_rate * 1e-3))
    code = sample_ca_code(prn, sample_rate, sample_count, code_phase_chips, CA_CODE_RATE_HZ)
    carrier = np.exp(1j * 2.0 * np.pi * doppler_hz * time)
    signal = cn0_like_amplitude * code * nav_stream * carrier
    noise = (np.random.default_rng(7).normal(0.0, noise_std, sample_count) + 1j * np.random.default_rng(8).normal(0.0, noise_std, sample_count)) / np.sqrt(2.0)
    samples = (signal + noise).astype(np.complex64)
    return DemoSignalResult(
        samples=samples,
        prn=prn,
        sample_rate=sample_rate,
        doppler_hz=doppler_hz,
        code_phase_samples=code_phase_samples,
        nav_bits=nav_bits,
        metadata={"duration_s": duration_s},
    )
