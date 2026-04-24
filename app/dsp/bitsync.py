"""Navigation bit timing and hard decisions from prompt integrations."""

from __future__ import annotations

import numpy as np

from app.models import BitDecisionResult, TrackingState


def _carrier_aligned_prompt_ms(tracking: TrackingState) -> np.ndarray:
    """Return prompt samples after a simple non-data-aided BPSK phase alignment."""

    prompt = tracking.iq_views.get("Integrated prompt")
    if prompt is None or prompt.size != tracking.prompt_i.size or prompt.size < 5:
        return tracking.prompt_i.astype(np.float64)

    complex_prompt = prompt.astype(np.complex128, copy=False)
    magnitude = np.abs(complex_prompt)
    valid = magnitude > 1e-12
    if np.count_nonzero(valid) < 5:
        return tracking.prompt_i.astype(np.float64)

    unit_squared = np.zeros_like(complex_prompt)
    unit_squared[valid] = (complex_prompt[valid] / magnitude[valid]) ** 2
    window = min(41, max(5, (unit_squared.size // 100) | 1))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smooth_real = np.convolve(unit_squared.real, kernel, mode="same")
    smooth_imag = np.convolve(unit_squared.imag, kernel, mode="same")
    phase = 0.5 * np.unwrap(np.angle(smooth_real + 1j * smooth_imag))
    aligned = complex_prompt * np.exp(-1j * phase)
    return aligned.real.astype(np.float64)


def form_navigation_bits(prompt_ms: np.ndarray) -> BitDecisionResult:
    """Find the best 20 ms bit boundary and form hard bit decisions from prompt values."""

    prompt_ms = np.asarray(prompt_ms, dtype=np.float64)
    if prompt_ms.size < 20:
        raise ValueError("Need at least 20 ms of prompt integrations for bit extraction.")

    best_offset = 0
    best_score = -np.inf
    best_sums = np.empty(0, dtype=np.float64)
    for offset in range(20):
        usable = prompt_ms[offset:]
        usable = usable[: (usable.size // 20) * 20]
        if usable.size == 0:
            continue
        sums = usable.reshape(-1, 20).sum(axis=1)
        score = float(np.sum(np.abs(sums)))
        if score > best_score:
            best_score = score
            best_offset = offset
            best_sums = sums

    bit_values = (best_sums >= 0.0).astype(np.int8)
    confidence_den = np.maximum(
        1e-9,
        np.sum(np.abs(prompt_ms[best_offset : best_offset + best_sums.size * 20].reshape(-1, 20)), axis=1),
    )
    confidences = np.abs(best_sums) / confidence_den
    bit_starts = best_offset + np.arange(best_sums.size) * 20

    return BitDecisionResult(
        prompt_ms=prompt_ms.astype(np.float32, copy=True),
        best_offset_ms=best_offset,
        bit_sums=best_sums.astype(np.float32),
        bit_values=bit_values,
        confidences=confidences.astype(np.float32),
        bit_start_ms=bit_starts.astype(np.int32),
    )


def extract_navigation_bits(tracking: TrackingState) -> BitDecisionResult:
    """Find the best 20 ms bit boundary and form hard bit decisions."""

    return form_navigation_bits(_carrier_aligned_prompt_ms(tracking))
