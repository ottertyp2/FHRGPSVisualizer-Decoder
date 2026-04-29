"""Tracking, bit extraction, and nav decode tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.dsp.acquisition import AcquisitionConfig, acquire_signal
from app.dsp.bitsync import extract_navigation_bits
from app.dsp.demo import generate_demo_signal
from app.dsp.navdecode import (
    PREAMBLE,
    build_subframes,
    check_lnav_word,
    compute_lnav_parity,
    decode_navigation_bits,
    decode_navigation_from_tracking,
    maybe_correct_word,
    parse_how,
)
from app.dsp.tracking import track_signal
from app.dsp.tracking import _CorrelatorOutput, _loop_discriminators
from app.models import AcquisitionCandidate, AcquisitionResult, BitDecisionResult, SessionConfig


def _int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _make_lnav_word(data_bits: list[int], previous_word: list[int] | None = None) -> list[int]:
    previous = previous_word or [0] * 30
    d29_star = previous[28]
    d30_star = previous[29]
    transmitted_data = [bit ^ d30_star for bit in data_bits]
    parity = compute_lnav_parity(data_bits, d29_star, d30_star)
    return transmitted_data + parity


def _make_synthetic_subframe(
    subframe_id: int = 1,
    tow_count: int = 100,
    page_id: int = 12,
) -> list[list[int]]:
    tlm_data = [int(bit) for bit in PREAMBLE] + _int_to_bits(0x1234, 16)
    how_data = _int_to_bits(tow_count, 17) + [0, 0] + _int_to_bits(subframe_id, 3) + [0, 0]
    payload = []
    for word_number in range(3, 11):
        data = _int_to_bits((word_number * 0x1555 + subframe_id) & 0xFFFFFF, 24)
        if word_number == 3 and subframe_id in (4, 5):
            data = [0, 1] + _int_to_bits(page_id, 6) + data[8:]
        payload.append(data)

    words: list[list[int]] = []
    previous_word: list[int] | None = None
    for data_bits in [tlm_data, how_data, *payload]:
        word = _make_lnav_word(data_bits, previous_word)
        words.append(word)
        previous_word = word
    return words


def _flatten_words(words: list[list[int]]) -> np.ndarray:
    return np.asarray([bit for word in words for bit in word], dtype=np.int8)


def _make_bit_result(bit_values: np.ndarray, confidences: np.ndarray | None = None) -> BitDecisionResult:
    if confidences is None:
        confidences = np.ones(bit_values.size, dtype=np.float32)
    return BitDecisionResult(
        prompt_ms=np.zeros(bit_values.size * 20, dtype=np.float32),
        best_offset_ms=0,
        bit_sums=np.where(bit_values > 0, 1.0, -1.0).astype(np.float32),
        bit_values=bit_values.astype(np.int8),
        confidences=confidences.astype(np.float32),
        bit_start_ms=np.arange(bit_values.size, dtype=np.int32) * 20,
    )


def test_lnav_parity_valid_word_remains_valid() -> None:
    words = _make_synthetic_subframe()

    assert check_lnav_word(words[0])

    nav_result = decode_navigation_bits(_make_bit_result(_flatten_words(words)))
    first_word = nav_result.words[0]

    assert first_word.parity_ok
    assert not first_word.corrected
    assert first_word.corrected_bit_index is None


def test_single_low_confidence_bit_is_corrected() -> None:
    words = _make_synthetic_subframe(subframe_id=2)
    bit_values = _flatten_words(words)
    flipped_index = 2 * 30 + 5
    bit_values[flipped_index] ^= 1
    confidences = np.ones(bit_values.size, dtype=np.float32)
    confidences[flipped_index] = 0.01

    nav_result = decode_navigation_bits(_make_bit_result(bit_values, confidences))
    corrected_word = next(word for word in nav_result.words if word.start_bit == 60)

    assert corrected_word.parity_ok
    assert corrected_word.corrected
    assert corrected_word.corrected_bit_index == 5
    assert nav_result.corrected_word_count >= 1
    assert nav_result.subframes[0].corrected_words == 1


def test_two_low_confidence_errors_are_not_corrected() -> None:
    words = _make_synthetic_subframe(subframe_id=2)
    bit_values = _flatten_words(words)
    flipped_indices = [2 * 30 + 5, 2 * 30 + 6]
    for index in flipped_indices:
        bit_values[index] ^= 1
    confidences = np.ones(bit_values.size, dtype=np.float32)
    for index in flipped_indices:
        confidences[index] = 0.01

    nav_result = decode_navigation_bits(_make_bit_result(bit_values, confidences))
    failed_word = next(word for word in nav_result.words if word.start_bit == 60)

    assert not failed_word.parity_ok
    assert not failed_word.corrected
    assert failed_word.corrected_bit_index is None


def test_maybe_correct_word_rejects_incomplete_words() -> None:
    corrected, was_corrected, corrected_index = maybe_correct_word([1, 0, 1])

    assert corrected == [1, 0, 1]
    assert not was_corrected
    assert corrected_index is None


def test_preamble_and_ten_words_group_into_navigation_subframe() -> None:
    words = _make_synthetic_subframe(subframe_id=1)
    bit_values = _flatten_words(words).astype(int).tolist()

    subframes = build_subframes(bit_values, [0], confidences=np.ones(len(bit_values), dtype=np.float32))

    assert len(subframes) == 1
    assert subframes[0].start_bit == 0
    assert len(subframes[0].words) == 10
    assert subframes[0].category == "Clock / satellite health"


def test_subframe_id_is_decoded_from_synthetic_how() -> None:
    words = _make_synthetic_subframe(subframe_id=3, tow_count=42)
    nav_result = decode_navigation_bits(_make_bit_result(_flatten_words(words)))
    subframe = nav_result.subframes[0]
    how = parse_how(subframe.words[1], subframe.words[0])

    assert subframe.subframe_id == 3
    assert subframe.tow_seconds == 252
    assert subframe.category == "Ephemeris part 2"
    assert how["subframe_id"] == 3
    assert how["tow_seconds"] == 252


def test_tracking_and_nav_pipeline() -> None:
    demo = generate_demo_signal(duration_s=0.5, doppler_hz=1500.0, code_phase_samples=300)
    acquisition = acquire_signal(
        demo.samples,
        AcquisitionConfig(
            sample_rate=demo.sample_rate,
            prn=demo.prn,
            doppler_min=-3000,
            doppler_max=3000,
            doppler_step=250,
            integration_ms=4,
        ),
    )
    session = SessionConfig(
        sample_rate=demo.sample_rate,
        sample_count=demo.samples.size,
        tracking_ms=300,
        prn=demo.prn,
    )
    tracking = track_signal(demo.samples, session, acquisition)
    assert tracking.prompt_i.size >= 200
    assert tracking.prompt_mag.mean() > 0.01
    assert tracking.lock_detected
    bit_result = extract_navigation_bits(tracking)
    assert bit_result.bit_values.size >= 5
    nav_result = decode_navigation_bits(bit_result)
    assert len(nav_result.summary_lines) >= 1


def test_tracking_pll_discriminator_ignores_bpsk_data_sign() -> None:
    positive = _CorrelatorOutput(prompt_code=None, early=1.0 + 0.0j, prompt=1.0 + 0.2j, late=1.0 + 0.0j)
    negative = _CorrelatorOutput(prompt_code=None, early=1.0 + 0.0j, prompt=-1.0 - 0.2j, late=1.0 + 0.0j)

    _dll_positive, pll_positive = _loop_discriminators(positive)
    _dll_negative, pll_negative = _loop_discriminators(negative)

    assert pll_positive == pytest.approx(pll_negative)


def test_tracking_rejects_prn_mismatch_between_session_and_acquisition() -> None:
    acquisition = AcquisitionResult(
        prn=3,
        sample_rate_hz=1_000_000.0,
        search_center_hz=0.0,
        doppler_bins_hz=np.asarray([0.0], dtype=np.float32),
        code_phases_samples=np.asarray([0], dtype=np.int32),
        heatmap=np.ones((1, 1), dtype=np.float32),
        best_candidate=AcquisitionCandidate(
            prn=3,
            doppler_hz=0.0,
            carrier_frequency_hz=0.0,
            code_phase_samples=0,
            metric=9.0,
        ),
    )
    session = SessionConfig(sample_rate=1_000_000.0, sample_count=1_000, tracking_ms=1, prn=8)
    samples = np.zeros(1_000, dtype=np.complex64)

    with pytest.raises(ValueError, match="Tracking PRN mismatch"):
        track_signal(samples, session, acquisition)


def test_tracking_rejects_prn_mismatch_inside_acquisition_result() -> None:
    acquisition = AcquisitionResult(
        prn=3,
        sample_rate_hz=1_000_000.0,
        search_center_hz=0.0,
        doppler_bins_hz=np.asarray([0.0], dtype=np.float32),
        code_phases_samples=np.asarray([0], dtype=np.int32),
        heatmap=np.ones((1, 1), dtype=np.float32),
        best_candidate=AcquisitionCandidate(
            prn=8,
            doppler_hz=0.0,
            carrier_frequency_hz=0.0,
            code_phase_samples=0,
            metric=9.0,
        ),
    )
    session = SessionConfig(sample_rate=1_000_000.0, sample_count=1_000, tracking_ms=1, prn=3)
    samples = np.zeros(1_000, dtype=np.complex64)

    with pytest.raises(ValueError, match="Acquisition PRN mismatch"):
        track_signal(samples, session, acquisition)


def test_tracking_rejects_sample_rate_mismatch() -> None:
    acquisition = AcquisitionResult(
        prn=3,
        sample_rate_hz=1_000_000.0,
        search_center_hz=0.0,
        doppler_bins_hz=np.asarray([0.0], dtype=np.float32),
        code_phases_samples=np.asarray([0], dtype=np.int32),
        heatmap=np.ones((1, 1), dtype=np.float32),
        best_candidate=AcquisitionCandidate(
            prn=3,
            doppler_hz=0.0,
            carrier_frequency_hz=0.0,
            code_phase_samples=0,
            metric=9.0,
        ),
    )
    session = SessionConfig(sample_rate=1_001_000.0, sample_count=1_001, tracking_ms=1, prn=3)
    samples = np.zeros(1_001, dtype=np.complex64)

    with pytest.raises(ValueError, match="sample-rate mismatch"):
        track_signal(samples, session, acquisition)


def test_tracking_rejects_search_center_mismatch() -> None:
    acquisition = AcquisitionResult(
        prn=3,
        sample_rate_hz=1_000_000.0,
        search_center_hz=0.0,
        doppler_bins_hz=np.asarray([0.0], dtype=np.float32),
        code_phases_samples=np.asarray([0], dtype=np.int32),
        heatmap=np.ones((1, 1), dtype=np.float32),
        best_candidate=AcquisitionCandidate(
            prn=3,
            doppler_hz=0.0,
            carrier_frequency_hz=0.0,
            code_phase_samples=0,
            metric=9.0,
        ),
    )
    session = SessionConfig(
        sample_rate=1_000_000.0,
        sample_count=1_000,
        tracking_ms=1,
        prn=3,
        is_baseband=False,
        if_frequency_hz=1_000.0,
    )
    samples = np.zeros(1_000, dtype=np.complex64)

    with pytest.raises(ValueError, match="search-center mismatch"):
        track_signal(samples, session, acquisition)


def test_navigation_decode_pipeline_reports_progress() -> None:
    demo = generate_demo_signal(duration_s=0.5, doppler_hz=1500.0, code_phase_samples=300)
    acquisition = acquire_signal(
        demo.samples,
        AcquisitionConfig(
            sample_rate=demo.sample_rate,
            prn=demo.prn,
            doppler_min=-3000,
            doppler_max=3000,
            doppler_step=250,
            integration_ms=4,
        ),
    )
    session = SessionConfig(
        sample_rate=demo.sample_rate,
        sample_count=demo.samples.size,
        tracking_ms=300,
        prn=demo.prn,
    )
    tracking = track_signal(demo.samples, session, acquisition)

    progress_updates: list[int] = []
    log_messages: list[str] = []
    bit_result, nav_result = decode_navigation_from_tracking(
        tracking,
        progress_callback=progress_updates.append,
        log_callback=log_messages.append,
    )

    assert progress_updates == [5, 55, 100]
    assert len(log_messages) == 3
    assert "extracting 20 ms navigation bits" in log_messages[0]
    assert "scanning" in log_messages[1]
    assert bit_result.bit_values.size >= 5
    assert len(nav_result.summary_lines) >= 1
    prompt_i_bits, prompt_i_nav = decode_navigation_from_tracking(tracking, bit_source="prompt_i")
    assert prompt_i_bits.bit_values.size >= 5
    assert prompt_i_nav.summary_lines[0] == "Bit source used: prompt I."
