"""Very first-step LNAV framing helpers."""

from __future__ import annotations

import numpy as np

from app.dsp.bitsync import extract_navigation_bits
from app.dsp.utils import bits_to_str
from app.models import BitDecisionResult, NavigationDecodeResult, NavigationWord, TrackingState

PREAMBLE = "10001011"


def _xor_selected(bits: list[int], indices: tuple[int, ...]) -> int:
    value = 0
    for index in indices:
        value ^= bits[index - 1]
    return value


def compute_lnav_parity(data_bits: list[int], d29_star: int, d30_star: int) -> list[int]:
    """Compute LNAV parity bits D25..D30 for a 24-bit data payload."""

    if len(data_bits) != 24:
        raise ValueError("LNAV parity needs exactly 24 data bits.")

    d = data_bits
    p25 = d29_star ^ _xor_selected(d, (1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23))
    p26 = d30_star ^ _xor_selected(d, (2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24))
    p27 = d29_star ^ _xor_selected(d, (1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22))
    p28 = d30_star ^ _xor_selected(d, (2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23))
    p29 = d30_star ^ _xor_selected(d, (1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24))
    p30 = d29_star ^ _xor_selected(d, (3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24))
    return [p25, p26, p27, p28, p29, p30]


def check_lnav_word(word_bits: list[int], previous_word: list[int] | None = None) -> bool:
    """Check the 6 parity bits of a 30-bit LNAV word."""

    if len(word_bits) != 30:
        return False
    previous_word = previous_word or [0] * 30
    d29_star = previous_word[28]
    d30_star = previous_word[29]
    expected = compute_lnav_parity(word_bits[:24], d29_star, d30_star)
    return expected == word_bits[24:]


def decode_navigation_bits(bits: BitDecisionResult) -> NavigationDecodeResult:
    """Search for the LNAV preamble, align 30-bit words, and run parity checks."""

    as_list = bits.bit_values.astype(int).tolist()
    normal = bits_to_str(as_list)
    inverted_list = [1 - value for value in as_list]
    inverted = bits_to_str(inverted_list)

    results = NavigationDecodeResult()

    def scan(bit_string: str, values: list[int], inverted_flag: bool) -> None:
        for index in range(max(0, len(values) - 30)):
            if bit_string[index : index + 8] != PREAMBLE:
                continue
            results.preamble_indices.append(index)
            previous_word: list[int] | None = None
            found_words = 0
            for word_start in range(index, len(values) - 29, 30):
                word = values[word_start : word_start + 30]
                parity_ok = check_lnav_word(word, previous_word)
                if found_words == 0 and word_start == index:
                    label = "TLM candidate"
                elif found_words == 1 and word_start == index + 30:
                    label = "HOW candidate"
                else:
                    label = ""
                results.words.append(
                    NavigationWord(
                        start_bit=word_start,
                        bits=bits_to_str(word),
                        hex_word=f"0x{int(bits_to_str(word), 2):08X}",
                        parity_ok=parity_ok,
                        is_inverted=inverted_flag,
                        label=label,
                    )
                )
                results.word_start_indices.append(word_start)
                if parity_ok:
                    results.parity_ok_count += 1
                previous_word = word
                found_words += 1
                if found_words >= 10:
                    break
            if found_words:
                polarity = "inverted" if inverted_flag else "normal"
                results.summary_lines.append(
                    f"Preamble at bit {index} ({polarity} polarity), captured {found_words} word candidates."
                )

    scan(normal, as_list, inverted_flag=False)
    scan(inverted, inverted_list, inverted_flag=True)

    if not results.summary_lines:
        results.summary_lines.append("No LNAV preamble detected in the current bit stream.")

    return results


def decode_navigation_from_tracking(
    tracking: TrackingState,
    progress_callback=None,
    log_callback=None,
) -> tuple[BitDecisionResult, NavigationDecodeResult]:
    """Run bit extraction and LNAV framing with coarse progress updates."""

    if log_callback:
        log_callback(f"Decoding PRN {tracking.prn}: extracting 20 ms navigation bits.")
    if progress_callback:
        progress_callback(5)

    bit_result = extract_navigation_bits(tracking)

    if log_callback:
        log_callback(
            f"Decoding PRN {tracking.prn}: scanning {bit_result.bit_values.size} hard decisions for LNAV preambles."
        )
    if progress_callback:
        progress_callback(55)

    nav_result = decode_navigation_bits(bit_result)

    if progress_callback:
        progress_callback(100)
    if log_callback:
        log_callback(
            f"Decoding PRN {tracking.prn}: found {len(nav_result.preamble_indices)} preambles and "
            f"{nav_result.parity_ok_count} parity-valid words."
        )
    return bit_result, nav_result
