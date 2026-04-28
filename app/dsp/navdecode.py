"""First-step LNAV framing and coarse field helpers.

The goal here is deliberately modest: find LNAV subframes, verify words,
make conservative one-bit repairs when confidence supports it, and expose
TLM/HOW plus raw payload areas for the GUI. Full ephemeris/almanac decoding
belongs in a later, more carefully tested layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.dsp.bitsync import _carrier_aligned_prompt_ms
from app.dsp.bitsync import extract_navigation_bits
from app.dsp.bitsync import form_navigation_bits
from app.dsp.utils import bits_to_str
from app.models import (
    BitDecisionResult,
    NavigationDecodeResult,
    NavigationField,
    NavigationSubframe,
    NavigationWord,
    TrackingState,
)

PREAMBLE = "10001011"
LNAV_WORD_BITS = 30
LNAV_DATA_BITS = 24
LNAV_SUBFRAME_WORDS = 10
LNAV_SUBFRAME_BITS = LNAV_WORD_BITS * LNAV_SUBFRAME_WORDS
MAX_TOW_COUNT = 604_800 // 6


def _xor_selected(bits: list[int], indices: tuple[int, ...]) -> int:
    value = 0
    for index in indices:
        value ^= bits[index - 1]
    return value


def _bits_to_int(bit_values: Sequence[int]) -> int:
    """Convert an MSB-first bit sequence to an integer."""

    if len(bit_values) == 0:
        return 0
    return int(bits_to_str([int(value) for value in bit_values]), 2)


def _coerce_word_bits(word: NavigationWord | Sequence[int] | str | None) -> list[int] | None:
    """Return a 0/1 list for a stored word, raw sequence, or bit string."""

    if word is None:
        return None
    if isinstance(word, NavigationWord):
        return [int(value) for value in word.bits]
    if isinstance(word, str):
        return [int(value) for value in word]
    return [int(value) for value in word]


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

    if len(word_bits) != LNAV_WORD_BITS:
        return False
    previous_word = previous_word or [0] * LNAV_WORD_BITS
    d29_star = previous_word[28]
    d30_star = previous_word[29]
    data_bits = extract_data_bits(word_bits, previous_word)
    expected = compute_lnav_parity(data_bits, d29_star, d30_star)
    return expected == word_bits[LNAV_DATA_BITS:]


def extract_data_bits(
    word: NavigationWord | Sequence[int] | str,
    previous_word: NavigationWord | Sequence[int] | str | None = None,
) -> list[int]:
    """Return the 24 recovered LNAV data bits from one transmitted word.

    IS-GPS-200 encodes D1..D24 as d1..d24 XOR D30* from the previous word.
    The first word of an isolated capture has no previous word available, so
    this helper falls back to D30*=0, matching the earlier simple decoder.
    """

    word_bits = _coerce_word_bits(word)
    previous_bits = _coerce_word_bits(previous_word) or [0] * LNAV_WORD_BITS
    if word_bits is None or len(word_bits) < LNAV_DATA_BITS:
        return []
    d30_star = previous_bits[29] if len(previous_bits) >= LNAV_WORD_BITS else 0
    return [bit ^ d30_star for bit in word_bits[:LNAV_DATA_BITS]]


def parse_tlm(
    word: NavigationWord | Sequence[int] | str,
    previous_word: NavigationWord | Sequence[int] | str | None = None,
) -> dict[str, int | str]:
    """Decode the minimal TLM fields used for teaching and display."""

    data_bits = extract_data_bits(word, previous_word)
    preamble = bits_to_str(data_bits[:8])
    tlm_message_bits = data_bits[8:22]
    raw_bits = data_bits[8:24]
    return {
        "preamble": preamble,
        "tlm_message": _bits_to_int(tlm_message_bits),
        "tlm_message_hex": f"0x{_bits_to_int(tlm_message_bits):04X}",
        "tlm_raw": _bits_to_int(raw_bits),
        "tlm_raw_hex": f"0x{_bits_to_int(raw_bits):04X}",
        "tlm_raw_bits": bits_to_str(raw_bits),
    }


def parse_how(
    word: NavigationWord | Sequence[int] | str,
    previous_word: NavigationWord | Sequence[int] | str | None = None,
) -> dict[str, int | str | None]:
    """Decode the minimal HOW fields: TOW count and subframe ID.

    Per IS-GPS-200, HOW bits 1..17 hold the truncated TOW-count message and
    bits 20..22 hold the subframe ID. This helper intentionally stops there;
    alert/anti-spoof/reserved details are shown only as raw bits for now.
    """

    data_bits = extract_data_bits(word, previous_word)
    tow_count = _bits_to_int(data_bits[:17])
    subframe_code = _bits_to_int(data_bits[19:22])
    tow_seconds = tow_count * 6 if 0 <= tow_count < MAX_TOW_COUNT else None
    subframe_id = subframe_code if 1 <= subframe_code <= 5 else None
    return {
        "tow_count": tow_count,
        "tow_seconds": tow_seconds,
        "subframe_id": subframe_id,
        "subframe_code": subframe_code,
        "how_raw": _bits_to_int(data_bits),
        "how_raw_hex": f"0x{_bits_to_int(data_bits):06X}",
        "how_raw_bits": bits_to_str(data_bits),
    }


def classify_subframe(subframe_id: int | None) -> str:
    """Return a didactic LNAV category label for a subframe ID."""

    if subframe_id == 1:
        return "Clock / satellite health"
    if subframe_id == 2:
        return "Ephemeris part 1"
    if subframe_id == 3:
        return "Ephemeris part 2"
    if subframe_id == 4:
        return "Almanac / ionosphere / UTC / special pages"
    if subframe_id == 5:
        return "Almanac / satellite health"
    return "Unknown"


def _confidence_slice(confidences: Sequence[float] | np.ndarray | None, start_bit: int) -> np.ndarray | None:
    if confidences is None:
        return None
    if start_bit < 0 or start_bit + LNAV_WORD_BITS > len(confidences):
        return None
    return np.asarray(confidences[start_bit : start_bit + LNAV_WORD_BITS], dtype=float)


def _word_confidence(confidences: np.ndarray | None) -> float | None:
    if confidences is None or confidences.size == 0:
        return None
    finite = confidences[np.isfinite(confidences)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _low_confidence_indices(confidences: np.ndarray | None) -> set[int]:
    """Return the lower-confidence quartile used as a conservative guard.

    A Hamming-style parity code can turn some two-bit errors into another
    valid word if a third bit is flipped. To avoid that overreach, confidence
    data must agree that the proposed bit was genuinely among the weak bits.
    """

    if confidences is None or confidences.size != LNAV_WORD_BITS:
        return set(range(LNAV_WORD_BITS))
    finite = np.nan_to_num(confidences.astype(float), nan=np.inf, posinf=np.inf, neginf=np.inf)
    valid = finite[np.isfinite(finite)]
    if valid.size == 0:
        return set(range(LNAV_WORD_BITS))
    threshold = float(np.quantile(valid, 0.25))
    return {index for index, value in enumerate(finite) if value <= threshold}


def maybe_correct_word(
    word_bits: list[int],
    previous_word: list[int] | None = None,
    confidences: Sequence[float] | np.ndarray | None = None,
) -> tuple[list[int], bool, int | None]:
    """Try a single-bit parity repair and accept only an unambiguous result."""

    if check_lnav_word(word_bits, previous_word):
        return list(word_bits), False, None

    confidence_values = None if confidences is None else np.asarray(confidences, dtype=float)
    if confidence_values is not None and confidence_values.size == LNAV_WORD_BITS:
        ordered_indices = list(np.argsort(np.nan_to_num(confidence_values, nan=np.inf)))
        low_confidence = _low_confidence_indices(confidence_values)
    else:
        ordered_indices = list(range(LNAV_WORD_BITS))
        low_confidence = set(range(LNAV_WORD_BITS))

    valid_flips: list[tuple[int, list[int]]] = []
    for bit_index in ordered_indices:
        candidate = list(word_bits)
        candidate[bit_index] ^= 1
        if check_lnav_word(candidate, previous_word):
            valid_flips.append((int(bit_index), candidate))

    if len(valid_flips) != 1:
        return list(word_bits), False, None

    corrected_bit_index, corrected_bits = valid_flips[0]
    if corrected_bit_index not in low_confidence:
        return list(word_bits), False, None
    return corrected_bits, True, corrected_bit_index


def _bit_range(subframe_start_bit: int, word_number: int, first_bit: int, last_bit: int) -> str:
    absolute_start = subframe_start_bit + (word_number - 1) * LNAV_WORD_BITS + first_bit - 1
    absolute_end = subframe_start_bit + (word_number - 1) * LNAV_WORD_BITS + last_bit - 1
    return f"W{word_number} D{first_bit}-D{last_bit} (bit {absolute_start}-{absolute_end})"


def _word_label(word_number: int, parity_unchecked: bool) -> str:
    if word_number == 1:
        label = "TLM candidate"
    elif word_number == 2:
        label = "HOW candidate"
    else:
        label = ""
    if parity_unchecked:
        suffix = "parity unchecked: previous word failed"
        return f"{label}; {suffix}" if label else suffix
    return label


def _decode_word_sequence(
    values: list[int],
    preamble_index: int,
    inverted_flag: bool,
    confidences: Sequence[float] | np.ndarray | None = None,
    max_words: int = LNAV_SUBFRAME_WORDS,
) -> list[NavigationWord]:
    """Decode up to ten LNAV words from one preamble location."""

    words: list[NavigationWord] = []
    previous_accepted_word: list[int] | None = None

    for word_offset in range(max_words):
        word_start = preamble_index + word_offset * LNAV_WORD_BITS
        if word_start + LNAV_WORD_BITS > len(values):
            break

        raw_word = list(values[word_start : word_start + LNAV_WORD_BITS])
        word_confidences = _confidence_slice(confidences, word_start)
        parity_unchecked = word_offset > 0 and previous_accepted_word is None
        accepted_word = raw_word
        corrected = False
        corrected_bit_index = None
        parity_ok = False

        if not parity_unchecked:
            accepted_word, corrected, corrected_bit_index = maybe_correct_word(
                raw_word,
                previous_accepted_word,
                word_confidences,
            )
            parity_ok = check_lnav_word(accepted_word, previous_accepted_word)

        word = NavigationWord(
            start_bit=word_start,
            bits=bits_to_str(accepted_word),
            hex_word=f"0x{_bits_to_int(accepted_word):08X}",
            parity_ok=parity_ok,
            is_inverted=inverted_flag,
            label=_word_label(word_offset + 1, parity_unchecked),
            corrected=corrected,
            corrected_bit_index=corrected_bit_index,
            confidence=_word_confidence(word_confidences),
        )
        words.append(word)
        previous_accepted_word = accepted_word if parity_ok else None

    return words


def _decode_page(words: list[NavigationWord], subframe_id: int | None) -> tuple[int | None, str]:
    """Return a coarse page label for subframes 4/5 when word 3 is reliable."""

    if subframe_id not in (4, 5):
        return None, ""
    if len(words) < 3 or not words[1].parity_ok or not words[2].parity_ok:
        return None, "page not decoded yet"

    # In LNAV pages, word 3 carries Data ID and SV/page ID near its start.
    # This is intentionally only a raw page/SV ID hint, not an almanac decoder.
    word3_data = extract_data_bits(words[2], words[1])
    page_id = _bits_to_int(word3_data[2:8])
    return page_id, f"raw page/SV ID {page_id} (not fully decoded yet)"


def _payload_field_name(subframe_id: int | None, word_number: int) -> str:
    if subframe_id == 1:
        return f"Clock/health raw word {word_number}"
    if subframe_id in (2, 3):
        return f"Ephemeris raw word {word_number}"
    if subframe_id in (4, 5):
        return f"Almanac raw word {word_number}"
    return f"Raw word {word_number}"


def _build_fields(
    words: list[NavigationWord],
    subframe_id: int | None,
    tow_seconds: int | None,
    category: str,
    page_id: int | None,
    page_label: str,
) -> list[NavigationField]:
    """Build GUI-ready field rows without pretending to decode every ICD field."""

    if not words:
        return []

    start_bit = words[0].start_bit
    tlm = parse_tlm(words[0])
    how = parse_how(words[1], words[0]) if len(words) > 1 and words[0].parity_ok else None
    parity_ok_words = sum(1 for word in words if word.parity_ok)
    corrected_words = sum(1 for word in words if word.corrected)
    failed_words = sum(1 for word in words if not word.parity_ok)

    fields = [
        NavigationField(
            "Preamble",
            str(tlm["preamble"]),
            _bit_range(start_bit, 1, 1, 8),
            "Expected GPS LNAV TLM preamble is 10001011.",
        ),
        NavigationField(
            "TLM raw",
            f"{tlm['tlm_raw_hex']} ({tlm['tlm_raw_bits']})",
            _bit_range(start_bit, 1, 9, 24),
            "TLM message/integrity/reserved bits shown as raw data.",
        ),
        NavigationField(
            "HOW raw",
            f"{how['how_raw_hex']} ({how['how_raw_bits']})" if how else "raw / not fully decoded yet",
            _bit_range(start_bit, 2, 1, 24),
            "HOW data bits; decoded only when TLM and HOW parity are reliable.",
        ),
        NavigationField(
            "TOW seconds",
            str(tow_seconds) if tow_seconds is not None else "not decoded",
            _bit_range(start_bit, 2, 1, 17),
            "HOW TOW count multiplied by 6 seconds when plausible.",
        ),
        NavigationField(
            "Subframe ID",
            str(subframe_id) if subframe_id is not None else "not decoded",
            _bit_range(start_bit, 2, 20, 22),
            "HOW bits 20..22 identify the LNAV subframe.",
        ),
        NavigationField(
            "Category",
            category,
            "HOW subframe ID",
            "Coarse LNAV bucket for orientation, not a full payload decoder.",
        ),
        NavigationField(
            "Parity summary",
            f"{parity_ok_words}/{len(words)} words parity-valid",
            "W1-W10 D25-D30",
            "Corrected words count as valid after the accepted single-bit repair.",
        ),
        NavigationField(
            "Correction summary",
            f"{corrected_words} corrected, {failed_words} failed/unchecked",
            "W1-W10",
            "Only one unambiguous low-confidence bit is flipped per word.",
        ),
    ]

    if subframe_id == 1:
        fields.append(
            NavigationField(
                "Week number / clock parameters / health",
                "raw / not fully decoded yet",
                "W3-W10",
                "Placeholder for future ICD-level clock and health decoding.",
            )
        )
    elif subframe_id in (2, 3):
        fields.append(
            NavigationField(
                "Ephemeris raw payload",
                "raw / not fully decoded yet",
                "W3-W10",
                "Ephemeris words are grouped here but not numerically decoded yet.",
            )
        )
    elif subframe_id in (4, 5):
        fields.append(
            NavigationField(
                "Almanac raw payload",
                "raw / not fully decoded yet",
                "W3-W10",
                "Almanac/page words are grouped here without invented values.",
            )
        )
        fields.append(
            NavigationField(
                "Page ID",
                str(page_id) if page_id is not None else page_label or "page not decoded yet",
                "W3 D3-D8",
                "Coarse raw page/SV ID hint; full almanac page parsing is still TODO.",
            )
        )

    for word_number, word in enumerate(words[2:], start=3):
        fields.append(
            NavigationField(
                _payload_field_name(subframe_id, word_number),
                f"{word.hex_word} {word.bits}",
                _bit_range(start_bit, word_number, 1, 30),
                "Raw 30-bit LNAV word, including parity bits.",
            )
        )

    return fields


def _build_subframe_from_words(words: list[NavigationWord]) -> NavigationSubframe:
    """Build one coarse subframe model from ten word candidates."""

    how_reliable = len(words) > 1 and words[0].parity_ok and words[1].parity_ok
    how = parse_how(words[1], words[0]) if how_reliable else None
    subframe_id = int(how["subframe_id"]) if how and how["subframe_id"] is not None else None
    tow_seconds = int(how["tow_seconds"]) if how and how["tow_seconds"] is not None else None
    category = classify_subframe(subframe_id)
    page_id, page_label = _decode_page(words, subframe_id)
    parity_ok_words = sum(1 for word in words if word.parity_ok)
    corrected_words = sum(1 for word in words if word.corrected)
    fields = _build_fields(words, subframe_id, tow_seconds, category, page_id, page_label)
    return NavigationSubframe(
        start_bit=words[0].start_bit,
        subframe_id=subframe_id,
        tow_seconds=tow_seconds,
        words=words,
        fields=fields,
        page_id=page_id,
        page_label=page_label,
        category=category,
        parity_ok_words=parity_ok_words,
        corrected_words=corrected_words,
        valid=len(words) == LNAV_SUBFRAME_WORDS and parity_ok_words == len(words),
    )


def build_subframes(
    values: list[int],
    preamble_indices: Sequence[int],
    inverted_flag: bool = False,
    confidences: Sequence[float] | np.ndarray | None = None,
) -> list[NavigationSubframe]:
    """Group each preamble into one 10-word LNAV subframe when enough bits exist."""

    subframes: list[NavigationSubframe] = []
    for preamble_index in preamble_indices:
        if preamble_index + LNAV_SUBFRAME_BITS > len(values):
            continue
        words = _decode_word_sequence(values, int(preamble_index), inverted_flag, confidences)
        if len(words) == LNAV_SUBFRAME_WORDS:
            subframes.append(_build_subframe_from_words(words))
    return subframes


def decode_navigation_bits(bits: BitDecisionResult) -> NavigationDecodeResult:
    """Search for the LNAV preamble, align 30-bit words, and run parity checks."""

    as_list = bits.bit_values.astype(int).tolist()
    normal = bits_to_str(as_list)
    inverted_list = [1 - value for value in as_list]
    inverted = bits_to_str(inverted_list)

    results = NavigationDecodeResult()

    def scan(bit_string: str, values: list[int], inverted_flag: bool) -> None:
        for index in range(max(0, len(values) - len(PREAMBLE) + 1)):
            if bit_string[index : index + 8] != PREAMBLE:
                continue
            results.preamble_indices.append(index)
            words = _decode_word_sequence(values, index, inverted_flag, bits.confidences)
            results.words.extend(words)
            results.word_start_indices.extend(word.start_bit for word in words)
            results.parity_ok_count += sum(1 for word in words if word.parity_ok)
            results.corrected_word_count += sum(1 for word in words if word.corrected)
            results.failed_word_count += sum(1 for word in words if not word.parity_ok)

            subframe: NavigationSubframe | None = None
            if len(words) == LNAV_SUBFRAME_WORDS:
                subframe = _build_subframe_from_words(words)
                results.subframes.append(subframe)

            if words:
                polarity = "inverted" if inverted_flag else "normal"
                if subframe is not None:
                    subframe_label = subframe.subframe_id if subframe.subframe_id is not None else "?"
                    results.summary_lines.append(
                        f"Preamble at bit {index} ({polarity} polarity), captured subframe "
                        f"{subframe_label}: {subframe.category}."
                    )
                else:
                    results.summary_lines.append(
                        f"Preamble at bit {index} ({polarity} polarity), captured {len(words)} "
                        "word candidates; not enough bits for a full subframe."
                    )

    scan(normal, as_list, inverted_flag=False)
    scan(inverted, inverted_list, inverted_flag=True)

    if not results.summary_lines:
        results.summary_lines.append("No LNAV preamble detected in the current bit stream.")
    else:
        results.summary_lines.append(
            f"Structured view: {len(results.subframes)} subframes, "
            f"{results.corrected_word_count} corrected words, {results.failed_word_count} failed/unchecked words."
        )

    return results


def decode_navigation_from_tracking(
    tracking: TrackingState,
    bit_source: str = "auto",
    progress_callback=None,
    log_callback=None,
) -> tuple[BitDecisionResult, NavigationDecodeResult]:
    """Run bit extraction and LNAV framing with coarse progress updates."""

    if log_callback:
        log_callback(f"Decoding PRN {tracking.prn}: extracting 20 ms navigation bits.")
    if progress_callback:
        progress_callback(5)

    source = bit_source.lower().strip()
    prompt = tracking.iq_views.get("Integrated prompt")
    candidates: list[tuple[str, BitDecisionResult]] = []
    if source in ("auto", "carrier_aligned"):
        candidates.append(("carrier-aligned prompt", extract_navigation_bits(tracking)))
    if source in ("auto", "prompt_i"):
        candidates.append(("prompt I", form_navigation_bits(tracking.prompt_i)))
    if source in ("auto", "prompt_q"):
        candidates.append(("prompt Q", form_navigation_bits(tracking.prompt_q)))
    if source not in ("auto", "carrier_aligned", "prompt_i", "prompt_q"):
        raise ValueError(f"Unsupported bit source {bit_source!r}.")
    if prompt is not None and prompt.size == tracking.prompt_i.size and source == "auto":
        candidates.append(("carrier-aligned prompt from IQ view", form_navigation_bits(_carrier_aligned_prompt_ms(tracking))))
    if not candidates:
        raise ValueError("No bit source produced a candidate stream.")
    selected_label, bit_result = candidates[0]

    if log_callback:
        log_callback(
            f"Decoding PRN {tracking.prn}: scanning {bit_result.bit_values.size} hard decisions for LNAV preambles."
        )
    if progress_callback:
        progress_callback(55)

    nav_result = decode_navigation_bits(bit_result)
    for candidate_label, candidate in candidates[1:]:
        candidate_nav = decode_navigation_bits(candidate)
        if (
            candidate_nav.parity_ok_count,
            len(candidate_nav.preamble_indices),
        ) > (
            nav_result.parity_ok_count,
            len(nav_result.preamble_indices),
        ):
            selected_label = candidate_label
            bit_result = candidate
            nav_result = candidate_nav
    nav_result.summary_lines.insert(0, f"Bit source used: {selected_label}.")

    if progress_callback:
        progress_callback(100)
    if log_callback:
        log_callback(
            f"Decoding PRN {tracking.prn}: found {len(nav_result.preamble_indices)} preambles and "
            f"{nav_result.parity_ok_count} parity-valid words."
        )
    return bit_result, nav_result
