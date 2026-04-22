"""Core dataclasses for session and processing state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SessionConfig:
    """User-selected session settings for offline analysis."""

    file_path: str | None = None
    sample_rate: float = 6_000_000.0
    center_frequency: float = 1_575_420_000.0
    is_baseband: bool = True
    if_frequency_hz: float = 0.0
    data_type: str = "complex64_le"
    start_sample: int = 0
    sample_count: int = 6_000_000
    prn: int = 1
    doppler_min: int = -6_000
    doppler_max: int = 6_000
    doppler_step: int = 250
    integration_ms: int = 80
    spread_acquisition_blocks: bool = False
    acquisition_segment_count: int = 8
    tracking_ms: int = 400
    early_late_spacing_chips: float = 0.5
    dll_gain: float = 0.08
    pll_gain: float = 10.0
    fll_gain: float = 0.15


@dataclass(slots=True)
class FileMetadata:
    """Description and preview statistics for an IQ file."""

    file_path: str
    file_name: str
    file_size_bytes: int
    data_type: str
    endianness: str
    sample_rate_hz: float
    total_samples: int
    estimated_duration_s: float
    preview_stats: dict[str, float] = field(default_factory=dict)
    preview_samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex64))


@dataclass(slots=True)
class AcquisitionCandidate:
    """One acquisition candidate peak."""

    prn: int
    doppler_hz: float
    carrier_frequency_hz: float
    code_phase_samples: int
    metric: float
    segment_start_sample: int = 0


@dataclass(slots=True)
class AcquisitionResult:
    """Full acquisition search output."""

    prn: int
    sample_rate_hz: float
    search_center_hz: float
    doppler_bins_hz: np.ndarray
    code_phases_samples: np.ndarray
    heatmap: np.ndarray
    best_candidate: AcquisitionCandidate
    candidates: list[AcquisitionCandidate] = field(default_factory=list)


@dataclass(slots=True)
class SearchCenterSweepEntry:
    """Best acquisition outcome for one searched IF / search-center hypothesis."""

    search_center_hz: float
    best_result: AcquisitionResult


@dataclass(slots=True)
class SearchCenterSweepResult:
    """Ranked results from sweeping multiple IF / search-center hypotheses."""

    entries: list[SearchCenterSweepEntry] = field(default_factory=list)


@dataclass(slots=True)
class TrackingState:
    """Time history from the simple tracking loop."""

    prn: int
    times_s: np.ndarray
    prompt_i: np.ndarray
    prompt_q: np.ndarray
    early_mag: np.ndarray
    prompt_mag: np.ndarray
    late_mag: np.ndarray
    code_error: np.ndarray
    carrier_error: np.ndarray
    doppler_est_hz: np.ndarray
    code_freq_est_hz: np.ndarray
    lock_metric: np.ndarray
    lock_detected: bool
    iq_views: dict[str, np.ndarray] = field(default_factory=dict)
    loop_states: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class BitDecisionResult:
    """Output of 20 ms navigation bit integration."""

    prompt_ms: np.ndarray
    best_offset_ms: int
    bit_sums: np.ndarray
    bit_values: np.ndarray
    confidences: np.ndarray
    bit_start_ms: np.ndarray


@dataclass(slots=True)
class NavigationWord:
    """A single 30-bit LNAV word."""

    start_bit: int
    bits: str
    hex_word: str
    parity_ok: bool
    is_inverted: bool
    label: str = ""


@dataclass(slots=True)
class NavigationDecodeResult:
    """Detected LNAV framing information."""

    preamble_indices: list[int] = field(default_factory=list)
    word_start_indices: list[int] = field(default_factory=list)
    words: list[NavigationWord] = field(default_factory=list)
    parity_ok_count: int = 0
    summary_lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkComponentResult:
    """One measured subsystem benchmark result."""

    name: str
    elapsed_s: float
    samples_processed: int
    bytes_processed: int
    throughput_samples_s: float
    throughput_mbytes_s: float
    realtime_factor_current: float
    realtime_factor_target: float
    estimated_time_for_10gb_s: float
    detail: str = ""


@dataclass(slots=True)
class BenchmarkResult:
    """Aggregate laptop suitability benchmark output."""

    target_sample_rate: float
    current_sample_rate: float
    components: list[BenchmarkComponentResult] = field(default_factory=list)
    bottleneck_name: str = ""
    suitability_summary: str = ""
    system_info: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DemoSignalResult:
    """Synthetic signal and reference values for demonstrations and tests."""

    samples: np.ndarray
    prn: int
    sample_rate: float
    doppler_hz: float
    code_phase_samples: int
    nav_bits: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
