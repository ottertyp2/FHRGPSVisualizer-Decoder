"""Tracking, bit extraction, and nav decode tests."""

from __future__ import annotations

from app.dsp.acquisition import AcquisitionConfig, acquire_signal
from app.dsp.bitsync import extract_navigation_bits
from app.dsp.demo import generate_demo_signal
from app.dsp.navdecode import decode_navigation_bits, decode_navigation_from_tracking
from app.dsp.tracking import track_signal
from app.models import SessionConfig


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
