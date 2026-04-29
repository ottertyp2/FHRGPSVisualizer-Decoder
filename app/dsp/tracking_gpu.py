"""Optional CuPy kernel for the GPS tracking correlator.

The CPU tracker stays in ``tracking.py``.  This module only contains the
compact GPU-specific E/P/L correlator used when the session selects a GPU
backend and CuPy is available.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.dsp.compute import get_cupy_module


TRACKING_CORRELATOR_CUDA = r"""
extern "C" __global__
void tracking_correlator(
    const float2* samples,
    const float* ca_code,
    const int sample_count,
    const double sample_rate,
    const double code_phase_chips,
    const double code_freq_hz,
    const double early_late_spacing_chips,
    const double carrier_phase_rad,
    const double carrier_freq_hz,
    double* out
) {
    extern __shared__ double shared[];
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    double* early_i = shared;
    double* early_q = early_i + threads;
    double* prompt_i = early_q + threads;
    double* prompt_q = prompt_i + threads;
    double* late_i = prompt_q + threads;
    double* late_q = late_i + threads;

    double early_i_sum = 0.0;
    double early_q_sum = 0.0;
    double prompt_i_sum = 0.0;
    double prompt_q_sum = 0.0;
    double late_i_sum = 0.0;
    double late_q_sum = 0.0;
    const double chip_step = code_freq_hz / sample_rate;
    const double phase_step = 6.2831853071795864769 * carrier_freq_hz / sample_rate;
    const double early_phase = code_phase_chips - 0.5 * early_late_spacing_chips;
    const double late_phase = code_phase_chips + 0.5 * early_late_spacing_chips;

    for (int index = tid; index < sample_count; index += threads) {
        const double carrier_angle = -(carrier_phase_rad + phase_step * (double)index);
        const float carrier_i = (float)cos(carrier_angle);
        const float carrier_q = (float)sin(carrier_angle);
        const float2 sample = samples[index];
        const double wiped_i = (double)sample.x * carrier_i - (double)sample.y * carrier_q;
        const double wiped_q = (double)sample.x * carrier_q + (double)sample.y * carrier_i;

        int early_index = (int)floor(early_phase + chip_step * (double)index);
        int prompt_index = (int)floor(code_phase_chips + chip_step * (double)index);
        int late_index = (int)floor(late_phase + chip_step * (double)index);
        early_index %= 1023;
        prompt_index %= 1023;
        late_index %= 1023;
        if (early_index < 0) early_index += 1023;
        if (prompt_index < 0) prompt_index += 1023;
        if (late_index < 0) late_index += 1023;

        const double early_code = (double)ca_code[early_index];
        const double prompt_code = (double)ca_code[prompt_index];
        const double late_code = (double)ca_code[late_index];
        early_i_sum += early_code * wiped_i;
        early_q_sum += early_code * wiped_q;
        prompt_i_sum += prompt_code * wiped_i;
        prompt_q_sum += prompt_code * wiped_q;
        late_i_sum += late_code * wiped_i;
        late_q_sum += late_code * wiped_q;
    }

    early_i[tid] = early_i_sum;
    early_q[tid] = early_q_sum;
    prompt_i[tid] = prompt_i_sum;
    prompt_q[tid] = prompt_q_sum;
    late_i[tid] = late_i_sum;
    late_q[tid] = late_q_sum;
    __syncthreads();

    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            early_i[tid] += early_i[tid + stride];
            early_q[tid] += early_q[tid + stride];
            prompt_i[tid] += prompt_i[tid + stride];
            prompt_q[tid] += prompt_q[tid + stride];
            late_i[tid] += late_i[tid + stride];
            late_q[tid] += late_q[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double scale = 1.0 / (double)sample_count;
        out[0] = early_i[0] * scale;
        out[1] = early_q[0] * scale;
        out[2] = prompt_i[0] * scale;
        out[3] = prompt_q[0] * scale;
        out[4] = late_i[0] * scale;
        out[5] = late_q[0] * scale;
    }
}
"""


@lru_cache(maxsize=1)
def get_tracking_correlator_kernel() -> Any:
    """Compile and return the optional CuPy tracking correlator kernel."""

    cupy = get_cupy_module()
    if cupy is None:
        raise RuntimeError("CuPy is not available for GPU tracking.")
    return cupy.RawKernel(TRACKING_CORRELATOR_CUDA, "tracking_correlator")
