// Arithmetic adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp and ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_ple_postops.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int N = strata::core::kCompiledGeometry.n_embd, H = strata::core::kCompiledGeometry.hc, D = N * H, HISTORY = 9;
static_assert(N == strata::core::kCompiledGeometry.n_embd && H == strata::core::kCompiledGeometry.hc, "native_ple_postops is compiled for core/geometry.hpp");
__device__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
__global__ void gate_kernel(const float* key, const float* query, float* gate, float scale) {
    // SUM_ROWS selects 512 threads for four rows on the target GPU. Preserve
    // its eight partial lanes and materialized MUL rounding (never a dot FMA).
    float sums[8] = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = threadIdx.x + j * 512;
        const float p = d < N ? __fmul_rn(key[blockIdx.x * N + d], query[blockIdx.x * N + d]) : 0.0f;
        sums[j] += p;
    }
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += sums[j];
    __shared__ float partials[32];
    sum = warp_sum(sum);
    const int lane = threadIdx.x % 32;
    if (!lane) partials[threadIdx.x / 32] = sum;
    __syncthreads();
    sum = lane < 16 ? partials[lane] : 0.0f;
    sum = warp_sum(sum);
    if (threadIdx.x == 0) {
        const float s = __fmaf_rn(scale, sum, 0.0f); // ggml SCALE's zero bias
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float sign = float((s > 0.0f) - (s < 0.0f));
        gate[blockIdx.x] = 1.0f / (1.0f + expf(-__fmul_rn(sign, mag)));
    }
}
__global__ void broadcast_kernel(const float* value, const float* gate, float* gated) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) gated[i] = __fmul_rn(value[i % N], gate[i / N]);
}
__global__ void conv_residual_kernel(const float* history, const float* normalized,
                                    const uint16_t* weights, const float* hidden,
                                    const float* gated, float* conv, float* result) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];
        const float w = __half2float(__ushort_as_half(weights[c * 4 + k]));
        const float term = __fmul_rn(x, w);
        sum = k == 0 ? term : __fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + expf(-sum));
    conv[c] = activation;
    // Exact hidden/result alias is safe: each thread owns one element.
    result[c] = __fadd_rn(hidden[c], __fadd_rn(gated[c], activation));
}
// ---- the batch: T tokens, the same arithmetic per element as the kernels above
// weighted_rms_norm (native_gr_norm.cu) with the gamma row repeating every H rows (one token's H groups)
__device__ float norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}
__global__ void rms_rep_kernel(const float* __restrict__ input, const float* __restrict__ gamma,
                               float* __restrict__ output) {
    constexpr int BlockSize = 1024;             // native_gr_rms_norm_weighted's choice for 2560 columns
    const int tid = threadIdx.x;
    const size_t row_offset = size_t(blockIdx.x) * N;
    input += row_offset;
    output += row_offset;
    gamma += size_t(blockIdx.x % H) * N;
    float partial = 0.0f;
    for (int col = tid; col < N; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }
    __shared__ float sums[32];
    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = 0.0f;
    if (lane < BlockSize / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);
    const float mean = partial / N;
    const float scale = rsqrtf(mean + NG_RMS_EPS);
    for (int col = tid; col < N; col += BlockSize) output[col] = scale * input[col] * gamma[col];
}
// rms_rep_kernel with the column count as a RUN-TIME divisor, as weighted_rms_norm<1024> (native_gr_rms_norm_weighted,
// the per-token path) divides: under --use_fast_math a division by the constant N may become a multiplication by its
// rounded reciprocal, which is not bitwise the run-time division.
__global__ void rms_rep_rt_kernel(const float* __restrict__ input, const float* __restrict__ gamma,
                                  float* __restrict__ output, int n_cols, float epsilon) {
    constexpr int BlockSize = 1024;
    const int tid = threadIdx.x;
    const size_t row_offset = size_t(blockIdx.x) * n_cols;
    input += row_offset;
    output += row_offset;
    gamma += size_t(blockIdx.x % H) * n_cols;
    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }
    __shared__ float sums[32];
    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = 0.0f;
    if (lane < BlockSize / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (int col = tid; col < n_cols; col += BlockSize) output[col] = scale * input[col] * gamma[col];
}
__global__ void broadcast_batch_kernel(const float* value, const float* gate, float* gated, int T) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(T) * D) return;
    const size_t t = i / D, d = i % D;
    gated[i] = __fmul_rn(value[t * N + d % N], gate[t * H + d / N]);
}
// the dilated conv (taps 9, 6, 3 tokens back and this one) and the residual; a tap before the chunk reads the history
__global__ void conv_residual_batch_kernel(const float* history, const float* normalized, const uint16_t* weights,
                                           float* hidden, const float* gated, int T) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(T) * D) return;
    const int t = int(i / D), c = int(i % D);
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = t - 9 + 3 * k;             // the token this tap reads (k == 3: this one)
        const float x = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (9 + p)];
        const float wk = __half2float(__ushort_as_half(weights[c * 4 + k]));
        const float term = __fmul_rn(x, wk);
        sum = k == 0 ? term : __fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + expf(-sum));
    hidden[i] = __fadd_rn(hidden[i], __fadd_rn(gated[i], activation));
}
// the history after the chunk: the last nine normalized rows (older ones from the history when T < 9)
__global__ void history_batch_kernel(float* history, const float* normalized, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float h[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) {
        const int p = T - HISTORY + r;
        h[r] = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (T + r)];
    }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) history[size_t(c) * HISTORY + r] = h[r];
}

// history_batch_kernel that also writes the history after each of the T tokens (`snap`, T x 9 x 10240): pure copies,
// the values T history advances (ple_history_advance) leave
__global__ void history_batch_snap_kernel(float* history, const float* normalized, int T, float* snap) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float h0[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) h0[r] = history[size_t(c) * HISTORY + r];
    for (int t = 0; t < T; ++t) {
        float* out = snap + size_t(t) * HISTORY * D + size_t(c) * HISTORY;
#pragma unroll
        for (int r = 0; r < HISTORY; ++r) {
            const int p = r + t + 1 - HISTORY;      // after t + 1 advances row r holds token p, or an older row
            out[r] = p >= 0 ? normalized[size_t(p) * D + c] : h0[r + t + 1];
        }
    }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) {
        const int p = r + T - HISTORY;
        history[size_t(c) * HISTORY + r] = p >= 0 ? normalized[size_t(p) * D + c] : h0[r + T];
    }
}


// PR 6.2b: Tree PLE dilated conv + history snapshot (bitwise identical to single-token PLE along each root-to-node path)
__global__ void conv_residual_tree_kernel(const float* __restrict__ history, const float* __restrict__ normalized,
                                          const uint16_t* __restrict__ weights, float* __restrict__ hidden,
                                          const float* __restrict__ gated, int T, const int32_t* __restrict__ ple_tap) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(T) * D) return;
    const int t = int(i / D), c = int(i % D);
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int src = (k < 3) ? ple_tap[t * 3 + k] : t;
        const float x = src >= 0 ? normalized[size_t(src) * D + c] : history[size_t(c) * HISTORY + (9 + src)];
        const float wk = __half2float(__ushort_as_half(weights[c * 4 + k]));
        const float term = __fmul_rn(x, wk);
        sum = k == 0 ? term : __fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + expf(-sum));
    hidden[i] = __fadd_rn(hidden[i], __fadd_rn(gated[i], activation));
}

__global__ void history_tree_snap_kernel(const float* __restrict__ history, const float* __restrict__ normalized, int T,
                                         const int32_t* __restrict__ ple_hist_src, float* __restrict__ snap) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float h0[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) h0[r] = history[size_t(c) * HISTORY + r];
    for (int t = 0; t < T; ++t) {
        float* out = snap + size_t(t) * HISTORY * D + size_t(c) * HISTORY;
#pragma unroll
        for (int r = 0; r < HISTORY; ++r) {
            const int src = ple_hist_src[t * HISTORY + r];
            out[r] = src >= 0 ? normalized[size_t(src) * D + c] : h0[HISTORY + src];
        }
    }
}

// ---- PR 6.3b: Fused Volta sm_70 PLE postops (3 kernels instead of 7, ~2.5x less HBM2 traffic, 0 extra VRAM,
// 100% bitwise identical to rms_rep[_rt] + gate + broadcast + rms_rep[_rt] + conv_residual + history).
// Kill-switch: STRATA_PLE_FUSED_V70=0 restores the original 7-kernel sequence.
static int g_ple_fused_v70 = -1;
static bool ple_fused_v70_enabled() {
    if (g_ple_fused_v70 >= 0) return g_ple_fused_v70 != 0;
    static const bool env_on = [] {
        const char* e = std::getenv("STRATA_PLE_FUSED_V70");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return env_on;
}

template <bool RtDiv>
__global__ void __launch_bounds__(512, 2)
ple_norm_gate_v70_kernel(const float* __restrict__ key, const float* __restrict__ hidden,
                         const float* __restrict__ gamma_k, const float* __restrict__ gamma_q,
                         float* __restrict__ gate, float scale_gate, int n_cols, float eps) {
    static_assert(N == 2560, "ple_norm_gate_v70_kernel assumes N == 2560 (5 * 512)");
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const size_t row_off = (size_t) blockIdx.x * N;
    const size_t g_off = (size_t) (blockIdx.x % H) * N;
    const float* kp = key + row_off;
    const float* hp = hidden + row_off;
    const float* gkp = gamma_k + g_off;
    const float* gqp = gamma_q + g_off;

    float kv[5], hv[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int col = tid + j * 512;
        kv[j] = kp[col];
        hv[j] = hp[col];
    }

    float k_even = 0.0f, k_odd = 0.0f, h_even = 0.0f, h_odd = 0.0f;
    k_even += kv[0] * kv[0]; k_even += kv[2] * kv[2]; k_even += kv[4] * kv[4];
    k_odd  += kv[1] * kv[1]; k_odd  += kv[3] * kv[3];
    h_even += hv[0] * hv[0]; h_even += hv[2] * hv[2]; h_even += hv[4] * hv[4];
    h_odd  += hv[1] * hv[1]; h_odd  += hv[3] * hv[3];

    __shared__ float sk[32], sh[32];
    k_even = norm_warp_sum(k_even); k_odd = norm_warp_sum(k_odd);
    h_even = norm_warp_sum(h_even); h_odd = norm_warp_sum(h_odd);
    if (lane == 0) {
        sk[warp] = k_even; sk[warp + 16] = k_odd;
        sh[warp] = h_even; sh[warp + 16] = h_odd;
    }
    __syncthreads();

    const float k_part = norm_warp_sum(sk[lane]);
    const float h_part = norm_warp_sum(sh[lane]);
    const float k_mean = RtDiv ? (k_part / n_cols) : (k_part / N);
    const float h_mean = RtDiv ? (h_part / n_cols) : (h_part / N);
    const float k_scale = rsqrtf(k_mean + (RtDiv ? eps : NG_RMS_EPS));
    const float h_scale = rsqrtf(h_mean + (RtDiv ? eps : NG_RMS_EPS));

    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int col = tid + j * 512;
        const float nk = k_scale * kv[j] * gkp[col];
        const float nq = h_scale * hv[j] * gqp[col];
        sum += __fmul_rn(nk, nq);
    }
    __shared__ float sg[16];
    sum = warp_sum(sum);
    if (lane == 0) sg[warp] = sum;
    __syncthreads();
    sum = lane < 16 ? sg[lane] : 0.0f;
    sum = warp_sum(sum);
    if (tid == 0) {
        const float s = __fmaf_rn(scale_gate, sum, 0.0f);
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float sign = float((s > 0.0f) - (s < 0.0f));
        gate[blockIdx.x] = 1.0f / (1.0f + expf(-__fmul_rn(sign, mag)));
    }
}

template <bool RtDiv>
__global__ void __launch_bounds__(512, 2)
ple_bcast_norm_v70_kernel(const float* __restrict__ value, const float* __restrict__ gate,
                          const float* __restrict__ gamma_conv, float* __restrict__ normalized,
                          int n_cols, float eps) {
    static_assert(N == 2560, "ple_bcast_norm_v70_kernel assumes N == 2560 (5 * 512)");
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int t = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const float g = gate[blockIdx.x];
    const float* vp = value + (size_t) t * N;
    const float* gp = gamma_conv + (size_t) h * N;
    float* out = normalized + (size_t) blockIdx.x * N;

    float gt[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) gt[j] = __fmul_rn(vp[tid + j * 512], g);

    float pe = 0.0f, po = 0.0f;
    pe += gt[0] * gt[0]; pe += gt[2] * gt[2]; pe += gt[4] * gt[4];
    po += gt[1] * gt[1]; po += gt[3] * gt[3];
    __shared__ float s[32];
    pe = norm_warp_sum(pe); po = norm_warp_sum(po);
    if (lane == 0) { s[warp] = pe; s[warp + 16] = po; }
    __syncthreads();
    const float part = norm_warp_sum(s[lane]);
    const float mean = RtDiv ? (part / n_cols) : (part / N);
    const float scale = rsqrtf(mean + (RtDiv ? eps : NG_RMS_EPS));

#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int col = tid + j * 512;
        out[col] = scale * gt[j] * gp[col];
    }
}

template <bool WriteSnap>
__global__ void __launch_bounds__(128, 4)
ple_conv_add_hist_v70_kernel(float* __restrict__ history, const float* __restrict__ normalized,
                             const uint16_t* __restrict__ weights, float* __restrict__ hidden,
                             const float* __restrict__ value, const float* __restrict__ gate,
                             int T, float* __restrict__ snap) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    const int h_idx = c / N;
    const int d_idx = c - h_idx * N;

    const uint2 wraw = *reinterpret_cast<const uint2*>(weights + (size_t) c * 4);
    const float w0 = __half2float(__ushort_as_half((uint16_t) (wraw.x & 0xFFFFu)));
    const float w1 = __half2float(__ushort_as_half((uint16_t) (wraw.x >> 16)));
    const float w2 = __half2float(__ushort_as_half((uint16_t) (wraw.y & 0xFFFFu)));
    const float w3 = __half2float(__ushort_as_half((uint16_t) (wraw.y >> 16)));

    float* hcol = history + (size_t) c * HISTORY;
    float h[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) h[r] = hcol[r];

    for (int t = 0; t < T; ++t) {
        const float xn = normalized[(size_t) t * D + c];
        float sum = __fmul_rn(h[0], w0);
        sum = __fadd_rn(sum, __fmul_rn(h[3], w1));
        sum = __fadd_rn(sum, __fmul_rn(h[6], w2));
        sum = __fadd_rn(sum, __fmul_rn(xn,   w3));
        const float act = sum / (1.0f + expf(-sum));
        const float gv = __fmul_rn(value[(size_t) t * N + d_idx], gate[(size_t) t * H + h_idx]);
        const size_t idx = (size_t) t * D + c;
        hidden[idx] = __fadd_rn(hidden[idx], __fadd_rn(gv, act));

#pragma unroll
        for (int r = 0; r < HISTORY - 1; ++r) h[r] = h[r + 1];
        h[HISTORY - 1] = xn;

        if constexpr (WriteSnap) {
            float* out = snap + (size_t) t * HISTORY * D + (size_t) c * HISTORY;
#pragma unroll
            for (int r = 0; r < HISTORY; ++r) out[r] = h[r];
        }
    }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) hcol[r] = h[r];
}

__global__ void __launch_bounds__(128, 4)
ple_conv_add_tree_v70_kernel(const float* __restrict__ history, const float* __restrict__ normalized,
                             const uint16_t* __restrict__ weights, float* __restrict__ hidden,
                             const float* __restrict__ value, const float* __restrict__ gate,
                             int T, const int32_t* __restrict__ ple_tap,
                             const int32_t* __restrict__ ple_hist_src, float* __restrict__ snap) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    const int h_idx = c / N;
    const int d_idx = c - h_idx * N;

    const uint2 wraw = *reinterpret_cast<const uint2*>(weights + (size_t) c * 4);
    const float w0 = __half2float(__ushort_as_half((uint16_t) (wraw.x & 0xFFFFu)));
    const float w1 = __half2float(__ushort_as_half((uint16_t) (wraw.x >> 16)));
    const float w2 = __half2float(__ushort_as_half((uint16_t) (wraw.y & 0xFFFFu)));
    const float w3 = __half2float(__ushort_as_half((uint16_t) (wraw.y >> 16)));

    float h0[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) h0[r] = history[(size_t) c * HISTORY + r];

    for (int t = 0; t < T; ++t) {
        const int s0 = ple_tap[t * 3 + 0];
        const int s1 = ple_tap[t * 3 + 1];
        const int s2 = ple_tap[t * 3 + 2];
        const float x0 = s0 >= 0 ? normalized[(size_t) s0 * D + c] : h0[HISTORY + s0];
        const float x1 = s1 >= 0 ? normalized[(size_t) s1 * D + c] : h0[HISTORY + s1];
        const float x2 = s2 >= 0 ? normalized[(size_t) s2 * D + c] : h0[HISTORY + s2];
        const float x3 = normalized[(size_t) t * D + c];
        float sum = __fmul_rn(x0, w0);
        sum = __fadd_rn(sum, __fmul_rn(x1, w1));
        sum = __fadd_rn(sum, __fmul_rn(x2, w2));
        sum = __fadd_rn(sum, __fmul_rn(x3, w3));
        const float act = sum / (1.0f + expf(-sum));
        const float gv = __fmul_rn(value[(size_t) t * N + d_idx], gate[(size_t) t * H + h_idx]);
        const size_t idx = (size_t) t * D + c;
        hidden[idx] = __fadd_rn(hidden[idx], __fadd_rn(gv, act));

        float* out = snap + (size_t) t * HISTORY * D + (size_t) c * HISTORY;
#pragma unroll
        for (int r = 0; r < HISTORY; ++r) {
            const int src = ple_hist_src[t * HISTORY + r];
            out[r] = src >= 0 ? normalized[(size_t) src * D + c] : h0[HISTORY + src];
        }
    }
}

__global__ void __launch_bounds__(128, 4)
ple_conv_add_prefill_v70_kernel(const float* __restrict__ history, const float* __restrict__ normalized,
                                const uint16_t* __restrict__ weights, float* __restrict__ hidden,
                                const float* __restrict__ value, const float* __restrict__ gate, int T) {
    constexpr int BT = 16;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    const int t0 = blockIdx.y * BT;
    const int t1 = (t0 + BT < T) ? (t0 + BT) : T;
    const int h_idx = c / N;
    const int d_idx = c - h_idx * N;

    const uint2 wraw = *reinterpret_cast<const uint2*>(weights + (size_t) c * 4);
    const float w0 = __half2float(__ushort_as_half((uint16_t) (wraw.x & 0xFFFFu)));
    const float w1 = __half2float(__ushort_as_half((uint16_t) (wraw.x >> 16)));
    const float w2 = __half2float(__ushort_as_half((uint16_t) (wraw.y & 0xFFFFu)));
    const float w3 = __half2float(__ushort_as_half((uint16_t) (wraw.y >> 16)));

    float ring[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) {
        const int p = t0 - HISTORY + r;
        ring[r] = p >= 0 ? normalized[(size_t) p * D + c] : history[(size_t) c * HISTORY + (HISTORY + p)];
    }

    for (int t = t0; t < t1; ++t) {
        const float xn = normalized[(size_t) t * D + c];
        float sum = __fmul_rn(ring[0], w0);
        sum = __fadd_rn(sum, __fmul_rn(ring[3], w1));
        sum = __fadd_rn(sum, __fmul_rn(ring[6], w2));
        sum = __fadd_rn(sum, __fmul_rn(xn,      w3));
        const float act = sum / (1.0f + expf(-sum));
        const float gv = __fmul_rn(value[(size_t) t * N + d_idx], gate[(size_t) t * H + h_idx]);
        const size_t idx = (size_t) t * D + c;
        hidden[idx] = __fadd_rn(hidden[idx], __fadd_rn(gv, act));
#pragma unroll
        for (int r = 0; r < HISTORY - 1; ++r) ring[r] = ring[r + 1];
        ring[HISTORY - 1] = xn;
    }
}

struct Span { const void* p; size_t bytes; size_t alignment; };
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("native PLE postops launch: ") + cudaGetErrorString(error));
}
} // namespace

void native_ple_set_fused_v70(bool enabled) { g_ple_fused_v70 = enabled ? 1 : 0; }
bool native_ple_fused_v70_enabled() { return ple_fused_v70_enabled(); }

void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key,D*4,4}, {hidden,D*4,4}, {value,N*4,4},
        {history,HISTORY*D*4,4}, {w.norm_key,D*4,4}, {w.norm_query,D*4,4},
        {w.norm_conv,D*4,4}, {w.conv1d_f16,4*D*2,2}};
    const Span outputs[] = {{b.key,D*4,4}, {b.query,D*4,4}, {b.gate,H*4,4},
        {b.gated,D*4,4}, {b.normalized,D*4,4}, {b.conv,D*4,4}, {b.result,D*4,4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    native_gr_rms_norm_weighted(projected_key,w.norm_key,b.key,N,H,NG_RMS_EPS,stream);
    native_gr_rms_norm_weighted(hidden,w.norm_query,b.query,N,H,NG_RMS_EPS,stream);
    auto st = static_cast<cudaStream_t>(stream);
    gate_kernel<<<H,512,0,st>>>(b.key,b.query,b.gate,1.0f / std::sqrt(float(N)));
    broadcast_kernel<<<D/256,256,0,st>>>(value,b.gate,b.gated);
    launch_check();
    native_gr_rms_norm_weighted(b.gated,w.norm_conv,b.normalized,N,H,NG_RMS_EPS,stream);
    conv_residual_kernel<<<D/256,256,0,st>>>(history,b.normalized,w.conv1d_f16,hidden,b.gated,b.conv,b.result);
    launch_check();
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int T, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");
    auto st = static_cast<cudaStream_t>(stream);
    const unsigned rows = unsigned(T) * H;
    if (ple_fused_v70_enabled()) {
        const float inv_sqrt_n = 1.0f / std::sqrt(float(N));
        ple_norm_gate_v70_kernel<false><<<rows, 512, 0, st>>>(key, hidden, w.norm_key, w.norm_query, gate, inv_sqrt_n, N, NG_RMS_EPS);
        ple_bcast_norm_v70_kernel<false><<<rows, 512, 0, st>>>(value, gate, w.norm_conv, query_norm, N, NG_RMS_EPS);
        if (T <= 64) {
            ple_conv_add_hist_v70_kernel<false><<<D / 128, 128, 0, st>>>(history, query_norm, w.conv1d_f16, hidden, value, gate, T, nullptr);
        } else {
            const dim3 grid(unsigned(D / 128), unsigned((T + 15) / 16));
            ple_conv_add_prefill_v70_kernel<<<grid, 128, 0, st>>>(history, query_norm, w.conv1d_f16, hidden, value, gate, T);
            history_batch_kernel<<<D / 256, 256, 0, st>>>(history, query_norm, T);
        }
        launch_check();
        return;
    }
    const unsigned blocks = unsigned((size_t(T) * D + 255) / 256);
    rms_rep_kernel<<<rows, 1024, 0, st>>>(key, w.norm_key, key);
    rms_rep_kernel<<<rows, 1024, 0, st>>>(hidden, w.norm_query, query_norm);
    gate_kernel<<<rows, 512, 0, st>>>(key, query_norm, gate, 1.0f / std::sqrt(float(N)));
    broadcast_batch_kernel<<<blocks, 256, 0, st>>>(value, gate, gated, T);
    rms_rep_kernel<<<rows, 1024, 0, st>>>(gated, w.norm_conv, query_norm);
    conv_residual_batch_kernel<<<blocks, 256, 0, st>>>(history, query_norm, w.conv1d_f16, hidden, gated, T);
    history_batch_kernel<<<D / 256, 256, 0, st>>>(history, query_norm, T);
    launch_check();
}

void native_ple_postops_batch_snap(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                                   float* query_norm, float* gated, float* gate, int T, float* snap, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate || !snap)
        throw std::invalid_argument("native PLE postops batch (snapshots): null input or empty batch");
    auto st = static_cast<cudaStream_t>(stream);
    const unsigned rows = unsigned(T) * H;
    if (ple_fused_v70_enabled()) {
        const float inv_sqrt_n = 1.0f / std::sqrt(float(N));
        ple_norm_gate_v70_kernel<true><<<rows, 512, 0, st>>>(key, hidden, w.norm_key, w.norm_query, gate, inv_sqrt_n, N, NG_RMS_EPS);
        ple_bcast_norm_v70_kernel<true><<<rows, 512, 0, st>>>(value, gate, w.norm_conv, query_norm, N, NG_RMS_EPS);
        ple_conv_add_hist_v70_kernel<true><<<D / 128, 128, 0, st>>>(history, query_norm, w.conv1d_f16, hidden, value, gate, T, snap);
        launch_check();
        return;
    }
    const unsigned blocks = unsigned((size_t(T) * D + 255) / 256);
    rms_rep_rt_kernel<<<rows, 1024, 0, st>>>(key, w.norm_key, key, N, NG_RMS_EPS);
    rms_rep_rt_kernel<<<rows, 1024, 0, st>>>(hidden, w.norm_query, query_norm, N, NG_RMS_EPS);
    gate_kernel<<<rows, 512, 0, st>>>(key, query_norm, gate, 1.0f / std::sqrt(float(N)));
    broadcast_batch_kernel<<<blocks, 256, 0, st>>>(value, gate, gated, T);
    rms_rep_rt_kernel<<<rows, 1024, 0, st>>>(gated, w.norm_conv, query_norm, N, NG_RMS_EPS);
    conv_residual_batch_kernel<<<blocks, 256, 0, st>>>(history, query_norm, w.conv1d_f16, hidden, gated, T);
    history_batch_snap_kernel<<<D / 256, 256, 0, st>>>(history, query_norm, T, snap);
    launch_check();
}

void native_ple_postops_tree_snap(float* key, float* hidden, const float* value, const float* history,
                                  const PleWeights& w, float* query_norm, float* gated, float* gate, int T,
                                  const int32_t* ple_tap, const int32_t* ple_hist_src, float* snap, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate || !snap ||
        !ple_tap || !ple_hist_src)
        throw std::invalid_argument("native PLE postops tree: null input or empty batch");
    auto st = static_cast<cudaStream_t>(stream);
    const unsigned rows = unsigned(T) * H;
    if (ple_fused_v70_enabled()) {
        const float inv_sqrt_n = 1.0f / std::sqrt(float(N));
        ple_norm_gate_v70_kernel<true><<<rows, 512, 0, st>>>(key, hidden, w.norm_key, w.norm_query, gate, inv_sqrt_n, N, NG_RMS_EPS);
        ple_bcast_norm_v70_kernel<true><<<rows, 512, 0, st>>>(value, gate, w.norm_conv, query_norm, N, NG_RMS_EPS);
        ple_conv_add_tree_v70_kernel<<<D / 128, 128, 0, st>>>(history, query_norm, w.conv1d_f16, hidden, value, gate, T, ple_tap, ple_hist_src, snap);
        launch_check();
        return;
    }
    const unsigned blocks = unsigned((size_t(T) * D + 255) / 256);
    rms_rep_rt_kernel<<<rows, 1024, 0, st>>>(key, w.norm_key, key, N, NG_RMS_EPS);
    rms_rep_rt_kernel<<<rows, 1024, 0, st>>>(hidden, w.norm_query, query_norm, N, NG_RMS_EPS);
    gate_kernel<<<rows, 512, 0, st>>>(key, query_norm, gate, 1.0f / std::sqrt(float(N)));
    broadcast_batch_kernel<<<blocks, 256, 0, st>>>(value, gate, gated, T);
    rms_rep_rt_kernel<<<rows, 1024, 0, st>>>(gated, w.norm_conv, query_norm, N, NG_RMS_EPS);
    conv_residual_tree_kernel<<<blocks, 256, 0, st>>>(history, query_norm, w.conv1d_f16, hidden, gated, T, ple_tap);
    history_tree_snap_kernel<<<D / 256, 256, 0, st>>>(history, query_norm, T, ple_hist_src, snap);
    launch_check();
}

} // namespace strata::kernels
