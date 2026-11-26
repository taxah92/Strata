// include/strata/kernels/q8_volta_tc.hpp - Clean-room MIT Volta (sm_70) FP16 Tensor Core Q8_0 kernels.
//
// MIT License
// Copyright (c) 2026 Strata contributors
//
// Direct Q8_0 (ggml_type == 8, 34 bytes per 32 elements) Volta mma.sync.aligned.m8n8k4 kernels:
//   1. Prefill Fused W8A16 GEMM (`q8_volta_prefill_gemm`): streams raw Q8_0 weights directly from HBM2,
//      dequantizes in registers/smem with coalesced loads, and executes FP16 Tensor Core MMA with FP32
//      accumulation — eliminating the global FP16 weight write + read round-trip of dequant_f16 + cuBLAS.
//      Kill-switch: STRATA_PF_Q8_V70=0 (default ON on sm_70).
//   2. Decode Single & Segmented Q8_0 x Q8_1 TC GEMM (`q8_volta_decode_mmvq`, `q8_volta_decode_seg`):
//      handles M = 1..16 tokens in a single pass over raw Q8_0 weights using coalesced 128/64/32-bit loads,
//      exact int8->FP16 conversion, exact integer dot products per 32-block via mma.m8n8k4, and identical
//      reduction order across M = 1..16 (bitwise equal to M = 1 for every column).
//      Kill-switch: STRATA_DEC_SEG_V70=0 (default 1 on sm_70: single + segmented; 2 = single only).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// True when the current CUDA device is Volta (sm_70, non-HIP).
bool q8_volta_tc_supported();

/// True when STRATA_PF_Q8_V70 is enabled for prefill Q8_0 GEMM (default ON on sm_70; =0 disables).
bool q8_volta_prefill_enabled();
bool q8_volta_prefill_should_use(int64_t T, int64_t N, int64_t K);

/// True when STRATA_DEC_SEG_V70 is enabled for single-matrix decode Q8_0 TC GEMM (default ON on sm_70; =0 disables).
bool q8_volta_decode_enabled();

/// True when STRATA_DEC_SEG_V70 enables multi-matrix segmented decode fusion (default ON on sm_70; =0 or =2 disables).
bool q8_volta_decode_seg_enabled();

/// Prefill Fused Q8_0 W8A16 GEMM:
///   Y[T, N] (fp32, row stride ldy; ldy <= 0 means N) = beta * Y + X_f16[T, K] * W_q8_0[N, K]^T
/// Requires contiguous X_f16 (ldx == K), K % 64 == 0, T >= 32, N >= 1, 0 extra VRAM.
/// Returns true if launched; false if caller should fall back to dequant_f16 + cuBLAS.
bool q8_volta_prefill_gemm(const uint16_t* X_f16, const void* W_q8_0, float* Y,
                           int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                           void* stream);

/// Decode single-matrix Q8_0 x Q8_1 TC GEMM (ncols in [1..16]):
///   Y[ncols, n_out] (column j at y + j * n_out) = X_q8_1[ncols, n_in/32] * W_q8_0[n_out, n_in]^T
/// Requires n_in % 64 == 0, n_out >= 1, 1 <= ncols <= 16.
/// Every column j is bitwise equal to a single-column (ncols == 1) call on column j.
/// Returns true if launched; false if caller should fall back to native_q8_0_mmvq.
bool q8_volta_decode_mmvq(const void* W_q8_0, const void* x_q8_1, float* y,
                          int n_in, int n_out, int ncols, void* stream);

struct Q8VoltaSeg {
    const void* w;
    float* y;
    int n_out;
};

/// Decode segmented multi-matrix Q8_0 x Q8_1 TC GEMM (1..4 matrices sharing the same x_q8_1, ncols in [1..16]):
/// Computes Y_s[ncols, n_out_s] = X_q8_1 * W_s^T for s = 0..n_segs-1 in a single launch.
/// Returns true if launched; false if caller should fall back to individual native_mmvq calls.
bool q8_volta_decode_seg(const Q8VoltaSeg* segs, int n_segs, const void* x_q8_1,
                         int n_in, int ncols, void* stream);

}  // namespace strata::kernels
