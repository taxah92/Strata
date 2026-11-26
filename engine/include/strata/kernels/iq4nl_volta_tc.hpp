// include/strata/kernels/iq4nl_volta_tc.hpp - Clean-room MIT Volta (sm_70) FP16 Tensor Core IQ4_NL down kernels.
//
// MIT License
// Copyright (c) 2026 Strata contributors
//
// Direct IQ4_NL (ggml_type == 20, 18 bytes per 32 elements) x Q8_1 Volta mma.sync.aligned.m8n8k4 kernels
// for MoE decode down projections in native_expert_grouped (0 extra VRAM, no ExpertCache repacking):
//   - Streams raw block_iq4_nl weights with coalesced 128-bit / paired 32-bit loads.
//   - Register-only 16-entry IQ4_NL LUT via __byte_perm (i8_table16 with pre-XOR 0x80 bias) -> exact FP16.
//   - Exact per-32-block integer dot products via mma.m8n8k4 + FP32 scale accumulation (rel_l2 ~1e-7 vs dp4a,
//     100% bitwise width-invariant across n_e = 1..16 entries per expert).
//   - Kill-switch: STRATA_IQ4_V70=0 (default ON on sm_70).
#pragma once

#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>

namespace strata::kernels {

/// True when the current CUDA device is Volta (sm_70, non-HIP).
bool iq4nl_volta_tc_supported();

/// Runtime mode from STRATA_IQ4_V70 (0 = disabled; >0 = enabled kernel variant; default on sm_70).
int iq4nl_volta_tc_mode();

/// Override runtime mode for parity/benchmarking (-1 restores env/default, 0 disables, >0 selects variant).
void iq4nl_volta_tc_set_mode(int mode);

/// Launch Volta sm_70 mma.m8n8k4 IQ4_NL down projection for native_expert_grouped.
/// Returns true if launched; false if caller should fall back to native_down_lds / native_down_multi.
bool iq4nl_volta_down_grouped(const NativeExpertLayout& L, int64_t gy,
                              const unsigned long long* grp_ptr, const int32_t* grp_start,
                              const int32_t* n_groups, const int32_t* ent_dst,
                              const void* hq_q8_1, float* out, void* stream);

}  // namespace strata::kernels
