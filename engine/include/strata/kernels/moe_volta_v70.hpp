// include/strata/kernels/moe_volta_v70.hpp - Clean-room MIT Volta (sm_70) MoE decode kernels (PR 5.3):
//   Step 5.3a (STRATA_EXP_BANK_V70): bank-aware split-plane IQ2_S / staged IQ3_XXS + PRMT register sign-masks
//                                    + int4 plane-staged q8_1 activations.
//   Step 5.3b (STRATA_MOE_PERSIST_V70): persistent weight-major fused Gate/Up + SwiGLU/Q8_1 + Down kernel
//                                       with lock-free epoch-tagged release/acquire ready protocol.
#pragma once

#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>

namespace strata::kernels {

// True if the current CUDA device is Volta sm_70 (compute capability 7.0).
bool moe_volta_v70_supported();

// Effective bank-aware gate/up mode: 0 = off (kill-switch STRATA_EXP_BANK_V70=0), 1 = on (default on sm_70).
int moe_volta_bank_mode();
void moe_volta_bank_set_mode(int mode);

// Effective persistent MoE mode: 0 = off (kill-switch STRATA_MOE_PERSIST_V70=0), 1 = on (default on sm_70).
int moe_volta_persist_mode();
void moe_volta_persist_set_mode(int mode);

// Step 5.3a: Fused Gate/Up + SwiGLU + Q8_1 quantization (IQ2_S=22, IQ3_XXS=18, n_embd=2560, n_ff%32==0).
// Bit-for-bit identical to native_gu_fused_kernel / expdec_gu_kernel + swiglu_q8_1_entries_kernel.
bool moe_volta_gu_fused(const NativeExpertLayout& L, int64_t gy,
                        const unsigned long long* grp_ptr, const int32_t* grp_start,
                        const int32_t* n_groups, const int32_t* ent_tok,
                        const void* x_q8_1, void* hq_q8_1, void* stream);

// Step 5.3a: Unfused Gate/Up writing FP32 gate & up (for Hadamard-folded down or expdec_bench).
// Bit-for-bit identical to expdec_gu_kernel.
bool moe_volta_gu_unfused(const NativeExpertLayout& L, int64_t gy,
                          const unsigned long long* grp_ptr, const int32_t* grp_start,
                          const int32_t* n_groups, const int32_t* ent_tok,
                          const void* x_q8_1, float* gate, float* up, void* stream);

// Step 5.3b: Persistent weight-major fused Gate/Up + SwiGLU/Q8_1 + Down (IQ2_S/IQ3_XXS x IQ4_NL/Q2_0,
// n_embd=2560, n_ff=640). Uses a strictly resident grid sized via cudaOccupancyMaxActiveBlocksPerMultiprocessor,
// priority Gate/Up draining + epoch-tagged release/acquire ready flags (zero volatile, zero cudaMemsetAsync,
// CUDA-Graph safe, deadlock-free). Bit-for-bit identical to Step 5.3a + PR 5.2 IQ4_NL / native Q2_0 down.
bool moe_volta_persist_grouped(const NativeExpertLayout& L, int64_t cap_groups,
                               const unsigned long long* grp_ptr, const int32_t* grp_start,
                               const int32_t* n_groups, const int32_t* ent_dst,
                               const int32_t* ent_tok, const void* x_q8_1,
                               void* hq_q8_1, float* out, void* stream);

}  // namespace strata::kernels
