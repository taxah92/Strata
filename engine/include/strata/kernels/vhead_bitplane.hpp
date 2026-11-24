// include/strata/kernels/vhead_bitplane.hpp - Clean-room MIT exact 2-stage bit-plane LM head (Q8_0 & Q6_K).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Accelerates large-vocabulary (e.g., 248,320 x 2,560) greedy argmax and sampled (min_p / top_k / top_p)
// projection for T = 1..16 tokens on sm_70 (Volta mma.m8n8k4) and sm_80+ (Ampere/Ada/Blackwell INT8 Tensor Cores):
//   - Load-time lossless zero-duplication split of Q8_0 (8-bit -> hi4 + lo4 + d + per-row residual L2 norm rho)
//     and Q6_K (6-bit -> hi4 + lo2 + sc + d + per-row residual L2 norm rho), freeing the raw weight copy.
//   - Stage 1: streams ONLY the compact 4-bit hi_tiled plane + scales using Tensor Cores across all T tokens in
//     one pass, recording per-tile upper/lower bounds `tile_hi[t][tile]`, `tile_lo[t][tile]` and global lower bound
//     `g_max_lo[t] = max_r (approx(r, t) - rho[r] * ||x_t||_2)`.
//   - Stage 2 (greedy): scans `tile_hi[t][tile] >= g_max_lo[t]` and evaluates exact dot products from HI + LO ONLY on
//     surviving 16-row tiles, reducing the bitwise-exact greedy argmax.
//   - Stage 2 (sampled): scans `tile_hi[t][tile] >= thresh[t]` (where `thresh[t]` is `max_lo + log(min_p)` or the
//     k-th tile_lo lower bound) to collect <= max_cand exact (token_id, logit) pairs for fast candidate sampling,
//     with clean full-logits fallback if any token overflows.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

struct BitplaneHeadHandle;

/// True when the current CUDA device supports the exact bit-plane LM head (sm_70 or sm_80+, non-HIP).
bool vhead_bitplane_supported();

/// True when STRATA_VHEAD_BITPLANE is enabled (default ON on sm_70 when supported, opt-in =1 on sm_80+; =0 disables).
bool vhead_bitplane_enabled();

/// Build a lossless zero-duplication bit-plane view from a device Q8_0 (ggml_type == 8) or Q6_K (ggml_type == 14)
/// output.weight matrix of shape [n_out, n_in] (n_in must be a multiple of 256). Returns nullptr if unsupported or OOM.
/// Once built, the handle owns all HI+LO+scale planes and does not reference `d_weights` again.
BitplaneHeadHandle* vhead_bitplane_create(int ggml_type, const void* d_weights, int n_in, int n_out);

/// Free all device buffers owned by `h`. Safe on nullptr.
void vhead_bitplane_destroy(BitplaneHeadHandle* h);

/// Approximate VRAM bytes held by `h` (for diagnostics / accounting).
size_t vhead_bitplane_bytes(const BitplaneHeadHandle* h);

/// True when `vhead_bitplane_argmax(..., out_logits)` computes full exact FP32 logits in one pass (sm_70 Volta).
bool vhead_bitplane_exact_logits(const BitplaneHeadHandle* h);

/// Exact greedy argmax over all `n_out` vocabulary rows for `T` tokens (`1 <= T <= 16`).
/// `x_q8_1` holds `T` contiguous rows of `n_in / 32` `block_q8_1` structs (matching `native_quantize_q8_1`).
/// `t_offset` selects the starting slot in [0, 16) for Stage-1 cached bounds (default 0).
/// Writes exact argmax token ids to `out_tokens[0..T-1]` (lowest index wins exact ties, matching `argmax_rows`).
/// When `out_nonfinite != nullptr` (`nonfinite_stride > 0`), populates `out_nonfinite[t * nonfinite_stride + 0..stride-1]`
/// with non-zero if token `t` produced a non-finite norm or winning logit (0 otherwise), matching `logits_nonfinite_rows`.
/// When `out_logits != nullptr` and `vhead_bitplane_exact_logits(h)` is true, also writes `[T, n_out]` FP32 logits.
/// Returns true if launched cleanly; false if fallback to full `native_mmvq` + `argmax_rows` is needed.
bool vhead_bitplane_argmax(BitplaneHeadHandle* h, const void* x_q8_1, int T, int32_t* out_tokens, void* stream,
                           uint32_t* out_nonfinite = nullptr, int nonfinite_stride = 0, int t_offset = 0,
                           float* out_logits = nullptr);

/// Extract exact surviving candidate rows `(token_id, exact_logit)` for sampled decoding (`T` tokens, `1 <= T <= 16`).
/// When `reuse_stage1` is true, reuses Stage-1 `d_tile_hi`/`d_tile_lo`/`d_max_lo` already computed by
/// `vhead_bitplane_argmax` in slots `[t_offset .. t_offset + T)`.
/// When `k_need > 0` (`top_p < 1.0`, `min_p <= 0`, or penalties), computes a guaranteed lower bound on the `k_need`-th
/// largest logit via `d_tile_lo`; when `use_min_p_cut` is true, tightens with `d_max_lo + logf(min_p) - 1e-4f`.
bool vhead_bitplane_candidates(BitplaneHeadHandle* h, const void* x_q8_1, int T, int k_need, float min_p,
                               bool use_min_p_cut, bool reuse_stage1, void* stream,
                               const int32_t** out_ids, const float** out_logits,
                               const int32_t** out_counts, const uint8_t** out_dev_overflow,
                               const uint8_t** out_host_overflow, int* out_max_cand, int t_offset = 0);

/// Compute full `[T, n_out]` FP32 logits directly from bit-plane HI + LO + scales (fallback / diagnostics / logprobs).
bool vhead_bitplane_logits(const BitplaneHeadHandle* h, const void* x_q8_1, int T, float* out_logits, void* stream);

/// Reconstruct exact raw GGUF blocks (Q8_0 or Q6_K) for `n_rows` vocabulary indices `d_rows` into `d_dst`.
bool vhead_bitplane_gather_raw(const BitplaneHeadHandle* h, const int32_t* d_rows, int64_t n_rows, void* d_dst,
                               void* stream);

}  // namespace strata::kernels
