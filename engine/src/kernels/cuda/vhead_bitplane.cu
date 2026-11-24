// src/kernels/cuda/vhead_bitplane.cu - Clean-room MIT exact 2-stage bit-plane LM head (Q8_0 & Q6_K).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Derived strictly from NVIDIA PTX ISA mma.sync.aligned.m8n8k4 (sm_70) / m16n8k32 / m16n8k16 (sm_80+) and
// Cauchy-Schwarz interval bounds on exact bit-plane decompositions:
//   - Q8_0 (int8 w in [-128..127]): w = 16 * hi4 + 8 + lo_c, hi4 = w >> 4 in [-8..7], lo_c = (w & 15) - 8 in [-8..7].
//   - Q6_K (int6 q in [-32..31]):   q =  4 * hi4 + 2 + lo_c, hi4 = q >> 2 in [-8..7], lo_c = (q &  3) - 2 in [-2..1].
//
// Zero-duplication 2-stage tile-bound architecture (frees raw output.weight on load):
//   - Stage 1 streams ONLY the coalesced 4-bit hi_tiled plane + scales using Tensor Cores across all T = 1..16 tokens,
//     computing exact per-tile upper/lower bounds `tile_hi[t][tile]`, `tile_lo[t][tile]` and global lower bound
//     `g_max_lo[t] = max_r (approx(r, t) - rho[r] * ||x_t||_2)`.
//   - Stage 2 (greedy) scans `tile_hi[t][tile] >= g_max_lo[t]` and evaluates exact dot products from HI + LO ONLY on
//     surviving 16-row tiles, reducing exact argmax via 64-bit atomicMax (lowest row index wins exact ties).
//   - Stage 2 (sampled) computes `thresh[t]` (either `g_max_lo[t] + log(min_p)` or k-th largest `tile_lo[t]` via a
//     2-pass 256-bucket histogram) and collects <= kMaxCand surviving `(row, exact_logit)` pairs from HI + LO.
#include "strata/kernels/vhead_bitplane.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

struct BitplaneHeadHandle {
    int type = 0;            // 8 = Q8_0, 14 = Q6_K
    int cc_mode = 0;         // 70 = sm_70 Volta, 80 = sm_80+ Ampere+
    int n_in = 0;
    int n_out = 0;
    int n_tiles = 0;         // (n_out + 15) / 16
    int n_tile_pairs = 0;    // (n_tiles + 1) / 2 (for sm_70 32-row tile pairs)
    int nb32 = 0;            // n_in / 32
    int n_scales = 0;        // Q8_0: n_in / 32; Q6_K: n_in / 16
    int nsb = 0;             // Q6_K: n_in / 256

    void* hi_tiled = nullptr;
    void* lo_plane = nullptr;      // Q6_K: uint32_t[n_out * n_scales]; Q8_0: uint4[n_out * nb32]
    uint16_t* d_tiled = nullptr;   // Q8_0 (70/80+) or Q6_K (80+): half scales
    int8_t* sc_tiled = nullptr;    // Q6_K (70): int8_t[n_tile_pairs * n_scales * 32] (or raw on 80+)
    uint16_t* d_all_tiled = nullptr; // Q6_K (70): uint16_t[n_tile_pairs * nsb * 32] (or raw on 80+)
    float* rho = nullptr;          // float[n_out] exact residual L2 norm * (1 + safety)

    static constexpr int kMaxT = 16;
    static constexpr int kMaxCand = 4096;

    float* d_x_norm = nullptr;     // [16]
    float* d_x_sum = nullptr;      // [16][n_scales]
    int32_t* d_max_lo = nullptr;   // [16] ordered-int32 for atomicMax
    float* d_tile_hi = nullptr;    // [16][n_tiles]
    float* d_tile_lo = nullptr;    // [16][n_tiles]
    float* d_thresh = nullptr;     // [16]
    unsigned long long* d_best = nullptr; // [16] packed (ordered_u32(val) << 32) | (~row)

    // Candidate buffers (lazy-allocated on first sampled call)
    int32_t* d_cand_ids = nullptr;      // [16][kMaxCand]
    float* d_cand_logits = nullptr;     // [16][kMaxCand]
    int32_t* d_cand_counts = nullptr;   // [16]
    uint8_t* d_cand_overflow = nullptr; // [16]
    uint8_t* h_cand_overflow = nullptr; // [16] pinned mapped host

    size_t total_bytes = 0;
};

namespace {

constexpr int kTypeQ8_0 = 8;
constexpr int kTypeQ6_K = 14;
constexpr float kBoundSafety = 1.0005f;

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

#if !defined(__HIPCC__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define STRATA_VHEAD_SM80 1
#else
#define STRATA_VHEAD_SM80 0
#endif

#if !defined(__HIPCC__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700)
#define STRATA_VHEAD_SM70 1
#else
#define STRATA_VHEAD_SM70 0
#endif

__device__ __forceinline__ int32_t float_to_ordered(float f) {
    const int32_t i = __float_as_int(f);
    return i >= 0 ? i : (i ^ 0x7fffffff);
}

__device__ __forceinline__ float ordered_to_float(int32_t o) {
    return __int_as_float(o >= 0 ? o : (o ^ 0x7fffffff));
}

__device__ __forceinline__ uint32_t float_to_u32_ordered(float f) {
    return (uint32_t) float_to_ordered(f) ^ 0x80000000u;
}

__device__ __forceinline__ float u32_ordered_to_float(uint32_t u) {
    return ordered_to_float((int32_t) (u ^ 0x80000000u));
}

__device__ __forceinline__ unsigned long long pack_val_row(float val, int32_t row) {
    return ((unsigned long long) float_to_u32_ordered(val) << 32) | (unsigned long long) (0xFFFFFFFFu - (uint32_t) row);
}

__device__ __forceinline__ void atomic_max_float(int32_t* addr, float val) {
    atomicMax(addr, float_to_ordered(val));
}

__device__ __forceinline__ uint32_t unpack_s4x4(uint16_t p) {
    const uint32_t v = (uint32_t) (p & 0x000Fu) |
                       ((uint32_t) (p & 0x00F0u) << 4) |
                       ((uint32_t) (p & 0x0F00u) << 8) |
                       ((uint32_t) (p & 0xF000u) << 12);
    return v | ((v & 0x08080808u) * 0x1Eu);
}

// Reconstruct 4 packed signed int8 Q6_K weights (hi4 << 2) | lo2 into one int32 for __dp4a.
__device__ __forceinline__ int pack_q6k_4(uint16_t hi16, uint8_t lo8) {
    const uint32_t h4 = (unpack_s4x4(hi16) << 2) & 0xFCFCFCFCu;
    const uint32_t l4 = ((uint32_t) (lo8 & 0x03u)) |
                        (((uint32_t) (lo8 & 0x0Cu)) << 6) |
                        (((uint32_t) (lo8 & 0x30u)) << 12) |
                        (((uint32_t) (lo8 & 0xC0u)) << 18);
    return (int) (h4 | l4);
}

// Reconstruct 4 unsigned 6-bit Q6_K weights [0..63] (= v + 32) from sm_70 packed (lo4, hi2) planes.
__device__ __forceinline__ uint32_t unpack_q6k_v70_u4(uint2 lo4, uint32_t hi2, int st) {
    if (st == 0) return (lo4.x & 0x0F0F0F0Fu) | ((hi2 << 4) & 0x30303030u);
    if (st == 1) return ((lo4.x >> 4) & 0x0F0F0F0Fu) | ((hi2 << 2) & 0x30303030u);
    if (st == 2) return (lo4.y & 0x0F0F0F0Fu) | (hi2 & 0x30303030u);
    return ((lo4.y >> 4) & 0x0F0F0F0Fu) | ((hi2 >> 2) & 0x30303030u);
}

__device__ __forceinline__ int pack_q6k_v70_s8_4(uint2 lo4, uint32_t hi2, int st) {
    const uint32_t q = unpack_q6k_v70_u4(lo4, hi2, st);
    const uint32_t neg_mask = ((~q) & 0x20202020u) >> 5;
    return (int) ((q & 0x1F1F1F1Fu) | (neg_mask * 0xE0u));
}

// Reconstruct 4 packed signed int8 Q8_0 weights (hi4 << 4) | lo4 into one int32 for __dp4a.
__device__ __forceinline__ int pack_q8_4(uint16_t hi16, uint16_t lo16) {
    const uint32_t h4 = ((uint32_t) (hi16 & 0x000Fu)) |
                        (((uint32_t) (hi16 & 0x00F0u)) << 4) |
                        (((uint32_t) (hi16 & 0x0F00u)) << 8) |
                        (((uint32_t) (hi16 & 0xF000u)) << 12);
    const uint32_t l4 = ((uint32_t) (lo16 & 0x000Fu)) |
                        (((uint32_t) (lo16 & 0x00F0u)) << 4) |
                        (((uint32_t) (lo16 & 0x0F00u)) << 8) |
                        (((uint32_t) (lo16 & 0xF000u)) << 12);
    return (int) ((h4 << 4) | l4);
}

__device__ __forceinline__ int lane_for_r32_v70(int r32) {
    // Inverse of r_in_tp(lane) = ((lane >> 2) & 3) * 8 + (lane & 3) + 4 * (lane >> 4)
    return ((r32 & 24) >> 1) | ((r32 & 4) << 2) | (r32 & 3);
}

__device__ __forceinline__ int decode_q6k_elem(const uint8_t* blk, int sub, int j) {
    const int n = sub >> 3;
    const int rem = sub & 7;
    const int l = (rem & 1) * 16 + j;
    const int qid = rem >> 1;
    const uint8_t* ql_n = blk + 64 * n;
    const uint8_t* qh_n = blk + 128 + 32 * n;
    uint8_t low4 = 0;
    if (qid == 0) low4 = ql_n[l] & 0x0Fu;
    else if (qid == 1) low4 = ql_n[l + 32] & 0x0Fu;
    else if (qid == 2) low4 = ql_n[l] >> 4;
    else low4 = ql_n[l + 32] >> 4;
    const uint8_t hi2 = (qh_n[l] >> (2 * qid)) & 0x03u;
    return (int) (low4 | (hi2 << 4)) - 32;
}

__device__ __forceinline__ uint16_t pack_q8_hi4(const int8_t* qs, int k0) {
    const uint16_t h0 = (uint16_t) ((qs[k0 + 0] >> 4) & 0xF);
    const uint16_t h1 = (uint16_t) ((qs[k0 + 1] >> 4) & 0xF);
    const uint16_t h2 = (uint16_t) ((qs[k0 + 2] >> 4) & 0xF);
    const uint16_t h3 = (uint16_t) ((qs[k0 + 3] >> 4) & 0xF);
    return (uint16_t) (h0 | (h1 << 4) | (h2 << 8) | (h3 << 12));
}

__device__ __forceinline__ uint16_t pack_q8_lo4(const int8_t* qs, int k0) {
    const uint16_t l0 = (uint16_t) (qs[k0 + 0] & 0xF);
    const uint16_t l1 = (uint16_t) (qs[k0 + 1] & 0xF);
    const uint16_t l2 = (uint16_t) (qs[k0 + 2] & 0xF);
    const uint16_t l3 = (uint16_t) (qs[k0 + 3] & 0xF);
    return (uint16_t) (l0 | (l1 << 4) | (l2 << 8) | (l3 << 12));
}

#if !defined(__HIPCC__)
// Exact int8x4 -> two half2 registers (sm_70 cooperative smem pack).
__device__ __forceinline__ uint2 vhead_i8h_cvt4(uint32_t w) {
    const uint32_t c = w ^ 0x80808080u;
    const uint32_t lo = __byte_perm(c, 0x64646464u, 0x5140);
    const uint32_t hi = __byte_perm(c, 0x64646464u, 0x7362);
    const __half2 m = __floats2half2_rn(1152.0f, 1152.0f);
    __half2 hl, hh;
    std::memcpy(&hl, &lo, 4);
    std::memcpy(&hh, &hi, 4);
    hl = __hsub2(hl, m);
    hh = __hsub2(hh, m);
    uint2 out;
    std::memcpy(&out.x, &hl, 4);
    std::memcpy(&out.y, &hh, 4);
    return out;
}

// Unsigned 6-bit x4 [0..63] (= v + 32) -> signed two half2 registers [-32..31], zero sign extension.
__device__ __forceinline__ uint2 vhead_u6h_cvt4(uint32_t q) {
    const uint32_t c = q + 0x60606060u;
    const uint32_t lo = __byte_perm(c, 0x64646464u, 0x5140);
    const uint32_t hi = __byte_perm(c, 0x64646464u, 0x7362);
    const __half2 m = __floats2half2_rn(1152.0f, 1152.0f);
    __half2 hl, hh;
    std::memcpy(&hl, &lo, 4);
    std::memcpy(&hh, &hi, 4);
    hl = __hsub2(hl, m);
    hh = __hsub2(hh, m);
    uint2 out;
    std::memcpy(&out.x, &hl, 4);
    std::memcpy(&out.y, &hh, 4);
    return out;
}
#endif

#if STRATA_VHEAD_SM70
__device__ __forceinline__ void mma_f16_m8n8k4(float (&d)[8], uint2 a, uint2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
#else
    (void) d; (void) a; (void) b;
    __trap();
#endif
}
#endif

#if STRATA_VHEAD_SM80
__device__ __forceinline__ void mma_s8_m16n8k32(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8_m16n8k16(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}
#endif

// ---- Load-time lossless tiled repack kernels (sm_70 Volta, zero duplication) ----

__global__ void repack_q8_0_v70_kernel(const uint8_t* __restrict__ src, uint4* __restrict__ hi_tiled,
                                       uint4* __restrict__ lo_tiled, uint16_t* __restrict__ d_tiled,
                                       float* __restrict__ rho, int n_in, int n_out) {
    const int tp = (int) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int n_tiles = (n_out + 15) >> 4;
    const int n_tp = (n_tiles + 1) >> 1;
    if (tp >= n_tp) return;
    const int lane = threadIdx.x & 31;
    const int q = (lane >> 2) & 3;
    const int g = lane & 3;
    const int h = lane >> 4;
    const int r_in_tp = q * 8 + g + 4 * h;
    const int row = tp * 32 + r_in_tp;
    const bool valid = row < n_out;
    const int nb = n_in >> 5;
    const uint8_t* rsrc = valid ? (src + (size_t) row * (size_t) nb * 34u) : nullptr;

    (void) rho;
    for (int b = 0; b < nb; ++b) {
        uint4 qs0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 qs1 = make_uint4(0u, 0u, 0u, 0u);
        uint16_t d_bits = 0;
        if (valid) {
            const uint8_t* blk = rsrc + (size_t) b * 34u;
            d_bits = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
            const uint16_t* q16 = reinterpret_cast<const uint16_t*>(blk + 2);
            auto rd32 = [&](int idx) -> uint32_t {
                return (uint32_t) q16[2 * idx] | ((uint32_t) q16[2 * idx + 1] << 16);
            };
            qs0 = make_uint4(rd32(0), rd32(1), rd32(2), rd32(3));
            qs1 = make_uint4(rd32(4), rd32(5), rd32(6), rd32(7));
        }
        hi_tiled[((size_t) tp * nb + b) * 32u + lane] = qs0;
        lo_tiled[((size_t) tp * nb + b) * 32u + lane] = qs1;
        d_tiled[((size_t) tp * nb + b) * 32u + lane] = d_bits;
    }
}

__global__ void repack_q6_k_v70_kernel(const uint8_t* __restrict__ src, uint2* __restrict__ hi_tiled,
                                       uint32_t* __restrict__ lo_tiled, int8_t* __restrict__ sc_tiled,
                                       uint16_t* __restrict__ d_all_tiled, float* __restrict__ rho,
                                       int n_in, int n_out) {
    const int tp = (int) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int n_tiles = (n_out + 15) >> 4;
    const int n_tp = (n_tiles + 1) >> 1;
    if (tp >= n_tp) return;
    const int lane = threadIdx.x & 31;
    const int q = (lane >> 2) & 3;
    const int g = lane & 3;
    const int h = lane >> 4;
    const int r_in_tp = q * 8 + g + 4 * h;
    const int row = tp * 32 + r_in_tp;
    const bool valid = row < n_out;
    const int nsb = n_in >> 8;
    const int n_scales = n_in >> 4;
    const uint8_t* rsrc = valid ? (src + (size_t) row * (size_t) nsb * 210u) : nullptr;
    (void) rho;

    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = valid ? (rsrc + (size_t) sb * 210u) : nullptr;
        const uint16_t d_bits = valid ? ((uint16_t) blk[208] | ((uint16_t) blk[209] << 8)) : 0;
        d_all_tiled[((size_t) tp * nsb + sb) * 32u + lane] = d_bits;

        for (int sub = 0; sub < 16; ++sub) {
            const int gsub = sb * 16 + sub;
            int8_t sc = 0;
            uint2 lo4_pk = make_uint2(0u, 0u);
            uint32_t hi2_w = 0u;
            if (valid) {
                sc = reinterpret_cast<const int8_t*>(blk + 192)[sub];
                uint32_t lo4_w[2] = {0u, 0u};
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint32_t q_u = (uint32_t) (decode_q6k_elem(blk, sub, j) + 32);
                    const uint32_t l4 = q_u & 0x0Fu;
                    const uint32_t h2 = (q_u >> 4) & 0x03u;
                    const int grp = j >> 2;
                    const int b = j & 3;
                    lo4_w[j >> 3] |= l4 << (8 * b + 4 * (grp & 1));
                    hi2_w |= h2 << (8 * b + 2 * grp);
                }
                lo4_pk = make_uint2(lo4_w[0], lo4_w[1]);
            }
            hi_tiled[((size_t) tp * n_scales + gsub) * 32u + lane] = lo4_pk;
            lo_tiled[((size_t) tp * n_scales + gsub) * 32u + lane] = hi2_w;
            sc_tiled[((size_t) tp * n_scales + gsub) * 32u + lane] = sc;
        }
    }
}

// ---- Load-time lossless tiled repack kernels (sm_80+, zero duplication) ----

__global__ void repack_q8_0_sm80_kernel(const uint8_t* __restrict__ src, uint2* __restrict__ hi_tiled,
                                        uint4* __restrict__ lo_row, uint16_t* __restrict__ d_tiled,
                                        float* __restrict__ rho, int n_in, int n_out) {
    const int tile = (int) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int n_tiles = (n_out + 15) >> 4;
    if (tile >= n_tiles) return;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int nb = n_in >> 5;
    const int r0 = tile * 16 + g;
    const int r1 = r0 + 8;
    const bool v0 = r0 < n_out;
    const bool v1 = r1 < n_out;

    const uint8_t* src0 = src + (size_t) (v0 ? r0 : 0) * (size_t) nb * 34u;
    const uint8_t* src1 = src + (size_t) (v1 ? r1 : 0) * (size_t) nb * 34u;

    double err_sq0 = 0.0, err_sq1 = 0.0;
    for (int b = 0; b < nb; ++b) {
        uint16_t p00 = 0, p01 = 0, p10 = 0, p11 = 0;
        if (v0) {
            const uint8_t* blk = src0 + (size_t) b * 34u;
            const uint16_t d_bits = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            p00 = pack_q8_hi4(qs, tig * 4);
            p01 = pack_q8_hi4(qs, 16 + tig * 4);
            const uint16_t l0 = pack_q8_lo4(qs, tig * 4);
            const uint16_t l1 = pack_q8_lo4(qs, 16 + tig * 4);
            uint16_t* dst_lo = reinterpret_cast<uint16_t*>(lo_row + (size_t) r0 * nb + b);
            dst_lo[tig] = l0;
            dst_lo[4 + tig] = l1;
            if (tig == 0) d_tiled[((size_t) tile * nb + b) * 16 + g] = d_bits;
            const float d = __half2float(__ushort_as_half(d_bits));
            int lsq = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ((int) qs[tig * 4 + j] & 15) - 8;
                const int c1 = ((int) qs[16 + tig * 4 + j] & 15) - 8;
                lsq += c0 * c0 + c1 * c1;
            }
            err_sq0 += (double) d * (double) d * (double) lsq;
        } else if (tig == 0) {
            d_tiled[((size_t) tile * nb + b) * 16 + g] = 0;
        }
        if (v1) {
            const uint8_t* blk = src1 + (size_t) b * 34u;
            const uint16_t d_bits = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            p10 = pack_q8_hi4(qs, tig * 4);
            p11 = pack_q8_hi4(qs, 16 + tig * 4);
            const uint16_t l0 = pack_q8_lo4(qs, tig * 4);
            const uint16_t l1 = pack_q8_lo4(qs, 16 + tig * 4);
            uint16_t* dst_lo = reinterpret_cast<uint16_t*>(lo_row + (size_t) r1 * nb + b);
            dst_lo[tig] = l0;
            dst_lo[4 + tig] = l1;
            if (tig == 0) d_tiled[((size_t) tile * nb + b) * 16 + 8 + g] = d_bits;
            const float d = __half2float(__ushort_as_half(d_bits));
            int lsq = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ((int) qs[tig * 4 + j] & 15) - 8;
                const int c1 = ((int) qs[16 + tig * 4 + j] & 15) - 8;
                lsq += c0 * c0 + c1 * c1;
            }
            err_sq1 += (double) d * (double) d * (double) lsq;
        } else if (tig == 0) {
            d_tiled[((size_t) tile * nb + b) * 16 + 8 + g] = 0;
        }
        hi_tiled[((size_t) tile * nb + b) * 32 + lane] =
            make_uint2((uint32_t) p00 | ((uint32_t) p10 << 16), (uint32_t) p01 | ((uint32_t) p11 << 16));
    }

    err_sq0 += __shfl_xor_sync(0xffffffffu, err_sq0, 1);
    err_sq0 += __shfl_xor_sync(0xffffffffu, err_sq0, 2);
    err_sq1 += __shfl_xor_sync(0xffffffffu, err_sq1, 1);
    err_sq1 += __shfl_xor_sync(0xffffffffu, err_sq1, 2);
    if (tig == 0) {
        if (v0) rho[r0] = (float) sqrt(err_sq0) * kBoundSafety;
        if (v1) rho[r1] = (float) sqrt(err_sq1) * kBoundSafety;
    }
}

__global__ void repack_q6_k_sm80_kernel(const uint8_t* __restrict__ src, uint32_t* __restrict__ hi_tiled,
                                        uint32_t* __restrict__ lo_row, uint16_t* __restrict__ d_tiled,
                                        int8_t* __restrict__ sc_raw, uint16_t* __restrict__ d_all_raw,
                                        float* __restrict__ rho, int n_in, int n_out) {
    const int tile = (int) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int n_tiles = (n_out + 15) >> 4;
    if (tile >= n_tiles) return;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int nsb = n_in >> 8;
    const int n_scales = n_in >> 4;
    const int r0 = tile * 16 + g;
    const int r1 = r0 + 8;
    const bool v0 = r0 < n_out;
    const bool v1 = r1 < n_out;

    const uint8_t* src0 = src + (size_t) (v0 ? r0 : 0) * (size_t) nsb * 210u;
    const uint8_t* src1 = src + (size_t) (v1 ? r1 : 0) * (size_t) nsb * 210u;

    double err_sq0 = 0.0, err_sq1 = 0.0;
    for (int gsub = 0; gsub < n_scales; ++gsub) {
        const int sb = gsub >> 4;
        const int sub = gsub & 15;
        uint16_t p0 = 0, p1 = 0;
        uint8_t l0 = 0, l1 = 0;

        if (v0) {
            const uint8_t* blk = src0 + (size_t) sb * 210u;
            const int8_t sc = reinterpret_cast<const int8_t*>(blk + 192)[sub];
            const uint16_t d_bits = (uint16_t) blk[208] | ((uint16_t) blk[209] << 8);
            const float d_all = __half2float(__ushort_as_half(d_bits));
            const float d_sub = d_all * (float) sc;
            const __half d_h = __float2half_rn(d_sub);
            const float d_q = __half2float(d_h);
            if (tig == 0) {
                d_tiled[((size_t) tile * n_scales + gsub) * 16 + g] = __half_as_ushort(d_h);
                sc_raw[(size_t) r0 * n_scales + gsub] = sc;
                if (sub == 0) d_all_raw[(size_t) r0 * nsb + sb] = d_bits;
            }
            const double d_err = fabs((double) d_sub - (double) d_q);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int v = decode_q6k_elem(blk, sub, tig * 4 + j);
                p0 |= (uint16_t) (((v >> 2) & 0xF) << (4 * j));
                l0 |= (uint8_t) ((v & 3) << (2 * j));
                const int c = (v & 3) - 2;
                const double e = (double) d_q * (double) c + d_err * (double) abs(v);
                err_sq0 += e * e;
            }
            reinterpret_cast<uint8_t*>(lo_row + (size_t) r0 * n_scales + gsub)[tig] = l0;
        } else if (tig == 0) {
            d_tiled[((size_t) tile * n_scales + gsub) * 16 + g] = 0;
        }

        if (v1) {
            const uint8_t* blk = src1 + (size_t) sb * 210u;
            const int8_t sc = reinterpret_cast<const int8_t*>(blk + 192)[sub];
            const uint16_t d_bits = (uint16_t) blk[208] | ((uint16_t) blk[209] << 8);
            const float d_all = __half2float(__ushort_as_half(d_bits));
            const float d_sub = d_all * (float) sc;
            const __half d_h = __float2half_rn(d_sub);
            const float d_q = __half2float(d_h);
            if (tig == 0) {
                d_tiled[((size_t) tile * n_scales + gsub) * 16 + 8 + g] = __half_as_ushort(d_h);
                sc_raw[(size_t) r1 * n_scales + gsub] = sc;
                if (sub == 0) d_all_raw[(size_t) r1 * nsb + sb] = d_bits;
            }
            const double d_err = fabs((double) d_sub - (double) d_q);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int v = decode_q6k_elem(blk, sub, tig * 4 + j);
                p1 |= (uint16_t) (((v >> 2) & 0xF) << (4 * j));
                l1 |= (uint8_t) ((v & 3) << (2 * j));
                const int c = (v & 3) - 2;
                const double e = (double) d_q * (double) c + d_err * (double) abs(v);
                err_sq1 += e * e;
            }
            reinterpret_cast<uint8_t*>(lo_row + (size_t) r1 * n_scales + gsub)[tig] = l1;
        } else if (tig == 0) {
            d_tiled[((size_t) tile * n_scales + gsub) * 16 + 8 + g] = 0;
        }

        hi_tiled[((size_t) tile * n_scales + gsub) * 32 + lane] = (uint32_t) p0 | ((uint32_t) p1 << 16);
    }

    err_sq0 += __shfl_xor_sync(0xffffffffu, err_sq0, 1);
    err_sq0 += __shfl_xor_sync(0xffffffffu, err_sq0, 2);
    err_sq1 += __shfl_xor_sync(0xffffffffu, err_sq1, 1);
    err_sq1 += __shfl_xor_sync(0xffffffffu, err_sq1, 2);
    if (tig == 0) {
        if (v0) rho[r0] = (float) sqrt(err_sq0) * kBoundSafety;
        if (v1) rho[r1] = (float) sqrt(err_sq1) * kBoundSafety;
    }
}

// ---- Per-step prep kernel ----
template <bool IS_Q6K>
__global__ void vhead_prep_kernel(const Q81Block* __restrict__ xq, int n_in, int T, int t_offset,
                                  float* __restrict__ x_norm, float* __restrict__ x_sum,
                                  int32_t* __restrict__ d_max_lo, unsigned long long* __restrict__ d_best) {
    const int t = (int) blockIdx.x;
    if (t >= T) return;
    const int ts = t_offset + t;
    const int tid = (int) threadIdx.x;
    const int nb32 = n_in >> 5;
    const Q81Block* xt = xq + (size_t) t * (size_t) nb32;
    const int n_scales = IS_Q6K ? (n_in >> 4) : nb32;

    if (tid == 0) {
        d_max_lo[ts] = float_to_ordered(-1e30f);
        d_best[ts] = 0ull;
    }

    double sq = 0.0;
    for (int b = tid; b < nb32; b += blockDim.x) {
        const float dx = __low2float(xt[b].ds);
        const int8_t* q = xt[b].qs;
        int s_lo = 0, s_hi = 0, q_sq = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int v = (int) q[j];
            s_lo += v;
            q_sq += v * v;
        }
#pragma unroll
        for (int j = 16; j < 32; ++j) {
            const int v = (int) q[j];
            s_hi += v;
            q_sq += v * v;
        }
        if constexpr (IS_Q6K) {
            x_sum[(size_t) ts * n_scales + 2 * b + 0] = (float) s_lo * dx;
            x_sum[(size_t) ts * n_scales + 2 * b + 1] = (float) s_hi * dx;
        } else {
            x_sum[(size_t) ts * n_scales + b] = (float) (s_lo + s_hi) * dx;
        }
        sq += (double) dx * (double) dx * (double) q_sq;
    }
    __shared__ double s_red[8];
    const int lane = tid & 31, warp = tid >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if (lane == 0) s_red[warp] = sq;
    __syncthreads();
    if (warp == 0) {
        double v = lane < (blockDim.x >> 5) ? s_red[lane] : 0.0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) x_norm[ts] = (float) sqrt(v);
    }
}

// ---- Single-pass exact LM head (sm_70 Volta mma.m8n8k4, 4 warps = 128 threads, 32 rows/warp = 128 rows/CTA) ----
__global__ void vhead_clear_v70_kernel(unsigned long long* __restrict__ d_best, uint32_t* __restrict__ d_nf,
                                         int32_t* __restrict__ d_max_lo, int T, int t_offset, bool clear_tiles) {
    const int t = (int) threadIdx.x;
    if (t < T) {
        const int ts = t_offset + t;
        d_best[ts] = 0ull;
        if (d_nf != nullptr) d_nf[ts] = 0u;
        if (clear_tiles && d_max_lo != nullptr) d_max_lo[ts] = float_to_ordered(-1e30f);
    }
}

template <bool IS_Q6K, int MAX_NTILES, bool WRITE_LOGITS, bool WRITE_BEST, bool WRITE_TILES>
__global__ void __launch_bounds__(128) vhead_exact_v70_kernel(
    const void* __restrict__ hi_tiled, const void* __restrict__ lo_tiled, const uint16_t* __restrict__ d_tiled,
    const int8_t* __restrict__ sc_tiled, const uint16_t* __restrict__ d_all_tiled, const Q81Block* __restrict__ xq,
    int n_in, int n_out, int T, int t_offset,
    float* __restrict__ out_logits, unsigned long long* __restrict__ d_best, uint32_t* __restrict__ d_nf,
    int32_t* __restrict__ d_max_lo, float* __restrict__ d_tile_hi, float* __restrict__ d_tile_lo) {
#if STRATA_VHEAD_SM70
    constexpr int WARPS = 4;
    constexpr int MAX_T = MAX_NTILES * 8;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q = (lane >> 2) & 3;
    const int g = lane & 3;
    const int h = lane >> 4;
    const int t_qp = g + 4 * h;

    const int nb32 = n_in >> 5;
    const int n_scales = IS_Q6K ? (n_in >> 4) : nb32;
    const int nsb = n_in >> 8;
    const int n_tiles = (n_out + 15) >> 4;
    const int n_tp = (n_tiles + 1) >> 1;
    const int tp = (int) blockIdx.x * WARPS + warp;
    const bool tp_active = tp < n_tp;

    __shared__ alignas(16) uint2 s_xh_u2[MAX_NTILES][16][32]; // 4 KB (T<=8) or 8 KB (T<=16), reused for s_log
    __shared__ float s_dx[MAX_T * 8];

    float acc[MAX_NTILES][8];
#pragma unroll
    for (int nt = 0; nt < MAX_NTILES; ++nt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[nt][i] = 0.0f;
    }

    const size_t tp_off = tp_active ? (size_t) tp : 0u;
    const uint4* hi8_base = reinterpret_cast<const uint4*>(hi_tiled) + tp_off * nb32 * 32u + lane;
    const uint4* lo8_base = reinterpret_cast<const uint4*>(lo_tiled) + tp_off * nb32 * 32u + lane;
    const uint2* hi6_base = reinterpret_cast<const uint2*>(hi_tiled) + tp_off * n_scales * 32u + lane;
    const uint32_t* lo6_base = reinterpret_cast<const uint32_t*>(lo_tiled) + tp_off * n_scales * 32u + lane;
    const uint16_t* dt8_base = d_tiled + tp_off * nb32 * 32u + lane;
    const int8_t* sc_base = sc_tiled + tp_off * n_scales * 32u + lane;
    const uint16_t* dall_base = d_all_tiled + tp_off * nsb * 32u + lane;

    for (int k_blk0 = 0; k_blk0 < nb32; k_blk0 += 8) {
        for (int idx = tid; idx < MAX_T * 64; idx += 128) {
            const int t = idx >> 6;
            const int sb = (idx >> 2) & 15;
            const int q4 = idx & 3;
            const int b = sb >> 1;
            const int half16 = sb & 1;
            const int nt = t >> 3;
            const int tx = t & 7;
            const int slot = ((tx & 4) << 2) | (q4 << 2) | (tx & 3);
            if (t < T) {
                const Q81Block& blk = xq[(size_t) t * nb32 + k_blk0 + b];
                const uint32_t w4 = *reinterpret_cast<const uint32_t*>(blk.qs + half16 * 16 + q4 * 4);
                s_xh_u2[nt][sb][slot] = vhead_i8h_cvt4(w4);
                if (q4 == 0 && half16 == 0) {
                    float dx = __low2float(blk.ds);
                    if (!isfinite(dx)) {
                        if constexpr (WRITE_BEST) {
                            if (d_nf != nullptr) atomicExch(d_nf + t_offset + t, 0x7fc00000u);
                        }
                        dx = 0.0f;
                    }
                    s_dx[t * 8 + b] = dx;
                }
            } else {
                s_xh_u2[nt][sb][slot] = make_uint2(0u, 0u);
                if (q4 == 0 && half16 == 0) s_dx[t * 8 + b] = 0.0f;
            }
        }
        __syncthreads();

        if (tp_active) {
            if constexpr (!IS_Q6K) {
#pragma unroll 4
                for (int b = 0; b < 8; ++b) {
                    const uint4 qs0 = __ldg(hi8_base + (size_t) (k_blk0 + b) * 32u);
                    const uint4 qs1 = __ldg(lo8_base + (size_t) (k_blk0 + b) * 32u);
                    const float dw_lane = __half2float(__ushort_as_half(__ldg(dt8_base + (size_t) (k_blk0 + b) * 32u)));
                    const float dw_a = __shfl_sync(0xffffffffu, dw_lane, lane & ~2);
                    const float dw_b = __shfl_sync(0xffffffffu, dw_lane, lane | 2);

                    float d[MAX_NTILES][8];
#pragma unroll
                    for (int nt = 0; nt < MAX_NTILES; ++nt)
#pragma unroll
                        for (int i = 0; i < 8; ++i) d[nt][i] = 0.0f;

                    const uint32_t w_steps[8] = {
                        qs0.x, qs0.y, qs0.z, qs0.w,
                        qs1.x, qs1.y, qs1.z, qs1.w
                    };
#pragma unroll
                    for (int st = 0; st < 8; ++st) {
                        const int sb = 2 * b + (st >> 2);
                        const int k4 = st & 3;
                        const uint2 a_op = vhead_i8h_cvt4(w_steps[st]);
                        const int slot_b = ((t_qp & 4) << 2) | (k4 << 2) | (t_qp & 3);
#pragma unroll
                        for (int nt = 0; nt < MAX_NTILES; ++nt) {
                            mma_f16_m8n8k4(d[nt], a_op, s_xh_u2[nt][sb][slot_b]);
                        }
                    }
#pragma unroll
                    for (int nt = 0; nt < MAX_NTILES; ++nt) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int tc = nt * 8 + (i & 4) + (g & 2) + (i & 1);
                            const float dw = (i & 2) ? dw_b : dw_a;
                            acc[nt][i] = fmaf(dw * s_dx[tc * 8 + b], d[nt][i], acc[nt][i]);
                        }
                    }
                }
            } else {
                const int sb_super = k_blk0 >> 3;
                const float d_all_lane = __half2float(__ushort_as_half(__ldg(dall_base + (size_t) sb_super * 32u)));
#pragma unroll 4
                for (int sb = 0; sb < 16; ++sb) {
                    const int b = sb >> 1;
                    const int gsub = k_blk0 * 2 + sb;
                    const uint2 pk_lo4 = __ldg(hi6_base + (size_t) gsub * 32u);
                    const uint32_t pk_hi2 = __ldg(lo6_base + (size_t) gsub * 32u);
                    const int8_t sc_lane = __ldg(sc_base + (size_t) gsub * 32u);
                    const float dw_lane = d_all_lane * (float) sc_lane;
                    const float dw_a = __shfl_sync(0xffffffffu, dw_lane, lane & ~2);
                    const float dw_b = __shfl_sync(0xffffffffu, dw_lane, lane | 2);

                    float d[MAX_NTILES][8];
#pragma unroll
                    for (int nt = 0; nt < MAX_NTILES; ++nt)
#pragma unroll
                        for (int i = 0; i < 8; ++i) d[nt][i] = 0.0f;

#pragma unroll
                    for (int st = 0; st < 4; ++st) {
                        const uint2 a_op = vhead_u6h_cvt4(unpack_q6k_v70_u4(pk_lo4, pk_hi2, st));
                        const int slot_b = ((t_qp & 4) << 2) | (st << 2) | (t_qp & 3);
#pragma unroll
                        for (int nt = 0; nt < MAX_NTILES; ++nt) {
                            mma_f16_m8n8k4(d[nt], a_op, s_xh_u2[nt][sb][slot_b]);
                        }
                    }
#pragma unroll
                    for (int nt = 0; nt < MAX_NTILES; ++nt) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int tc = nt * 8 + (i & 4) + (g & 2) + (i & 1);
                            const float dw = (i & 2) ? dw_b : dw_a;
                            acc[nt][i] = fmaf(dw * s_dx[tc * 8 + b], d[nt][i], acc[nt][i]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!tp_active) return;

    float* s_log = reinterpret_cast<float*>(s_xh_u2) + (size_t) warp * (size_t) MAX_T * 32u;
    const int r_in_tp_a = q * 8 + (g & 1) + 4 * h;
    const int r_in_tp_b = r_in_tp_a + 2;
    constexpr int kIA[4] = {0, 1, 4, 5};
    constexpr int kIB[4] = {2, 3, 6, 7};
    constexpr int kCol[4] = {0, 1, 4, 5};

#pragma unroll
    for (int nt = 0; nt < MAX_NTILES; ++nt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int tc = nt * 8 + (g & 2) + kCol[j];
            if (tc < T) {
                s_log[(size_t) tc * 32u + r_in_tp_a] = acc[nt][kIA[j]];
                s_log[(size_t) tc * 32u + r_in_tp_b] = acc[nt][kIB[j]];
            }
        }
    }
    __syncwarp();

    const int row = tp * 32 + lane;
    const bool valid = row < n_out;
    for (int t = 0; t < T; ++t) {
        const float val = s_log[(size_t) t * 32u + lane];
        if constexpr (WRITE_LOGITS) {
            if (valid) out_logits[(size_t) t * (size_t) n_out + (size_t) row] = val;
        }
        if constexpr (WRITE_BEST) {
            const bool nf = valid && !isfinite(val);
            if (__any_sync(0xffffffffu, nf) && lane == 0 && d_nf != nullptr) {
                atomicExch(d_nf + t_offset + t, 0x7fc00000u);
            }
            unsigned long long b = (valid && isfinite(val)) ? pack_val_row(val, row) : 0ull;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const unsigned long long ob = __shfl_xor_sync(0xffffffffu, b, o);
                if (ob > b) b = ob;
            }
            if (lane == 0 && b != 0ull) atomicMax(d_best + t_offset + t, b);
        }
        if constexpr (WRITE_TILES) {
            float tmax = (valid && isfinite(val)) ? val : -1e30f;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
            if ((lane & 15) == 0) {
                const int tile = 2 * tp + (lane >> 4);
                if (tile < n_tiles) {
                    const int ts = t_offset + t;
                    d_tile_hi[(size_t) ts * (size_t) n_tiles + (size_t) tile] = tmax;
                    d_tile_lo[(size_t) ts * (size_t) n_tiles + (size_t) tile] = tmax;
                    if (tmax > ordered_to_float(*reinterpret_cast<volatile int32_t*>(d_max_lo + ts))) {
                        atomic_max_float(d_max_lo + ts, tmax);
                    }
                }
            }
        }
    }
#else
    (void) hi_tiled; (void) lo_tiled; (void) d_tiled; (void) sc_tiled; (void) d_all_tiled; (void) xq;
    (void) n_in; (void) n_out; (void) T; (void) t_offset; (void) out_logits; (void) d_best; (void) d_nf;
    (void) d_max_lo; (void) d_tile_hi; (void) d_tile_lo;
#endif
}

// ---- Stage 1 Screen (sm_80+ INT8 TC) ----
template <bool IS_Q6K, int MAX_NTILES>
__global__ void __launch_bounds__(128) vhead_stage1_tc_kernel(
    const void* __restrict__ hi_tiled, const uint16_t* __restrict__ d_tiled, const float* __restrict__ rho,
    const Q81Block* __restrict__ xq, const float* __restrict__ x_norm, const float* __restrict__ x_sum,
    int n_in, int n_out, int T, int t_offset,
    int32_t* __restrict__ d_max_lo, float* __restrict__ d_tile_hi, float* __restrict__ d_tile_lo) {
#if STRATA_VHEAD_SM80
    constexpr int WARPS = 4;
    constexpr int MAX_T = MAX_NTILES * 8;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int nb32 = n_in >> 5;
    const int n_scales = IS_Q6K ? (n_in >> 4) : nb32;
    const int n_tiles = (n_out + 15) >> 4;
    const int tile = (int) blockIdx.x * WARPS + warp;
    const bool tile_active = tile < n_tiles;

    __shared__ alignas(16) int8_t s_xq[MAX_T * 256];
    __shared__ float s_dx[MAX_T * 8];
    __shared__ float s_xsum[MAX_T * (IS_Q6K ? 16 : 8)];
    __shared__ float s_xnorm[MAX_T];

    if (tid < MAX_T) s_xnorm[tid] = tid < T ? x_norm[t_offset + tid] : 0.0f;

    const int r_base = tile << 4;
    const int r0 = r_base + g;
    const int r1 = r_base + 8 + g;
    const bool v0 = tile_active && r0 < n_out;
    const bool v1 = tile_active && r1 < n_out;

    float acc[MAX_NTILES][4];
#pragma unroll
    for (int nt = 0; nt < MAX_NTILES; ++nt) {
        acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
    }

    const size_t tile_off = tile_active ? (size_t) tile : 0u;
    const uint2* hi2_base = reinterpret_cast<const uint2*>(hi_tiled) + tile_off * nb32 * 32u + lane;
    const uint32_t* hi1_base = reinterpret_cast<const uint32_t*>(hi_tiled) + tile_off * n_scales * 32u + lane;
    const uint16_t* dt_base = d_tiled + tile_off * n_scales * 16u;

    for (int k_blk0 = 0; k_blk0 < nb32; k_blk0 += 8) {
        for (int idx = tid; idx < MAX_T * 8; idx += 128) {
            const int t = idx >> 3;
            const int b = idx & 7;
            if (t < T) {
                const Q81Block& blk = xq[(size_t) t * nb32 + k_blk0 + b];
                const float dx = __low2float(blk.ds);
                s_dx[idx] = IS_Q6K ? (4.0f * dx) : (16.0f * dx);
                if constexpr (IS_Q6K) {
                    s_xsum[t * 16 + 2 * b + 0] = 2.0f * x_sum[(size_t) (t_offset + t) * n_scales + (k_blk0 + b) * 2 + 0];
                    s_xsum[t * 16 + 2 * b + 1] = 2.0f * x_sum[(size_t) (t_offset + t) * n_scales + (k_blk0 + b) * 2 + 1];
                } else {
                    s_xsum[idx] = 8.0f * x_sum[(size_t) (t_offset + t) * n_scales + (k_blk0 + b)];
                }
                const uint32_t* src32 = reinterpret_cast<const uint32_t*>(blk.qs);
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + idx * 32);
                dst4[0] = make_uint4(src32[0], src32[1], src32[2], src32[3]);
                dst4[1] = make_uint4(src32[4], src32[5], src32[6], src32[7]);
            } else {
                s_dx[idx] = 0.0f;
                if constexpr (IS_Q6K) {
                    s_xsum[t * 16 + 2 * b + 0] = 0.0f;
                    s_xsum[t * 16 + 2 * b + 1] = 0.0f;
                } else {
                    s_xsum[idx] = 0.0f;
                }
                uint4* dst4 = reinterpret_cast<uint4*>(s_xq + idx * 32);
                dst4[0] = make_uint4(0u, 0u, 0u, 0u);
                dst4[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        if (tile_active) {
            if constexpr (!IS_Q6K) {
#pragma unroll
                for (int b = 0; b < 8; ++b) {
                    const uint2 pk = __ldg(hi2_base + (size_t) (k_blk0 + b) * 32u);
                    const uint32_t a0 = unpack_s4x4((uint16_t) (pk.x & 0xFFFFu));
                    const uint32_t a1 = unpack_s4x4((uint16_t) (pk.x >> 16));
                    const uint32_t a2 = unpack_s4x4((uint16_t) (pk.y & 0xFFFFu));
                    const uint32_t a3 = unpack_s4x4((uint16_t) (pk.y >> 16));
                    const uint16_t* db = dt_base + (size_t) (k_blk0 + b) * 16u;
                    const float dw0 = __half2float(__ushort_as_half(__ldg(db + g)));
                    const float dw1 = __half2float(__ushort_as_half(__ldg(db + 8 + g)));

#pragma unroll
                    for (int nt = 0; nt < MAX_NTILES; ++nt) {
                        const int tg = nt * 8 + g;
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + tig * 4);
                        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + 16 + tig * 4);
                        int32_t d[4] = {0, 0, 0, 0};
                        mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1);

                        const int t0 = nt * 8 + 2 * tig;
                        const int t1 = t0 + 1;
                        acc[nt][0] = fmaf(dw0, fmaf(s_dx[t0 * 8 + b], (float) d[0], s_xsum[t0 * 8 + b]), acc[nt][0]);
                        acc[nt][1] = fmaf(dw0, fmaf(s_dx[t1 * 8 + b], (float) d[1], s_xsum[t1 * 8 + b]), acc[nt][1]);
                        acc[nt][2] = fmaf(dw1, fmaf(s_dx[t0 * 8 + b], (float) d[2], s_xsum[t0 * 8 + b]), acc[nt][2]);
                        acc[nt][3] = fmaf(dw1, fmaf(s_dx[t1 * 8 + b], (float) d[3], s_xsum[t1 * 8 + b]), acc[nt][3]);
                    }
                }
            } else {
#pragma unroll
                for (int sb = 0; sb < 16; ++sb) {
                    const int b = sb >> 1;
                    const int half16 = sb & 1;
                    const int gsub = k_blk0 * 2 + sb;
                    const uint32_t pk = __ldg(hi1_base + (size_t) gsub * 32u);
                    const uint32_t a0 = unpack_s4x4((uint16_t) (pk & 0xFFFFu));
                    const uint32_t a1 = unpack_s4x4((uint16_t) (pk >> 16));
                    const uint16_t* db = dt_base + (size_t) gsub * 16u;
                    const float dw0 = __half2float(__ushort_as_half(__ldg(db + g)));
                    const float dw1 = __half2float(__ushort_as_half(__ldg(db + 8 + g)));

#pragma unroll
                    for (int nt = 0; nt < MAX_NTILES; ++nt) {
                        const int tg = nt * 8 + g;
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_xq + (tg * 8 + b) * 32 + half16 * 16 + tig * 4);
                        int32_t d[4] = {0, 0, 0, 0};
                        mma_s8_m16n8k16(d, a0, a1, b0);

                        const int t0 = nt * 8 + 2 * tig;
                        const int t1 = t0 + 1;
                        acc[nt][0] = fmaf(dw0, fmaf(s_dx[t0 * 8 + b], (float) d[0], s_xsum[t0 * 16 + sb]), acc[nt][0]);
                        acc[nt][1] = fmaf(dw0, fmaf(s_dx[t1 * 8 + b], (float) d[1], s_xsum[t1 * 16 + sb]), acc[nt][1]);
                        acc[nt][2] = fmaf(dw1, fmaf(s_dx[t0 * 8 + b], (float) d[2], s_xsum[t0 * 16 + sb]), acc[nt][2]);
                        acc[nt][3] = fmaf(dw1, fmaf(s_dx[t1 * 8 + b], (float) d[3], s_xsum[t1 * 16 + sb]), acc[nt][3]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!tile_active) return;
    const float rho0 = v0 ? __ldg(rho + r0) : 0.0f;
    const float rho1 = v1 ? __ldg(rho + r1) : 0.0f;

#pragma unroll
    for (int nt = 0; nt < MAX_NTILES; ++nt) {
        const int t0 = nt * 8 + 2 * tig;
        const int t1 = t0 + 1;
        const int ts[2] = {t0, t1};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int tc = ts[c];
            const bool active = tc < T;
            const float xn = active ? s_xnorm[tc] : 0.0f;
            const float ap0 = acc[nt][c];
            const float rad0 = rho0 * xn + 1e-5f * fabsf(ap0) + 1e-5f;
            const float lo0 = (active && v0) ? (ap0 - rad0) : -1e30f;
            const float hi0 = (active && v0) ? (ap0 + rad0) : -1e38f;
            const float ap1 = acc[nt][2 + c];
            const float rad1 = rho1 * xn + 1e-5f * fabsf(ap1) + 1e-5f;
            const float lo1 = (active && v1) ? (ap1 - rad1) : -1e30f;
            const float hi1 = (active && v1) ? (ap1 + rad1) : -1e38f;

            float tile_lo = fmaxf(lo0, lo1);
            float tile_hi = fmaxf(hi0, hi1);
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                tile_lo = fmaxf(tile_lo, __shfl_xor_sync(0xffffffffu, tile_lo, o));
                tile_hi = fmaxf(tile_hi, __shfl_xor_sync(0xffffffffu, tile_hi, o));
            }
            if (active && g == 0) {
                const int tslot = t_offset + tc;
                d_tile_hi[(size_t) tslot * (size_t) n_tiles + (size_t) tile] = tile_hi;
                d_tile_lo[(size_t) tslot * (size_t) n_tiles + (size_t) tile] = tile_lo;
                if (tile_lo > ordered_to_float(*reinterpret_cast<volatile int32_t*>(d_max_lo + tslot))) {
                    atomic_max_float(d_max_lo + tslot, tile_lo);
                }
            }
        }
    }
#else
    (void) hi_tiled; (void) d_tiled; (void) rho; (void) xq; (void) x_norm; (void) x_sum;
    (void) n_in; (void) n_out; (void) T; (void) t_offset; (void) d_max_lo; (void) d_tile_hi; (void) d_tile_lo;
#endif
}

// ---- Exact dot helpers from zero-duplication HI + LO planes ----

template <int CC_MODE>
__device__ __forceinline__ float exact_dot_q8_0_planes_warp(
    const void* __restrict__ hi_tiled, const uint4* __restrict__ lo_row, const uint16_t* __restrict__ d_tiled,
    int row, const Q81Block* __restrict__ xt, int nb32, int lane) {
    const uint4* hi_v70 = reinterpret_cast<const uint4*>(hi_tiled);
    const uint2* hi_sm80 = reinterpret_cast<const uint2*>(hi_tiled);
    const int tp = row >> 5;
    const int l_row = lane_for_r32_v70(row & 31);
    const int tile = row >> 4;
    const int r16 = row & 15;
    const int g_sm80 = r16 & 7;
    const int half_sm80 = r16 >> 3;

    float sum = 0.0f;
    for (int b = lane; b < nb32; b += 32) {
        uint4 lo4;
        uint4 hi4;
        uint16_t d_bits;
        if constexpr (CC_MODE == 70) {
            lo4 = __ldg(lo_row + ((size_t) tp * nb32 + b) * 32u + l_row);
            hi4 = __ldg(hi_v70 + ((size_t) tp * nb32 + b) * 32u + l_row);
            d_bits = __ldg(d_tiled + ((size_t) tp * nb32 + b) * 32u + l_row);
        } else {
            lo4 = __ldg(lo_row + (size_t) row * nb32 + b);
            const size_t base = ((size_t) tile * nb32 + b) * 32u + (size_t) (g_sm80 << 2);
            const uint2 p0 = __ldg(hi_sm80 + base + 0);
            const uint2 p1 = __ldg(hi_sm80 + base + 1);
            const uint2 p2 = __ldg(hi_sm80 + base + 2);
            const uint2 p3 = __ldg(hi_sm80 + base + 3);
            if (half_sm80 == 0) {
                hi4 = make_uint4((p0.x & 0xFFFFu) | ((p1.x & 0xFFFFu) << 16),
                                 (p2.x & 0xFFFFu) | ((p3.x & 0xFFFFu) << 16),
                                 (p0.y & 0xFFFFu) | ((p1.y & 0xFFFFu) << 16),
                                 (p2.y & 0xFFFFu) | ((p3.y & 0xFFFFu) << 16));
            } else {
                hi4 = make_uint4((p0.x >> 16) | ((p1.x >> 16) << 16),
                                 (p2.x >> 16) | ((p3.x >> 16) << 16),
                                 (p0.y >> 16) | ((p1.y >> 16) << 16),
                                 (p2.y >> 16) | ((p3.y >> 16) << 16));
            }
            d_bits = __ldg(d_tiled + ((size_t) tile * nb32 + b) * 16u + r16);
        }
        const float dw = __half2float(__ushort_as_half(d_bits));
        const float dx = __low2float(xt[b].ds);
        const int* x4 = reinterpret_cast<const int*>(xt[b].qs);
        int dot = 0;
        if constexpr (CC_MODE == 70) {
            dot = STRATA_DP4A((int) hi4.x, x4[0], dot);
            dot = STRATA_DP4A((int) hi4.y, x4[1], dot);
            dot = STRATA_DP4A((int) hi4.z, x4[2], dot);
            dot = STRATA_DP4A((int) hi4.w, x4[3], dot);
            dot = STRATA_DP4A((int) lo4.x, x4[4], dot);
            dot = STRATA_DP4A((int) lo4.y, x4[5], dot);
            dot = STRATA_DP4A((int) lo4.z, x4[6], dot);
            dot = STRATA_DP4A((int) lo4.w, x4[7], dot);
        } else {
            dot = STRATA_DP4A(pack_q8_4((uint16_t) (hi4.x & 0xFFFFu), (uint16_t) (lo4.x & 0xFFFFu)), x4[0], dot);
            dot = STRATA_DP4A(pack_q8_4((uint16_t) (hi4.x >> 16), (uint16_t) (lo4.x >> 16)), x4[1], dot);
            dot = STRATA_DP4A(pack_q8_4((uint16_t) (hi4.y & 0xFFFFu), (uint16_t) (lo4.y & 0xFFFFu)), x4[2], dot);
            dot = STRATA_DP4A(pack_q8_4((uint16_t) (hi4.y >> 16), (uint16_t) (lo4.y >> 16)), x4[3], dot);
            dot = STRATA_DP4A(pack_q8_4((uint16_t) (hi4.z & 0xFFFFu), (uint16_t) (lo4.z & 0xFFFFu)), x4[4], dot);
            dot = STRATA_DP4A(pack_q8_4((uint16_t) (hi4.z >> 16), (uint16_t) (lo4.z >> 16)), x4[5], dot);
            dot = STRATA_DP4A(pack_q8_4((uint16_t) (hi4.w & 0xFFFFu), (uint16_t) (lo4.w & 0xFFFFu)), x4[6], dot);
            dot = STRATA_DP4A(pack_q8_4((uint16_t) (hi4.w >> 16), (uint16_t) (lo4.w >> 16)), x4[7], dot);
        }
        sum = fmaf(dw * dx, (float) dot, sum);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    return sum;
}

template <int CC_MODE>
__device__ __forceinline__ float exact_dot_q6_k_planes_warp(
    const void* __restrict__ hi_tiled, const uint32_t* __restrict__ lo_row,
    const int8_t* __restrict__ sc_arr, const uint16_t* __restrict__ dall_arr,
    int row, const Q81Block* __restrict__ xt, int n_in, int lane) {
    const uint2* hi_v70 = reinterpret_cast<const uint2*>(hi_tiled);
    const uint32_t* hi_sm80 = reinterpret_cast<const uint32_t*>(hi_tiled);
    const int n_scales = n_in >> 4;
    const int nsb = n_in >> 8;
    const int tp = row >> 5;
    const int l_row = lane_for_r32_v70(row & 31);
    const int tile = row >> 4;
    const int r16 = row & 15;
    const int g_sm80 = r16 & 7;
    const int half_sm80 = r16 >> 3;

    float sum = 0.0f;
    for (int gsub = lane; gsub < n_scales; gsub += 32) {
        const int sb = gsub >> 4;
        uint32_t lo32;
        uint2 hi2;
        int8_t sc;
        uint16_t d_bits;
        if constexpr (CC_MODE == 70) {
            lo32 = __ldg(lo_row + ((size_t) tp * n_scales + gsub) * 32u + l_row);
            hi2 = __ldg(hi_v70 + ((size_t) tp * n_scales + gsub) * 32u + l_row);
            sc = __ldg(sc_arr + ((size_t) tp * n_scales + gsub) * 32u + l_row);
            d_bits = __ldg(dall_arr + ((size_t) tp * nsb + sb) * 32u + l_row);
        } else {
            lo32 = __ldg(lo_row + (size_t) row * n_scales + gsub);
            const size_t base = ((size_t) tile * n_scales + gsub) * 32u + (size_t) (g_sm80 << 2);
            const uint32_t p0 = __ldg(hi_sm80 + base + 0);
            const uint32_t p1 = __ldg(hi_sm80 + base + 1);
            const uint32_t p2 = __ldg(hi_sm80 + base + 2);
            const uint32_t p3 = __ldg(hi_sm80 + base + 3);
            if (half_sm80 == 0) {
                hi2 = make_uint2((p0 & 0xFFFFu) | ((p1 & 0xFFFFu) << 16),
                                 (p2 & 0xFFFFu) | ((p3 & 0xFFFFu) << 16));
            } else {
                hi2 = make_uint2((p0 >> 16) | ((p1 >> 16) << 16),
                                 (p2 >> 16) | ((p3 >> 16) << 16));
            }
            sc = __ldg(sc_arr + (size_t) row * n_scales + gsub);
            d_bits = __ldg(dall_arr + (size_t) row * nsb + sb);
        }
        const float d_sub = __half2float(__ushort_as_half(d_bits)) * (float) sc;
        const int b32 = gsub >> 1;
        const int half16 = gsub & 1;
        const float dx = __low2float(xt[b32].ds);
        const int* x4 = reinterpret_cast<const int*>(xt[b32].qs + half16 * 16);
        int dot = 0;
        if constexpr (CC_MODE == 70) {
            dot = STRATA_DP4A(pack_q6k_v70_s8_4(hi2, lo32, 0), x4[0], dot);
            dot = STRATA_DP4A(pack_q6k_v70_s8_4(hi2, lo32, 1), x4[1], dot);
            dot = STRATA_DP4A(pack_q6k_v70_s8_4(hi2, lo32, 2), x4[2], dot);
            dot = STRATA_DP4A(pack_q6k_v70_s8_4(hi2, lo32, 3), x4[3], dot);
        } else {
            dot = STRATA_DP4A(pack_q6k_4((uint16_t) (hi2.x & 0xFFFFu), (uint8_t) (lo32 & 0xFFu)), x4[0], dot);
            dot = STRATA_DP4A(pack_q6k_4((uint16_t) (hi2.x >> 16), (uint8_t) ((lo32 >> 8) & 0xFFu)), x4[1], dot);
            dot = STRATA_DP4A(pack_q6k_4((uint16_t) (hi2.y & 0xFFFFu), (uint8_t) ((lo32 >> 16) & 0xFFu)), x4[2], dot);
            dot = STRATA_DP4A(pack_q6k_4((uint16_t) (hi2.y >> 16), (uint8_t) (lo32 >> 24)), x4[3], dot);
        }
        sum = fmaf(d_sub * dx, (float) dot, sum);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    return sum;
}

// ---- Stage 2 Multi-CTA Exact Tile Refinement for Greedy Argmax ----
template <bool IS_Q6K, int CC_MODE>
__global__ void __launch_bounds__(128) vhead_stage2_tiles_kernel(
    const void* __restrict__ hi_tiled, const void* __restrict__ lo_plane,
    const uint16_t* __restrict__ d_tiled, const int8_t* __restrict__ sc_arr,
    const uint16_t* __restrict__ dall_arr, const Q81Block* __restrict__ xq,
    int n_in, int n_out, int n_tiles, int t_offset,
    const int32_t* __restrict__ d_max_lo, const float* __restrict__ d_tile_hi,
    unsigned long long* __restrict__ d_best) {
    const int t = (int) blockIdx.y;
    const int ts = t_offset + t;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int global_warp = (int) blockIdx.x * 4 + warp;
    const int total_warps = (int) gridDim.x * 4;

    const float final_lo = ordered_to_float(d_max_lo[ts]);
    const float* th = d_tile_hi + (size_t) ts * (size_t) n_tiles;
    const int nb32 = n_in >> 5;
    const Q81Block* xt = xq + (size_t) t * (size_t) nb32;

    unsigned long long best = 0ull;

    for (int tile = global_warp; tile < n_tiles; tile += total_warps) {
        if (th[tile] < final_lo) continue;
        const int r0 = tile << 4;
        const int r_end = min(r0 + 16, n_out);
        for (int row = r0; row < r_end; ++row) {
            const float val = IS_Q6K
                ? exact_dot_q6_k_planes_warp<CC_MODE>(hi_tiled, reinterpret_cast<const uint32_t*>(lo_plane),
                                                      sc_arr, dall_arr, row, xt, n_in, lane)
                : exact_dot_q8_0_planes_warp<CC_MODE>(hi_tiled, reinterpret_cast<const uint4*>(lo_plane),
                                                      d_tiled, row, xt, nb32, lane);
            if (lane == 0) {
                const unsigned long long cand = pack_val_row(val, row);
                if (cand > best) best = cand;
            }
        }
    }
    if (lane == 0 && best != 0ull) {
        atomicMax(d_best + ts, best);
    }
}

__global__ void vhead_extract_tokens_kernel(const unsigned long long* __restrict__ d_best,
                                            const float* __restrict__ d_x_norm, int T, int t_offset,
                                            int32_t* __restrict__ out_tokens,
                                            uint32_t* __restrict__ out_nonfinite, int nf_stride) {
    const int t = (int) threadIdx.x;
    if (t < T) {
        const int ts = t_offset + t;
        const unsigned long long packed = d_best[ts];
        out_tokens[t] = (int32_t) (0xFFFFFFFFu - (uint32_t) (packed & 0xFFFFFFFFull));
        if (out_nonfinite != nullptr && nf_stride > 0) {
            const float best_v = ordered_to_float((int32_t) ((uint32_t) (packed >> 32) ^ 0x80000000u));
            const uint32_t bad = (packed != 0ull && isfinite(d_x_norm[ts]) && isfinite(best_v)) ? 0u : 1u;
            uint32_t* dst = out_nonfinite + (size_t) t * (size_t) nf_stride;
            dst[0] = bad;
            for (int b = 1; b < nf_stride; ++b) dst[b] = 0u;
        }
    }
}

// ---- Threshold Kernel for Sampled Candidate Extraction ----
// Computes guaranteed lower bound on k_need-th largest tile_lo via 2-pass 256-bucket histogram (1 KB smem),
// optionally tightened by max_lo + logf(min_p) - 1e-4f when use_min_p_cut is true.
__global__ void vhead_sample_thresh_kernel(
    const float* __restrict__ d_tile_lo, const int32_t* __restrict__ d_max_lo,
    const float* __restrict__ d_x_norm,
    int n_tiles, int T, int t_offset, int k_need, float min_p, bool use_min_p_cut,
    float* __restrict__ d_thresh, int32_t* __restrict__ d_cand_counts, uint8_t* __restrict__ d_cand_overflow) {
    const int t = (int) blockIdx.x;
    if (t >= T) return;
    const int ts = t_offset + t;
    const int tid = (int) threadIdx.x;

    if (tid == 0) {
        d_cand_counts[t] = 0;
        d_cand_overflow[t] = 0;
    }

    float thr = -1e30f;
    __shared__ uint32_t s_found;
    if (tid == 0) s_found = 0u;
    __syncthreads();

    if (k_need > 0 && k_need <= n_tiles) {
        __shared__ uint32_t hist[256];
        __shared__ uint32_t s_b0, s_b1, s_rem;
        const float* tl = d_tile_lo + (size_t) ts * (size_t) n_tiles;

        for (int b = tid; b < 256; b += blockDim.x) hist[b] = 0u;
        __syncthreads();
        for (int i = tid; i < n_tiles; i += blockDim.x) {
            const float v = tl[i];
            if (isfinite(v)) {
                const uint32_t u = float_to_u32_ordered(v);
                atomicAdd(&hist[u >> 24], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0u;
            uint32_t b0 = 0u;
            for (int b = 255; b >= 0; --b) {
                const uint32_t c = hist[b];
                if (acc + c >= (uint32_t) k_need) {
                    b0 = (uint32_t) b;
                    s_rem = (uint32_t) k_need - acc;
                    s_found = 1u;
                    break;
                }
                acc += c;
            }
            s_b0 = b0;
        }
        __syncthreads();

        if (s_found != 0u) {
            for (int b = tid; b < 256; b += blockDim.x) hist[b] = 0u;
            __syncthreads();
            const uint32_t target_b0 = s_b0;
            for (int i = tid; i < n_tiles; i += blockDim.x) {
                const float v = tl[i];
                if (isfinite(v)) {
                    const uint32_t u = float_to_u32_ordered(v);
                    if ((u >> 24) == target_b0) atomicAdd(&hist[(u >> 16) & 0xFFu], 1u);
                }
            }
            __syncthreads();
            if (tid == 0) {
                uint32_t acc = 0u;
                uint32_t b1 = 0u;
                const uint32_t rem = s_rem;
                for (int b = 255; b >= 0; --b) {
                    const uint32_t c = hist[b];
                    if (acc + c >= rem) {
                        b1 = (uint32_t) b;
                        break;
                    }
                    acc += c;
                }
                s_b1 = b1;
            }
            __syncthreads();
            thr = u32_ordered_to_float((s_b0 << 24) | (s_b1 << 16)) - 1e-4f;
        }
    }

    if (use_min_p_cut && min_p > 0.0f && min_p < 1.0f) {
        const float max_lo = ordered_to_float(d_max_lo[ts]);
        const float mp_thr = max_lo + logf(min_p) - 1e-4f;
        thr = (k_need > 0 && s_found != 0u) ? fmaxf(thr, mp_thr) : mp_thr;
    }

    if (tid == 0) {
        const float max_lo = ordered_to_float(d_max_lo[ts]);
        const float xn = d_x_norm[ts];
        const bool k_missing = (k_need > 0 && s_found == 0u && !(use_min_p_cut && min_p > 0.0f && min_p < 1.0f));
        if (!isfinite(xn) || !isfinite(max_lo) || !isfinite(thr) || k_missing) {
            thr = -1e30f;
            d_cand_overflow[t] = 1;
        }
        d_thresh[t] = thr;
    }
}

// ---- Stage 2 Candidate Extraction (for Sampled Decoding) ----
template <bool IS_Q6K, int CC_MODE>
__global__ void __launch_bounds__(128) vhead_stage2_candidates_kernel(
    const void* __restrict__ hi_tiled, const void* __restrict__ lo_plane,
    const uint16_t* __restrict__ d_tiled, const int8_t* __restrict__ sc_arr,
    const uint16_t* __restrict__ dall_arr, const Q81Block* __restrict__ xq,
    int n_in, int n_out, int n_tiles, int t_offset, int max_cand,
    const float* __restrict__ d_tile_hi, const float* __restrict__ d_thresh,
    int32_t* __restrict__ d_cand_ids, float* __restrict__ d_cand_logits,
    int32_t* __restrict__ d_cand_counts, uint8_t* __restrict__ d_cand_overflow) {
    const int t = (int) blockIdx.y;
    if (d_cand_overflow[t] != 0) return;
    const int ts = t_offset + t;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int global_warp = (int) blockIdx.x * 4 + warp;
    const int total_warps = (int) gridDim.x * 4;

    const float thr = d_thresh[t];
    const float* th = d_tile_hi + (size_t) ts * (size_t) n_tiles;
    const int nb32 = n_in >> 5;
    const Q81Block* xt = xq + (size_t) t * (size_t) nb32;
    int32_t* ids_out = d_cand_ids + (size_t) t * (size_t) max_cand;
    float* log_out = d_cand_logits + (size_t) t * (size_t) max_cand;

    for (int tile = global_warp; tile < n_tiles; tile += total_warps) {
        if (*reinterpret_cast<volatile uint8_t*>(d_cand_overflow + t) != 0) return;
        if (th[tile] < thr) continue;
        const int r0 = tile << 4;
        const int r_end = min(r0 + 16, n_out);
        for (int row = r0; row < r_end; ++row) {
            const float val = IS_Q6K
                ? exact_dot_q6_k_planes_warp<CC_MODE>(hi_tiled, reinterpret_cast<const uint32_t*>(lo_plane),
                                                      sc_arr, dall_arr, row, xt, n_in, lane)
                : exact_dot_q8_0_planes_warp<CC_MODE>(hi_tiled, reinterpret_cast<const uint4*>(lo_plane),
                                                      d_tiled, row, xt, nb32, lane);
            if (lane == 0) {
                if (!isfinite(val)) {
                    d_cand_overflow[t] = 1;
                } else if (val >= thr) {
                    const int pos = atomicAdd(d_cand_counts + t, 1);
                    if (pos < max_cand) {
                        ids_out[pos] = row;
                        log_out[pos] = val;
                    } else {
                        d_cand_overflow[t] = 1;
                    }
                }
            }
        }
    }
}

__global__ void vhead_copy_overflow_kernel(const uint8_t* __restrict__ d_ov, uint8_t* __restrict__ h_ov, int T) {
    const int t = (int) threadIdx.x;
    if (t < T) h_ov[t] = d_ov[t];
}

// ---- Full Logits Kernel (from zero-duplication HI + LO planes) ----
template <bool IS_Q6K, int CC_MODE>
__global__ void __launch_bounds__(128) vhead_full_logits_kernel(
    const void* __restrict__ hi_tiled, const void* __restrict__ lo_plane,
    const uint16_t* __restrict__ d_tiled, const int8_t* __restrict__ sc_arr,
    const uint16_t* __restrict__ dall_arr, const Q81Block* __restrict__ xq,
    int n_in, int n_out, int T, float* __restrict__ out_logits) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = (int) blockIdx.x * 4 + warp;
    if (row >= n_out) return;

    const int nb32 = n_in >> 5;
    const int n_scales = n_in >> 4;
    const int nsb = n_in >> 8;
    const int tp = row >> 5;
    const int l_row = lane_for_r32_v70(row & 31);
    const int tile = row >> 4;
    const int r16 = row & 15;
    const int g_sm80 = r16 & 7;
    const int half_sm80 = r16 >> 3;

    float sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if constexpr (!IS_Q6K) {
        const uint4* lo_row = reinterpret_cast<const uint4*>(lo_plane);
        const uint4* hi_v70 = reinterpret_cast<const uint4*>(hi_tiled);
        const uint2* hi_sm80 = reinterpret_cast<const uint2*>(hi_tiled);
        for (int b = lane; b < nb32; b += 32) {
            uint4 lo4;
            uint4 hi4;
            uint16_t d_bits;
            if constexpr (CC_MODE == 70) {
                lo4 = __ldg(lo_row + ((size_t) tp * nb32 + b) * 32u + l_row);
                hi4 = __ldg(hi_v70 + ((size_t) tp * nb32 + b) * 32u + l_row);
                d_bits = __ldg(d_tiled + ((size_t) tp * nb32 + b) * 32u + l_row);
            } else {
                lo4 = __ldg(lo_row + (size_t) row * nb32 + b);
                const size_t base = ((size_t) tile * nb32 + b) * 32u + (size_t) (g_sm80 << 2);
                const uint2 p0 = __ldg(hi_sm80 + base + 0);
                const uint2 p1 = __ldg(hi_sm80 + base + 1);
                const uint2 p2 = __ldg(hi_sm80 + base + 2);
                const uint2 p3 = __ldg(hi_sm80 + base + 3);
                if (half_sm80 == 0) {
                    hi4 = make_uint4((p0.x & 0xFFFFu) | ((p1.x & 0xFFFFu) << 16),
                                     (p2.x & 0xFFFFu) | ((p3.x & 0xFFFFu) << 16),
                                     (p0.y & 0xFFFFu) | ((p1.y & 0xFFFFu) << 16),
                                     (p2.y & 0xFFFFu) | ((p3.y & 0xFFFFu) << 16));
                } else {
                    hi4 = make_uint4((p0.x >> 16) | ((p1.x >> 16) << 16),
                                     (p2.x >> 16) | ((p3.x >> 16) << 16),
                                     (p0.y >> 16) | ((p1.y >> 16) << 16),
                                     (p2.y >> 16) | ((p3.y >> 16) << 16));
                }
                d_bits = __ldg(d_tiled + ((size_t) tile * nb32 + b) * 16u + r16);
            }
            const float dw = __half2float(__ushort_as_half(d_bits));
            int w0, w1, w2, w3, w4, w5, w6, w7;
            if constexpr (CC_MODE == 70) {
                w0 = (int) hi4.x; w1 = (int) hi4.y; w2 = (int) hi4.z; w3 = (int) hi4.w;
                w4 = (int) lo4.x; w5 = (int) lo4.y; w6 = (int) lo4.z; w7 = (int) lo4.w;
            } else {
                w0 = pack_q8_4((uint16_t) (hi4.x & 0xFFFFu), (uint16_t) (lo4.x & 0xFFFFu));
                w1 = pack_q8_4((uint16_t) (hi4.x >> 16), (uint16_t) (lo4.x >> 16));
                w2 = pack_q8_4((uint16_t) (hi4.y & 0xFFFFu), (uint16_t) (lo4.y & 0xFFFFu));
                w3 = pack_q8_4((uint16_t) (hi4.y >> 16), (uint16_t) (lo4.y >> 16));
                w4 = pack_q8_4((uint16_t) (hi4.z & 0xFFFFu), (uint16_t) (lo4.z & 0xFFFFu));
                w5 = pack_q8_4((uint16_t) (hi4.z >> 16), (uint16_t) (lo4.z >> 16));
                w6 = pack_q8_4((uint16_t) (hi4.w & 0xFFFFu), (uint16_t) (lo4.w & 0xFFFFu));
                w7 = pack_q8_4((uint16_t) (hi4.w >> 16), (uint16_t) (lo4.w >> 16));
            }
            for (int t = 0; t < T; ++t) {
                const Q81Block& xb = xq[(size_t) t * nb32 + b];
                const float dx = __low2float(xb.ds);
                const int* x4 = reinterpret_cast<const int*>(xb.qs);
                int dot = 0;
                dot = STRATA_DP4A(w0, x4[0], dot);
                dot = STRATA_DP4A(w1, x4[1], dot);
                dot = STRATA_DP4A(w2, x4[2], dot);
                dot = STRATA_DP4A(w3, x4[3], dot);
                dot = STRATA_DP4A(w4, x4[4], dot);
                dot = STRATA_DP4A(w5, x4[5], dot);
                dot = STRATA_DP4A(w6, x4[6], dot);
                dot = STRATA_DP4A(w7, x4[7], dot);
                sum[t] = fmaf(dw * dx, (float) dot, sum[t]);
            }
        }
    } else {
        const uint32_t* lo_row = reinterpret_cast<const uint32_t*>(lo_plane);
        const uint2* hi_v70 = reinterpret_cast<const uint2*>(hi_tiled);
        const uint32_t* hi_sm80 = reinterpret_cast<const uint32_t*>(hi_tiled);
        for (int gsub = lane; gsub < n_scales; gsub += 32) {
            const int sb = gsub >> 4;
            uint32_t lo32;
            uint2 hi2;
            int8_t sc;
            uint16_t d_bits;
            if constexpr (CC_MODE == 70) {
                lo32 = __ldg(lo_row + ((size_t) tp * n_scales + gsub) * 32u + l_row);
                hi2 = __ldg(hi_v70 + ((size_t) tp * n_scales + gsub) * 32u + l_row);
                sc = __ldg(sc_arr + ((size_t) tp * n_scales + gsub) * 32u + l_row);
                d_bits = __ldg(dall_arr + ((size_t) tp * nsb + sb) * 32u + l_row);
            } else {
                lo32 = __ldg(lo_row + (size_t) row * n_scales + gsub);
                const size_t base = ((size_t) tile * n_scales + gsub) * 32u + (size_t) (g_sm80 << 2);
                const uint32_t p0 = __ldg(hi_sm80 + base + 0);
                const uint32_t p1 = __ldg(hi_sm80 + base + 1);
                const uint32_t p2 = __ldg(hi_sm80 + base + 2);
                const uint32_t p3 = __ldg(hi_sm80 + base + 3);
                if (half_sm80 == 0) {
                    hi2 = make_uint2((p0 & 0xFFFFu) | ((p1 & 0xFFFFu) << 16),
                                     (p2 & 0xFFFFu) | ((p3 & 0xFFFFu) << 16));
                } else {
                    hi2 = make_uint2((p0 >> 16) | ((p1 >> 16) << 16),
                                     (p2 >> 16) | ((p3 >> 16) << 16));
                }
                sc = __ldg(sc_arr + (size_t) row * n_scales + gsub);
                d_bits = __ldg(dall_arr + (size_t) row * nsb + sb);
            }
            const float d_sub = __half2float(__ushort_as_half(d_bits)) * (float) sc;
            int w0, w1, w2, w3;
            if constexpr (CC_MODE == 70) {
                w0 = pack_q6k_v70_s8_4(hi2, lo32, 0);
                w1 = pack_q6k_v70_s8_4(hi2, lo32, 1);
                w2 = pack_q6k_v70_s8_4(hi2, lo32, 2);
                w3 = pack_q6k_v70_s8_4(hi2, lo32, 3);
            } else {
                w0 = pack_q6k_4((uint16_t) (hi2.x & 0xFFFFu), (uint8_t) (lo32 & 0xFFu));
                w1 = pack_q6k_4((uint16_t) (hi2.x >> 16), (uint8_t) ((lo32 >> 8) & 0xFFu));
                w2 = pack_q6k_4((uint16_t) (hi2.y & 0xFFFFu), (uint8_t) ((lo32 >> 16) & 0xFFu));
                w3 = pack_q6k_4((uint16_t) (hi2.y >> 16), (uint8_t) (lo32 >> 24));
            }
            const int b32 = gsub >> 1;
            const int half16 = gsub & 1;
            for (int t = 0; t < T; ++t) {
                const Q81Block& xb = xq[(size_t) t * nb32 + b32];
                const float dx = __low2float(xb.ds);
                const int* x4 = reinterpret_cast<const int*>(xb.qs + half16 * 16);
                int dot = 0;
                dot = STRATA_DP4A(w0, x4[0], dot);
                dot = STRATA_DP4A(w1, x4[1], dot);
                dot = STRATA_DP4A(w2, x4[2], dot);
                dot = STRATA_DP4A(w3, x4[3], dot);
                sum[t] = fmaf(d_sub * dx, (float) dot, sum[t]);
            }
        }
    }

    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], o);
        if (lane == 0) out_logits[(size_t) t * n_out + row] = sum[t];
    }
}

// ---- Exact Raw GGUF Row Reconstruction Kernel (for MtpDrafter / Dflash2Drafter bind) ----
template <bool IS_Q6K, int CC_MODE>
__global__ void vhead_gather_raw_kernel(
    const void* __restrict__ hi_tiled, const void* __restrict__ lo_plane,
    const uint16_t* __restrict__ d_tiled, const int8_t* __restrict__ sc_arr,
    const uint16_t* __restrict__ dall_arr, const int32_t* __restrict__ d_rows,
    int64_t n_rows, int n_in, uint8_t* __restrict__ dst) {
    const int64_t out_r = (int64_t) blockIdx.x;
    if (out_r >= n_rows) return;
    const int row = d_rows[out_r];
    const int tid = (int) threadIdx.x;

    const int tp = row >> 5;
    const int l_row = lane_for_r32_v70(row & 31);
    const int tile = row >> 4;
    const int r16 = row & 15;
    const int g_sm80 = r16 & 7;
    const int half_sm80 = r16 >> 3;

    if constexpr (!IS_Q6K) {
        const int nb32 = n_in >> 5;
        const uint4* lo_row = reinterpret_cast<const uint4*>(lo_plane);
        const uint4* hi_v70 = reinterpret_cast<const uint4*>(hi_tiled);
        const uint2* hi_sm80 = reinterpret_cast<const uint2*>(hi_tiled);
        uint8_t* rdst = dst + (size_t) out_r * (size_t) nb32 * 34u;
        for (int b = tid; b < nb32; b += blockDim.x) {
            uint4 lo4;
            uint4 hi4;
            uint16_t d_bits;
            if constexpr (CC_MODE == 70) {
                lo4 = lo_row[((size_t) tp * nb32 + b) * 32u + l_row];
                hi4 = hi_v70[((size_t) tp * nb32 + b) * 32u + l_row];
                d_bits = d_tiled[((size_t) tp * nb32 + b) * 32u + l_row];
            } else {
                lo4 = lo_row[(size_t) row * nb32 + b];
                const size_t base = ((size_t) tile * nb32 + b) * 32u + (size_t) (g_sm80 << 2);
                const uint2 p0 = hi_sm80[base + 0];
                const uint2 p1 = hi_sm80[base + 1];
                const uint2 p2 = hi_sm80[base + 2];
                const uint2 p3 = hi_sm80[base + 3];
                if (half_sm80 == 0) {
                    hi4 = make_uint4((p0.x & 0xFFFFu) | ((p1.x & 0xFFFFu) << 16),
                                     (p2.x & 0xFFFFu) | ((p3.x & 0xFFFFu) << 16),
                                     (p0.y & 0xFFFFu) | ((p1.y & 0xFFFFu) << 16),
                                     (p2.y & 0xFFFFu) | ((p3.y & 0xFFFFu) << 16));
                } else {
                    hi4 = make_uint4((p0.x >> 16) | ((p1.x >> 16) << 16),
                                     (p2.x >> 16) | ((p3.x >> 16) << 16),
                                     (p0.y >> 16) | ((p1.y >> 16) << 16),
                                     (p2.y >> 16) | ((p3.y >> 16) << 16));
                }
                d_bits = d_tiled[((size_t) tile * nb32 + b) * 16u + r16];
            }
            uint8_t* blk = rdst + (size_t) b * 34u;
            blk[0] = (uint8_t) (d_bits & 0xFFu);
            blk[1] = (uint8_t) (d_bits >> 8);
            const uint32_t hw[8] = {hi4.x & 0xFFFFu, hi4.x >> 16, hi4.y & 0xFFFFu, hi4.y >> 16,
                                    hi4.z & 0xFFFFu, hi4.z >> 16, hi4.w & 0xFFFFu, hi4.w >> 16};
            const uint32_t lw[8] = {lo4.x & 0xFFFFu, lo4.x >> 16, lo4.y & 0xFFFFu, lo4.y >> 16,
                                    lo4.z & 0xFFFFu, lo4.z >> 16, lo4.w & 0xFFFFu, lo4.w >> 16};
            if constexpr (CC_MODE == 70) {
                uint16_t* q16 = reinterpret_cast<uint16_t*>(blk + 2);
                auto wr32 = [&](int idx, uint32_t v) {
                    q16[2 * idx] = (uint16_t) (v & 0xFFFFu);
                    q16[2 * idx + 1] = (uint16_t) (v >> 16);
                };
                wr32(0, hi4.x); wr32(1, hi4.y); wr32(2, hi4.z); wr32(3, hi4.w);
                wr32(4, lo4.x); wr32(5, lo4.y); wr32(6, lo4.z); wr32(7, lo4.w);
            } else {
                for (int j = 0; j < 8; ++j) {
                    const int w4 = pack_q8_4((uint16_t) hw[j], (uint16_t) lw[j]);
                    blk[2 + j * 4 + 0] = (uint8_t) (w4 & 0xFF);
                    blk[2 + j * 4 + 1] = (uint8_t) ((w4 >> 8) & 0xFF);
                    blk[2 + j * 4 + 2] = (uint8_t) ((w4 >> 16) & 0xFF);
                    blk[2 + j * 4 + 3] = (uint8_t) ((w4 >> 24) & 0xFF);
                }
            }
        }
    } else {
        const int nsb = n_in >> 8;
        const int n_scales = n_in >> 4;
        const uint32_t* lo_row = reinterpret_cast<const uint32_t*>(lo_plane);
        const uint2* hi_v70 = reinterpret_cast<const uint2*>(hi_tiled);
        const uint32_t* hi_sm80 = reinterpret_cast<const uint32_t*>(hi_tiled);
        uint8_t* rdst = dst + (size_t) out_r * (size_t) nsb * 210u;

        // One thread per super-block (typical nsb=10) writes zero-initialized ql/qh, then fills sub-blocks.
        for (int sb = tid; sb < nsb; sb += blockDim.x) {
            uint8_t* blk = rdst + (size_t) sb * 210u;
            for (int i = 0; i < 192; ++i) blk[i] = 0u;
            uint16_t d_bits;
            if constexpr (CC_MODE == 70) {
                d_bits = dall_arr[((size_t) tp * nsb + sb) * 32u + l_row];
            } else {
                d_bits = dall_arr[(size_t) row * nsb + sb];
            }
            blk[208] = (uint8_t) (d_bits & 0xFFu);
            blk[209] = (uint8_t) (d_bits >> 8);
            for (int sub = 0; sub < 16; ++sub) {
                const int gsub = sb * 16 + sub;
                uint32_t lo32;
                uint2 hi2;
                int8_t sc;
                if constexpr (CC_MODE == 70) {
                    lo32 = lo_row[((size_t) tp * n_scales + gsub) * 32u + l_row];
                    hi2 = hi_v70[((size_t) tp * n_scales + gsub) * 32u + l_row];
                    sc = sc_arr[((size_t) tp * n_scales + gsub) * 32u + l_row];
                } else {
                    lo32 = lo_row[(size_t) row * n_scales + gsub];
                    const size_t base = ((size_t) tile * n_scales + gsub) * 32u + (size_t) (g_sm80 << 2);
                    const uint32_t p0 = hi_sm80[base + 0];
                    const uint32_t p1 = hi_sm80[base + 1];
                    const uint32_t p2 = hi_sm80[base + 2];
                    const uint32_t p3 = hi_sm80[base + 3];
                    if (half_sm80 == 0) {
                        hi2 = make_uint2((p0 & 0xFFFFu) | ((p1 & 0xFFFFu) << 16),
                                         (p2 & 0xFFFFu) | ((p3 & 0xFFFFu) << 16));
                    } else {
                        hi2 = make_uint2((p0 >> 16) | ((p1 >> 16) << 16),
                                         (p2 >> 16) | ((p3 >> 16) << 16));
                    }
                    sc = sc_arr[(size_t) row * n_scales + gsub];
                }
                reinterpret_cast<int8_t*>(blk + 192)[sub] = sc;
                const int n = sub >> 3;
                const int rem = sub & 7;
                const int l_base = (rem & 1) * 16;
                const int qid = rem >> 1;
                uint8_t* ql_n = blk + 64 * n + ((qid & 1) ? 32 : 0);
                uint8_t* qh_n = blk + 128 + 32 * n;
                if constexpr (CC_MODE == 70) {
                    const uint32_t qu4[4] = {
                        unpack_q6k_v70_u4(hi2, lo32, 0),
                        unpack_q6k_v70_u4(hi2, lo32, 1),
                        unpack_q6k_v70_u4(hi2, lo32, 2),
                        unpack_q6k_v70_u4(hi2, lo32, 3)
                    };
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const uint32_t q_u = (qu4[j >> 2] >> (8 * (j & 3))) & 0x3Fu;
                        const int l = l_base + j;
                        if (qid < 2) ql_n[l] |= (uint8_t) (q_u & 0x0Fu);
                        else ql_n[l] |= (uint8_t) ((q_u & 0x0Fu) << 4);
                        qh_n[l] |= (uint8_t) (((q_u >> 4) & 0x03u) << (2 * qid));
                    }
                } else {
                    const int w4[4] = {
                        pack_q6k_4((uint16_t) (hi2.x & 0xFFFFu), (uint8_t) (lo32 & 0xFFu)),
                        pack_q6k_4((uint16_t) (hi2.x >> 16), (uint8_t) ((lo32 >> 8) & 0xFFu)),
                        pack_q6k_4((uint16_t) (hi2.y & 0xFFFFu), (uint8_t) ((lo32 >> 16) & 0xFFu)),
                        pack_q6k_4((uint16_t) (hi2.y >> 16), (uint8_t) (lo32 >> 24))
                    };
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const int q_s = (int) (int8_t) ((w4[j >> 2] >> (8 * (j & 3))) & 0xFF);
                        const uint32_t q_u = (uint32_t) (q_s + 32);
                        const int l = l_base + j;
                        if (qid < 2) ql_n[l] |= (uint8_t) (q_u & 0x0Fu);
                        else ql_n[l] |= (uint8_t) ((q_u & 0x0Fu) << 4);
                        qh_n[l] |= (uint8_t) (((q_u >> 4) & 0x03u) << (2 * qid));
                    }
                }
            }
        }
    }
}

int current_cc() {
#if defined(__HIPCC__)
    return 0;
#else
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) { (void) cudaGetLastError(); return 0; }
    int mj = 0, mn = 0;
    if (cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&mn, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    return mj * 10 + mn;
#endif
}

bool launch_stage1(BitplaneHeadHandle* h, const Q81Block* xq, int T, int t_offset, cudaStream_t s) {
    const bool is_q6k = (h->type == kTypeQ6_K);
    const unsigned grid_s1 = (unsigned) ((h->n_tiles + 3) / 4);
    const int ntiles = T <= 8 ? 1 : 2;
    if (is_q6k) {
        if (ntiles == 1)
            vhead_stage1_tc_kernel<true, 1><<<grid_s1, 128, 0, s>>>(
                h->hi_tiled, h->d_tiled, h->rho, xq, h->d_x_norm, h->d_x_sum, h->n_in, h->n_out, T, t_offset,
                h->d_max_lo, h->d_tile_hi, h->d_tile_lo);
        else
            vhead_stage1_tc_kernel<true, 2><<<grid_s1, 128, 0, s>>>(
                h->hi_tiled, h->d_tiled, h->rho, xq, h->d_x_norm, h->d_x_sum, h->n_in, h->n_out, T, t_offset,
                h->d_max_lo, h->d_tile_hi, h->d_tile_lo);
    } else {
        if (ntiles == 1)
            vhead_stage1_tc_kernel<false, 1><<<grid_s1, 128, 0, s>>>(
                h->hi_tiled, h->d_tiled, h->rho, xq, h->d_x_norm, h->d_x_sum, h->n_in, h->n_out, T, t_offset,
                h->d_max_lo, h->d_tile_hi, h->d_tile_lo);
        else
            vhead_stage1_tc_kernel<false, 2><<<grid_s1, 128, 0, s>>>(
                h->hi_tiled, h->d_tiled, h->rho, xq, h->d_x_norm, h->d_x_sum, h->n_in, h->n_out, T, t_offset,
                h->d_max_lo, h->d_tile_hi, h->d_tile_lo);
    }
    return true;
}

template <bool WRITE_LOGITS, bool WRITE_BEST, bool WRITE_TILES>
void launch_exact_v70(const BitplaneHeadHandle* h, const Q81Block* xq, int T, int t_offset,
                      float* out_logits, unsigned long long* d_best, uint32_t* d_nf,
                      int32_t* d_max_lo, float* d_tile_hi, float* d_tile_lo, cudaStream_t s) {
    const unsigned grid_tp = (unsigned) ((h->n_tile_pairs + 3) / 4);
    const bool is_q6k = (h->type == kTypeQ6_K);
    for (int c0 = 0; c0 < T; c0 += 8) {
        const int HT = (T - c0) < 8 ? (T - c0) : 8;
        const Q81Block* xq_c = xq + (size_t) c0 * (size_t) h->nb32;
        float* log_c = out_logits ? (out_logits + (size_t) c0 * (size_t) h->n_out) : nullptr;
        if (is_q6k) {
            vhead_exact_v70_kernel<true, 1, WRITE_LOGITS, WRITE_BEST, WRITE_TILES><<<grid_tp, 128, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq_c,
                h->n_in, h->n_out, HT, t_offset + c0, log_c, d_best, d_nf, d_max_lo, d_tile_hi, d_tile_lo);
        } else {
            vhead_exact_v70_kernel<false, 1, WRITE_LOGITS, WRITE_BEST, WRITE_TILES><<<grid_tp, 128, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq_c,
                h->n_in, h->n_out, HT, t_offset + c0, log_c, d_best, d_nf, d_max_lo, d_tile_hi, d_tile_lo);
        }
    }
}

bool ensure_candidate_buffers(BitplaneHeadHandle* h) {
    if (h->d_cand_ids != nullptr) return true;
    const size_t ids_b = (size_t) BitplaneHeadHandle::kMaxT * BitplaneHeadHandle::kMaxCand * sizeof(int32_t);
    const size_t log_b = (size_t) BitplaneHeadHandle::kMaxT * BitplaneHeadHandle::kMaxCand * sizeof(float);
    const size_t cnt_b = (size_t) BitplaneHeadHandle::kMaxT * sizeof(int32_t);
    const size_t ov_b = (size_t) BitplaneHeadHandle::kMaxT * sizeof(uint8_t);
    bool ok = cudaMalloc(&h->d_cand_ids, ids_b) == cudaSuccess &&
              cudaMalloc(&h->d_cand_logits, log_b) == cudaSuccess &&
              cudaMalloc(&h->d_cand_counts, cnt_b) == cudaSuccess &&
              cudaMalloc(&h->d_cand_overflow, ov_b) == cudaSuccess &&
              cudaHostAlloc(&h->h_cand_overflow, ov_b, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess;
    if (!ok) {
        (void) cudaGetLastError();
        if (h->d_cand_ids) { cudaFree(h->d_cand_ids); h->d_cand_ids = nullptr; }
        if (h->d_cand_logits) { cudaFree(h->d_cand_logits); h->d_cand_logits = nullptr; }
        if (h->d_cand_counts) { cudaFree(h->d_cand_counts); h->d_cand_counts = nullptr; }
        if (h->d_cand_overflow) { cudaFree(h->d_cand_overflow); h->d_cand_overflow = nullptr; }
        if (h->h_cand_overflow) { cudaFreeHost(h->h_cand_overflow); h->h_cand_overflow = nullptr; }
        return false;
    }
    h->total_bytes += ids_b + log_b + cnt_b + ov_b;
    return true;
}

}  // namespace

bool vhead_bitplane_supported() {
    const int cc = current_cc();
    return cc == 70 || cc >= 80;
}

bool vhead_bitplane_enabled() {
    if (!vhead_bitplane_supported()) return false;
    const char* v = std::getenv("STRATA_VHEAD_BITPLANE");
    if (v != nullptr && v[0] != '\0') return std::atoi(v) != 0;
    return current_cc() == 70;
}

BitplaneHeadHandle* vhead_bitplane_create(int ggml_type, const void* d_weights, int n_in, int n_out) {
    if (!vhead_bitplane_supported() || d_weights == nullptr || n_in <= 0 || n_out <= 0 || (n_in & 255) != 0)
        return nullptr;
    if (ggml_type != kTypeQ8_0 && ggml_type != kTypeQ6_K) return nullptr;

    const int cc = current_cc();
    auto* h = new BitplaneHeadHandle();
    h->type = ggml_type;
    h->cc_mode = (cc == 70) ? 70 : 80;
    h->n_in = n_in;
    h->n_out = n_out;
    h->n_tiles = (n_out + 15) >> 4;
    h->n_tile_pairs = (h->n_tiles + 1) >> 1;
    h->nb32 = n_in >> 5;
    h->n_scales = (ggml_type == kTypeQ6_K) ? (n_in >> 4) : h->nb32;
    h->nsb = n_in >> 8;

    size_t hi_bytes = 0, lo_bytes = 0, d_bytes = 0, sc_bytes = 0, dall_bytes = 0;
    if (h->cc_mode == 70) {
        if (ggml_type == kTypeQ8_0) {
            hi_bytes = (size_t) h->n_tile_pairs * (size_t) h->nb32 * 32u * sizeof(uint4);
            lo_bytes = (size_t) h->n_tile_pairs * (size_t) h->nb32 * 32u * sizeof(uint4);
            d_bytes = (size_t) h->n_tile_pairs * (size_t) h->nb32 * 32u * sizeof(uint16_t);
        } else {
            hi_bytes = (size_t) h->n_tile_pairs * (size_t) h->n_scales * 32u * sizeof(uint2);
            lo_bytes = (size_t) h->n_tile_pairs * (size_t) h->n_scales * 32u * sizeof(uint32_t);
            sc_bytes = (size_t) h->n_tile_pairs * (size_t) h->n_scales * 32u * sizeof(int8_t);
            dall_bytes = (size_t) h->n_tile_pairs * (size_t) h->nsb * 32u * sizeof(uint16_t);
        }
    } else {
        if (ggml_type == kTypeQ8_0) {
            hi_bytes = (size_t) h->n_tiles * (size_t) h->nb32 * 32u * sizeof(uint2);
            lo_bytes = (size_t) n_out * (size_t) h->nb32 * sizeof(uint4);
            d_bytes = (size_t) h->n_tiles * (size_t) h->n_scales * 16u * sizeof(uint16_t);
        } else {
            hi_bytes = (size_t) h->n_tiles * (size_t) h->n_scales * 32u * sizeof(uint32_t);
            lo_bytes = (size_t) n_out * (size_t) h->n_scales * sizeof(uint32_t);
            d_bytes = (size_t) h->n_tiles * (size_t) h->n_scales * 16u * sizeof(uint16_t);
            sc_bytes = (size_t) n_out * (size_t) h->n_scales * sizeof(int8_t);
            dall_bytes = (size_t) n_out * (size_t) h->nsb * sizeof(uint16_t);
        }
    }

    const size_t rho_bytes = (h->cc_mode == 70) ? 0u : ((size_t) n_out * sizeof(float));
    const size_t scratch_xnorm = BitplaneHeadHandle::kMaxT * sizeof(float);
    const size_t scratch_xsum = (h->cc_mode == 70) ? 0u : ((size_t) BitplaneHeadHandle::kMaxT * (size_t) h->n_scales * sizeof(float));
    const size_t scratch_maxlo = BitplaneHeadHandle::kMaxT * sizeof(int32_t);
    const size_t scratch_tilehi = (size_t) BitplaneHeadHandle::kMaxT * (size_t) h->n_tiles * sizeof(float);
    const size_t scratch_tilelo = (size_t) BitplaneHeadHandle::kMaxT * (size_t) h->n_tiles * sizeof(float);
    const size_t scratch_thresh = BitplaneHeadHandle::kMaxT * sizeof(float);
    const size_t scratch_best = BitplaneHeadHandle::kMaxT * sizeof(unsigned long long);

    bool ok = cudaMalloc(&h->hi_tiled, hi_bytes) == cudaSuccess &&
              cudaMalloc(&h->lo_plane, lo_bytes) == cudaSuccess &&
              (d_bytes == 0 || cudaMalloc(&h->d_tiled, d_bytes) == cudaSuccess) &&
              (sc_bytes == 0 || cudaMalloc(&h->sc_tiled, sc_bytes) == cudaSuccess) &&
              (dall_bytes == 0 || cudaMalloc(&h->d_all_tiled, dall_bytes) == cudaSuccess) &&
              (rho_bytes == 0 || cudaMalloc(&h->rho, rho_bytes) == cudaSuccess) &&
              cudaMalloc(&h->d_x_norm, scratch_xnorm) == cudaSuccess &&
              (scratch_xsum == 0 || cudaMalloc(&h->d_x_sum, scratch_xsum) == cudaSuccess) &&
              cudaMalloc(&h->d_max_lo, scratch_maxlo) == cudaSuccess &&
              cudaMalloc(&h->d_tile_hi, scratch_tilehi) == cudaSuccess &&
              cudaMalloc(&h->d_tile_lo, scratch_tilelo) == cudaSuccess &&
              cudaMalloc(&h->d_thresh, scratch_thresh) == cudaSuccess &&
              cudaMalloc(&h->d_best, scratch_best) == cudaSuccess;
    if (!ok) {
        (void) cudaGetLastError();
        vhead_bitplane_destroy(h);
        return nullptr;
    }

    h->total_bytes = hi_bytes + lo_bytes + d_bytes + sc_bytes + dall_bytes + rho_bytes +
                     scratch_xnorm + scratch_xsum + scratch_maxlo +
                     scratch_tilehi + scratch_tilelo + scratch_thresh + scratch_best;

    if (h->cc_mode == 70) {
        const unsigned grid = (unsigned) ((h->n_tile_pairs + 7) / 8);
        if (ggml_type == kTypeQ8_0) {
            repack_q8_0_v70_kernel<<<grid, 256>>>(reinterpret_cast<const uint8_t*>(d_weights),
                                                  reinterpret_cast<uint4*>(h->hi_tiled),
                                                  reinterpret_cast<uint4*>(h->lo_plane),
                                                  h->d_tiled, h->rho, n_in, n_out);
        } else {
            repack_q6_k_v70_kernel<<<grid, 256>>>(reinterpret_cast<const uint8_t*>(d_weights),
                                                  reinterpret_cast<uint2*>(h->hi_tiled),
                                                  reinterpret_cast<uint32_t*>(h->lo_plane),
                                                  h->sc_tiled, h->d_all_tiled, h->rho, n_in, n_out);
        }
    } else {
        const unsigned grid = (unsigned) ((h->n_tiles + 7) / 8);
        if (ggml_type == kTypeQ8_0) {
            repack_q8_0_sm80_kernel<<<grid, 256>>>(reinterpret_cast<const uint8_t*>(d_weights),
                                                   reinterpret_cast<uint2*>(h->hi_tiled),
                                                   reinterpret_cast<uint4*>(h->lo_plane),
                                                   h->d_tiled, h->rho, n_in, n_out);
        } else {
            repack_q6_k_sm80_kernel<<<grid, 256>>>(reinterpret_cast<const uint8_t*>(d_weights),
                                                   reinterpret_cast<uint32_t*>(h->hi_tiled),
                                                   reinterpret_cast<uint32_t*>(h->lo_plane),
                                                   h->d_tiled, h->sc_tiled, h->d_all_tiled, h->rho, n_in, n_out);
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        vhead_bitplane_destroy(h);
        return nullptr;
    }
    return h;
}

void vhead_bitplane_destroy(BitplaneHeadHandle* h) {
    if (h == nullptr) return;
    if (h->hi_tiled) cudaFree(h->hi_tiled);
    if (h->lo_plane) cudaFree(h->lo_plane);
    if (h->d_tiled) cudaFree(h->d_tiled);
    if (h->sc_tiled) cudaFree(h->sc_tiled);
    if (h->d_all_tiled) cudaFree(h->d_all_tiled);
    if (h->rho) cudaFree(h->rho);
    if (h->d_x_norm) cudaFree(h->d_x_norm);
    if (h->d_x_sum) cudaFree(h->d_x_sum);
    if (h->d_max_lo) cudaFree(h->d_max_lo);
    if (h->d_tile_hi) cudaFree(h->d_tile_hi);
    if (h->d_tile_lo) cudaFree(h->d_tile_lo);
    if (h->d_thresh) cudaFree(h->d_thresh);
    if (h->d_best) cudaFree(h->d_best);
    if (h->d_cand_ids) cudaFree(h->d_cand_ids);
    if (h->d_cand_logits) cudaFree(h->d_cand_logits);
    if (h->d_cand_counts) cudaFree(h->d_cand_counts);
    if (h->d_cand_overflow) cudaFree(h->d_cand_overflow);
    if (h->h_cand_overflow) cudaFreeHost(h->h_cand_overflow);
    delete h;
}

size_t vhead_bitplane_bytes(const BitplaneHeadHandle* h) {
    return h ? h->total_bytes : 0;
}

bool vhead_bitplane_exact_logits(const BitplaneHeadHandle* h) {
    return h != nullptr && h->cc_mode == 70;
}

bool vhead_bitplane_argmax(BitplaneHeadHandle* h, const void* x_q8_1, int T, int32_t* out_tokens, void* stream,
                           uint32_t* out_nonfinite, int nonfinite_stride, int t_offset, float* out_logits) {
    if (h == nullptr || x_q8_1 == nullptr || out_tokens == nullptr || T < 1 || t_offset < 0 ||
        t_offset + T > BitplaneHeadHandle::kMaxT)
        return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const auto* xq = reinterpret_cast<const Q81Block*>(x_q8_1);
    const bool is_q6k = (h->type == kTypeQ6_K);

    if (h->cc_mode == 70) {
        const bool write_tiles = (out_logits == nullptr);
        vhead_clear_v70_kernel<<<1, (unsigned) T, 0, s>>>(
            h->d_best, reinterpret_cast<uint32_t*>(h->d_x_norm), h->d_max_lo, T, t_offset, write_tiles);
        if (out_logits != nullptr) {
            launch_exact_v70<true, true, false>(h, xq, T, t_offset, out_logits, h->d_best,
                                                reinterpret_cast<uint32_t*>(h->d_x_norm),
                                                h->d_max_lo, h->d_tile_hi, h->d_tile_lo, s);
        } else {
            launch_exact_v70<false, true, true>(h, xq, T, t_offset, nullptr, h->d_best,
                                                reinterpret_cast<uint32_t*>(h->d_x_norm),
                                                h->d_max_lo, h->d_tile_hi, h->d_tile_lo, s);
        }
        vhead_extract_tokens_kernel<<<1, (unsigned) T, 0, s>>>(h->d_best, h->d_x_norm, T, t_offset,
                                                               out_tokens, out_nonfinite, nonfinite_stride);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    if (is_q6k) {
        vhead_prep_kernel<true><<<(unsigned) T, 128, 0, s>>>(xq, h->n_in, T, t_offset, h->d_x_norm, h->d_x_sum,
                                                             h->d_max_lo, h->d_best);
    } else {
        vhead_prep_kernel<false><<<(unsigned) T, 128, 0, s>>>(xq, h->n_in, T, t_offset, h->d_x_norm, h->d_x_sum,
                                                              h->d_max_lo, h->d_best);
    }

    launch_stage1(h, xq, T, t_offset, s);

    const dim3 grid_s2(32u, (unsigned) T);
    if (is_q6k)
        vhead_stage2_tiles_kernel<true, 80><<<grid_s2, 128, 0, s>>>(
            h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq,
            h->n_in, h->n_out, h->n_tiles, t_offset, h->d_max_lo, h->d_tile_hi, h->d_best);
    else
        vhead_stage2_tiles_kernel<false, 80><<<grid_s2, 128, 0, s>>>(
            h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq,
            h->n_in, h->n_out, h->n_tiles, t_offset, h->d_max_lo, h->d_tile_hi, h->d_best);

    vhead_extract_tokens_kernel<<<1, (unsigned) T, 0, s>>>(h->d_best, h->d_x_norm, T, t_offset,
                                                           out_tokens, out_nonfinite, nonfinite_stride);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool vhead_bitplane_candidates(BitplaneHeadHandle* h, const void* x_q8_1, int T, int k_need, float min_p,
                               bool use_min_p_cut, bool reuse_stage1, void* stream,
                               const int32_t** out_ids, const float** out_logits,
                               const int32_t** out_counts, const uint8_t** out_dev_overflow,
                               const uint8_t** out_host_overflow, int* out_max_cand, int t_offset) {
    if (h == nullptr || x_q8_1 == nullptr || T < 1 || t_offset < 0 ||
        t_offset + T > BitplaneHeadHandle::kMaxT || !ensure_candidate_buffers(h))
        return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const auto* xq = reinterpret_cast<const Q81Block*>(x_q8_1);
    const bool is_q6k = (h->type == kTypeQ6_K);

    if (!reuse_stage1) {
        if (h->cc_mode == 70) {
            vhead_clear_v70_kernel<<<1, (unsigned) T, 0, s>>>(
                h->d_best, reinterpret_cast<uint32_t*>(h->d_x_norm), h->d_max_lo, T, t_offset, true);
            launch_exact_v70<false, false, true>(h, xq, T, t_offset, nullptr, h->d_best,
                                                 reinterpret_cast<uint32_t*>(h->d_x_norm),
                                                 h->d_max_lo, h->d_tile_hi, h->d_tile_lo, s);
        } else {
            if (is_q6k) {
                vhead_prep_kernel<true><<<(unsigned) T, 128, 0, s>>>(xq, h->n_in, T, t_offset, h->d_x_norm, h->d_x_sum,
                                                                     h->d_max_lo, h->d_best);
            } else {
                vhead_prep_kernel<false><<<(unsigned) T, 128, 0, s>>>(xq, h->n_in, T, t_offset, h->d_x_norm, h->d_x_sum,
                                                                      h->d_max_lo, h->d_best);
            }
            launch_stage1(h, xq, T, t_offset, s);
        }
    }

    vhead_sample_thresh_kernel<<<(unsigned) T, 256, 0, s>>>(
        h->d_tile_lo, h->d_max_lo, h->d_x_norm, h->n_tiles, T, t_offset, k_need, min_p, use_min_p_cut,
        h->d_thresh, h->d_cand_counts, h->d_cand_overflow);

    const dim3 grid_s2(32u, (unsigned) T);
    if (h->cc_mode == 70) {
        if (is_q6k)
            vhead_stage2_candidates_kernel<true, 70><<<grid_s2, 128, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq,
                h->n_in, h->n_out, h->n_tiles, t_offset, BitplaneHeadHandle::kMaxCand,
                h->d_tile_hi, h->d_thresh, h->d_cand_ids, h->d_cand_logits, h->d_cand_counts, h->d_cand_overflow);
        else
            vhead_stage2_candidates_kernel<false, 70><<<grid_s2, 128, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq,
                h->n_in, h->n_out, h->n_tiles, t_offset, BitplaneHeadHandle::kMaxCand,
                h->d_tile_hi, h->d_thresh, h->d_cand_ids, h->d_cand_logits, h->d_cand_counts, h->d_cand_overflow);
    } else {
        if (is_q6k)
            vhead_stage2_candidates_kernel<true, 80><<<grid_s2, 128, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq,
                h->n_in, h->n_out, h->n_tiles, t_offset, BitplaneHeadHandle::kMaxCand,
                h->d_tile_hi, h->d_thresh, h->d_cand_ids, h->d_cand_logits, h->d_cand_counts, h->d_cand_overflow);
        else
            vhead_stage2_candidates_kernel<false, 80><<<grid_s2, 128, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq,
                h->n_in, h->n_out, h->n_tiles, t_offset, BitplaneHeadHandle::kMaxCand,
                h->d_tile_hi, h->d_thresh, h->d_cand_ids, h->d_cand_logits, h->d_cand_counts, h->d_cand_overflow);
    }

    if (out_host_overflow != nullptr) {
        vhead_copy_overflow_kernel<<<1, (unsigned) T, 0, s>>>(h->d_cand_overflow, h->h_cand_overflow, T);
    }

    if (out_ids) *out_ids = h->d_cand_ids;
    if (out_logits) *out_logits = h->d_cand_logits;
    if (out_counts) *out_counts = h->d_cand_counts;
    if (out_dev_overflow) *out_dev_overflow = h->d_cand_overflow;
    if (out_host_overflow) *out_host_overflow = h->h_cand_overflow;
    if (out_max_cand) *out_max_cand = BitplaneHeadHandle::kMaxCand;
    return cudaPeekAtLastError() == cudaSuccess;
}

bool vhead_bitplane_logits(const BitplaneHeadHandle* h, const void* x_q8_1, int T, float* out_logits, void* stream) {
    if (h == nullptr || x_q8_1 == nullptr || out_logits == nullptr || T < 1 || T > 8) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const auto* xq = reinterpret_cast<const Q81Block*>(x_q8_1);
    if (h->cc_mode == 70) {
        launch_exact_v70<true, false, false>(h, xq, T, 0, out_logits, nullptr, nullptr, nullptr, nullptr, nullptr, s);
        return cudaPeekAtLastError() == cudaSuccess;
    }
    const unsigned grid = (unsigned) ((h->n_out + 3) / 4);
    const bool is_q6k = (h->type == kTypeQ6_K);
    if (is_q6k)
        vhead_full_logits_kernel<true, 80><<<grid, 128, 0, s>>>(
            h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq, h->n_in, h->n_out, T, out_logits);
    else
        vhead_full_logits_kernel<false, 80><<<grid, 128, 0, s>>>(
            h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled, xq, h->n_in, h->n_out, T, out_logits);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool vhead_bitplane_gather_raw(const BitplaneHeadHandle* h, const int32_t* d_rows, int64_t n_rows, void* d_dst,
                               void* stream) {
    if (h == nullptr || d_rows == nullptr || d_dst == nullptr || n_rows < 0) return false;
    if (n_rows == 0) return true;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const unsigned grid = (unsigned) n_rows;
    const bool is_q6k = (h->type == kTypeQ6_K);
    if (h->cc_mode == 70) {
        if (is_q6k)
            vhead_gather_raw_kernel<true, 70><<<grid, 64, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled,
                d_rows, n_rows, h->n_in, reinterpret_cast<uint8_t*>(d_dst));
        else
            vhead_gather_raw_kernel<false, 70><<<grid, 128, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled,
                d_rows, n_rows, h->n_in, reinterpret_cast<uint8_t*>(d_dst));
    } else {
        if (is_q6k)
            vhead_gather_raw_kernel<true, 80><<<grid, 64, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled,
                d_rows, n_rows, h->n_in, reinterpret_cast<uint8_t*>(d_dst));
        else
            vhead_gather_raw_kernel<false, 80><<<grid, 128, 0, s>>>(
                h->hi_tiled, h->lo_plane, h->d_tiled, h->sc_tiled, h->d_all_tiled,
                d_rows, n_rows, h->n_in, reinterpret_cast<uint8_t*>(d_dst));
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::kernels
