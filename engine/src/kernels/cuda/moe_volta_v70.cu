// src/kernels/cuda/moe_volta_v70.cu - Clean-room MIT Volta (sm_70) MoE decode kernels (PR 5.3).
//
// Step 5.3a (STRATA_EXP_BANK_V70):
//   - IQ2_S (22): splits the 8 KB iq2s_grid into two separate 32-bit shared-memory planes
//     (sgrid_lo[1024], sgrid_hi[1024]) so every 32-bit lookup spreads across all 32 Volta SMEM banks.
//   - IQ3_XXS (18): stages the 1 KB iq3xxs_grid[256] in SMEM with ZERO sign-mask SMEM tables.
//   - Inline PTX prmt.b32 (0xBA98 sign-extend) register-only sign negation: every byte of iq2s_grid (>=8)
//     and iq3xxs_grid (>=4) is non-zero, so (g ^ m) + (m & 0x01010101u) is exact 4x int8 two's-complement
//     negation with zero carry and zero __vcmpne4/__vsub4 emulation.
//   - int4 plane-staged q8_1 activations (2x LDS.128 + 1x pre-converted float LDS.32, conflict-free).
//
// Step 5.3b (STRATA_MOE_PERSIST_V70):
//   - Single strictly-resident persistent CTA grid (grid = num_sms * bpsm from
//     cudaOccupancyMaxActiveBlocksPerMultiprocessor) fusing Gate/Up + SwiGLU/Q8_1 + Down (IQ4_NL & Q2_0).
//   - Loads sgrid_lo/hi ONCE per resident CTA instead of 20 * cap_groups times per layer.
//   - Priority Gate/Up drain before any Down task + PTX release/acquire epoch-tagged ready flags
//     (zero volatile, zero cudaMemsetAsync, self-zeroing state keyed per (dev, stream), CUDA-Graph safe).
//   - Unioned SMEM between Phase 1 (Gate/Up) and Phase 2 (Down): 29,440 B for IQ4_NL (3 CTA/SM in 96 KB)
//     and 20,736 B for Q2_0 (3 CTA/SM in 64 KB).
//   - Down IQ4_NL reuses PR 5.2's burst uint4 staging + hybrid dp4a (n_e<=4) / mma.m8n8k4 (n_e>=5) math
//     with BR=64 rows/tile (8 warps = 256 threads), 100% bit-for-bit identical to iq4nl_down_v70_kernel<20>.
//   - Down Q2_0 uses int4 plane-staged hq + Split<42> exact reduction tree, 100% bit-for-bit identical.
#include "strata/kernels/moe_volta_v70.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::kernels {

#if !defined(__HIPCC__)

namespace {

constexpr int kV70Nt = 4;    // Gate/Up entries staged in SMEM per pass
constexpr int kV70Xb = 80;   // Gate/Up q8_1 blocks per row (n_embd = 2560)
constexpr int kV70Hb = 20;   // Down q8_1 blocks per row (n_ff = 640)
constexpr int kMaxPersistGroups = 256;

struct alignas(4) Q81Raw {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Raw) == 36, "Q81Raw must be 36 bytes");

// ============================================================================
// PTX memory ordering helpers (sm_70 PTX ISA 6.0+, .gpu scope, zero volatile)
// ============================================================================
__device__ __forceinline__ uint32_t ld_acquire_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t atomic_add_acq_rel_gpu(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ uint32_t atomic_add_relaxed_gpu(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

struct alignas(128) PersistState {
    uint32_t ticket_gu;
    uint32_t ticket_down;
    uint32_t cta_done;
    uint32_t epoch;   // zero-initialized in .bss; launch k uses my_epoch = epoch + 1 (always >= 1)
    uint32_t gu_done_cnt[kMaxPersistGroups];
    uint32_t ready_epoch[kMaxPersistGroups];
};

constexpr int kMaxStreamSlots = 16;
__device__ PersistState g_dev_state[kMaxStreamSlots];

// ============================================================================
// Step 5.3a helpers: bank-aware IQ2_S / IQ3_XXS + PRMT sign-masks + int4 s_u
// ============================================================================
__device__ __forceinline__ uint32_t load_u32_b2(const void* x, int i32) {
    const uint16_t* x16 = reinterpret_cast<const uint16_t*>(x);
    return (uint32_t) x16[2 * i32 + 0] | ((uint32_t) x16[2 * i32 + 1] << 16);
}

__device__ __forceinline__ int prmt_apply_sign4(uint32_t g, uint32_t nib4) {
    const uint32_t t = nib4 * 0x10204080u;
    uint32_t m;
    asm("prmt.b32 %0, %1, 0, 0xBA98;" : "=r"(m) : "r"(t));
    return (int) ((g ^ m) + (m & 0x01010101u));
}

template <int XB>
__device__ __forceinline__ void stage_q8_1_entry(int4* __restrict__ s_u, float* __restrict__ s_ds,
                                                 int c, const block_q8_1* __restrict__ src, int tid) {
    const int* w32 = reinterpret_cast<const int*>(src);
    const int k = tid >> 1;
    const int h = tid & 1;
    if (k < XB) {
        const int* p = w32 + k * 9 + 1 + 4 * h;
        s_u[(c * 2 + h) * XB + k] = make_int4(p[0], p[1], p[2], p[3]);
    }
    if (tid < XB) {
        const int v = w32[tid * 9];
        half2 h2;
        std::memcpy(&h2, &v, 4);
        s_ds[c * XB + tid] = __low2float(h2);
    }
}

struct Iq2sW {
    int g[8];
    int ls0, ls1;
    float dw;
};

__device__ __forceinline__ Iq2sW load_iq2s_v70(const uint8_t* __restrict__ row, int kbx, int q,
                                               const uint32_t* __restrict__ sgrid_lo,
                                               const uint32_t* __restrict__ sgrid_hi) {
    const block_iq2_s* bq2 = reinterpret_cast<const block_iq2_s*>(row) + kbx;
    const uint32_t qs_packed = load_u32_b2(bq2->qs, q);
    const uint32_t signs_packed = load_u32_b2(bq2->qs, QK_K / 32 + q);
    const uint32_t qh = bq2->qh[q];
    const uint32_t sc = bq2->scales[q];
    const float dw = __half2float(bq2->d);

    Iq2sW out;
    out.ls0 = (int) (sc & 0x0Fu);
    out.ls1 = (int) (sc >> 4);
    out.dw = dw;

#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t qs_b = (qs_packed >> (8 * l)) & 0xFFu;
        const uint32_t idx = qs_b | ((qh << (8 - 2 * l)) & 0x300u);
        const uint32_t gl = sgrid_lo[idx];
        const uint32_t gh = sgrid_hi[idx];
        const uint32_t sg = (signs_packed >> (8 * l)) & 0xFFu;
        out.g[2 * l + 0] = prmt_apply_sign4(gl, sg & 0x0Fu);
        out.g[2 * l + 1] = prmt_apply_sign4(gh, sg >> 4);
    }
    return out;
}

// Stream 4+4 ints directly from s_u (2x LDS.128) to save 4 live registers (eliminates STACK:8 spill!).
__device__ __forceinline__ float apply_iq2s_staged(const Iq2sW& r, const int4* __restrict__ s_u,
                                                   int c, int k, float d8) {
    const int4 a = s_u[(c * 2 + 0) * kV70Xb + k];
    int sumi0 = 0;
    sumi0 = __dp4a(r.g[0], a.x, sumi0);
    sumi0 = __dp4a(r.g[1], a.y, sumi0);
    sumi0 = __dp4a(r.g[2], a.z, sumi0);
    sumi0 = __dp4a(r.g[3], a.w, sumi0);
    const int4 b = s_u[(c * 2 + 1) * kV70Xb + k];
    int sumi1 = 0;
    sumi1 = __dp4a(r.g[4], b.x, sumi1);
    sumi1 = __dp4a(r.g[5], b.y, sumi1);
    sumi1 = __dp4a(r.g[6], b.z, sumi1);
    sumi1 = __dp4a(r.g[7], b.w, sumi1);
    const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
    return (r.dw * d8) * sumi;
}

struct Iq3xxsW {
    int g[8];
    int ls;
    float dw;
};

__device__ __forceinline__ Iq3xxsW load_iq3xxs_v70(const uint8_t* __restrict__ row, int kbx, int q,
                                                   const uint32_t* __restrict__ sgrid) {
    const block_iq3_xxs* bq3 = reinterpret_cast<const block_iq3_xxs*>(row) + kbx;
    const uint32_t q3_0 = load_u32_b2(bq3->qs, 2 * q + 0);
    const uint32_t q3_1 = load_u32_b2(bq3->qs, 2 * q + 1);
    const uint32_t aux32 = load_u32_b2(bq3->qs, QK_K / 16 + q);
    const float dw = __half2float(bq3->d);

    Iq3xxsW out;
    out.ls = (int) (aux32 >> 28);
    out.dw = dw;

#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t pair = (l < 2) ? (q3_0 >> (16 * l)) : (q3_1 >> (16 * (l - 2)));
        const uint32_t g0 = sgrid[pair & 0xFFu];
        const uint32_t g1 = sgrid[(pair >> 8) & 0xFFu];
        const uint32_t v = (aux32 >> (7 * l)) & 0x7Fu;
        const uint32_t hi4 = (v >> 4) | ((__popc(v) & 1u) << 3);
        out.g[2 * l + 0] = prmt_apply_sign4(g0, v & 0x0Fu);
        out.g[2 * l + 1] = prmt_apply_sign4(g1, hi4);
    }
    return out;
}

__device__ __forceinline__ float apply_iq3xxs_staged(const Iq3xxsW& r, const int4* __restrict__ s_u,
                                                     int c, int k, float d8) {
    const int4 a = s_u[(c * 2 + 0) * kV70Xb + k];
    int sumi = 0;
    sumi = __dp4a(r.g[0], a.x, sumi);
    sumi = __dp4a(r.g[1], a.y, sumi);
    sumi = __dp4a(r.g[2], a.z, sumi);
    sumi = __dp4a(r.g[3], a.w, sumi);
    const int4 b = s_u[(c * 2 + 1) * kV70Xb + k];
    sumi = __dp4a(r.g[4], b.x, sumi);
    sumi = __dp4a(r.g[5], b.y, sumi);
    sumi = __dp4a(r.g[6], b.z, sumi);
    sumi = __dp4a(r.g[7], b.w, sumi);
    sumi = (r.ls * sumi + sumi / 2) / 2;
    return (r.dw * d8) * sumi;
}

template <int TG, int NTC>
__device__ __forceinline__ void v70_row_dot2_80_sub16(const uint8_t* __restrict__ ra,
                                                      const uint8_t* __restrict__ rb,
                                                      const int4* __restrict__ s_u,
                                                      const float* __restrict__ s_ds,
                                                      int t16,
                                                      float (&sa)[NTC], float (&sb)[NTC],
                                                      const uint32_t* __restrict__ sgrid_lo,
                                                      const uint32_t* __restrict__ sgrid_hi) {
    const int q = t16 & 7;
#pragma unroll
    for (int c = 0; c < NTC; ++c) { sa[c] = 0.0f; sb[c] = 0.0f; }

    auto step = [&](int k, float (&acc_a)[NTC], float (&acc_b)[NTC]) {
        const int kbx = k >> 3;
        if constexpr (TG == 22) {
            {
                const Iq2sW wa = load_iq2s_v70(ra, kbx, q, sgrid_lo, sgrid_hi);
#pragma unroll
                for (int c = 0; c < NTC; ++c) {
                    acc_a[c] += apply_iq2s_staged(wa, s_u, c, k, s_ds[c * kV70Xb + k]);
                }
            }
            {
                const Iq2sW wb = load_iq2s_v70(rb, kbx, q, sgrid_lo, sgrid_hi);
#pragma unroll
                for (int c = 0; c < NTC; ++c) {
                    acc_b[c] += apply_iq2s_staged(wb, s_u, c, k, s_ds[c * kV70Xb + k]);
                }
            }
        } else {
            {
                const Iq3xxsW wa = load_iq3xxs_v70(ra, kbx, q, sgrid_lo);
#pragma unroll
                for (int c = 0; c < NTC; ++c) {
                    acc_a[c] += apply_iq3xxs_staged(wa, s_u, c, k, s_ds[c * kV70Xb + k]);
                }
            }
            {
                const Iq3xxsW wb = load_iq3xxs_v70(rb, kbx, q, sgrid_lo);
#pragma unroll
                for (int c = 0; c < NTC; ++c) {
                    acc_b[c] += apply_iq3xxs_staged(wb, s_u, c, k, s_ds[c * kV70Xb + k]);
                }
            }
        }
    };

#pragma unroll 1
    for (int j = 0; j < 3; ++j) step(t16 + 32 * j, sa, sb);

    float a16[NTC], b16[NTC];
#pragma unroll
    for (int c = 0; c < NTC; ++c) { a16[c] = 0.0f; b16[c] = 0.0f; }

#pragma unroll 1
    for (int j = 0; j < 2; ++j) step(t16 + 16 + 32 * j, a16, b16);

#pragma unroll
    for (int c = 0; c < NTC; ++c) {
        sa[c] = __fadd_rn(sa[c], a16[c]);
        sb[c] = __fadd_rn(sb[c], b16[c]);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            sa[c] += __shfl_xor_sync(0xffffffffu, sa[c], o);
            sb[c] += __shfl_xor_sync(0xffffffffu, sb[c], o);
        }
    }
}

template <int TG, int NTC>
__device__ __forceinline__ void v70_row_dot1_80_sub16(const uint8_t* __restrict__ ra,
                                                      const int4* __restrict__ s_u,
                                                      const float* __restrict__ s_ds,
                                                      int t16,
                                                      float (&sa)[NTC],
                                                      const uint32_t* __restrict__ sgrid_lo,
                                                      const uint32_t* __restrict__ sgrid_hi) {
    const int q = t16 & 7;
#pragma unroll
    for (int c = 0; c < NTC; ++c) sa[c] = 0.0f;

    auto step = [&](int k, float (&acc)[NTC]) {
        const int kbx = k >> 3;
        if constexpr (TG == 22) {
            const Iq2sW w = load_iq2s_v70(ra, kbx, q, sgrid_lo, sgrid_hi);
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c] += apply_iq2s_staged(w, s_u, c, k, s_ds[c * kV70Xb + k]);
            }
        } else {
            const Iq3xxsW w = load_iq3xxs_v70(ra, kbx, q, sgrid_lo);
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c] += apply_iq3xxs_staged(w, s_u, c, k, s_ds[c * kV70Xb + k]);
            }
        }
    };

#pragma unroll 1
    for (int j = 0; j < 3; ++j) step(t16 + 32 * j, sa);

    float a16[NTC];
#pragma unroll
    for (int c = 0; c < NTC; ++c) a16[c] = 0.0f;

#pragma unroll 1
    for (int j = 0; j < 2; ++j) step(t16 + 16 + 32 * j, a16);

#pragma unroll
    for (int c = 0; c < NTC; ++c) {
        sa[c] = __fadd_rn(sa[c], a16[c]);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) sa[c] += __shfl_xor_sync(0xffffffffu, sa[c], o);
    }
}

// ============================================================================
// Step 5.3a Kernels (standalone fused & unfused Gate/Up)
// ============================================================================
template <int TG>
__global__ void __launch_bounds__(256, 3) moe_gu_fused_v70_kernel(
    const unsigned long long* __restrict__ grp_ptr,
    const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups,
    const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq,
    NativeExpertLayout L,
    block_q8_1* __restrict__ hq) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const int ng = *n_groups;
    if ((int) blockIdx.y >= ng) return;

    __shared__ alignas(16) int4 s_u[kV70Nt * 2 * kV70Xb];
    __shared__ float s_ds[kV70Nt * kV70Xb];
    __shared__ float res[kV70Nt][64];
    __shared__ uint32_t sgrid_lo[(TG == 22) ? 1024 : 256];
    __shared__ uint32_t sgrid_hi[(TG == 22) ? 1024 : 1];

    const int tid = threadIdx.x;
    if constexpr (TG == 22) {
        const uint2* gsrc = reinterpret_cast<const uint2*>(iq2s_grid);
#pragma unroll
        for (int i = tid; i < 1024; i += 256) {
            const uint2 v = gsrc[i];
            sgrid_lo[i] = v.x;
            sgrid_hi[i] = v.y;
        }
    } else {
        sgrid_lo[tid] = iq3xxs_grid[tid];
    }

    const int sub = tid >> 4;
    const int t16 = tid & 15;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int hb = (int) (L.n_ff >> 5);
    const int ch = (int) blockIdx.x;
    const int r0 = ch * 32;

    for (int g = (int) blockIdx.y; g < ng; g += (int) gridDim.y) {
        const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[g]);
        const int e0 = grp_start[g];
        const int e1 = grp_start[g + 1];

        for (int c0 = e0; c0 < e1; c0 += kV70Nt) {
            const int cn = min(kV70Nt, e1 - c0);
            if (g > (int) blockIdx.y || c0 > e0) __syncthreads();
            for (int j = 0; j < cn; ++j) {
                stage_q8_1_entry<kV70Xb>(s_u, s_ds, j, xq + (size_t) ent_tok[c0 + j] * kV70Xb, tid);
            }
            __syncthreads();

#pragma unroll 1
            for (int p = 0; p < 2; ++p) {
                const int i0 = p * 16 + sub;
                const uint8_t* w0 = blob + (size_t) (r0 + i0) * L.gu_row;
                const uint8_t* w1 = blob + L.up_off + (size_t) (r0 + i0) * L.gu_row;
                auto run = [&](auto ntc) {
                    constexpr int NTC = decltype(ntc)::value;
                    if constexpr (NTC <= 3) {
                        float sa[NTC], sb[NTC];
                        v70_row_dot2_80_sub16<TG, NTC>(w0, w1, s_u, s_ds, t16, sa, sb, sgrid_lo, sgrid_hi);
                        if (t16 == 0) {
#pragma unroll
                            for (int j = 0; j < NTC; ++j) { res[j][i0] = sa[j]; res[j][32 + i0] = sb[j]; }
                        }
                    } else {
                        float srow[NTC];
                        v70_row_dot1_80_sub16<TG, NTC>(w0, s_u, s_ds, t16, srow, sgrid_lo, sgrid_hi);
                        if (t16 == 0) {
#pragma unroll
                            for (int j = 0; j < NTC; ++j) res[j][i0] = srow[j];
                        }
                        v70_row_dot1_80_sub16<TG, NTC>(w1, s_u, s_ds, t16, srow, sgrid_lo, sgrid_hi);
                        if (t16 == 0) {
#pragma unroll
                            for (int j = 0; j < NTC; ++j) res[j][32 + i0] = srow[j];
                        }
                    }
                };
                switch (cn) {
                    case 1: run(std::integral_constant<int, 1>{}); break;
                    case 2: run(std::integral_constant<int, 2>{}); break;
                    case 3: run(std::integral_constant<int, 3>{}); break;
                    default: run(std::integral_constant<int, 4>{}); break;
                }
            }

            __syncthreads();
            if (warp < cn) {
                const float gg = res[warp][lane];
                const float uu = res[warp][32 + lane];
                const float xi = (gg / (1.0f + __expf(-gg))) * uu;
                float amax = fabsf(xi), sum = xi;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                    sum += __shfl_xor_sync(0xffffffffu, sum, o);
                }
                const float d = q8_1_finite(amax / 127.0f);
                const int8_t qv = q8_1_quant(xi, d, amax);
                block_q8_1* y = hq + (size_t) (c0 + warp) * hb + ch;
                y->qs[lane] = qv;
                if (lane == 0) y->ds = q8_1_ds(d, sum);
            }
        }
    }
#else
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_tok; (void) xq; (void) L; (void) hq;
#endif
}

template <int TG>
__global__ void __launch_bounds__(256, 3) moe_gu_unfused_v70_kernel(
    const unsigned long long* __restrict__ grp_ptr,
    const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups,
    const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq,
    uint64_t up_off, uint32_t gu_row, int n_ff,
    float* __restrict__ gate,
    float* __restrict__ up,
    int rb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const int ng = *n_groups;
    if ((int) blockIdx.y >= ng) return;

    __shared__ alignas(16) int4 s_u[kV70Nt * 2 * kV70Xb];
    __shared__ float s_ds[kV70Nt * kV70Xb];
    __shared__ uint32_t sgrid_lo[(TG == 22) ? 1024 : 256];
    __shared__ uint32_t sgrid_hi[(TG == 22) ? 1024 : 1];

    const int tid = threadIdx.x;
    if constexpr (TG == 22) {
        const uint2* gsrc = reinterpret_cast<const uint2*>(iq2s_grid);
#pragma unroll
        for (int i = tid; i < 1024; i += 256) {
            const uint2 v = gsrc[i];
            sgrid_lo[i] = v.x;
            sgrid_hi[i] = v.y;
        }
    } else {
        sgrid_lo[tid] = iq3xxs_grid[tid];
    }

    const int sub = tid >> 4;
    const int t16 = tid & 15;
    const int nrow = 2 * n_ff;
    const int row_base = (int) blockIdx.x * rb;

    for (int g = (int) blockIdx.y; g < ng; g += (int) gridDim.y) {
        const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[g]);
        const int e0 = grp_start[g];
        const int e1 = grp_start[g + 1];
        for (int c0 = e0; c0 < e1; c0 += kV70Nt) {
            const int cn = min(kV70Nt, e1 - c0);
            if (g > (int) blockIdx.y || c0 > e0) __syncthreads();
            for (int j = 0; j < cn; ++j) {
                stage_q8_1_entry<kV70Xb>(s_u, s_ds, j, xq + (size_t) ent_tok[c0 + j] * kV70Xb, tid);
            }
            __syncthreads();

            for (int p = 0; p * 16 < rb; ++p) {
                const int row0 = row_base + p * 16 + sub;
                if (row0 >= nrow) break;
                const bool is_up = row0 >= n_ff;
                const int r0 = is_up ? row0 - n_ff : row0;
                const uint8_t* wr = blob + (is_up ? up_off : 0) + (size_t) r0 * gu_row;
                float* dst = is_up ? up : gate;
                auto run = [&](auto ntc) {
                    constexpr int NTC = decltype(ntc)::value;
                    float sa[NTC];
                    v70_row_dot1_80_sub16<TG, NTC>(wr, s_u, s_ds, t16, sa, sgrid_lo, sgrid_hi);
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        if (t16 == j) dst[(size_t) (c0 + j) * n_ff + r0] = sa[j];
                    }
                };
                switch (cn) {
                    case 1: run(std::integral_constant<int, 1>{}); break;
                    case 2: run(std::integral_constant<int, 2>{}); break;
                    case 3: run(std::integral_constant<int, 3>{}); break;
                    default: run(std::integral_constant<int, 4>{}); break;
                }
            }
        }
    }
#else
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_tok;
    (void) xq; (void) up_off; (void) gu_row; (void) n_ff; (void) gate; (void) up; (void) rb;
#endif
}

// ============================================================================
// Step 5.3b helpers: IQ4_NL (PR 5.2 exact math) & Q2_0 Down for Persistent MoE
// ============================================================================
constexpr uint32_t kIq4Sig0 = 0xBFAD9881u;
constexpr uint32_t kIq4Sig1 = 0xF6EADDCFu;
constexpr uint32_t kIq4Sig2 = 0x26190D01u;
constexpr uint32_t kIq4Sig3 = 0x71594535u;

constexpr uint32_t kIq4Lut0 = 0x3F2D1801u;
constexpr uint32_t kIq4Lut1 = 0x766A5D4Fu;
constexpr uint32_t kIq4Lut2 = 0xA6998D81u;
constexpr uint32_t kIq4Lut3 = 0xF1D9C5B5u;

__device__ __forceinline__ void iq4_decode_4bytes_signed(uint32_t q4, uint32_t& lo, uint32_t& hi) {
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    const uint32_t l0 = __byte_perm(kIq4Sig0, kIq4Sig1, q4);
    const uint32_t h0 = __byte_perm(kIq4Sig2, kIq4Sig3, q4);
    const uint32_t t0 = __byte_perm(l0, h0, sel);
    const uint32_t q4h = q4 >> 16;
    const uint32_t l1 = __byte_perm(kIq4Sig0, kIq4Sig1, q4h);
    const uint32_t h1 = __byte_perm(kIq4Sig2, kIq4Sig3, q4h);
    const uint32_t t1 = __byte_perm(l1, h1, sel >> 16);
    lo = __byte_perm(t0, t1, 0x6420u);
    hi = __byte_perm(t0, t1, 0x7531u);
}

__device__ __forceinline__ void iq4_decode_4bytes_bias128(uint32_t q4, uint32_t& lo, uint32_t& hi) {
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    const uint32_t l0 = __byte_perm(kIq4Lut0, kIq4Lut1, q4);
    const uint32_t h0 = __byte_perm(kIq4Lut2, kIq4Lut3, q4);
    const uint32_t t0 = __byte_perm(l0, h0, sel);
    const uint32_t q4h = q4 >> 16;
    const uint32_t l1 = __byte_perm(kIq4Lut0, kIq4Lut1, q4h);
    const uint32_t h1 = __byte_perm(kIq4Lut2, kIq4Lut3, q4h);
    const uint32_t t1 = __byte_perm(l1, h1, sel >> 16);
    lo = __byte_perm(t0, t1, 0x6420u);
    hi = __byte_perm(t0, t1, 0x7531u);
}

__device__ __forceinline__ uint2 u8x4_bias128_to_f16x4(uint32_t c) {
    const uint32_t lo_bits = __byte_perm(c, 0x64646464u, 0x5140u);
    const uint32_t hi_bits = __byte_perm(c, 0x64646464u, 0x7362u);
    const half2 bias = __float2half2_rn(-1152.0f);
    half2 h0, h1;
    std::memcpy(&h0, &lo_bits, 4);
    std::memcpy(&h1, &hi_bits, 4);
    h0 = __hadd2(h0, bias);
    h1 = __hadd2(h1, bias);
    uint2 out;
    std::memcpy(&out.x, &h0, 4);
    std::memcpy(&out.y, &h1, 4);
    return out;
}

__device__ __forceinline__ void mma_m8n8k4(float (&c)[8], uint2 a, uint2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
          "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
#else
    (void) c; (void) a; (void) b;
#endif
}

__device__ __forceinline__ void load_iq4_block(const uint32_t* __restrict__ sw_row, int bi, int odd,
                                               uint16_t& d_bits, uint32_t& q0, uint32_t& q1,
                                               uint32_t& q2, uint32_t& q3) {
    const uint32_t* const wp = sw_row + (bi >> 1) * 9;
    if (!odd) {
        const uint32_t w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3], w4 = wp[4];
        d_bits = (uint16_t) (w0 & 0xFFFFu);
        q0 = __funnelshift_r(w0, w1, 16);
        q1 = __funnelshift_r(w1, w2, 16);
        q2 = __funnelshift_r(w2, w3, 16);
        q3 = __funnelshift_r(w3, w4, 16);
    } else {
        d_bits = (uint16_t) (wp[4] >> 16);
        q0 = wp[5]; q1 = wp[6]; q2 = wp[7]; q3 = wp[8];
    }
}

// Two-batch burst uint4 staging (3x uint4/thread per batch = 12 regs instead of 24 -> keeps REG<=80, STACK:0!)
__device__ __forceinline__ void stage_iq4_w64_burst(const uint8_t* __restrict__ w_base,
                                                    uint32_t* __restrict__ s_w, int tid) {
    // 64 rows * 90 uint32 = 5760 uint32 = 1440 uint4 (64*360 = 23040 B, always 16B aligned when down_off%16==0)
    constexpr int N_U4 = 1440;
    if ((((uintptr_t) w_base) & 15u) == 0u) {
        const uint4* const src = reinterpret_cast<const uint4*>(w_base);
        uint4* const dst = reinterpret_cast<uint4*>(s_w);
        {
            uint4 b0 = __ldg(src + tid);
            uint4 b1 = __ldg(src + tid + 256);
            uint4 b2 = __ldg(src + tid + 512);
            dst[tid] = b0;
            dst[tid + 256] = b1;
            dst[tid + 512] = b2;
        }
        {
            uint4 b0 = __ldg(src + tid + 768);
            uint4 b1 = __ldg(src + tid + 1024);
            uint4 b2 = (tid + 1280 < N_U4) ? __ldg(src + tid + 1280) : make_uint4(0, 0, 0, 0);
            dst[tid + 768] = b0;
            dst[tid + 1024] = b1;
            if (tid + 1280 < N_U4) dst[tid + 1280] = b2;
        }
    } else {
        const uint32_t* const src = reinterpret_cast<const uint32_t*>(w_base);
        for (int i = tid; i < 64 * 90; i += 256) s_w[i] = __ldg(src + i);
    }
}

__device__ __forceinline__ void iq4_scale_accum(float (&acc)[8], const float (&d)[8],
                                                float dw, int lane, int c0_col,
                                                const float* __restrict__ xd_bi) {
    const float dw_other = __shfl_xor_sync(0xffffffffu, dw, 2);
    const float dw0 = (lane & 2) ? dw_other : dw;
    const float dw1 = (lane & 2) ? dw : dw_other;
    const float dx0 = xd_bi[c0_col + 0];
    const float dx1 = xd_bi[c0_col + 1];
    const float dx2 = xd_bi[c0_col + 4];
    const float dx3 = xd_bi[c0_col + 5];
    acc[0] = fmaf(dw0 * dx0, d[0], acc[0]);
    acc[1] = fmaf(dw0 * dx1, d[1], acc[1]);
    acc[2] = fmaf(dw1 * dx0, d[2], acc[2]);
    acc[3] = fmaf(dw1 * dx1, d[3], acc[3]);
    acc[4] = fmaf(dw0 * dx2, d[4], acc[4]);
    acc[5] = fmaf(dw0 * dx3, d[5], acc[5]);
    acc[6] = fmaf(dw1 * dx2, d[6], acc[6]);
    acc[7] = fmaf(dw1 * dx3, d[7], acc[7]);
}

__device__ __forceinline__ void iq4_store_tile8(const float (&acc)[8], int r_base, int nrow,
                                                int gg, int hh, int cn, int c0,
                                                const int32_t* __restrict__ ent_dst,
                                                float* __restrict__ out) {
    const int r0 = r_base + (gg & 1) + 4 * hh;
    const int r1 = r0 + 2;
    const int c0_col = gg & 2;
    if (c0_col + 0 < cn) {
        const size_t base = (size_t) ent_dst[c0 + c0_col + 0] * (size_t) nrow;
        out[base + r0] = acc[0]; out[base + r1] = acc[2];
    }
    if (c0_col + 1 < cn) {
        const size_t base = (size_t) ent_dst[c0 + c0_col + 1] * (size_t) nrow;
        out[base + r0] = acc[1]; out[base + r1] = acc[3];
    }
    if (c0_col + 4 < cn) {
        const size_t base = (size_t) ent_dst[c0 + c0_col + 4] * (size_t) nrow;
        out[base + r0] = acc[4]; out[base + r1] = acc[6];
    }
    if (c0_col + 5 < cn) {
        const size_t base = (size_t) ent_dst[c0 + c0_col + 5] * (size_t) nrow;
        out[base + r0] = acc[5]; out[base + r1] = acc[7];
    }
}

// Q2_0 (TD = 42) exact Split<42> load + apply against int4-staged hq (hoisted load across NTC entries)
struct Q20W {
    int16_t q[4];
    float d2;
};

__device__ __forceinline__ Q20W load_q2_0_w(const uint8_t* __restrict__ wrow, int k) {
    const int kbx = k >> 1;
    const int iqs = k & 1;
    const block_q2_0* bq = reinterpret_cast<const block_q2_0*>(wrow) + kbx;
    Q20W r;
    r.d2 = bq->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(bq->qs) + iqs * 4;
    r.q[0] = qs[0]; r.q[1] = qs[1]; r.q[2] = qs[2]; r.q[3] = qs[3];
    return r;
}

__device__ __forceinline__ float apply_q2_0_staged(const Q20W& r, const int4* __restrict__ s_u,
                                                   int c, int k, float d8) {
    const int4 ua = s_u[(c * 2 + 0) * kV70Hb + k];
    const int4 ub = s_u[(c * 2 + 1) * kV70Hb + k];
    int sumi = 0;
    auto acc_pair = [&](int16_t qv, int u0, int u1) {
        const int qe = __byte_perm(0x020100FFu, 0x020100FFu, (int) qv >> 0);
        const int qo = __byte_perm(0x020100FFu, 0x020100FFu, (int) qv >> 2);
        sumi = __dp4a(u0, (int) __byte_perm(qe, qo, 0x5140u), sumi);
        sumi = __dp4a(u1, (int) __byte_perm(qe, qo, 0x7362u), sumi);
    };
    acc_pair(r.q[0], ua.x, ua.y);
    acc_pair(r.q[1], ua.z, ua.w);
    acc_pair(r.q[2], ub.x, ub.y);
    acc_pair(r.q[3], ub.z, ub.w);
    return r.d2 * d8 * sumi;
}

// ============================================================================
// Step 5.3b: Persistent Weight-Major MoE Kernel (TG in {22, 18}, TD in {20, 42})
//   n_embd = 2560 (80 q8_1 blocks), n_ff = 640 (20 chunks of 32 rows),
//   Down BR = 64 rows/tile -> 40 Down tiles per expert.
//   Strictly resident grid sized by cudaOccupancyMaxActiveBlocksPerMultiprocessor.
// ============================================================================
template <int TG, int TD>
__global__ void __launch_bounds__(256, 3) moe_persist_v70_kernel(
    const unsigned long long* __restrict__ grp_ptr,
    const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups,
    const int32_t* __restrict__ ent_dst,
    const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq,
    NativeExpertLayout L,
    block_q8_1* __restrict__ hq,
    float* __restrict__ out,
    int state_slot) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    constexpr int kGuChunks = 20;      // 640 / 32
    constexpr int kDnTiles = 40;       // 2560 / 64
    constexpr int kXqStride = 72;      // bank-free stride for mma.m8n8k4
    constexpr int kRowW = 90;          // 20 * 18 / 4 uint32s per IQ4_NL down row

    __shared__ uint32_t s_epoch;
    __shared__ int s_ng;
    __shared__ uint32_t s_task;

    // Unioned SMEM: Phase 1 (Gate/Up) and Phase 2 (Down) never overlap within a CTA!
    __shared__ union {
        struct {
            alignas(16) int4 s_u[kV70Nt * 2 * kV70Xb];
            float s_ds[kV70Nt * kV70Xb];
            float res[kV70Nt][64];
            uint32_t sgrid_lo[(TG == 22) ? 1024 : 256];
            uint32_t sgrid_hi[(TG == 22) ? 1024 : 1];
        } gu;
        struct {
            alignas(16) uint32_t s_w[(TD == 20) ? (64 * kRowW) : 4];
            union {
                struct {
                    int xq[4][kV70Hb][8];
                    float xd[4][kV70Hb];
                } dp;
                struct {
                    uint32_t xq[kV70Hb * kXqStride];
                    float xd[kV70Hb][8];
                } tc;
                struct {
                    alignas(16) int4 s_u[kV70Nt * 2 * kV70Hb];
                    float s_ds[kV70Nt * kV70Hb];
                } q2;
            } x;
        } dn;
    } sm;

    const int tid = threadIdx.x;
    const int sub = tid >> 4;
    const int t16 = tid & 15;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    PersistState* const state = &g_dev_state[state_slot];
    if (tid == 0) {
        s_epoch = ld_acquire_gpu(&state->epoch) + 1u;
        s_ng = *n_groups;
    }
    __syncthreads();

    const uint32_t my_epoch = s_epoch;
    const int ng = s_ng;

    if (ng > 0) {
        // --------------------------------------------------------------------
        // Phase 1: Priority Gate/Up drain (all ng * 20 chunks before any Down)
        // Load sgrid ONCE per resident CTA!
        // --------------------------------------------------------------------
        if constexpr (TG == 22) {
            const uint2* gsrc = reinterpret_cast<const uint2*>(iq2s_grid);
#pragma unroll
            for (int i = tid; i < 1024; i += 256) {
                const uint2 v = gsrc[i];
                sm.gu.sgrid_lo[i] = v.x;
                sm.gu.sgrid_hi[i] = v.y;
            }
        } else {
            sm.gu.sgrid_lo[tid] = iq3xxs_grid[tid];
        }
        __syncthreads();

        const uint32_t total_gu = (uint32_t) ng * (uint32_t) kGuChunks;
        int cached_gu_g = -1;

        while (true) {
            if (tid == 0) s_task = atomic_add_relaxed_gpu(&state->ticket_gu, 1u);
            __syncthreads();
            const uint32_t t = s_task;
            if (t >= total_gu) break;

            const int g = (int) (t / (uint32_t) kGuChunks);
            const int ch = (int) (t - (uint32_t) g * (uint32_t) kGuChunks);
            const int r0 = ch * 32;
            const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[g]);
            const int e0 = grp_start[g];
            const int e1 = grp_start[g + 1];
            const int n_e = e1 - e0;

            for (int c0 = e0; c0 < e1; c0 += kV70Nt) {
                const int cn = min(kV70Nt, e1 - c0);
                // Reuse already-staged s_u/s_ds if same expert g with n_e <= 4!
                const bool reuse_x = (g == cached_gu_g) && (n_e <= kV70Nt) && (c0 == e0);
                if (!reuse_x) {
                    if (c0 > e0) __syncthreads();
                    for (int j = 0; j < cn; ++j) {
                        stage_q8_1_entry<kV70Xb>(sm.gu.s_u, sm.gu.s_ds, j,
                                                 xq + (size_t) ent_tok[c0 + j] * kV70Xb, tid);
                    }
                    __syncthreads();
                }

#pragma unroll 1
                for (int p = 0; p < 2; ++p) {
                    const int i0 = p * 16 + sub;
                    const uint8_t* w0 = blob + (size_t) (r0 + i0) * L.gu_row;
                    const uint8_t* w1 = blob + L.up_off + (size_t) (r0 + i0) * L.gu_row;
                    auto run = [&](auto ntc) {
                        constexpr int NTC = decltype(ntc)::value;
                        if constexpr (NTC <= 3) {
                            float sa[NTC], sb[NTC];
                            v70_row_dot2_80_sub16<TG, NTC>(w0, w1, sm.gu.s_u, sm.gu.s_ds, t16, sa, sb,
                                                           sm.gu.sgrid_lo, sm.gu.sgrid_hi);
                            if (t16 == 0) {
#pragma unroll
                                for (int j = 0; j < NTC; ++j) { sm.gu.res[j][i0] = sa[j]; sm.gu.res[j][32 + i0] = sb[j]; }
                            }
                        } else {
                            float srow[NTC];
                            v70_row_dot1_80_sub16<TG, NTC>(w0, sm.gu.s_u, sm.gu.s_ds, t16, srow,
                                                           sm.gu.sgrid_lo, sm.gu.sgrid_hi);
                            if (t16 == 0) {
#pragma unroll
                                for (int j = 0; j < NTC; ++j) sm.gu.res[j][i0] = srow[j];
                            }
                            v70_row_dot1_80_sub16<TG, NTC>(w1, sm.gu.s_u, sm.gu.s_ds, t16, srow,
                                                           sm.gu.sgrid_lo, sm.gu.sgrid_hi);
                            if (t16 == 0) {
#pragma unroll
                                for (int j = 0; j < NTC; ++j) sm.gu.res[j][32 + i0] = srow[j];
                            }
                        }
                    };
                    switch (cn) {
                        case 1: run(std::integral_constant<int, 1>{}); break;
                        case 2: run(std::integral_constant<int, 2>{}); break;
                        case 3: run(std::integral_constant<int, 3>{}); break;
                        default: run(std::integral_constant<int, 4>{}); break;
                    }
                }

                __syncthreads();
                if (warp < cn) {
                    const float gg = sm.gu.res[warp][lane];
                    const float uu = sm.gu.res[warp][32 + lane];
                    const float xi = (gg / (1.0f + __expf(-gg))) * uu;
                    float amax = fabsf(xi), sum = xi;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                        sum += __shfl_xor_sync(0xffffffffu, sum, o);
                    }
                    const float d = q8_1_finite(amax / 127.0f);
                    const int8_t qv = q8_1_quant(xi, d, amax);
                    block_q8_1* y = hq + (size_t) (c0 + warp) * kV70Hb + ch;
                    y->qs[lane] = qv;
                    if (lane == 0) y->ds = q8_1_ds(d, sum);
                }
            }
            cached_gu_g = (n_e <= kV70Nt && n_e > 0) ? g : -1;

            // Publish chunk completion with GPU acq_rel scope; 20th chunk releases expert g for Down.
            __syncthreads();
            if (tid == 0) {
                const uint32_t prev = atomic_add_acq_rel_gpu(&state->gu_done_cnt[g], 1u);
                if (prev == (uint32_t) (kGuChunks - 1)) {
                    state->gu_done_cnt[g] = 0u;
                    st_release_gpu(&state->ready_epoch[g], my_epoch);
                }
            }
            __syncthreads();
        }

        // --------------------------------------------------------------------
        // Phase 2: Weight-Major Down (all Gate/Up tickets already claimed -> deadlock-free)
        // --------------------------------------------------------------------
        __syncthreads();
        const uint32_t total_dn = (uint32_t) ng * (uint32_t) kDnTiles;
        const Q81Raw* hq_raw = reinterpret_cast<const Q81Raw*>(hq);
        int cached_dn_g = -1;

        while (true) {
            if (tid == 0) s_task = atomic_add_relaxed_gpu(&state->ticket_down, 1u);
            __syncthreads();
            const uint32_t t = s_task;
            if (t >= total_dn) break;

            const int g = (int) (t / (uint32_t) kDnTiles);
            const int tile = (int) (t - (uint32_t) g * (uint32_t) kDnTiles);
            const int row_block = tile * 64;
            const int e0 = grp_start[g];
            const int e1 = grp_start[g + 1];
            if (e0 >= e1) continue;
            const int n_e = e1 - e0;

            // Acquire readiness of expert g (already ready in >99% of tiles since g ascends in both phases)
            if (tid == 0) {
                while (ld_acquire_gpu(&state->ready_epoch[g]) != my_epoch) {}
            }
            __syncthreads();

            if constexpr (TD == 20) {
                // Burst-stage 64 IQ4_NL weight rows (23,040 B)
                const uint8_t* const w_base =
                    reinterpret_cast<const uint8_t*>(grp_ptr[g]) + L.down_off + (size_t) row_block * (size_t) (kV70Hb * 18);
                stage_iq4_w64_burst(w_base, sm.dn.s_w, tid);

                if (n_e <= 4) {
                    if (g != cached_dn_g) {
                        const int total_b = n_e * kV70Hb;
                        for (int idx = tid; idx < total_b * 2; idx += 256) {
                            const int half_b = idx & 1;
                            const int lin_b = idx >> 1;
                            const int bi = lin_b % kV70Hb;
                            const int c = lin_b / kV70Hb;
                            const Q81Raw* const xb = hq_raw + (size_t) (e0 + c) * kV70Hb + bi;
                            const int* const xqp = reinterpret_cast<const int*>(xb->qs) + half_b * 4;
                            sm.dn.x.dp.xq[c][bi][half_b * 4 + 0] = xqp[0];
                            sm.dn.x.dp.xq[c][bi][half_b * 4 + 1] = xqp[1];
                            sm.dn.x.dp.xq[c][bi][half_b * 4 + 2] = xqp[2];
                            sm.dn.x.dp.xq[c][bi][half_b * 4 + 3] = xqp[3];
                            if (half_b == 0) sm.dn.x.dp.xd[c][bi] = __low2float(xb->ds);
                        }
                        cached_dn_g = g;
                    }
                    __syncthreads();

                    const int sub_row = tid >> 2;   // 0..63
                    const int t4 = tid & 3;         // 0..3
                    const int odd4 = t4 & 1;
                    const uint32_t* const sw_row = sm.dn.s_w + sub_row * kRowW;

                    auto run_dp = [&](auto cn_c) {
                        constexpr int CN = decltype(cn_c)::value;
                        float acc[CN];
#pragma unroll
                        for (int c = 0; c < CN; ++c) acc[c] = 0.0f;
#pragma unroll 1
                        for (int bi = t4; bi < kV70Hb; bi += 4) {
                            uint16_t d_bits;
                            uint32_t q0, q1, q2, q3;
                            load_iq4_block(sw_row, bi, odd4, d_bits, q0, q1, q2, q3);
                            const float dw = __half2float(__ushort_as_half(d_bits));
                            uint32_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
                            iq4_decode_4bytes_signed(q0, lo0, hi0);
                            iq4_decode_4bytes_signed(q1, lo1, hi1);
                            iq4_decode_4bytes_signed(q2, lo2, hi2);
                            iq4_decode_4bytes_signed(q3, lo3, hi3);
#pragma unroll
                            for (int c = 0; c < CN; ++c) {
                                const int* const xqp = sm.dn.x.dp.xq[c][bi];
                                int sumi = 0;
                                sumi = __dp4a((int) lo0, xqp[0], sumi);
                                sumi = __dp4a((int) lo1, xqp[1], sumi);
                                sumi = __dp4a((int) lo2, xqp[2], sumi);
                                sumi = __dp4a((int) lo3, xqp[3], sumi);
                                sumi = __dp4a((int) hi0, xqp[4], sumi);
                                sumi = __dp4a((int) hi1, xqp[5], sumi);
                                sumi = __dp4a((int) hi2, xqp[6], sumi);
                                sumi = __dp4a((int) hi3, xqp[7], sumi);
                                acc[c] = fmaf(dw * sm.dn.x.dp.xd[c][bi], (float) sumi, acc[c]);
                            }
                        }
#pragma unroll
                        for (int c = 0; c < CN; ++c) {
                            acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], 2);
                            acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], 1);
                        }
                        if (t4 == 0) {
                            const int out_r = row_block + sub_row;
#pragma unroll
                            for (int c = 0; c < CN; ++c) {
                                out[(size_t) ent_dst[e0 + c] * 2560u + out_r] = acc[c];
                            }
                        }
                    };
                    if (n_e == 1) run_dp(std::integral_constant<int, 1>{});
                    else if (n_e == 2) run_dp(std::integral_constant<int, 2>{});
                    else if (n_e == 3) run_dp(std::integral_constant<int, 3>{});
                    else run_dp(std::integral_constant<int, 4>{});
                    __syncthreads();
                } else {
                    // Exact mma.m8n8k4 path for n_e >= 5 (8 warps x 8 rows = 64 rows!)
                    const int qp = (lane & 15) >> 2;
                    const int tl = (lane & 3) + 4 * (lane >> 4);
                    const int gg = lane & 3;
                    const int hh = lane >> 4;
                    const int c0_col = gg & 2;
                    const int odd = qp & 1;
                    const uint32_t* const sw_row = sm.dn.s_w + (warp * 8 + tl) * kRowW;

                    for (int c0 = e0; c0 < e1; c0 += 8) {
                        const int cn = min(8, e1 - c0);
                        const bool reuse_tc = (g == cached_dn_g) && (n_e <= 8) && (c0 == e0);
                        if (!reuse_tc) {
                            if (c0 > e0) __syncthreads();
                            for (int idx = tid; idx < 8 * kV70Hb * 2; idx += 256) {
                                const int half_b = idx & 1;
                                const int lin_b = idx >> 1;
                                const int bi = lin_b % kV70Hb;
                                const int c = lin_b / kV70Hb;
                                uint32_t* const xq_dst = sm.dn.x.tc.xq + bi * kXqStride + half_b * 32 + c;
                                if (c < cn) {
                                    const Q81Raw* const xb = hq_raw + (size_t) (c0 + c) * kV70Hb + bi;
                                    const uint32_t* const xqp = reinterpret_cast<const uint32_t*>(xb->qs) + half_b * 4;
                                    xq_dst[0]  = xqp[0] ^ 0x80808080u;
                                    xq_dst[8]  = xqp[1] ^ 0x80808080u;
                                    xq_dst[16] = xqp[2] ^ 0x80808080u;
                                    xq_dst[24] = xqp[3] ^ 0x80808080u;
                                    if (half_b == 0) sm.dn.x.tc.xd[bi][c] = __low2float(xb->ds);
                                } else {
                                    xq_dst[0] = xq_dst[8] = xq_dst[16] = xq_dst[24] = 0x80808080u;
                                    if (half_b == 0) sm.dn.x.tc.xd[bi][c] = 0.0f;
                                }
                            }
                        }
                        __syncthreads();

                        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
                        for (int bi = qp; bi < kV70Hb; bi += 4) {
                            uint16_t d_bits;
                            uint32_t q0, q1, q2, q3;
                            load_iq4_block(sw_row, bi, odd, d_bits, q0, q1, q2, q3);
                            const float dw = __half2float(__ushort_as_half(d_bits));
                            const uint32_t* const xq_bi = sm.dn.x.tc.xq + bi * kXqStride + tl;
                            float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                            auto step_q = [&](uint32_t qv, int j) {
                                uint32_t lo, hi;
                                iq4_decode_4bytes_bias128(qv, lo, hi);
                                mma_m8n8k4(d, u8x4_bias128_to_f16x4(lo), u8x4_bias128_to_f16x4(xq_bi[j * 8]));
                                mma_m8n8k4(d, u8x4_bias128_to_f16x4(hi), u8x4_bias128_to_f16x4(xq_bi[(j + 4) * 8]));
                            };
                            step_q(q0, 0); step_q(q1, 1); step_q(q2, 2); step_q(q3, 3);
                            iq4_scale_accum(acc, d, dw, lane, c0_col, sm.dn.x.tc.xd[bi]);
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 8);
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 4);
                        if (qp == 0) {
                            iq4_store_tile8(acc, row_block + warp * 8, 2560, gg, hh, cn, c0, ent_dst, out);
                        }
                    }
                    cached_dn_g = (n_e <= 8) ? g : -1;
                    __syncthreads();
                }
            } else {
                // Q2_0 (TD == 42): 64 rows = 4 passes x 16 sub-warps of 16 lanes, exact Split<42> reduction tree
                const uint8_t* const blob = reinterpret_cast<const uint8_t*>(grp_ptr[g]);
                for (int c0 = e0; c0 < e1; c0 += kV70Nt) {
                    const int cn = min(kV70Nt, e1 - c0);
                    const bool reuse_q2 = (g == cached_dn_g) && (n_e <= kV70Nt) && (c0 == e0);
                    if (!reuse_q2) {
                        if (c0 > e0) __syncthreads();
                        for (int j = 0; j < cn; ++j) {
                            stage_q8_1_entry<kV70Hb>(sm.dn.x.q2.s_u, sm.dn.x.q2.s_ds, j,
                                                     hq + (size_t) (c0 + j) * kV70Hb, tid);
                        }
                        __syncthreads();
                    }

                    const int sub8 = tid >> 3;
                    const int slane = tid & 7;
#pragma unroll 1
                    for (int p = 0; p < 2; ++p) {
                        const int r = row_block + p * 32 + sub8;
                        const uint8_t* const wrow = blob + L.down_off + (size_t) r * L.d_row;
                        auto run_q2 = [&](auto ntc) {
                            constexpr int NTC = decltype(ntc)::value;
                            float acc[NTC];
#pragma unroll
                            for (int c = 0; c < NTC; ++c) acc[c] = 0.0f;
#pragma unroll 1
                            for (int k = slane; k < kV70Hb; k += 8) {
                                const Q20W rw = load_q2_0_w(wrow, k);
#pragma unroll
                                for (int c = 0; c < NTC; ++c) {
                                    acc[c] += apply_q2_0_staged(rw, sm.dn.x.q2.s_u, c, k, sm.dn.x.q2.s_ds[c * kV70Hb + k]);
                                }
                            }
#pragma unroll
                            for (int c = 0; c < NTC; ++c) {
                                float v = acc[c];
#pragma unroll
                                for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                                if (slane == 0) out[(size_t) ent_dst[c0 + c] * 2560u + r] = v;
                            }
                        };
                        switch (cn) {
                            case 1: run_q2(std::integral_constant<int, 1>{}); break;
                            case 2: run_q2(std::integral_constant<int, 2>{}); break;
                            case 3: run_q2(std::integral_constant<int, 3>{}); break;
                            default: run_q2(std::integral_constant<int, 4>{}); break;
                        }
                    }
                    __syncthreads();
                }
                cached_dn_g = (n_e <= kV70Nt) ? g : -1;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Self-zeroing epilogue: last CTA resets tickets & increments epoch.
    // ------------------------------------------------------------------------
    __syncthreads();
    if (tid == 0) {
        const uint32_t done = atomic_add_acq_rel_gpu(&state->cta_done, 1u);
        if (done == gridDim.x - 1u) {
            state->ticket_gu = 0u;
            state->ticket_down = 0u;
            state->cta_done = 0u;
            uint32_t next_stored = my_epoch;   // next launch uses next_stored + 1
            if (next_stored == 0xFFFFFFFFu) {
                next_stored = 0u;
                for (int i = 0; i < kMaxPersistGroups; ++i) state->ready_epoch[i] = 0u;
            }
            st_release_gpu(&state->epoch, next_stored);
        }
    }
#else
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_dst; (void) ent_tok;
    (void) xq; (void) L; (void) hq; (void) out; (void) state_slot;
#endif
}

// ============================================================================
// Runtime switches, occupancy caching, and per-(dev, stream) PersistState pool
// ============================================================================
std::atomic<int> g_bank_override{-1};
std::atomic<int> g_persist_override{-1};

int default_bank_mode() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_EXP_BANK_V70");
        if (e == nullptr || e[0] == 0) return 1;
        return std::atoi(e);
    }();
    return v;
}

int default_persist_mode() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_MOE_PERSIST_V70");
        if (e == nullptr || e[0] == 0) return 1;
        return std::atoi(e);
    }();
    return v;
}

struct DevMeta {
    bool init = false;
    int num_sms = 0;
    int grid_22_20 = 0;
    int grid_18_20 = 0;
    int grid_22_42 = 0;
    int grid_18_42 = 0;
};

DevMeta g_dev_meta[64];
std::mutex g_state_mu;

struct StreamSlot {
    bool used = false;
    int dev = -1;
    void* stream = nullptr;
};
StreamSlot g_stream_slots[kMaxStreamSlots];

int get_persist_slot(int dev, void* stream) {
    std::lock_guard<std::mutex> lk(g_state_mu);
    for (int i = 0; i < kMaxStreamSlots; ++i) {
        if (g_stream_slots[i].used && g_stream_slots[i].dev == dev && g_stream_slots[i].stream == stream) {
            return i;
        }
    }
    for (int i = 0; i < kMaxStreamSlots; ++i) {
        if (!g_stream_slots[i].used) {
            g_stream_slots[i].used = true;
            g_stream_slots[i].dev = dev;
            g_stream_slots[i].stream = stream;
            return i;
        }
    }
    return 0;
}

void ensure_dev_init(int dev) {
    if (dev < 0 || dev >= 64 || g_dev_meta[dev].init) return;
    std::lock_guard<std::mutex> lk(g_state_mu);
    if (g_dev_meta[dev].init) return;
    DevMeta& m = g_dev_meta[dev];
    cudaDeviceGetAttribute(&m.num_sms, cudaDevAttrMultiProcessorCount, dev);

    // Step 5.3a carveout: 64 KB SMEM + 64 KB L1 (fits 3 CTA/SM x 20.7 KB with 64 KB L1!)
    cudaFuncSetAttribute(moe_gu_fused_v70_kernel<22>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(moe_gu_fused_v70_kernel<18>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(moe_gu_unfused_v70_kernel<22>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(moe_gu_unfused_v70_kernel<18>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);

    // Step 5.3b carveout: 96 KB SMEM for IQ4_NL (29.4 KB x 3 = 88.3 KB <= 96 KB!), 50% for Q2_0
    cudaFuncSetAttribute(moe_persist_v70_kernel<22, 20>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(moe_persist_v70_kernel<18, 20>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(moe_persist_v70_kernel<22, 42>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(moe_persist_v70_kernel<18, 42>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);

    auto query_grid = [&](auto fn) -> int {
        int bpsm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, fn, 256, 0) != cudaSuccess || bpsm <= 0) {
            return 0;
        }
        return m.num_sms * bpsm;
    };
    m.grid_22_20 = query_grid(moe_persist_v70_kernel<22, 20>);
    m.grid_18_20 = query_grid(moe_persist_v70_kernel<18, 20>);
    m.grid_22_42 = query_grid(moe_persist_v70_kernel<22, 42>);
    m.grid_18_42 = query_grid(moe_persist_v70_kernel<18, 42>);
    (void) cudaGetLastError();
    m.init = true;
}

}  // namespace

bool moe_volta_v70_supported() {
    static int per_dev[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (per_dev[dev] == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        per_dev[dev] = (major == 7 && minor == 0) ? 1 : -1;
    }
    return per_dev[dev] == 1;
}

int moe_volta_bank_mode() {
    if (!moe_volta_v70_supported()) return 0;
    const int ov = g_bank_override.load(std::memory_order_relaxed);
    return ov >= 0 ? ov : default_bank_mode();
}

void moe_volta_bank_set_mode(int mode) {
    g_bank_override.store(mode, std::memory_order_relaxed);
}

int moe_volta_persist_mode() {
    if (!moe_volta_v70_supported()) return 0;
    const int ov = g_persist_override.load(std::memory_order_relaxed);
    return ov >= 0 ? ov : default_persist_mode();
}

void moe_volta_persist_set_mode(int mode) {
    g_persist_override.store(mode, std::memory_order_relaxed);
}

bool moe_volta_gu_fused(const NativeExpertLayout& L, int64_t gy,
                        const unsigned long long* grp_ptr, const int32_t* grp_start,
                        const int32_t* n_groups, const int32_t* ent_tok,
                        const void* x_q8_1, void* hq_q8_1, void* stream) {
    if (moe_volta_bank_mode() <= 0) return false;
    if ((L.gu_type != 22 && L.gu_type != 18) || L.tail_off != 0 || L.had_block > 0) return false;
    if (L.n_embd != 2560 || L.n_ff <= 0 || (L.n_ff % 32) != 0 || gy <= 0) return false;

    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) ensure_dev_init(dev);

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const dim3 grid((unsigned) (L.n_ff / 32), (unsigned) gy);
    const auto* X = static_cast<const block_q8_1*>(x_q8_1);
    auto* hq = static_cast<block_q8_1*>(hq_q8_1);
    if (L.gu_type == 22) {
        moe_gu_fused_v70_kernel<22><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq);
    } else {
        moe_gu_fused_v70_kernel<18><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool moe_volta_gu_unfused(const NativeExpertLayout& L, int64_t gy,
                          const unsigned long long* grp_ptr, const int32_t* grp_start,
                          const int32_t* n_groups, const int32_t* ent_tok,
                          const void* x_q8_1, float* gate, float* up, void* stream) {
    if (moe_volta_bank_mode() <= 0) return false;
    if ((L.gu_type != 22 && L.gu_type != 18) || L.tail_off != 0) return false;
    if (L.n_embd != 2560 || L.n_ff <= 0 || (L.n_ff % 32) != 0 || gy <= 0) return false;

    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) ensure_dev_init(dev);

    const int nrow = 2 * (int) L.n_ff;
    const int rb = (gy >= 20) ? 32 : 16;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const dim3 grid((unsigned) ((nrow + rb - 1) / rb), (unsigned) gy);
    const auto* X = static_cast<const block_q8_1*>(x_q8_1);
    if (L.gu_type == 22) {
        moe_gu_unfused_v70_kernel<22><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, (uint64_t) L.up_off, (uint32_t) L.gu_row, (int) L.n_ff, gate, up, rb);
    } else {
        moe_gu_unfused_v70_kernel<18><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, (uint64_t) L.up_off, (uint32_t) L.gu_row, (int) L.n_ff, gate, up, rb);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool moe_volta_persist_grouped(const NativeExpertLayout& L, int64_t cap_groups,
                               const unsigned long long* grp_ptr, const int32_t* grp_start,
                               const int32_t* n_groups, const int32_t* ent_dst,
                               const int32_t* ent_tok, const void* x_q8_1,
                               void* hq_q8_1, float* out, void* stream) {
    if (moe_volta_persist_mode() <= 0) return false;
    if ((L.gu_type != 22 && L.gu_type != 18) || (L.d_type != 20 && L.d_type != 42)) return false;
    if (L.n_embd != 2560 || L.n_ff != 640 || L.tail_off != 0 || L.had_block > 0) return false;
    if (cap_groups <= 0 || cap_groups > kMaxPersistGroups) return false;
    if (L.d_type == 20 && (L.d_row != 360 || (L.down_off & 3u) != 0)) return false;

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    ensure_dev_init(dev);

    const int slot = get_persist_slot(dev, stream);
    const DevMeta& m = g_dev_meta[dev];
    int grid_ctas = 0;
    if (L.gu_type == 22 && L.d_type == 20) grid_ctas = m.grid_22_20;
    else if (L.gu_type == 18 && L.d_type == 20) grid_ctas = m.grid_18_20;
    else if (L.gu_type == 22 && L.d_type == 42) grid_ctas = m.grid_22_42;
    else grid_ctas = m.grid_18_42;
    if (grid_ctas <= 0) return false;

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const auto* X = static_cast<const block_q8_1*>(x_q8_1);
    auto* hq = static_cast<block_q8_1*>(hq_q8_1);

    if (L.gu_type == 22 && L.d_type == 20) {
        moe_persist_v70_kernel<22, 20><<<(unsigned) grid_ctas, 256, 0, s>>>(
            grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, L, hq, out, slot);
    } else if (L.gu_type == 18 && L.d_type == 20) {
        moe_persist_v70_kernel<18, 20><<<(unsigned) grid_ctas, 256, 0, s>>>(
            grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, L, hq, out, slot);
    } else if (L.gu_type == 22 && L.d_type == 42) {
        moe_persist_v70_kernel<22, 42><<<(unsigned) grid_ctas, 256, 0, s>>>(
            grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, L, hq, out, slot);
    } else {
        moe_persist_v70_kernel<18, 42><<<(unsigned) grid_ctas, 256, 0, s>>>(
            grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, L, hq, out, slot);
    }
    return cudaGetLastError() == cudaSuccess;
}

#else

bool moe_volta_v70_supported() { return false; }
int moe_volta_bank_mode() { return 0; }
void moe_volta_bank_set_mode(int) {}
int moe_volta_persist_mode() { return 0; }
void moe_volta_persist_set_mode(int) {}
bool moe_volta_gu_fused(const NativeExpertLayout&, int64_t, const unsigned long long*, const int32_t*,
                        const int32_t*, const int32_t*, const void*, void*, void*) { return false; }
bool moe_volta_gu_unfused(const NativeExpertLayout&, int64_t, const unsigned long long*, const int32_t*,
                          const int32_t*, const int32_t*, const void*, float*, float*, void*) { return false; }
bool moe_volta_persist_grouped(const NativeExpertLayout&, int64_t, const unsigned long long*, const int32_t*,
                               const int32_t*, const int32_t*, const int32_t*, const void*, void*,
                               float*, void*) { return false; }

#endif

}  // namespace strata::kernels
