// src/kernels/cuda/q8_volta_tc.cu - Clean-room MIT Volta (sm_70) FP16 Tensor Core Q8_0 kernels (V3).
//
// MIT License
// Copyright (c) 2026 Strata contributors
//
// Direct Q8_0 (ggml_type == 8, 34 bytes/block: half d + int8_t qs[32]) Volta mma.sync.aligned.m8n8k4 kernels:
//   1. Prefill Fused W8A16 GEMM (128x128x64 / 64x128x64 / 64x64x64):
//      Dynamic-smem double-buffered XOR-swizzled LDS.128 with register fragment ping-pong (a[2], b[2])
//      and register-prefetched coalesced X (uint4) & W (16-lane uint32 + distributed 17th word) overlapped
//      with HMMA. Zero stack spills across all instantiations.
//   2. Decode Single & Segmented Q8_0 x Q8_1 TC GEMM (M = 1..16):
//      - Large-N (N > 3072, K % 256 == 0): BR=16, NB_STEP=8 (BK=256), NT=128, double-buffered s_w[2]/s_xh[2]
//        (1 __syncthreads()/step, register-prefetched uint4 W loads overlapped with MMA).
//      - Medium/Small-N (N <= 3072, K % 512 == 0): BR=8, NB_STEP=16 (BK=512), NT=128, single-buffered smem
//        (14.1 KB for M<=8 -> 5 CTA/SM; 23.9 KB for M<=16 -> 4 CTA/SM -> 0 wave tail on N=2560 = 320 CTAs!)
//        with register-prefetched uint4 W loads overlapped with MMA.
//      Bitwise identical across M = 1..16 for every column.

#include "strata/kernels/q8_volta_tc.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);

#if !defined(__HIPCC__)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
#define STRATA_Q8V_SM70 1
#else
#define STRATA_Q8V_SM70 0
#endif

__device__ __forceinline__ uint2 q8v_i8x4_to_f16x4(uint32_t w) {
    const uint32_t c = w ^ 0x80808080u;
    const uint32_t lo = __byte_perm(c, 0x64646464u, 0x5140);
    const uint32_t hi = __byte_perm(c, 0x64646464u, 0x7362);
    const __half2 m = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 l = __hsub2(*reinterpret_cast<const __half2*>(&lo), m);
    const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&hi), m);
    return make_uint2(*reinterpret_cast<const uint32_t*>(&l), *reinterpret_cast<const uint32_t*>(&h));
}

__device__ __forceinline__ uint2 q8v_i8x4_to_f16x4_scaled(uint32_t w, __half2 dh2) {
    const uint2 q = q8v_i8x4_to_f16x4(w);
    const __half2 l = __hmul2(*reinterpret_cast<const __half2*>(&q.x), dh2);
    const __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&q.y), dh2);
    return make_uint2(*reinterpret_cast<const uint32_t*>(&l), *reinterpret_cast<const uint32_t*>(&h));
}

#if STRATA_Q8V_SM70
__device__ __forceinline__ void q8v_mma_m8n8k4(float (&d)[8], uint2 a, uint2 b) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}
#endif

// =================================================================================================
// 1. Prefill Fused Q8_0 W8A16 GEMM V3: Double-Buffered Dynamic Smem + LDS.128 + Frag Ping-Pong
// =================================================================================================

__device__ __forceinline__ int q8v_swz8(int r, int c8) {
    return c8 ^ ((r & 3) | ((r >> 1) & 4));
}

template <int BM, int BN, int WARPS_M, int WARPS_N, bool HAS_BETA, int MIN_BLOCKS>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, MIN_BLOCKS)
q8_volta_prefill_kernel(const uint16_t* __restrict__ X_f16, const uint8_t* __restrict__ W_q8_0,
                        float* __restrict__ Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
#if STRATA_Q8V_SM70
    constexpr int NT = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M;
    constexpr int WN = BN / WARPS_N;
    constexpr int WTM = WM / 16;
    constexpr int WTN = WN / 16;
    constexpr int X_ITERS = BM / (NT >> 3);
    constexpr int W_ITERS = BN / (NT >> 4);
    static_assert(W_ITERS <= 16, "my_w16 packs 1 word per lane16");

    extern __shared__ uint4 dyn_smem[];
    // Layout: s_X[2][BM][8], then s_W[2][BN][8] (each uint4 = 8 halves = 16 B; 0 padding)
    uint4* const s_X_base = dyn_smem;
    uint4* const s_W_base = dyn_smem + 2 * BM * 8;
    auto sx_at = [&](int stg, int r, int c8) -> uint4& { return s_X_base[(stg * BM + r) * 8 + q8v_swz8(r, c8)]; };
    auto sw_at = [&](int stg, int r, int c8) -> uint4& { return s_W_base[(stg * BN + r) * 8 + q8v_swz8(r, c8)]; };

    constexpr int kGroupM = 8;
    const int grid_m = (int) ((T + BM - 1) / BM);
    const int grid_n = (int) ((N + BN - 1) / BN);
    const int lin = (int) blockIdx.x;
    const int tiles_per_grp = kGroupM * grid_n;
    const int grp = lin / tiles_per_grp;
    const int first_m = grp * kGroupM;
    const int grp_m = min(kGroupM, grid_m - first_m);
    const int in_grp = lin - grp * tiles_per_grp;
    const int bm_idx = first_m + (in_grp % grp_m);
    const int bn_idx = in_grp / grp_m;
    const int64_t block_m = (int64_t) bm_idx * BM;
    const int64_t block_n = (int64_t) bn_idx * BN;

    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    const int qp = (lane & 15) >> 2;
    const int tl = (lane & 3) + 4 * (lane >> 4);

    const int x_r0 = tid >> 3;
    const int x_c8 = tid & 7;
    constexpr int x_rstep = NT >> 3;

    const int w_hw = tid >> 4;
    const int w_l16 = tid & 15;
    constexpr int w_rstep = NT >> 4;
    const size_t w_row_bytes = (size_t) (K >> 5) * 34u;

    const int a_row_base = wm * WM + (qp & 1) * 8 + tl;
    const int b_row_base = wn * WN + (qp >> 1) * 8 + tl;

    float acc[WTM][WTN][8];
#pragma unroll
    for (int tm = 0; tm < WTM; ++tm)
#pragma unroll
        for (int tn = 0; tn < WTN; ++tn)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[tm][tn][i] = 0.0f;

    const int n_steps = (int) (K >> 6);

    uint4 x_reg[X_ITERS];
    uint32_t w_reg[W_ITERS];
    uint32_t my_w16 = 0u;

    auto fetch_regs = [&](int step) {
        const int64_t k0 = (int64_t) step * 64;
#pragma unroll
        for (int xi = 0; xi < X_ITERS; ++xi) {
            const int r = x_r0 + xi * x_rstep;
            const int64_t gt = block_m + r;
            x_reg[xi] = (gt < T)
                            ? __ldg(reinterpret_cast<const uint4*>(X_f16 + (size_t) gt * (size_t) K + (size_t) k0) + x_c8)
                            : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int wi = 0; wi < W_ITERS; ++wi) {
            const int r = w_hw + wi * w_rstep;
            const int64_t gn = block_n + r;
            const bool valid = gn < N;
            const uint32_t* const wp = reinterpret_cast<const uint32_t*>(
                W_q8_0 + (size_t) (valid ? gn : 0) * w_row_bytes + (size_t) step * 68u);
            w_reg[wi] = valid ? __ldg(wp + w_l16) : 0u;
        }
        if (w_l16 < W_ITERS) {
            const int r = w_hw + w_l16 * w_rstep;
            const int64_t gn = block_n + r;
            const bool valid = gn < N;
            const uint32_t* const wp = reinterpret_cast<const uint32_t*>(
                W_q8_0 + (size_t) (valid ? gn : 0) * w_row_bytes + (size_t) step * 68u);
            my_w16 = valid ? __ldg(wp + 16) : 0u;
        }
    };

    auto commit_regs = [&](int stage) {
#pragma unroll
        for (int xi = 0; xi < X_ITERS; ++xi) {
            const int r = x_r0 + xi * x_rstep;
            sx_at(stage, r, x_c8) = x_reg[xi];
        }
#pragma unroll
        for (int wi = 0; wi < W_ITERS; ++wi) {
            const int r = w_hw + wi * w_rstep;
            const uint32_t w = w_reg[wi];
            uint32_t w_next = __shfl_down_sync(0xffffffffu, w, 1, 16);
            const uint32_t w16_b = __shfl_sync(0xffffffffu, my_w16, wi, 16);
            if (w_l16 == 15) w_next = w16_b;
            const uint32_t d0_u = __shfl_sync(0xffffffffu, w, 0, 16) & 0xFFFFu;
            const uint32_t d1_u = __shfl_sync(0xffffffffu, w, 8, 16) >> 16;
            const uint32_t q4 = (w_l16 < 8) ? __funnelshift_r(w, w_next, 16) : w_next;
            const __half2 dh2 = __half2half2(__ushort_as_half((uint16_t) (w_l16 < 8 ? d0_u : d1_u)));
            const uint2 h4 = q8v_i8x4_to_f16x4_scaled(q4, dh2);
            const int c8 = w_l16 >> 1;
            const int half_u4 = w_l16 & 1;
            uint2* const dst = reinterpret_cast<uint2*>(&sw_at(stage, r, c8));
            dst[half_u4] = h4;
        }
    };

    auto compute_stage = [&](int stage) {
        uint4 a[2][WTM];
        uint4 b[2][WTN];
        // Preload k8 = 0
#pragma unroll
        for (int tm = 0; tm < WTM; ++tm) a[0][tm] = sx_at(stage, a_row_base + tm * 16, 0);
#pragma unroll
        for (int tn = 0; tn < WTN; ++tn) b[0][tn] = sw_at(stage, b_row_base + tn * 16, 0);

#pragma unroll
        for (int k8 = 0; k8 < 8; ++k8) {
            const int c = k8 & 1;
            const int n = c ^ 1;
            // First k=4 half of k8
#pragma unroll
            for (int tm = 0; tm < WTM; ++tm)
#pragma unroll
                for (int tn = 0; tn < WTN; ++tn)
                    q8v_mma_m8n8k4(acc[tm][tn], make_uint2(a[c][tm].x, a[c][tm].y), make_uint2(b[c][tn].x, b[c][tn].y));
            // Issue LDS.128 for k8 + 1 while second k=4 half executes!
            if (k8 + 1 < 8) {
#pragma unroll
                for (int tm = 0; tm < WTM; ++tm) a[n][tm] = sx_at(stage, a_row_base + tm * 16, k8 + 1);
#pragma unroll
                for (int tn = 0; tn < WTN; ++tn) b[n][tn] = sw_at(stage, b_row_base + tn * 16, k8 + 1);
            }
            // Second k=4 half of k8
#pragma unroll
            for (int tm = 0; tm < WTM; ++tm)
#pragma unroll
                for (int tn = 0; tn < WTN; ++tn)
                    q8v_mma_m8n8k4(acc[tm][tn], make_uint2(a[c][tm].z, a[c][tm].w), make_uint2(b[c][tn].z, b[c][tn].w));
        }
    };

    fetch_regs(0);
    commit_regs(0);
    __syncthreads();

    for (int step = 0; step < n_steps; ++step) {
        const int cur = step & 1;
        const int nxt = cur ^ 1;
        if (step + 1 < n_steps) fetch_regs(step + 1);
        compute_stage(cur);
        if (step + 1 < n_steps) commit_regs(nxt);
        __syncthreads();
    }

    const int gg = lane & 3;
    const int hh = lane >> 4;
    const int r0 = (gg & 1) + 4 * hh;
    const int c0 = gg & 2;

    auto store_elem = [&](int64_t gt, int64_t gn, float val) {
        if (gt < T && gn < N) {
            float* const p = Y + (size_t) gt * (size_t) ldy + (size_t) gn;
            *p = HAS_BETA ? fmaf(beta, *p, val) : val;
        }
    };

#pragma unroll
    for (int tm = 0; tm < WTM; ++tm) {
        const int64_t t0 = block_m + wm * WM + tm * 16 + (qp & 1) * 8 + r0;
        const int64_t t1 = t0 + 2;
#pragma unroll
        for (int tn = 0; tn < WTN; ++tn) {
            const int64_t n0 = block_n + wn * WN + tn * 16 + (qp >> 1) * 8 + c0;
            const int64_t n4 = n0 + 4;
            const float* const d = acc[tm][tn];
            store_elem(t0, n0, d[0]);
            store_elem(t0, n0 + 1, d[1]);
            store_elem(t1, n0, d[2]);
            store_elem(t1, n0 + 1, d[3]);
            store_elem(t0, n4, d[4]);
            store_elem(t0, n4 + 1, d[5]);
            store_elem(t1, n4, d[6]);
            store_elem(t1, n4 + 1, d[7]);
        }
    }
#else
    (void) X_f16; (void) W_q8_0; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) beta;
    __trap();
#endif
}

// =================================================================================================
// 2. Decode Single & Segmented Q8_0 x Q8_1 TC GEMM V3 (M = 1..16)
// =================================================================================================

struct Q8VoltaSegPlan {
    int n_segs;
    int seg_blk_end[4];
    const uint8_t* w[4];
    float* y[4];
    int n_out[4];
};

__device__ __forceinline__ void q8v_resolve_seg(const Q8VoltaSegPlan& plan, int blk,
                                                const uint8_t*& w_base, float*& y_base,
                                                int& seg_n_out, int& blk_in_seg) {
    if (blk < plan.seg_blk_end[0]) {
        w_base = plan.w[0]; y_base = plan.y[0]; seg_n_out = plan.n_out[0]; blk_in_seg = blk;
    } else if (plan.n_segs > 1 && blk < plan.seg_blk_end[1]) {
        w_base = plan.w[1]; y_base = plan.y[1]; seg_n_out = plan.n_out[1]; blk_in_seg = blk - plan.seg_blk_end[0];
    } else if (plan.n_segs > 2 && blk < plan.seg_blk_end[2]) {
        w_base = plan.w[2]; y_base = plan.y[2]; seg_n_out = plan.n_out[2]; blk_in_seg = blk - plan.seg_blk_end[1];
    } else {
        w_base = plan.w[3]; y_base = plan.y[3]; seg_n_out = plan.n_out[3]; blk_in_seg = blk - plan.seg_blk_end[2];
    }
}

// Pipelined BR=16, NB_STEP=8 (BK=256), NT=128 (4 warps), double-buffered s_w[2] + s_xh[2]:
template <bool WIDE_M>
__global__ void __launch_bounds__(128, WIDE_M ? 3 : 4)
q8_volta_decode_seg_nb8_kernel(Q8VoltaSegPlan plan, const Q81Block* __restrict__ x_q8_1, int n_in, int ncols) {
#if STRATA_Q8V_SM70
    constexpr int NT = 128;
    constexpr int NB_STEP = 8;
    constexpr int W_STRIDE = 70;
    constexpr int MCAP = WIDE_M ? 16 : 8;

    __shared__ union {
        struct {
            uint32_t w[2][16][W_STRIDE];
            __half xh[2][NB_STEP][MCAP][36];
            float xd[2][MCAP][NB_STEP];
        } pipe;
        float out[4][16][MCAP];
    } smem;

    const uint8_t* w_base = nullptr;
    float* y_base = nullptr;
    int seg_n_out = 0, blk_in_seg = 0;
    q8v_resolve_seg(plan, (int) blockIdx.x, w_base, y_base, seg_n_out, blk_in_seg);
    const int row0 = blk_in_seg * 16;

    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int qp = (lane & 15) >> 2;
    const int tl = (lane & 3) + 4 * (lane >> 4);
    const int gg = lane & 3;
    const int hh = lane >> 4;
    const int rg = qp & 1;
    const int c0 = gg & 2;

    const int nb = n_in >> 5;
    const int steps = nb >> 3;
    const size_t row_bytes = (size_t) nb * 34u;

    if (ncols < MCAP) {
        const int pad_c = MCAP - ncols;
        const int total_z = pad_c * NB_STEP * 8;
        for (int stg = 0; stg < 2; ++stg) {
            for (int i = tid; i < total_z; i += NT) {
                const int k4 = i & 7;
                const int t0 = i >> 3;
                const int bi = t0 & 7;
                const int c = ncols + (t0 >> 3);
                *reinterpret_cast<uint2*>(&smem.pipe.xh[stg][bi][c][k4 * 4]) = make_uint2(0u, 0u);
            }
            const int total_zd = pad_c * NB_STEP;
            for (int i = tid; i < total_zd; i += NT) {
                const int bi = i & 7;
                const int c = ncols + (i >> 3);
                smem.pipe.xd[stg][c][bi] = 0.0f;
            }
        }
    }

    const int hw = tid >> 4;
    const int l16 = tid & 15;
    const bool valid0 = (row0 + hw) < seg_n_out;
    const bool valid1 = (row0 + hw + 8) < seg_n_out;
    const uint8_t* const rp0_base = w_base + (size_t) (valid0 ? (row0 + hw) : 0) * row_bytes;
    const uint8_t* const rp1_base = w_base + (size_t) (valid1 ? (row0 + hw + 8) : 0) * row_bytes;
    const int total_x_halves = ncols * NB_STEP * 2;

    uint4 w_v0, w_v1, w_tail = make_uint4(0u, 0u, 0u, 0u);
    auto fetch_w = [&](int st) {
        const uint4* const p0 = reinterpret_cast<const uint4*>(rp0_base + (size_t) st * 272u);
        const uint4* const p1 = reinterpret_cast<const uint4*>(rp1_base + (size_t) st * 272u);
        w_v0 = valid0 ? __ldg(p0 + l16) : make_uint4(0u, 0u, 0u, 0u);
        w_v1 = valid1 ? __ldg(p1 + l16) : make_uint4(0u, 0u, 0u, 0u);
        if (l16 == 0) w_tail = valid0 ? __ldg(p0 + 16) : make_uint4(0u, 0u, 0u, 0u);
        else if (l16 == 1) w_tail = valid1 ? __ldg(p1 + 16) : make_uint4(0u, 0u, 0u, 0u);
    };
    auto store_w = [&](int stg) {
        uint2* const d0 = reinterpret_cast<uint2*>(&smem.pipe.w[stg][hw][l16 * 4]);
        d0[0] = make_uint2(w_v0.x, w_v0.y);
        d0[1] = make_uint2(w_v0.z, w_v0.w);
        uint2* const d1 = reinterpret_cast<uint2*>(&smem.pipe.w[stg][hw + 8][l16 * 4]);
        d1[0] = make_uint2(w_v1.x, w_v1.y);
        d1[1] = make_uint2(w_v1.z, w_v1.w);
        if (l16 == 0) {
            uint2* const dt = reinterpret_cast<uint2*>(&smem.pipe.w[stg][hw][64]);
            dt[0] = make_uint2(w_tail.x, w_tail.y);
            dt[1] = make_uint2(w_tail.z, w_tail.w);
        } else if (l16 == 1) {
            uint2* const dt = reinterpret_cast<uint2*>(&smem.pipe.w[stg][hw + 8][64]);
            dt[0] = make_uint2(w_tail.x, w_tail.y);
            dt[1] = make_uint2(w_tail.z, w_tail.w);
        }
    };
    auto load_x = [&](int st, int stg) {
        for (int idx = tid; idx < total_x_halves; idx += NT) {
            const int half_b = idx & 1;
            const int lin_b = idx >> 1;
            const int bi = lin_b & 7;
            const int c = lin_b >> 3;
            const Q81Block* const xb = x_q8_1 + (size_t) c * (size_t) nb + st * 8 + bi;
            const uint32_t* const xqp = reinterpret_cast<const uint32_t*>(xb->qs) + half_b * 4;
            uint2* const dst = reinterpret_cast<uint2*>(&smem.pipe.xh[stg][bi][c][half_b * 16]);
            dst[0] = q8v_i8x4_to_f16x4(__ldg(xqp + 0));
            dst[1] = q8v_i8x4_to_f16x4(__ldg(xqp + 1));
            dst[2] = q8v_i8x4_to_f16x4(__ldg(xqp + 2));
            dst[3] = q8v_i8x4_to_f16x4(__ldg(xqp + 3));
            if (half_b == 0) smem.pipe.xd[stg][c][bi] = __low2float(__ldg(&xb->ds));
        }
    };

    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    auto mma_one_block = [&](int stg, int bi, int col_off, float (&acc)[8]) {
        const int r = rg * 8 + tl;
        const int pair = bi >> 1;
        const int odd = bi & 1;
        const uint32_t* const wr = &smem.pipe.w[stg][r][pair * 17];
        float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const __half* const xb = &smem.pipe.xh[stg][bi][col_off + tl][0];
        uint32_t prev = odd ? 0u : wr[0];
        const float dw = __half2float(__ushort_as_half((uint16_t) (odd ? (wr[8] >> 16) : (prev & 0xFFFFu))));
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t q4;
            if (odd == 0) {
                const uint32_t cur = wr[kk + 1];
                q4 = __funnelshift_r(prev, cur, 16);
                prev = cur;
            } else {
                q4 = wr[9 + kk];
            }
            q8v_mma_m8n8k4(d, q8v_i8x4_to_f16x4(q4), *reinterpret_cast<const uint2*>(xb + kk * 4));
        }
        const float dw_other = __shfl_xor_sync(0xffffffffu, dw, 2);
        const float dw0 = (lane & 2) ? dw_other : dw;
        const float dw1 = (lane & 2) ? dw : dw_other;
        const float dx0 = smem.pipe.xd[stg][col_off + c0][bi];
        const float dx1 = smem.pipe.xd[stg][col_off + c0 + 1][bi];
        const float dx2 = smem.pipe.xd[stg][col_off + c0 + 4][bi];
        const float dx3 = smem.pipe.xd[stg][col_off + c0 + 5][bi];
        acc[0] = fmaf(dw0 * dx0, d[0], acc[0]);
        acc[1] = fmaf(dw0 * dx1, d[1], acc[1]);
        acc[2] = fmaf(dw1 * dx0, d[2], acc[2]);
        acc[3] = fmaf(dw1 * dx1, d[3], acc[3]);
        acc[4] = fmaf(dw0 * dx2, d[4], acc[4]);
        acc[5] = fmaf(dw0 * dx3, d[5], acc[5]);
        acc[6] = fmaf(dw1 * dx2, d[6], acc[6]);
        acc[7] = fmaf(dw1 * dx3, d[7], acc[7]);
    };

    fetch_w(0);
    store_w(0);
    load_x(0, 0);
    __syncthreads();

    for (int st = 0; st < steps; ++st) {
        const int cur = st & 1;
        const int nxt = cur ^ 1;
        if (st + 1 < steps) fetch_w(st + 1);
        if constexpr (!WIDE_M) {
            mma_one_block(cur, warp * 2 + (qp >> 1), 0, acc0);
        } else {
            const int cg = qp >> 1;
            mma_one_block(cur, warp * 2 + 0, cg * 8, acc0);
            mma_one_block(cur, warp * 2 + 1, cg * 8, acc1);
        }
        if (st + 1 < steps) {
            store_w(nxt);
            load_x(st + 1, nxt);
        }
        __syncthreads();
    }

    if constexpr (!WIDE_M) {
        const int kb = qp >> 1;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc0[i] += __shfl_xor_sync(0xffffffffu, acc0[i], 8);
        if (kb == 0) {
            const int r0 = rg * 8 + (gg & 1) + 4 * hh;
            const int r1 = r0 + 2;
            smem.out[warp][r0][c0] = acc0[0];
            smem.out[warp][r0][c0 + 1] = acc0[1];
            smem.out[warp][r1][c0] = acc0[2];
            smem.out[warp][r1][c0 + 1] = acc0[3];
            smem.out[warp][r0][c0 + 4] = acc0[4];
            smem.out[warp][r0][c0 + 5] = acc0[5];
            smem.out[warp][r1][c0 + 4] = acc0[6];
            smem.out[warp][r1][c0 + 5] = acc0[7];
        }
    } else {
        const int cg = qp >> 1;
        const int col_off = cg * 8;
        const int r0 = rg * 8 + (gg & 1) + 4 * hh;
        const int r1 = r0 + 2;
        smem.out[warp][r0][col_off + c0] = acc0[0] + acc1[0];
        smem.out[warp][r0][col_off + c0 + 1] = acc0[1] + acc1[1];
        smem.out[warp][r1][col_off + c0] = acc0[2] + acc1[2];
        smem.out[warp][r1][col_off + c0 + 1] = acc0[3] + acc1[3];
        smem.out[warp][r0][col_off + c0 + 4] = acc0[4] + acc1[4];
        smem.out[warp][r0][col_off + c0 + 5] = acc0[5] + acc1[5];
        smem.out[warp][r1][col_off + c0 + 4] = acc0[6] + acc1[6];
        smem.out[warp][r1][col_off + c0 + 5] = acc0[7] + acc1[7];
    }
    __syncthreads();

    constexpr int total_out = 16 * MCAP;
    for (int idx = tid; idx < total_out; idx += NT) {
        const int r = idx / MCAP;
        const int c = idx % MCAP;
        if (c < ncols && row0 + r < seg_n_out) {
            const float sum = (smem.out[0][r][c] + smem.out[1][r][c]) + (smem.out[2][r][c] + smem.out[3][r][c]);
            y_base[(size_t) c * (size_t) seg_n_out + (size_t) (row0 + r)] = sum;
        }
    }
#else
    (void) plan; (void) x_q8_1; (void) n_in; (void) ncols;
    __trap();
#endif
}

// High-occupancy zero-wave-tail BR=8, NB_STEP=16 (BK=512), NT=128 (4 warps) for N<=3072, K%512==0!
// Single-buffered smem (14.1 KB for M<=8 -> 5 CTA/SM; 23.9 KB for M<=16 -> 4 CTA/SM -> exact 1 wave on N=2560 = 320 CTAs!)
// with register-prefetched uint4 W loads overlapped with MMA!
template <bool WIDE_M>
__global__ void __launch_bounds__(128, WIDE_M ? 4 : 5)
q8_volta_decode_seg_br8_nb16_kernel(Q8VoltaSegPlan plan, const Q81Block* __restrict__ x_q8_1, int n_in, int ncols) {
#if STRATA_Q8V_SM70
    constexpr int NT = 128;
    constexpr int NB_STEP = 16;
    constexpr int W_STRIDE = 138;
    constexpr int MCAP = WIDE_M ? 16 : 8;

    __shared__ union {
        struct {
            uint32_t w[8][W_STRIDE];      //  4,416 B
            __half xh[NB_STEP][MCAP][36]; //  9,216 B (M<=8) or 18,432 B (M<=16)
            float xd[MCAP][NB_STEP];      //    512 B (M<=8) or  1,024 B (M<=16) -> total 14.1 KB / 23.9 KB <= 24 KB!
        } pipe;
        float out[4][8][MCAP];
    } smem;

    const uint8_t* w_base = nullptr;
    float* y_base = nullptr;
    int seg_n_out = 0, blk_in_seg = 0;
    q8v_resolve_seg(plan, (int) blockIdx.x, w_base, y_base, seg_n_out, blk_in_seg);
    const int row0 = blk_in_seg * 8;

    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int qp = (lane & 15) >> 2;
    const int tl = (lane & 3) + 4 * (lane >> 4);
    const int gg = lane & 3;
    const int hh = lane >> 4;
    const int c0 = gg & 2;

    const int nb = n_in >> 5;
    const int steps = nb >> 4;
    const size_t row_bytes = (size_t) nb * 34u;

    if (ncols < MCAP) {
        const int pad_c = MCAP - ncols;
        const int total_z = pad_c * NB_STEP * 8;
        for (int i = tid; i < total_z; i += NT) {
            const int k4 = i & 7;
            const int t0 = i >> 3;
            const int bi = t0 & 15;
            const int c = ncols + (t0 >> 4);
            *reinterpret_cast<uint2*>(&smem.pipe.xh[bi][c][k4 * 4]) = make_uint2(0u, 0u);
        }
        const int total_zd = pad_c * NB_STEP;
        for (int i = tid; i < total_zd; i += NT) {
            const int bi = i & 15;
            const int c = ncols + (i >> 4);
            smem.pipe.xd[c][bi] = 0.0f;
        }
    }

    const int hw = tid >> 4;
    const int l16 = tid & 15;
    const bool valid = (row0 + hw) < seg_n_out;
    const uint8_t* const rp_base = w_base + (size_t) (valid ? (row0 + hw) : 0) * row_bytes;
    const int total_x_halves = ncols * NB_STEP * 2;

    uint4 w_v0, w_v1, w_tail = make_uint4(0u, 0u, 0u, 0u);
    auto fetch_w = [&](int st) {
        const uint4* const p = reinterpret_cast<const uint4*>(rp_base + (size_t) st * 544u);
        w_v0 = valid ? __ldg(p + l16) : make_uint4(0u, 0u, 0u, 0u);
        w_v1 = valid ? __ldg(p + 16 + l16) : make_uint4(0u, 0u, 0u, 0u);
        if (l16 < 2) w_tail = valid ? __ldg(p + 32 + l16) : make_uint4(0u, 0u, 0u, 0u);
    };
    auto store_w = [&]() {
        uint2* const d0 = reinterpret_cast<uint2*>(&smem.pipe.w[hw][l16 * 4]);
        d0[0] = make_uint2(w_v0.x, w_v0.y);
        d0[1] = make_uint2(w_v0.z, w_v0.w);
        uint2* const d1 = reinterpret_cast<uint2*>(&smem.pipe.w[hw][64 + l16 * 4]);
        d1[0] = make_uint2(w_v1.x, w_v1.y);
        d1[1] = make_uint2(w_v1.z, w_v1.w);
        if (l16 < 2) {
            uint2* const dt = reinterpret_cast<uint2*>(&smem.pipe.w[hw][128 + l16 * 4]);
            dt[0] = make_uint2(w_tail.x, w_tail.y);
            dt[1] = make_uint2(w_tail.z, w_tail.w);
        }
    };
    auto load_x = [&](int st) {
        for (int idx = tid; idx < total_x_halves; idx += NT) {
            const int half_b = idx & 1;
            const int lin_b = idx >> 1;
            const int bi = lin_b & 15;
            const int c = lin_b >> 4;
            const Q81Block* const xb = x_q8_1 + (size_t) c * (size_t) nb + st * 16 + bi;
            const uint32_t* const xqp = reinterpret_cast<const uint32_t*>(xb->qs) + half_b * 4;
            uint2* const dst = reinterpret_cast<uint2*>(&smem.pipe.xh[bi][c][half_b * 16]);
            dst[0] = q8v_i8x4_to_f16x4(__ldg(xqp + 0));
            dst[1] = q8v_i8x4_to_f16x4(__ldg(xqp + 1));
            dst[2] = q8v_i8x4_to_f16x4(__ldg(xqp + 2));
            dst[3] = q8v_i8x4_to_f16x4(__ldg(xqp + 3));
            if (half_b == 0) smem.pipe.xd[c][bi] = __low2float(__ldg(&xb->ds));
        }
    };

    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    auto mma_one_block = [&](int bi, int col_off, float (&acc)[8]) {
        const int r = tl;
        const int pair = bi >> 1;
        const int odd = bi & 1;
        const uint32_t* const wr = &smem.pipe.w[r][pair * 17];
        float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const __half* const xb = &smem.pipe.xh[bi][col_off + tl][0];
        uint32_t prev = odd ? 0u : wr[0];
        const float dw = __half2float(__ushort_as_half((uint16_t) (odd ? (wr[8] >> 16) : (prev & 0xFFFFu))));
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t q4;
            if (odd == 0) {
                const uint32_t cur = wr[kk + 1];
                q4 = __funnelshift_r(prev, cur, 16);
                prev = cur;
            } else {
                q4 = wr[9 + kk];
            }
            q8v_mma_m8n8k4(d, q8v_i8x4_to_f16x4(q4), *reinterpret_cast<const uint2*>(xb + kk * 4));
        }
        const float dw_other = __shfl_xor_sync(0xffffffffu, dw, 2);
        const float dw0 = (lane & 2) ? dw_other : dw;
        const float dw1 = (lane & 2) ? dw : dw_other;
        const float dx0 = smem.pipe.xd[col_off + c0][bi];
        const float dx1 = smem.pipe.xd[col_off + c0 + 1][bi];
        const float dx2 = smem.pipe.xd[col_off + c0 + 4][bi];
        const float dx3 = smem.pipe.xd[col_off + c0 + 5][bi];
        acc[0] = fmaf(dw0 * dx0, d[0], acc[0]);
        acc[1] = fmaf(dw0 * dx1, d[1], acc[1]);
        acc[2] = fmaf(dw1 * dx0, d[2], acc[2]);
        acc[3] = fmaf(dw1 * dx1, d[3], acc[3]);
        acc[4] = fmaf(dw0 * dx2, d[4], acc[4]);
        acc[5] = fmaf(dw0 * dx3, d[5], acc[5]);
        acc[6] = fmaf(dw1 * dx2, d[6], acc[6]);
        acc[7] = fmaf(dw1 * dx3, d[7], acc[7]);
    };

    fetch_w(0);
    store_w();
    load_x(0);
    __syncthreads();

    for (int st = 0; st < steps; ++st) {
        if (st + 1 < steps) fetch_w(st + 1);
        if constexpr (!WIDE_M) {
            mma_one_block(warp * 4 + qp, 0, acc0);
        } else {
            const int kb2 = qp & 1;
            const int cg = qp >> 1;
            mma_one_block(warp * 4 + kb2, cg * 8, acc0);
            mma_one_block(warp * 4 + 2 + kb2, cg * 8, acc1);
        }
        __syncthreads();
        if (st + 1 < steps) {
            store_w();
            load_x(st + 1);
            __syncthreads();
        }
    }

    if constexpr (!WIDE_M) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] += __shfl_xor_sync(0xffffffffu, acc0[i], 4);
            acc0[i] += __shfl_xor_sync(0xffffffffu, acc0[i], 8);
        }
        if (qp == 0) {
            const int r0 = (gg & 1) + 4 * hh;
            const int r1 = r0 + 2;
            smem.out[warp][r0][c0] = acc0[0];
            smem.out[warp][r0][c0 + 1] = acc0[1];
            smem.out[warp][r1][c0] = acc0[2];
            smem.out[warp][r1][c0 + 1] = acc0[3];
            smem.out[warp][r0][c0 + 4] = acc0[4];
            smem.out[warp][r0][c0 + 5] = acc0[5];
            smem.out[warp][r1][c0 + 4] = acc0[6];
            smem.out[warp][r1][c0 + 5] = acc0[7];
        }
    } else {
        const int kb2 = qp & 1;
        const int cg = qp >> 1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] += __shfl_xor_sync(0xffffffffu, acc0[i], 4);
            acc1[i] += __shfl_xor_sync(0xffffffffu, acc1[i], 4);
            acc0[i] += acc1[i];
        }
        if (kb2 == 0) {
            const int col_off = cg * 8;
            const int r0 = (gg & 1) + 4 * hh;
            const int r1 = r0 + 2;
            smem.out[warp][r0][col_off + c0] = acc0[0];
            smem.out[warp][r0][col_off + c0 + 1] = acc0[1];
            smem.out[warp][r1][col_off + c0] = acc0[2];
            smem.out[warp][r1][col_off + c0 + 1] = acc0[3];
            smem.out[warp][r0][col_off + c0 + 4] = acc0[4];
            smem.out[warp][r0][col_off + c0 + 5] = acc0[5];
            smem.out[warp][r1][col_off + c0 + 4] = acc0[6];
            smem.out[warp][r1][col_off + c0 + 5] = acc0[7];
        }
    }
    __syncthreads();

    constexpr int total_out = 8 * MCAP;
    for (int idx = tid; idx < total_out; idx += NT) {
        const int r = idx / MCAP;
        const int c = idx % MCAP;
        if (c < ncols && row0 + r < seg_n_out) {
            const float sum = (smem.out[0][r][c] + smem.out[1][r][c]) + (smem.out[2][r][c] + smem.out[3][r][c]);
            y_base[(size_t) c * (size_t) seg_n_out + (size_t) (row0 + r)] = sum;
        }
    }
#else
    (void) plan; (void) x_q8_1; (void) n_in; (void) ncols;
    __trap();
#endif
}

template <int NB_STEP>
struct Q8DecSmallCfg;

template <>
struct Q8DecSmallCfg<4> {
    static constexpr int NWARPS = 2;
    static constexpr int NT = 64;
    static constexpr int W_STRIDE = 38;
    static constexpr int ROW_STEP_B = 136;
};

template <>
struct Q8DecSmallCfg<2> {
    static constexpr int NWARPS = 1;
    static constexpr int NT = 32;
    static constexpr int W_STRIDE = 22;
    static constexpr int ROW_STEP_B = 68;
};

template <int NB_STEP, bool WIDE_M>
__global__ void __launch_bounds__(Q8DecSmallCfg<NB_STEP>::NT, WIDE_M ? 4 : 6)
q8_volta_decode_seg_small_kernel(Q8VoltaSegPlan plan, const Q81Block* __restrict__ x_q8_1, int n_in, int ncols) {
#if STRATA_Q8V_SM70
    using Cfg = Q8DecSmallCfg<NB_STEP>;
    constexpr int NT = Cfg::NT;
    constexpr int NWARPS = Cfg::NWARPS;
    constexpr int W_STRIDE = Cfg::W_STRIDE;
    constexpr int MCAP = WIDE_M ? 16 : 8;

    __shared__ __align__(16) uint32_t s_w[16][W_STRIDE];
    __shared__ __align__(16) __half s_xh[NB_STEP][MCAP][36];
    __shared__ float s_xd[MCAP][NB_STEP];
    __shared__ float s_out[NWARPS][16][MCAP];

    const uint8_t* w_base = nullptr;
    float* y_base = nullptr;
    int seg_n_out = 0, blk_in_seg = 0;
    q8v_resolve_seg(plan, (int) blockIdx.x, w_base, y_base, seg_n_out, blk_in_seg);
    const int row0 = blk_in_seg * 16;

    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int qp = (lane & 15) >> 2;
    const int tl = (lane & 3) + 4 * (lane >> 4);
    const int gg = lane & 3;
    const int hh = lane >> 4;
    const int rg = qp & 1;
    const int c0 = gg & 2;

    const int nb = n_in >> 5;
    const int steps = nb / NB_STEP;
    const size_t row_bytes = (size_t) nb * 34u;

    if (ncols < MCAP) {
        const int pad_c = MCAP - ncols;
        const int total_z = pad_c * NB_STEP * 8;
        for (int i = tid; i < total_z; i += NT) {
            const int k4 = i & 7;
            const int t0 = i >> 3;
            const int bi = t0 & (NB_STEP - 1);
            const int c = ncols + (t0 / NB_STEP);
            *reinterpret_cast<uint2*>(&s_xh[bi][c][k4 * 4]) = make_uint2(0u, 0u);
        }
        const int total_zd = pad_c * NB_STEP;
        for (int i = tid; i < total_zd; i += NT) {
            const int bi = i & (NB_STEP - 1);
            const int c = ncols + (i / NB_STEP);
            s_xd[c][bi] = 0.0f;
        }
    }

    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int hw = tid >> 4;
    const int l16 = tid & 15;
    constexpr int n_hw = NT >> 4;
    const int total_x_halves = ncols * NB_STEP * 2;

    auto mma_one_block = [&](int bi, int col_off, float (&acc)[8]) {
        const int r = rg * 8 + tl;
        const int pair = bi >> 1;
        const int odd = bi & 1;
        const uint32_t* const wr = &s_w[r][pair * 17];
        float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const __half* const xb = &s_xh[bi][col_off + tl][0];
        uint32_t prev = odd ? 0u : wr[0];
        const float dw = __half2float(__ushort_as_half((uint16_t) (odd ? (wr[8] >> 16) : (prev & 0xFFFFu))));
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t q4;
            if (odd == 0) {
                const uint32_t cur = wr[kk + 1];
                q4 = __funnelshift_r(prev, cur, 16);
                prev = cur;
            } else {
                q4 = wr[9 + kk];
            }
            q8v_mma_m8n8k4(d, q8v_i8x4_to_f16x4(q4), *reinterpret_cast<const uint2*>(xb + kk * 4));
        }
        const float dw_other = __shfl_xor_sync(0xffffffffu, dw, 2);
        const float dw0 = (lane & 2) ? dw_other : dw;
        const float dw1 = (lane & 2) ? dw : dw_other;
        const float dx0 = s_xd[col_off + c0][bi];
        const float dx1 = s_xd[col_off + c0 + 1][bi];
        const float dx2 = s_xd[col_off + c0 + 4][bi];
        const float dx3 = s_xd[col_off + c0 + 5][bi];
        acc[0] = fmaf(dw0 * dx0, d[0], acc[0]);
        acc[1] = fmaf(dw0 * dx1, d[1], acc[1]);
        acc[2] = fmaf(dw1 * dx0, d[2], acc[2]);
        acc[3] = fmaf(dw1 * dx1, d[3], acc[3]);
        acc[4] = fmaf(dw0 * dx2, d[4], acc[4]);
        acc[5] = fmaf(dw0 * dx3, d[5], acc[5]);
        acc[6] = fmaf(dw1 * dx2, d[6], acc[6]);
        acc[7] = fmaf(dw1 * dx3, d[7], acc[7]);
    };

    for (int st = 0; st < steps; ++st) {
#pragma unroll
        for (int r = hw; r < 16; r += n_hw) {
            const int gr = row0 + r;
            const bool valid = gr < seg_n_out;
            const uint8_t* const rp = w_base + (size_t) (valid ? gr : 0) * row_bytes + (size_t) st * Cfg::ROW_STEP_B;
            if constexpr (NB_STEP == 4) {
                const uint2* const wp2 = reinterpret_cast<const uint2*>(rp);
                *reinterpret_cast<uint2*>(&s_w[r][l16 * 2]) = valid ? __ldg(wp2 + l16) : make_uint2(0u, 0u);
                if (l16 == 0) *reinterpret_cast<uint2*>(&s_w[r][32]) = valid ? __ldg(wp2 + 16) : make_uint2(0u, 0u);
            } else {
                const uint32_t* const wp1 = reinterpret_cast<const uint32_t*>(rp);
                s_w[r][l16] = valid ? __ldg(wp1 + l16) : 0u;
                if (l16 == 0) s_w[r][16] = valid ? __ldg(wp1 + 16) : 0u;
            }
        }
        for (int idx = tid; idx < total_x_halves; idx += NT) {
            const int half_b = idx & 1;
            const int lin_b = idx >> 1;
            const int bi = lin_b & (NB_STEP - 1);
            const int c = lin_b / NB_STEP;
            const Q81Block* const xb = x_q8_1 + (size_t) c * (size_t) nb + st * NB_STEP + bi;
            const uint32_t* const xqp = reinterpret_cast<const uint32_t*>(xb->qs) + half_b * 4;
            uint2* const dst = reinterpret_cast<uint2*>(&s_xh[bi][c][half_b * 16]);
            dst[0] = q8v_i8x4_to_f16x4(__ldg(xqp + 0));
            dst[1] = q8v_i8x4_to_f16x4(__ldg(xqp + 1));
            dst[2] = q8v_i8x4_to_f16x4(__ldg(xqp + 2));
            dst[3] = q8v_i8x4_to_f16x4(__ldg(xqp + 3));
            if (half_b == 0) s_xd[c][bi] = __low2float(__ldg(&xb->ds));
        }
        __syncthreads();
        if constexpr (!WIDE_M) {
            mma_one_block(warp * 2 + (qp >> 1), 0, acc0);
        } else {
            const int cg = qp >> 1;
            mma_one_block(warp * 2 + 0, cg * 8, acc0);
            mma_one_block(warp * 2 + 1, cg * 8, acc1);
        }
        __syncthreads();
    }

    if constexpr (!WIDE_M) {
        const int kb = qp >> 1;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc0[i] += __shfl_xor_sync(0xffffffffu, acc0[i], 8);
        if (kb == 0) {
            const int r0 = rg * 8 + (gg & 1) + 4 * hh;
            const int r1 = r0 + 2;
            s_out[warp][r0][c0] = acc0[0];
            s_out[warp][r0][c0 + 1] = acc0[1];
            s_out[warp][r1][c0] = acc0[2];
            s_out[warp][r1][c0 + 1] = acc0[3];
            s_out[warp][r0][c0 + 4] = acc0[4];
            s_out[warp][r0][c0 + 5] = acc0[5];
            s_out[warp][r1][c0 + 4] = acc0[6];
            s_out[warp][r1][c0 + 5] = acc0[7];
        }
    } else {
        const int cg = qp >> 1;
        const int col_off = cg * 8;
        const int r0 = rg * 8 + (gg & 1) + 4 * hh;
        const int r1 = r0 + 2;
        s_out[warp][r0][col_off + c0] = acc0[0] + acc1[0];
        s_out[warp][r0][col_off + c0 + 1] = acc0[1] + acc1[1];
        s_out[warp][r1][col_off + c0] = acc0[2] + acc1[2];
        s_out[warp][r1][col_off + c0 + 1] = acc0[3] + acc1[3];
        s_out[warp][r0][col_off + c0 + 4] = acc0[4] + acc1[4];
        s_out[warp][r0][col_off + c0 + 5] = acc0[5] + acc1[5];
        s_out[warp][r1][col_off + c0 + 4] = acc0[6] + acc1[6];
        s_out[warp][r1][col_off + c0 + 5] = acc0[7] + acc1[7];
    }
    __syncthreads();

    constexpr int total_out = 16 * MCAP;
    for (int idx = tid; idx < total_out; idx += NT) {
        const int r = idx / MCAP;
        const int c = idx % MCAP;
        if (c < ncols && row0 + r < seg_n_out) {
            const float sum = (NWARPS == 2) ? (s_out[0][r][c] + s_out[1][r][c]) : s_out[0][r][c];
            y_base[(size_t) c * (size_t) seg_n_out + (size_t) (row0 + r)] = sum;
        }
    }
#else
    (void) plan; (void) x_q8_1; (void) n_in; (void) ncols;
    __trap();
#endif
}

bool device_is_sm70() {
    static int cache[16] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    if (cache[dev] == 0) {
        int maj = 0, min = 0;
        if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
            maj == 7 && min == 0) {
            cache[dev] = 1;
        } else {
            cache[dev] = -1;
        }
    }
    return cache[dev] > 0;
}

int dec_seg_mode() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_DEC_SEG_V70");
        if (e == nullptr || e[0] == 0) return 1;
        return std::atoi(e);
    }();
    return v;
}

int pf_tile_override() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_PF_Q8_V70_TILE");
        return (e != nullptr && e[0] != 0) ? std::atoi(e) : 0;
    }();
    return v;
}

int dec_br8_max_n() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_DEC_Q8_V70_BR8_MAX_N");
        return (e != nullptr && e[0] != 0) ? std::atoi(e) : 3072;
    }();
    return v;
}

template <int BM, int BN, int WM, int WN, bool HAS_BETA, int MB>
void ensure_dyn_smem(int bytes) {
    static int done[16] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 16 && !done[dev]) {
        cudaFuncSetAttribute(q8_volta_prefill_kernel<BM, BN, WM, WN, HAS_BETA, MB>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        done[dev] = 1;
    }
}

#endif  // !__HIPCC__

}  // namespace

bool q8_volta_tc_supported() {
#if !defined(__HIPCC__)
    return device_is_sm70();
#else
    return false;
#endif
}

bool q8_volta_prefill_enabled() {
#if !defined(__HIPCC__)
    static const bool env_on = [] {
        const char* e = std::getenv("STRATA_PF_Q8_V70");
        return e == nullptr || e[0] == 0 || std::atoi(e) != 0;
    }();
    return env_on && device_is_sm70();
#else
    return false;
#endif
}

bool q8_volta_prefill_should_use(int64_t T, int64_t N, int64_t K) {
#if !defined(__HIPCC__)
    if (!q8_volta_prefill_enabled() || T < 32 || K < 64 || (K & 63) != 0) return false;
    static const int64_t max_t_override = [] {
        const char* e = std::getenv("STRATA_PF_Q8_V70_MAX_T");
        return (e != nullptr && e[0] != 0) ? (int64_t) std::atoll(e) : (int64_t) 0;
    }();
    if (max_t_override > 0) return T <= max_t_override;
    // Tuned on Tesla V100 (sm_70): fused Q8_0 W8A16 beats dequant_f16+cuBLAS by 1.17-2.86x
    // on wide dense projections at short/medium T where global FP16 weight write/read dominates;
    // at larger T, cuBLAS + STRATA_DEQUANT_SIDE reaches 85-98.7 TFLOP/s.
    if (N >= 10240) return T <= 256;
    if (N >= 6144) return T <= 128;
    return false;
#else
    (void) T; (void) N; (void) K;
    return false;
#endif
}

bool q8_volta_decode_enabled() {
#if !defined(__HIPCC__)
    return dec_seg_mode() != 0 && device_is_sm70();
#else
    return false;
#endif
}

bool q8_volta_decode_seg_enabled() {
#if !defined(__HIPCC__)
    return dec_seg_mode() == 1 && device_is_sm70();
#else
    return false;
#endif
}

bool q8_volta_prefill_gemm(const uint16_t* X_f16, const void* W_q8_0, float* Y,
                           int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                           void* stream) {
#if !defined(__HIPCC__)
    if (!device_is_sm70() || !X_f16 || !W_q8_0 || !Y || !stream) return false;
    if (T < 32 || N < 1 || K < 64 || (K & 63) != 0) return false;
    if (((uintptr_t) X_f16 & 15u) != 0 || ((uintptr_t) W_q8_0 & 3u) != 0) return false;

    const int64_t ldy_eff = ldy > 0 ? ldy : N;
    const auto* wb = static_cast<const uint8_t*>(W_q8_0);
    const auto s = static_cast<cudaStream_t>(stream);
    const bool has_beta = (beta != 0.0f);

    int tile = pf_tile_override();
    if (tile == 0) {
        if (T >= 128 && N >= 1024) tile = 3;
        else if (N >= 1024) tile = 2;
        else tile = 1;
    }

    auto launch = [&](auto bm_tag, auto bn_tag, auto wm_tag, auto wn_tag, auto mb_tag) {
        constexpr int BM = decltype(bm_tag)::value;
        constexpr int BN = decltype(bn_tag)::value;
        constexpr int WM = decltype(wm_tag)::value;
        constexpr int WN = decltype(wn_tag)::value;
        constexpr int MB = decltype(mb_tag)::value;
        constexpr int smem_bytes = 2 * (BM + BN) * 8 * (int) sizeof(uint4);
        const unsigned grid = (unsigned) (((T + BM - 1) / BM) * ((N + BN - 1) / BN));
        if (has_beta) {
            if constexpr (smem_bytes > 48 * 1024) ensure_dyn_smem<BM, BN, WM, WN, true, MB>(smem_bytes);
            q8_volta_prefill_kernel<BM, BN, WM, WN, true, MB><<<grid, WM * WN * 32, smem_bytes, s>>>(
                X_f16, wb, Y, T, N, K, ldy_eff, beta);
        } else {
            if constexpr (smem_bytes > 48 * 1024) ensure_dyn_smem<BM, BN, WM, WN, false, MB>(smem_bytes);
            q8_volta_prefill_kernel<BM, BN, WM, WN, false, MB><<<grid, WM * WN * 32, smem_bytes, s>>>(
                X_f16, wb, Y, T, N, K, ldy_eff, 0.0f);
        }
    };

    if (tile == 3) {
        // 128x128x64 DB (64 KB dyn smem, 1 CTA/SM, 8 warps/SM, zero spill, full frag ping-pong)
        launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{},
               std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{},
               std::integral_constant<int, 1>{});
    } else if (tile == 2) {
        // 64x128x64 DB (48 KB dyn smem, 1 CTA/SM, 8 warps/SM, zero spill)
        launch(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{},
               std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{},
               std::integral_constant<int, 1>{});
    } else {
        // 64x64x64 DB (32 KB dyn smem, 2 CTA/SM, 8 warps/SM, zero spill)
        launch(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{},
               std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{},
               std::integral_constant<int, 2>{});
    }
    return true;
#else
    (void) X_f16; (void) W_q8_0; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) beta; (void) stream;
    return false;
#endif
}

bool q8_volta_decode_seg(const Q8VoltaSeg* segs, int n_segs, const void* x_q8_1,
                         int n_in, int ncols, void* stream) {
#if !defined(__HIPCC__)
    if (!device_is_sm70() || !segs || n_segs < 1 || n_segs > 4 || !x_q8_1 || !stream) return false;
    if (ncols < 1 || ncols > 16 || n_in < 64 || (n_in & 63) != 0) return false;

    bool align16 = ((n_in & 255) == 0);
    bool align8 = ((n_in & 127) == 0);
    int total_n = 0;
    for (int i = 0; i < n_segs; ++i) {
        if (!segs[i].w || !segs[i].y || segs[i].n_out < 1) return false;
        const uintptr_t p = reinterpret_cast<uintptr_t>(segs[i].w);
        if ((p & 3u) != 0) return false;
        if ((p & 15u) != 0) align16 = false;
        if ((p & 7u) != 0) align8 = false;
        total_n += segs[i].n_out;
    }

    const bool use_br8_nb16 = align16 && ((n_in & 511) == 0) && (total_n <= dec_br8_max_n());
    const int br = use_br8_nb16 ? 8 : 16;

    Q8VoltaSegPlan plan{};
    plan.n_segs = n_segs;
    int total_blks = 0;
    for (int i = 0; i < n_segs; ++i) {
        total_blks += (segs[i].n_out + br - 1) / br;
        plan.seg_blk_end[i] = total_blks;
        plan.w[i] = static_cast<const uint8_t*>(segs[i].w);
        plan.y[i] = segs[i].y;
        plan.n_out[i] = segs[i].n_out;
    }

    const auto* xq = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const bool wide = (ncols > 8);

    if (use_br8_nb16) {
        if (wide) q8_volta_decode_seg_br8_nb16_kernel<true><<<(unsigned) total_blks, 128, 0, s>>>(plan, xq, n_in, ncols);
        else q8_volta_decode_seg_br8_nb16_kernel<false><<<(unsigned) total_blks, 128, 0, s>>>(plan, xq, n_in, ncols);
        return true;
    }

    if (align16) {
        if (wide) q8_volta_decode_seg_nb8_kernel<true><<<(unsigned) total_blks, 128, 0, s>>>(plan, xq, n_in, ncols);
        else q8_volta_decode_seg_nb8_kernel<false><<<(unsigned) total_blks, 128, 0, s>>>(plan, xq, n_in, ncols);
    } else if (align8) {
        if (wide) q8_volta_decode_seg_small_kernel<4, true><<<(unsigned) total_blks, 64, 0, s>>>(plan, xq, n_in, ncols);
        else q8_volta_decode_seg_small_kernel<4, false><<<(unsigned) total_blks, 64, 0, s>>>(plan, xq, n_in, ncols);
    } else {
        if (wide) q8_volta_decode_seg_small_kernel<2, true><<<(unsigned) total_blks, 32, 0, s>>>(plan, xq, n_in, ncols);
        else q8_volta_decode_seg_small_kernel<2, false><<<(unsigned) total_blks, 32, 0, s>>>(plan, xq, n_in, ncols);
    }
    return true;
#else
    (void) segs; (void) n_segs; (void) x_q8_1; (void) n_in; (void) ncols; (void) stream;
    return false;
#endif
}

bool q8_volta_decode_mmvq(const void* W_q8_0, const void* x_q8_1, float* y,
                          int n_in, int n_out, int ncols, void* stream) {
    const Q8VoltaSeg seg{W_q8_0, y, n_out};
    return q8_volta_decode_seg(&seg, 1, x_q8_1, n_in, ncols, stream);
}

}  // namespace strata::kernels
