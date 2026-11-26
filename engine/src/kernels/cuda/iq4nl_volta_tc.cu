// src/kernels/cuda/iq4nl_volta_tc.cu - Clean-room MIT Volta (sm_70) IQ4_NL down kernel
// (coalesced burst uint4 weight staging + exact dp4a for n_e <= 4 + exact mma.m8n8k4 for n_e >= 5).
//
// MIT License
// Copyright (c) 2026 Strata contributors
#include "strata/kernels/iq4nl_volta_tc.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

#if !defined(__HIPCC__)

namespace {

struct __align__(4) Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36, "block_q8_1 size must be 36 bytes");

// Signed int8 kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38, 53,69,89,113}
constexpr uint32_t kIq4Sig0 = 0xBFAD9881u;
constexpr uint32_t kIq4Sig1 = 0xF6EADDCFu;
constexpr uint32_t kIq4Sig2 = 0x26190D01u;
constexpr uint32_t kIq4Sig3 = 0x71594535u;

// Pre-XORed with 0x80 (+128 bias) for register-only exact int8 -> FP16 conversion
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
    const bool vr0 = r0 < nrow;
    const bool vr1 = r1 < nrow;
    if (c0_col + 0 < cn) {
        const size_t base = (size_t) ent_dst[c0 + c0_col + 0] * (size_t) nrow;
        if (vr0) out[base + r0] = acc[0];
        if (vr1) out[base + r1] = acc[2];
    }
    if (c0_col + 1 < cn) {
        const size_t base = (size_t) ent_dst[c0 + c0_col + 1] * (size_t) nrow;
        if (vr0) out[base + r0] = acc[1];
        if (vr1) out[base + r1] = acc[3];
    }
    if (c0_col + 4 < cn) {
        const size_t base = (size_t) ent_dst[c0 + c0_col + 4] * (size_t) nrow;
        if (vr0) out[base + r0] = acc[4];
        if (vr1) out[base + r1] = acc[6];
    }
    if (c0_col + 5 < cn) {
        const size_t base = (size_t) ent_dst[c0 + c0_col + 5] * (size_t) nrow;
        if (vr0) out[base + r0] = acc[5];
        if (vr1) out[base + r1] = acc[7];
    }
}

__device__ __forceinline__ void load_iq4_block(const uint32_t* __restrict__ sw_row, int bi, int odd,
                                               uint16_t& d_bits, uint32_t& q0, uint32_t& q1,
                                               uint32_t& q2, uint32_t& q3) {
    const uint32_t* const wp = sw_row + (bi >> 1) * 9;
    if (!odd) {
        const uint32_t w0 = wp[0];
        const uint32_t w1 = wp[1];
        const uint32_t w2 = wp[2];
        const uint32_t w3 = wp[3];
        const uint32_t w4 = wp[4];
        d_bits = (uint16_t) (w0 & 0xFFFFu);
        q0 = __funnelshift_r(w0, w1, 16);
        q1 = __funnelshift_r(w1, w2, 16);
        q2 = __funnelshift_r(w2, w3, 16);
        q3 = __funnelshift_r(w3, w4, 16);
    } else {
        d_bits = (uint16_t) (wp[4] >> 16);
        q0 = wp[5];
        q1 = wp[6];
        q2 = wp[7];
        q3 = wp[8];
    }
}

template <int NT, int MAX_ITERS>
__device__ __forceinline__ void stage_w_burst(const uint8_t* __restrict__ w_base,
                                              uint32_t* __restrict__ s_w,
                                              int valid_rows, int row_w, int tid) {
    const int n_u4 = (valid_rows * row_w) >> 2;
    if ((((uintptr_t) w_base) & 15u) == 0u) {
        const uint4* const src = reinterpret_cast<const uint4*>(w_base);
        uint4* const dst = reinterpret_cast<uint4*>(s_w);
        uint4 buf[MAX_ITERS];
#pragma unroll
        for (int k = 0; k < MAX_ITERS; ++k) {
            const int i = tid + k * NT;
            if (i < n_u4) buf[k] = __ldg(src + i);
        }
#pragma unroll
        for (int k = 0; k < MAX_ITERS; ++k) {
            const int i = tid + k * NT;
            if (i < n_u4) dst[i] = buf[k];
        }
    } else {
        const uint32_t* const src = reinterpret_cast<const uint32_t*>(w_base);
        const int n_u32 = valid_rows * row_w;
        for (int i = tid; i < n_u32; i += NT) s_w[i] = __ldg(src + i);
    }
}

// ----------------------------------------------------------------------------
// Volta sm_70 IQ4_NL down kernel:
//   BR = 32 rows/CTA, NT = 128 threads (4 warps), 5 CTA/SM = 20 warps/SM,
//   17.9 KB smem (NB=20) / 14.3 KB (NB=16), REG <= 96, STACK:0 LOCAL:0.
//   - Burst-issued coalesced uint4 staging of raw block_iq4_nl weights (once per group)
//   - Per-group CTA-uniform adaptive dispatch with 100% BIT-IDENTICAL accumulation order:
//     * n_e <= 4: exact dp4a + register-only i8_table16 (4 threads/row, 0 FP16 conversion)
//     * n_e >= 5: exact mma.m8n8k4 over bank-free int8^0x80 X (stride 72 -> 0 bank conflicts)
// ----------------------------------------------------------------------------
template <int NB>
__global__ void __launch_bounds__(128, 5)
iq4nl_down_v70_kernel(const unsigned long long* __restrict__ grp_ptr,
                      const int32_t* __restrict__ grp_start,
                      const int32_t* __restrict__ n_groups,
                      const int32_t* __restrict__ ent_dst,
                      const Q81Block* __restrict__ hq,
                      uint64_t down_off, int nrow,
                      float* __restrict__ out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    constexpr int BR = 32;
    constexpr int NT = 128;
    constexpr int ROW_W = (NB * 18) / 4;
    constexpr int MAX_U4_ITERS = ((BR * ROW_W) / 4 + NT - 1) / NT;
    constexpr int XQ_STRIDE = 72;   // 64 + 8 uint32 -> 72 mod 32 = 8 -> 0 bank conflicts across 4 qps

    __shared__ __align__(16) uint32_t s_w[BR * ROW_W];
    __shared__ union {
        struct {
            int xq[4][NB][8];
            float xd[4][NB];
        } dp;
        struct {
            uint32_t xq[NB * XQ_STRIDE];
            float xd[NB][8];
        } tc;
    } u;

    const int ng = *n_groups;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int qp = (lane & 15) >> 2;
    const int tl = (lane & 3) + 4 * (lane >> 4);
    const int gg = lane & 3;
    const int hh = lane >> 4;
    const int c0_col = gg & 2;

    const int row_block = (int) blockIdx.x * BR;
    if (row_block >= nrow) return;
    const int valid_rows = min(BR, nrow - row_block);

    for (int g = (int) blockIdx.y; g < ng; g += (int) gridDim.y) {
        const int e0 = grp_start[g];
        const int e1 = grp_start[g + 1];
        if (e0 >= e1) continue;
        const int n_e = e1 - e0;

        const uint8_t* const w_base =
            reinterpret_cast<const uint8_t*>(grp_ptr[g]) + down_off + (size_t) row_block * (size_t) (NB * 18);
        stage_w_burst<NT, MAX_U4_ITERS>(w_base, s_w, valid_rows, ROW_W, tid);

        if (n_e <= 4) {
            const int total_b = n_e * NB;
            for (int idx = tid; idx < total_b * 2; idx += NT) {
                const int half_b = idx & 1;
                const int lin_b = idx >> 1;
                const int bi = lin_b % NB;
                const int c = lin_b / NB;
                const Q81Block* const xb = hq + (size_t) (e0 + c) * NB + bi;
                const int* const xqp = reinterpret_cast<const int*>(xb->qs) + half_b * 4;
                u.dp.xq[c][bi][half_b * 4 + 0] = __ldg(xqp + 0);
                u.dp.xq[c][bi][half_b * 4 + 1] = __ldg(xqp + 1);
                u.dp.xq[c][bi][half_b * 4 + 2] = __ldg(xqp + 2);
                u.dp.xq[c][bi][half_b * 4 + 3] = __ldg(xqp + 3);
                if (half_b == 0) u.dp.xd[c][bi] = __low2float(__ldg(&xb->ds));
            }
            __syncthreads();

            const int sub_row = tid >> 2;
            const int t4 = tid & 3;
            const int odd4 = t4 & 1;
            const int safe_r = (sub_row < valid_rows) ? sub_row : 0;
            const uint32_t* const sw_row = s_w + safe_r * ROW_W;

            auto run_dp = [&](auto cn_tag) {
                constexpr int CN = decltype(cn_tag)::value;
                float acc[CN];
#pragma unroll
                for (int c = 0; c < CN; ++c) acc[c] = 0.0f;
#pragma unroll 1
                for (int bi = t4; bi < NB; bi += 4) {
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
                        const int* const xq = u.dp.xq[c][bi];
                        int sumi = 0;
                        sumi = __dp4a((int) lo0, xq[0], sumi);
                        sumi = __dp4a((int) lo1, xq[1], sumi);
                        sumi = __dp4a((int) lo2, xq[2], sumi);
                        sumi = __dp4a((int) lo3, xq[3], sumi);
                        sumi = __dp4a((int) hi0, xq[4], sumi);
                        sumi = __dp4a((int) hi1, xq[5], sumi);
                        sumi = __dp4a((int) hi2, xq[6], sumi);
                        sumi = __dp4a((int) hi3, xq[7], sumi);
                        acc[c] = fmaf(dw * u.dp.xd[c][bi], (float) sumi, acc[c]);
                    }
                }
#pragma unroll
                for (int c = 0; c < CN; ++c) {
                    acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], 2);
                    acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], 1);
                }
                if (t4 == 0 && sub_row < valid_rows) {
                    const int out_r = row_block + sub_row;
#pragma unroll
                    for (int c = 0; c < CN; ++c) {
                        out[(size_t) ent_dst[e0 + c] * (size_t) nrow + out_r] = acc[c];
                    }
                }
            };
            if (n_e == 1) run_dp(std::integral_constant<int, 1>{});
            else if (n_e == 2) run_dp(std::integral_constant<int, 2>{});
            else if (n_e == 3) run_dp(std::integral_constant<int, 3>{});
            else run_dp(std::integral_constant<int, 4>{});
            __syncthreads();
            continue;
        }

        // Exact mma.m8n8k4 path for n_e >= 5 (chunks of 8 cols; W stays resident in s_w!)
        const int odd = qp & 1;
        const int local_r = warp * 8 + tl;
        const int safe_r = (local_r < valid_rows) ? local_r : 0;
        const uint32_t* const sw_row = s_w + safe_r * ROW_W;

        for (int c0 = e0; c0 < e1; c0 += 8) {
            const int cn = min(8, e1 - c0);
            if (c0 > e0) __syncthreads();

            for (int idx = tid; idx < 8 * NB * 2; idx += NT) {
                const int half_b = idx & 1;
                const int lin_b = idx >> 1;
                const int bi = lin_b % NB;
                const int c = lin_b / NB;
                uint32_t* const xq_dst = u.tc.xq + bi * XQ_STRIDE + half_b * 32 + c;
                if (c < cn) {
                    const Q81Block* const xb = hq + (size_t) (c0 + c) * NB + bi;
                    const uint32_t* const xqp = reinterpret_cast<const uint32_t*>(xb->qs) + half_b * 4;
                    xq_dst[0]  = __ldg(xqp + 0) ^ 0x80808080u;
                    xq_dst[8]  = __ldg(xqp + 1) ^ 0x80808080u;
                    xq_dst[16] = __ldg(xqp + 2) ^ 0x80808080u;
                    xq_dst[24] = __ldg(xqp + 3) ^ 0x80808080u;
                    if (half_b == 0) u.tc.xd[bi][c] = __low2float(__ldg(&xb->ds));
                } else {
                    xq_dst[0]  = 0x80808080u;
                    xq_dst[8]  = 0x80808080u;
                    xq_dst[16] = 0x80808080u;
                    xq_dst[24] = 0x80808080u;
                    if (half_b == 0) u.tc.xd[bi][c] = 0.0f;
                }
            }
            __syncthreads();

            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

#pragma unroll 1
            for (int bi = qp; bi < NB; bi += 4) {
                uint16_t d_bits;
                uint32_t q0, q1, q2, q3;
                load_iq4_block(sw_row, bi, odd, d_bits, q0, q1, q2, q3);
                const float dw = __half2float(__ushort_as_half(d_bits));
                const uint32_t* const xq_bi = u.tc.xq + bi * XQ_STRIDE + tl;
                float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                auto step_q = [&](uint32_t qv, int j) {
                    uint32_t lo, hi;
                    iq4_decode_4bytes_bias128(qv, lo, hi);
                    mma_m8n8k4(d, u8x4_bias128_to_f16x4(lo), u8x4_bias128_to_f16x4(xq_bi[j * 8]));
                    mma_m8n8k4(d, u8x4_bias128_to_f16x4(hi), u8x4_bias128_to_f16x4(xq_bi[(j + 4) * 8]));
                };
                step_q(q0, 0); step_q(q1, 1); step_q(q2, 2); step_q(q3, 3);
                iq4_scale_accum(acc, d, dw, lane, c0_col, u.tc.xd[bi]);
            }

#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 4);
            if (qp == 0) {
                iq4_store_tile8(acc, row_block + warp * 8, nrow, gg, hh, cn, c0, ent_dst, out);
            }
        }
        __syncthreads();
    }
#else
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) ent_dst; (void) hq;
    (void) down_off; (void) nrow; (void) out;
#endif
}

std::atomic<int> g_mode_override{-1};

int default_env_mode() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_IQ4_V70");
        if (e == nullptr || e[0] == 0) return 1;
        return std::atoi(e);
    }();
    return v;
}

}  // namespace

bool iq4nl_volta_tc_supported() {
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

int iq4nl_volta_tc_mode() {
    if (!iq4nl_volta_tc_supported()) return 0;
    const int ov = g_mode_override.load(std::memory_order_relaxed);
    return ov >= 0 ? ov : default_env_mode();
}

void iq4nl_volta_tc_set_mode(int mode) {
    g_mode_override.store(mode, std::memory_order_relaxed);
}

bool iq4nl_volta_down_grouped(const NativeExpertLayout& L, int64_t gy,
                              const unsigned long long* grp_ptr, const int32_t* grp_start,
                              const int32_t* n_groups, const int32_t* ent_dst,
                              const void* hq_q8_1, float* out, void* stream) {
    if (iq4nl_volta_tc_mode() <= 0) return false;
    if (L.d_type != 20 || L.tail_off != 0 || gy <= 0 || L.n_embd <= 0) return false;
    if ((L.n_embd % 32) != 0) return false;
    const int64_t nb = L.n_ff / 32;
    if ((L.n_ff != 640 && L.n_ff != 512) || L.d_row != (size_t) nb * 18u) return false;
    if ((L.down_off & 3u) != 0) return false;

    const int nrow = (int) L.n_embd;
    const auto* hq = reinterpret_cast<const Q81Block*>(hq_q8_1);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const dim3 grid((unsigned) ((nrow + 31) / 32), (unsigned) gy);
    if (nb == 20) {
        iq4nl_down_v70_kernel<20><<<grid, 128, 0, s>>>(
            grp_ptr, grp_start, n_groups, ent_dst, hq, L.down_off, nrow, out);
    } else {
        iq4nl_down_v70_kernel<16><<<grid, 128, 0, s>>>(
            grp_ptr, grp_start, n_groups, ent_dst, hq, L.down_off, nrow, out);
    }
    return cudaGetLastError() == cudaSuccess;
}

#else  // __HIPCC__

bool iq4nl_volta_tc_supported() { return false; }
int iq4nl_volta_tc_mode() { return 0; }
void iq4nl_volta_tc_set_mode(int) {}
bool iq4nl_volta_down_grouped(const NativeExpertLayout&, int64_t, const unsigned long long*, const int32_t*,
                              const int32_t*, const int32_t*, const void*, float*, void*) {
    return false;
}

#endif

}  // namespace strata::kernels
