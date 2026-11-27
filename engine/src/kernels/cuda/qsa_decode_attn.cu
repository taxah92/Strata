// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/core/sm75_series.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#if defined(STRATA_HIP_GFX906)
#include "strata/kernels/gfx_arch.hpp"
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv;   // query heads per KV head (12 for Flash-Next; 1..16, a multiple of 4)
static_assert(G % 4 == 0 && G <= 16, "qsa_decode_attn: query heads per KV head must be 4, 8, 12 or 16 (float4 probability rows, the 16-lane score reduce)");
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv, "qsa_decode_attn is compiled for core/geometry.hpp");
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
// Per-format bodies; `load8` below is the KV_MODE dispatcher. value=false is the K side, true the V side.
__device__ __forceinline__ void load8_f16(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const uint4 raw = *reinterpret_cast<const uint4*>(base);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(h2[j]);
        out[2 * j] = f.x;
        out[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void load8_q8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = __half2float(__ushort_as_half(sbits));
    const uint2 raw = *reinterpret_cast<const uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__device__ __forceinline__ void load8_q4(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = __half2float(__ushort_as_half(blk->d));
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
    }
}
template <int KV_MODE>
__device__ __forceinline__ void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 0) load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1) load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        // hybrid K8V4: both sides are defined - K unrotated INT8, V rotated Q4_0 - so a value=true call
        // reads the Q4_0 pool instead of dereferencing the null v_q (no call site does today; PR review)
        if (value) load8_q4(p, value, row, d0, out);
        else load8_q8(p, value, row, d0, out);
    } else load8_q4(p, value, row, d0, out);
}

// One cell's G probabilities (G/4 broadcast float4 loads) into the G accumulators: every accumulator takes one fmaf,
// so the sums are the per-head loop's, bit for bit, for any G.
template <int GG> __device__ __forceinline__ void fold_cell(float (&acc)[GG], const float* w, float v) {
    const float4* w4 = reinterpret_cast<const float4*>(w);
    if constexpr (GG == 12) {   // Flash-Next: the expressions this replaced, so its device code is unchanged
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        acc[0] = fmaf(w0.x, v, acc[0]);  acc[1] = fmaf(w0.y, v, acc[1]);
        acc[2] = fmaf(w0.z, v, acc[2]);  acc[3] = fmaf(w0.w, v, acc[3]);
        acc[4] = fmaf(w1.x, v, acc[4]);  acc[5] = fmaf(w1.y, v, acc[5]);
        acc[6] = fmaf(w1.z, v, acc[6]);  acc[7] = fmaf(w1.w, v, acc[7]);
        acc[8] = fmaf(w2.x, v, acc[8]);  acc[9] = fmaf(w2.y, v, acc[9]);
        acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
        return;
    }
#pragma unroll
    for (int i = 0; i < GG / 4; ++i) {
        const float4 x = w4[i];
        acc[4 * i + 0] = fmaf(x.x, v, acc[4 * i + 0]);
        acc[4 * i + 1] = fmaf(x.y, v, acc[4 * i + 1]);
        acc[4 * i + 2] = fmaf(x.z, v, acc[4 * i + 2]);
        acc[4 * i + 3] = fmaf(x.w, v, acc[4 * i + 3]);
    }
}

// the q8_v3 kernel's value loop: Flash-Next (12 heads per KV head) keeps the expressions this kernel always had (its device code is
// unchanged); any other G folds the cell with fold_cell.  A template, so the branch not taken is not instantiated (G is not 12 in a
// non-Flash-Next build, and the 12-element initialiser below would not compile).
template <int GG> __device__ __forceinline__ void fold_cell_v3(float (&acc)[GG], const float* w, float v) {
    if constexpr (GG == 12) {
        const float4 p0 = *reinterpret_cast<const float4*>(&w[0]);
        const float4 p1 = *reinterpret_cast<const float4*>(&w[4]);
        const float4 p2 = *reinterpret_cast<const float4*>(&w[8]);
        const float pr[GG] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w, p2.x, p2.y, p2.z, p2.w};
#pragma unroll
        for (int h = 0; h < GG; ++h) acc[h] = fmaf(pr[h], v, acc[h]);
    } else {
        fold_cell<GG>(acc, w, v);
    }
}


// Exact-intent score reduce-scatter: the same butterfly pairs as twelve warp_sum calls,
// with 16 shuffle exchanges instead of 60. Used only by the opt-in gfx906 INT8
// query-swizzled specialization; softmax, value accumulation and merge are unchanged.
__device__ __forceinline__ float reduce12_gfx906(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int reduce12_head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// The chunk kernel's body; `p` is the query's KV pools (the call's one set, or blockIdx.z's slot's set).
// (the body is a device function so the batch window's rows kernel below can run it over each query's own pools without
// the default kernel's signature, or the function-pointer types the pre-sm_75 / HIP selections build, changing)
template <int KV_MODE, bool LANE_CELL, bool QUERY_SWIZZLE, bool REDUCE12>
__device__ __forceinline__ void attn_chunk_body(const float* __restrict__ q, QsaAttnPools p,
                                                const int32_t* __restrict__ ids, const int32_t* __restrict__ step,
                                                int n_kv_heads, int page_size, float scale,
                                                float* __restrict__ part_acc, float* __restrict__ part_m,
                                                float* __restrict__ part_l, int n_chunks, int cap,
                                                long long scratch_stride) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads
    __shared__ float sp[G][CHUNK];                // scores, then probabilities
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    // Opt-in gfx906 batched attention: a bijective XOR of query dimensions in LDS only. It swaps the two
    // float4 halves of lanes whose bit 2 is set, retaining 16-byte alignment and the 12 KB tile.
    // The scoring reads undo this permutation; FP operands and operation order are unchanged.
    static_assert(!QUERY_SWIZZLE || !LANE_CELL, "query swizzle is for warp scoring only");
    for (int i = t; i < G * HD; i += THREADS) {
        const int d = i % HD;
        const int sd = QUERY_SWIZZLE ? (d ^ ((d >> 3) & 4)) : d;
        sq[i / HD][sd] = q[(size_t) (kvh * G) * HD + i];
    }
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    if constexpr (LANE_CELL) {
        // S25 (STRATA_ATTN_LANECELL=1): thread t scores cell t % 64 for heads 3 (t / 64) .. +2 by itself - the same
        // 8-dimension partial per former lane (the identical expression), then the former warp_sum's butterfly as
        // lane 0 ran it (a[l] += a[l + o] for o = 16 .. 1), so every score is bit-identical without 12 shuffle
        // reductions per cell
        constexpr int HPT = G * CHUNK / THREADS;   // heads per thread (3)
        const int c = t % CHUNK, h0 = (t / CHUNK) * HPT;
        if (c >= n_here || srow[c] < 0) {
#pragma unroll
            for (int j = 0; j < HPT; ++j) sp[h0 + j][c] = -FLT_MAX;
        } else {
            float a[HPT][32];
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], l * 8, k8);
#pragma unroll
                for (int j = 0; j < HPT; ++j) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h0 + j][l * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h0 + j][l * 8 + 4]);
                    float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                    a[j][l] = s;
                }
            }
#pragma unroll
            for (int j = 0; j < HPT; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                    for (int l = 0; l < o; ++l) a[j][l] = a[j][l] + a[j][l + o];
                sp[h0 + j][c] = a[j][0] * scale;
            }
        }
    } else
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        if constexpr (REDUCE12) {
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
        float part[16];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const int qflip = QUERY_SWIZZLE ? (lane & 4) : 0;
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + qflip]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + (qflip ^ 4)]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            part[h] = s;
        }
#pragma unroll
        for (int h = G; h < 16; ++h) part[h] = 0.0f;
        const float score = reduce12_gfx906(part, lane);
        const int head = reduce12_head_of_lane(lane);
        if ((lane & 1) == 0 && head < G) sp[head][c] = score * scale;
        } else {
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const int qflip = QUERY_SWIZZLE ? (lane & 4) : 0;
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + qflip]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + (qflip ^ 4)]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        float v;
        if constexpr (KV_MODE == 0) {
            v = __half2float(__ushort_as_half(p.v_pool[srow[c] * HD + t]));
        } else if constexpr (KV_MODE == 1) {
            const float sc = __half2float(__ushort_as_half(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d = __half2float(__ushort_as_half(blk->d));
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

template <int KV_MODE, bool LANE_CELL = false, bool QUERY_SWIZZLE = false, bool REDUCE12 = false>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    attn_chunk_body<KV_MODE, LANE_CELL, QUERY_SWIZZLE, REDUCE12>(q, p, ids, step, n_kv_heads, page_size, scale,
                                                                 part_acc, part_m, part_l, n_chunks, cap,
                                                                 scratch_stride);
}

// The same blocks for queries of different slots: query blockIdx.z reads its own slot's pools, everything else is
// attn_chunk_kernel's batched form (qsa_decode_attn_slots).
template <int KV_MODE, bool LANE_CELL>
__global__ void __launch_bounds__(THREADS) attn_chunk_slots_kernel(const float* __restrict__ q, QsaAttnSlotPools ps,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ step, int n_kv_heads,
                                                                   int page_size, float scale,
                                                                   float* __restrict__ part_acc,
                                                                   float* __restrict__ part_m,
                                                                   float* __restrict__ part_l, int n_chunks, int cap,
                                                                   long long scratch_stride) {
    attn_chunk_body<KV_MODE, LANE_CELL, false, false>(q, ps.p[blockIdx.z], ids, step, n_kv_heads, page_size, scale,
                                                      part_acc, part_m, part_l, n_chunks, cap, scratch_stride);
}

// Batch windows: query blockIdx.z attends over its own slot's pools (rows.p[z]); the default kernel otherwise.
template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) attn_chunk_rows_kernel(const float* __restrict__ q, QsaAttnPoolsRows rows,
                                                                  const int32_t* __restrict__ ids,
                                                                  const int32_t* __restrict__ step, int n_kv_heads,
                                                                  int page_size, float scale,
                                                                  float* __restrict__ part_acc,
                                                                  float* __restrict__ part_m,
                                                                  float* __restrict__ part_l, int n_chunks, int cap,
                                                                  long long scratch_stride) {
    attn_chunk_body<KV_MODE, false, false, false>(q, rows.p[blockIdx.z], ids, step, n_kv_heads, page_size, scale,
                                                  part_acc, part_m, part_l, n_chunks, cap, scratch_stride);
}

#if !defined(__HIPCC__)
// Turing (INT8 pools): attn_chunk_kernel<1> bit for bit, rearranged for the SM's shared-memory pipe, which the
// default kernel saturates (its time grew with the queries even with the SMs half empty).  Three changes, none to an
// operand or the order of an operation:
//   - the lane's slice of the 12 query heads (8 dimensions each) is read from shared memory once, into registers,
//     instead of two 16-byte reads per head per cell (with a two-way bank conflict at lane * 8);
//   - the 12 heads' lane sums are reduced with reduce12_gfx906 (16 shuffles, each sum warp_sum's pairs) instead of
//     12 warp_sums (60 shuffles);
//   - the probabilities are stored cell-major, so the value loop reads a cell's 12 as three 16-byte broadcasts.
// RTX 2080 Ti, 2051 selected cells: 1 / 2 / 3 / 4 queries 51.6 / 78.5 / 110 / 146 -> 34.5 / 41.0 / 73.0 / 84.9 us
// (with the merge); 600 cells 45.9 / 74.6 / 105 / 137 -> 28.6 / 32.8 / 33.5 / 33.3 us.  STRATA_ATTN_V3=0 keeps the
// default kernel; =1 takes this one on any CUDA card.
__global__ void __launch_bounds__(THREADS) attn_chunk_q8_v3_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ step, int n_kv_heads,
                                                                   int page_size, float scale, float* __restrict__ part_acc,
                                                                   float* __restrict__ part_m, float* __restrict__ part_l,
                                                                   int n_chunks, int cap = 0, long long scratch_stride = 0) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];
    __shared__ __align__(16) float spt[CHUNK][G];   // cell-major: one cell's 12 heads are 3 float4
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    // stored with the gfx906 swizzle (the two float4 halves of lanes with bit 2 swapped): fewer conflicts on the one read
    for (int i = t; i < G * HD; i += THREADS) {
        const int d = i % HD;
        sq[i / HD][d ^ ((d >> 3) & 4)] = q[(size_t) (kvh * G) * HD + i];
    }
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    float qr[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const int qflip = lane & 4;
        const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + qflip]);
        const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + (qflip ^ 4)]);
        qr[h][0] = qa.x; qr[h][1] = qa.y; qr[h][2] = qa.z; qr[h][3] = qa.w;
        qr[h][4] = qb.x; qr[h][5] = qb.y; qr[h][6] = qb.z; qr[h][7] = qb.w;
    }
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) spt[c][lane] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<1>(p, false, srow[c], lane * 8, k8);
        float part[16];
#pragma unroll
        for (int h = 0; h < G; ++h)
            part[h] = k8[0] * qr[h][0] + k8[1] * qr[h][1] + k8[2] * qr[h][2] + k8[3] * qr[h][3] +
                      k8[4] * qr[h][4] + k8[5] * qr[h][5] + k8[6] * qr[h][6] + k8[7] * qr[h][7];
#pragma unroll
        for (int h = G; h < 16; ++h) part[h] = 0.0f;
        const float score = reduce12_gfx906(part, lane);
        const int head = reduce12_head_of_lane(lane);
        if ((lane & 1) == 0 && head < G) spt[c][head] = score * scale;
    }
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = spt[lane][h], b = spt[lane + 32][h];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        const float sc = __half2float(__ushort_as_half(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
        const float v = (float) p.v_q[srow[c] * HD + t] * sc;
        fold_cell_v3<G>(acc, &spt[c][0], v);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}
bool attn_v3_on() {
    static int on[16] = {};   // by device ordinal: 0 unknown, 1 yes, -1 no
    static const int env = [] { const char* v = std::getenv("STRATA_ATTN_V3"); return v ? (v[0] == '0' ? 0 : 1) : -1; }();
    if (env >= 0) return env == 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    if (on[dev] == 0) {
        int maj = 0, mnr = 0;
        const bool ok = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        on[dev] = ok && strata::sm75_default(maj, mnr) ? 1 : -1;   // Turing only, with STRATA_SM75_PERF=1
    }
    return on[dev] > 0;
}
#endif

// PR #540 (sskver): the same attention with fewer shuffles and less shared-memory traffic, bit for bit the kernel
// above (same operands, same order); +7% prompt speed on a V100, where this kernel reads every prompt chunk.  It
// uses more registers (80 vs 38 on sm_70), so on CUDA it runs on cards below sm_75 only, and exists only in the
// experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON).  HIP: built for wave32 cards (not the gfx906 wave64 backend) and
// run on gfx103x (RDNA2), with 8 cells per step and the butterfly's lane exchanges in registers (DPP / permlanex16)
// instead of ds_bpermute: there the score loop is bound by the LDS pipe (the q reads and ds_bpermute share it), not by
// occupancy - 8 cells per step at 7 waves/SIMD beat 1 cell at 9.  Every other card keeps exactly the kernel above.
// Fork (lab/upsync): built on every CUDA build too, for STRATA_ATTN_DAILY=1 (see daily_attn below).  With NC 2 and
// no DPP it is the Daily's (be4cf1c) attn_chunk_kernel line for line - the same reduce12 scores, the same cell-major
// probabilities and the same four-cell value fold - except that `spt` shares `sq`'s storage here (no arithmetic
// change).  Without the variable nothing selects it on a card at or above sm_75.
#if defined(STRATA_EXPERIMENTAL_SM60) || !defined(__HIPCC__) || !defined(STRATA_HIP_GFX906)
#define STRATA_ATTN_PRE75_BUILT 1
#endif
#if defined(STRATA_ATTN_PRE75_BUILT)
// A butterfly step's lane exchange, `M` = 1, 2, 4, 8 or 16.  DPP (HIP, wave32): in registers - DPP row_xmask within a
// 16-lane row, v_permlanex16 across the two rows - instead of ds_bpermute; the same value moves, so every sum is
// unchanged bit for bit.
template <bool DPP, int M> __device__ __forceinline__ float shfl_x(float v) {
#if defined(__HIPCC__)
    if constexpr (DPP) {
        const int x = __float_as_int(v);
        if constexpr (M == 16) return __int_as_float(__builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false));
        else return __int_as_float(__builtin_amdgcn_mov_dpp(x, 0x160 | M, 0xF, 0xF, false));   // row_xmask:M
    }
#endif
    return __shfl_xor_sync(0xffffffffu, v, M);
}
// warp_sum / warp_max with the same pairing order (xor 16, 8, 4, 2, 1)
template <bool DPP> __device__ __forceinline__ float warp_sum_x(float v) {
    v += shfl_x<DPP, 16>(v); v += shfl_x<DPP, 8>(v); v += shfl_x<DPP, 4>(v); v += shfl_x<DPP, 2>(v); v += shfl_x<DPP, 1>(v);
    return v;
}
template <bool DPP> __device__ __forceinline__ float warp_max_x(float v) {
    v = fmaxf(v, shfl_x<DPP, 16>(v)); v = fmaxf(v, shfl_x<DPP, 8>(v)); v = fmaxf(v, shfl_x<DPP, 4>(v));
    v = fmaxf(v, shfl_x<DPP, 2>(v)); v = fmaxf(v, shfl_x<DPP, 1>(v));
    return v;
}
// The 12 heads' lane sums (`part[12..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
// every output adds the same two operands at every level, so each sum is bit for bit warp_sum's (float add commutes),
// for 16 shuffles instead of 12 x 5.  Lane l ends holding head head_of_lane(l); lanes l and l^1 agree.
template <bool DPP> __device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + shfl_x<DPP, 16>(hi ? part[i] : part[i + 8]);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + shfl_x<DPP, 8>(hi ? r8[i] : r8[i + 4]);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + shfl_x<DPP, 4>(hi ? r4[i] : r4[i + 2]);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + shfl_x<DPP, 2>(hi2 ? r2[0] : r2[1]);
    return r1 + shfl_x<DPP, 1>(r1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own thread index), per KV format.
template <int KV_MODE>
__device__ __forceinline__ float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return __half2float(__ushort_as_half(p.v_pool[row * HD + d]));
    } else if constexpr (KV_MODE == 1) {
        const float sc = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
        return (float) p.v_q[row * HD + d] * sc;
    } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = __half2float(__ushort_as_half(blk->d));
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nibble * dd;
    }
}

template <int KV_MODE, int NC_ = 2, bool DPP_ = false>
__device__ __forceinline__ void attn_chunk_pre75_body(const float* __restrict__ q, const QsaAttnPools& p,
                                                      const int32_t* __restrict__ ids,
                                                      const int32_t* __restrict__ step, int n_kv_heads,
                                                      int page_size, float scale, float* __restrict__ part_acc,
                                                      float* __restrict__ part_m, float* __restrict__ part_l,
                                                      int n_chunks, int cap, long long scratch_stride) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads (dead once the scores are done)
    __shared__ float sp[G][CHUNK];                // scores
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    // probabilities, cell-major: the value loop reads one cell's 12 as three float4.  They are first written after the
    // __syncthreads that ends the score phase, the last reader of sq, so they share its storage (64 x 12 of its 256 x 12
    // floats) and the block's shared memory stays what it was before the cell-major layout.
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sq[0][0]);
    static_assert(CHUNK * G <= G * HD, "spt must fit in sq");
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.  NC cells per step
    // (warp + 8 * (i + j)): a lane's q slice comes out of shared memory once for all of them, which cuts the traffic
    // that bounds this phase (24 x 16 B per lane per cell, against the FMAs) by NC; every dot product is the same
    // expression as for a single cell, so the scores are unchanged.
    constexpr int NC = NC_;   // CUDA 2 (4 measured slower on a V100: the registers cost occupancy); HIP gfx103x 8
    const int hd = head_of_lane(lane);
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) load8<KV_MODE>(p, false, srow[cc[j]], lane * 8, kk[j]);
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12<DPP_>(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < NC; ++k) {   // a masked or out-of-range cell in the group: one cell at a time
                const int c = cc[k];
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12<DPP_>(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x<DPP_>(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum_x<DPP_>(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    // One cell's 12 weights (three broadcast float4 loads) into the 12 accumulators; the cells are folded in
    // ascending order, so the sums are the plain loop's, bit for bit.
    const auto fold = [&](int c, float v) {
        fold_cell<G>(acc, spt[c], v);
    };
    int c = 0;
    // four cells at a time with their V loads issued together (the loop was one dependent load chain per cell);
    // a group holding a masked cell (page -1, rare) leaves this loop and the scalar one below finishes the chunk.
    for (; c + 4 <= n_here; c += 4) {
        const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) break;
        const float v0 = load_v1<KV_MODE>(p, r0, t), v1 = load_v1<KV_MODE>(p, r1, t);
        const float v2 = load_v1<KV_MODE>(p, r2, t), v3 = load_v1<KV_MODE>(p, r3, t);
        fold(c, v0);
        fold(c + 1, v1);
        fold(c + 2, v2);
        fold(c + 3, v3);
    }
    for (; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        fold(c, load_v1<KV_MODE>(p, srow[c], t));
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

template <int KV_MODE, int NC_ = 2, bool DPP_ = false>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel_pre75(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ step, int n_kv_heads,
                                                                   int page_size, float scale,
                                                                   float* __restrict__ part_acc,
                                                                   float* __restrict__ part_m,
                                                                   float* __restrict__ part_l, int n_chunks,
                                                                   int cap = 0, long long scratch_stride = 0) {
    attn_chunk_pre75_body<KV_MODE, NC_, DPP_>(q, p, ids, step, n_kv_heads, page_size, scale, part_acc, part_m, part_l,
                                              n_chunks, cap, scratch_stride);
}

// Multi-slot batched form of attn_chunk_kernel_pre75: query blockIdx.z attends over ps.p[blockIdx.z]
// with the exact pre75 arithmetic and accumulation order (bitwise identical to per-query attn_chunk_kernel_pre75).
template <int KV_MODE, int NC_ = 2, bool DPP_ = false>
__global__ void __launch_bounds__(THREADS) attn_chunk_slots_pre75_kernel(
    const float* __restrict__ q, QsaAttnSlotPools ps, const int32_t* __restrict__ ids,
    const int32_t* __restrict__ step, int n_kv_heads, int page_size, float scale, float* __restrict__ part_acc,
    float* __restrict__ part_m, float* __restrict__ part_l, int n_chunks, int cap, long long scratch_stride) {
    attn_chunk_pre75_body<KV_MODE, NC_, DPP_>(q, ps.p[blockIdx.z], ids, step, n_kv_heads, page_size, scale, part_acc,
                                              part_m, part_l, n_chunks, cap, scratch_stride);
}

// Per-row batched form of attn_chunk_kernel_pre75 (STRATA_BATCH_ATTN_ROWS=1 with STRATA_ATTN_DAILY=1).
template <int KV_MODE, int NC_ = 2, bool DPP_ = false>
__global__ void __launch_bounds__(THREADS) attn_chunk_rows_pre75_kernel(
    const float* __restrict__ q, QsaAttnPoolsRows rows, const int32_t* __restrict__ ids,
    const int32_t* __restrict__ step, int n_kv_heads, int page_size, float scale, float* __restrict__ part_acc,
    float* __restrict__ part_m, float* __restrict__ part_l, int n_chunks, int cap, long long scratch_stride) {
    attn_chunk_pre75_body<KV_MODE, NC_, DPP_>(q, rows.p[blockIdx.z], ids, step, n_kv_heads, page_size, scale, part_acc,
                                              part_m, part_l, n_chunks, cap, scratch_stride);
}
#endif  // STRATA_ATTN_PRE75_BUILT

#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
// PR 3.3a (Volta sm_70): multi-chunk split-K decode attention on mma.m8n8k4 Tensor Cores + 128-bit uint4 KV staging.
// Targets ~160 CTAs (2 full waves on 80-SM V100) using `nsp <= ceil(cap / 64)` splits over the existing scratch
// buffer (0 extra VRAM, 100% CUDA-Graph-safe, exact match with attn_merge_v2_kernel's `ceil(width / 64)` bound).
// Each CTA loads & packs Q once into smem, then strides 64-cell blocks (`blk64 = split, split+nsp, ...`) in 32-cell
// sub-chunks (`CH_V70 = 32`) with online softmax:
//   - K and V (all 256 dims of 32 cells) are loaded together in 128-bit uint4 bursts (1 barrier for modes 1/2/3!).
//   - Q*K^T: 4 warps x 8 cells = 32 cells; QP (qp&1) picks head block (0..7 / 8..15), (qp>>1) picks dim half
//     (0..127 / 128..255), reduced in-warp via 1 __shfl_xor_sync(8) (0 smem, 0 cross-warp barriers).
//   - Online softmax over 32 cells -> updates running mrow/lsum and alpha.
//   - P*V: 4 warps x 64 dims = 256 dims; QP (qp&1) owns 32 dims (1 block_q4_0 for k8v4/q4_0, half of a KV_Q8_GROUP
//     for int8). Running vmax_run rescales `acc` directly in-place (0 duplicate `tmp` register array).
// STRATA_ATTN_V70: 0 disables (fallback to attn_chunk_kernel_pre75), 1 1-pass FP16 Q/P,
// 2 (default on sm_70) 2-pass hi+lo FP16 Q/P (~2e-6 FP32-level precision, tied speed on k8v4).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
#define STRATA_DEC_ATTN_VOLTA 1
#else
#define STRATA_DEC_ATTN_VOLTA 0
#endif

__device__ __forceinline__ void dec_mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if STRATA_DEC_ATTN_VOLTA
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    (void) c; (void) a0; (void) a1; (void) b0; (void) b1;
    __trap();
#endif
}

__device__ __forceinline__ uint32_t dec_i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t dec_u4x2_to_h2(uint32_t n0, uint32_t n1) {
    const uint32_t y = 0x64006400u | (n0 & 0x0Fu) | ((n1 & 0x0Fu) << 16);
    __half2 h = *reinterpret_cast<const __half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1032.f), __float2half(1032.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t dec_pack_h2(float lo_k, float hi_k) {
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t*>(&h);
}

constexpr int THREADS_V70 = 128;
constexpr int CH_V70 = 32;

template <int KV_MODE, bool SPLIT_HL>
struct SmemDecV70 {
    long long srow[CH_V70];          // 256 B
    __align__(16) float sp[16][CH_V70 + 4];   // 2,304 B
    float mrow[16];                  // 64 B
    float lsum[16];                  // 64 B
    float alpha[16];                 // 64 B
    float qmax[4];                   // 16 B
    __align__(16) __half qh[G][HD + 4];                                     // 6,240 B (G=12)
    __align__(16) __half ql[SPLIT_HL ? G : 1][SPLIT_HL ? (HD + 4) : 1];     // 6,240 B if SPLIT_HL
    // K and V staged together for modes 1/2/3 (only 1 barrier per 32 cells!); mode 0 (fp16) unions K and V.
    static constexpr int K_BYTES = (KV_MODE == 0) ? (CH_V70 * 528) : (KV_MODE == 2 ? (CH_V70 * 144) : (CH_V70 * 272));
    static constexpr int V_BYTES = (KV_MODE == 0) ? 16 : (KV_MODE == 1 ? (CH_V70 * 272) : (CH_V70 * 144));
    __align__(16) uint8_t kbuf[K_BYTES];
    __align__(16) uint8_t vbuf[V_BYTES];
    float ks[(KV_MODE == 1 || KV_MODE == 3) ? CH_V70 : 1][4];
    float vs[(KV_MODE == 1) ? CH_V70 : 1][4];
    float vs_q4[(KV_MODE == 2 || KV_MODE == 3) ? CH_V70 : 1][8];
};

template <int KV_MODE, bool SPLIT_HL>
__global__ void __launch_bounds__(THREADS_V70, SPLIT_HL ? 2 : 3)
attn_chunk_v70_mma_kernel(const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ids,
                          const int32_t* __restrict__ step, int n_kv_heads, int page_size, float scale,
                          float* __restrict__ part_acc, float* __restrict__ part_m, float* __restrict__ part_l,
                          int n_splits, int cap = 0, long long scratch_stride = 0) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;

    const int n_ids = __ldg(step + kStepWidth);
    const int sp_idx = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int slot = kvh * n_splits + sp_idx;
    if (sp_idx * CHUNK >= n_ids) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }

    __shared__ SmemDecV70<KV_MODE, SPLIT_HL> S;
    const float* qkvh = q + (size_t) (kvh * G) * HD;

    // 0. Load & scale Q ONCE for all 64-cell blocks handled by this split!
    float qm = 0.0f;
    const float4* q4 = reinterpret_cast<const float4*>(qkvh);
    for (int i = t; i < G * (HD / 4); i += THREADS_V70) {
        const float4 v = __ldg(q4 + i);
        qm = fmaxf(qm, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    qm = warp_max(qm);
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -FLT_MAX; S.lsum[t] = 0.0f; S.alpha[t] = 0.0f; }
    for (int i = t; i < (16 - G) * CH_V70; i += THREADS_V70)
        S.sp[G + i / CH_V70][i % CH_V70] = 0.0f;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = qm > 0.0f ? ldexpf(1.0f, 14 - qe) : 0.0f;
    const float qdown = qm > 0.0f ? ldexpf(scale, qe - 14) : 0.0f;

    for (int i = t; i < G * (HD / 4); i += THREADS_V70) {
        const int h = i / (HD / 4);
        const int d = (i % (HD / 4)) * 4;
        const float4 v = __ldg(q4 + i);
        const float x0 = v.x * qup, x1 = v.y * qup, x2 = v.z * qup, x3 = v.w * qup;
        const __half2 h01 = __floats2half2_rn(x0, x1);
        const __half2 h23 = __floats2half2_rn(x2, x3);
        *reinterpret_cast<__half2*>(&S.qh[h][d]) = h01;
        *reinterpret_cast<__half2*>(&S.qh[h][d + 2]) = h23;
        if constexpr (SPLIT_HL) {
            const float2 f01 = __half22float2(h01);
            const float2 f23 = __half22float2(h23);
            *reinterpret_cast<__half2*>(&S.ql[h][d]) = __floats2half2_rn(x0 - f01.x, x1 - f01.y);
            *reinterpret_cast<__half2*>(&S.ql[h][d + 2]) = __floats2half2_rn(x2 - f23.x, x3 - f23.y);
        }
    }

    const int qp = (lane >> 2) & 3;
    const int jj = (lane & 3) + 4 * (lane >> 4);
    const int crow = (lane & 1) + 4 * (lane >> 4);
    const int ccol = lane & 2;
    // Q*K^T mapping: warp w owns cells 8w..8w+7; (qp&1) picks head block rb; (qp>>1) picks dim half (0..127 / 128..255).
    const int qk_rb = qp & 1;
    const int qk_dh = qp >> 1;
    const int qk_head = qk_rb * 8 + jj;
    const int qk_cell = warp * 8 + jj;
    // P*V mapping: warp w owns 64 output dims [64w..64w+63]; (qp>>1) picks head block pv_rb; (qp&1) picks 32-dim half db.
    const int pv_rb = qp >> 1;
    const int pv_db = qp & 1;
    const int pv_head = pv_rb * 8 + jj;

    float acc[4][8];
    float accl[4][8];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[nt][i] = 0.0f; accl[nt][i] = 0.0f; }
    }
    float vmax_run = (KV_MODE == 0) ? 1.0f : 0.0f;

    // Stride 64-cell blocks (exact match to attn_merge_v2_kernel's ceil(width / 64)), 2 x 32-cell sub-chunks each.
    const int n_ids_bound = cap > 0 ? min(n_ids, ((cap + CHUNK - 1) / CHUNK) * CHUNK) : n_ids;
    for (int blk64 = sp_idx; blk64 * CHUNK < n_ids_bound; blk64 += n_splits) {
        for (int sub = 0; sub < 2; ++sub) {
            const int c0 = blk64 * CHUNK + sub * CH_V70;
            if (c0 >= n_ids_bound) break;
            const int nh = min(CH_V70, n_ids_bound - c0);

            __syncthreads();
            if (t < CH_V70) {
                long long r = -1;
                if (t < nh) {
                    const int cell = __ldg(ids + c0 + t);
                    const long long page = (long long) __ldg(p.page_table + cell / page_size);
                    if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
                }
                S.srow[t] = r;
            }
            __syncthreads();

            // 1. Stage K (and V in parallel for modes 1/2/3!) in 128-bit uint4 bursts.
            if constexpr (KV_MODE == 1 || KV_MODE == 3) {
                uint4 kx[4];
                int kc[4], ku[4];
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const int idx = t + s * THREADS_V70;
                    kc[s] = idx >> 4;
                    ku[s] = idx & 15;
                    const long long r = S.srow[kc[s]];
                    kx[s] = r >= 0 ? __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD) + ku[s]) : make_uint4(0, 0, 0, 0);
                }
                if constexpr (KV_MODE == 3) {
                    // K8V4: also burst-load all 32 cells of Q4_0 V (288 uint4 / 128 threads <= 3 loads/thread)!
                    constexpr int BYTES = (HD / QK4_0) * (int) sizeof(block_q4_0);
                    uint4 vx[3];
                    int vidx[3];
#pragma unroll
                    for (int s = 0; s < 3; ++s) {
                        const int idx = t + s * THREADS_V70;
                        vidx[s] = idx;
                        vx[s] = make_uint4(0, 0, 0, 0);
                        if (idx < CH_V70 * 9) {
                            const int c = idx / 9, u = idx - c * 9;
                            const long long r = S.srow[c];
                            if (r >= 0) vx[s] = __ldg(reinterpret_cast<const uint4*>(p.v_q4 + r * BYTES) + u);
                        }
                    }
                    const int sc_c = t >> 2, sc_g = t & 3;
                    const long long sc_r = S.srow[sc_c];
                    const float ks_val = sc_r >= 0 ? __half2float(__ushort_as_half(__ldg(p.k_scale + sc_r * 4 + sc_g))) : 0.0f;
#pragma unroll
                    for (int s = 0; s < 4; ++s)
                        *reinterpret_cast<uint4*>(S.kbuf + kc[s] * 272 + ku[s] * 16) = kx[s];
                    uint4* vb4 = reinterpret_cast<uint4*>(S.vbuf);
#pragma unroll
                    for (int s = 0; s < 3; ++s)
                        if (vidx[s] < CH_V70 * 9) vb4[vidx[s]] = vx[s];
                    S.ks[sc_c][sc_g] = ks_val;
                } else {
                    // INT8: burst-load both K and V (4 uint4 each) + K/V scales in parallel!
                    uint4 vx[4];
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const long long r = S.srow[kc[s]];
                        vx[s] = r >= 0 ? __ldg(reinterpret_cast<const uint4*>(p.v_q + r * HD) + ku[s]) : make_uint4(0, 0, 0, 0);
                    }
                    const int sc_c = t >> 2, sc_g = t & 3;
                    const long long sc_r = S.srow[sc_c];
                    const float ks_val = sc_r >= 0 ? __half2float(__ushort_as_half(__ldg(p.k_scale + sc_r * 4 + sc_g))) : 0.0f;
                    const float vs_val = sc_r >= 0 ? __half2float(__ushort_as_half(__ldg(p.v_scale + sc_r * 4 + sc_g))) : 0.0f;
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        *reinterpret_cast<uint4*>(S.kbuf + kc[s] * 272 + ku[s] * 16) = kx[s];
                        *reinterpret_cast<uint4*>(S.vbuf + kc[s] * 272 + ku[s] * 16) = vx[s];
                    }
                    S.ks[sc_c][sc_g] = ks_val;
                    S.vs[sc_c][sc_g] = vs_val;
                }
            } else if constexpr (KV_MODE == 2) {
                // Q4_0: burst-load both K and V (288 uint4 each)!
                constexpr int BYTES = (HD / QK4_0) * (int) sizeof(block_q4_0);
                uint4 kx[3], vx[3];
#pragma unroll
                for (int s = 0; s < 3; ++s) {
                    const int idx = t + s * THREADS_V70;
                    kx[s] = make_uint4(0, 0, 0, 0);
                    vx[s] = make_uint4(0, 0, 0, 0);
                    if (idx < CH_V70 * 9) {
                        const int c = idx / 9, u = idx - c * 9;
                        const long long r = S.srow[c];
                        if (r >= 0) {
                            kx[s] = __ldg(reinterpret_cast<const uint4*>(p.k_q4 + r * BYTES) + u);
                            vx[s] = __ldg(reinterpret_cast<const uint4*>(p.v_q4 + r * BYTES) + u);
                        }
                    }
                }
                uint4* kb4 = reinterpret_cast<uint4*>(S.kbuf);
                uint4* vb4 = reinterpret_cast<uint4*>(S.vbuf);
#pragma unroll
                for (int s = 0; s < 3; ++s) {
                    const int idx = t + s * THREADS_V70;
                    if (idx < CH_V70 * 9) { kb4[idx] = kx[s]; vb4[idx] = vx[s]; }
                }
            } else {
                // FP16: burst-load K (32 x 512 B = 1,024 uint4 -> 8 uint4/thread).
                uint4 kx[8];
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const int idx = t + s * THREADS_V70;
                    const int c = idx >> 5, u = idx & 31;
                    const long long r = S.srow[c];
                    kx[s] = r >= 0 ? __ldg(reinterpret_cast<const uint4*>(p.k_pool + r * HD) + u) : make_uint4(0, 0, 0, 0);
                }
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const int idx = t + s * THREADS_V70;
                    const int c = idx >> 5, u = idx & 31;
                    *reinterpret_cast<uint4*>(S.kbuf + c * 528 + u * 16) = kx[s];
                }
            }
            __syncthreads();

            // 2. Q * K^T (0 extra barriers; 1 in-warp shuffle XOR 8 merges dims 0..127 and 128..255).
            float total[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const int cb0 = warp * 8 + ccol;
            if constexpr (KV_MODE == 1 || KV_MODE == 3) {
                const uint8_t* krow = S.kbuf + qk_cell * 272;
#pragma unroll
                for (int gl = 0; gl < 2; ++gl) {
                    const int g = qk_dh * 2 + gl;
                    float tg[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    float tgl[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int ks = 0; ks < 16; ++ks) {
                        const int d0 = g * 64 + ks * 4;
                        const uint32_t raw = *reinterpret_cast<const uint32_t*>(krow + d0);
                        const uint32_t b0 = dec_i8x2_to_h2(raw & 0xffffu);
                        const uint32_t b1 = dec_i8x2_to_h2(raw >> 16);
                        const uint2 ah = qk_head < G ? *reinterpret_cast<const uint2*>(&S.qh[qk_head][d0]) : make_uint2(0, 0);
                        dec_mma884(tg, ah.x, ah.y, b0, b1);
                        if constexpr (SPLIT_HL) {
                            const uint2 al = qk_head < G ? *reinterpret_cast<const uint2*>(&S.ql[qk_head][d0]) : make_uint2(0, 0);
                            dec_mma884(tgl, al.x, al.y, b0, b1);
                        }
                    }
                    const float sc0 = S.ks[cb0][g], sc1 = S.ks[cb0 + 1][g];
                    const float sc4 = S.ks[cb0 + 4][g], sc5 = S.ks[cb0 + 5][g];
                    const float sc[8] = {sc0, sc1, sc0, sc1, sc4, sc5, sc4, sc5};
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if constexpr (SPLIT_HL) total[i] = fmaf(tg[i] + tgl[i], sc[i], total[i]);
                        else total[i] = fmaf(tg[i], sc[i], total[i]);
                    }
                }
            } else if constexpr (KV_MODE == 2) {
                constexpr int BYTES = (HD / QK4_0) * (int) sizeof(block_q4_0);
#pragma unroll
                for (int bl = 0; bl < 4; ++bl) {
                    const int b = qk_dh * 4 + bl;
                    float tg[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    float tgl[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    const uint8_t* qs = S.kbuf + qk_cell * BYTES + b * 18 + 2;
#pragma unroll
                    for (int ks = 0; ks < 8; ++ks) {
                        const int j0 = (ks & 3) * 4;
                        const uint32_t w0 = qs[j0], w1 = qs[j0 + 1], w2 = qs[j0 + 2], w3 = qs[j0 + 3];
                        const uint32_t b0 = (ks < 4) ? dec_u4x2_to_h2(w0, w1) : dec_u4x2_to_h2(w0 >> 4, w1 >> 4);
                        const uint32_t b1 = (ks < 4) ? dec_u4x2_to_h2(w2, w3) : dec_u4x2_to_h2(w2 >> 4, w3 >> 4);
                        const int d0 = b * 32 + ks * 4;
                        const uint2 ah = qk_head < G ? *reinterpret_cast<const uint2*>(&S.qh[qk_head][d0]) : make_uint2(0, 0);
                        dec_mma884(tg, ah.x, ah.y, b0, b1);
                        if constexpr (SPLIT_HL) {
                            const uint2 al = qk_head < G ? *reinterpret_cast<const uint2*>(&S.ql[qk_head][d0]) : make_uint2(0, 0);
                            dec_mma884(tgl, al.x, al.y, b0, b1);
                        }
                    }
                    auto get_kd = [&](int c) -> float {
                        if (c >= nh || S.srow[c] < 0) return 0.0f;
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(S.kbuf + c * BYTES) + b;
                        return __half2float(__ushort_as_half(blk->d));
                    };
                    const float sc0 = get_kd(cb0), sc1 = get_kd(cb0 + 1), sc4 = get_kd(cb0 + 4), sc5 = get_kd(cb0 + 5);
                    const float sc[8] = {sc0, sc1, sc0, sc1, sc4, sc5, sc4, sc5};
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if constexpr (SPLIT_HL) total[i] = fmaf(tg[i] + tgl[i], sc[i], total[i]);
                        else total[i] = fmaf(tg[i], sc[i], total[i]);
                    }
                }
            } else {
                const uint8_t* krow = S.kbuf + qk_cell * 528;
                float tot_l[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int ks = 0; ks < 32; ++ks) {
                    const int d0 = qk_dh * 128 + ks * 4;
                    const uint2 raw = *reinterpret_cast<const uint2*>(krow + d0 * 2);
                    const uint2 ah = qk_head < G ? *reinterpret_cast<const uint2*>(&S.qh[qk_head][d0]) : make_uint2(0, 0);
                    dec_mma884(total, ah.x, ah.y, raw.x, raw.y);
                    if constexpr (SPLIT_HL) {
                        const uint2 al = qk_head < G ? *reinterpret_cast<const uint2*>(&S.ql[qk_head][d0]) : make_uint2(0, 0);
                        dec_mma884(tot_l, al.x, al.y, raw.x, raw.y);
                    }
                }
                if constexpr (SPLIT_HL) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) total[i] += tot_l[i];
                }
            }
            // Merge dim-half 0 (qp 0,1) and dim-half 1 (qp 2,3) via XOR-8 shuffle!
#pragma unroll
            for (int i = 0; i < 8; ++i) total[i] += __shfl_xor_sync(0xffffffffu, total[i], 8);
            if (qk_dh == 0) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = qk_rb * 8 + crow + 2 * ((i >> 1) & 1);
                    const int c = warp * 8 + ccol + 4 * (i >> 2) + (i & 1);
                    if (row < G) S.sp[row][c] = (c < nh && S.srow[c] >= 0) ? total[i] * qdown : -FLT_MAX;
                }
            }
            __syncthreads();

            // 3. Online softmax over 32 cells (1 warp pass per head; for FP16 also load V into kbuf!).
            if constexpr (KV_MODE == 0) {
                uint4 vx[8];
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const int idx = t + s * THREADS_V70;
                    const int c = idx >> 5, u = idx & 31;
                    const long long r = S.srow[c];
                    vx[s] = r >= 0 ? __ldg(reinterpret_cast<const uint4*>(p.v_pool + r * HD) + u) : make_uint4(0, 0, 0, 0);
                }
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const int idx = t + s * THREADS_V70;
                    const int c = idx >> 5, u = idx & 31;
                    *reinterpret_cast<uint4*>(S.kbuf + c * 528 + u * 16) = vx[s];
                }
            }
            for (int h = warp; h < G; h += 4) {
                const float a = S.sp[h][lane];
                const float m_ch = warp_max(a);
                const float m_old = __shfl_sync(0xffffffffu, lane == 0 ? S.mrow[h] : 0.0f, 0);
                const float m_new = fmaxf(m_old, m_ch);
                const float e = (lane < nh && S.srow[lane] >= 0 && m_new > -FLT_MAX) ? __expf(a - m_new) : 0.0f;
                S.sp[h][lane] = e;
                const float l_ch = warp_sum(e);
                if (lane == 0) {
                    const float al = (m_old > -FLT_MAX && m_new > -FLT_MAX) ? __expf(m_old - m_new) : 0.0f;
                    S.alpha[h] = al;
                    S.lsum[h] = fmaf(S.lsum[h], al, l_ch);
                    S.mrow[h] = m_new;
                }
            }
            __syncthreads();

            // 4. P * V (V is ALREADY in smem! Running vmax_run rescales `acc` directly in-place).
            float cur_vmax = 1.0f;
            if constexpr (KV_MODE == 2 || KV_MODE == 3) {
                constexpr int BYTES = (HD / QK4_0) * (int) sizeof(block_q4_0);
                // Pre-decode all 32x8 Q4_0 V scales once across 128 threads (2 scales/thread, 0 branches in inner loop).
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const int idx = t + s * THREADS_V70;
                    const int c = idx >> 3, b = idx & 7;
                    float dv = 0.0f;
                    if (c < nh && S.srow[c] >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(S.vbuf + c * BYTES) + b;
                        dv = __half2float(__ushort_as_half(blk->d));
                    }
                    S.vs_q4[c][b] = dv;
                }
                __syncthreads();
                const float vm0 = warp_max(fabsf(S.vs_q4[lane][warp * 2]));
                const float vm1 = warp_max(fabsf(S.vs_q4[lane][warp * 2 + 1]));
                cur_vmax = pv_db == 0 ? vm0 : vm1;
            } else if constexpr (KV_MODE == 1) {
                cur_vmax = warp_max(fabsf(S.vs[lane][warp]));
            }
            const float new_vmax = fmaxf(vmax_run, cur_vmax);
            const float v_ratio = (vmax_run > 0.0f && new_vmax > 0.0f) ? (vmax_run / new_vmax) : 0.0f;
            const float a0 = S.alpha[pv_rb * 8 + crow] * v_ratio;
            const float a1 = (pv_rb * 8 + crow + 2 < G) ? S.alpha[pv_rb * 8 + crow + 2] * v_ratio : 0.0f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float sc_a = ((i >> 1) & 1) ? a1 : a0;
                    acc[nt][i] *= sc_a;
                    if constexpr (SPLIT_HL) accl[nt][i] *= sc_a;
                }
            }
            vmax_run = new_vmax;
            const float vup = vmax_run > 0.0f ? (16384.0f / vmax_run) : 0.0f;

            if constexpr (KV_MODE == 2 || KV_MODE == 3) {
                constexpr int BYTES = (HD / QK4_0) * (int) sizeof(block_q4_0);
                const int b_blk = warp * 2 + pv_db;
                const int b_off = b_blk * 18;
                for (int c_rel = 0; c_rel < nh; c_rel += 4) {
                    const float4 pv = *reinterpret_cast<const float4*>(&S.sp[pv_head][c_rel]);
                    const float p0 = pv.x * (S.vs_q4[c_rel + 0][b_blk] * vup);
                    const float p1 = pv.y * (S.vs_q4[c_rel + 1][b_blk] * vup);
                    const float p2 = pv.z * (S.vs_q4[c_rel + 2][b_blk] * vup);
                    const float p3 = pv.w * (S.vs_q4[c_rel + 3][b_blk] * vup);
                    const uint32_t ah0 = dec_pack_h2(p0, p1);
                    const uint32_t ah1 = dec_pack_h2(p2, p3);
                    uint32_t al0 = 0, al1 = 0;
                    if constexpr (SPLIT_HL) {
                        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&ah0));
                        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&ah1));
                        al0 = dec_pack_h2(p0 - f01.x, p1 - f01.y);
                        al1 = dec_pack_h2(p2 - f23.x, p3 - f23.y);
                    }
                    const uint8_t* qs0 = S.vbuf + (c_rel + 0) * BYTES + b_off + 2;
                    const uint8_t* qs1 = S.vbuf + (c_rel + 1) * BYTES + b_off + 2;
                    const uint8_t* qs2 = S.vbuf + (c_rel + 2) * BYTES + b_off + 2;
                    const uint8_t* qs3 = S.vbuf + (c_rel + 3) * BYTES + b_off + 2;
                    // Pair (nt, nt+2): read each qs byte ONCE and unpack both low (nt) and high (nt+2) nibbles!
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const int bi = nt * 8 + jj;
                        const uint32_t y0 = qs0[bi], y1 = qs1[bi], y2 = qs2[bi], y3 = qs3[bi];
                        const uint32_t lo0 = dec_u4x2_to_h2(y0, y1);
                        const uint32_t lo1 = dec_u4x2_to_h2(y2, y3);
                        const uint32_t hi0 = dec_u4x2_to_h2(y0 >> 4, y1 >> 4);
                        const uint32_t hi1 = dec_u4x2_to_h2(y2 >> 4, y3 >> 4);
                        dec_mma884(acc[nt], ah0, ah1, lo0, lo1);
                        dec_mma884(acc[nt + 2], ah0, ah1, hi0, hi1);
                        if constexpr (SPLIT_HL) {
                            dec_mma884(accl[nt], al0, al1, lo0, lo1);
                            dec_mma884(accl[nt + 2], al0, al1, hi0, hi1);
                        }
                    }
                }
            } else if constexpr (KV_MODE == 1) {
                for (int c_rel = 0; c_rel < nh; c_rel += 4) {
                    float w[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) w[e] = S.vs[c_rel + e][warp] * vup;
                    const float4 pv = *reinterpret_cast<const float4*>(&S.sp[pv_head][c_rel]);
                    const float p0 = pv.x * w[0], p1 = pv.y * w[1], p2 = pv.z * w[2], p3 = pv.w * w[3];
                    const uint32_t ah0 = dec_pack_h2(p0, p1);
                    const uint32_t ah1 = dec_pack_h2(p2, p3);
                    uint32_t al0 = 0, al1 = 0;
                    if constexpr (SPLIT_HL) {
                        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&ah0));
                        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&ah1));
                        al0 = dec_pack_h2(p0 - f01.x, p1 - f01.y);
                        al1 = dec_pack_h2(p2 - f23.x, p3 - f23.y);
                    }
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        const int d = warp * 64 + pv_db * 32 + nt * 8 + jj;
                        const uint32_t x0 = (uint32_t) S.vbuf[(c_rel + 0) * 272 + d] |
                                            ((uint32_t) S.vbuf[(c_rel + 1) * 272 + d] << 8);
                        const uint32_t x1 = (uint32_t) S.vbuf[(c_rel + 2) * 272 + d] |
                                            ((uint32_t) S.vbuf[(c_rel + 3) * 272 + d] << 8);
                        const uint32_t b0 = dec_i8x2_to_h2(x0);
                        const uint32_t b1 = dec_i8x2_to_h2(x1);
                        dec_mma884(acc[nt], ah0, ah1, b0, b1);
                        if constexpr (SPLIT_HL) dec_mma884(accl[nt], al0, al1, b0, b1);
                    }
                }
            } else {
                for (int c_rel = 0; c_rel < nh; c_rel += 4) {
                    const float4 pv = *reinterpret_cast<const float4*>(&S.sp[pv_head][c_rel]);
                    const float p0 = pv.x * vup, p1 = pv.y * vup, p2 = pv.z * vup, p3 = pv.w * vup;
                    const uint32_t ah0 = dec_pack_h2(p0, p1);
                    const uint32_t ah1 = dec_pack_h2(p2, p3);
                    uint32_t al0 = 0, al1 = 0;
                    if constexpr (SPLIT_HL) {
                        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&ah0));
                        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&ah1));
                        al0 = dec_pack_h2(p0 - f01.x, p1 - f01.y);
                        al1 = dec_pack_h2(p2 - f23.x, p3 - f23.y);
                    }
                    const __half* v0 = reinterpret_cast<const __half*>(S.kbuf + (c_rel + 0) * 528);
                    const __half* v1 = reinterpret_cast<const __half*>(S.kbuf + (c_rel + 1) * 528);
                    const __half* v2 = reinterpret_cast<const __half*>(S.kbuf + (c_rel + 2) * 528);
                    const __half* v3 = reinterpret_cast<const __half*>(S.kbuf + (c_rel + 3) * 528);
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        const int d = warp * 64 + pv_db * 32 + nt * 8 + jj;
                        const __half2 h0 = __halves2half2(v0[d], v1[d]);
                        const __half2 h1 = __halves2half2(v2[d], v3[d]);
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                        dec_mma884(acc[nt], ah0, ah1, b0, b1);
                        if constexpr (SPLIT_HL) dec_mma884(accl[nt], al0, al1, b0, b1);
                    }
                }
            }
        }
    }

    // Write final split partials (only once per split!).
    if (t < G) {
        part_m[slot * G + t] = S.mrow[t];
        part_l[slot * G + t] = S.lsum[t];
    }
    const float vdown = vmax_run * (1.0f / 16384.0f);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = pv_rb * 8 + crow + 2 * ((i >> 1) & 1);
            const int col = warp * 64 + pv_db * 32 + nt * 8 + ccol + 4 * (i >> 2) + (i & 1);
            if (row < G) {
                const float val = (SPLIT_HL ? (acc[nt][i] + accl[nt][i]) : acc[nt][i]) * vdown;
                part_acc[((size_t) slot * G + row) * HD + col] = val;
            }
        }
    }
}

int attn_v70_mode() {
    static const int env = [] {
        const char* v = std::getenv("STRATA_ATTN_V70");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::atoi(v);
    }();
    if (env == 0) return 0;
    static int cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        cc[dev] = 10 * major + minor;
    }
    if (cc[dev] != 70) return 0;
    return env > 0 ? env : 2;   // default ON (mode 2: 2-pass hi+lo FP16 Q/P mma.m8n8k4, ~2e-6 precision) on sm_70
}

// Target <= 160 CTAs (2 full waves on 80-SM V100, zero tail wave), bounded by [1, n_chunks_cap] so partials always
// fit in the caller's scratch and merge in 1..5 U=16 blocks.
int attn_v70_splits(int64_t cap, int64_t n_head_kv, int64_t n_q) {
    const int n_chunks_cap = (int) ((cap + CHUNK - 1) / CHUNK);
    int want = (int) (160 / (n_q * n_head_kv));
    if (want < 1) want = 1;
    return want < n_chunks_cap ? want : n_chunks_cap;
}

void launch_v70_mma_chunks(int mode, int kv_mode, int n_splits, int64_t n_q, cudaStream_t st, const float* q,
                           const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int n_kv_heads,
                           int page_size, float scale, float* part_acc, float* part_m, float* part_l, int cap,
                           long long stride) {
    const dim3 grid((unsigned) n_splits, (unsigned) n_kv_heads, (unsigned) n_q);
#define STRATA_ATTN_V70_L(M, HL) \
    attn_chunk_v70_mma_kernel<M, HL><<<grid, THREADS_V70, 0, st>>>(q, pools, ids, steps, n_kv_heads, page_size, scale, \
                                                                   part_acc, part_m, part_l, n_splits, cap, stride)
    if (mode >= 2) {
        if (kv_mode == 3) STRATA_ATTN_V70_L(3, true);
        else if (kv_mode == 2) STRATA_ATTN_V70_L(2, true);
        else if (kv_mode == 1) STRATA_ATTN_V70_L(1, true);
        else STRATA_ATTN_V70_L(0, true);
    } else {
        if (kv_mode == 3) STRATA_ATTN_V70_L(3, false);
        else if (kv_mode == 2) STRATA_ATTN_V70_L(2, false);
        else if (kv_mode == 1) STRATA_ATTN_V70_L(1, false);
        else STRATA_ATTN_V70_L(0, false);
    }
#undef STRATA_ATTN_V70_L
}
#endif  // !__HIPCC__ && STRATA_EXPERIMENTAL_SM60

__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks,
                                                        float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int h = blockIdx.x;                 // global query head
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

// attn_merge_kernel's arithmetic in the same chunk order, with two changes that leave every output bit as it was
// (the Daily's 3effa9d5, from Hardin22; qsa_merge_daily_parity compares the two kernels on the same partials, all four
// KV formats):
//
// (1) Bounded by the width.  The chunk kernel writes m = -FLT_MAX, l = 0 for every chunk with c0 >= step[kStepWidth]
//     (its `n_here <= 0` exit) and the merge skips exactly those, so it can stop at n_act = ceil(width / CHUNK):
//     fmaxf with -FLT_MAX never changes M, and a skipped chunk never touches L or acc.  The drafter's merge walks its
//     whole capacity (--mtp-window 32768: 512 chunks) at every context, where a short context fills one or two.
// (2) Loads batched.  attn_merge_kernel loads part_l / part_acc behind its `m == -FLT_MAX` branch, so the compiler
//     issues each chunk's l and acc loads inside that chunk's branch: a few dependent L2 round trips per chunk.  Here
//     the m, l and acc of U chunks are loaded first, guarded only by the chunk index, then consumed in chunk order.
//     The skip is a select on the loaded m instead of a branch, so the loads cannot move back behind it.  The values
//     selected are the very fmaf's attn_merge_kernel computes (same operands, same order), and a skipped chunk leaves
//     L and acc as they were, as its `continue` does.
constexpr int kMergeU = 16;

template <int U, bool TAIL>
__device__ __forceinline__ void merge_max_block(const float* __restrict__ pm, int c0, int n_act, float& M) {
    float mm[U];
#pragma unroll
    for (int u = 0; u < U; ++u) mm[u] = (!TAIL || c0 + u < n_act) ? pm[(size_t) (c0 + u) * G] : -FLT_MAX;
#pragma unroll
    for (int u = 0; u < U; ++u)
        if (!TAIL || c0 + u < n_act) M = fmaxf(M, mm[u]);
}

template <int U, bool TAIL>
__device__ __forceinline__ void merge_sum_block(const float* __restrict__ pm, const float* __restrict__ pl,
                                                const float* __restrict__ pa, int c0, int n_act, float M, float& L,
                                                float& acc) {
    float mm[U], ll[U], aa[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const bool in = !TAIL || c0 + u < n_act;
        mm[u] = in ? pm[(size_t) (c0 + u) * G] : -FLT_MAX;
        ll[u] = in ? pl[(size_t) (c0 + u) * G] : 0.0f;
        aa[u] = in ? pa[(size_t) (c0 + u) * G * HD] : 0.0f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const float w = __expf(mm[u] - M);
        const float L_next = fmaf(ll[u], w, L);
        const float acc_next = fmaf(aa[u], w, acc);
        const bool live = !(mm[u] == -FLT_MAX);   // attn_merge_kernel: `if (m == -FLT_MAX) continue;`
        L = live ? L_next : L;
        acc = live ? acc_next : acc;
    }
}

template <int U>
__global__ void __launch_bounds__(HD) attn_merge_v2_kernel(const float* __restrict__ part_acc,
                                                           const float* __restrict__ part_m,
                                                           const float* __restrict__ part_l, int n_chunks,
                                                           const int32_t* __restrict__ steps,
                                                           float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    // the step record the chunk kernel read for this query (attn_chunk_kernel: step + blockIdx.z * kStepCount)
    const int width = __ldg(steps + (size_t) blockIdx.y * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    const int h = blockIdx.x;                 // global query head
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    // chunk c of this head: part_m / part_l [(kvh * n_chunks + c) * G + hl], part_acc [that * HD + d]
    const size_t base = (size_t) kvh * (size_t) n_chunks * G + (size_t) hl;
    const float* pm = part_m + base;
    const float* pl = part_l + base;
    const float* pa = part_acc + base * HD + (size_t) d;
    const int n_full = n_act - n_act % U;
    float M = -FLT_MAX;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_max_block<U, false>(pm, c0, n_act, M);
    if (n_full < n_act) merge_max_block<U, true>(pm, n_full, n_act, M);
    float L = 0.0f, acc = 0.0f;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_sum_block<U, false>(pm, pl, pa, c0, n_act, M, L, acc);
    if (n_full < n_act) merge_sum_block<U, true>(pm, pl, pa, n_full, n_act, M, L, acc);
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

#if defined(STRATA_HIP_GFX906)
// The compat backend can target other architectures. Check the calling thread's current device,
// caching only its last ordinal so a layer split or another thread cannot reuse the wrong answer.
bool query_swizzle_gfx906_device() {
    static thread_local int cached_device = -1;
    static thread_local bool cached_supported = false;
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device < 0) {
        cached_device = -1;
        cached_supported = false;
        (void) hipGetLastError();
        return false;
    }
    if (device != cached_device) {
        cached_device = -1;
        cached_supported = false;
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
            (void) hipGetLastError();
            return false;
        }
        cached_supported = gfx_arch_is(prop.gcnArchName, "gfx906") && prop.warpSize == 64;
        cached_device = device;
    }
    return cached_supported;
}
#endif

// Fork (lab/upsync): STRATA_ATTN_DAILY=1 runs the Daily's (be4cf1c) decode attention on every CUDA card: the chunk
// pass on PR #540's kernel (attn_chunk_kernel_pre75<M, 2, false>, the Daily's attn_chunk_kernel), bit for bit the
// default kernel's partials.  CUDA only; unset or anything but "1" keeps upstream 0.1.42's kernels.
// STRATA_ATTN_PRE75=0 still turns the chunk kernel off (A/B).
// 147-verify: unset, it is on for a compute capability 12.0 card (sm120_series.hpp; RTX PRO 5000, a 4-token verify round
// -2.6%, bitwise); STRATA_ATTN_DAILY=0 (anything but "1") keeps upstream's kernels there too.
[[maybe_unused]] bool daily_attn() {
#if defined(__HIPCC__)
    return false;
#else
    static const int env = [] {
        const char* v = std::getenv("STRATA_ATTN_DAILY");
        if (v == nullptr) return -1;
        return v[0] == '1' && v[1] == '\0' ? 1 : 0;
    }();
    if (env >= 0) return env != 0;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (!per_dev[dev]) {
        int maj = 0, mnr = 0;
        const bool ok = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        per_dev[dev] = ok && strata::sm12x_default(maj, mnr, strata::kGb10AttnDaily) ? 2 : 1;
    }
    return per_dev[dev] == 2;
#endif
}

#if defined(STRATA_ATTN_PRE75_BUILT)
// CUDA: the current device is below sm_75 (experimental build) or STRATA_ATTN_DAILY=1 (any card); HIP: it is gfx103x
// (per device: a layer split can mix cards).
// STRATA_ATTN_PRE75=0 turns PR #540's kernel off (A/B); on HIP =1 also runs it on another wave32 card.
bool pre75_attn() {
    static const int env = [] {
        const char* e = std::getenv("STRATA_ATTN_PRE75");
        return e == nullptr ? -1 : e[0] == '0' ? 0 : e[0] == '1' ? 1 : -1;
    }();
    if (env == 0) return false;
#if !defined(__HIPCC__) && !defined(STRATA_EXPERIMENTAL_SM60)
    return daily_attn();   // the ready-made engine: upstream's attn_chunk_kernel unless STRATA_ATTN_DAILY=1
#else
#if !defined(__HIPCC__)
    if (daily_attn()) return true;
#endif
    static int cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
#if defined(__HIPCC__)
    if (cc[dev] == 0) {   // 2: gfx103x, 1: another card
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        cc[dev] = std::strncmp(prop.gcnArchName, "gfx103", 6) == 0 ? 2 : 1;
    }
    return env == 1 || cc[dev] == 2;
#else
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = 10 * major + minor;
    }
    return cc[dev] < 75;
#endif
#endif  // ready-made CUDA engine
}
// `PRE` picks PR #540's kernel; the environment's choice is STRATA_ATTN_CHUNK.
#if defined(__HIPCC__)
#define STRATA_ATTN_CHUNK_SEL(M, PRE) ((PRE) ? attn_chunk_kernel_pre75<M, 8, true> : attn_chunk_kernel<M>)
#else
#define STRATA_ATTN_CHUNK_SEL(M, PRE) ((PRE) ? attn_chunk_kernel_pre75<M> : attn_chunk_kernel<M>)
#endif
#define STRATA_ATTN_CHUNK(M) STRATA_ATTN_CHUNK_SEL(M, pre75_attn())
#else
#define STRATA_ATTN_CHUNK_SEL(M, PRE) attn_chunk_kernel<M>
#define STRATA_ATTN_CHUNK(M) attn_chunk_kernel<M>
#endif

#if defined(STRATA_ATTN_PRE75_BUILT)
// Opt-in (default OFF): unlock batched multi-slot / multi-row pre75 attention when STRATA_ATTN_DAILY=1 / pre75_attn()
// is active. Explicit STRATA_ATTN_PRE75_SLOTS=0/1 wins; otherwise follows STRATA_BATCH_QSA_STREAM_OPT.
bool pre75_slots_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ATTN_PRE75_SLOTS");
        if (v != nullptr) return std::atoi(v) != 0;
        const char* b = std::getenv("STRATA_BATCH_QSA_STREAM_OPT");
        return b != nullptr && std::atoi(b) != 0;
    }();
    return on;
}
#endif

// S25: STRATA_ATTN_LANECELL=1 - the score phase one cell per thread (bit-identical scores, no shuffle reductions)
bool lane_cell_attn() {
    static const bool on = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
    return on;
}

// 0 fp16, 1 int8, 2 q4_0, 3 int8 K with q4_0 V
int attn_kv_mode(const QsaAttnPools& p) {
    return p.k_q4 != nullptr ? 2 : (p.k_q != nullptr && p.v_q4 != nullptr ? 3 : (p.k_q != nullptr ? 1 : 0));
}

// STRATA_ATTN_MERGE_V2: 1 merges with attn_merge_v2_kernel, 0 with attn_merge_kernel.  Unset it follows
// STRATA_ATTN_DAILY (the Daily had it on by default on NVIDIA), so upstream's merger stays the default.
int attn_merge_v2() {
    static const int env = [] {
        const char* v = std::getenv("STRATA_ATTN_MERGE_V2");
        return v != nullptr && *v != '\0' ? (std::atoi(v) != 0 ? 1 : 0) : -1;
    }();
    return env >= 0 ? env : (daily_attn() ? 1 : 0);
}

// the merge of `n_q` queries' partials: v2 = 0 attn_merge_kernel, 1 attn_merge_v2_kernel (bitwise the same output)
void launch_merge(int v2, const float* part_acc, const float* part_m, const float* part_l, int n_chunks,
                  const int32_t* steps, float* attn, long long stride, unsigned n_head, unsigned n_q, cudaStream_t st) {
    if (v2)
        attn_merge_v2_kernel<kMergeU><<<dim3(n_head, n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, steps,
                                                                         attn, stride);
    else
        attn_merge_kernel<<<dim3(n_head, n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn, stride);
}

// qsa_decode_attn_batch's chunk pass: `n_q` queries' partials into `scratch` (one stride each).  `kernel` < 0 is the
// environment's choice (lane cell, gfx906 swizzle, STRATA_ATTN_DAILY / PR #540); 0 is attn_chunk_kernel and 1 PR #540's
// kernel where it is built (the parity harness: never the environment).
void launch_chunks_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                         int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, cudaStream_t st, int kernel) {
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
#if defined(STRATA_ATTN_PRE75_BUILT)
    const bool pre = kernel < 0 ? pre75_attn() : kernel == 1;
#endif
    // S25: STRATA_ATTN_LANECELL=1 - the score phase one cell per thread (bit-identical scores, no shuffle reductions)
    const bool lane_cell = kernel < 0 && lane_cell_attn();
#if defined(STRATA_HIP_GFX906)
    static const bool query_swizzle = [] {
        const char* v = std::getenv("STRATA_GFX906_ATTN_QUERY_SWIZZLE"); return v && v[0] == '1';
    }();
    const bool use_query_swizzle = kernel < 0 && query_swizzle && !lane_cell && kv_mode == 1 &&
                                   query_swizzle_gfx906_device();
    static const bool query_swizzle_trace = std::getenv("STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE") != nullptr;
    static thread_local bool query_swizzle_said = false;
    if (query_swizzle_trace && !query_swizzle_said) {
        query_swizzle_said = true;
        std::fprintf(stderr, "strata attn-query-swizzle: active=%d nq=%lld cap=%lld kv_mode=%d lane_cell=%d\n",
                     use_query_swizzle, (long long) n_q, (long long) cap, kv_mode, lane_cell);
    }
#endif
#define STRATA_ATTN_LC(M) attn_chunk_kernel<M, true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
    if (lane_cell) {
        if (kv_mode == 3) STRATA_ATTN_LC(3);
        else if (kv_mode == 2) STRATA_ATTN_LC(2);
        else if (kv_mode == 1) STRATA_ATTN_LC(1);
        else STRATA_ATTN_LC(0);
    } else
#undef STRATA_ATTN_LC
#if defined(STRATA_HIP_GFX906)
    // Opt-in INT8 batched attention: prefill, verification and MTP (including n_q=1).
    // Other pool formats and lane-cell scoring keep their paths; qsa_decode_attn_step is unchanged.
    // A k8v4 model's MTP drafter uses INT8 pools and can therefore take this path.
    if (use_query_swizzle) {
        // use_query_swizzle already requires INT8, warp scoring, and the calling
        // thread's current device to be exactly gfx906 with a 64-lane wavefront.
        static const bool reduce12 = [] {
            const char* v = std::getenv("STRATA_GFX906_ATTN_REDUCE12");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        if (reduce12) {
            attn_chunk_kernel<1, false, true, true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps,
                (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
            static const bool trace = [] {
                const char* v = std::getenv("STRATA_GFX906_ATTN_REDUCE12_TRACE");
                return v && v[0] == '1' && v[1] == '\0';
            }();
            static thread_local bool said = false;
            if (trace && !said) {
                said = true;
                std::fprintf(stderr, "strata attn-reduce12: selected=1 query_swizzle=1 kv_mode=1 nq=%lld cap=%lld\n",
                             (long long) n_q, (long long) cap);
            }
        } else {
            attn_chunk_kernel<1, false, true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps,
                (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
        }
    } else
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    if (kernel < 0 && !lane_cell) {
        const int v70 = attn_v70_mode();
        if (v70 > 0) {
            const int nsp = attn_v70_splits(cap, s.n_head_kv, n_q);
            float* pa = scratch;
            float* pm = scratch + (size_t) nsp * s.n_head * HD;
            float* pl = pm + (size_t) nsp * s.n_head;
            launch_v70_mma_chunks(v70, kv_mode, nsp, n_q, st, q, pools, ids, steps, (int) s.n_head_kv,
                                  (int) s.page_size, scale, pa, pm, pl, (int) cap, stride);
            return;
        }
    }
#endif
#if !defined(__HIPCC__)
    if (kernel < 0 && kv_mode == 1 && attn_v3_on())   // (the parity harness names its kernel: never the environment)
        attn_chunk_q8_v3_kernel<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else
#endif
    if (kv_mode == 3)
        STRATA_ATTN_CHUNK_SEL(3, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
                                                                n_chunks, (int) cap, stride);
    else if (kv_mode == 2)
        STRATA_ATTN_CHUNK_SEL(2, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
                                                                n_chunks, (int) cap, stride);
    else if (kv_mode == 1)
        STRATA_ATTN_CHUNK_SEL(1, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
                                                                n_chunks, (int) cap, stride);
    else
        STRATA_ATTN_CHUNK_SEL(0, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
                                                                n_chunks, (int) cap, stride);
}

// qsa_decode_attn_batch's merge pass over the partials launch_chunks_batch wrote; `v2` names the merge kernel
void launch_merge_batch(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch, float* attn,
                        int64_t n_q, int v2, cudaStream_t st) {
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    const float* part_acc = scratch;
    const float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    const float* part_l = part_m + (size_t) n_chunks * s.n_head;
    launch_merge(v2, part_acc, part_m, part_l, n_chunks, steps, attn, stride, (unsigned) s.n_head, (unsigned) n_q, st);
}

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    if (!lane_cell_attn() && attn_v70_mode() > 0) {
        const int nsp = attn_v70_splits(cap, s.n_head_kv, n_q);
        const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
        float* pa = scratch;
        float* pm = scratch + (size_t) nsp * s.n_head * HD;
        float* pl = pm + (size_t) nsp * s.n_head;
        const int kv_mode = attn_kv_mode(pools);
        const float scale = 1.0f / sqrtf((float) HD);
        launch_v70_mma_chunks(attn_v70_mode(), kv_mode, nsp, n_q, st, q, pools, ids, steps, (int) s.n_head_kv,
                              (int) s.page_size, scale, pa, pm, pl, (int) cap, stride);
        launch_merge(attn_merge_v2(), pa, pm, pl, nsp, steps, attn, stride, (unsigned) s.n_head, (unsigned) n_q, st);
    } else
#endif
    {
        launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, st, -1);
        launch_merge_batch(steps, cap, s, scratch, attn, n_q, attn_merge_v2(), st);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool qsa_decode_attn_slots(const float* q, const QsaAttnSlotPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    if (n_q > kQsaAttnMaxSlots) return false;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps) {
        std::fprintf(stderr, "qsa_decode_attn_slots: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = attn_kv_mode(pools.p[0]);
    for (int64_t z = 0; z < n_q; ++z) {
        if (!pools.p[z].page_table) {
            std::fprintf(stderr, "qsa_decode_attn_slots: missing page table\n");
            std::exit(1);
        }
        if (attn_kv_mode(pools.p[z]) != kv_mode) return false;
    }
    const bool lane_cell = lane_cell_attn();
#if defined(STRATA_ATTN_PRE75_BUILT)
    bool use_pre75 = false;
    if (!lane_cell && pre75_attn()) {
        if (!pre75_slots_on()) return false;
        use_pre75 = true;
    }
#endif
#if defined(STRATA_HIP_GFX906)
    static const bool query_swizzle = [] {
        const char* v = std::getenv("STRATA_GFX906_ATTN_QUERY_SWIZZLE"); return v && v[0] == '1';
    }();
    if (query_swizzle && !lane_cell && kv_mode == 1 && query_swizzle_gfx906_device()) return false;
#endif
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    cudaStream_t st = (cudaStream_t) stream;
#if defined(STRATA_ATTN_PRE75_BUILT)
    if (use_pre75) {
#if defined(__HIPCC__)
#define STRATA_ATTN_SLOTS_PRE75(M) attn_chunk_slots_pre75_kernel<M, 8, true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
#else
#define STRATA_ATTN_SLOTS_PRE75(M) attn_chunk_slots_pre75_kernel<M, 2, false><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
#endif
        if (kv_mode == 3) STRATA_ATTN_SLOTS_PRE75(3);
        else if (kv_mode == 2) STRATA_ATTN_SLOTS_PRE75(2);
        else if (kv_mode == 1) STRATA_ATTN_SLOTS_PRE75(1);
        else STRATA_ATTN_SLOTS_PRE75(0);
#undef STRATA_ATTN_SLOTS_PRE75
    } else
#endif
    {
#define STRATA_ATTN_SLOTS(M, LC) attn_chunk_slots_kernel<M, LC><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
        if (lane_cell) {
            if (kv_mode == 3) STRATA_ATTN_SLOTS(3, true);
            else if (kv_mode == 2) STRATA_ATTN_SLOTS(2, true);
            else if (kv_mode == 1) STRATA_ATTN_SLOTS(1, true);
            else STRATA_ATTN_SLOTS(0, true);
        } else {
            if (kv_mode == 3) STRATA_ATTN_SLOTS(3, false);
            else if (kv_mode == 2) STRATA_ATTN_SLOTS(2, false);
            else if (kv_mode == 1) STRATA_ATTN_SLOTS(1, false);
            else STRATA_ATTN_SLOTS(0, false);
        }
#undef STRATA_ATTN_SLOTS
    }
    launch_merge(attn_merge_v2(), part_acc, part_m, part_l, n_chunks, steps, attn, stride, (unsigned) s.n_head,
                 (unsigned) n_q, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_slots: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

void qsa_decode_attn_batch_rows(const float* q, const QsaAttnPoolsRows& rows, const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    const QsaAttnPools& pools = rows.p[0];
    if (rows.n != n_q || n_q > kAttnRowsMax || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 ||
        !scratch || !ids || !steps) {
        std::fprintf(stderr, "qsa_decode_attn_batch_rows: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = attn_kv_mode(pools);
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    cudaStream_t st = (cudaStream_t) stream;
    // The plain kernel is what qsa_decode_attn_batch runs unless one of its variants is on (lane-cell scoring, the
    // pre-sm_75 / gfx103x kernel, the gfx906 swizzle): there each row goes through that call, one launch per row.
    static const bool lane_cell_rows = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
    bool per_row = lane_cell_rows;
    for (int64_t r = 0; !per_row && r < n_q; ++r) {
        if (!rows.p[r].page_table || attn_kv_mode(rows.p[r]) != kv_mode) per_row = true;
    }
#if defined(STRATA_ATTN_PRE75_BUILT)
    bool use_pre75 = false;
    if (!per_row && pre75_attn()) {
        if (pre75_slots_on()) use_pre75 = true;
        else per_row = true;
    }
#endif
#if defined(STRATA_HIP_GFX906)
    per_row = true;
#if defined(STRATA_ATTN_PRE75_BUILT)
    use_pre75 = false;
#endif
#endif
    if (per_row) {
        for (int64_t r = 0; r < n_q; ++r)
            qsa_decode_attn_batch(q + (size_t) r * s.n_head * HD, rows.p[r], ids + (size_t) r * cap,
                                  steps + (size_t) r * kStepCount, cap, s, scratch + (size_t) r * stride,
                                  attn + (size_t) r * s.n_head * HD, 1, stream);
        return;
    }
#if defined(STRATA_ATTN_PRE75_BUILT)
    if (use_pre75) {
#if defined(__HIPCC__)
#define STRATA_ATTN_ROWS_PRE75(M) attn_chunk_rows_pre75_kernel<M, 8, true><<<grid, THREADS, 0, st>>>(q, rows, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
#else
#define STRATA_ATTN_ROWS_PRE75(M) attn_chunk_rows_pre75_kernel<M, 2, false><<<grid, THREADS, 0, st>>>(q, rows, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
#endif
        if (kv_mode == 3) STRATA_ATTN_ROWS_PRE75(3);
        else if (kv_mode == 2) STRATA_ATTN_ROWS_PRE75(2);
        else if (kv_mode == 1) STRATA_ATTN_ROWS_PRE75(1);
        else STRATA_ATTN_ROWS_PRE75(0);
#undef STRATA_ATTN_ROWS_PRE75
    } else
#endif
    {
#define STRATA_ATTN_ROWS(M) attn_chunk_rows_kernel<M><<<grid, THREADS, 0, st>>>(q, rows, ids, steps, (int) s.n_head_kv, \
        (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
        if (kv_mode == 3) STRATA_ATTN_ROWS(3);
        else if (kv_mode == 2) STRATA_ATTN_ROWS(2);
        else if (kv_mode == 1) STRATA_ATTN_ROWS(1);
        else STRATA_ATTN_ROWS(0);
#undef STRATA_ATTN_ROWS
    }
    launch_merge(attn_merge_v2(), part_acc, part_m, part_l, n_chunks, steps, attn, stride, (unsigned) s.n_head,
                 (unsigned) n_q, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch_rows: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv);
    cudaStream_t st = (cudaStream_t) stream;
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    if (const int v70 = attn_v70_mode(); v70 > 0) {
        const int nsp = attn_v70_splits(cap, s.n_head_kv, 1);
        float* pa = scratch;
        float* pm = scratch + (size_t) nsp * s.n_head * HD;
        float* pl = pm + (size_t) nsp * s.n_head;
        launch_v70_mma_chunks(v70, kv_mode, nsp, 1, st, q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                              scale, pa, pm, pl, (int) cap, 0);
        launch_merge(attn_merge_v2(), pa, pm, pl, nsp, step, attn, 0, (unsigned) s.n_head, 1u, st);
    } else
#endif
    {
#if !defined(__HIPCC__)
    if (kv_mode == 1 && attn_v3_on())
        attn_chunk_q8_v3_kernel<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else
#endif
    if (kv_mode == 3)
        STRATA_ATTN_CHUNK(3)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else if (kv_mode == 2)
        STRATA_ATTN_CHUNK(2)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else if (kv_mode == 1)
        STRATA_ATTN_CHUNK(1)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else
        STRATA_ATTN_CHUNK(0)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    launch_merge(attn_merge_v2(), part_acc, part_m, part_l, n_chunks, step, attn, 0, (unsigned) s.n_head, 1u, st);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// qsa_merge_daily_parity: qsa_decode_attn_batch's two passes apart, the chunk kernel and the merge named by the
// arguments (never the environment), so the Daily's kernels and upstream's can be run on the same inputs and the
// outputs compared bit for bit.  The engine never calls these.
void qsa_decode_attn_chunks_only(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                 int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, int daily_chunk,
                                 void* stream) {
    if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
        !q || !ids || !steps || !pools.page_table || (daily_chunk != 0 && daily_chunk != 1)) {
        std::fprintf(stderr, "qsa_decode_attn_chunks_only: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, (cudaStream_t) stream, daily_chunk);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_chunks_only: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void qsa_decode_attn_merge_only(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch,
                                float* attn, int64_t n_q, int v2, void* stream) {
    if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
        (v2 != 0 && v2 != 1) || !steps || !attn) {
        std::fprintf(stderr, "qsa_decode_attn_merge_only: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    launch_merge_batch(steps, cap, s, scratch, attn, n_q, v2, (cudaStream_t) stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_merge_only: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
