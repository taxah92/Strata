// src/kernels/cuda/i8h_volta.cuh - the Volta form of the int8 expert kernels (TUNE-0216 / PR 4.3): the same products as i8_kernel, on the tensor cores.
// Included by i8_expert.cuh (inside iq_kernels.cu's anonymous namespace) after the tile kernels; it uses their decode (i8_load / i8_pack / i8_convert), their
// activation image and their tile tables.  CUDA only; the device code is Volta's (__CUDA_ARCH__ 700): every other architecture compiles a trap.
//
// Why.  On a Tesla V100 (no int8 matrix cores) the dp4a tile kernels read prompts of 8,192+ tokens at 0.75-0.95 of MMQ's speed per layer, and with every expert
// resident (#6053's 6x V100) a 26K prompt's expert products took 1.30 times MMQ's time.  Volta's FP16 tensor cores do 512 FMA a clock per SM against dp4a's 256.
//
// Exact, by construction.  The weights' int8 codes (the decoded sub-blocks i8_convert writes) and the activations' int8 codes become FP16 values (integers of
// -128..127: exact), mma.m8n8k4 multiplies them exactly and adds them in FP32; every partial sum is an integer below 2^24 (32 products of at most 127 x 127), so it is
// exact in any order.  A fresh accumulator collects each 32-value sub-block (each 16-value half for the two-scale formats), and the epilogue then does what the dp4a
// kernels do with their integer sums, in the same order: t = s1 m1 + s0 m0 (or s m, or s), acc = fmaf(t, base * d_x, acc), sub-block after sub-block.  The products
// t are integers below 2^24 as well (the dp4a kernels' own bound), so t is the same number whether made in int or float: the output is bit-identical to i8_kernel's.
//
// Shape.  64 weight rows (gate/up: 32 gate + the same 32 up features, as i8_kernel) x BM entries of one expert, K in stages of 128 values (4 sub-blocks):
//   - BM = 64 (tile code 5): 256 threads (8 warps), 2 blocks/SM (<= 41.7 KB smem), warp w owns 16 rows x 32 entries.
//   - BM = 32 (tile code 6): 128 threads (4 warps), 3 blocks/SM (<= 32.6 KB smem), warp w owns 16 rows x 32 entries (each thread decodes 2 weight rows + 1 act block).
// Both keep 3 LDS.64 per 2 HMMA.884 (TC-bound, vs 2 LDS.64 per 1 HMMA for an 8-warp 16x16 BM=32 layout).
// I8H_LDH = 132 halves (264 B = 66 banks = 2 mod 32) eliminates 2-way bank conflicts across 16 rows/cols on LDS.64; I8H_RS = 10 words packs {int16 m0, int16 m1, float base}
// with 0 bank conflicts on s_r and fits every i8h_type format in <= 32,640 B for BM=32 (3 blocks per 96 KB Volta SM).
// Warp-uniform partial-tile guards skip empty 16-entry halves (do_j0 = warp_c0 < nv, do_j1 = warp_c0 + 16 < nv) and activation loads for entries >= nv.
//
// The m8n8k4 fragments (PTX ISA, mma.m8n8k4 .row.col, f16 inputs, f32 accumulators; checked on the V100 by phaseA-tests/tune0216/probe_m884.cu): quad pair
// q = (lane % 16) / 4 holds lanes 4q..4q+3 and 4q+16..4q+19, local index t = lane % 4 + 4 (lane / 16); thread t supplies row t of A (k 0..3) and column t of B
// (k 0..3); accumulator element i of thread (g = lane % 4, h = lane / 16) is row (g & 1) + (i & 2) + 4 h, column (i & 4) + (g & 2) + (i & 1).
#pragma once

constexpr int I8H_BM = 64;            // entries of the wide Volta tile (tile code 5)
constexpr int I8H_LDH = 132;          // halves per shared row: 128 + 4 (2-bank stride -> 0 bank conflicts across 16 rows/cols on LDS.64)
constexpr int I8H_RS = 10;            // words per row of packed scale records: 4 sub-blocks x {m01, base} + 2 pad (0 bank conflicts on LDS.64)
// formats of the Volta form: the decodes with no per-32 minimum (the minimum formats Q4_K / Q5_K / Q5_1 keep the dp4a kernels)
__host__ __device__ constexpr bool i8h_type(int t) { return !i8_min_type(t) && (i8_gu_type(t) || i8_d_type(t)); }

template <int T, int BM = I8H_BM>
struct I8HSmem {
    static constexpr int GW = i8_gg_type(T) ? 0 : (i8_grid_words(T) + 3) / 4 * 4;   // a big grid is read from global memory (as I8Smem's GG)
    static constexpr int SG = i8_sign_type(T) ? 512 : 0;
    static constexpr int SZ_W = 64 * I8H_LDH / 2, SZ_A = BM * I8H_LDH / 2;          // words (4224; 2112 for BM=32, 4224 for BM=64)
    static constexpr int SZ_R = 64 * I8H_RS, SZ_D = BM * 4;
    static constexpr int o_sg = GW, o_w = o_sg + SG, o_a = o_w + SZ_W, o_r = o_a + SZ_A, o_d = o_r + SZ_R, o_src = o_d + SZ_D,
                         total = (o_src + BM + 3) / 4 * 4;
    static constexpr size_t bytes = (size_t) total * 4;
};

static_assert(I8HSmem<16, 32>::bytes <= 32768 && I8HSmem<17, 32>::bytes <= 32768 && I8HSmem<18, 32>::bytes <= 32768 &&
              I8HSmem<21, 32>::bytes <= 32768 && I8HSmem<22, 32>::bytes <= 32768 && I8HSmem<23, 32>::bytes <= 32768 &&
              I8HSmem<29, 32>::bytes <= 32768 && I8HSmem<20, 32>::bytes <= 32768 && I8HSmem<42, 32>::bytes <= 32768 &&
              I8HSmem<6, 32>::bytes <= 32768 && I8HSmem<8, 32>::bytes <= 32768 && I8HSmem<14, 32>::bytes <= 32768,
              "BM=32 Volta i8h tile must fit <= 32 KB (3 blocks per 96 KB SM)");

// four int8 (bytes of w) as four FP16 values, exactly: x + 128 is the mantissa of 1024 + x + 128, minus 1152
__device__ __forceinline__ uint2 i8h_cvt4(uint32_t w) {
    const uint32_t c = w ^ 0x80808080u;
    const uint32_t lo = __byte_perm(c, 0x64646464u, 0x5140), hi = __byte_perm(c, 0x64646464u, 0x7362);
    const __half2 m = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 l = __hsub2(*(const __half2*) &lo, m), h = __hsub2(*(const __half2*) &hi, m);
    return make_uint2(*(const uint32_t*) &l, *(const uint32_t*) &h);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
__device__ __forceinline__ void i8h_mma(float (&d)[8], uint2 a, uint2 b) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}
#endif

template <int T, bool GU, int BM = I8H_BM>
__global__ void __launch_bounds__(BM * 4, BM == 32 ? 3 : 2)
i8h_kernel(const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
           const int32_t* __restrict__ tile_off, const int32_t* __restrict__ ent_idx, const block_q8_1* __restrict__ A,
           block_q8_1* __restrict__ hq, float* __restrict__ out, NativeExpertLayout L) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    static_assert(BM == 32 || BM == 64, "i8h_kernel supports BM=32 and BM=64");
    constexpr int NT = BM * 4;   // 256 (8 warps) for BM=64, 128 (4 warps) for BM=32
    using SM = I8HSmem<T, BM>;
    extern __shared__ uint4 i8_dyn[];
    uint32_t* const smem = (uint32_t*) i8_dyn;
    uint32_t* const s_grid = smem;
    uint32_t* const s_sg = smem + SM::o_sg;
    __half* const s_w = (__half*) (smem + SM::o_w);
    __half* const s_a = (__half*) (smem + SM::o_a);
    uint32_t* const s_r = smem + SM::o_r;
    float* const s_ad = (float*) (smem + SM::o_d);
    int* const s_src = (int*) (smem + SM::o_src);
    const uint32_t* const grid_rd = i8_gg_type(T) ? i8_grid_src<T>() : s_grid;

    const int ng = *n_groups;
    const int ti = blockIdx.x;
    if (ti >= tile_off[ng]) return;
    int lo = 0, hi = ng - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tile_off[mid] <= ti) lo = mid; else hi = mid - 1;
    }
    const int g = lo;
    const int e_lo = grp_start[g] + (ti - tile_off[g]) * BM;
    const int e_hi = min(grp_start[g + 1], e_lo + BM);
    const int nv = e_hi - e_lo;
    I8P blob = i8_uniform(I8_GPTR(grp_ptr[g]));
    const int tid = threadIdx.x;
    const int K = GU ? (int) L.n_embd : (int) L.n_ff;
    const int ns = K / I8_KS;
    const size_t wrb = GU ? L.gu_row : L.d_row;

    if constexpr (i8_grid_words(T) > 4 && !i8_gg_type(T)) {
        const uint32_t* src = i8_grid_src<T>();
        for (int i = tid; i < i8_grid_words(T); i += NT) s_grid[i] = src[i];
    }
    if constexpr (i8_sign_type(T)) {   // as i8_kernel's table
        for (uint32_t b = (uint32_t) tid; b < 256u; b += NT) {
            uint32_t sb = b;
            if constexpr (T != 21 && T != 22) sb = (b & 127u) | (((uint32_t) __popc(b & 127u) & 1u) << 7);
            uint32_t mlo = 0, mhi = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                mlo |= ((sb >> k) & 1u) ? (0xFFu << (8 * k)) : 0u;
                mhi |= ((sb >> (4 + k)) & 1u) ? (0xFFu << (8 * k)) : 0u;
            }
            *(uint2*) (s_sg + 2 * b) = make_uint2(mlo, mhi);
        }
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (T == 20 || T == 23) {
        const uint32_t* t = (const uint32_t*) kvalues_iq4nl;
#pragma unroll
        for (int i = 0; i < 4; ++i) kv[i] = t[i];
    }
    if (tid < BM) s_src[tid] = (e_lo + tid < e_hi) ? (GU ? ent_idx[e_lo + tid] : e_lo + tid) : -1;
    __syncthreads();

    // decode thread: BM=64 -> 1 weight row (wr0) + 1 act block; BM=32 -> 2 weight rows (wr0, wr0+32) + 1 act block
    const int wr0 = tid >> 2, wp = tid & 3;
    uint32_t row_off0;
    if constexpr (GU) row_off0 = (uint32_t) ((wr0 >= 32 ? L.up_off : (size_t) 0) + (size_t) (blockIdx.y * 32 + (wr0 & 31)) * wrb);
    else row_off0 = (uint32_t) (L.down_off + (size_t) (blockIdx.y * 64 + wr0) * wrb);
    const I8Off woff0 = i8_offs<T>(row_off0, wp);
    uint32_t row_off1 = 0u;
    if constexpr (BM == 32) {
        if constexpr (GU) row_off1 = (uint32_t) (L.up_off + (size_t) (blockIdx.y * 32 + wr0) * wrb);
        else row_off1 = (uint32_t) (L.down_off + (size_t) (blockIdx.y * 64 + wr0 + 32) * wrb);
    }
    const I8Off woff1 = i8_offs<T>(row_off1, wp);

    I8P const abase = I8_GPTR(A);
    const bool a_active = wr0 < nv;
    uint32_t a_vo = 0u;
    if (a_active) {
        const int src = s_src[wr0];
        a_vo = (uint32_t) ((size_t) src * (size_t) ((K / 32) * 36) + (size_t) wp * 36);
    } else {
        // Initialize inactive padding rows in s_a/s_ad once so partial-tile MMA reads clean zeros (initcheck-safe)
        uint2* d = (uint2*) (s_a + wr0 * I8H_LDH + wp * 32);
#pragma unroll
        for (int k = 0; k < 8; ++k) d[k] = make_uint2(0u, 0u);
        s_ad[tid] = 0.0f;
    }

    auto store_w_one = [&](int wr, int st, const uint32_t (&r)[I8NR<T>]) {
        uint32_t q[8], w[I8NW<T>];
        int m0, m1;
        float base, minv = 0.0f;
        const int ibs = (st * 4 + wp) & 7;
        i8_pack<T>(r, ibs, w);
        i8_convert<T>(w, grid_rd, (const uint2*) s_sg, kv, q, m0, m1, base, ibs, &minv);
        uint2* dst = (uint2*) (s_w + wr * I8H_LDH + wp * 32);
#pragma unroll
        for (int k = 0; k < 8; ++k) dst[k] = i8h_cvt4(q[k]);
        const uint32_t m01 = ((uint32_t) (uint16_t) (int16_t) m0) | (((uint32_t) (uint16_t) (int16_t) m1) << 16);
        *(uint2*) (s_r + wr * I8H_RS + wp * 2) = make_uint2(m01, __float_as_uint(base));
    };
    auto store_a = [&](const uint32_t (&ar)[9]) {
        if (a_active) {
            uint2* d = (uint2*) (s_a + wr0 * I8H_LDH + wp * 32);
#pragma unroll
            for (int k = 0; k < 8; ++k) d[k] = i8h_cvt4(ar[1 + k]);
            s_ad[tid] = __uint_as_float(ar[0]);
        }
    };

    const int lane = tid & 31, warp = tid >> 5;
    const int qp = (lane & 15) >> 2, tl = (lane & 3) + 4 * (lane >> 4);
    const int gg = lane & 3, hh = lane >> 4;
    const int r0 = 16 * (warp & 3) + 8 * (qp & 1);                       // the quad pair's 8 rows
    const int warp_c0 = (BM == 64) ? (32 * (warp >> 2)) : 0;             // warp-uniform starting entry of this warp's 32-entry band
    const int c0 = warp_c0 + 8 * (qp >> 1);                              // quad pair's 8 entries of first 16-entry half (second: +16)
    const bool do_j0 = warp_c0 < nv;                                     // warp-uniform: active entries in [warp_c0, warp_c0 + 16)
    const bool do_j1 = warp_c0 + 16 < nv;                                // warp-uniform: active entries in [warp_c0 + 16, warp_c0 + 32)
    const __half* const a_row = s_w + (r0 + tl) * I8H_LDH;
    const __half* const b_col0 = s_a + (c0 + tl) * I8H_LDH;
    const __half* const b_col1 = b_col0 + 16 * I8H_LDH;
    float acc[2][8];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[j][i] = 0.0f;

    uint32_t wraw0[I8NR<T>], ar[9];
    uint32_t wraw1[BM == 32 ? I8NR<T> : 1];
    i8_load<T>(blob, woff0, 0, wraw0);
    if constexpr (BM == 32) i8_load<T>(blob, woff1, 0, wraw1);
    if (a_active) {
        I8P p = abase + a_vo;
#pragma unroll
        for (int k = 0; k < 9; ++k) ar[k] = i8_ldw(p + 4 * k);
    }
    store_w_one(wr0, 0, wraw0);
    if constexpr (BM == 32) store_w_one(wr0 + 32, 0, wraw1);
    store_a(ar);
    __syncthreads();

    for (int st = 0; st < ns; ++st) {
        if (st + 1 < ns) {   // the next stage's raw bytes into registers while this one computes
            i8_load<T>(blob, woff0, st + 1, wraw0);
            if constexpr (BM == 32) i8_load<T>(blob, woff1, st + 1, wraw1);
            if (a_active) {
                I8P p = abase + (size_t) (st + 1) * 144 + a_vo;
#pragma unroll
                for (int k = 0; k < 9; ++k) ar[k] = i8_ldw(p + 4 * k);
            }
        }
        if (do_j1) {   // common full-32-entry band for this warp (uniform per warp)
#pragma unroll 1
            for (int blk = 0; blk < 4; ++blk) {
                float p0[2][8], p1[2][8];
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        p0[j][i] = 0.0f;
                        if constexpr (i8_two_scales(T)) p1[j][i] = 0.0f;
                    }
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const int k = 32 * blk + 4 * kk;
                    const uint2 a = *(const uint2*) (a_row + k);
                    const uint2 b0 = *(const uint2*) (b_col0 + k), b1 = *(const uint2*) (b_col1 + k);
                    if constexpr (i8_two_scales(T)) {
                        if (kk >= 4) { i8h_mma(p1[0], a, b0); i8h_mma(p1[1], a, b1); }
                        else { i8h_mma(p0[0], a, b0); i8h_mma(p0[1], a, b1); }
                    } else {
                        i8h_mma(p0[0], a, b0);
                        i8h_mma(p0[1], a, b1);
                    }
                }
                float axv[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int cc = 0; cc < 4; ++cc) axv[j][cc] = s_ad[(c0 + 16 * j + ((cc & 2) << 1) + (gg & 2) + (cc & 1)) * 4 + blk];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const int row = r0 + (gg & 1) + 2 * rr + 4 * hh;
                    const uint2 rc = *(const uint2*) (s_r + row * I8H_RS + blk * 2);
                    const float base = __uint_as_float(rc.y);
                    const int m0 = (int) (int16_t) (rc.x & 0xFFFFu);
                    const int m1 = (int) (int16_t) (rc.x >> 16);
#pragma unroll
                    for (int j = 0; j < 2; ++j)
#pragma unroll
                        for (int cc = 0; cc < 4; ++cc) {
                            const int i = (cc & 1) | (rr << 1) | ((cc & 2) << 1);
                            const float ax = axv[j][cc];
                            float t;
                            if constexpr (i8_two_scales(T)) t = p1[j][i] * (float) m1 + p0[j][i] * (float) m0;
                            else if constexpr (i8_unit_scale(T)) t = p0[j][i];
                            else t = p0[j][i] * (float) m0;
                            acc[j][i] = fmaf(t, base * ax, acc[j][i]);
                        }
                }
            }
        } else if (do_j0) {   // partial tile: only the first 16 entries of this warp's 32-entry band are active (uniform per warp)
#pragma unroll 1
            for (int blk = 0; blk < 4; ++blk) {
                float p0[8], p1[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    p0[i] = 0.0f;
                    if constexpr (i8_two_scales(T)) p1[i] = 0.0f;
                }
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const int k = 32 * blk + 4 * kk;
                    const uint2 a = *(const uint2*) (a_row + k);
                    const uint2 b0 = *(const uint2*) (b_col0 + k);
                    if constexpr (i8_two_scales(T)) {
                        if (kk >= 4) i8h_mma(p1, a, b0);
                        else i8h_mma(p0, a, b0);
                    } else {
                        i8h_mma(p0, a, b0);
                    }
                }
                float axv[4];
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) axv[cc] = s_ad[(c0 + ((cc & 2) << 1) + (gg & 2) + (cc & 1)) * 4 + blk];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const int row = r0 + (gg & 1) + 2 * rr + 4 * hh;
                    const uint2 rc = *(const uint2*) (s_r + row * I8H_RS + blk * 2);
                    const float base = __uint_as_float(rc.y);
                    const int m0 = (int) (int16_t) (rc.x & 0xFFFFu);
                    const int m1 = (int) (int16_t) (rc.x >> 16);
#pragma unroll
                    for (int cc = 0; cc < 4; ++cc) {
                        const int i = (cc & 1) | (rr << 1) | ((cc & 2) << 1);
                        const float ax = axv[cc];
                        float t;
                        if constexpr (i8_two_scales(T)) t = p1[i] * (float) m1 + p0[i] * (float) m0;
                        else if constexpr (i8_unit_scale(T)) t = p0[i];
                        else t = p0[i] * (float) m0;
                        acc[0][i] = fmaf(t, base * ax, acc[0][i]);
                    }
                }
            }
        }
        if (st + 1 < ns) {
            __syncthreads();
            store_w_one(wr0, st + 1, wraw0);
            if constexpr (BM == 32) store_w_one(wr0 + 32, st + 1, wraw1);
            store_a(ar);
        }
        __syncthreads();
    }

    // epilogues
    if constexpr (GU) {
        constexpr int F_STRIDE = BM + 1;   // 65 for BM=64 (16.6 KB), 33 for BM=32 (8.4 KB) inside s_w (16.9 KB)
        float* const F = (float*) s_w;
        if (do_j0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = r0 + (gg & 1) + (i & 2) + 4 * hh, col = c0 + (i & 4) + (gg & 2) + (i & 1);
                F[row * F_STRIDE + col] = acc[0][i];
            }
        }
        if (do_j1) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = r0 + (gg & 1) + (i & 2) + 4 * hh, col = c0 + 16 + (i & 4) + (gg & 2) + (i & 1);
                F[row * F_STRIDE + col] = acc[1][i];
            }
        }
        __syncthreads();
        const int nb = (int) (L.n_ff / 32);
        for (int e8 = 0; e8 < 8; ++e8) {   // warp w: entries 8 w .. 8 w + 7 (8 warps x 8 = 64 for BM=64; 4 warps x 8 = 32 for BM=32)
            const int el = 8 * warp + e8;
            if (el >= nv) break;   // uniform per warp
            const float gv = F[lane * F_STRIDE + el], uv = F[(32 + lane) * F_STRIDE + el];
            const float h = (gv / (1.0f + __expf(-gv))) * uv;
            float amax = fabsf(h);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            const float d_inv = 127.0f / amax;
            block_q8_1* hb = hq + (size_t) (e_lo + el) * nb + blockIdx.y;
            hb->qs[lane] = i8_code(h, d_inv, amax);
            if (lane == 0) i8_put_scale(hb, amax, d_inv);
        }
    } else {
        if (do_j0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = r0 + (gg & 1) + (i & 2) + 4 * hh, col = c0 + (i & 4) + (gg & 2) + (i & 1);
                if (col < nv) out[(size_t) ent_idx[e_lo + col] * L.n_embd + (size_t) blockIdx.y * 64 + row] = acc[0][i];
            }
        }
        if (do_j1) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = r0 + (gg & 1) + (i & 2) + 4 * hh, col = c0 + 16 + (i & 4) + (gg & 2) + (i & 1);
                if (col < nv) out[(size_t) ent_idx[e_lo + col] * L.n_embd + (size_t) blockIdx.y * 64 + row] = acc[1][i];
            }
        }
    }
#else
    (void) grp_ptr; (void) grp_start; (void) n_groups; (void) tile_off; (void) ent_idx; (void) A; (void) hq; (void) out; (void) L;
    __trap();
#endif
}
