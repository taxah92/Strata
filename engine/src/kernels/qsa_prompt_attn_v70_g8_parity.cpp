// src/kernels/qsa_prompt_attn_v70_g8_parity.cpp - PR 4.1 Volta (sm_70) grouped 8q/CTA QSA prompt attention parity & bench.
//
// Tests prompt_attn_v70_g8_kernel + qsa_group8_plan_kernel against prompt_attn_v70_kernel (G8=0),
// qsa_decode_attn_batch (FP32), and an FP64 host reference across:
//   - KV formats: k8v4 (mode 3, signed q4_0 scales over 2 decades), int8 (mode 1), fp16 (mode 0)
//   - Modes: G8=1 (H_CTA=3) and G8=2 (H_CTA=4), SPLIT=1 (hi+lo) and SPLIT=0 (1-pass FP16 Q/P)
//   - Selection patterns: Page4-aligned blocks + partial tail, arbitrary random cells, identity short contexts,
//     non-resident page holes (page_table == -1), all-masked queries, zero Q, 100x Q dynamic range within a group,
//     zero overlap (disjoint 8 queries), 100% overlap, non-multiple-of-8 n_q, and multi-sub-batch n_q (>1024).
// Usage: qsa_prompt_attn_v70_g8_parity [--bench] [--quick]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    if (!h.empty()) ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
float h2f(uint16_t b) { __half h = __ushort_as_half(b); return __half2float(h); }
uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

enum SelPattern {
    SEL_PAGE4_REAL = 0,   // Page4-aligned blocks (drift ~10% per query) + recent 512 + tail
    SEL_RANDOM_CELL = 1,  // Arbitrary non-aligned sorted cells (as qsa_prompt_attn_parity)
    SEL_IDENTITY = 2,     // Short context: every cell 0..pos
    SEL_DISJOINT = 3,     // Worst-case 0 overlap across 8 queries
    SEL_SAME = 4,         // 100% overlap across 8 queries
    SEL_HOLES = 5,        // Non-resident pages (-1) including one query with 100% masked cells
};

struct TestCase {
    const char* name;
    int fmt;          // 0 fp16, 1 int8, 3 k8v4
    int64_t ctx;
    int64_t nq;
    SelPattern pat;
    bool zero_q;
    bool dyn_range_q;
};

double q4_val(const std::vector<uint8_t>& pool, int64_t row, int64_t d) {
    constexpr int64_t B4 = 18, ROW4 = 8 * B4;
    const uint8_t* blk = pool.data() + row * ROW4 + (d / 32) * B4;
    uint16_t sc;
    std::memcpy(&sc, blk, 2);
    const int j = (int) (d % 32);
    const int code = j < 16 ? (blk[2 + j] & 0x0F) : (blk[2 + j - 16] >> 4);
    return (double) (code - 8) * (double) h2f(sc);
}

int run_case(const TestCase& tc, int g8_mode, int split_hl) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (tc.ctx + PS - 1) / PS, rows = pages * NKV * PS;
    std::mt19937 rng(4242 + tc.fmt * 97 + (int) tc.nq * 13 + (int) tc.pat * 31);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> lg(std::log(0.01f), std::log(1.0f));

    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    std::vector<uint8_t> vq4;
    if (tc.fmt == 0) {
        kh.resize((size_t) (rows * HD));
        vh.resize((size_t) (rows * HD));
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
    } else {
        kq.resize((size_t) (rows * HD));
        ks.resize((size_t) (rows * 4));
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
        if (tc.fmt == 1) {
            vq.resize((size_t) (rows * HD));
            vs.resize((size_t) (rows * 4));
            for (auto& x : vq) x = (int8_t) code(rng);
            for (auto& x : vs) x = f2h(sc(rng));
        } else {
            // k8v4: signed q4_0 scales spanning two decades (hard stress test!)
            vq4.resize((size_t) (rows * (HD / 32) * 18));
            for (size_t b = 0; b < vq4.size(); b += 18) {
                const float dval = std::exp(lg(rng)) * ((byte_d(rng) & 1) ? -1.0f : 1.0f);
                const uint16_t d = f2h(dval);
                std::memcpy(vq4.data() + b, &d, 2);
                for (int j = 2; j < 18; ++j) vq4[b + j] = (uint8_t) byte_d(rng);
            }
        }
    }

    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    if (tc.pat == SEL_HOLES) {
        // Mark ~15% of pages non-resident (-1), plus pages 0..31 so a query selecting only 0..127 is 100% masked!
        for (int64_t p = 0; p < std::min<int64_t>(32, pages); ++p) table[(size_t) p] = -1;
        for (int64_t p = 32; p < pages; ++p)
            if ((p % 7) == 0) table[(size_t) p] = -1;
    }

    const int64_t cap = k::qsa_selection_width(tc.ctx, s);
    std::vector<int32_t> ids((size_t) (tc.nq * cap), 0);
    std::vector<int32_t> steps((size_t) (tc.nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (tc.nq * NH * HD), 0.0f);
    if (!tc.zero_q) {
        for (int64_t i = 0; i < tc.nq; ++i) {
            const float scale_q = tc.dyn_range_q ? ((i & 7) == 0 ? 100.0f : ((i & 7) == 3 ? 0.05f : 2.0f)) : 2.0f;
            for (int64_t j = 0; j < NH * HD; ++j) q[(size_t) (i * NH * HD + j)] = nd(rng) * scale_q;
        }
    }

    // Build selections according to pattern
    std::vector<int32_t> old_blks;
    std::vector<int32_t> old_cells;
    for (int64_t i = 0; i < tc.nq; ++i) {
        const int64_t pos = tc.ctx - tc.nq + i;
        const int64_t nkv = pos + 1;
        const int64_t w = k::qsa_selection_width(nkv, s);
        int32_t* sel = ids.data() + i * cap;
        steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = (int32_t) w;

        if (tc.pat == SEL_HOLES && i == 2) {
            // Query 2: selects only cells in non-resident pages 0..31 (100% masked!)
            const int64_t w2 = std::min<int64_t>(64, cap);
            steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = (int32_t) w2;
            for (int64_t c = 0; c < w2; ++c) sel[c] = (int32_t) c;
            continue;
        }
        if (tc.pat == SEL_HOLES && i == 5) {
            // Query 5: n == 0!
            steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = 0;
            continue;
        }
        if (tc.pat == SEL_IDENTITY || w == nkv) {
            for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c;
            continue;
        }
        if (tc.pat == SEL_PAGE4_REAL || tc.pat == SEL_HOLES) {
            const int64_t n_full_blks = nkv / PS;
            const int64_t tail = nkv % PS;
            const int64_t rec_blks = std::min<int64_t>(128, n_full_blks);
            const int64_t want_blks = (w - tail) / PS;
            const int64_t older_blks = std::max<int64_t>(0, want_blks - rec_blks);
            const int64_t avail = n_full_blks - rec_blks;
            if ((int64_t) old_blks.size() != older_blks) {
                std::vector<int32_t> all((size_t) avail);
                for (int64_t b = 0; b < avail; ++b) all[(size_t) b] = (int32_t) b;
                std::shuffle(all.begin(), all.end(), rng);
                old_blks.assign(all.begin(), all.begin() + older_blks);
            } else if (avail > older_blks && older_blks > 0) {
                std::uniform_int_distribution<int64_t> pick(0, older_blks - 1), any(0, avail - 1);
                for (int r = 0; r < std::max<int64_t>(1, older_blks / 16); ++r) {
                    const int32_t b = (int32_t) any(rng);
                    if (std::find(old_blks.begin(), old_blks.end(), b) == old_blks.end()) old_blks[(size_t) pick(rng)] = b;
                }
            }
            std::vector<int32_t> blks(old_blks);
            for (int64_t b = n_full_blks - rec_blks; b < n_full_blks; ++b) blks.push_back((int32_t) b);
            std::sort(blks.begin(), blks.end());
            int64_t out_c = 0;
            for (int32_t b : blks)
                for (int64_t j = 0; j < PS; ++j) sel[out_c++] = (int32_t) (b * PS + j);
            for (int64_t j = 0; j < tail; ++j) sel[out_c++] = (int32_t) (n_full_blks * PS + j);
            steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = (int32_t) out_c;
        } else if (tc.pat == SEL_DISJOINT) {
            const int64_t w_d = std::min<int64_t>(w, 512);
            steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = (int32_t) w_d;
            const int64_t base = ((i & 7) * w_d) % std::max<int64_t>(1, nkv - w_d);
            for (int64_t c = 0; c < w_d; ++c) sel[c] = (int32_t) (base + c);
        } else if (tc.pat == SEL_SAME) {
            for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c;
        } else {
            const int64_t recent = 512, older = w - recent;
            if ((int64_t) old_cells.size() != older) {
                std::vector<int32_t> all((size_t) (nkv - recent));
                for (int64_t c = 0; c < nkv - recent; ++c) all[(size_t) c] = (int32_t) c;
                std::shuffle(all.begin(), all.end(), rng);
                old_cells.assign(all.begin(), all.begin() + older);
            } else {
                std::uniform_int_distribution<int64_t> pick(0, older - 1), any(0, nkv - recent - 1);
                for (int r = 0; r < older / 32; ++r) {
                    const int32_t c = (int32_t) any(rng);
                    if (std::find(old_cells.begin(), old_cells.end(), c) == old_cells.end()) old_cells[(size_t) pick(rng)] = c;
                }
            }
            std::vector<int32_t> v(old_cells);
            for (int64_t c = nkv - recent; c < nkv; ++c) v.push_back((int32_t) c);
            std::sort(v.begin(), v.end());
            std::copy(v.begin(), v.end(), sel);
        }
    }

    k::QsaAttnPools pl;
    if (tc.fmt == 0) { pl.k_pool = up(kh); pl.v_pool = up(vh); }
    else if (tc.fmt == 1) { pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); }
    else { pl.k_q = up(kq); pl.k_scale = up(ks); pl.v_q4 = up(vq4); }
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    float *d_ref = nullptr, *d_g8 = nullptr, *d_scratch = nullptr;
    const int64_t ref_batch = 32;
    ck(cudaMalloc(&d_ref, (size_t) (tc.nq * NH * HD) * 4), "malloc ref");
    ck(cudaMalloc(&d_g8, (size_t) (tc.nq * NH * HD) * 4), "malloc g8");
    ck(cudaMalloc(&d_scratch, (size_t) (ref_batch * k::qsa_decode_attn_scratch_floats(cap, s)) * 4), "malloc scratch");
    ck(cudaMemset(d_ref, 0x5a, (size_t) (tc.nq * NH * HD) * 4), "memset ref");
    ck(cudaMemset(d_g8, 0xa5, (size_t) (tc.nq * NH * HD) * 4), "memset g8");

    for (int64_t t0 = 0; t0 < tc.nq; t0 += ref_batch) {
        k::qsa_decode_attn_batch(d_q + t0 * NH * HD, pl, d_ids + t0 * cap, d_steps + t0 * k::kStepCount, cap, s,
                                 d_scratch, d_ref + t0 * NH * HD, std::min(ref_batch, tc.nq - t0), nullptr);
    }
    k::qsa_prompt_attn_set_v70_g8(g8_mode, split_hl);
    if (!k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_g8, tc.nq, nullptr)) {
        std::fprintf(stderr, "FAIL %s: g8 launch refused\n", tc.name);
        return 1;
    }
    k::qsa_prompt_attn_set_v70_g8(-1, -1);
    ck(cudaDeviceSynchronize(), "sync");
    cudaFree(d_scratch);

    std::vector<float> h_ref((size_t) (tc.nq * NH * HD)), h_g8(h_ref.size());
    ck(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * 4, cudaMemcpyDeviceToHost), "down ref");
    ck(cudaMemcpy(h_g8.data(), d_g8, h_g8.size() * 4, cudaMemcpyDeviceToHost), "down g8");

    double max_diff = 0.0, scale = 0.0, l2_num = 0.0, l2_den = 0.0;
    int nans = 0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
        if (!std::isfinite(h_g8[i])) ++nans;
        const double d = std::fabs((double) h_ref[i] - (double) h_g8[i]);
        max_diff = std::max(max_diff, d);
        scale = std::max(scale, (double) std::fabs(h_ref[i]));
        l2_num += d * d;
        l2_den += (double) h_ref[i] * (double) h_ref[i];
    }
    const double rel_l2 = l2_den > 0.0 ? std::sqrt(l2_num / l2_den) : 0.0;

    // FP64 check on sampled queries
    double err_fp64 = 0.0, err_ref_fp64 = 0.0, fp64_scale = 0.0;
    const int64_t step_i = std::max<int64_t>(1, tc.nq / 8);
    for (int64_t i = 0; i < tc.nq; i += step_i) {
        const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
        const int32_t* sel = ids.data() + i * cap;
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / (NH / NKV);
            std::vector<double> sco((size_t) w, -1e300);
            std::vector<int64_t> prow((size_t) w, -1);
            double mx = -1e300;
            bool any_valid = false;
            for (int64_t c = 0; c < w; ++c) {
                const int64_t cell = sel[c];
                const int64_t pg = table[(size_t) (cell / PS)];
                if (pg < 0) continue;
                any_valid = true;
                const int64_t row = (pg * NKV + kvh) * PS + (cell % PS);
                prow[(size_t) c] = row;
                double a = 0.0;
                for (int64_t d = 0; d < HD; ++d) {
                    const double kv = tc.fmt == 0 ? (double) h2f(kh[(size_t) (row * HD + d)])
                                                  : (double) kq[(size_t) (row * HD + d)] * (double) h2f(ks[(size_t) (row * 4 + d / 64)]);
                    a += (double) q[(size_t) ((i * NH + h) * HD + d)] * kv;
                }
                sco[(size_t) c] = a / 16.0;
                mx = std::max(mx, sco[(size_t) c]);
            }
            double l = 0.0;
            if (any_valid) {
                for (int64_t c = 0; c < w; ++c) {
                    if (prow[(size_t) c] < 0) { sco[(size_t) c] = 0.0; continue; }
                    sco[(size_t) c] = std::exp(sco[(size_t) c] - mx);
                    l += sco[(size_t) c];
                }
            }
            for (int64_t d = 0; d < HD; ++d) {
                double r = 0.0;
                if (any_valid && l > 0.0) {
                    double a = 0.0;
                    for (int64_t c = 0; c < w; ++c) {
                        const int64_t row = prow[(size_t) c];
                        if (row < 0) continue;
                        const double vv = tc.fmt == 0 ? (double) h2f(vh[(size_t) (row * HD + d)])
                                        : tc.fmt == 1 ? (double) vq[(size_t) (row * HD + d)] * (double) h2f(vs[(size_t) (row * 4 + d / 64)])
                                                      : q4_val(vq4, row, d);
                        a += sco[(size_t) c] * vv;
                    }
                    r = a / l;
                }
                const size_t at = (size_t) ((i * NH + h) * HD + d);
                fp64_scale = std::max(fp64_scale, std::fabs(r));
                err_fp64 = std::max(err_fp64, std::fabs((double) h_g8[at] - r));
                err_ref_fp64 = std::max(err_ref_fp64, std::fabs((double) h_ref[at] - r));
            }
        }
    }

    // Thresholds: SPLIT=1 must match FP64 to FP32 level (<= max(4*err_ref, 2e-5*scale) and rel_l2 <= 2e-5);
    // SPLIT=0 (1-pass FP16) allows FP16 rounding (<= 5e-3*scale, rel_l2 <= 2e-3).
    const double eff_scale = std::max(scale, 1e-6);
    const double fp64_eff = std::max(fp64_scale, 1e-6);
    const bool ok = (nans == 0) &&
                    (split_hl
                         ? (err_fp64 <= std::max(4.0 * err_ref_fp64, 2e-5 * fp64_eff) &&
                            max_diff <= std::max(1e-4 * eff_scale, 1e-6))
                         : (rel_l2 <= 5e-3));

    std::printf("%s %-24s [G8=%d SPLIT=%d fmt=%d ctx=%lld nq=%lld]: vs_fp64=%.3g (fp32=%.3g) vs_fp32=%.3g rel_l2=%.3g nans=%d\n",
                ok ? "PASS" : "FAIL", tc.name, g8_mode, split_hl, tc.fmt, (long long) tc.ctx, (long long) tc.nq,
                err_fp64, err_ref_fp64, max_diff, rel_l2, nans);

    cudaFree((void*) d_ids); cudaFree((void*) d_steps); cudaFree((void*) d_q);
    cudaFree(d_ref); cudaFree(d_g8);
    cudaFree((void*) pl.k_q); cudaFree((void*) pl.v_q); cudaFree((void*) pl.k_scale); cudaFree((void*) pl.v_scale);
    cudaFree((void*) pl.v_q4); cudaFree((void*) pl.k_pool); cudaFree((void*) pl.v_pool); cudaFree((void*) pl.page_table);
    return ok ? 0 : 1;
}

float time_ms(const std::function<void()>& f, int reps) {
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "e0");
    ck(cudaEventCreate(&e1), "e1");
    f();
    ck(cudaDeviceSynchronize(), "warm");
    ck(cudaEventRecord(e0), "r0");
    for (int r = 0; r < reps; ++r) f();
    ck(cudaEventRecord(e1), "r1");
    ck(cudaEventSynchronize(e1), "sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / reps;
}

void run_bench(int fmt, int64_t ctx, int64_t nq, int drift_div, int reps) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    std::mt19937 rng(777 + fmt * 19 + drift_div);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);

    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    std::vector<uint8_t> vq4;
    if (fmt == 0) {
        kh.resize((size_t) (rows * HD)); vh.resize((size_t) (rows * HD));
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
    } else {
        kq.resize((size_t) (rows * HD)); ks.resize((size_t) (rows * 4));
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
        if (fmt == 1) {
            vq.resize((size_t) (rows * HD)); vs.resize((size_t) (rows * 4));
            for (auto& x : vq) x = (int8_t) code(rng);
            for (auto& x : vs) x = f2h(sc(rng));
        } else {
            vq4.resize((size_t) (rows * (HD / 32) * 18));
            for (size_t i = 0; i < vq4.size(); i += 18) {
                const uint16_t d = f2h(sc(rng));
                std::memcpy(vq4.data() + i, &d, 2);
                for (int j = 2; j < 18; ++j) vq4[i + j] = (uint8_t) rng();
            }
        }
    }
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    const int64_t cap = k::qsa_selection_width(ctx, s);
    std::vector<int32_t> ids((size_t) (nq * cap), 0), steps((size_t) (nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (nq * NH * HD));
    for (auto& x : q) x = nd(rng) * 2.0f;

    // Build Page4-aligned realistic drift selections
    std::vector<int32_t> old_blks;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t pos = ctx - nq + i, nkv = pos + 1;
        const int64_t w = k::qsa_selection_width(nkv, s);
        int32_t* sel = ids.data() + i * cap;
        const int64_t n_full_blks = nkv / PS, tail = nkv % PS;
        const int64_t rec_blks = std::min<int64_t>(128, n_full_blks);
        const int64_t want_blks = (w - tail) / PS;
        const int64_t older_blks = std::max<int64_t>(0, want_blks - rec_blks);
        const int64_t avail = n_full_blks - rec_blks;
        if ((int64_t) old_blks.size() != older_blks) {
            std::vector<int32_t> all((size_t) avail);
            for (int64_t b = 0; b < avail; ++b) all[(size_t) b] = (int32_t) b;
            std::shuffle(all.begin(), all.end(), rng);
            old_blks.assign(all.begin(), all.begin() + older_blks);
        } else if (avail > older_blks && drift_div > 0) {
            std::uniform_int_distribution<int64_t> pick(0, older_blks - 1), any(0, avail - 1);
            for (int r = 0; r < std::max<int64_t>(1, older_blks / drift_div); ++r) {
                const int32_t b = (int32_t) any(rng);
                if (std::find(old_blks.begin(), old_blks.end(), b) == old_blks.end()) old_blks[(size_t) pick(rng)] = b;
            }
        }
        std::vector<int32_t> blks(old_blks);
        for (int64_t b = n_full_blks - rec_blks; b < n_full_blks; ++b) blks.push_back((int32_t) b);
        std::sort(blks.begin(), blks.end());
        int64_t out_c = 0;
        for (int32_t b : blks)
            for (int64_t j = 0; j < PS; ++j) sel[out_c++] = (int32_t) (b * PS + j);
        for (int64_t j = 0; j < tail; ++j) sel[out_c++] = (int32_t) (n_full_blks * PS + j);
        steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = (int32_t) out_c;
    }

    // Measure actual average 8-query union ratio
    double sum_u = 0.0, sum_n = 0.0;
    int64_t ng = 0;
    for (int64_t g0 = 0; g0 + 8 <= nq; g0 += 8) {
        std::vector<int32_t> u;
        for (int r = 0; r < 8; ++r) {
            const int w = steps[(size_t) ((g0 + r) * k::kStepCount + k::kStepWidth)];
            u.insert(u.end(), ids.begin() + (size_t) ((g0 + r) * cap), ids.begin() + (size_t) ((g0 + r) * cap + w));
            sum_n += w;
        }
        std::sort(u.begin(), u.end());
        sum_u += (double) (std::unique(u.begin(), u.end()) - u.begin());
        ++ng;
    }
    const double union_x = ng > 0 ? (sum_u / (sum_n / 8.0)) : 1.0;

    k::QsaAttnPools pl;
    if (fmt == 0) { pl.k_pool = up(kh); pl.v_pool = up(vh); }
    else if (fmt == 1) { pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); }
    else { pl.k_q = up(kq); pl.k_scale = up(ks); pl.v_q4 = up(vq4); }
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    float* d_out = nullptr;
    ck(cudaMalloc(&d_out, (size_t) (nq * NH * HD) * 4), "malloc out");

    auto run_m = [&](int g8, int sp) {
        k::qsa_prompt_attn_set_v70_g8(g8, sp);
        return time_ms([&]() { k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_out, nq, nullptr); }, reps);
    };
    const float ms_base = run_m(0, 1);
    const float ms_v2_s1 = run_m(3, 1);
    const float ms_v2_s0 = run_m(3, 0);
    const float ms_g8_s1 = run_m(1, 1);
    const float ms_g8_s0 = run_m(1, 0);
    k::qsa_prompt_attn_set_v70_g8(-1, -1);

    std::printf("BENCH %-4s ctx=%6lld nq=%4lld union=%.2fx | base=%7.3f ms | V2/HL=%6.3f (%4.2fx) V2/1P=%6.3f (%4.2fx) | G8/HL=%6.3f (%4.2fx) G8/1P=%6.3f (%4.2fx)\n",
                fmt == 3 ? "k8v4" : (fmt == 1 ? "int8" : "fp16"), (long long) ctx, (long long) nq, union_x,
                ms_base, ms_v2_s1, ms_base / ms_v2_s1, ms_v2_s0, ms_base / ms_v2_s0,
                ms_g8_s1, ms_base / ms_g8_s1, ms_g8_s0, ms_base / ms_g8_s0);

    cudaFree((void*) d_ids); cudaFree((void*) d_steps); cudaFree((void*) d_q); cudaFree(d_out);
    cudaFree((void*) pl.k_q); cudaFree((void*) pl.v_q); cudaFree((void*) pl.k_scale); cudaFree((void*) pl.v_scale);
    cudaFree((void*) pl.v_q4); cudaFree((void*) pl.k_pool); cudaFree((void*) pl.v_pool); cudaFree((void*) pl.page_table);
}
}  // namespace

int main(int argc, char** argv) {
    int dev = 0, maj = 0, min = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        std::fprintf(stderr, "No CUDA device\n");
        return 77;
    }
    if (maj != 7 || min >= 5) {
        std::printf("SKIP: sm_%d%d is not Volta (sm_70)\n", maj, min);
        return 77;
    }
    bool do_bench = false, quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    }

    const std::vector<TestCase> cases = quick ? std::vector<TestCase>{
        {"k8v4_page4_33",     3, 8192,  33, SEL_PAGE4_REAL,  false, false},
        {"k8v4_holes_17",     3, 4096,  17, SEL_HOLES,       false, true},
        {"int8_rand_25",      1, 8192,  25, SEL_RANDOM_CELL, false, false},
        {"fp16_id_19",        0, 256,   19, SEL_IDENTITY,    false, false},
    } : std::vector<TestCase>{
        {"k8v4_page4_256",    3, 32768, 256,  SEL_PAGE4_REAL,  false, false},
        {"k8v4_rand_65",      3, 32768, 65,   SEL_RANDOM_CELL, false, true},
        {"k8v4_tail_7",       3, 8192,  7,    SEL_PAGE4_REAL,  false, false},
        {"k8v4_id_33",        3, 1500,  33,   SEL_IDENTITY,    false, false},
        {"k8v4_holes_17",     3, 8192,  17,   SEL_HOLES,       false, true},
        {"k8v4_disjoint_16",  3, 16384, 16,   SEL_DISJOINT,    false, false},
        {"k8v4_same_16",      3, 8192,  16,   SEL_SAME,        false, false},
        {"k8v4_zeroq_9",      3, 4096,  9,    SEL_PAGE4_REAL,  true,  false},
        {"k8v4_subbatch_1035",3, 8192,  1035, SEL_PAGE4_REAL,  false, false},
        {"int8_page4_256",    1, 32768, 256,  SEL_PAGE4_REAL,  false, false},
        {"int8_rand_65",      1, 32768, 65,   SEL_RANDOM_CELL, false, true},
        {"int8_holes_17",     1, 8192,  17,   SEL_HOLES,       false, false},
        {"fp16_page4_128",    0, 16384, 128,  SEL_PAGE4_REAL,  false, false},
        {"fp16_holes_17",     0, 4096,  17,   SEL_HOLES,       false, true},
    };

    int fails = 0;
    for (const auto& tc : cases) {
        fails += run_case(tc, 3, 1);
        fails += run_case(tc, 3, 0);
        fails += run_case(tc, 1, 1);
        fails += run_case(tc, 1, 0);
        if (!quick) {
            fails += run_case(tc, 2, 1);
            fails += run_case(tc, 2, 0);
        }
    }
    std::printf("PARITY FAILURES: %d\n", fails);

    if (do_bench && fails == 0) {
        std::printf("\n--- V100 Grouped QSA Prefill Microbench ---\n");
        for (int fmt : {3, 1, 0}) {
            run_bench(fmt, 32768, 2048, 32, 8);   // ~1.18x union (high overlap)
            run_bench(fmt, 32768, 2048, 8,  8);   // ~1.7x union (typical prompt)
            run_bench(fmt, 32768, 2048, 3,  8);   // ~2.6x union (low overlap)
            run_bench(fmt, 131072, 2048, 8, 5);   // 131K context
        }
    }
    return fails ? 1 : 0;
}
