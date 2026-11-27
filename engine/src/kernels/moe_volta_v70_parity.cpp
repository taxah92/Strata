// src/kernels/moe_volta_v70_parity.cpp - Parity, edge-case routes, 1000-replay CUDA Graph stress,
// and realistic 48-layer V100 microbenchmark for PR 5.3 (Steps 5.3a & 5.3b).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq4nl_volta_tc.hpp"
#include "strata/kernels/moe_volta_v70.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

constexpr int kEmbd = 2560;
constexpr int kFf = 640;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_volta_v70_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

int qk_of(int ty) { return (ty == 20) ? 32 : (ty == 42) ? 64 : 256; }

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

void fill_blob(uint8_t* b, const K::NativeExpertLayout& L, std::mt19937& rng) {
    for (size_t i = 0; i + 4 <= L.bytes; i += 4) {
        const uint32_t r = rng();
        std::memcpy(b + i, &r, 4);
    }
    const size_t gu_block = K::iq_row_bytes(L.gu_type, qk_of(L.gu_type));
    const size_t d_block = K::iq_row_bytes(L.d_type, qk_of(L.d_type));
    const size_t gu_blocks = (size_t) 2 * (size_t) L.n_ff * (size_t) (L.n_embd / qk_of(L.gu_type));
    for (size_t k = 0; k < gu_blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(b + k * gu_block, &h, 2);
    }
    const size_t d_blocks = (size_t) L.n_embd * (size_t) (L.n_ff / qk_of(L.d_type));
    for (size_t k = 0; k < d_blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(b + L.down_off + k * d_block, &h, 2);
    }
}

struct Route {
    int G = 0, NE = 0, T = 0;
    std::vector<int> exp_id;
    std::vector<int32_t> start, tok, dst;
};

Route make_skewed_route(int G, int NE, int T, int pool_n, std::mt19937& rng) {
    Route r;
    r.G = G; r.NE = NE; r.T = T;
    r.start.push_back(0);
    std::vector<int> counts((size_t) G, 0);
    if (G > 0 && NE > 0) {
        for (int g = 0; g < G && g < NE; ++g) counts[(size_t) g] = 1;
        int rem = NE - std::min(G, NE);
        // Zipf-like skew: first few experts get multiple entries (up to 12)
        for (int i = 0; i < rem; ++i) {
            int g = (i % 3 == 0) ? (int) (rng() % (unsigned) std::min(G, 6))
                                 : (int) (rng() % (unsigned) G);
            counts[(size_t) g]++;
        }
    }
    std::vector<int> perm((size_t) pool_n);
    for (int i = 0; i < pool_n; ++i) perm[(size_t) i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> dsts((size_t) NE);
    for (int i = 0; i < NE; ++i) dsts[(size_t) i] = i;
    std::shuffle(dsts.begin(), dsts.end(), rng);
    int cur = 0;
    for (int g = 0; g < G; ++g) {
        r.exp_id.push_back(perm[(size_t) (g % pool_n)]);
        for (int c = 0; c < counts[(size_t) g]; ++c) {
            r.tok.push_back(T > 0 ? (int32_t) ((cur + c) % T) : 0);
            r.dst.push_back(dsts[(size_t) cur]);
            cur++;
        }
        r.start.push_back(cur);
    }
    return r;
}

Route make_single_expert_route(int NE, int T) {
    Route r;
    r.G = 1; r.NE = NE; r.T = T;
    r.exp_id = {0};
    r.start = {0, NE};
    for (int i = 0; i < NE; ++i) {
        r.tok.push_back(i % T);
        r.dst.push_back(NE - 1 - i);   // reversed dst to check ent_dst mapping
    }
    return r;
}

Route make_all_distinct_route(int G, int T, int pool_n) {
    Route r;
    r.G = G; r.NE = G; r.T = T;
    r.start.push_back(0);
    for (int g = 0; g < G; ++g) {
        r.exp_id.push_back(g % pool_n);
        r.tok.push_back(g % T);
        r.dst.push_back(g);
        r.start.push_back(g + 1);
    }
    return r;
}

Route make_empty_route(int T) {
    Route r;
    r.G = 0; r.NE = 0; r.T = T;
    r.start = {0};
    return r;
}

struct DiffStats {
    int bit_diffs = 0;
    double max_abs = 0.0;
    double rel_l2 = 0.0;
    int non_finite = 0;
};

DiffStats compare_vec(const std::vector<float>& a, const std::vector<float>& b) {
    DiffStats s;
    double d2 = 0.0, r2 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { s.non_finite++; continue; }
        uint32_t ua, ub;
        std::memcpy(&ua, &a[i], 4);
        std::memcpy(&ub, &b[i], 4);
        if (ua != ub) s.bit_diffs++;
        const double diff = std::fabs((double) a[i] - (double) b[i]);
        if (diff > s.max_abs) s.max_abs = diff;
        d2 += diff * diff;
        r2 += (double) b[i] * (double) b[i];
    }
    s.rel_l2 = (r2 > 0.0) ? std::sqrt(d2 / r2) : std::sqrt(d2);
    return s;
}

struct TestCtx {
    int gu_type = 22, d_type = 20;
    K::NativeExpertLayout L{};
    size_t stride = 0;
    int pool_n = 0;
    uint8_t* d_pool = nullptr;
    unsigned long long* d_ptr = nullptr;
    int32_t *d_start = nullptr, *d_n = nullptr, *d_dst = nullptr, *d_tok = nullptr;
    void *d_xq = nullptr, *d_scr = nullptr;
    float *d_xf = nullptr, *d_out = nullptr;
    cudaStream_t stream = nullptr;

    void init(int gu, int d, int n_blobs, int max_cap, int max_T, uint32_t seed) {
        gu_type = gu; d_type = d; pool_n = n_blobs;
        L = K::native_expert_layout(gu, d, kEmbd, kFf);
        stride = (L.bytes + 255) & ~(size_t) 255;
        std::mt19937 rng(seed);
        std::vector<uint8_t> host_pool((size_t) n_blobs * stride, 0);
        for (int i = 0; i < n_blobs; ++i) fill_blob(host_pool.data() + (size_t) i * stride, L, rng);
        ck(cudaMalloc(&d_pool, host_pool.size()), "malloc pool");
        ck(cudaMemcpy(d_pool, host_pool.data(), host_pool.size(), cudaMemcpyHostToDevice), "copy pool");

        ck(cudaMalloc(&d_ptr, (size_t) max_cap * sizeof(unsigned long long)), "malloc ptr");
        ck(cudaMalloc(&d_start, (size_t) (max_cap + 1) * sizeof(int32_t)), "malloc start");
        ck(cudaMalloc(&d_n, sizeof(int32_t)), "malloc n");
        ck(cudaMalloc(&d_dst, (size_t) max_cap * sizeof(int32_t)), "malloc dst");
        ck(cudaMalloc(&d_tok, (size_t) max_cap * sizeof(int32_t)), "malloc tok");
        ck(cudaMalloc(&d_xf, (size_t) max_T * kEmbd * sizeof(float)), "malloc xf");
        ck(cudaMalloc(&d_xq, (size_t) max_T * (kEmbd / 32) * 36), "malloc xq");
        ck(cudaMalloc(&d_scr, K::native_expert_scratch_bytes(max_cap, kFf)), "malloc scr");
        ck(cudaMalloc(&d_out, (size_t) max_cap * kEmbd * sizeof(float)), "malloc out");
        ck(cudaStreamCreate(&stream), "streamCreate");
    }

    void upload_x(const std::vector<float>& xh) {
        const int T = (int) (xh.size() / kEmbd);
        ck(cudaMemcpyAsync(d_xf, xh.data(), xh.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy x");
        for (int off = 0; off < T; off += 8) {
            const int cnt = std::min(8, T - off);
            K::native_quantize_q8_1(d_xf + (size_t) off * kEmbd,
                                    (uint8_t*) d_xq + (size_t) off * (kEmbd / 32) * 36,
                                    kEmbd, cnt, stream);
        }
    }

    void upload_route(const Route& r) {
        std::vector<unsigned long long> ptrs((size_t) r.G);
        for (int g = 0; g < r.G; ++g) {
            ptrs[(size_t) g] = (unsigned long long) (d_pool + (size_t) r.exp_id[(size_t) g] * stride);
        }
        if (r.G > 0) ck(cudaMemcpyAsync(d_ptr, ptrs.data(), (size_t) r.G * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream), "ptr");
        ck(cudaMemcpyAsync(d_start, r.start.data(), r.start.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "start");
        ck(cudaMemcpyAsync(d_n, &r.G, sizeof(int32_t), cudaMemcpyHostToDevice, stream), "n");
        if (r.NE > 0) {
            ck(cudaMemcpyAsync(d_dst, r.dst.data(), (size_t) r.NE * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "dst");
            ck(cudaMemcpyAsync(d_tok, r.tok.data(), (size_t) r.NE * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "tok");
        }
    }

    std::vector<float> run_mode(int cap, int exp_mode, int bank_m, int persist_m, int iq4_m) {
        K::native_expert_set_mode(exp_mode, 0);
        K::moe_volta_bank_set_mode(bank_m);
        K::moe_volta_persist_set_mode(persist_m);
        K::iq4nl_volta_tc_set_mode(iq4_m);
        const size_t n_floats = (size_t) cap * kEmbd;
        ck(cudaMemsetAsync(d_out, 0, n_floats * sizeof(float), stream), "zero out");
        K::native_expert_grouped(L, d_ptr, d_start, d_n, d_dst, d_tok, cap, cap, d_xq, d_scr, d_out, stream, 0);
        std::vector<float> h(n_floats);
        ck(cudaMemcpyAsync(h.data(), d_out, n_floats * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy out");
        ck(cudaStreamSynchronize(stream), "sync");
        return h;
    }

    void destroy() {
        if (stream) cudaStreamDestroy(stream);
        cudaFree(d_pool); cudaFree(d_ptr); cudaFree(d_start); cudaFree(d_n);
        cudaFree(d_dst); cudaFree(d_tok); cudaFree(d_xf); cudaFree(d_xq);
        cudaFree(d_scr); cudaFree(d_out);
    }
};

int check_pair_parity(int gu, int d, const char* label, int n_replays, bool race_only) {
    int fails = 0;
    TestCtx ctx;
    const int pool_n = 64, max_cap = 140, max_T = 16;
    ctx.init(gu, d, pool_n, max_cap, max_T, 1337u + (unsigned) (gu * 31 + d));
    std::mt19937 rng(4242u + (unsigned) (gu * 17 + d));
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> x_rand((size_t) max_T * kEmbd);
    for (float& v : x_rand) v = nd(rng);
    std::vector<float> x_zero((size_t) max_T * kEmbd, 0.0f);

    struct Case { const char* name; Route r; int cap; bool zero_x; };
    std::vector<Case> cases;
    cases.push_back({"empty_ng0", make_empty_route(4), 40, false});
    cases.push_back({"single_exp_ne1", make_single_expert_route(1, 1), 10, false});
    cases.push_back({"single_exp_ne4", make_single_expert_route(4, 4), 40, false});
    cases.push_back({"single_exp_ne5_tc", make_single_expert_route(5, 5), 50, false});
    cases.push_back({"single_exp_ne12_tc", make_single_expert_route(12, 12), 120, false});
    cases.push_back({"single_exp_ne16_tc", make_single_expert_route(16, 16), 140, false});
    cases.push_back({"distinct_50", make_all_distinct_route(50, 5, pool_n), 50, false});
    cases.push_back({"distinct_120", make_all_distinct_route(120, 12, pool_n), 120, false});
    cases.push_back({"mtp_t5_skew", make_skewed_route(36, 50, 5, pool_n, rng), 50, false});
    cases.push_back({"df2_t7_skew", make_skewed_route(42, 70, 7, pool_n, rng), 70, false});
    cases.push_back({"df2_t12_skew", make_skewed_route(56, 120, 12, pool_n, rng), 120, false});
    cases.push_back({"zero_act_36", make_skewed_route(36, 50, 5, pool_n, rng), 50, true});

    if (race_only) {
        std::vector<Case> race_cases;
        for (const auto& c : cases) {
            const std::string nm(c.name);
            if (nm == "ng_zero" || nm == "single_exp_ne1" || nm == "single_exp_ne4_dp4a" ||
                nm == "single_exp_ne6_tc" || nm == "skew_1_1_1_1_12") {
                race_cases.push_back(c);
            }
        }
        cases.swap(race_cases);
    }
    for (const auto& c : cases) {
        ctx.upload_x(c.zero_x ? x_zero : x_rand);
        ctx.upload_route(c.r);
        auto ref_pr52 = ctx.run_mode(c.cap, 8, 0, 0, 1);    // PR 5.2 baseline (BANK=0, PERSIST=0, IQ4=1)
        auto out_53a  = ctx.run_mode(c.cap, 8, 1, 0, 1);    // Step 5.3a (BANK=1, PERSIST=0, IQ4=1)
        auto out_53b  = ctx.run_mode(c.cap, 8, 1, 1, 1);    // Step 5.3b (BANK=1, PERSIST=1, IQ4=1)

        auto da = compare_vec(out_53a, ref_pr52);
        auto db = compare_vec(out_53b, ref_pr52);
        const bool ok_a = (da.bit_diffs == 0 && da.non_finite == 0);
        const bool ok_b = (db.bit_diffs == 0 && db.non_finite == 0);
        if (!ok_a || !ok_b) {
            std::printf("  FAIL [%s] %-18s: 5.3a bit_diffs=%d rel=%.3e | 5.3b bit_diffs=%d rel=%.3e\n",
                        label, c.name, da.bit_diffs, da.rel_l2, db.bit_diffs, db.rel_l2);
            fails++;
        }
    }

    // 1000-replay CUDA Graph stress test with changing routes, ng (0..70), and inputs!
    const int graph_cap = 120;
    std::vector<Route> groutes = {
        make_empty_route(12),
        make_single_expert_route(12, 12),
        make_single_expert_route(4, 4),
        make_all_distinct_route(50, 5, pool_n),
        make_skewed_route(36, 50, 5, pool_n, rng),
        make_skewed_route(42, 70, 7, pool_n, rng),
        make_skewed_route(56, 120, 12, pool_n, rng),
    };
    ctx.upload_x(x_rand);
    ctx.upload_route(groutes[4]);
    ck(cudaStreamSynchronize(ctx.stream), "pre-capture sync");

    // Pre-compute reference outputs for every route
    std::vector<std::vector<float>> grefs;
    for (const auto& gr : groutes) {
        ctx.upload_route(gr);
        grefs.push_back(ctx.run_mode(graph_cap, 8, 0, 0, 1));
    }

    // Capture Step 5.3b persistent kernel into a CUDA Graph
    K::native_expert_set_mode(8, 0);
    K::moe_volta_bank_set_mode(1);
    K::moe_volta_persist_set_mode(1);
    K::iq4nl_volta_tc_set_mode(1);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(ctx.stream, cudaStreamCaptureModeGlobal), "begin capture");
    K::native_expert_grouped(ctx.L, ctx.d_ptr, ctx.d_start, ctx.d_n, ctx.d_dst, ctx.d_tok,
                             graph_cap, graph_cap, ctx.d_xq, ctx.d_scr, ctx.d_out, ctx.stream, 0);
    ck(cudaStreamEndCapture(ctx.stream, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "graph instantiate");

    int graph_bad = 0;
    for (int rep = 0; rep < n_replays; ++rep) {
        const size_t ri = (size_t) (rep % (int) groutes.size());
        ctx.upload_route(groutes[ri]);
        if (rep < 14 || rep >= n_replays - 7) {
            ck(cudaMemsetAsync(ctx.d_out, 0, (size_t) graph_cap * kEmbd * sizeof(float), ctx.stream), "zero");
        }
        ck(cudaGraphLaunch(exec, ctx.stream), "graph launch");
        if (rep < 14 || rep >= n_replays - 7) {
            std::vector<float> got((size_t) graph_cap * kEmbd);
            ck(cudaMemcpyAsync(got.data(), ctx.d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream), "copy");
            ck(cudaStreamSynchronize(ctx.stream), "sync");
            auto dg = compare_vec(got, grefs[ri]);
            if (dg.bit_diffs != 0 || dg.non_finite != 0) {
                std::printf("  FAIL [%s] graph replay %d (route %zu): bit_diffs=%d rel=%.3e\n",
                            label, rep, ri, dg.bit_diffs, dg.rel_l2);
                graph_bad++;
                break;
            }
        }
    }
    ck(cudaStreamSynchronize(ctx.stream), "final graph sync");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    if (graph_bad) fails++;

    std::fflush(stdout); std::printf("  parity [%s]: 12 edge/skew cases + %d CUDA Graph replays -> %s\n",
                label, n_replays, fails == 0 ? "BITWISE OK" : "FAILED");

    K::native_expert_set_mode(-1, 0);
    K::moe_volta_bank_set_mode(-1);
    K::moe_volta_persist_set_mode(-1);
    K::iq4nl_volta_tc_set_mode(-1);
    ctx.destroy();
    return fails;
}

void run_bench() {
    std::printf("\n# PR 5.3 V100 MoE decode microbenchmark (cold pool ~220 MB, 2560x640, 50 reps x 5 rounds)\n");
    struct PairInfo { int gu, d; const char* name; int weight_layers; };
    const PairInfo pairs[3] = {
        {22, 20, "IQ2_S/IQ4_NL", 25},
        {18, 20, "IQ3_XXS/IQ4_NL", 20},
        {22, 42, "IQ2_S/Q2_0", 3},
    };
    struct Scenario { const char* name; int G, NE, T, cap; };
    const Scenario scenarios[] = {
        {"T=1 (10e/10)",     10,  10,  1,  10},
        {"T=2 (20e/20)",     20,  20,  2,  20},
        {"T=4 (36e/40)",     36,  40,  4,  40},
        {"MTP_T5 (36e/50)",  36,  50,  5,  50},
        {"DF2_T7 (42e/70)",  42,  70,  7,  70},
        {"DF2_T12 (56e/120)",56, 120, 12, 120},
    };

    double tot48[6][3] = {};   // [scenario][0=base, 1=5.3a, 2=5.3b]

    for (const auto& pi : pairs) {
        TestCtx ctx;
        const int pool_n = 128;
        ctx.init(pi.gu, pi.d, pool_n, 140, 16, 9001u + (unsigned) (pi.gu * 13 + pi.d));
        std::mt19937 rng(2026u + (unsigned) (pi.gu * 7 + pi.d));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xh((size_t) 16 * kEmbd);
        for (float& v : xh) v = nd(rng);
        ctx.upload_x(xh);

        std::printf("\n--- %s (%d/48 layers, blob %.2f MB) ---\n",
                    pi.name, pi.weight_layers, (double) ctx.L.bytes / 1e6);
        std::printf("%-18s | %8s %8s %8s | %7s %7s | %7s\n",
                    "scenario", "PR5.2_us", "5.3a_us", "5.3b_us", "5.3a/b", "5.3b/b", "BW(GB/s)");

        cudaEvent_t ev0, ev1;
        ck(cudaEventCreate(&ev0), "ev0");
        ck(cudaEventCreate(&ev1), "ev1");

        for (int si = 0; si < 6; ++si) {
            const auto& sc = scenarios[si];
            Route r = make_skewed_route(sc.G, sc.NE, sc.T, pool_n, rng);
            ctx.upload_route(r);
            ck(cudaStreamSynchronize(ctx.stream), "sync");

            auto measure = [&](int bank_m, int persist_m) -> double {
                K::native_expert_set_mode(8, 0);
                K::moe_volta_bank_set_mode(bank_m);
                K::moe_volta_persist_set_mode(persist_m);
                K::iq4nl_volta_tc_set_mode(1);
                // Warmup
                for (int w = 0; w < 5; ++w) {
                    K::native_expert_grouped(ctx.L, ctx.d_ptr, ctx.d_start, ctx.d_n, ctx.d_dst, ctx.d_tok,
                                             sc.cap, sc.cap, ctx.d_xq, ctx.d_scr, ctx.d_out, ctx.stream, 0);
                }
                ck(cudaStreamSynchronize(ctx.stream), "warm");
                std::vector<double> rds;
                for (int rd = 0; rd < 5; ++rd) {
                    ck(cudaEventRecord(ev0, ctx.stream), "rec0");
                    for (int rep = 0; rep < 40; ++rep) {
                        K::native_expert_grouped(ctx.L, ctx.d_ptr, ctx.d_start, ctx.d_n, ctx.d_dst, ctx.d_tok,
                                                 sc.cap, sc.cap, ctx.d_xq, ctx.d_scr, ctx.d_out, ctx.stream, 0);
                    }
                    ck(cudaEventRecord(ev1, ctx.stream), "rec1");
                    ck(cudaEventSynchronize(ev1), "evsync");
                    float ms = 0.0f;
                    ck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
                    rds.push_back((double) ms * 1000.0 / 40.0);
                }
                std::sort(rds.begin(), rds.end());
                return rds[2];
            };

            const double t_base = measure(0, 0);
            const double t_53a  = measure(1, 0);
            const double t_53b  = measure(1, 1);
            const double bw_b   = ((double) sc.G * (double) ctx.L.bytes) / (t_53b * 1e3);
            std::printf("%-18s | %8.2f %8.2f %8.2f | %6.2fx %6.2fx | %7.1f\n",
                        sc.name, t_base, t_53a, t_53b, t_base / t_53a, t_base / t_53b, bw_b);
            tot48[si][0] += pi.weight_layers * t_base;
            tot48[si][1] += pi.weight_layers * t_53a;
            tot48[si][2] += pi.weight_layers * t_53b;
        }
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        ctx.destroy();
    }

    std::printf("\n=== 48-layer weighted MoE GPU total (25x 22/20 + 20x 18/20 + 3x 22/42), ms/window ===\n");
    std::printf("%-18s | %8s %8s %8s | %9s %9s | %7s %7s\n",
                "scenario", "PR5.2_ms", "5.3a_ms", "5.3b_ms", "d5.3a_ms", "d5.3b_ms", "5.3a/b", "5.3b/b");
    for (int si = 0; si < 6; ++si) {
        const double b = tot48[si][0] / 1000.0;
        const double a = tot48[si][1] / 1000.0;
        const double p = tot48[si][2] / 1000.0;
        std::printf("%-18s | %8.3f %8.3f %8.3f | %+9.3f %+9.3f | %6.2fx %6.2fx\n",
                    scenarios[si].name, b, a, p, a - b, p - b, b / a, b / p);
    }
    K::native_expert_set_mode(-1, 0);
    K::moe_volta_bank_set_mode(-1);
    K::moe_volta_persist_set_mode(-1);
    K::iq4nl_volta_tc_set_mode(-1);
}

}  // namespace

int main(int argc, char** argv) {
    if (!K::moe_volta_v70_supported()) {
        std::printf("moe_volta_v70_parity: requires sm_70 Volta, skipped\n");
        return 77;
    }
    bool do_bench = false;
    int n_replays = 1000;
    bool race_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") do_bench = true;
        if (std::string(argv[i]) == "--fast") n_replays = 21;
        if (std::string(argv[i]) == "--race") { n_replays = 3; race_only = true; }
    }
    std::printf("moe_volta_v70_parity: running PR 5.3 (5.3a + 5.3b) parity & CUDA Graph stress suite on sm_70\n");
    int fails = 0;
    fails += check_pair_parity(22, 20, "IQ2_S/IQ4_NL", n_replays, race_only);
    fails += check_pair_parity(18, 20, "IQ3_XXS/IQ4_NL", n_replays, race_only);
    fails += check_pair_parity(22, 42, "IQ2_S/Q2_0", n_replays, race_only);
    fails += check_pair_parity(18, 42, "IQ3_XXS/Q2_0", n_replays, race_only);
    std::printf("moe_volta_v70_parity: %d failures\n", fails);
    if (fails == 0 && do_bench) run_bench();
    return fails == 0 ? 0 : 1;
}
