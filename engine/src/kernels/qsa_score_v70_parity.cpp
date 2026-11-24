// src/kernels/qsa_score_v70_parity.cpp - PR 3.1: Volta (sm_70) shared-key mma.m8n8k4 QSA indexer scorer parity,
// top-k agreement, CUDA Graph multi-replay test, and microbenchmark (decode nq=1..12 + prefill qsa_block_scores_tc).
//
// Usage:
//   qsa_score_v70_parity [--bench] [--device N]
// Exit 0: PASS; 77: not an sm_70 CUDA device; 1: parity/graph failure.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_score_v70_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 64), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "cudaMemset");
    return p;
}

template <typename T> T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

std::vector<int32_t> make_steps(int nq, int64_t pos0, const k::QsaShapes& s) {
    std::vector<int32_t> st((size_t) nq * k::kStepCount, 0);
    for (int i = 0; i < nq; ++i) {
        int32_t* p = st.data() + (size_t) i * k::kStepCount;
        const int64_t pos = pos0 + i;
        p[k::kStepPos] = (int32_t) pos;
        p[k::kStepNKv] = (int32_t) (pos + 1);
        p[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        p[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    return st;
}

// dist: 0 = realistic shared-direction (like real QSA indexer), 1 = standard normal, 2 = wide dynamic range 2^[-20..20] + zeros
void fill_data(std::mt19937& rng, int dist, int64_t max_blocks, int nq,
               std::vector<float>& pooled, std::vector<float>& dead, std::vector<float>& q) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> expo(-20, 20), coin(0, 9);
    if (dist == 0) {
        std::vector<float> dir(128);
        for (auto& x : dir) x = nd(rng);
        for (int64_t b = 0; b < max_blocks; ++b) {
            const float a = nd(rng);
            for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * a * dir[d] + nd(rng);
        }
        for (auto& x : dead) x = nd(rng);
        for (int i = 0; i < nq; ++i)
            for (int d = 0; d < 512; ++d) q[(size_t) (i * 512 + d)] = 0.2f * dir[d % 128] + 0.1f * nd(rng);
    } else if (dist == 1) {
        for (auto& x : pooled) x = nd(rng);
        for (auto& x : dead) x = nd(rng);
        for (auto& x : q) x = nd(rng);
    } else {
        auto val = [&] { return coin(rng) == 0 ? 0.0f : std::ldexp(nd(rng), expo(rng)); };
        for (auto& x : pooled) x = val();
        for (auto& x : dead) x = val();
        for (auto& x : q) x = val();
    }
}

struct Stats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double scale = 0.0;
    double tail_max_abs = 0.0;
    int64_t topk_total = 0;
    int64_t topk_common = 0;
};

Stats compare_scores_and_topk(int nq, int64_t max_blocks, int64_t cap,
                              const std::vector<int32_t>& steps,
                              const std::vector<float>& ref, const std::vector<float>& got,
                              const std::vector<int32_t>& ids_ref, const std::vector<int32_t>& ids_got) {
    Stats st{};
    for (int i = 0; i < nq; ++i) {
        const int64_t nb = steps[(size_t) i * k::kStepCount + k::kStepNBid];
        for (int64_t b = 0; b <= nb; ++b) {
            const double a = ref[(size_t) (i * max_blocks + b)];
            const double g = got[(size_t) (i * max_blocks + b)];
            const double diff = std::fabs(a - g);
            if (b == nb) {
                st.tail_max_abs = std::max(st.tail_max_abs, diff);
            } else {
                st.scale = std::max(st.scale, std::fabs(a));
                st.max_abs = std::max(st.max_abs, diff);
                st.max_rel = std::max(st.max_rel, diff / std::max(1.0, std::fabs(a)));
            }
        }
        const int64_t w = steps[(size_t) i * k::kStepCount + k::kStepWidth];
        std::vector<int32_t> x(ids_ref.begin() + i * cap, ids_ref.begin() + i * cap + w);
        std::vector<int32_t> y(ids_got.begin() + i * cap, ids_got.begin() + i * cap + w);
        std::vector<int32_t> inter;
        std::set_intersection(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(inter));
        st.topk_total += w;
        st.topk_common += (int64_t) inter.size();
    }
    return st;
}

bool check_decode_case(const k::QsaShapes& s, int64_t ctx, int nq, int dist, int mode, std::mt19937& rng) {
    const int64_t max_blocks = std::max<int64_t>(ctx / s.idx_block + 4, 16);
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<float> pooled((size_t) (max_blocks * s.idx_dim)), dead((size_t) s.idx_dim),
        q((size_t) (nq * s.idx_n_head * s.idx_dim));
    fill_data(rng, dist, max_blocks, nq, pooled, dead, q);
    const int64_t pos0 = std::max<int64_t>(0, ctx - nq);
    const std::vector<int32_t> steps = make_steps(nq, pos0, s);

    float *dp = upload(pooled), *dd = upload(dead), *dq = upload(q);
    int32_t* dst = upload(steps);
    float *sc_ref = dalloc<float>((size_t) (nq * max_blocks)), *sc_got = dalloc<float>((size_t) (nq * max_blocks));
    int32_t *id_ref = dalloc<int32_t>((size_t) (nq * cap)), *id_got = dalloc<int32_t>((size_t) (nq * cap));
    ck(cudaMemset(sc_ref, 0x7f, (size_t) (nq * max_blocks) * 4), "fill ref");
    ck(cudaMemset(sc_got, 0x3e, (size_t) (nq * max_blocks) * 4), "fill got");

    k::qsa_set_score_v70(0);
    k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc_ref, nullptr, -1);
    k::qsa_block_topk(sc_ref, dst, nq, max_blocks, cap, s, id_ref, nullptr);

    k::qsa_set_score_v70(mode);
    k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc_got, nullptr, -1);
    k::qsa_block_topk(sc_got, dst, nq, max_blocks, cap, s, id_got, nullptr);
    ck(cudaDeviceSynchronize(), "sync");

    std::vector<float> href((size_t) (nq * max_blocks)), hgot(href.size());
    std::vector<int32_t> iref((size_t) (nq * cap)), igot(iref.size());
    ck(cudaMemcpy(href.data(), sc_ref, href.size() * 4, cudaMemcpyDeviceToHost), "down ref");
    ck(cudaMemcpy(hgot.data(), sc_got, hgot.size() * 4, cudaMemcpyDeviceToHost), "down got");
    ck(cudaMemcpy(iref.data(), id_ref, iref.size() * 4, cudaMemcpyDeviceToHost), "down iref");
    ck(cudaMemcpy(igot.data(), id_got, igot.size() * 4, cudaMemcpyDeviceToHost), "down igot");

    const Stats st = compare_scores_and_topk(nq, max_blocks, cap, steps, href, hgot, iref, igot);
    const double overlap = st.topk_total ? 100.0 * (double) st.topk_common / (double) st.topk_total : 100.0;
    const bool is_hilo = (mode == 3);
    // FP16 half-ULP is 2^-11 = 4.88e-4 (1-pass hi); 3-pass hi+lo retains ~22 mantissa bits (~1e-6).
    const double tol_rel = is_hilo ? 1e-5 : 1e-3;
    const double rel_vs_scale = st.scale > 0.0 ? st.max_abs / st.scale : st.max_abs;
    const bool ok_score = (st.tail_max_abs == 0.0) && (rel_vs_scale <= tol_rel);
    const bool ok_topk = overlap >= 99.9;
    const bool ok = ok_score && ok_topk;
    std::printf("%s dec  mode=%d dist=%d ctx=%7lld nq=%2d: max_abs=%.3e (%.3e of scale %.3g) tail=%.1e topk=%.4f%%\n",
                ok ? "ok  " : "FAIL", mode, dist, (long long) ctx, nq, st.max_abs, rel_vs_scale, st.scale,
                st.tail_max_abs, overlap);

    for (void* p : {(void*) dp, (void*) dd, (void*) dq, (void*) dst, (void*) sc_ref, (void*) sc_got, (void*) id_ref, (void*) id_got})
        cudaFree(p);
    return ok;
}

bool check_prefill_case(const k::QsaShapes& s, int64_t ctx, int nq, int dist, int mode, std::mt19937& rng) {
    const int64_t max_blocks = ctx / s.idx_block + 4;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<float> pooled((size_t) (max_blocks * s.idx_dim)), dead((size_t) s.idx_dim),
        q((size_t) (nq * s.idx_n_head * s.idx_dim));
    fill_data(rng, dist, max_blocks, nq, pooled, dead, q);
    const std::vector<int32_t> steps = make_steps(nq, ctx - nq, s);
    const int64_t active = steps[(size_t) (nq - 1) * k::kStepCount + k::kStepNBid] + 1;

    float *dp = upload(pooled), *dd = upload(dead), *dq = upload(q);
    int32_t* dst = upload(steps);
    float *sc_ref = dalloc<float>((size_t) (nq * max_blocks)), *sc_got = dalloc<float>((size_t) (nq * max_blocks));
    int32_t *id_ref = dalloc<int32_t>((size_t) (nq * cap)), *id_got = dalloc<int32_t>((size_t) (nq * cap));

    k::qsa_set_score_v70(0);
    k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc_ref, nullptr, active);
    k::qsa_block_topk(sc_ref, dst, nq, max_blocks, cap, s, id_ref, nullptr, active);

    k::qsa_set_score_v70(mode);
    const bool ran = k::qsa_block_scores_tc(dp, dd, dq, dst, nq, max_blocks, s, sc_got, nullptr, active);
    k::qsa_block_topk(sc_got, dst, nq, max_blocks, cap, s, id_got, nullptr, active);
    ck(cudaDeviceSynchronize(), "sync");

    std::vector<float> href((size_t) (nq * max_blocks)), hgot(href.size());
    std::vector<int32_t> iref((size_t) (nq * cap)), igot(iref.size());
    ck(cudaMemcpy(href.data(), sc_ref, href.size() * 4, cudaMemcpyDeviceToHost), "down ref");
    ck(cudaMemcpy(hgot.data(), sc_got, hgot.size() * 4, cudaMemcpyDeviceToHost), "down got");
    ck(cudaMemcpy(iref.data(), id_ref, iref.size() * 4, cudaMemcpyDeviceToHost), "down iref");
    ck(cudaMemcpy(igot.data(), id_got, igot.size() * 4, cudaMemcpyDeviceToHost), "down igot");

    const Stats st = compare_scores_and_topk(nq, max_blocks, cap, steps, href, hgot, iref, igot);
    const double overlap = st.topk_total ? 100.0 * (double) st.topk_common / (double) st.topk_total : 100.0;
    const double rel_vs_scale = st.scale > 0.0 ? st.max_abs / st.scale : st.max_abs;
    const double tol_rel = (mode == 1) ? 1e-5 : 3e-3;
    const bool ok = ran && (st.tail_max_abs == 0.0) && (rel_vs_scale <= tol_rel) && (overlap >= 99.9);
    std::printf("%s pf   mode=%d dist=%d ctx=%7lld nq=%3d: max_abs=%.3e (%.3e of scale %.3g) tail=%.1e topk=%.4f%%\n",
                ok ? "ok  " : "FAIL", mode, dist, (long long) ctx, nq, st.max_abs, rel_vs_scale, st.scale,
                st.tail_max_abs, overlap);

    for (void* p : {(void*) dp, (void*) dd, (void*) dq, (void*) dst, (void*) sc_ref, (void*) sc_got, (void*) id_ref, (void*) id_got})
        cudaFree(p);
    return ok;
}

bool check_cuda_graph_replay(const k::QsaShapes& s, int nq, int mode, std::mt19937& rng) {
    const int64_t max_ctx = 131072;
    const int64_t max_blocks = max_ctx / s.idx_block + 4;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<float> pooled((size_t) (max_blocks * s.idx_dim)), dead((size_t) s.idx_dim),
        q((size_t) (nq * s.idx_n_head * s.idx_dim));
    fill_data(rng, 0, max_blocks, nq, pooled, dead, q);

    float *dp = upload(pooled), *dd = upload(dead), *dq = upload(q);
    int32_t* dst = dalloc<int32_t>((size_t) (nq * k::kStepCount));
    float *sc_got = dalloc<float>((size_t) (nq * max_blocks)), *sc_ref = dalloc<float>((size_t) (nq * max_blocks));
    int32_t *id_got = dalloc<int32_t>((size_t) (nq * cap)), *id_ref = dalloc<int32_t>((size_t) (nq * cap));

    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "stream");

    // Capture at initial tiny context (2K) — graph must grow dynamically up to 131K on replay!
    const std::vector<int32_t> init_steps = make_steps(nq, 2048 - nq, s);
    ck(cudaMemcpy(dst, init_steps.data(), init_steps.size() * 4, cudaMemcpyHostToDevice), "init steps");

    // Warm once before capture (for >48KB smem attribute setup when nq > 8)
    k::qsa_set_score_v70(mode);
    k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc_got, stream, -1);
    ck(cudaStreamSynchronize(stream), "warm");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc_got, stream, -1);
    k::qsa_block_topk(sc_got, dst, nq, max_blocks, cap, s, id_got, stream);
    ck(cudaStreamEndCapture(stream, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");

    const int64_t replay_ctxs[] = {4, 17, 2048, 32769, 131072};
    bool ok = true;
    for (int64_t ctx : replay_ctxs) {
        const std::vector<int32_t> steps = make_steps(nq, std::max<int64_t>(0, ctx - nq), s);
        // Perturb queries + dead on every replay to verify no stale state is baked into the graph
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& x : dead) x = nd(rng);
        for (auto& x : q) x = 0.25f * nd(rng);
        ck(cudaMemcpyAsync(dd, dead.data(), dead.size() * 4, cudaMemcpyHostToDevice, stream), "up dead");
        ck(cudaMemcpyAsync(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice, stream), "up q");
        ck(cudaMemcpyAsync(dst, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice, stream), "up steps");
        ck(cudaMemsetAsync(sc_got, 0x3e, (size_t) (nq * max_blocks) * 4, stream), "clear got");
        ck(cudaGraphLaunch(exec, stream), "launch graph");
        ck(cudaStreamSynchronize(stream), "sync graph");

        k::qsa_set_score_v70(0);
        k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc_ref, stream, -1);
        k::qsa_block_topk(sc_ref, dst, nq, max_blocks, cap, s, id_ref, stream);
        ck(cudaStreamSynchronize(stream), "sync ref");

        std::vector<float> href((size_t) (nq * max_blocks)), hgot(href.size());
        std::vector<int32_t> iref((size_t) (nq * cap)), igot(iref.size());
        ck(cudaMemcpy(href.data(), sc_ref, href.size() * 4, cudaMemcpyDeviceToHost), "down ref");
        ck(cudaMemcpy(hgot.data(), sc_got, hgot.size() * 4, cudaMemcpyDeviceToHost), "down got");
        ck(cudaMemcpy(iref.data(), id_ref, iref.size() * 4, cudaMemcpyDeviceToHost), "down iref");
        ck(cudaMemcpy(igot.data(), id_got, igot.size() * 4, cudaMemcpyDeviceToHost), "down igot");
        const Stats st = compare_scores_and_topk(nq, max_blocks, cap, steps, href, hgot, iref, igot);
        const double rel = st.scale > 0.0 ? st.max_abs / st.scale : st.max_abs;
        const double overlap = st.topk_total ? 100.0 * (double) st.topk_common / (double) st.topk_total : 100.0;
        const bool pass = (st.tail_max_abs == 0.0) && (rel <= (mode == 3 ? 1e-5 : 1e-3)) && (overlap >= 99.9);
        if (!pass) {
            std::fprintf(stderr, "FAIL graph nq=%d mode=%d ctx=%lld: rel=%.3e tail=%.1e topk=%.4f%%\n",
                         nq, mode, (long long) ctx, rel, st.tail_max_abs, overlap);
            ok = false;
            break;
        }
    }
    std::printf("%s graph replay mode=%d nq=%2d (5 dynamic lengths 4..131K)\n", ok ? "ok  " : "FAIL", mode, nq);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    for (void* p : {(void*) dp, (void*) dd, (void*) dq, (void*) dst, (void*) sc_ref, (void*) sc_got, (void*) id_ref, (void*) id_got})
        cudaFree(p);
    return ok;
}

void run_bench(const k::QsaShapes& s) {
    std::mt19937 rng(42);
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "e0");
    ck(cudaEventCreate(&e1), "e1");
    auto time_us = [&](auto fn, int reps) {
        for (int i = 0; i < 5; ++i) fn();
        ck(cudaDeviceSynchronize(), "warm");
        double med[3];
        for (int r = 0; r < 3; ++r) {
            ck(cudaEventRecord(e0), "r0");
            for (int i = 0; i < reps; ++i) fn();
            ck(cudaEventRecord(e1), "r1");
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0.0f;
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            med[r] = 1000.0 * ms / reps;
        }
        std::sort(med, med + 3);
        return med[1];
    };

    std::puts("\n=== DECODE BENCH (qsa_block_scores, active=-1, 1 call us | 12 layers ms) ===");
    const int64_t ctxs[] = {32768, 131072, 300000};
    const int nqs[] = {2, 3, 4, 5, 6, 8, 12};
    for (int64_t ctx : ctxs) {
        const int64_t max_blocks = ctx / s.idx_block + 4;
        std::vector<float> pooled((size_t) (max_blocks * s.idx_dim)), dead((size_t) s.idx_dim),
            q((size_t) (12 * s.idx_n_head * s.idx_dim));
        fill_data(rng, 0, max_blocks, 12, pooled, dead, q);
        float *dp = upload(pooled), *dd = upload(dead), *dq = upload(q);
        float* sc = dalloc<float>((size_t) (12 * max_blocks));
        for (int nq : nqs) {
            const std::vector<int32_t> steps = make_steps(nq, ctx - nq, s);
            int32_t* dst = upload(steps);
            const int reps = ctx >= 300000 ? 40 : 100;
            k::qsa_set_score_v70(0);
            const double t0 = time_us([&] { k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc, nullptr, -1); }, reps);
            k::qsa_set_score_v70(3);
            const double t3 = time_us([&] { k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc, nullptr, -1); }, reps);
            k::qsa_set_score_v70(1);
            const double t1 = time_us([&] { k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc, nullptr, -1); }, reps);
            const double gb = (double) (ctx / 4) * 512.0 / 1e9;
            std::printf("ctx=%6lld nq=%2d: base %6.1f us (%.2f ms/12L) | v70_hilo %6.1f us (%.2f ms/12L, %4.2fx, %3.0f GB/s) | v70_def %6.1f us (%.2f ms/12L, %4.2fx, %3.0f GB/s)\n",
                        (long long) ctx, nq, t0, t0 * 12.0 / 1000.0,
                        t3, t3 * 12.0 / 1000.0, t0 / t3, gb / (t3 / 1e6),
                        t1, t1 * 12.0 / 1000.0, t0 / t1, gb / (t1 / 1e6));
            cudaFree(dst);
        }
        cudaFree(dp); cudaFree(dd); cudaFree(dq); cudaFree(sc);
    }

    std::puts("\n=== PREFILL BENCH (qsa_block_scores_tc, active>0, ms/call) ===");
    const int64_t pf_ctxs[] = {32768, 131072, 280000};
    for (int64_t ctx : pf_ctxs) {
        const int nq = 256;
        const int64_t max_blocks = ctx / s.idx_block + 4;
        std::vector<float> pooled((size_t) (max_blocks * s.idx_dim)), dead((size_t) s.idx_dim),
            q((size_t) (nq * s.idx_n_head * s.idx_dim));
        fill_data(rng, 0, max_blocks, nq, pooled, dead, q);
        const std::vector<int32_t> steps = make_steps(nq, ctx - nq, s);
        const int64_t active = steps[(size_t) (nq - 1) * k::kStepCount + k::kStepNBid] + 1;
        float *dp = upload(pooled), *dd = upload(dead), *dq = upload(q);
        int32_t* dst = upload(steps);
        float* sc = dalloc<float>((size_t) (nq * max_blocks));
        const int reps = ctx >= 280000 ? 10 : 20;
        k::qsa_set_score_v70(0);
        const double t_warp = time_us([&] { k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, sc, nullptr, active); }, reps) / 1000.0;
        const bool have_simt = k::qsa_block_scores_tc(dp, dd, dq, dst, nq, max_blocks, s, sc, nullptr, active);
        const double t_simt = have_simt
            ? time_us([&] { k::qsa_block_scores_tc(dp, dd, dq, dst, nq, max_blocks, s, sc, nullptr, active); }, reps) / 1000.0
            : t_warp;
        k::qsa_set_score_v70(1);
        const double t_v70_1 = time_us([&] { k::qsa_block_scores_tc(dp, dd, dq, dst, nq, max_blocks, s, sc, nullptr, active); }, reps) / 1000.0;
        k::qsa_set_score_v70(2);
        const double t_v70_2 = time_us([&] { k::qsa_block_scores_tc(dp, dd, dq, dst, nq, max_blocks, s, sc, nullptr, active); }, reps) / 1000.0;
        std::printf("ctx=%6lld nq=%3d: warp %.3f ms | simt %.3f ms | v70_hilo %.3f ms (%.2fx vs simt, %.2fx vs warp) | v70_hi %.3f ms (%.2fx vs simt)\n",
                    (long long) ctx, nq, t_warp, t_simt, t_v70_1, t_simt / t_v70_1, t_warp / t_v70_1, t_v70_2, t_simt / t_v70_2);
        cudaFree(dp); cudaFree(dd); cudaFree(dq); cudaFree(dst); cudaFree(sc);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}
}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) bench = true;
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = std::atoi(argv[++i]);
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= device) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    ck(cudaSetDevice(device), "setDevice");
    int major = 0, minor = 0;
    ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "cc major");
    ck(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "cc minor");
    if (major != 7 || minor != 0) {
        std::printf("SKIP: device %d is sm_%d%d (need sm_70)\n", device, major, minor);
        return 77;
    }

    const k::QsaShapes s = k::qsa_real_shapes();
    std::mt19937 rng(20261124);
    bool ok = true;

    // 1. Edge cases + full decode sweep (mode 1 = default 1-pass hi with pow2 scale)
    const int64_t edge_ctxs[] = {2, 4, 5, 31, 32, 33, 128, 129, 2048};
    for (int64_t ctx : edge_ctxs) ok &= check_decode_case(s, ctx, 5, 0, 1, rng);
    const int64_t dec_ctxs[] = {2048, 32768, 131072, 300000};
    const int dec_nqs[] = {1, 2, 3, 4, 5, 6, 8, 9, 12};
    for (int64_t ctx : dec_ctxs) {
        for (int nq : dec_nqs) {
            if (ctx == 300000 && nq != 5 && nq != 8 && nq != 12) continue;
            ok &= check_decode_case(s, ctx, nq, 0, 1, rng);
        }
    }
    // Wide dynamic range (2^[-20..20] + zeros) and standard normal: both mode 1 (hi) and mode 3 (hi+lo)
    ok &= check_decode_case(s, 32768, 5, 1, 1, rng);
    ok &= check_decode_case(s, 32768, 5, 2, 1, rng);
    ok &= check_decode_case(s, 32768, 12, 2, 1, rng);
    ok &= check_decode_case(s, 32768, 5, 1, 3, rng);
    ok &= check_decode_case(s, 32768, 5, 2, 3, rng);
    ok &= check_decode_case(s, 131072, 5, 0, 3, rng);
    ok &= check_decode_case(s, 300000, 12, 0, 3, rng);

    // 2. Prefill qsa_block_scores_tc sweep (mode 1 = 3-pass hi+lo FP64-exact; mode 2 = 1-pass hi)
    ok &= check_prefill_case(s, 2048, 16, 0, 1, rng);
    ok &= check_prefill_case(s, 32768, 64, 0, 1, rng);
    ok &= check_prefill_case(s, 32768, 256, 2, 1, rng);
    ok &= check_prefill_case(s, 131072, 256, 0, 1, rng);
    ok &= check_prefill_case(s, 32768, 256, 0, 2, rng);

    // 3. CUDA Graph capture + multi-replay with dynamic context growth & changing inputs
    ok &= check_cuda_graph_replay(s, 3, 1, rng);
    ok &= check_cuda_graph_replay(s, 5, 1, rng);
    ok &= check_cuda_graph_replay(s, 8, 1, rng);
    ok &= check_cuda_graph_replay(s, 12, 1, rng);
    ok &= check_cuda_graph_replay(s, 5, 3, rng);

    k::qsa_set_score_v70(-1);
    if (!ok) {
        std::puts("FAIL: qsa_score_v70_parity");
        return 1;
    }
    std::puts("PASS: qsa_score_v70_parity");

    if (bench) run_bench(s);
    return 0;
}
