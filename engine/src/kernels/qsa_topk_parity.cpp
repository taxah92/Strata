// src/kernels/qsa_topk_parity.cpp - the QSA top-k the engine dispatches (qsa_block_topk) against the original kernel
// (qsa_block_topk_ref): the selected cells must be identical (GPU, synthetic, no model).
//
// A case is a context under a capacity (the engine's --max-context; max_blocks = capacity / 4 + 2, and without an
// active-block count the dispatch follows it), with consecutive queries at the context's end so that the tail block
// holds 0 to 3 cells.  Three score sets per case: continuous, 64 levels (many ties at the threshold) and all equal
// (the rule's ties go to the lowest index).  The dispatch is called as a decode window calls it (no block count) and as
// the prompt path does (with the call's largest block count).  The reference and the dispatched kernel are also timed.
//
// Usage: qsa_topk_parity --selftest
//        qsa_topk_parity CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] [COUNT=1]   (COUNT=0: no block count)
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

bool run_case(int64_t ctx, int64_t nq, int reps, int64_t capacity, bool counted) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    std::mt19937 rng((uint32_t) (ctx * 31 + capacity));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> scores((size_t) (nq * max_blocks)), tied(scores.size()), equal(scores.size(), 0.25f);
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* st = steps.data() + i * k::kStepCount;
        float* row = scores.data() + i * max_blocks;
        for (int64_t b = 0; b < max_blocks; ++b) row[b] = nd(rng);
        // the tail block as the scorers leave it: far above the others when it holds cells
        if (st[k::kStepNKv] % s.idx_block != 0) row[st[k::kStepNBid]] += 1e9f;
        for (int64_t b = 0; b < max_blocks; ++b)
            tied[(size_t) (i * max_blocks + b)] = row[b] > 1e8f ? row[b] : std::floor(row[b] * 8.0f) / 8.0f;   // ~64 levels
    }
    // Edge rows for STRATA_TOPK_COMPACT (adapted from Strata-V100 PR #103): signed zeros, both infinities, tiny
    // denormals and exact ties in one set, NaN mixed into continuous scores in another, and 4,096 spread ties.
    std::vector<float> edges(scores.size()), nanrow(scores.size()), spread(scores.size());
    const float special[8] = {0.0f, -0.0f, INFINITY, -INFINITY, 1e-30f, -1e-30f, 0.25f, 0.5f};
    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t b = 0; b < max_blocks; ++b) {
            const size_t j = (size_t) (i * max_blocks + b);
            edges[j] = special[(size_t) ((i * 7 + b) & 7)];
            spread[j] = (float) ((b * 37) & 4095) * 0.0625f;
            nanrow[j] = scores[j];
            if (b % 97 == 0) nanrow[j] = NAN;
            else if (b % 89 == 0) nanrow[j] = INFINITY;
            else if (b % 83 == 0) nanrow[j] = -0.0f;
            else if (b % 79 == 0) nanrow[j] = -INFINITY;
        }
    }
    int32_t *d_steps = nullptr, *ids_ref = nullptr, *ids_new = nullptr;
    float* d_scores = nullptr;
    ck(cudaMalloc(&d_steps, steps.size() * 4), "malloc");
    ck(cudaMalloc(&d_scores, scores.size() * 4), "malloc");
    ck(cudaMalloc(&ids_ref, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&ids_new, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "upload");
    const int64_t active = counted ? steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1 : -1;
    std::vector<int32_t> a((size_t) (nq * cap)), b(a.size());
    auto same = [&](const std::vector<float>& sc) -> int64_t {
        ck(cudaMemcpy(d_scores, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "upload");
        ck(cudaMemset(ids_ref, 0xff, a.size() * 4), "memset");
        ck(cudaMemset(ids_new, 0xff, a.size() * 4), "memset");
        k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr);
        k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active);
        ck(cudaDeviceSynchronize(), "sync");
        ck(cudaMemcpy(a.data(), ids_ref, a.size() * 4, cudaMemcpyDeviceToHost), "down");
        ck(cudaMemcpy(b.data(), ids_new, b.size() * 4, cudaMemcpyDeviceToHost), "down");
        int64_t n = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
            n += std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap);
        }
        return n;
    };
    const int64_t same_tied = same(tied), same_equal = same(equal), same_cont = same(scores);   // d_scores: continuous
    const int64_t same_edges = same(edges), same_nan = same(nanrow), same_spread = same(spread);
    ck(cudaMemcpy(d_scores, scores.data(), scores.size() * 4, cudaMemcpyHostToDevice), "upload");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto timed = [&](auto f) {
        ck(cudaEventRecord(e0), "record");
        for (int r = 0; r < reps; ++r) f();
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "time");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return ms / (float) reps;
    };
    const float t_ref = timed([&] { k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr); });
    const float t_new = timed([&] { k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active); });
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_steps); cudaFree(d_scores); cudaFree(ids_ref); cudaFree(ids_new);
    const bool ok = same_cont == nq && same_tied == nq && same_equal == nq && same_edges == nq && same_nan == nq &&
                    same_spread == nq;
    std::printf("%s context %lld under a capacity of %lld (%lld blocks), %lld queries %s: reference %.3f ms, "
                "dispatched %.3f ms (%.1fx); identical ids: continuous %lld/%lld, 64 levels %lld/%lld, equal scores "
                "%lld/%lld, signed-zero/infinities %lld/%lld, NaN rows %lld/%lld, spread ties %lld/%lld\n",
                ok ? "PASS" : "FAIL", (long long) ctx, (long long) capacity, (long long) max_blocks, (long long) nq,
                counted ? "with their block count" : "without a block count", t_ref, t_new, t_ref / t_new,
                (long long) same_cont, (long long) nq, (long long) same_tied, (long long) nq,
                (long long) same_equal, (long long) nq, (long long) same_edges, (long long) nq,
                (long long) same_nan, (long long) nq, (long long) same_spread, (long long) nq);
    return ok;
}

bool run_graph_replay_case() {
    // Capture qsa_block_topk once at max-context 524,288 (131,074 blocks), then replay across changing n_kv
    // (identity <= width, short < TC_MIN_CELLS, 150K, 312K, NaN, all-equal, spread ties, 524K) to verify
    // zero state leak across CUDA Graph replays.
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t capacity = 524288;
    const int64_t max_blocks = capacity / s.idx_block + 2;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t nq = 6;
    int32_t *d_steps = nullptr, *ids_ref = nullptr, *ids_new = nullptr;
    float* d_scores = nullptr;
    ck(cudaMalloc(&d_steps, (size_t) (nq * k::kStepCount) * 4), "graph malloc");
    ck(cudaMalloc(&d_scores, (size_t) (nq * max_blocks) * 4), "graph malloc");
    ck(cudaMalloc(&ids_ref, (size_t) (nq * cap) * 4), "graph malloc");
    ck(cudaMalloc(&ids_new, (size_t) (nq * cap) * 4), "graph malloc");
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "graph stream");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, stream);
    ck(cudaStreamEndCapture(stream, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "graph instantiate");

    struct Step { int64_t ctx; int kind; const char* name; };
    const Step seq[] = {
        {1200, 0, "identity<=width"},
        {16384, 0, "short<32K"},
        {150003, 0, "150K-cont"},
        {312001, 2, "312K-NaN"},
        {312001, 1, "312K-equal"},
        {312002, 3, "312K-spread"},
        {524288, 0, "524K-cont"},
        {242235, 0, "242K-cont"},
    };
    std::mt19937 rng(424242u);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    std::vector<float> scores((size_t) (nq * max_blocks));
    std::vector<int32_t> a((size_t) (nq * cap)), b(a.size());
    bool ok = true;
    for (const auto& tc : seq) {
        for (int64_t i = 0; i < nq; ++i) {
            int32_t* st = steps.data() + i * k::kStepCount;
            const int64_t pos = tc.ctx - nq + i;
            st[k::kStepPos] = (int32_t) pos;
            st[k::kStepNKv] = (int32_t) (pos + 1);
            st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
            st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
            float* row = scores.data() + i * max_blocks;
            for (int64_t j = 0; j < max_blocks; ++j) {
                if (tc.kind == 1) row[j] = 0.25f;
                else if (tc.kind == 3) row[j] = (float) ((j * 37) & 4095) * 0.0625f;
                else row[j] = nd(rng);
            }
            if (tc.kind == 2) {
                for (int64_t j = 0; j <= st[k::kStepNBid]; j += 97) row[j] = NAN;
            }
            if (tc.kind == 0 && st[k::kStepNKv] % s.idx_block != 0) row[st[k::kStepNBid]] += 1e9f;
        }
        ck(cudaMemcpyAsync(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice, stream), "upload steps");
        ck(cudaMemcpyAsync(d_scores, scores.data(), scores.size() * 4, cudaMemcpyHostToDevice, stream), "upload scores");
        ck(cudaMemsetAsync(ids_ref, 0xff, a.size() * 4, stream), "memset ref");
        ck(cudaMemsetAsync(ids_new, 0xff, a.size() * 4, stream), "memset new");
        k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, stream);
        ck(cudaGraphLaunch(exec, stream), "graph launch");
        ck(cudaStreamSynchronize(stream), "graph sync");
        ck(cudaMemcpy(a.data(), ids_ref, a.size() * 4, cudaMemcpyDeviceToHost), "down ref");
        ck(cudaMemcpy(b.data(), ids_new, b.size() * 4, cudaMemcpyDeviceToHost), "down new");
        int64_t same_n = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
            same_n += std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap);
        }
        if (same_n != nq) {
            std::printf("FAIL CUDA graph replay step %s (ctx=%lld): %lld/%lld\n",
                        tc.name, (long long) tc.ctx, (long long) same_n, (long long) nq);
            ok = false;
        }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_steps); cudaFree(d_scores); cudaFree(ids_ref); cudaFree(ids_new);
    std::printf("%s CUDA graph replay across 8 dynamic n_kv/edge transitions (nq=%lld, cap=524288)\n",
                ok ? "PASS" : "FAIL", (long long) nq);
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0) {
        // {context, capacity}: both sides of the register kernel's reach (33,792 blocks = 135,168 cells), short
        // contexts under a long capacity (fewer blocks than the kernel's threads; a prompt batch below the dispatch's
        // 4,608 blocks), and the long contexts themselves
        const int64_t cases[][2] = {{9000, 9000},    {135160, 135160}, {135176, 135176}, {2100, 262144},
                                    {9000, 262144},  {32768, 262144},  {262144, 262144}, {300001, 524288},
                                    {524288, 524288}};
        bool ok = true;
        for (const auto& c : cases) {
            ok = run_case(c[0], 8, 3, c[1], false) && ok;     // a decode window's call
            ok = run_case(c[0], 64, 3, c[1], true) && ok;     // the prompt path's
        }
        ok = run_graph_replay_case() && ok;
        std::printf("%s\n", ok ? "qsa_topk_parity: PASS" : "qsa_topk_parity: FAIL");
        return ok ? 0 : 1;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: qsa_topk_parity --selftest | CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] "
                             "[COUNT=1]\n");
        return 2;
    }
    const int64_t ctx = std::atoll(argv[1]);
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : ctx;
    if (ctx < nq || nq <= 0 || reps <= 0 || capacity < ctx) {
        std::fprintf(stderr, "qsa_topk_parity: CONTEXT >= QUERIES > 0, REPS > 0, CAPACITY >= CONTEXT\n");
        return 2;
    }
    return run_case(ctx, nq, reps, capacity, argc <= 5 || std::atoi(argv[5]) != 0) ? 0 : 1;
}
