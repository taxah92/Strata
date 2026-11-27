// src/kernels/mtp_zero_d2h_parity.cpp - PR 6.2a parity, CUDA Graph conditional gate, and sanitizer test
// for mtp_topk_prob_select and mtp_zero_d2h_gate.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/spec/draft_policy.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mtp_zero_d2h_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Top4Ref {
    int32_t ids[4];
    float probs[4];
};

Top4Ref ref_top4(const float* logits, int n, const int32_t* dvocab) {
    std::vector<std::pair<float, int>> items;
    items.reserve((size_t) n);
    double mx = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(logits[i])) {
            items.push_back({logits[i], i});
            if ((double) logits[i] > mx) mx = (double) logits[i];
        }
    }
    std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    double sum = 0.0;
    for (const auto& it : items) sum += std::exp((double) it.first - mx);
    Top4Ref r{};
    for (int k = 0; k < 4; ++k) {
        if (k < (int) items.size() && std::isfinite(mx) && sum > 0.0) {
            const int raw = items[(size_t) k].second;
            r.ids[k] = dvocab ? dvocab[raw] : raw;
            r.probs[k] = (float) (std::exp((double) items[(size_t) k].first - mx) / sum);
        } else {
            r.ids[k] = (k == 0 ? 0 : -1);
            r.probs[k] = 0.0f;
        }
    }
    return r;
}

int test_topk_select(cudaStream_t st) {
    int fails = 0;
    std::mt19937 rng(42);
    const int vocabs[] = {64, 4096, 65536, 248320};
    const int n_rows_list[] = {1, 2, 4};
    constexpr int64_t R_stride = 10240;

    float *d_Rsrc = nullptr, *d_Rdst = nullptr;
    int32_t *d_rowdev = nullptr, *d_tokdst = nullptr;
    ck(cudaMalloc(&d_Rsrc, (size_t) (4 * R_stride) * sizeof(float)), "malloc Rsrc");
    ck(cudaMalloc(&d_Rdst, (size_t) R_stride * sizeof(float)), "malloc Rdst");
    ck(cudaMalloc(&d_rowdev, 4), "malloc rowdev");
    ck(cudaMalloc(&d_tokdst, 4), "malloc tokdst");

    std::vector<float> h_Rsrc((size_t) (4 * R_stride));
    for (size_t i = 0; i < h_Rsrc.size(); ++i) h_Rsrc[i] = (float) ((int)(i % 97) - 48) * 0.125f;
    ck(cudaMemcpy(d_Rsrc, h_Rsrc.data(), h_Rsrc.size() * sizeof(float), cudaMemcpyHostToDevice), "copy Rsrc");

    for (int nv : vocabs) {
        for (int n_rows : n_rows_list) {
            for (int use_sub = 0; use_sub <= 1; ++use_sub) {
                std::vector<float> h_lg((size_t) n_rows * (size_t) nv);
                std::normal_distribution<float> nd(0.0f, 3.0f);
                for (auto& x : h_lg) x = nd(rng);
                // Inject tie and NaN edge cases on small vocab
                if (nv == 64 && n_rows == 1) {
                    h_lg[10] = 15.0f;
                    h_lg[27] = 15.0f;  // tie: index 10 must win over 27
                    h_lg[3] = std::numeric_limits<float>::quiet_NaN();
                    h_lg[50] = 14.5f;
                    h_lg[51] = 14.25f;
                }
                std::vector<int32_t> h_sub((size_t) nv);
                for (int i = 0; i < nv; ++i) h_sub[(size_t) i] = 100000 + (i * 7) % 140000;

                float *d_lg = nullptr, *d_probs = nullptr;
                int32_t *d_sub = nullptr, *d_ids = nullptr;
                void* d_scr = nullptr;
                ck(cudaMalloc(&d_lg, h_lg.size() * sizeof(float)), "malloc lg");
                ck(cudaMemcpy(d_lg, h_lg.data(), h_lg.size() * sizeof(float), cudaMemcpyHostToDevice), "copy lg");
                if (use_sub) {
                    ck(cudaMalloc(&d_sub, h_sub.size() * sizeof(int32_t)), "malloc sub");
                    ck(cudaMemcpy(d_sub, h_sub.data(), h_sub.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy sub");
                }
                ck(cudaMalloc(&d_ids, (size_t) n_rows * 4), "malloc ids");
                ck(cudaMalloc(&d_probs, (size_t) n_rows * 4), "malloc probs");
                const uint64_t sb = strata::kernels::mtp_topk_prob_select_scratch_bytes(n_rows);
                ck(cudaMalloc(&d_scr, sb), "malloc scr");
                ck(cudaMemset(d_scr, 0, sb), "zero scr");

                int32_t *h_mout = nullptr, *m_mout = nullptr;
                float *h_mprob = nullptr, *m_mprob = nullptr;
                int32_t *h_t4i = nullptr, *m_t4i = nullptr;
                float *h_t4p = nullptr, *m_t4p = nullptr;
                ck(cudaHostAlloc(&h_mout, 32, cudaHostAllocMapped), "alloc mout");
                ck(cudaHostGetDevicePointer(&m_mout, h_mout, 0), "dev mout");
                ck(cudaHostAlloc(&h_mprob, 32, cudaHostAllocMapped), "alloc mprob");
                ck(cudaHostGetDevicePointer(&m_mprob, h_mprob, 0), "dev mprob");
                ck(cudaHostAlloc(&h_t4i, 64 * 4, cudaHostAllocMapped), "alloc t4i");
                ck(cudaHostGetDevicePointer(&m_t4i, h_t4i, 0), "dev t4i");
                ck(cudaHostAlloc(&h_t4p, 64 * 4, cudaHostAllocMapped), "alloc t4p");
                ck(cudaHostGetDevicePointer(&m_t4p, h_t4p, 0), "dev t4p");

                const int rsel = (n_rows == 1) ? 2 : 0;
                ck(cudaMemcpy(d_rowdev, &rsel, 4, cudaMemcpyHostToDevice), "copy rsel");
                ck(cudaMemset(d_Rdst, 0, (size_t) R_stride * sizeof(float)), "zero Rdst");
                const int step_j = 1;

                strata::kernels::mtp_topk_prob_select(
                    d_lg, n_rows, nv, d_sub,
                    n_rows == 1 ? d_Rsrc : nullptr, n_rows == 1 ? R_stride : 0, d_rowdev,
                    n_rows == 1 ? d_Rdst : nullptr, n_rows == 1 ? d_tokdst : nullptr,
                    d_ids, d_probs, n_rows == 1 ? m_mout : nullptr, n_rows == 1 ? m_mprob : nullptr,
                    m_t4i, m_t4p, step_j, d_scr, st);
                ck(cudaStreamSynchronize(st), "sync topk");

                std::vector<int32_t> g_ids((size_t) n_rows);
                std::vector<float> g_probs((size_t) n_rows);
                ck(cudaMemcpy(g_ids.data(), d_ids, (size_t) n_rows * 4, cudaMemcpyDeviceToHost), "d2h ids");
                ck(cudaMemcpy(g_probs.data(), d_probs, (size_t) n_rows * 4, cudaMemcpyDeviceToHost), "d2h probs");

                for (int r = 0; r < n_rows; ++r) {
                    const Top4Ref ref = ref_top4(h_lg.data() + (size_t) r * (size_t) nv, nv, use_sub ? h_sub.data() : nullptr);
                    if (g_ids[(size_t) r] != ref.ids[0] || std::fabs(g_probs[(size_t) r] - ref.probs[0]) > 2e-5f) {
                        std::fprintf(stderr, "FAIL top1 nv=%d nr=%d sub=%d r=%d: got (%d, %.7f) vs ref (%d, %.7f)\n",
                                     nv, n_rows, use_sub, r, g_ids[(size_t) r], g_probs[(size_t) r], ref.ids[0], ref.probs[0]);
                        ++fails;
                    }
                    for (int k = 0; k < 4; ++k) {
                        const int idx = (step_j + r) * 4 + k;
                        if (h_t4i[idx] != ref.ids[k] || std::fabs(h_t4p[idx] - ref.probs[k]) > 2e-5f) {
                            std::fprintf(stderr, "FAIL top4[%d] nv=%d nr=%d sub=%d r=%d: got (%d, %.7f) vs ref (%d, %.7f)\n",
                                         k, nv, n_rows, use_sub, r, h_t4i[idx], h_t4p[idx], ref.ids[k], ref.probs[k]);
                            ++fails;
                        }
                    }
                    if (n_rows == 1) {
                        int32_t td = -1;
                        ck(cudaMemcpy(&td, d_tokdst, 4, cudaMemcpyDeviceToHost), "d2h tokdst");
                        if (td != ref.ids[0] || h_mout[step_j] != ref.ids[0] || std::fabs(h_mprob[step_j] - ref.probs[0]) > 2e-5f) {
                            std::fprintf(stderr, "FAIL select nv=%d sub=%d: td=%d mout=%d mprob=%.7f ref=(%d, %.7f)\n",
                                         nv, use_sub, td, h_mout[step_j], h_mprob[step_j], ref.ids[0], ref.probs[0]);
                            ++fails;
                        }
                        std::vector<float> h_Rdst((size_t) R_stride);
                        ck(cudaMemcpy(h_Rdst.data(), d_Rdst, (size_t) R_stride * sizeof(float), cudaMemcpyDeviceToHost), "d2h Rdst");
                        if (std::memcmp(h_Rdst.data(), h_Rsrc.data() + (size_t) rsel * (size_t) R_stride, (size_t) R_stride * sizeof(float)) != 0) {
                            std::fprintf(stderr, "FAIL R_dst copy mismatch nv=%d sub=%d\n", nv, use_sub);
                            ++fails;
                        }
                    }
                }
                cudaFree(d_lg); if (d_sub) cudaFree(d_sub); cudaFree(d_ids); cudaFree(d_probs); cudaFree(d_scr);
                cudaFreeHost(h_mout); cudaFreeHost(h_mprob); cudaFreeHost(h_t4i); cudaFreeHost(h_t4p);
            }
        }
    }

    // 200-replay CUDA Graph stress test with changing logits
    {
        const int nv = 65536;
        float *d_lg = nullptr, *d_probs = nullptr;
        int32_t *d_ids = nullptr;
        void* d_scr = nullptr;
        ck(cudaMalloc(&d_lg, (size_t) nv * 4), "g malloc lg");
        ck(cudaMalloc(&d_ids, 4), "g malloc ids");
        ck(cudaMalloc(&d_probs, 4), "g malloc probs");
        const uint64_t sb = strata::kernels::mtp_topk_prob_select_scratch_bytes(1);
        ck(cudaMalloc(&d_scr, sb), "g malloc scr");
        ck(cudaMemset(d_scr, 0, sb), "g zero scr");

        cudaGraph_t gr = nullptr;
        cudaGraphExec_t ex = nullptr;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin cap");
        strata::kernels::mtp_topk_prob_select(d_lg, 1, nv, nullptr, d_Rsrc, R_stride, d_rowdev, d_Rdst, d_tokdst,
                                              d_ids, d_probs, nullptr, nullptr, nullptr, nullptr, 0, d_scr, st);
        ck(cudaStreamEndCapture(st, &gr), "end cap");
        ck(cudaGraphInstantiate(&ex, gr, 0), "inst");
        cudaGraphDestroy(gr);

        std::vector<float> h_lg((size_t) nv, -5.0f);
        for (int rep = 0; rep < 200; ++rep) {
            const int target = (rep * 997 + 13) % nv;
            h_lg[(size_t) target] = 10.0f;
            ck(cudaMemcpyAsync(d_lg, h_lg.data(), (size_t) nv * 4, cudaMemcpyHostToDevice, st), "h2d lg");
            ck(cudaGraphLaunch(ex, st), "launch");
            ck(cudaStreamSynchronize(st), "sync");
            int32_t got = -1;
            ck(cudaMemcpy(&got, d_ids, 4, cudaMemcpyDeviceToHost), "d2h");
            if (got != target) {
                std::fprintf(stderr, "FAIL graph replay %d: got %d expect %d\n", rep, got, target);
                ++fails;
                break;
            }
            h_lg[(size_t) target] = -5.0f;
        }
        cudaGraphExecDestroy(ex);
        cudaFree(d_lg); cudaFree(d_ids); cudaFree(d_probs); cudaFree(d_scr);
    }

    cudaFree(d_Rsrc); cudaFree(d_Rdst); cudaFree(d_rowdev); cudaFree(d_tokdst);
    return fails;
}

int test_zero_d2h_gate(cudaStream_t st) {
    if (!strata::kernels::mtp_zero_d2h_supported()) {
        std::printf("mtp_zero_d2h_parity: conditional graph not supported on this build, skipping gate test\n");
        return 0;
    }
    int fails = 0;
#if !defined(STRATA_USE_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 12040
    constexpr int K = 4;
    strata::kernels::MtpZeroD2hCtrl *h_ctrl = nullptr, *m_ctrl = nullptr;
    int32_t *h_out = nullptr, *m_out = nullptr;
    float *h_probs = nullptr, *m_probs = nullptr;
    strata::kernels::MtpZeroD2hState* d_st = nullptr;
    ck(cudaHostAlloc(&h_ctrl, sizeof(*h_ctrl), cudaHostAllocMapped), "alloc ctrl");
    ck(cudaHostGetDevicePointer(&m_ctrl, h_ctrl, 0), "dev ctrl");
    ck(cudaHostAlloc(&h_out, 16, cudaHostAllocMapped), "alloc out");
    ck(cudaHostGetDevicePointer(&m_out, h_out, 0), "dev out");
    ck(cudaHostAlloc(&h_probs, K * sizeof(float), cudaHostAllocMapped), "alloc probs");
    ck(cudaHostGetDevicePointer(&m_probs, h_probs, 0), "dev probs");
    ck(cudaMalloc(&d_st, sizeof(*d_st)), "malloc st");

    cudaGraph_t root = nullptr;
    ck(cudaGraphCreate(&root, 0), "create root");
    cudaGraphConditionalHandle ch[K] = {};
    for (int j = 1; j < K; ++j)
        ck(cudaGraphConditionalHandleCreate(&ch[j], root, 0, cudaGraphCondAssignDefault), "cond handle");

    cudaGraph_t g0 = nullptr;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cap 0");
    strata::kernels::mtp_zero_d2h_gate(0, (unsigned long long) ch[1], m_ctrl, d_st, m_probs + 0, m_out, m_out + 1, st);
    ck(cudaStreamEndCapture(st, &g0), "end 0");
    cudaGraphNode_t prev = nullptr;
    ck(cudaGraphAddChildGraphNode(&prev, root, nullptr, 0, g0), "add 0");
    cudaGraphDestroy(g0);

    for (int j = 1; j < K; ++j) {
        cudaGraphNodeParams p{};
        p.type = cudaGraphNodeTypeConditional;
        p.conditional.handle = ch[j];
        p.conditional.type = cudaGraphCondTypeIf;
        p.conditional.size = 1;
        cudaGraphNode_t cn = nullptr;
        ck(cudaGraphAddNode(&cn, root, &prev, 1, &p), "add cond");
        cudaGraph_t gj = nullptr;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cap j");
        strata::kernels::mtp_zero_d2h_gate(j, (j + 1 < K) ? (unsigned long long) ch[j + 1] : 0ull,
                                           m_ctrl, d_st, m_probs + j, m_out, m_out + 1, st);
        ck(cudaStreamEndCapture(st, &gj), "end j");
        cudaGraphNode_t child = nullptr;
        ck(cudaGraphAddChildGraphNode(&child, p.conditional.phGraph_out[0], nullptr, 0, gj), "add child");
        cudaGraphDestroy(gj);
        prev = cn;
    }
    cudaGraphExec_t exec = nullptr;
    ck(cudaGraphInstantiate(&exec, root, 0), "inst root");
    cudaGraphDestroy(root);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ur(0.05f, 0.99f);
    strata::spec::DraftPolicy pol(8);
    pol.set_robust(true);
    // Warm up costs & calibration realistically
    for (int r = 0; r < 25; ++r) {
        pol.observe(false, 2, r % 2, 0, 20.5 + (r % 3));
        pol.observe(false, 3, r % 3, 0, 22.8 + (r % 3));
        pol.observe(false, 4, r % 4, 0, 25.1 + (r % 3));
        pol.observe(false, 5, r % 5, 0, 27.6 + (r % 3));
        float sp[4] = {0.9f, 0.7f, 0.5f, 0.3f};
        pol.observe_probs(sp, 4, r % 5);
    }

    for (int tc = 0; tc < 120; ++tc) {
        const int mode = tc % 3;
        const int max_steps = 1 + (tc % K);
        const int cap = max_steps + 1;
        for (int j = 0; j < K; ++j) h_probs[j] = ur(rng);

        *h_ctrl = {};
        h_ctrl->max_steps = max_steps;
        h_ctrl->cap = cap;
        h_ctrl->mode = mode;
        h_ctrl->min_p = 0.5f;
        if (mode == 2) {
            pol.fill_gpu_ctrl(cap, h_ctrl->cost_ms, h_ctrl->cost_n, h_ctrl->pa_ok, h_ctrl->pa_n,
                              h_ctrl->margin, h_ctrl->fixed);
        }
        h_out[0] = 0; h_out[1] = 1;
        ck(cudaGraphLaunch(exec, st), "launch gate");
        ck(cudaStreamSynchronize(st), "sync gate");

        // CPU reference
        int exp_n = 1, exp_rows = 2;
        if (mode == 0) {
            exp_n = max_steps;
            exp_rows = std::min(cap, exp_n + 1);
        } else if (mode == 1) {
            exp_n = 1;
            for (int j = 1; j < max_steps && h_probs[j - 1] >= h_ctrl->min_p; ++j) ++exp_n;
            exp_rows = std::min(cap, exp_n + 1);
        } else {
            strata::spec::DraftPolicy::ChainState cst;
            exp_n = 1;
            bool cont = pol.keep_draft(cst, 0, h_probs[0], cap);
            for (int j = 1; j < max_steps && cont; ++j) {
                ++exp_n;
                cont = pol.keep_draft(cst, j, h_probs[j], cap);
            }
            exp_rows = cst.best_rows;
        }
        if (h_out[0] != exp_n || (mode == 2 && h_out[1] != exp_rows)) {
            std::fprintf(stderr, "FAIL gate tc=%d mode=%d max_steps=%d: got (n=%d, rows=%d) vs exp (n=%d, rows=%d) probs=(%.3f,%.3f,%.3f,%.3f)\n",
                         tc, mode, max_steps, h_out[0], h_out[1], exp_n, exp_rows, h_probs[0], h_probs[1], h_probs[2], h_probs[3]);
            ++fails;
        }
    }

    cudaGraphExecDestroy(exec);
    cudaFree(d_st);
    cudaFreeHost(h_ctrl); cudaFreeHost(h_out); cudaFreeHost(h_probs);
#endif
    return fails;
}

}  // namespace

int main() {
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    int fails = 0;
    fails += test_topk_select(st);
    fails += test_zero_d2h_gate(st);
    cudaStreamDestroy(st);
    std::printf("mtp_zero_d2h_parity: %s (%d failures)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
}
