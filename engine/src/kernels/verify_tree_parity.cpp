// src/kernels/verify_tree_parity.cpp - PR 6.2b bitwise parity & CUDA Graph stress test for Tree Verify/Commit kernels
// (GDN conv/step/commit, PLE postops tree snap, QSA sel filter, QSA KV compact, path gather).
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "verify_tree_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct TreeCase {
    const char* name;
    int T;
    int32_t parent[8];
};

const TreeCase kCases[] = {
    {"linear_5",     5, {-1, 0, 1, 2, 3}},
    {"fan_w4_root",  7, {-1, 0, 1, 0, 3, 0, 0}},          // root->1->2, root->3->4, root->5, root->6 (1 stack slot)
    {"bin_tree_7",   7, {-1, 0, 1, 1, 0, 4, 4}},          // full binary depth-3 tree (2 stack slots)
    {"beam_8_asym",  8, {-1, 0, 1, 2, 1, 0, 5, 6}},       // 0->1->{2->3, 4}, 0->5->6->7 (2 stack slots)
    {"deep_branch",  6, {-1, 0, 1, 2, 3, 2}},             // 0->1->2->{3->4, 5} (1 stack slot)
};

int test_gdn_tree(cudaStream_t st) {
    int fails = 0;
    constexpr int S = 128, HK = 16, HV = 48;
    constexpr int C = 2 * HK * S + HV * S;  // 10240
    constexpr int ZV = HV * S;              // 6144
    constexpr int ST_N = S * HV * S;
    constexpr int HIST_N = C * 3;
    constexpr float EPS = 1e-6f;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> ud(-0.25f, 0.25f);
    std::vector<float> h_state0(ST_N), h_hist0(HIST_N), h_cw(C * 4), h_gamma(S);
    for (auto& x : h_state0) x = ud(rng);
    for (auto& x : h_hist0) x = ud(rng);
    for (auto& x : h_cw) x = ud(rng);
    for (auto& x : h_gamma) x = 0.5f + ud(rng);

    float *d_state = nullptr, *d_hist = nullptr, *d_cw = nullptr, *d_gamma = nullptr;
    float *d_qkv = nullptr, *d_h = nullptr, *d_gate = nullptr, *d_beta = nullptr, *d_z = nullptr, *d_y = nullptr;
    float *d_state_ref = nullptr, *d_hist_ref = nullptr, *d_h_ref = nullptr, *d_y_ref = nullptr;
    strata::kernels::TreeVerifyMeta *h_meta = nullptr, *m_meta = nullptr;
    int32_t *d_nkeep = nullptr, *d_path = nullptr;

    ck(cudaMalloc(&d_state, (size_t)(ST_N + HIST_N) * 4), "m state");
    d_hist = d_state + ST_N;
    ck(cudaMalloc(&d_state_ref, (size_t)(ST_N + HIST_N) * 4), "m state_ref");
    d_hist_ref = d_state_ref + ST_N;
    ck(cudaMalloc(&d_cw, (size_t)C * 4 * 4), "m cw");
    ck(cudaMalloc(&d_gamma, (size_t)S * 4), "m gamma");
    ck(cudaMalloc(&d_qkv, (size_t)8 * C * 4), "m qkv");
    ck(cudaMalloc(&d_h, (size_t)8 * C * 4), "m h");
    ck(cudaMalloc(&d_h_ref, (size_t)8 * C * 4), "m h_ref");
    ck(cudaMalloc(&d_gate, (size_t)8 * HV * 4), "m gate");
    ck(cudaMalloc(&d_beta, (size_t)8 * HV * 4), "m beta");
    ck(cudaMalloc(&d_z, (size_t)8 * ZV * 4), "m z");
    ck(cudaMalloc(&d_y, (size_t)8 * ZV * 4), "m y");
    ck(cudaMalloc(&d_y_ref, (size_t)8 * ZV * 4), "m y_ref");
    ck(cudaHostAlloc(&h_meta, sizeof(*h_meta), cudaHostAllocMapped), "m meta");
    ck(cudaHostGetDevicePointer(&m_meta, h_meta, 0), "dp meta");
    ck(cudaMalloc(&d_nkeep, 4), "m nk");
    ck(cudaMalloc(&d_path, 32), "m path");

    ck(cudaMemcpy(d_cw, h_cw.data(), h_cw.size() * 4, cudaMemcpyHostToDevice), "cp cw");
    ck(cudaMemcpy(d_gamma, h_gamma.data(), h_gamma.size() * 4, cudaMemcpyHostToDevice), "cp gam");

    for (const auto& tc : kCases) {
        if (!strata::kernels::build_tree_verify_meta(tc.parent, tc.T, *h_meta)) {
            std::fprintf(stderr, "FAIL build_tree_verify_meta(%s)\n", tc.name);
            ++fails;
            continue;
        }
        std::vector<float> h_qkv((size_t)tc.T * C), h_gate((size_t)tc.T * HV), h_beta((size_t)tc.T * HV), h_z((size_t)tc.T * ZV);
        for (auto& x : h_qkv) x = ud(rng);
        for (auto& x : h_gate) x = -0.5f - std::fabs(ud(rng));
        for (auto& x : h_beta) x = 0.2f + std::fabs(ud(rng));
        for (auto& x : h_z) x = ud(rng);

        ck(cudaMemcpy(d_state, h_state0.data(), ST_N * 4, cudaMemcpyHostToDevice), "cp st");
        ck(cudaMemcpy(d_hist, h_hist0.data(), HIST_N * 4, cudaMemcpyHostToDevice), "cp hi");
        ck(cudaMemcpy(d_qkv, h_qkv.data(), h_qkv.size() * 4, cudaMemcpyHostToDevice), "cp qkv");
        ck(cudaMemcpy(d_gate, h_gate.data(), h_gate.size() * 4, cudaMemcpyHostToDevice), "cp gate");
        ck(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * 4, cudaMemcpyHostToDevice), "cp beta");
        ck(cudaMemcpy(d_z, h_z.data(), h_z.size() * 4, cudaMemcpyHostToDevice), "cp z");

        strata::kernels::gdn_conv_l2_tree(d_hist, d_qkv, d_cw, d_h, C, 2 * HK, EPS, tc.T, m_meta, st);
        strata::kernels::gdn_step_norm_tree(d_state, d_h, C, d_gate, d_beta, d_z, d_gamma, EPS, d_y, HK, HV, tc.T, m_meta, st);
        ck(cudaStreamSynchronize(st), "sync tree");

        std::vector<float> got_h((size_t)tc.T * C), got_y((size_t)tc.T * ZV);
        ck(cudaMemcpy(got_h.data(), d_h, got_h.size() * 4, cudaMemcpyDeviceToHost), "d2h h");
        ck(cudaMemcpy(got_y.data(), d_y, got_y.size() * 4, cudaMemcpyDeviceToHost), "d2h y");

        // Compare EVERY node t against linear multi-token reference along root->t path!
        for (int target = 0; target < tc.T; ++target) {
            std::vector<int32_t> path;
            for (int u = target; u >= 0; u = tc.parent[u]) path.push_back(u);
            std::reverse(path.begin(), path.end());
            const int L = (int) path.size();
            std::vector<float> p_qkv((size_t)L * C), p_gate((size_t)L * HV), p_beta((size_t)L * HV), p_z((size_t)L * ZV);
            for (int k = 0; k < L; ++k) {
                const int u = path[(size_t)k];
                std::memcpy(p_qkv.data() + (size_t)k * C, h_qkv.data() + (size_t)u * C, C * 4);
                std::memcpy(p_gate.data() + (size_t)k * HV, h_gate.data() + (size_t)u * HV, HV * 4);
                std::memcpy(p_beta.data() + (size_t)k * HV, h_beta.data() + (size_t)u * HV, HV * 4);
                std::memcpy(p_z.data() + (size_t)k * ZV, h_z.data() + (size_t)u * ZV, ZV * 4);
            }
            float *d_pq = nullptr, *d_pg = nullptr, *d_pb = nullptr, *d_pz = nullptr;
            ck(cudaMalloc(&d_pq, p_qkv.size() * 4), "pq"); ck(cudaMemcpy(d_pq, p_qkv.data(), p_qkv.size() * 4, cudaMemcpyHostToDevice), "cp pq");
            ck(cudaMalloc(&d_pg, p_gate.size() * 4), "pg"); ck(cudaMemcpy(d_pg, p_gate.data(), p_gate.size() * 4, cudaMemcpyHostToDevice), "cp pg");
            ck(cudaMalloc(&d_pb, p_beta.size() * 4), "pb"); ck(cudaMemcpy(d_pb, p_beta.data(), p_beta.size() * 4, cudaMemcpyHostToDevice), "cp pb");
            ck(cudaMalloc(&d_pz, p_z.size() * 4), "pz"); ck(cudaMemcpy(d_pz, p_z.data(), p_z.size() * 4, cudaMemcpyHostToDevice), "cp pz");

            ck(cudaMemcpy(d_state_ref, h_state0.data(), ST_N * 4, cudaMemcpyHostToDevice), "cp st ref");
            ck(cudaMemcpy(d_hist_ref, h_hist0.data(), HIST_N * 4, cudaMemcpyHostToDevice), "cp hi ref");
            strata::kernels::gdn_conv_l2_multi(d_hist_ref, d_pq, d_cw, d_h_ref, C, 2 * HK, EPS, L, st);
            strata::kernels::gdn_step_norm_multi(d_state_ref, d_h_ref, C, d_pg, d_pb, d_pz, d_gamma, EPS, d_y_ref, HK, HV, L, nullptr, st);
            ck(cudaStreamSynchronize(st), "sync ref");

            std::vector<float> ref_h(C), ref_y(ZV);
            ck(cudaMemcpy(ref_h.data(), d_h_ref + (size_t)(L - 1) * C, C * 4, cudaMemcpyDeviceToHost), "d2h rh");
            ck(cudaMemcpy(ref_y.data(), d_y_ref + (size_t)(L - 1) * ZV, ZV * 4, cudaMemcpyDeviceToHost), "d2h ry");

            if (std::memcmp(got_h.data() + (size_t)target * C, ref_h.data(), C * 4) != 0) {
                std::fprintf(stderr, "FAIL GDN conv bitwise mismatch case=%s node=%d\n", tc.name, target);
                ++fails;
            }
            if (std::memcmp(got_y.data() + (size_t)target * ZV, ref_y.data(), ZV * 4) != 0) {
                std::fprintf(stderr, "FAIL GDN step bitwise mismatch case=%s node=%d\n", tc.name, target);
                ++fails;
            }

            // Also verify gdn_commit_tree_layers against linear commit on this branch!
            int32_t nk = L;
            ck(cudaMemcpy(d_nkeep, &nk, 4, cudaMemcpyHostToDevice), "cp nk");
            ck(cudaMemcpy(d_path, path.data(), (size_t)L * 4, cudaMemcpyHostToDevice), "cp path");
            ck(cudaMemcpy(d_state, h_state0.data(), ST_N * 4, cudaMemcpyHostToDevice), "cp st2");
            ck(cudaMemcpy(d_hist, h_hist0.data(), HIST_N * 4, cudaMemcpyHostToDevice), "cp hi2");
            strata::kernels::gdn_commit_tree_layers(d_state, ST_N + HIST_N, ST_N, d_qkv, (int64_t)tc.T * C,
                                                    d_h, (int64_t)tc.T * C, d_gate, d_beta, (int64_t)tc.T * HV,
                                                    C, HK, HV, 1, d_nkeep, d_path, st);
            strata::kernels::gdn_conv_commit(d_hist_ref, d_pq, C, d_nkeep, st);
            strata::kernels::gdn_step_norm_multi(d_state_ref, d_h_ref, C, d_pg, d_pb, d_pz, d_gamma, EPS, d_y_ref, HK, HV, L, d_nkeep, st, L);
            ck(cudaStreamSynchronize(st), "sync commit");

            std::vector<float> got_st(ST_N + HIST_N), ref_st(ST_N + HIST_N);
            ck(cudaMemcpy(got_st.data(), d_state, got_st.size() * 4, cudaMemcpyDeviceToHost), "d2h gst");
            ck(cudaMemcpy(ref_st.data(), d_state_ref, ref_st.size() * 4, cudaMemcpyDeviceToHost), "d2h rst");
            if (std::memcmp(got_st.data(), ref_st.data(), got_st.size() * 4) != 0) {
                std::fprintf(stderr, "FAIL GDN commit bitwise mismatch case=%s node=%d\n", tc.name, target);
                ++fails;
            }
            cudaFree(d_pq); cudaFree(d_pg); cudaFree(d_pb); cudaFree(d_pz);
        }
    }

    // 100-replay CUDA Graph stress test with changing TreeVerifyMeta topologies!
    {
        cudaGraph_t gr = nullptr; cudaGraphExec_t ex = nullptr;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cap gdn");
        strata::kernels::gdn_conv_l2_tree(d_hist, d_qkv, d_cw, d_h, C, 2 * HK, EPS, 7, m_meta, st);
        strata::kernels::gdn_step_norm_tree(d_state, d_h, C, d_gate, d_beta, d_z, d_gamma, EPS, d_y, HK, HV, 7, m_meta, st);
        ck(cudaStreamEndCapture(st, &gr), "end gdn");
        ck(cudaGraphInstantiate(&ex, gr, 0), "inst gdn");
        cudaGraphDestroy(gr);
        for (int rep = 0; rep < 100; ++rep) {
            const auto& tc = kCases[1 + (rep % 2)]; // alternates fan_w4_root and bin_tree_7 (both T=7)
            strata::kernels::build_tree_verify_meta(tc.parent, tc.T, *h_meta);
            ck(cudaGraphLaunch(ex, st), "launch gdn graph");
            ck(cudaStreamSynchronize(st), "sync gdn graph");
        }
        cudaGraphExecDestroy(ex);
    }

    cudaFree(d_state); cudaFree(d_state_ref); cudaFree(d_cw); cudaFree(d_gamma);
    cudaFree(d_qkv); cudaFree(d_h); cudaFree(d_h_ref); cudaFree(d_gate); cudaFree(d_beta);
    cudaFree(d_z); cudaFree(d_y); cudaFree(d_y_ref); cudaFree(d_nkeep); cudaFree(d_path);
    cudaFreeHost(h_meta);
    return fails;
}

int test_ple_tree(cudaStream_t st) {
    int fails = 0;
    constexpr int D = 10240;   // 10240
    constexpr int N = 2560;     // 2560
    constexpr int H = 9;     // 9
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);

    std::vector<float> h_hist0((size_t)H * D), h_nk(D), h_nq(D), h_nc(D);
    std::vector<uint16_t> h_conv((size_t)D * 4);
    for (auto& x : h_hist0) x = ud(rng);
    for (auto& x : h_nk) x = 1.0f + 0.1f * ud(rng);
    for (auto& x : h_nq) x = 1.0f + 0.1f * ud(rng);
    for (auto& x : h_nc) x = 1.0f + 0.1f * ud(rng);
    for (auto& x : h_conv) x = 0x3800u; // 0.5 in fp16

    strata::kernels::PleWeights w{};
    float *d_nk = nullptr, *d_nq = nullptr, *d_nc = nullptr;
    uint16_t* d_conv = nullptr;
    ck(cudaMalloc(&d_nk, D * 4), "nk"); ck(cudaMemcpy(d_nk, h_nk.data(), D * 4, cudaMemcpyHostToDevice), "cp nk");
    ck(cudaMalloc(&d_nq, D * 4), "nq"); ck(cudaMemcpy(d_nq, h_nq.data(), D * 4, cudaMemcpyHostToDevice), "cp nq");
    ck(cudaMalloc(&d_nc, D * 4), "nc"); ck(cudaMemcpy(d_nc, h_nc.data(), D * 4, cudaMemcpyHostToDevice), "cp nc");
    ck(cudaMalloc(&d_conv, D * 4 * 2), "conv"); ck(cudaMemcpy(d_conv, h_conv.data(), D * 4 * 2, cudaMemcpyHostToDevice), "cp conv");
    w.norm_key = d_nk; w.norm_query = d_nq; w.norm_conv = d_nc; w.conv1d_f16 = d_conv;

    const auto& tc = kCases[3]; // beam_8_asym (T=8)
    strata::kernels::TreeVerifyMeta *h_meta = nullptr, *m_meta = nullptr;
    ck(cudaHostAlloc(&h_meta, sizeof(*h_meta), cudaHostAllocMapped), "ple meta");
    ck(cudaHostGetDevicePointer(&m_meta, h_meta, 0), "ple dp");
    strata::kernels::build_tree_verify_meta(tc.parent, tc.T, *h_meta);

    std::vector<float> h_key((size_t)tc.T * D), h_hid((size_t)tc.T * D), h_val((size_t)tc.T * N);
    for (auto& x : h_key) x = ud(rng);
    for (auto& x : h_hid) x = ud(rng);
    for (auto& x : h_val) x = ud(rng);

    float *d_key = nullptr, *d_hid = nullptr, *d_val = nullptr, *d_hist = nullptr, *d_qn = nullptr, *d_gat = nullptr, *d_gate = nullptr, *d_snap = nullptr;
    ck(cudaMalloc(&d_key, h_key.size() * 4), "k"); ck(cudaMemcpy(d_key, h_key.data(), h_key.size() * 4, cudaMemcpyHostToDevice), "cp k");
    ck(cudaMalloc(&d_hid, h_hid.size() * 4), "h"); ck(cudaMemcpy(d_hid, h_hid.data(), h_hid.size() * 4, cudaMemcpyHostToDevice), "cp h");
    ck(cudaMalloc(&d_val, h_val.size() * 4), "v"); ck(cudaMemcpy(d_val, h_val.data(), h_val.size() * 4, cudaMemcpyHostToDevice), "cp v");
    ck(cudaMalloc(&d_hist, h_hist0.size() * 4), "hi"); ck(cudaMemcpy(d_hist, h_hist0.data(), h_hist0.size() * 4, cudaMemcpyHostToDevice), "cp hi");
    ck(cudaMalloc(&d_qn, h_key.size() * 4), "qn");
    ck(cudaMalloc(&d_gat, h_key.size() * 4), "gat");
    ck(cudaMalloc(&d_gate, (size_t)tc.T * 4 * 4), "gt");
    ck(cudaMalloc(&d_snap, (size_t)tc.T * H * D * 4), "sn");

    strata::kernels::native_ple_postops_tree_snap(
        d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, tc.T,
        &m_meta->ple_tap[0][0], &m_meta->ple_hist_src[0][0], d_snap, st);
    ck(cudaStreamSynchronize(st), "sync ple tree");

    std::vector<float> got_hid(h_hid.size()), got_snap((size_t)tc.T * H * D);
    ck(cudaMemcpy(got_hid.data(), d_hid, got_hid.size() * 4, cudaMemcpyDeviceToHost), "d2h ph");
    ck(cudaMemcpy(got_snap.data(), d_snap, got_snap.size() * 4, cudaMemcpyDeviceToHost), "d2h ps");

    for (int target = 0; target < tc.T; ++target) {
        std::vector<int32_t> path;
        for (int u = target; u >= 0; u = tc.parent[u]) path.push_back(u);
        std::reverse(path.begin(), path.end());
        const int L = (int) path.size();
        std::vector<float> pk((size_t)L * D), ph((size_t)L * D), pv((size_t)L * N);
        for (int k = 0; k < L; ++k) {
            const int u = path[(size_t)k];
            std::memcpy(pk.data() + (size_t)k * D, h_key.data() + (size_t)u * D, D * 4);
            std::memcpy(ph.data() + (size_t)k * D, h_hid.data() + (size_t)u * D, D * 4);
            std::memcpy(pv.data() + (size_t)k * N, h_val.data() + (size_t)u * N, N * 4);
        }
        ck(cudaMemcpy(d_key, pk.data(), pk.size() * 4, cudaMemcpyHostToDevice), "cp pk");
        ck(cudaMemcpy(d_hid, ph.data(), ph.size() * 4, cudaMemcpyHostToDevice), "cp ph");
        ck(cudaMemcpy(d_val, pv.data(), pv.size() * 4, cudaMemcpyHostToDevice), "cp pv");
        ck(cudaMemcpy(d_hist, h_hist0.data(), h_hist0.size() * 4, cudaMemcpyHostToDevice), "cp phi");
        strata::kernels::native_ple_postops_batch_snap(d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, L, d_snap, st);
        ck(cudaStreamSynchronize(st), "sync ple ref");

        std::vector<float> ref_hid(D), ref_snap((size_t)H * D);
        ck(cudaMemcpy(ref_hid.data(), d_hid + (size_t)(L - 1) * D, D * 4, cudaMemcpyDeviceToHost), "d2h rph");
        ck(cudaMemcpy(ref_snap.data(), d_snap + (size_t)(L - 1) * H * D, (size_t)H * D * 4, cudaMemcpyDeviceToHost), "d2h rps");

        if (std::memcmp(got_hid.data() + (size_t)target * D, ref_hid.data(), D * 4) != 0) {
            std::fprintf(stderr, "FAIL PLE hidden bitwise mismatch node=%d\n", target);
            ++fails;
        }
        if (std::memcmp(got_snap.data() + (size_t)target * H * D, ref_snap.data(), (size_t)H * D * 4) != 0) {
            std::fprintf(stderr, "FAIL PLE snap bitwise mismatch node=%d\n", target);
            ++fails;
        }
    }
    cudaFree(d_nk); cudaFree(d_nq); cudaFree(d_nc); cudaFree(d_conv);
    cudaFree(d_key); cudaFree(d_hid); cudaFree(d_val); cudaFree(d_hist);
    cudaFree(d_qn); cudaFree(d_gat); cudaFree(d_gate); cudaFree(d_snap);
    cudaFreeHost(h_meta);
    return fails;
}

int test_qsa_filter_and_compact(cudaStream_t st) {
    int fails = 0;
    const auto& tc = kCases[3]; // beam_8_asym: 0->1->{2->3, 4}, 0->5->6->7
    strata::kernels::TreeVerifyMeta *h_meta = nullptr, *m_meta = nullptr;
    ck(cudaHostAlloc(&h_meta, sizeof(*h_meta), cudaHostAllocMapped), "qf meta");
    ck(cudaHostGetDevicePointer(&m_meta, h_meta, 0), "qf dp");
    strata::kernels::build_tree_verify_meta(tc.parent, tc.T, *h_meta);

    // 1. Test qsa_tree_filter_sel (short ctx p0=20 < cap=64, and long ctx p0=5000 > cap=64)
    constexpr int cap = 64;
    int32_t *d_sel = nullptr, *d_steps = nullptr;
    ck(cudaMalloc(&d_sel, (size_t)tc.T * cap * 4), "sel");
    ck(cudaMalloc(&d_steps, (size_t)tc.T * strata::kernels::kStepCount * 4), "steps");

    for (int p0 : {20, 5000}) {
        std::vector<int32_t> h_sel((size_t)tc.T * cap, 0);
        std::vector<int32_t> h_st((size_t)tc.T * strata::kernels::kStepCount, 0);
        for (int t = 0; t < tc.T; ++t) {
            h_st[(size_t)t * 4 + 0] = p0 + t;
            h_st[(size_t)t * 4 + 1] = p0 + t + 1;
            h_st[(size_t)t * 4 + 2] = (p0 + t + 1) / 4;
            int w = std::min(p0 + t + 1, cap);
            h_st[(size_t)t * 4 + 3] = w;
            if (p0 < cap) {
                for (int i = 0; i < w; ++i) h_sel[(size_t)t * cap + i] = i;
            } else {
                // Sorted blocks before p0, then window tail [p0..p0+t]
                const int tail = t + 1;
                const int pre = w - tail;
                for (int i = 0; i < pre; ++i) h_sel[(size_t)t * cap + i] = i * 50;
                for (int i = 0; i < tail; ++i) h_sel[(size_t)t * cap + pre + i] = p0 + i;
            }
        }
        ck(cudaMemcpy(d_sel, h_sel.data(), h_sel.size() * 4, cudaMemcpyHostToDevice), "cp sel");
        ck(cudaMemcpy(d_steps, h_st.data(), h_st.size() * 4, cudaMemcpyHostToDevice), "cp st");
        strata::kernels::qsa_tree_filter_sel(d_sel, d_steps, tc.T, cap, m_meta, st);
        ck(cudaStreamSynchronize(st), "sync qf");
        ck(cudaMemcpy(h_sel.data(), d_sel, h_sel.size() * 4, cudaMemcpyDeviceToHost), "d2h sel");
        ck(cudaMemcpy(h_st.data(), d_steps, h_st.size() * 4, cudaMemcpyDeviceToHost), "d2h st");

        for (int t = 0; t < tc.T; ++t) {
            const int w = h_st[(size_t)t * 4 + 3];
            // Check strictly ascending and exact ancestor set for >= p0
            uint32_t seen_mask = 0;
            for (int i = 0; i < w; ++i) {
                const int c = h_sel[(size_t)t * cap + i];
                if (i > 0 && c <= h_sel[(size_t)t * cap + i - 1]) {
                    std::fprintf(stderr, "FAIL qsa_tree_filter_sel not ascending p0=%d t=%d\n", p0, t);
                    ++fails;
                }
                if (c >= p0) seen_mask |= (1u << (c - p0));
            }
            if (seen_mask != h_meta->anc_mask[t]) {
                std::fprintf(stderr, "FAIL qsa_tree_filter_sel mask p0=%d t=%d got=0x%x exp=0x%x\n",
                             p0, t, seen_mask, h_meta->anc_mask[t]);
                ++fails;
            }
        }
    }
    cudaFree(d_sel); cudaFree(d_steps);

    // 2. Test qsa_kv_compact_tree_layers (k8v4 hybrid + host copy across page boundary!)
    constexpr int PS = 16, NKV = 4, HD = 256, NPAGES = 4;
    const int k_bytes = HD, v_bytes = (HD / 32) * 18, sc_bytes = (HD / 64) * 2;
    const int tot_rows = NPAGES * NKV * PS;
    std::vector<uint8_t> hk(tot_rows * k_bytes), hv(tot_rows * v_bytes), hsc(tot_rows * sc_bytes);
    for (size_t i = 0; i < hk.size(); ++i) hk[i] = (uint8_t)((i * 31 + 7) & 0xff);
    for (size_t i = 0; i < hv.size(); ++i) hv[i] = (uint8_t)((i * 17 + 3) & 0xff);
    for (size_t i = 0; i < hsc.size(); ++i) hsc[i] = (uint8_t)((i * 13 + 1) & 0xff);

    uint8_t *dk = nullptr, *dv = nullptr, *dsc = nullptr, *d_hk = nullptr, *d_hv = nullptr, *d_hsc = nullptr;
    int32_t *d_pt = nullptr, *d_p0 = nullptr, *d_nk = nullptr, *d_path = nullptr, *d_nc = nullptr;
    ck(cudaMalloc(&dk, hk.size()), "dk"); ck(cudaMemcpy(dk, hk.data(), hk.size(), cudaMemcpyHostToDevice), "cp dk");
    ck(cudaMalloc(&dv, hv.size()), "dv"); ck(cudaMemcpy(dv, hv.data(), hv.size(), cudaMemcpyHostToDevice), "cp dv");
    ck(cudaMalloc(&dsc, hsc.size()), "dsc"); ck(cudaMemcpy(dsc, hsc.data(), hsc.size(), cudaMemcpyHostToDevice), "cp dsc");
    ck(cudaMalloc(&d_hk, hk.size()), "dhk"); ck(cudaMemcpy(d_hk, hk.data(), hk.size(), cudaMemcpyHostToDevice), "cp dhk");
    ck(cudaMalloc(&d_hv, hv.size()), "dhv"); ck(cudaMemcpy(d_hv, hv.data(), hv.size(), cudaMemcpyHostToDevice), "cp dhv");
    ck(cudaMalloc(&d_hsc, hsc.size()), "dhsc"); ck(cudaMemcpy(d_hsc, hsc.data(), hsc.size(), cudaMemcpyHostToDevice), "cp dhsc");
    int32_t pt[NPAGES] = {2, 0, 3, 1}; // scrambled page table
    ck(cudaMalloc(&d_pt, sizeof(pt)), "pt"); ck(cudaMemcpy(d_pt, pt, sizeof(pt), cudaMemcpyHostToDevice), "cp pt");
    int32_t p0 = 14; // crosses page boundary 15 -> 16!
    int32_t path[4] = {0, 5, 6, 7};
    int32_t nk = 4, nc = 1;
    ck(cudaMalloc(&d_p0, 4), "p0"); ck(cudaMemcpy(d_p0, &p0, 4, cudaMemcpyHostToDevice), "cp p0");
    ck(cudaMalloc(&d_nk, 4), "nk"); ck(cudaMemcpy(d_nk, &nk, 4, cudaMemcpyHostToDevice), "cp nk");
    ck(cudaMalloc(&d_path, 16), "path"); ck(cudaMemcpy(d_path, path, 16, cudaMemcpyHostToDevice), "cp path");
    ck(cudaMalloc(&d_nc, 4), "nc"); ck(cudaMemcpy(d_nc, &nc, 4, cudaMemcpyHostToDevice), "cp nc");

    strata::kernels::QsaKvCompactLayers L{};
    L.n = 1;
    L.layer[0].kv_mode = 3; // k8v4 hybrid
    L.layer[0].n_kv_heads = NKV; L.layer[0].head_dim = HD; L.layer[0].page_size = PS;
    L.layer[0].page_table = d_pt;
    L.layer[0].k_pool = dk; L.layer[0].v_pool = dv; L.layer[0].k_scale = (uint16_t*)dsc;
    L.layer[0].h_k_pool = d_hk; L.layer[0].h_v_pool = d_hv; L.layer[0].h_k_scale = (uint16_t*)d_hsc;
    strata::kernels::qsa_kv_compact_tree_layers(L, d_p0, d_nk, d_path, d_nc, st);
    ck(cudaStreamSynchronize(st), "sync compact");

    std::vector<uint8_t> gk(hk.size()), gv(hv.size()), gsc(hsc.size()), ghk(hk.size()), ghv(hv.size());
    ck(cudaMemcpy(gk.data(), dk, gk.size(), cudaMemcpyDeviceToHost), "d2h gk");
    ck(cudaMemcpy(gv.data(), dv, gv.size(), cudaMemcpyDeviceToHost), "d2h gv");
    ck(cudaMemcpy(gsc.data(), dsc, gsc.size(), cudaMemcpyDeviceToHost), "d2h gsc");
    ck(cudaMemcpy(ghk.data(), d_hk, ghk.size(), cudaMemcpyDeviceToHost), "d2h ghk");
    ck(cudaMemcpy(ghv.data(), d_hv, ghv.size(), cudaMemcpyDeviceToHost), "d2h ghv");

    for (int k = 0; k < nk; ++k) {
        for (int h = 0; h < NKV; ++h) {
            const int sp = p0 + path[k], dp = p0 + k;
            const int srow = (pt[sp / PS] * NKV + h) * PS + (sp % PS);
            const int drow = (pt[dp / PS] * NKV + h) * PS + (dp % PS);
            if (std::memcmp(gk.data() + drow * k_bytes, hk.data() + srow * k_bytes, k_bytes) != 0 ||
                std::memcmp(gv.data() + drow * v_bytes, hv.data() + srow * v_bytes, v_bytes) != 0 ||
                std::memcmp(gsc.data() + drow * sc_bytes, hsc.data() + srow * sc_bytes, sc_bytes) != 0) {
                std::fprintf(stderr, "FAIL KV GPU compact mismatch k=%d h=%d\n", k, h);
                ++fails;
            }
            const int hsrow = ((sp / PS) * NKV + h) * PS + (sp % PS);
            const int hdrow = ((dp / PS) * NKV + h) * PS + (dp % PS);
            if (std::memcmp(ghk.data() + hdrow * k_bytes, hk.data() + hsrow * k_bytes, k_bytes) != 0 ||
                std::memcmp(ghv.data() + hdrow * v_bytes, hv.data() + hsrow * v_bytes, v_bytes) != 0) {
                std::fprintf(stderr, "FAIL KV Host compact mismatch k=%d h=%d\n", k, h);
                ++fails;
            }
        }
    }

    cudaFree(dk); cudaFree(dv); cudaFree(dsc); cudaFree(d_hk); cudaFree(d_hv); cudaFree(d_hsc);
    cudaFree(d_pt); cudaFree(d_p0); cudaFree(d_nk); cudaFree(d_path); cudaFree(d_nc);
    cudaFreeHost(h_meta);
    return fails;
}

}  // namespace

int main() {
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    int fails = 0;
    fails += test_gdn_tree(st);
    fails += test_ple_tree(st);
    fails += test_qsa_filter_and_compact(st);
    cudaStreamDestroy(st);
    std::printf("verify_tree_parity: %s (%d failures)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
}
