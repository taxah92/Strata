// src/kernels/ple_fused_v70_parity.cpp - PR 6.3b bitwise parity, CUDA Graph replay & microbench
// for fused Volta sm_70 PLE postops (ple_norm_gate_v70, ple_bcast_norm_v70, ple_conv_add_*_v70).
#include "strata/kernels/native_ple_postops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ple_fused_v70_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int N = 2560, H = 4, D = N * H, HIST = 9;

} // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--bench") do_bench = true;

    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");

    std::mt19937 rng(6302);
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
    std::vector<float> h_nk(D), h_nq(D), h_nc(D), h_hist0((size_t)HIST * D);
    std::vector<uint16_t> h_cw((size_t)4 * D);
    for (auto& x : h_nk) x = 0.5f + ud(rng);
    for (auto& x : h_nq) x = 0.5f + ud(rng);
    for (auto& x : h_nc) x = 0.5f + ud(rng);
    for (auto& x : h_hist0) x = ud(rng);
    for (auto& x : h_cw) x = 0x3800u + (uint16_t)(rng() % 512);

    float *d_nk = nullptr, *d_nq = nullptr, *d_nc = nullptr;
    uint16_t* d_cw = nullptr;
    ck(cudaMalloc(&d_nk, D * 4), "nk"); ck(cudaMemcpy(d_nk, h_nk.data(), D * 4, cudaMemcpyHostToDevice), "cp nk");
    ck(cudaMalloc(&d_nq, D * 4), "nq"); ck(cudaMemcpy(d_nq, h_nq.data(), D * 4, cudaMemcpyHostToDevice), "cp nq");
    ck(cudaMalloc(&d_nc, D * 4), "nc"); ck(cudaMemcpy(d_nc, h_nc.data(), D * 4, cudaMemcpyHostToDevice), "cp nc");
    ck(cudaMalloc(&d_cw, 4 * D * 2), "cw"); ck(cudaMemcpy(d_cw, h_cw.data(), 4 * D * 2, cudaMemcpyHostToDevice), "cp cw");

    strata::kernels::PleWeights w{};
    w.norm_key = d_nk; w.norm_query = d_nq; w.norm_conv = d_nc; w.conv1d_f16 = d_cw;

    int fails = 0;

    // 1. Prefill native_ple_postops_batch bitwise parity across T <= 64 and T > 64
    for (int T : {1, 4, 7, 12, 64, 65, 257, 1024}) {
        std::vector<float> h_key((size_t)T * D), h_hid((size_t)T * D), h_val((size_t)T * N);
        for (auto& x : h_key) x = ud(rng);
        for (auto& x : h_hid) x = ud(rng);
        for (auto& x : h_val) x = ud(rng);
        float *d_key, *d_hid, *d_val, *d_hist, *d_qn, *d_gat, *d_gate;
        ck(cudaMalloc(&d_key, h_key.size() * 4), "k");
        ck(cudaMalloc(&d_hid, h_hid.size() * 4), "h");
        ck(cudaMalloc(&d_val, h_val.size() * 4), "v");
        ck(cudaMalloc(&d_hist, h_hist0.size() * 4), "hi");
        ck(cudaMalloc(&d_qn, h_key.size() * 4), "qn");
        ck(cudaMalloc(&d_gat, h_key.size() * 4), "gat");
        ck(cudaMalloc(&d_gate, (size_t)T * H * 4), "gt");

        auto run = [&](bool fused, std::vector<float>& out_h, std::vector<float>& out_hi) {
            strata::kernels::native_ple_set_fused_v70(fused);
            ck(cudaMemcpy(d_key, h_key.data(), h_key.size() * 4, cudaMemcpyHostToDevice), "cp k");
            ck(cudaMemcpy(d_hid, h_hid.data(), h_hid.size() * 4, cudaMemcpyHostToDevice), "cp h");
            ck(cudaMemcpy(d_val, h_val.data(), h_val.size() * 4, cudaMemcpyHostToDevice), "cp v");
            ck(cudaMemcpy(d_hist, h_hist0.data(), h_hist0.size() * 4, cudaMemcpyHostToDevice), "cp hi");
            strata::kernels::native_ple_postops_batch(d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, T, st);
            ck(cudaStreamSynchronize(st), "sync");
            out_h.resize(h_hid.size()); out_hi.resize(h_hist0.size());
            ck(cudaMemcpy(out_h.data(), d_hid, out_h.size() * 4, cudaMemcpyDeviceToHost), "d2h h");
            ck(cudaMemcpy(out_hi.data(), d_hist, out_hi.size() * 4, cudaMemcpyDeviceToHost), "d2h hi");
        };
        std::vector<float> ref_h, ref_hi, got_h, got_hi;
        run(false, ref_h, ref_hi);
        run(true, got_h, got_hi);
        if (std::memcmp(ref_h.data(), got_h.data(), ref_h.size() * 4) != 0) {
            std::fprintf(stderr, "FAIL batch hidden bitwise mismatch T=%d\n", T);
            ++fails;
        }
        if (std::memcmp(ref_hi.data(), got_hi.data(), ref_hi.size() * 4) != 0) {
            std::fprintf(stderr, "FAIL batch hist bitwise mismatch T=%d\n", T);
            ++fails;
        }
        cudaFree(d_key); cudaFree(d_hid); cudaFree(d_val); cudaFree(d_hist);
        cudaFree(d_qn); cudaFree(d_gat); cudaFree(d_gate);
    }

    // 2. Decode verify native_ple_postops_batch_snap bitwise parity + 100x CUDA Graph replay
    for (int T : {1, 5, 7, 12}) {
        std::vector<float> h_key((size_t)T * D), h_hid((size_t)T * D), h_val((size_t)T * N);
        for (auto& x : h_key) x = ud(rng);
        for (auto& x : h_hid) x = ud(rng);
        for (auto& x : h_val) x = ud(rng);
        float *d_key, *d_hid, *d_val, *d_hist, *d_qn, *d_gat, *d_gate, *d_snap;
        ck(cudaMalloc(&d_key, h_key.size() * 4), "k");
        ck(cudaMalloc(&d_hid, h_hid.size() * 4), "h");
        ck(cudaMalloc(&d_val, h_val.size() * 4), "v");
        ck(cudaMalloc(&d_hist, h_hist0.size() * 4), "hi");
        ck(cudaMalloc(&d_qn, h_key.size() * 4), "qn");
        ck(cudaMalloc(&d_gat, h_key.size() * 4), "gat");
        ck(cudaMalloc(&d_gate, (size_t)T * H * 4), "gt");
        ck(cudaMalloc(&d_snap, (size_t)T * HIST * D * 4), "sn");

        auto run = [&](bool fused, std::vector<float>& out_h, std::vector<float>& out_hi, std::vector<float>& out_sn) {
            strata::kernels::native_ple_set_fused_v70(fused);
            ck(cudaMemcpy(d_key, h_key.data(), h_key.size() * 4, cudaMemcpyHostToDevice), "cp k");
            ck(cudaMemcpy(d_hid, h_hid.data(), h_hid.size() * 4, cudaMemcpyHostToDevice), "cp h");
            ck(cudaMemcpy(d_val, h_val.data(), h_val.size() * 4, cudaMemcpyHostToDevice), "cp v");
            ck(cudaMemcpy(d_hist, h_hist0.data(), h_hist0.size() * 4, cudaMemcpyHostToDevice), "cp hi");
            strata::kernels::native_ple_postops_batch_snap(d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, T, d_snap, st);
            ck(cudaStreamSynchronize(st), "sync");
            out_h.resize(h_hid.size()); out_hi.resize(h_hist0.size()); out_sn.resize((size_t)T * HIST * D);
            ck(cudaMemcpy(out_h.data(), d_hid, out_h.size() * 4, cudaMemcpyDeviceToHost), "d2h h");
            ck(cudaMemcpy(out_hi.data(), d_hist, out_hi.size() * 4, cudaMemcpyDeviceToHost), "d2h hi");
            ck(cudaMemcpy(out_sn.data(), d_snap, out_sn.size() * 4, cudaMemcpyDeviceToHost), "d2h sn");
        };
        std::vector<float> ref_h, ref_hi, ref_sn, got_h, got_hi, got_sn;
        run(false, ref_h, ref_hi, ref_sn);
        run(true, got_h, got_hi, got_sn);
        if (std::memcmp(ref_h.data(), got_h.data(), ref_h.size() * 4) != 0) {
            std::fprintf(stderr, "FAIL batch_snap hidden bitwise mismatch T=%d\n", T); ++fails;
        }
        if (std::memcmp(ref_hi.data(), got_hi.data(), ref_hi.size() * 4) != 0) {
            std::fprintf(stderr, "FAIL batch_snap hist bitwise mismatch T=%d\n", T); ++fails;
        }
        if (std::memcmp(ref_sn.data(), got_sn.data(), ref_sn.size() * 4) != 0) {
            std::fprintf(stderr, "FAIL batch_snap snap bitwise mismatch T=%d\n", T); ++fails;
        }

        // CUDA Graph capture & 100 replays for T==7
        if (T == 7) {
            strata::kernels::native_ple_set_fused_v70(true);
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), "begin cap");
            strata::kernels::native_ple_postops_batch_snap(d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, T, d_snap, st);
            ck(cudaStreamEndCapture(st, &graph), "end cap");
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "inst");
            for (int r = 0; r < 100; ++r) {
                ck(cudaMemcpyAsync(d_key, h_key.data(), h_key.size() * 4, cudaMemcpyHostToDevice, st), "cp k");
                ck(cudaMemcpyAsync(d_hid, h_hid.data(), h_hid.size() * 4, cudaMemcpyHostToDevice, st), "cp h");
                ck(cudaMemcpyAsync(d_val, h_val.data(), h_val.size() * 4, cudaMemcpyHostToDevice, st), "cp v");
                ck(cudaMemcpyAsync(d_hist, h_hist0.data(), h_hist0.size() * 4, cudaMemcpyHostToDevice, st), "cp hi");
                ck(cudaGraphLaunch(exec, st), "launch");
            }
            ck(cudaStreamSynchronize(st), "sync graph");
            std::vector<float> gh(h_hid.size()), ghi(h_hist0.size()), gsn((size_t)T * HIST * D);
            ck(cudaMemcpy(gh.data(), d_hid, gh.size() * 4, cudaMemcpyDeviceToHost), "d2h gh");
            ck(cudaMemcpy(ghi.data(), d_hist, ghi.size() * 4, cudaMemcpyDeviceToHost), "d2h ghi");
            ck(cudaMemcpy(gsn.data(), d_snap, gsn.size() * 4, cudaMemcpyDeviceToHost), "d2h gsn");
            if (std::memcmp(ref_h.data(), gh.data(), ref_h.size() * 4) != 0 ||
                std::memcmp(ref_hi.data(), ghi.data(), ref_hi.size() * 4) != 0 ||
                std::memcmp(ref_sn.data(), gsn.data(), ref_sn.size() * 4) != 0) {
                std::fprintf(stderr, "FAIL CUDA Graph replay bitwise mismatch T=%d\n", T); ++fails;
            }
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
        }

        cudaFree(d_key); cudaFree(d_hid); cudaFree(d_val); cudaFree(d_hist);
        cudaFree(d_qn); cudaFree(d_gat); cudaFree(d_gate); cudaFree(d_snap);
    }

    // 3. Tree snap bitwise parity (T=6 branching tree)
    {
        constexpr int T = 6;
        // parent: [-1, 0, 1, 0, 3, 1]
        int32_t h_tap[T * 3] = {
            -9, -6, -3,
            -8, -5, -2,
            -7, -4, -1,
            -8, -5, -2,
            -7, -4, -1,
            -7, -4, -1
        };
        int32_t h_hsrc[T * HIST] = {
            -8,-7,-6,-5,-4,-3,-2,-1, 0,
            -7,-6,-5,-4,-3,-2,-1, 0, 1,
            -6,-5,-4,-3,-2,-1, 0, 1, 2,
            -7,-6,-5,-4,-3,-2,-1, 0, 3,
            -6,-5,-4,-3,-2,-1, 0, 3, 4,
            -6,-5,-4,-3,-2,-1, 0, 1, 5
        };
        int32_t *d_tap, *d_hsrc;
        ck(cudaMalloc(&d_tap, sizeof(h_tap)), "tap"); ck(cudaMemcpy(d_tap, h_tap, sizeof(h_tap), cudaMemcpyHostToDevice), "cp tap");
        ck(cudaMalloc(&d_hsrc, sizeof(h_hsrc)), "hsrc"); ck(cudaMemcpy(d_hsrc, h_hsrc, sizeof(h_hsrc), cudaMemcpyHostToDevice), "cp hsrc");
        std::vector<float> h_key((size_t)T * D), h_hid((size_t)T * D), h_val((size_t)T * N);
        for (auto& x : h_key) x = ud(rng);
        for (auto& x : h_hid) x = ud(rng);
        for (auto& x : h_val) x = ud(rng);
        float *d_key, *d_hid, *d_val, *d_hist, *d_qn, *d_gat, *d_gate, *d_snap;
        ck(cudaMalloc(&d_key, h_key.size() * 4), "k");
        ck(cudaMalloc(&d_hid, h_hid.size() * 4), "h");
        ck(cudaMalloc(&d_val, h_val.size() * 4), "v");
        ck(cudaMalloc(&d_hist, h_hist0.size() * 4), "hi");
        ck(cudaMalloc(&d_qn, h_key.size() * 4), "qn");
        ck(cudaMalloc(&d_gat, h_key.size() * 4), "gat");
        ck(cudaMalloc(&d_gate, (size_t)T * H * 4), "gt");
        ck(cudaMalloc(&d_snap, (size_t)T * HIST * D * 4), "sn");
        auto run = [&](bool fused, std::vector<float>& oh, std::vector<float>& os) {
            strata::kernels::native_ple_set_fused_v70(fused);
            ck(cudaMemcpy(d_key, h_key.data(), h_key.size() * 4, cudaMemcpyHostToDevice), "cp k");
            ck(cudaMemcpy(d_hid, h_hid.data(), h_hid.size() * 4, cudaMemcpyHostToDevice), "cp h");
            ck(cudaMemcpy(d_val, h_val.data(), h_val.size() * 4, cudaMemcpyHostToDevice), "cp v");
            ck(cudaMemcpy(d_hist, h_hist0.data(), h_hist0.size() * 4, cudaMemcpyHostToDevice), "cp hi");
            strata::kernels::native_ple_postops_tree_snap(d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, T, d_tap, d_hsrc, d_snap, st);
            ck(cudaStreamSynchronize(st), "sync tree");
            oh.resize(h_hid.size()); os.resize((size_t)T * HIST * D);
            ck(cudaMemcpy(oh.data(), d_hid, oh.size() * 4, cudaMemcpyDeviceToHost), "d2h h");
            ck(cudaMemcpy(os.data(), d_snap, os.size() * 4, cudaMemcpyDeviceToHost), "d2h s");
        };
        std::vector<float> rh, rs, gh, gs;
        run(false, rh, rs);
        run(true, gh, gs);
        if (std::memcmp(rh.data(), gh.data(), rh.size() * 4) != 0 ||
            std::memcmp(rs.data(), gs.data(), rs.size() * 4) != 0) {
            std::fprintf(stderr, "FAIL tree_snap bitwise mismatch\n"); ++fails;
        }
        cudaFree(d_tap); cudaFree(d_hsrc);
        cudaFree(d_key); cudaFree(d_hid); cudaFree(d_val); cudaFree(d_hist);
        cudaFree(d_qn); cudaFree(d_gat); cudaFree(d_gate); cudaFree(d_snap);
    }

    if (do_bench && fails == 0) {
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "e0"); ck(cudaEventCreate(&e1), "e1");
        auto bench = [&](const char* label, int T, bool snap, int iters) {
            float *d_key, *d_hid, *d_val, *d_hist, *d_qn, *d_gat, *d_gate, *d_snap = nullptr;
            ck(cudaMalloc(&d_key, (size_t)T * D * 4), "bk");
            ck(cudaMalloc(&d_hid, (size_t)T * D * 4), "bh");
            ck(cudaMalloc(&d_val, (size_t)T * N * 4), "bv");
            ck(cudaMalloc(&d_hist, (size_t)HIST * D * 4), "bhi");
            ck(cudaMalloc(&d_qn, (size_t)T * D * 4), "bqn");
            ck(cudaMalloc(&d_gat, (size_t)T * D * 4), "bgat");
            ck(cudaMalloc(&d_gate, (size_t)T * H * 4), "bgt");
            if (snap) ck(cudaMalloc(&d_snap, (size_t)T * HIST * D * 4), "bsn");
            ck(cudaMemset(d_key, 0, (size_t)T * D * 4), "z");
            ck(cudaMemset(d_hid, 0, (size_t)T * D * 4), "z");
            ck(cudaMemset(d_val, 0, (size_t)T * N * 4), "z");
            ck(cudaMemset(d_hist, 0, (size_t)HIST * D * 4), "z");
            auto time_arm = [&](bool fused) {
                strata::kernels::native_ple_set_fused_v70(fused);
                for (int i = 0; i < 10; ++i) {
                    if (snap) strata::kernels::native_ple_postops_batch_snap(d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, T, d_snap, st);
                    else strata::kernels::native_ple_postops_batch(d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, T, st);
                }
                ck(cudaStreamSynchronize(st), "w");
                ck(cudaEventRecord(e0, st), "r0");
                for (int i = 0; i < iters; ++i) {
                    if (snap) strata::kernels::native_ple_postops_batch_snap(d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, T, d_snap, st);
                    else strata::kernels::native_ple_postops_batch(d_key, d_hid, d_val, d_hist, w, d_qn, d_gat, d_gate, T, st);
                }
                ck(cudaEventRecord(e1, st), "r1");
                ck(cudaEventSynchronize(e1), "s1");
                float ms = 0; ck(cudaEventElapsedTime(&ms, e0, e1), "el");
                return (double)ms * 1000.0 / (double)iters;
            };
            const double base_us = time_arm(false);
            const double fused_us = time_arm(true);
            std::printf("BENCH %-16s T=%-5d: base=%7.2f us  fused=%7.2f us  speedup=%.2fx\n",
                        label, T, base_us, fused_us, base_us / fused_us);
            cudaFree(d_key); cudaFree(d_hid); cudaFree(d_val); cudaFree(d_hist);
            cudaFree(d_qn); cudaFree(d_gat); cudaFree(d_gate);
            if (d_snap) cudaFree(d_snap);
        };
        bench("decode_snap", 5, true, 500);
        bench("decode_snap", 7, true, 500);
        bench("decode_snap", 12, true, 500);
        bench("prefill_batch", 512, false, 100);
        bench("prefill_batch", 2048, false, 50);
        bench("prefill_batch", 8192, false, 20);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    cudaFree(d_nk); cudaFree(d_nq); cudaFree(d_nc); cudaFree(d_cw);
    cudaStreamDestroy(st);
    if (fails == 0) {
        std::printf("ple_fused_v70_parity: ALL PASS (batch T=1..1024 + batch_snap T=1..12 + 100x CUDA Graph + tree_snap bitwise 0 diffs)\n");
        return 0;
    }
    return 1;
}
