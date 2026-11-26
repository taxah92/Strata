// src/kernels/iq4nl_volta_tc_parity.cpp - Parity, width-invariance, CUDA Graph replay & microbenchmark
// for Volta sm_70 IQ4_NL down kernel (iq4nl_down_v70_kernel).
//
// MIT License
// Copyright (c) 2026 Strata contributors
#include "strata/kernels/iq4nl_volta_tc.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
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

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "iq4nl_volta_tc_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void fill_iq4nl_blob(uint8_t* blob, const K::NativeExpertLayout& L, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> scale_d(-0.08f, 0.08f);
    const int nb = (int) (L.n_ff / 32);
    uint8_t* dn = blob + L.down_off;
    for (int64_t r = 0; r < L.n_embd; ++r) {
        uint8_t* row = dn + (size_t) r * L.d_row;
        for (int b = 0; b < nb; ++b) {
            uint8_t* blk = row + b * 18;
            const __half dh = __float2half(scale_d(rng));
            std::memcpy(blk, &dh, 2);
            for (int j = 0; j < 16; ++j) blk[2 + j] = (uint8_t) byte_d(rng);
        }
    }
}

struct Metrics {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
};

Metrics compare(const std::vector<float>& a, const std::vector<float>& b) {
    Metrics m;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        m.max_abs = std::max(m.max_abs, std::fabs(d));
        num += d * d;
        den += (double) b[i] * (double) b[i];
    }
    m.rel_l2 = std::sqrt(num / std::max(den, 1e-30));
    return m;
}

int run_parity_shape(int64_t n_embd, int64_t n_ff, const std::vector<int>& group_counts,
                     int64_t gy, int seed, cudaStream_t s) {
    const K::NativeExpertLayout L = K::native_expert_layout(18, 20, n_embd, n_ff);
    const int G = (int) group_counts.size();
    std::vector<int32_t> start(G + 1, 0);
    for (int g = 0; g < G; ++g) start[g + 1] = start[g] + group_counts[g];
    const int NE = start[G];

    std::mt19937 rng((unsigned) seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const size_t stride = (L.bytes + 255) & ~(size_t) 255;
    std::vector<uint8_t> blobs((size_t) std::max(G, 1) * stride, 0);
    for (int g = 0; g < G; ++g) fill_iq4nl_blob(blobs.data() + (size_t) g * stride, L, rng);

    std::vector<float> h_in((size_t) std::max(NE, 1) * (size_t) n_ff);
    for (float& v : h_in) v = nd(rng);

    std::vector<int32_t> dst(NE);
    for (int i = 0; i < NE; ++i) dst[i] = i;
    std::shuffle(dst.begin(), dst.end(), rng);

    uint8_t* d_blobs = nullptr;
    float* d_h = nullptr;
    void* d_hq = nullptr;
    float* d_out = nullptr;
    unsigned long long* d_ptr = nullptr;
    int32_t *d_start = nullptr, *d_ng = nullptr, *d_dst = nullptr, *d_tok = nullptr;

    ck(cudaMalloc(&d_blobs, blobs.size()), "malloc blobs");
    ck(cudaMemcpy(d_blobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "cpy blobs");
    ck(cudaMalloc(&d_h, h_in.size() * 4), "malloc h");
    ck(cudaMemcpy(d_h, h_in.data(), h_in.size() * 4, cudaMemcpyHostToDevice), "cpy h");
    const size_t scr_bytes = K::native_expert_scratch_bytes(std::max(NE, 1), n_ff);
    ck(cudaMalloc(&d_hq, scr_bytes), "malloc scr");
    ck(cudaMalloc(&d_out, (size_t) std::max(NE, 1) * (size_t) n_embd * 4), "malloc out");
    ck(cudaMalloc(&d_ptr, (size_t) std::max(G, 1) * 8), "malloc ptr");
    ck(cudaMalloc(&d_start, (size_t) (G + 1) * 4), "malloc start");
    ck(cudaMalloc(&d_ng, 4), "malloc ng");
    ck(cudaMalloc(&d_dst, (size_t) std::max(NE, 1) * 4), "malloc dst");
    ck(cudaMalloc(&d_tok, (size_t) std::max(NE, 1) * 4), "malloc tok");

    std::vector<unsigned long long> h_ptr(std::max(G, 1), (unsigned long long) d_blobs);
    for (int g = 0; g < G; ++g) h_ptr[g] = (unsigned long long) (d_blobs + (size_t) g * stride);
    ck(cudaMemcpy(d_ptr, h_ptr.data(), h_ptr.size() * 8, cudaMemcpyHostToDevice), "cpy ptr");
    ck(cudaMemcpy(d_start, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "cpy start");
    ck(cudaMemcpy(d_ng, &G, 4, cudaMemcpyHostToDevice), "cpy ng");
    if (NE > 0) {
        ck(cudaMemcpy(d_dst, dst.data(), (size_t) NE * 4, cudaMemcpyHostToDevice), "cpy dst");
        ck(cudaMemcpy(d_tok, dst.data(), (size_t) NE * 4, cudaMemcpyHostToDevice), "cpy tok");
    }

    const size_t f = (size_t) std::max(NE, 1) * (size_t) n_ff * sizeof(float);
    const size_t fa = (f + 255) & ~(size_t) 255;
    void* hq_at_scr = (uint8_t*) d_hq + 3 * fa;
    if (NE > 0) K::quantize_q8_1_rows(d_h, NE, n_ff, hq_at_scr, s);

    // Reference: mode 8 and mode 0 with STRATA_IQ4_V70 disabled
    std::vector<float> ref8((size_t) std::max(NE, 1) * (size_t) n_embd, 0.0f);
    std::vector<float> ref0(ref8.size(), 0.0f);
    K::iq4nl_volta_tc_set_mode(0);
    K::native_expert_set_mode(8, 2);
    ck(cudaMemsetAsync(d_out, 0, ref8.size() * 4, s), "z ref8");
    if (G > 0 && NE > 0) K::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, G, NE, d_hq, d_hq, d_out, s, gy);
    ck(cudaStreamSynchronize(s), "sync ref8");
    ck(cudaMemcpy(ref8.data(), d_out, ref8.size() * 4, cudaMemcpyDeviceToHost), "cp ref8");

    K::native_expert_set_mode(0, 2);
    ck(cudaMemsetAsync(d_out, 0, ref0.size() * 4, s), "z ref0");
    if (G > 0 && NE > 0) K::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, G, NE, d_hq, d_hq, d_out, s, gy);
    ck(cudaStreamSynchronize(s), "sync ref0");
    ck(cudaMemcpy(ref0.data(), d_out, ref0.size() * 4, cudaMemcpyDeviceToHost), "cp ref0");

    // V70 kernel
    int fails = 0;
    K::iq4nl_volta_tc_set_mode(1);
    K::native_expert_set_mode(8, 2);
    ck(cudaMemsetAsync(d_out, 0, ref8.size() * 4, s), "z v70");
    if (G > 0 && NE > 0) K::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, G, NE, d_hq, d_hq, d_out, s, gy);
    ck(cudaStreamSynchronize(s), "sync v70");
    std::vector<float> got(ref8.size(), 0.0f);
    ck(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost), "cp v70");

    const Metrics m8 = compare(got, ref8);
    const Metrics m0 = compare(got, ref0);
    bool nonfinite = false;
    for (float v : got) if (!std::isfinite(v)) nonfinite = true;
    const bool ok = !nonfinite && m8.rel_l2 < 1e-6 && m8.max_abs < 1e-4 && m0.rel_l2 < 1e-6;
    std::printf("  parity %lldx%lld G=%d NE=%d gy=%lld: vs_m8 rel_l2=%.3e max_abs=%.3e | vs_m0 rel_l2=%.3e %s\n",
                (long long) n_embd, (long long) n_ff, G, NE, (long long) gy,
                m8.rel_l2, m8.max_abs, m0.rel_l2, ok ? "ok" : "FAIL");
    if (!ok) ++fails;

    // Bit-exact width-invariance across EVERY group (proves dp4a n_e<=4 and mma.m8n8k4 n_e>=5 are bit-identical!)
    if (G > 0 && NE > 0) {
        const int32_t st1[2] = {0, 1};
        const int32_t ng1 = 1, dst1 = 0;
        unsigned long long* d_p1 = nullptr;
        int32_t *d_s1 = nullptr, *d_n1 = nullptr, *d_d1 = nullptr;
        float* d_o1 = nullptr;
        void* d_hq1 = nullptr;
        ck(cudaMalloc(&d_p1, 8), "p1");
        ck(cudaMalloc(&d_s1, 8), "s1");
        ck(cudaMalloc(&d_n1, 4), "n1");
        ck(cudaMalloc(&d_d1, 4), "d1");
        ck(cudaMalloc(&d_o1, (size_t) n_embd * 4), "o1");
        ck(cudaMalloc(&d_hq1, (size_t) (n_ff / 32) * 36u), "hq1");
        ck(cudaMemcpy(d_s1, st1, 8, cudaMemcpyHostToDevice), "cp s1");
        ck(cudaMemcpy(d_n1, &ng1, 4, cudaMemcpyHostToDevice), "cp n1");
        ck(cudaMemcpy(d_d1, &dst1, 4, cudaMemcpyHostToDevice), "cp d1");

        size_t total_bit_diffs = 0;
        int checked_entries = 0;
        const size_t row_qbytes = (size_t) (n_ff / 32) * 36u;
        std::vector<float> one((size_t) n_embd);
        for (int g = 0; g < G; ++g) {
            if (group_counts[g] == 0) continue;
            ck(cudaMemcpy(d_p1, &h_ptr[g], 8, cudaMemcpyHostToDevice), "cp p1");
            for (int e = start[g]; e < start[g + 1]; ++e) {
                ck(cudaMemcpyAsync(d_hq1, (const uint8_t*) hq_at_scr + (size_t) e * row_qbytes, row_qbytes,
                                   cudaMemcpyDeviceToDevice, s), "cp hq1");
                K::iq4nl_volta_down_grouped(L, 1, d_p1, d_s1, d_n1, d_d1, d_hq1, d_o1, s);
                ck(cudaStreamSynchronize(s), "sync one");
                ck(cudaMemcpy(one.data(), d_o1, one.size() * 4, cudaMemcpyDeviceToHost), "cp one");
                const float* multi_row = got.data() + (size_t) dst[e] * (size_t) n_embd;
                for (int64_t r = 0; r < n_embd; ++r) {
                    if (std::memcmp(&one[(size_t) r], &multi_row[(size_t) r], 4) != 0) ++total_bit_diffs;
                }
                ++checked_entries;
            }
        }
        std::printf("  width-invariance (all %d entries across G=%d, dp4a<->mma): %zu bit diffs %s\n",
                    checked_entries, G, total_bit_diffs, total_bit_diffs == 0 ? "ok" : "FAIL");
        if (total_bit_diffs != 0) ++fails;
        cudaFree(d_p1); cudaFree(d_s1); cudaFree(d_n1); cudaFree(d_d1); cudaFree(d_o1); cudaFree(d_hq1);
    }

    // CUDA Graph capture + replay (ng=0 and ng=G x20)
    if (G >= 4 && NE >= 8) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "begin cap");
        K::iq4nl_volta_down_grouped(L, gy, d_ptr, d_start, d_ng, d_dst, hq_at_scr, d_out, s);
        ck(cudaStreamEndCapture(s, &graph), "end cap");
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "inst");

        int32_t zero_ng = 0;
        ck(cudaMemcpyAsync(d_ng, &zero_ng, 4, cudaMemcpyHostToDevice, s), "ng=0");
        ck(cudaMemsetAsync(d_out, 0, ref8.size() * 4, s), "z out");
        ck(cudaGraphLaunch(exec, s), "launch ng=0");
        ck(cudaMemcpyAsync(d_ng, &G, 4, cudaMemcpyHostToDevice, s), "ng=G");
        for (int rep = 0; rep < 20; ++rep) ck(cudaGraphLaunch(exec, s), "launch ng=G");
        ck(cudaStreamSynchronize(s), "sync graph");
        std::vector<float> g_out(ref8.size());
        ck(cudaMemcpy(g_out.data(), d_out, g_out.size() * 4, cudaMemcpyDeviceToHost), "cp graph");
        const bool same = std::memcmp(g_out.data(), got.data(), got.size() * 4) == 0;
        std::printf("  CUDA Graph capture+replay (ng=0 and ng=%d x20): %s\n", G, same ? "bitwise ok" : "FAIL");
        if (!same) ++fails;
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }

    K::iq4nl_volta_tc_set_mode(-1);
    K::native_expert_set_mode(-1, 0);
    cudaFree(d_blobs); cudaFree(d_h); cudaFree(d_hq); cudaFree(d_out);
    cudaFree(d_ptr); cudaFree(d_start); cudaFree(d_ng); cudaFree(d_dst); cudaFree(d_tok);
    return fails;
}

int run_bench(cudaStream_t s) {
    constexpr int64_t H = 2560, FF = 640;
    const K::NativeExpertLayout L = K::native_expert_layout(18, 20, H, FF);
    const size_t stride = (L.bytes + 255) & ~(size_t) 255;
    const size_t POOL = 128;
    std::mt19937 rng(42);
    std::vector<uint8_t> hpool(POOL * stride, 0);
    for (size_t i = 0; i < POOL; ++i) fill_iq4nl_blob(hpool.data() + i * stride, L, rng);

    uint8_t* dpool = nullptr;
    ck(cudaMalloc(&dpool, hpool.size()), "malloc pool");
    ck(cudaMemcpy(dpool, hpool.data(), hpool.size(), cudaMemcpyHostToDevice), "cpy pool");

    constexpr int MAX_G = 64, MAX_T = 16, MAX_NE = MAX_G * MAX_T;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> h_in((size_t) MAX_NE * FF);
    for (float& v : h_in) v = nd(rng);

    float *d_h = nullptr, *d_out = nullptr;
    void* d_scr = nullptr;
    unsigned long long* d_ptr = nullptr;
    int32_t *d_start = nullptr, *d_ng = nullptr, *d_dst = nullptr, *d_tok = nullptr;

    ck(cudaMalloc(&d_h, h_in.size() * 4), "d_h");
    ck(cudaMemcpy(d_h, h_in.data(), h_in.size() * 4, cudaMemcpyHostToDevice), "cp h");
    ck(cudaMalloc(&d_scr, K::native_expert_scratch_bytes(MAX_NE, FF)), "d_scr");
    ck(cudaMalloc(&d_out, (size_t) MAX_NE * H * 4), "d_out");
    ck(cudaMalloc(&d_ptr, POOL * (size_t) MAX_G * 8), "d_ptr");
    ck(cudaMalloc(&d_start, (size_t) (MAX_G + 1) * 4), "d_start");
    ck(cudaMalloc(&d_ng, 4), "d_ng");
    ck(cudaMalloc(&d_dst, (size_t) MAX_NE * 4), "d_dst");
    ck(cudaMalloc(&d_tok, (size_t) MAX_NE * 4), "d_tok");

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    std::printf("\n# IQ4_NL down microbenchmark (2560 x 640, cold pool %.0f MB, down=921.6 KB/expert)\n",
                (double) hpool.size() / 1e6);
    std::printf("%-22s | %9s %9s %9s | vs_m0   vs_m8   BW(GB/s)\n",
                "case", "mode0_us", "mode8_us", "iq4_v70");

    auto run_case = [&](const char* label, const std::vector<int>& counts) {
        const int G = (int) counts.size();
        std::vector<int32_t> st(G + 1, 0);
        for (int g = 0; g < G; ++g) st[g + 1] = st[g] + counts[g];
        const int NE = st[G];
        std::vector<int32_t> dst(NE);
        for (int i = 0; i < NE; ++i) dst[i] = i;
        std::vector<unsigned long long> hp(POOL * (size_t) G);
        for (size_t p = 0; p < POOL; ++p)
            for (int g = 0; g < G; ++g)
                hp[p * (size_t) G + (size_t) g] = (unsigned long long) (dpool + ((p * (size_t) G + (size_t) g) % POOL) * stride);
        ck(cudaMemcpy(d_ptr, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice), "cp ptrs");
        ck(cudaMemcpy(d_ng, &G, 4, cudaMemcpyHostToDevice), "cp ng");
        ck(cudaMemcpy(d_start, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "cp st");
        ck(cudaMemcpy(d_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "cp dst");
        ck(cudaMemcpy(d_tok, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "cp tok");
        const size_t fa = ((size_t) NE * FF * 4 + 255) & ~(size_t) 255;
        K::quantize_q8_1_rows(d_h, NE, FF, (uint8_t*) d_scr + 3 * fa, s);
        ck(cudaStreamSynchronize(s), "q8_1");

        auto measure = [&](int exp_m, int iq4_m) -> double {
            K::iq4nl_volta_tc_set_mode(iq4_m);
            K::native_expert_set_mode(exp_m, 2);
            const int warmup = 20, reps = 60, rounds = 5;
            for (int i = 0; i < warmup; ++i)
                K::native_expert_grouped(L, d_ptr + (size_t) (i % POOL) * (size_t) G, d_start, d_ng, d_dst, d_tok,
                                         G, NE, d_scr, d_scr, d_out, s, 0);
            std::vector<double> r(rounds);
            for (int k = 0; k < rounds; ++k) {
                cudaEventRecord(e0, s);
                for (int i = 0; i < reps; ++i)
                    K::native_expert_grouped(L, d_ptr + (size_t) ((k * reps + i) % POOL) * (size_t) G, d_start, d_ng,
                                             d_dst, d_tok, G, NE, d_scr, d_scr, d_out, s, 0);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                r[k] = 1000.0 * (double) ms / reps;
            }
            std::sort(r.begin(), r.end());
            return r[rounds / 2];
        };

        const double t0 = measure(0, 0);
        const double t8 = measure(8, 0);
        const double tv = measure(8, 1);
        const double down_bytes = (double) G * (double) (H * (FF / 32) * 18);
        const double bw = down_bytes / (tv * 1e-6) / 1e9;
        std::printf("%-22s | %9.2f %9.2f %9.2f | %5.2fx  %5.2fx  %6.1f\n",
                    label, t0, t8, tv, t0 / tv, t8 / tv, bw);
    };

    for (int G : {10, 20, 36, 48}) {
        for (int T : {1, 2, 4, 6, 8, 12}) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "G=%d T=%d", G, T);
            run_case(buf, std::vector<int>((size_t) G, T));
        }
    }
    std::vector<int> mtp5;
    for (int i = 0; i < 24; ++i) mtp5.push_back(1);
    for (int i = 0; i < 8; ++i) mtp5.push_back(2);
    for (int i = 0; i < 3; ++i) mtp5.push_back(3);
    mtp5.push_back(5);
    run_case("MTP_T5(36e/50ent)", mtp5);

    std::vector<int> df7;
    for (int i = 0; i < 24; ++i) df7.push_back(1);
    for (int i = 0; i < 10; ++i) df7.push_back(2);
    for (int i = 0; i < 4; ++i) df7.push_back(3);
    for (int i = 0; i < 2; ++i) df7.push_back(4);
    df7.push_back(5); df7.push_back(7);
    run_case("DF2_T7(42e/70ent)", df7);

    std::vector<int> df12;
    for (int i = 0; i < 22; ++i) df12.push_back(1);
    for (int i = 0; i < 14; ++i) df12.push_back(2);
    for (int i = 0; i < 8; ++i) df12.push_back(3);
    for (int i = 0; i < 6; ++i) df12.push_back(4);
    for (int i = 0; i < 3; ++i) df12.push_back(6);
    for (int i = 0; i < 2; ++i) df12.push_back(8);
    df12.push_back(10);
    run_case("DF2_T12(56e/120ent)", df12);

    K::iq4nl_volta_tc_set_mode(-1);
    K::native_expert_set_mode(-1, 0);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dpool); cudaFree(d_h); cudaFree(d_scr); cudaFree(d_out);
    cudaFree(d_ptr); cudaFree(d_start); cudaFree(d_ng); cudaFree(d_dst); cudaFree(d_tok);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool do_bench = argc >= 2 && std::strcmp(argv[1], "--bench") == 0;
    if (!K::iq4nl_volta_tc_supported()) {
        std::printf("iq4nl_volta_tc_parity: requires sm_70 Volta, skipped\n");
        return 77;
    }
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");

    int fails = 0;
    std::printf("iq4nl_volta_tc_parity: running parity suite on sm_70\n");
    fails += run_parity_shape(2560, 640, {1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 0, 19}, 12, 101, s);
    fails += run_parity_shape(2560, 640, {1, 1, 1, 1, 1, 1, 1, 1}, 4, 202, s);
    fails += run_parity_shape(2560, 640, {0, 0, 0, 0}, 4, 303, s);
    fails += run_parity_shape(2048, 512, {1, 4, 5, 8, 12}, 5, 404, s);
    fails += run_parity_shape(288, 640, {1, 4, 5, 9, 13}, 5, 505, s);
    std::printf("iq4nl_volta_tc_parity: %d failures\n", fails);

    if (fails == 0 && do_bench) {
        run_bench(s);
    }
    cudaStreamDestroy(s);
    return fails ? 1 : 0;
}
