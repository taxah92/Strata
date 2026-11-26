// src/kernels/q8_volta_tc_bench.cpp - Microbenchmark for PR 5.1 Volta Q8_0 kernels:
//   1. Prefill: dequant_f16 + cuBLAS (Gemm::native baseline) vs q8_volta_prefill_gemm
//   2. Decode: native_q8_0_mmvq (baseline) vs q8_volta_decode_mmvq vs q8_volta_decode_seg

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_volta_tc.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

static void quant_q8_1_any(const float* dx, void* dxq, int n_in, int ncols, cudaStream_t st) {
    const size_t col_b = k::native_q8_1_bytes(n_in, 1);
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = std::min(8, ncols - c0);
        k::native_quantize_q8_1(dx + (size_t) c0 * n_in, static_cast<uint8_t*>(dxq) + (size_t) c0 * col_b, n_in, nc, st);
    }
}

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

void bench_prefill(cudaStream_t st) {
    struct Shape { const char* name; int64_t N, K; };
    const Shape shapes[] = {
        {"GDN qkv   2560->10240", 10240, 2560},
        {"GDN gate  2560->6144 ", 6144,  2560},
        {"GDN out   6144->2560 ", 2560,  6144},
        {"QSA q     2560->12288", 12288, 2560},
        {"QSA k/v   2560->512  ", 512,   2560},
        {"SH  g/u   2560->640  ", 640,   2560},
        {"SH  down  640->2560  ", 2560,  640},
    };
    const int64_t Ts[] = {32, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096};

    // Baseline Gemm uses dequant_f16 + cuBLAS (we call dequant_f16 + gemm.f16 directly so we can compare in one run).
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(st, 32ll << 20, err)) {
        std::fprintf(stderr, "gemm.init failed: %s\n", err.c_str());
        std::exit(1);
    }

    uint16_t* dX = nullptr;
    uint16_t* dW16 = nullptr;
    float* dY = nullptr;
    void* dW = nullptr;
    ck(cudaMalloc(&dX, (size_t) 12288 * 6144 * 2), "pf dX");
    ck(cudaMalloc(&dW16, (size_t) 12288 * 2560 * 2), "pf dW16");
    ck(cudaMalloc(&dY, (size_t) 12288 * 12288 * 4), "pf dY");
    ck(cudaMalloc(&dW, (size_t) 12288 * (2560 / 32) * 34), "pf dW");
    cudaMemset(dX, 0x3c, (size_t) 12288 * 6144 * 2);
    cudaMemset(dW, 0x21, (size_t) 12288 * (2560 / 32) * 34);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    std::printf("\n=== PREFILL BENCHMARK: dequant_f16+cuBLAS vs q8_volta_prefill_gemm (tile env=%s) ===\n",
                std::getenv("STRATA_PF_Q8_V70_TILE") ? std::getenv("STRATA_PF_Q8_V70_TILE") : "auto");
    std::printf("%-22s %6s | %8s %7s | %8s %7s | %7s\n",
                "Shape", "T", "cuBLASms", "TFLOP/s", "V70ms", "TFLOP/s", "Speedup");

    for (const auto& s : shapes) {
        for (int64_t T : Ts) {
            const int iters = (T <= 1024) ? 20 : (T <= 4096 ? 10 : 6);
            // Warmup + measure dequant_f16 + cuBLAS f16
            for (int i = 0; i < 2; ++i) {
                k::dequant_f16(8, dW, 0, s.N, s.K, dW16, st);
                gemm.f16(dX, dW16, dY, T, s.N, s.K, s.N, 0.0f);
            }
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) {
                k::dequant_f16(8, dW, 0, s.N, s.K, dW16, st);
                gemm.f16(dX, dW16, dY, T, s.N, s.K, s.N, 0.0f);
            }
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "pf base sync");
            float ms_base = 0;
            cudaEventElapsedTime(&ms_base, e0, e1);
            ms_base /= iters;

            // Warmup + measure V70 fused Q8_0
            for (int i = 0; i < 2; ++i) k::q8_volta_prefill_gemm(dX, dW, dY, T, s.N, s.K, s.N, 0.0f, st);
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) k::q8_volta_prefill_gemm(dX, dW, dY, T, s.N, s.K, s.N, 0.0f, st);
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "pf v70 sync");
            float ms_v70 = 0;
            cudaEventElapsedTime(&ms_v70, e0, e1);
            ms_v70 /= iters;

            const double flops = 2.0 * (double) T * (double) s.N * (double) s.K;
            std::printf("%-22s %6lld | %8.3f %7.1f | %8.3f %7.1f | %6.2fx\n",
                        s.name, (long long) T,
                        ms_base, flops / (ms_base * 1e9),
                        ms_v70, flops / (ms_v70 * 1e9),
                        ms_base / ms_v70);
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(dX); cudaFree(dW16); cudaFree(dY); cudaFree(dW);
}

void bench_decode(cudaStream_t st) {
    struct Bundle {
        const char* name;
        int n_in;
        std::vector<int> n_outs;
    };
    const Bundle bundles[] = {
        {"GDN qkv+gate (10240+6144)", 2560, {10240, 6144}},
        {"GDN out      (2560)      ", 6144, {2560}},
        {"QSA q+k+v    (12288+512+512)", 2560, {12288, 512, 512}},
        {"QSA k+v      (512+512)   ", 2560, {512, 512}},
        {"QSA q        (12288)     ", 2560, {12288}},
        {"SH  gate+up  (640+640)   ", 2560, {640, 640}},
        {"SH  down     (2560)      ", 640,  {2560}},
    };
    const int Ms[] = {1, 2, 4, 5, 6, 8, 10, 12, 16};
    const int iters = 40;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    std::printf("\n=== DECODE BENCHMARK: baseline native_q8_0_mmvq (pass STRATA_DEC_SEG_V70=0) vs q8_volta_decode ===\n");
    std::printf("%-28s %3s | %8s %7s | %8s %7s | %8s %7s | %7s\n",
                "Bundle", "M", "Base us", "GB/s", "V70-1 us", "GB/s", "V70-seg", "GB/s", "Speedup");

    for (const auto& b : bundles) {
        const int n_segs = (int) b.n_outs.size();
        size_t total_bytes = 0;
        for (int no : b.n_outs) total_bytes += (size_t) no * (b.n_in / 32) * 34;
        const int copies = (int) std::max<size_t>(2, std::min<size_t>(12, ((size_t) 256 << 20) / total_bytes));

        std::vector<std::vector<void*>> ws(copies, std::vector<void*>(n_segs, nullptr));
        std::vector<float*> ys(n_segs, nullptr);
        for (int c = 0; c < copies; ++c) {
            for (int s = 0; s < n_segs; ++s) {
                const size_t bytes = (size_t) b.n_outs[s] * (b.n_in / 32) * 34;
                ck(cudaMalloc(&ws[c][s], bytes), "dec w");
                cudaMemset(ws[c][s], 0x21, bytes);
            }
        }
        for (int s = 0; s < n_segs; ++s) ck(cudaMalloc(&ys[s], (size_t) 16 * b.n_outs[s] * 4), "dec y");
        float* dx = nullptr;
        void* dxq = nullptr;
        ck(cudaMalloc(&dx, (size_t) 16 * b.n_in * 4), "dec dx");
        ck(cudaMalloc(&dxq, (size_t) 16 * k::native_q8_1_bytes(b.n_in, 1)), "dec dxq");
        cudaMemset(dx, 0x3f, (size_t) 16 * b.n_in * 4);
        quant_q8_1_any(dx, dxq, b.n_in, 16, st);

        for (int m : Ms) {
            // 1. Baseline native_q8_0_mmvq (runs old dp4a when STRATA_DEC_SEG_V70=0 is set in env, else V70 single)
            auto run_base = [&](int i) {
                for (int s = 0; s < n_segs; ++s)
                    for (int c0 = 0; c0 < m; c0 += 8) {
                        const int nc = std::min(8, m - c0);
                        const size_t qcb = k::native_q8_1_bytes(b.n_in, 1);
                        k::native_q8_0_mmvq(ws[i % copies][s], static_cast<const uint8_t*>(dxq) + (size_t) c0 * qcb,
                                            ys[s] + (size_t) c0 * b.n_outs[s], b.n_in, b.n_outs[s], nc, st);
                    }
            };
            auto run_v70_single = [&](int i) {
                for (int s = 0; s < n_segs; ++s)
                    k::q8_volta_decode_mmvq(ws[i % copies][s], dxq, ys[s], b.n_in, b.n_outs[s], m, st);
            };
            auto run_v70_seg = [&](int i) {
                k::Q8VoltaSeg segs[4];
                for (int s = 0; s < n_segs; ++s) segs[s] = {ws[i % copies][s], ys[s], b.n_outs[s]};
                k::q8_volta_decode_seg(segs, n_segs, dxq, b.n_in, m, st);
            };

            auto time_fn = [&](auto&& fn) {
                for (int i = 0; i < 4; ++i) fn(i);
                cudaEventRecord(e0, st);
                for (int i = 0; i < iters; ++i) fn(i);
                cudaEventRecord(e1, st);
                ck(cudaEventSynchronize(e1), "dec sync");
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                return 1000.0 * ms / iters;
            };

            const double us_base = time_fn(run_base);
            const double us_v1 = time_fn(run_v70_single);
            const double us_seg = time_fn(run_v70_seg);
            std::printf("%-28s %3d | %8.1f %7.1f | %8.1f %7.1f | %8.1f %7.1f | %6.2fx\n",
                        b.name, m,
                        us_base, total_bytes / (us_base * 1e3),
                        us_v1, total_bytes / (us_v1 * 1e3),
                        us_seg, total_bytes / (us_seg * 1e3),
                        us_base / us_seg);
        }

        for (int c = 0; c < copies; ++c)
            for (int s = 0; s < n_segs; ++s) cudaFree(ws[c][s]);
        for (int s = 0; s < n_segs; ++s) cudaFree(ys[s]);
        cudaFree(dx); cudaFree(dxq);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool do_pf = true, do_dec = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--prefill-only") == 0) do_dec = false;
        if (std::strcmp(argv[i], "--decode-only") == 0) do_pf = false;
    }
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("q8_volta_tc_bench: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    if (do_pf) bench_prefill(st);
    if (do_dec) bench_decode(st);
    cudaStreamDestroy(st);
    return 0;
}
