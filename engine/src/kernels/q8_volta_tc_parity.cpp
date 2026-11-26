// src/kernels/q8_volta_tc_parity.cpp - Parity, bitwise multi-column, segmented, and CUDA Graph tests
// for Volta (sm_70) Q8_0 FP16 Tensor Core kernels (PR 5.1).
//
// Usage: build-sm70/q8_volta_tc_parity [--quick]

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_volta_tc.hpp"

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

namespace k = strata::kernels;

namespace {

static void quant_q8_1_any(const float* dx, void* dxq, int n_in, int ncols, cudaStream_t st) {
    const size_t col_b = k::native_q8_1_bytes(n_in, 1);
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = std::min(8, ncols - c0);
        k::native_quantize_q8_1(dx + (size_t) c0 * n_in, static_cast<uint8_t*>(dxq) + (size_t) c0 * col_b, n_in, nc, st);
    }
}

struct Q80Block {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(Q80Block) == 34);

struct Q81Block {
    uint16_t d;
    uint16_t s;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

bool ck(cudaError_t e, const char* w) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
    return false;
}

float fp16_to_f32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

uint16_t f32_to_fp16(float f) {
    __half h = __float2half(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

std::vector<Q80Block> make_q8_0(int N, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qd(-128, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.04f);
    const int nb = K / 32;
    std::vector<Q80Block> out((size_t) N * nb);
    for (int r = 0; r < N; ++r) {
        for (int b = 0; b < nb; ++b) {
            Q80Block& blk = out[(size_t) r * nb + b];
            const float sign = (rng() & 1u) ? 1.0f : -1.0f;
            blk.d = f32_to_fp16(sign * sd(rng));
            for (int i = 0; i < 32; ++i) blk.qs[i] = (int8_t) qd(rng);
        }
    }
    return out;
}

double rel_l2(const std::vector<float>& got, const std::vector<double>& ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) return 1e30;
        const double d = (double) got[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

int test_prefill_case(int T, int N, int K, int ldy, float beta, uint32_t seed, cudaStream_t st) {
    auto w = make_q8_0(N, K, seed);
    std::mt19937 rng(seed + 1000u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> xh((size_t) T * K);
    std::vector<float> xf((size_t) T * K);
    for (size_t i = 0; i < xh.size(); ++i) {
        const float v = std::max(-8.0f, std::min(8.0f, nd(rng)));
        xh[i] = f32_to_fp16(v);
        xf[i] = fp16_to_f32(xh[i]);
    }
    const int ldy_eff = ldy > 0 ? ldy : N;
    std::vector<float> y_init((size_t) T * ldy_eff, 0.0f);
    for (auto& v : y_init) v = nd(rng);

    const int nb = K / 32;
    std::vector<double> ref((size_t) T * N, 0.0);
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int b = 0; b < nb; ++b) {
                const Q80Block& blk = w[(size_t) n * nb + b];
                const double dw = (double) fp16_to_f32(blk.d);
                double s = 0.0;
                for (int i = 0; i < 32; ++i) s += (double) blk.qs[i] * (double) xf[(size_t) t * K + b * 32 + i];
                acc += dw * s;
            }
            if (beta != 0.0f) acc += (double) beta * (double) y_init[(size_t) t * ldy_eff + n];
            ref[(size_t) t * N + n] = acc;
        }
    }

    void* dw = nullptr;
    uint16_t* dx = nullptr;
    float* dy = nullptr;
    if (!ck(cudaMalloc(&dw, w.size() * sizeof(Q80Block)), "pf dw") ||
        !ck(cudaMalloc(&dx, xh.size() * sizeof(uint16_t)), "pf dx") ||
        !ck(cudaMalloc(&dy, (size_t) T * ldy_eff * sizeof(float)), "pf dy")) {
        return 1;
    }
    cudaMemcpy(dw, w.data(), w.size() * sizeof(Q80Block), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, xh.data(), xh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (beta == 0.0f) cudaMemset(dy, 0xff, (size_t) T * ldy_eff * sizeof(float));   // NaN poison guard
    else cudaMemcpy(dy, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);

    const bool launched = k::q8_volta_prefill_gemm(dx, dw, dy, T, N, K, ldy, beta, st);
    if (!launched || !ck(cudaStreamSynchronize(st), "pf sync")) {
        std::printf("  FAIL prefill T=%d N=%d K=%d ldy=%d beta=%.1f: launch=%d\n", T, N, K, ldy, beta, (int) launched);
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
        return 1;
    }
    std::vector<float> y_got_raw((size_t) T * ldy_eff);
    cudaMemcpy(y_got_raw.data(), dy, y_got_raw.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> y_got((size_t) T * N);
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < N; ++n) y_got[(size_t) t * N + n] = y_got_raw[(size_t) t * ldy_eff + n];

    const double err = rel_l2(y_got, ref);
    const int fail = (err < 5e-4) ? 0 : 1;
    std::printf("  prefill T=%-4d N=%-5d K=%-4d ldy=%-5d beta=%.1f: rel_l2=%.2e %s\n",
                T, N, K, ldy_eff, beta, err, fail ? "FAIL" : "ok");
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    return fail;
}

int test_decode_case(int n_in, const std::vector<int>& n_outs, const std::vector<int>& ms, uint32_t seed,
                     bool graph_check, cudaStream_t st) {
    const int n_segs = (int) n_outs.size();
    const int max_m = *std::max_element(ms.begin(), ms.end());
    const int nb = n_in / 32;

    std::vector<std::vector<Q80Block>> ws(n_segs);
    std::vector<void*> dws(n_segs, nullptr);
    std::vector<float*> dys(n_segs, nullptr);
    for (int s = 0; s < n_segs; ++s) {
        ws[s] = make_q8_0(n_outs[s], n_in, seed + (uint32_t) s * 17u);
        if (!ck(cudaMalloc(&dws[s], ws[s].size() * sizeof(Q80Block)), "dec dw") ||
            !ck(cudaMalloc(&dys[s], (size_t) max_m * n_outs[s] * sizeof(float)), "dec dy"))
            return 1;
        cudaMemcpy(dws[s], ws[s].data(), ws[s].size() * sizeof(Q80Block), cudaMemcpyHostToDevice);
    }

    std::mt19937 rng(seed + 5000u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> xf((size_t) max_m * n_in);
    for (auto& v : xf) v = nd(rng);
    for (int j = 0; j < max_m; ++j) xf[(size_t) j * n_in + (size_t) (j * 29 % n_in)] *= 10.0f;

    float* dx = nullptr;
    void* dxq = nullptr;
    if (!ck(cudaMalloc(&dx, xf.size() * sizeof(float)), "dec dx") ||
        !ck(cudaMalloc(&dxq, (size_t) max_m * k::native_q8_1_bytes(n_in, 1)), "dec dxq"))
        return 1;
    cudaMemcpy(dx, xf.data(), xf.size() * sizeof(float), cudaMemcpyHostToDevice);
    quant_q8_1_any(dx, dxq, n_in, max_m, st);
    cudaStreamSynchronize(st);

    std::vector<Q81Block> xq_host((size_t) max_m * nb);
    cudaMemcpy(xq_host.data(), dxq, xq_host.size() * sizeof(Q81Block), cudaMemcpyDeviceToHost);

    // Exact FP64 Q8_0 x Q8_1 reference.
    std::vector<std::vector<double>> refs(n_segs);
    for (int s = 0; s < n_segs; ++s) {
        refs[s].assign((size_t) max_m * n_outs[s], 0.0);
        for (int c = 0; c < max_m; ++c) {
            for (int r = 0; r < n_outs[s]; ++r) {
                double acc = 0.0;
                for (int b = 0; b < nb; ++b) {
                    const Q80Block& wb = ws[s][(size_t) r * nb + b];
                    const Q81Block& xb = xq_host[(size_t) c * nb + b];
                    int sumi = 0;
                    for (int i = 0; i < 32; ++i) sumi += (int) wb.qs[i] * (int) xb.qs[i];
                    acc += (double) fp16_to_f32(wb.d) * (double) fp16_to_f32(xb.d) * (double) sumi;
                }
                refs[s][(size_t) c * n_outs[s] + r] = acc;
            }
        }
    }

    int fails = 0;
    // Reference M=1 single-column outputs for bitwise multi-column verification.
    std::vector<std::vector<float>> ref_m1(n_segs);
    for (int s = 0; s < n_segs; ++s) {
        ref_m1[s].resize((size_t) max_m * n_outs[s]);
        for (int c = 0; c < max_m; ++c) {
            const k::Q8VoltaSeg seg{dws[s], dys[s], n_outs[s]};
            const Q81Block* xcol = static_cast<const Q81Block*>(dxq) + (size_t) c * nb;
            cudaMemset(dys[s], 0xff, (size_t) n_outs[s] * sizeof(float));
            if (!k::q8_volta_decode_seg(&seg, 1, xcol, n_in, 1, st)) { ++fails; break; }
            cudaStreamSynchronize(st);
            cudaMemcpy(ref_m1[s].data() + (size_t) c * n_outs[s], dys[s], (size_t) n_outs[s] * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
    }

    double worst_err = 0.0;
    bool all_bitwise = true;
    for (int m : ms) {
        k::Q8VoltaSeg segs[4];
        for (int s = 0; s < n_segs; ++s) {
            cudaMemset(dys[s], 0xff, (size_t) m * n_outs[s] * sizeof(float));
            segs[s] = {dws[s], dys[s], n_outs[s]};
        }
        if (!k::q8_volta_decode_seg(segs, n_segs, dxq, n_in, m, st) || !ck(cudaStreamSynchronize(st), "dec sync")) {
            ++fails;
            continue;
        }
        for (int s = 0; s < n_segs; ++s) {
            std::vector<float> got((size_t) m * n_outs[s]);
            cudaMemcpy(got.data(), dys[s], got.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::vector<double> rr(refs[s].begin(), refs[s].begin() + (ptrdiff_t) got.size());
            worst_err = std::max(worst_err, rel_l2(got, rr));
            if (!(worst_err < 1e-5)) ++fails;
            // Bitwise equality only guaranteed when n_segs==1 (same BR selection as single-seg M=1).
            if (n_segs == 1 && std::memcmp(got.data(), ref_m1[s].data(), got.size() * sizeof(float)) != 0)
                all_bitwise = false;
        }
    }
    if (n_segs == 1 && !all_bitwise) ++fails;

    // Optional CUDA Graph capture & replay with mutated inputs.
    bool graph_ok = true;
    if (graph_check) {
        for (int gm : {5, 12}) {
            if (gm > max_m) continue;
            k::Q8VoltaSeg segs[4];
            for (int s = 0; s < n_segs; ++s) segs[s] = {dws[s], dys[s], n_outs[s]};
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            if (!ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), "begin cap")) { graph_ok = false; break; }
            k::q8_volta_decode_seg(segs, n_segs, dxq, n_in, gm, st);
            if (!ck(cudaStreamEndCapture(st, &graph), "end cap") ||
                !ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "inst")) {
                graph_ok = false;
                if (graph) cudaGraphDestroy(graph);
                break;
            }
            for (int rep = 0; rep < 3; ++rep) {
                for (auto& v : xf) v = nd(rng);
                cudaMemcpy(dx, xf.data(), (size_t) gm * n_in * sizeof(float), cudaMemcpyHostToDevice);
                quant_q8_1_any(dx, dxq, n_in, gm, st);
                cudaGraphLaunch(exec, st);
                cudaStreamSynchronize(st);
                std::vector<std::vector<float>> g_out(n_segs);
                for (int s = 0; s < n_segs; ++s) {
                    g_out[s].resize((size_t) gm * n_outs[s]);
                    cudaMemcpy(g_out[s].data(), dys[s], g_out[s].size() * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemset(dys[s], 0, g_out[s].size() * sizeof(float));
                }
                k::q8_volta_decode_seg(segs, n_segs, dxq, n_in, gm, st);
                cudaStreamSynchronize(st);
                for (int s = 0; s < n_segs; ++s) {
                    std::vector<float> e_out((size_t) gm * n_outs[s]);
                    cudaMemcpy(e_out.data(), dys[s], e_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    if (std::memcmp(g_out[s].data(), e_out.data(), e_out.size() * sizeof(float)) != 0) graph_ok = false;
                }
            }
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
        }
        if (!graph_ok) ++fails;
    }

    std::string ns_str;
    for (int s = 0; s < n_segs; ++s) { if (s) ns_str += "+"; ns_str += std::to_string(n_outs[s]); }
    std::printf("  decode n_in=%-4d n_out=%-18s rel_l2=%.2e%s%s  %s\n",
                n_in, ns_str.c_str(), worst_err,
                n_segs == 1 ? (all_bitwise ? " bitwise=yes" : " bitwise=NO") : "",
                graph_check ? (graph_ok ? " graph=ok" : " graph=FAIL") : "",
                fails ? "FAIL" : "ok");

    for (auto p : dws) cudaFree(p);
    for (auto p : dys) cudaFree(p);
    cudaFree(dx);
    cudaFree(dxq);
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool quick = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1 || !k::q8_volta_tc_supported()) {
        std::printf("q8_volta_tc_parity: not sm_70 (skipped)\n");
        return 77;
    }

    cudaStream_t st = nullptr;
    if (!ck(cudaStreamCreate(&st), "stream")) return 1;

    int fails = 0;
    std::printf("=== Prefill Fused Q8_0 W8A16 GEMM ===\n");
    if (quick) {
        fails += test_prefill_case(33, 65, 64, 0, 0.0f, 1, st);
        fails += test_prefill_case(65, 1025, 128, 1088, 1.0f, 2, st);
        fails += test_prefill_case(129, 1025, 128, 0, 0.0f, 3, st);
        fails += test_prefill_case(128, 512, 256, 0, 0.5f, 4, st);
    } else {
        // Edge shapes covering all 3 tile configs + unaligned tails + beta + ldy
        fails += test_prefill_case(32, 64, 64, 0, 0.0f, 10, st);
        fails += test_prefill_case(33, 65, 128, 80, 1.0f, 11, st);
        fails += test_prefill_case(64, 1024, 320, 0, 0.0f, 12, st);
        fails += test_prefill_case(65, 1025, 640, 1088, 0.5f, 13, st);
        fails += test_prefill_case(129, 1025, 640, 0, 0.0f, 14, st);
        // Real Flash-Next dense Q8_0 shapes
        fails += test_prefill_case(130, 512, 2560, 0, 0.0f, 20, st);
        fails += test_prefill_case(130, 640, 2560, 0, 0.0f, 21, st);
        fails += test_prefill_case(130, 2560, 640, 0, 0.0f, 22, st);
        fails += test_prefill_case(130, 2560, 6144, 0, 1.0f, 23, st);
        fails += test_prefill_case(130, 6144, 2560, 0, 0.0f, 24, st);
        fails += test_prefill_case(130, 10240, 2560, 0, 0.0f, 25, st);
        fails += test_prefill_case(130, 12288, 2560, 0, 0.0f, 26, st);
    }

    std::printf("=== Decode Single & Segmented Q8_0 x Q8_1 TC ===\n");
    const std::vector<int> ms_quick = {1, 5, 12};
    const std::vector<int> ms_full = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16};
    const auto& ms = quick ? ms_quick : ms_full;

    // NB_STEP=8 (BR=8 and BR=16), NB_STEP=4 (640), NB_STEP=2 (320), partial row tails
    fails += test_decode_case(256, {17}, ms, 100, true, st);
    fails += test_decode_case(128, {31}, ms, 101, false, st);
    fails += test_decode_case(64, {19}, ms, 102, false, st);
    if (!quick) {
        fails += test_decode_case(320, {512}, ms, 103, false, st);
        fails += test_decode_case(640, {2560}, ms, 104, true, st);
        fails += test_decode_case(2560, {512}, ms, 105, false, st);
        fails += test_decode_case(2560, {640}, ms, 106, false, st);
        fails += test_decode_case(2560, {6144}, ms, 107, true, st);
        fails += test_decode_case(6144, {2560}, ms, 108, true, st);
        // Segmented Flash-Next bundles
        fails += test_decode_case(2560, {640, 640}, ms, 110, true, st);             // shared gate+up
        fails += test_decode_case(2560, {512, 512}, ms, 111, true, st);             // QSA k+v
        fails += test_decode_case(2560, {10240, 6144}, ms, 112, true, st);          // GDN qkv+gate
        fails += test_decode_case(2560, {12288, 512, 512}, ms, 113, true, st);      // QSA q+k+v
        fails += test_decode_case(640, {13, 17, 31, 65}, ms, 114, false, st);       // 4-seg unaligned
    } else {
        fails += test_decode_case(256, {33, 19}, ms, 115, true, st);
        fails += test_decode_case(256, {2560, 640}, ms, 116, false, st);            // exercises BR=16 (>2560)
    }

    cudaStreamDestroy(st);
    std::printf("q8_volta_tc_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
