// src/core/dflash2_v70_parity.cpp - Volta sm_70 mma.m8n8k4 DFlash2 parity & CUDA Graph test (attention, Q4_0/Q8_0 GEMM, batch GEMM, n_new > 8 draft).
//
//     dflash2_v70_parity [pack_path]
#include "strata/core/dflash2.hpp"
#include "strata/kernels/dflash2.hpp"

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

namespace {

uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t) (sign | r);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t r = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t) (sign | r);
}

float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t x;
    if (e == 0) {
        if (m == 0) x = sign;
        else {
            int s = 0;
            uint32_t mm = m;
            while (!(mm & 0x400)) { mm <<= 1; ++s; }
            x = sign | ((uint32_t) (127 - 15 - s + 1) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

void quant_q4_0(const float* x, int n, uint8_t* out) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0, mx = 0;
        for (int i = 0; i < 32; ++i)
            if (std::fabs(x[b * 32 + i]) > amax) { amax = std::fabs(x[b * 32 + i]); mx = x[b * 32 + i]; }
        const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = f2h(d);
        out[b * 18] = (uint8_t) (dh & 0xff);
        out[b * 18 + 1] = (uint8_t) (dh >> 8);
        for (int i = 0; i < 16; ++i) {
            const int q0 = std::min(15, std::max(0, (int) (x[b * 32 + i] * id + 8.5f)));
            const int q1 = std::min(15, std::max(0, (int) (x[b * 32 + 16 + i] * id + 8.5f)));
            out[b * 18 + 2 + i] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

void quant_q8_0(const float* x, int n, uint8_t* out) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = f2h(d);
        out[b * 34] = (uint8_t) (dh & 0xff);
        out[b * 34 + 1] = (uint8_t) (dh >> 8);
        for (int i = 0; i < 32; ++i) {
            const int q = std::max(-127, std::min(127, (int) std::lrintf(x[b * 32 + i] * id)));
            out[b * 34 + 2 + i] = (uint8_t) (int8_t) q;
        }
    }
}

int test_attention_parity() {
    using namespace strata::kernels;
    constexpr int ring = 2064, window = 2048;
    std::mt19937 rng(20261125);
    std::normal_distribution<float> nd(0.0f, 0.7f);

    std::vector<float> hq(8 * 32 * 128), hkr((size_t) ring * 8 * 128), hvr((size_t) ring * 8 * 128);
    std::vector<float> hkb(8 * 8 * 128), hvb(8 * 8 * 128);
    for (float& v : hq) v = nd(rng);
    for (float& v : hkr) v = nd(rng);
    for (float& v : hvr) v = nd(rng);
    for (float& v : hkb) v = nd(rng);
    for (float& v : hvb) v = nd(rng);

    float *dq, *dkr, *dvr, *dkb, *dvb, *dpart, *dxd;
    int8_t* dxq;
    int32_t *dpos, *dep;
    int64_t* dtag;
    cudaMalloc(&dq, hq.size() * 4);
    cudaMalloc(&dkr, hkr.size() * 4);
    cudaMalloc(&dvr, hvr.size() * 4);
    cudaMalloc(&dkb, hkb.size() * 4);
    cudaMalloc(&dvb, hvb.size() * 4);
    cudaMalloc(&dpart, (size_t) kDf2MaxChunks * 8 * 32 * 130 * 4);
    cudaMalloc(&dxq, 8 * 4096);
    cudaMalloc(&dxd, 8 * 128 * 4);
    cudaMalloc(&dpos, 4);
    cudaMalloc(&dep, 4);
    cudaMalloc(&dtag, (size_t) ring * 8);

    cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkr, hkr.data(), hkr.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dvr, hvr.data(), hvr.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkb, hkb.data(), hkb.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dvb, hvb.data(), hvb.size() * 4, cudaMemcpyHostToDevice);

    const int32_t positions[] = {1, 7, 31, 128, 513, 2047, 5000, 260000};
    int bad = 0;
    const double scale = 1.0 / std::sqrt(128.0);
    for (int ci = 0; ci < (int) (sizeof(positions) / sizeof(positions[0])); ++ci) {
        const int32_t s = positions[ci];
        const int32_t epoch = 3 + ci;
        const int lo = std::max(0, s - (window - 1));
        const int nctx = s - lo;
        const int nkeys = nctx + 8;
        std::vector<int64_t> htag(ring, -1);
        for (int k = 0; k < nctx; ++k) {
            // Leave every 17th key stale on odd cases to exercise tag masking.
            if ((ci & 1) && (k % 17 == 5)) continue;
            htag[(size_t) ((lo + k) % ring)] = ((int64_t) epoch << 32) | (int64_t) (lo + k);
        }
        cudaMemcpy(dpos, &s, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dep, &epoch, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtag, htag.data(), htag.size() * 8, cudaMemcpyHostToDevice);

        df2_attention(dq, dkr, dvr, dkb, dvb, dpos, ring, window, dpart, dxq, dxd, nullptr, dtag, dep);
        cudaDeviceSynchronize();

        std::vector<int8_t> oxq(8 * 4096);
        std::vector<float> oxd(8 * 128);
        cudaMemcpy(oxq.data(), dxq, oxq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(oxd.data(), dxd, oxd.size() * 4, cudaMemcpyDeviceToHost);

        // FP64 CPU reference for all 8 rows x 32 heads x 128 dims
        double dot = 0, na = 0, nb = 0, max_diff = 0, max_ref = 0;
        for (int row = 0; row < 8; ++row) {
            for (int kvh = 0; kvh < 8; ++kvh) {
                for (int g = 0; g < 4; ++g) {
                    const int h = kvh * 4 + g;
                    const float* qv = &hq[((size_t) row * 32 + h) * 128];
                    std::vector<double> scores((size_t) nkeys, -1e300);
                    double mx = -1e300;
                    for (int k = 0; k < nkeys; ++k) {
                        const float* kv = nullptr;
                        if (k < nctx) {
                            const int pos_k = lo + k;
                            if (pos_k <= s + row - window) continue;
                            const int slot = pos_k % ring;
                            if (htag[(size_t) slot] != (((int64_t) epoch << 32) | (int64_t) pos_k)) continue;
                            kv = &hkr[((size_t) kvh * ring + slot) * 128];
                        } else {
                            kv = &hkb[((size_t) (k - nctx) * 8 + kvh) * 128];
                        }
                        double acc = 0;
                        for (int d = 0; d < 128; ++d) acc += (double) qv[d] * (double) kv[d];
                        acc *= scale;
                        scores[(size_t) k] = acc;
                        mx = std::max(mx, acc);
                    }
                    double Z = 0;
                    double out[128] = {};
                    for (int k = 0; k < nkeys; ++k) {
                        if (scores[(size_t) k] <= -1e200) continue;
                        const double w = std::exp(scores[(size_t) k] - mx);
                        Z += w;
                        const float* vv = (k < nctx)
                            ? &hvr[((size_t) kvh * ring + ((lo + k) % ring)) * 128]
                            : &hvb[((size_t) (k - nctx) * 8 + kvh) * 128];
                        for (int d = 0; d < 128; ++d) out[d] += w * (double) vv[d];
                    }
                    for (int d = 0; d < 128; ++d) {
                        const double ref = Z > 0 ? out[d] / Z : 0.0;
                        const int blk = h * 4 + (d >> 5);
                        const double got = (double) oxq[(size_t) row * 4096 + h * 128 + d] * (double) oxd[(size_t) row * 128 + blk];
                        if (!std::isfinite(got)) ++bad;
                        dot += got * ref;
                        na += got * got;
                        nb += ref * ref;
                        max_diff = std::max(max_diff, std::fabs(got - ref));
                        max_ref = std::max(max_ref, std::fabs(ref));
                    }
                }
            }
        }
        const double cos_sim = dot / std::sqrt(na * nb + 1e-30);
        // Output is quantized to int8 (step ~1/127 ~= 0.008), so cos_sim > 0.999 and max_diff < 0.03 * max_ref.
        const bool ok = cos_sim > 0.999 && max_diff < 0.03 * std::max(max_ref, 1.0);
        if (!ok) ++bad;
        std::printf("attn s=%6d (nctx=%4d, mask=%d): cos=%.6f max_diff=%.4g (ref_max=%.3g) %s\n",
                    s, nctx, ci & 1, cos_sim, max_diff, max_ref, ok ? "ok" : "FAIL");
    }

    // CUDA Graph replay test with changing anchor_pos & epoch
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaGraph_t gr = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
    df2_attention(dq, dkr, dvr, dkb, dvb, dpos, ring, window, dpart, dxq, dxd, st, dtag, dep);
    cudaStreamEndCapture(st, &gr);
    cudaGraphInstantiate(&exec, gr, 0);
    for (int it = 0; it < 20; ++it) {
        const int32_t s = positions[it % 8];
        const int32_t epoch = 100 + it;
        const int lo = std::max(0, s - (window - 1));
        std::vector<int64_t> htag(ring, -1);
        for (int k = 0; k < s - lo; ++k) htag[(size_t) ((lo + k) % ring)] = ((int64_t) epoch << 32) | (int64_t) (lo + k);
        cudaMemcpyAsync(dpos, &s, 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dep, &epoch, 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dtag, htag.data(), htag.size() * 8, cudaMemcpyHostToDevice, st);
        cudaGraphLaunch(exec, st);
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) { std::printf("attn graph replay FAIL\n"); ++bad; }
    else std::printf("attn CUDA Graph replay (20 varying pos/epoch): ok\n");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(gr);
    cudaStreamDestroy(st);

    cudaFree(dq); cudaFree(dkr); cudaFree(dvr); cudaFree(dkb); cudaFree(dvb);
    cudaFree(dpart); cudaFree(dxq); cudaFree(dxd); cudaFree(dpos); cudaFree(dep); cudaFree(dtag);
    return bad;
}

int test_gemm_parity(bool q4) {
    using namespace strata::kernels;
    std::mt19937 rng(q4 ? 20261126 : 20261127);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    struct Case { int n_in; std::vector<int> n_out; };
    const Case cases[] = {
        {2560, {16}},
        {2560, {256}},
        {2560, {640}},
        {2560, {1024, 1024}},
        {2560, {4096, 1024, 1024}},
        {4096, {2560}},
        {7680, {2560}},
        {12800, {2560}},
        {2560, {7680, 7680}},
    };
    const int ncols_list[] = {1, 2, 3, 4, 6, 7, 8, 9, 12, 16, 31, 64, 65};
    constexpr int kMaxCols = 65;
    const int blk_bytes = q4 ? 18 : 34;

    int bad = 0, total = 0;
    for (const Case& cs : cases) {
        const int n_in = cs.n_in, nb = n_in / 32, nm = (int) cs.n_out.size();
        std::vector<std::vector<uint8_t>> blocks(nm);
        std::vector<uint8_t*> dev_w(nm);
        std::vector<float*> dev_y(nm);
        for (int m = 0; m < nm; ++m) {
            const int n_out = cs.n_out[m];
            std::vector<float> w((size_t) n_out * n_in);
            for (float& v : w) v = nd(rng);
            blocks[m].resize((size_t) n_out * nb * blk_bytes);
            for (int r = 0; r < n_out; ++r) {
                if (q4) quant_q4_0(&w[(size_t) r * n_in], n_in, &blocks[m][(size_t) r * nb * blk_bytes]);
                else quant_q8_0(&w[(size_t) r * n_in], n_in, &blocks[m][(size_t) r * nb * blk_bytes]);
            }
            uint8_t* tmp;
            cudaMalloc(&tmp, blocks[m].size());
            cudaMemcpy(tmp, blocks[m].data(), blocks[m].size(), cudaMemcpyHostToDevice);
            cudaMalloc(&dev_w[m], blocks[m].size());
            if (q4) df2_q4_tile(tmp, dev_w[m], dev_w[m] + (size_t) n_out * n_in / 2, n_in, (int64_t) n_out * nb, nullptr);
            else df2_q8_tile(tmp, dev_w[m], dev_w[m] + (size_t) n_out * n_in, n_in, (int64_t) n_out * nb, nullptr);
            cudaDeviceSynchronize();
            cudaFree(tmp);
            cudaMalloc(&dev_y[m], (size_t) kMaxCols * n_out * 4);
        }
        std::vector<float> x((size_t) kMaxCols * n_in);
        for (float& v : x) v = nd(rng) * 4.0f;
        float *dx, *dxd;
        int8_t* dxq;
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dxq, (size_t) kMaxCols * n_in + 4096);
        cudaMalloc(&dxd, (size_t) kMaxCols * nb * 4 + 64);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        df2_quant_rows(dx, n_in, kMaxCols, dxq, dxd, nullptr);
        cudaDeviceSynchronize();
        std::vector<int8_t> xq((size_t) kMaxCols * n_in);
        std::vector<float> xd((size_t) kMaxCols * nb);
        cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(xd.data(), dxd, xd.size() * 4, cudaMemcpyDeviceToHost);

        for (int ncols : ncols_list) {
            // Keep runtime bounded on the largest multi-matrix shapes while testing every boundary.
            if (cs.n_out[0] >= 7680 && ncols > 12 && ncols != 64) continue;
            Df2Mat mats[3];
            for (int m = 0; m < nm; ++m) mats[m] = Df2Mat{dev_w[m], dev_y[m], cs.n_out[m], q4};
            df2_gemm_variant(50);   // exercise Volta mma.m8n8k4 TC (both <=8 and batch >8)
            df2_gemm(mats, nm, dxq, dxd, n_in, ncols, nullptr);
            df2_gemm_variant(0);
            cudaDeviceSynchronize();
            double worst = 0, big = 0;
            for (int m = 0; m < nm; ++m) {
                const int n_out = cs.n_out[m];
                std::vector<float> y((size_t) ncols * n_out);
                cudaMemcpy(y.data(), dev_y[m], y.size() * 4, cudaMemcpyDeviceToHost);
                const int r_step = n_out > 256 ? 37 : 1;
                for (int c = 0; c < ncols; ++c) {
                    for (int r = 0; r < n_out; r += r_step) {
                        double ref = 0;
                        for (int b = 0; b < nb; ++b) {
                            const uint8_t* blk = &blocks[m][((size_t) r * nb + b) * blk_bytes];
                            const uint16_t dh = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
                            long isum = 0;
                            if (q4) {
                                for (int i = 0; i < 16; ++i) {
                                    isum += (long) ((blk[2 + i] & 15) - 8) * xq[(size_t) c * n_in + b * 32 + i];
                                    isum += (long) ((blk[2 + i] >> 4) - 8) * xq[(size_t) c * n_in + b * 32 + 16 + i];
                                }
                            } else {
                                for (int i = 0; i < 32; ++i)
                                    isum += (long) (int8_t) blk[2 + i] * xq[(size_t) c * n_in + b * 32 + i];
                            }
                            ref += (double) h2f(dh) * (double) xd[(size_t) c * nb + b] * (double) isum;
                        }
                        const double got = y[(size_t) c * n_out + r];
                        if (!std::isfinite(got)) ++bad;
                        worst = std::max(worst, std::fabs(got - ref));
                        big = std::max(big, std::fabs(ref));
                    }
                }
            }
            const bool ok = worst <= 2e-5 * std::max(big, 1e-9);
            ++total;
            if (!ok) {
                ++bad;
                std::printf("FAIL %s n_in=%d n_out0=%d nm=%d ncols=%d: rel=%.3g\n",
                            q4 ? "Q4" : "Q8", n_in, cs.n_out[0], nm, ncols, worst / std::max(big, 1e-9));
            }
        }
        for (int m = 0; m < nm; ++m) { cudaFree(dev_w[m]); cudaFree(dev_y[m]); }
        cudaFree(dx); cudaFree(dxq); cudaFree(dxd);
    }
    std::printf("gemm %s parity: %d/%d cases ok\n", q4 ? "Q4_0" : "Q8_0", total - bad, total);
    return bad;
}

int test_long_lookup_draft(const char* pack_path) {
    using strata::core::Dflash2Drafter;
    std::string err;
    constexpr int H = 2560, tapn = 5 * H, vp = 1024;
    std::mt19937 rng(20261128);
    std::normal_distribution<float> nd(0.0f, 0.25f);
    std::vector<float> taps((size_t) 20 * tapn), emb(2 * H);
    for (float& v : taps) v = nd(rng);
    for (float& v : emb) v = nd(rng);
    std::vector<int32_t> dvocab(vp);
    for (int i = 0; i < vp; ++i) dvocab[i] = 100 + i;
    const size_t row_bytes = (size_t) (H / 32) * 34;
    std::vector<uint8_t> head((size_t) vp * row_bytes);
    std::vector<float> hw((size_t) vp * H);
    for (float& v : hw) v = nd(rng) * 0.1f;
    for (int r = 0; r < vp; ++r) quant_q8_0(&hw[(size_t) r * H], H, &head[(size_t) r * row_bytes]);

    float *dtaps = nullptr, *demb = nullptr;
    uint8_t* dhead = nullptr;
    cudaMalloc(&dtaps, taps.size() * 4);
    cudaMalloc(&demb, emb.size() * 4);
    cudaMalloc(&dhead, head.size());
    cudaMemcpy(dtaps, taps.data(), taps.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(demb, emb.data(), emb.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dhead, head.data(), head.size(), cudaMemcpyHostToDevice);

    auto embed = [&](const int32_t*, int n, float* out, cudaStream_t cs) {
        cudaMemcpyAsync(out, demb, (size_t) H * 4, cudaMemcpyDeviceToDevice, cs);
        for (int i = 1; i < n; ++i) cudaMemcpyAsync(out + (size_t) i * H, demb + H, (size_t) H * 4, cudaMemcpyDeviceToDevice, cs);
    };

    // Reference: context(19 rows) + draft(1 row at pos 19)
    // Test:      context(8 rows) + draft(12 rows at pos 8 -> exercises big_ctx n_new=12 + exec_[0]!)
    int32_t tok_ref[7] = {}, tok_big[7] = {};
    float p_ref[7] = {}, p_big[7] = {};
    std::vector<float> rows_ref, rows_big;

    {
        Dflash2Drafter d;
        if (!d.load(pack_path, 0, err) || !d.bind_raw(embed, dhead, 8, vp, dvocab, err) || !d.prepare(dtaps + 19 * tapn, err)) {
            std::printf("long_lookup ref setup FAIL: %s\n", err.c_str());
            return 1;
        }
        if (!d.context(dtaps, 19, 0, err) || !d.draft(dtaps + 19 * tapn, 1, 19, 123, tok_ref, p_ref, err) || !d.debug_rows(rows_ref, err)) {
            std::printf("long_lookup ref run FAIL: %s\n", err.c_str());
            return 1;
        }
    }
    {
        Dflash2Drafter d;
        if (!d.load(pack_path, 0, err) || !d.bind_raw(embed, dhead, 8, vp, dvocab, err) || !d.prepare(dtaps + 8 * tapn, err)) {
            std::printf("long_lookup big setup FAIL: %s\n", err.c_str());
            return 1;
        }
        if (!d.context(dtaps, 8, 0, err) || !d.draft(dtaps + 8 * tapn, 12, 8, 123, tok_big, p_big, err) || !d.debug_rows(rows_big, err)) {
            std::printf("long_lookup big run FAIL: %s\n", err.c_str());
            return 1;
        }
        // Also verify normal n_new=3 draft works cleanly right after big_ctx!
        int32_t tok2[7] = {};
        float p2[7] = {};
        if (!d.draft(dtaps + 8 * tapn, 3, 20, 124, tok2, p2, err)) {
            std::printf("long_lookup follow-up draft FAIL: %s\n", err.c_str());
            return 1;
        }
    }

    int bad = 0;
    double max_diff = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < rows_ref.size(); ++i) {
        max_diff = std::max(max_diff, (double) std::fabs(rows_ref[i] - rows_big[i]));
        dot += (double) rows_ref[i] * (double) rows_big[i];
        na += (double) rows_ref[i] * (double) rows_ref[i];
        nb += (double) rows_big[i] * (double) rows_big[i];
    }
    const double cos_sim = dot / std::sqrt(na * nb + 1e-30);
    int same = 0;
    for (int i = 0; i < 7; ++i) same += (tok_ref[i] == tok_big[i]);
    if (cos_sim < 0.999 || same < 7) ++bad;
    std::printf("long_lookup n_new=12 vs 19+1: tokens_match=%d/7 cos=%.6f max_row_diff=%.3g %s\n",
                same, cos_sim, max_diff, bad == 0 ? "ok" : "FAIL");

    cudaFree(dtaps); cudaFree(demb); cudaFree(dhead);
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    int bad = 0;
    bad += test_attention_parity();
    bad += test_gemm_parity(true);
    bad += test_gemm_parity(false);
    if (argc > 1) bad += test_long_lookup_draft(argv[1]);
    std::printf(bad == 0 ? "dflash2_v70_parity: PASS\n" : "dflash2_v70_parity: FAIL (%d)\n", bad);
    return bad == 0 ? 0 : 1;
}
