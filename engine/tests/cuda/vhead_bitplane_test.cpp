// tests/cuda/vhead_bitplane_test.cpp - Parity & latency test for exact 2-stage bit-plane LM head (Q8_0 & Q6_K).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Verifies on sm_70 (Volta) and sm_80+ that:
//   1. vhead_bitplane_gather_raw matches raw GGUF blocks 100% bit-for-bit.
//   2. vhead_bitplane_logits matches native_mmvq within FP32 rounding.
//   3. vhead_bitplane_argmax matches brute-force native_mmvq + argmax_rows 100% bit-for-bit across
//      T = 1, 2, 3, 4, 5, 6, 8, 16, including CUDA Graph capture + replay with changing inputs.
//   4. vhead_bitplane_candidates + sample_tokens_candidates matches full native_mmvq + sample_tokens
//      100% bit-for-bit across min_p, top_p, top_k, temperature, and repeat/freq/present penalties.
#include "strata/kernels/vhead_bitplane.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "vhead_bitplane_test: %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

void test_type(int ggml_type, const char* name, int n_in, int n_out, bool bench) {
    const size_t w_bytes = strata::kernels::native_mmvq_weight_bytes(ggml_type, n_in, n_out);
    const size_t row_bytes = w_bytes / (size_t) n_out;
    std::mt19937 rng(ggml_type == 8 ? 202601u : 202602u);
    std::vector<uint8_t> h_w(w_bytes);

    if (ggml_type == 8) {
        const int nb = n_in / 32;
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = h_w.data() + ((size_t) r * nb + b) * 34u;
                const __half dh = __float2half_rn(0.005f + 0.015f * ((float) (rng() & 255) / 255.0f));
                const uint16_t db = __half_as_ushort(dh);
                blk[0] = (uint8_t) (db & 0xffu);
                blk[1] = (uint8_t) (db >> 8);
                for (int j = 0; j < 32; ++j) {
                    blk[2 + j] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
                }
            }
        }
    } else {
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* blk = h_w.data() + ((size_t) r * nsb + sb) * 210u;
                for (int j = 0; j < 192; ++j) blk[j] = (uint8_t) (rng() & 0xffu);
                for (int s = 0; s < 16; ++s) blk[192 + s] = (uint8_t) (int8_t) ((int) (rng() % 63) - 31);
                const __half dh = __float2half_rn(0.004f + 0.008f * ((float) (rng() & 255) / 255.0f));
                const uint16_t db = __half_as_ushort(dh);
                blk[208] = (uint8_t) (db & 0xffu);
                blk[209] = (uint8_t) (db >> 8);
            }
        }
    }

    std::vector<float> h_x((size_t) 16 * (size_t) n_in);
    std::vector<float> h_x2((size_t) 16 * (size_t) n_in);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : h_x) v = nd(rng);
    for (float& v : h_x2) v = nd(rng);

    // Plant realistic peaks + near-peaks across 25 rows per token so top_k/min_p/top_p have realistic structure,
    // while leaving some tokens unplanted in parity mode to exercise fallback/overflow paths too.
    for (int t = 0; t < 16; ++t) {
        if (!bench && (t % 4 == 3)) continue;
        for (int rank = 0; rank < 24; ++rank) {
            const int target = (t * 7919 + rank * 521 + 137) % n_out;
            const float gain = 1.0f - 0.03f * (float) rank;
            if (ggml_type == 8) {
                const int nb = n_in / 32;
                for (int b = 0; b < nb; ++b) {
                    uint8_t* blk = h_w.data() + ((size_t) target * nb + b) * 34u;
                    const __half dh = __float2half_rn(0.02f * gain);
                    const uint16_t db = __half_as_ushort(dh);
                    blk[0] = (uint8_t) (db & 0xffu);
                    blk[1] = (uint8_t) (db >> 8);
                    for (int j = 0; j < 32; ++j) {
                        const float xv = h_x[(size_t) t * n_in + b * 32 + j];
                        blk[2 + j] = (uint8_t) (int8_t) std::max(-127, std::min(127, (int) std::lrintf(xv * 40.0f)));
                    }
                }
            } else {
                const int nsb = n_in / 256;
                for (int sb = 0; sb < nsb; ++sb) {
                    uint8_t* blk = h_w.data() + ((size_t) target * nsb + sb) * 210u;
                    for (int s = 0; s < 16; ++s) blk[192 + s] = (uint8_t) (int8_t) 16;
                    const __half dh = __float2half_rn(0.01f * gain);
                    const uint16_t db = __half_as_ushort(dh);
                    blk[208] = (uint8_t) (db & 0xffu);
                    blk[209] = (uint8_t) (db >> 8);
                    std::memset(blk, 0, 192);
                    for (int n = 0; n < 2; ++n) {
                        for (int l = 0; l < 32; ++l) {
                            for (int qid = 0; qid < 4; ++qid) {
                                const int elem = sb * 256 + n * 128 + qid * 32 + l;
                                const float xv = h_x[(size_t) t * n_in + elem];
                                const int q = std::max(0, std::min(63, 32 + (int) std::lrintf(xv * 10.0f)));
                                const uint8_t lo4 = (uint8_t) (q & 0xF);
                                const uint8_t hi2 = (uint8_t) ((q >> 4) & 3);
                                uint8_t& ql_b = blk[64 * n + ((qid & 1) ? 32 : 0) + l];
                                if (qid < 2) ql_b |= lo4; else ql_b |= (uint8_t) (lo4 << 4);
                                blk[128 + 32 * n + l] |= (uint8_t) (hi2 << (2 * qid));
                            }
                        }
                    }
                }
            }
        }
    }

    void *d_w = nullptr, *d_xq = nullptr;
    float *d_x = nullptr, *d_logits = nullptr, *d_bp_logits = nullptr;
    int32_t* d_out = nullptr;
    int32_t* d_ref_out = nullptr;
    ck(cudaMalloc(&d_w, w_bytes), "malloc w");
    ck(cudaMalloc(&d_x, h_x.size() * sizeof(float)), "malloc x");
    ck(cudaMalloc(&d_xq, 2u * strata::kernels::native_q8_1_bytes(n_in, 8)), "malloc xq");
    ck(cudaMalloc(&d_logits, (size_t) 16 * (size_t) n_out * sizeof(float)), "malloc logits");
    ck(cudaMalloc(&d_bp_logits, (size_t) 8 * (size_t) n_out * sizeof(float)), "malloc bp logits");
    ck(cudaMalloc(&d_out, 16 * sizeof(int32_t)), "malloc out");
    ck(cudaMalloc(&d_ref_out, 16 * sizeof(int32_t)), "malloc ref out");

    ck(cudaMemcpy(d_w, h_w.data(), w_bytes, cudaMemcpyHostToDevice), "H2D w");
    ck(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D x");

    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");

    auto* bh = strata::kernels::vhead_bitplane_create(ggml_type, d_w, n_in, n_out);
    if (bh == nullptr) {
        std::fprintf(stderr, "vhead_bitplane_test: create failed for %s\n", name);
        std::exit(1);
    }
    // Free d_w after building bh? Wait: we keep d_w only to run reference native_mmvq in the test,
    // verify first that gather_raw reproduces raw h_w 100% bit-for-bit!
    if (!bench) {
        constexpr int kGatherN = 257;
        std::vector<int32_t> h_rows(kGatherN);
        for (int i = 0; i < kGatherN; ++i) h_rows[i] = (i * 97 + 13) % n_out;
        int32_t* d_rows = nullptr;
        uint8_t* d_gathered = nullptr;
        ck(cudaMalloc(&d_rows, h_rows.size() * sizeof(int32_t)), "malloc rows");
        ck(cudaMalloc(&d_gathered, h_rows.size() * row_bytes), "malloc gathered");
        ck(cudaMemcpy(d_rows, h_rows.data(), h_rows.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "H2D rows");
        if (!strata::kernels::vhead_bitplane_gather_raw(bh, d_rows, (int64_t) h_rows.size(), d_gathered, s)) {
            std::fprintf(stderr, "vhead_bitplane_test: gather_raw failed for %s\n", name);
            std::exit(1);
        }
        ck(cudaStreamSynchronize(s), "gather sync");
        std::vector<uint8_t> h_gathered(h_rows.size() * row_bytes);
        ck(cudaMemcpy(h_gathered.data(), d_gathered, h_gathered.size(), cudaMemcpyDeviceToHost), "D2H gathered");
        for (size_t i = 0; i < h_rows.size(); ++i) {
            const uint8_t* exp = h_w.data() + (size_t) h_rows[i] * row_bytes;
            const uint8_t* got = h_gathered.data() + i * row_bytes;
            if (std::memcmp(exp, got, row_bytes) != 0) {
                std::fprintf(stderr, "FAIL %s gather_raw mismatch on row %d\n", name, h_rows[i]);
                std::exit(1);
            }
        }
        ck(cudaFree(d_rows), "free rows");
        ck(cudaFree(d_gathered), "free gathered");
        std::printf("  %s %dx%d: gather_raw 257 rows bit-exact OK\n", name, n_out, n_in);
    }

    // History buffer for penalty sampling parity test
    constexpr int kHistLen = 32;
    std::vector<int32_t> h_hist((size_t) 16 * kHistLen);
    for (int t = 0; t < 16; ++t) {
        for (int j = 0; j < kHistLen; ++j) {
            h_hist[(size_t) t * kHistLen + j] = (t * 7919 + (j % 4) * 521 + 137) % n_out;
        }
    }
    int32_t* d_hist = nullptr;
    ck(cudaMalloc(&d_hist, h_hist.size() * sizeof(int32_t)), "malloc hist");
    ck(cudaMemcpy(d_hist, h_hist.data(), h_hist.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "H2D hist");

    const std::vector<int> Ts = bench ? std::vector<int>{1, 2, 3, 4, 5, 6, 8, 16}
                                      : std::vector<int>{1, 2, 3, 4, 5, 6, 8, 16};
    for (int T : Ts) {
        for (int t0 = 0; t0 < T; t0 += 8) {
            const int nt = std::min(8, T - t0);
            strata::kernels::native_quantize_q8_1(
                d_x + (size_t) t0 * (size_t) n_in,
                reinterpret_cast<uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                n_in, nt, s);
        }
        for (int t0 = 0; t0 < T; t0 += 4) {
            const int nt = std::min(4, T - t0);
            strata::kernels::native_mmvq(
                ggml_type, d_w,
                reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                d_logits + (size_t) t0 * (size_t) n_out, n_in, n_out, nt, s);
        }
        ck(cudaStreamSynchronize(s), "ref sync");
        std::vector<float> h_logits((size_t) T * (size_t) n_out);
        ck(cudaMemcpy(h_logits.data(), d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H logits");

        // Verify vhead_bitplane_logits matches native_mmvq
        if (!bench && T <= 8) {
            if (!strata::kernels::vhead_bitplane_logits(bh, d_xq, T, d_bp_logits, s)) {
                std::fprintf(stderr, "FAIL %s T=%d vhead_bitplane_logits launch failed\n", name, T);
                std::exit(1);
            }
            ck(cudaStreamSynchronize(s), "bp logits sync");
            std::vector<float> h_bp((size_t) T * (size_t) n_out);
            ck(cudaMemcpy(h_bp.data(), d_bp_logits, h_bp.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H bp logits");
            float max_diff = 0.0f;
            float max_rel = 0.0f;
            for (size_t i = 0; i < h_bp.size(); ++i) {
                const float d = std::fabs(h_bp[i] - h_logits[i]);
                max_diff = std::max(max_diff, d);
                max_rel = std::max(max_rel, d / std::max(1.0f, std::fabs(h_logits[i])));
            }
            if (!(max_diff < 5e-4f || max_rel < 1e-4f)) {
                std::fprintf(stderr, "FAIL %s T=%d vhead_bitplane_logits max_diff=%.6g max_rel=%.6g\n",
                             name, T, max_diff, max_rel);
                std::exit(1);
            }
        }

        // Greedy argmax parity
        std::vector<int32_t> ref_tok(T, 0);
        for (int t = 0; t < T; ++t) {
            const float* row = h_logits.data() + (size_t) t * n_out;
            float bv = row[0];
            int32_t br = 0;
            for (int r = 1; r < n_out; ++r) {
                if (row[r] > bv) { bv = row[r]; br = r; }
            }
            ref_tok[t] = br;
        }

        for (int h0 = 0; h0 < T; h0 += 8) {
            const int HT = std::min(8, T - h0);
            const void* xq_ptr = reinterpret_cast<const uint8_t*>(d_xq) +
                                 (size_t) h0 * strata::kernels::native_q8_1_bytes(n_in, 1);
            if (!strata::kernels::vhead_bitplane_argmax(bh, xq_ptr, HT, d_out + h0, s, nullptr, 0, h0)) {
                std::fprintf(stderr, "vhead_bitplane_test: launch failed for %s T=%d\n", name, T);
                std::exit(1);
            }
        }
        ck(cudaStreamSynchronize(s), "bitplane sync");
        std::vector<int32_t> got_tok(T, -1);
        ck(cudaMemcpy(got_tok.data(), d_out, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H out");

        for (int t = 0; t < T; ++t) {
            if (got_tok[t] != ref_tok[t]) {
                const float g_val = h_logits[(size_t) t * n_out + got_tok[t]];
                const float r_val = h_logits[(size_t) t * n_out + ref_tok[t]];
                std::fprintf(stderr, "FAIL %s T=%d tok=%d: got=%d (%.6f) vs ref=%d (%.6f)\n",
                             name, T, t, got_tok[t], g_val, ref_tok[t], r_val);
                std::exit(1);
            }
        }

        // Sampled parity (min_p + top_k + top_p + penalties)
        if (!bench && (T == 1 || T == 5 || T == 8 || T == 16)) {
            struct Case { const char* label; int top_k; float top_p; float min_p; float temp; bool pen; };
            const Case cases[] = {
                {"min_p=0.05,top_p=1", 20, 1.00f, 0.05f, 0.7f, false},
                {"min_p=0.05,top_p=0.95", 20, 0.95f, 0.05f, 0.6f, false},
                {"top_k=64", 64, 1.00f, 0.00f, 1.0f, false},
                {"min_p+penalties", 20, 0.95f, 0.05f, 0.7f, true},
            };
            for (const Case& cs : cases) {
                strata::kernels::SamplerParams sp;
                sp.top_k = cs.top_k;
                sp.top_p = cs.top_p;
                sp.min_p = cs.min_p;
                sp.temperature = cs.temp;
                sp.seed = 424242ull;
                sp.counter = 1000ull;
                sp.greedy = false;
                if (cs.pen) {
                    sp.penalty_last_n = kHistLen;
                    sp.penalty_repeat = 1.15f;
                    sp.penalty_freq = 0.1f;
                    sp.penalty_present = 0.2f;
                }
                strata::kernels::sample_tokens(d_logits, T, n_out, cs.pen ? d_hist : nullptr, cs.pen ? kHistLen : 0,
                                               sp, d_ref_out, s);
                ck(cudaStreamSynchronize(s), "ref sample sync");
                std::vector<int32_t> ref_s(T), got_s(T, -1);
                ck(cudaMemcpy(ref_s.data(), d_ref_out, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H ref s");

                const int k_top = std::min(cs.top_k > 0 && cs.top_k < 64 ? cs.top_k : 64, n_out);
                const int pen_b = cs.pen ? kHistLen : 0;
                const bool use_mp = (cs.top_p >= 1.0f && cs.min_p > 0.0f && cs.min_p < 1.0f && pen_b == 0);
                const int k_need = use_mp ? 0 : (k_top + pen_b);

                for (int h0 = 0; h0 < T; h0 += 8) {
                    const int HT = std::min(8, T - h0);
                    const bool reuse = (T <= 8);
                    const void* xq_ptr = reinterpret_cast<const uint8_t*>(d_xq) +
                                         (size_t) h0 * strata::kernels::native_q8_1_bytes(n_in, 1);
                    const int32_t* c_ids = nullptr;
                    const float* c_log = nullptr;
                    const int32_t* c_cnt = nullptr;
                    const uint8_t* c_ov_d = nullptr;
                    const uint8_t* c_ov_h = nullptr;
                    int max_cand = 0;
                    if (!strata::kernels::vhead_bitplane_candidates(
                            bh, xq_ptr, HT, k_need, cs.min_p, use_mp, reuse, s,
                            &c_ids, &c_log, &c_cnt, &c_ov_d, &c_ov_h, &max_cand, h0)) {
                        std::fprintf(stderr, "FAIL %s T=%d %s candidates failed\n", name, T, cs.label);
                        std::exit(1);
                    }
                    strata::kernels::SamplerParams sp_c = sp;
                    sp_c.counter = sp.counter + (uint64_t) h0;
                    if (!strata::kernels::sample_tokens_candidates(
                            c_ids, c_log, c_cnt, c_ov_d, max_cand, HT, n_out,
                            cs.pen ? (d_hist + (size_t) h0 * kHistLen) : nullptr, cs.pen ? kHistLen : 0,
                            sp_c, d_out + h0, s)) {
                        std::fprintf(stderr, "FAIL %s T=%d %s sample_tokens_candidates failed\n", name, T, cs.label);
                        std::exit(1);
                    }
                    ck(cudaStreamSynchronize(s), "cand sample sync");
                    // For planted tokens (t % 4 != 3) when k_need <= 20 (< 24 planted peaks), candidate extraction
                    // MUST NOT overflow; for top_k=64 or unplanted flat tail (t % 4 == 3), fallback may run.
                    for (int t = 0; t < HT; ++t) {
                        if (!cs.pen && cs.top_k <= 20 && ((h0 + t) % 4 != 3) && c_ov_h[t] != 0) {
                            std::fprintf(stderr, "FAIL %s T=%d %s tok=%d unexpectedly overflowed\n",
                                         name, T, cs.label, h0 + t);
                            std::exit(1);
                        }
                        if (c_ov_h[t] != 0) {
                            const void* xq_t = reinterpret_cast<const uint8_t*>(xq_ptr) +
                                               (size_t) t * strata::kernels::native_q8_1_bytes(n_in, 1);
                            strata::kernels::vhead_bitplane_logits(bh, xq_t, 1, d_bp_logits, s);
                            strata::kernels::SamplerParams sp_1 = sp_c;
                            sp_1.counter = sp_c.counter + (uint64_t) t;
                            strata::kernels::sample_tokens(d_bp_logits, 1, n_out,
                                                           cs.pen ? (d_hist + (size_t) (h0 + t) * kHistLen) : nullptr,
                                                           cs.pen ? kHistLen : 0, sp_1, d_out + h0 + t, s);
                            ck(cudaStreamSynchronize(s), "fallback sample sync");
                        }
                    }
                }
                ck(cudaMemcpy(got_s.data(), d_out, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H got s");
                for (int t = 0; t < T; ++t) {
                    if (got_s[t] != ref_s[t]) {
                        std::fprintf(stderr, "FAIL %s T=%d %s tok=%d: got=%d vs ref=%d\n",
                                     name, T, cs.label, t, got_s[t], ref_s[t]);
                        std::exit(1);
                    }
                }
            }
        }

        if (bench) {
            cudaEvent_t e0, e1;
            ck(cudaEventCreate(&e0), "e0"); ck(cudaEventCreate(&e1), "e1");
            constexpr int NITER = 20;
            for (int i = 0; i < 3; ++i)
                for (int t0 = 0; t0 < T; t0 += 4)
                    strata::kernels::native_mmvq(ggml_type, d_w,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        d_logits + (size_t) t0 * (size_t) n_out, n_in, n_out, std::min(4, T - t0), s);
            strata::kernels::SamplerParams sp_g;
            sp_g.greedy = true; sp_g.temperature = 0.0f;
            strata::kernels::sample_tokens(d_logits, T, n_out, nullptr, 0, sp_g, d_ref_out, s);
            ck(cudaEventRecord(e0, s), "rec");
            for (int i = 0; i < NITER; ++i) {
                for (int t0 = 0; t0 < T; t0 += 4)
                    strata::kernels::native_mmvq(ggml_type, d_w,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        d_logits + (size_t) t0 * (size_t) n_out, n_in, n_out, std::min(4, T - t0), s);
                strata::kernels::sample_tokens(d_logits, T, n_out, nullptr, 0, sp_g, d_ref_out, s);
            }
            ck(cudaEventRecord(e1, s), "rec");
            ck(cudaEventSynchronize(e1), "sync");
            float ms_ref = 0.0f;
            ck(cudaEventElapsedTime(&ms_ref, e0, e1), "el");
            ms_ref /= NITER;

            for (int i = 0; i < 3; ++i)
                for (int h0 = 0; h0 < T; h0 += 8)
                    strata::kernels::vhead_bitplane_argmax(bh,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) h0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        std::min(8, T - h0), d_out + h0, s, nullptr, 0, h0);
            ck(cudaEventRecord(e0, s), "rec");
            for (int i = 0; i < NITER; ++i)
                for (int h0 = 0; h0 < T; h0 += 8)
                    strata::kernels::vhead_bitplane_argmax(bh,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) h0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        std::min(8, T - h0), d_out + h0, s, nullptr, 0, h0);
            ck(cudaEventRecord(e1, s), "rec");
            ck(cudaEventSynchronize(e1), "sync");
            float ms_bp = 0.0f;
            ck(cudaEventElapsedTime(&ms_bp, e0, e1), "el");
            ms_bp /= NITER;

            // Sampled benchmark (min_p=0.05, top_p=1.0, top_k=20, temp=0.7)
            strata::kernels::SamplerParams sp_s;
            sp_s.top_k = 20; sp_s.top_p = 1.0f; sp_s.min_p = 0.05f; sp_s.temperature = 0.7f;
            const int32_t* c_ids = nullptr; const float* c_log = nullptr; const int32_t* c_cnt = nullptr;
            const uint8_t* c_ov_d = nullptr; const uint8_t* c_ov_h = nullptr; int max_cand = 0;
            for (int i = 0; i < 3; ++i) {
                for (int t0 = 0; t0 < T; t0 += 4)
                    strata::kernels::native_mmvq(ggml_type, d_w,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        d_logits + (size_t) t0 * (size_t) n_out, n_in, n_out, std::min(4, T - t0), s);
                strata::kernels::sample_tokens(d_logits, T, n_out, nullptr, 0, sp_s, d_ref_out, s);
            }
            ck(cudaEventRecord(e0, s), "rec");
            for (int i = 0; i < NITER; ++i) {
                for (int t0 = 0; t0 < T; t0 += 4)
                    strata::kernels::native_mmvq(ggml_type, d_w,
                        reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                        d_logits + (size_t) t0 * (size_t) n_out, n_in, n_out, std::min(4, T - t0), s);
                strata::kernels::sample_tokens(d_logits, T, n_out, nullptr, 0, sp_s, d_ref_out, s);
            }
            ck(cudaEventRecord(e1, s), "rec");
            ck(cudaEventSynchronize(e1), "sync");
            float ms_ref_s = 0.0f;
            ck(cudaEventElapsedTime(&ms_ref_s, e0, e1), "el");
            ms_ref_s /= NITER;

            for (int i = 0; i < 3; ++i) {
                for (int h0 = 0; h0 < T; h0 += 8) {
                    const int HT = std::min(8, T - h0);
                    const void* xp = reinterpret_cast<const uint8_t*>(d_xq) + (size_t) h0 * strata::kernels::native_q8_1_bytes(n_in, 1);
                    strata::kernels::vhead_bitplane_candidates(bh, xp, HT, 0, sp_s.min_p, true, false, s,
                                                               &c_ids, &c_log, &c_cnt, &c_ov_d, &c_ov_h, &max_cand, h0);
                    strata::kernels::sample_tokens_candidates(c_ids, c_log, c_cnt, c_ov_d, max_cand, HT, n_out,
                                                              nullptr, 0, sp_s, d_out + h0, s);
                }
            }
            ck(cudaEventRecord(e0, s), "rec");
            for (int i = 0; i < NITER; ++i) {
                for (int h0 = 0; h0 < T; h0 += 8) {
                    const int HT = std::min(8, T - h0);
                    const void* xp = reinterpret_cast<const uint8_t*>(d_xq) + (size_t) h0 * strata::kernels::native_q8_1_bytes(n_in, 1);
                    strata::kernels::vhead_bitplane_candidates(bh, xp, HT, 0, sp_s.min_p, true, false, s,
                                                               &c_ids, &c_log, &c_cnt, &c_ov_d, &c_ov_h, &max_cand, h0);
                    strata::kernels::sample_tokens_candidates(c_ids, c_log, c_cnt, c_ov_d, max_cand, HT, n_out,
                                                              nullptr, 0, sp_s, d_out + h0, s);
                }
            }
            ck(cudaEventRecord(e1, s), "rec");
            ck(cudaEventSynchronize(e1), "sync");
            float ms_bp_s = 0.0f;
            ck(cudaEventElapsedTime(&ms_bp_s, e0, e1), "el");
            ms_bp_s /= NITER;

            std::printf("  %s %dx%d T=%2d: parity OK | greedy: ref=%.3f ms bp=%.3f ms (%.2fx) | min_p: ref=%.3f ms bp=%.3f ms (%.2fx)\n",
                        name, n_out, n_in, T, ms_ref, ms_bp, ms_ref / ms_bp, ms_ref_s, ms_bp_s, ms_ref_s / ms_bp_s);
            ck(cudaEventDestroy(e0), "d0"); ck(cudaEventDestroy(e1), "d1");
        } else {
            std::printf("  %s %dx%d T=%2d: exact parity OK\n", name, n_out, n_in, T);
        }
    }

    // CUDA Graph capture + replay parity with changing inputs (T = 5)
    if (!bench) {
        constexpr int TG = 5;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "begin cap");
        strata::kernels::native_quantize_q8_1(d_x, d_xq, n_in, TG, s);
        strata::kernels::vhead_bitplane_argmax(bh, d_xq, TG, d_out, s, nullptr, 0, 0);
        ck(cudaStreamEndCapture(s, &graph), "end cap");
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
        ck(cudaGraphDestroy(graph), "destroy graph");

        // Replay with h_x2 (different activations!)
        ck(cudaMemcpy(d_x, h_x2.data(), (size_t) TG * n_in * sizeof(float), cudaMemcpyHostToDevice), "H2D x2");
        ck(cudaGraphLaunch(exec, s), "graph launch");
        ck(cudaStreamSynchronize(s), "graph sync");
        std::vector<int32_t> got_g(TG);
        ck(cudaMemcpy(got_g.data(), d_out, (size_t) TG * sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H graph");

        strata::kernels::native_quantize_q8_1(d_x, d_xq, n_in, TG, s);
        for (int t0 = 0; t0 < TG; t0 += 4) {
            const int nt = std::min(4, TG - t0);
            strata::kernels::native_mmvq(
                ggml_type, d_w,
                reinterpret_cast<const uint8_t*>(d_xq) + (size_t) t0 * strata::kernels::native_q8_1_bytes(n_in, 1),
                d_logits + (size_t) t0 * (size_t) n_out, n_in, n_out, nt, s);
        }
        ck(cudaStreamSynchronize(s), "ref g sync");
        std::vector<float> h_lg((size_t) TG * n_out);
        ck(cudaMemcpy(h_lg.data(), d_logits, h_lg.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H lg");
        for (int t = 0; t < TG; ++t) {
            const float* row = h_lg.data() + (size_t) t * n_out;
            int br = 0;
            for (int r = 1; r < n_out; ++r) if (row[r] > row[br]) br = r;
            if (got_g[t] != br) {
                std::fprintf(stderr, "FAIL %s CUDA Graph replay tok=%d: got=%d vs ref=%d\n", name, t, got_g[t], br);
                std::exit(1);
            }
        }
        ck(cudaGraphExecDestroy(exec), "destroy exec");
        std::printf("  %s %dx%d: CUDA Graph capture+replay T=%d OK\n", name, n_out, n_in, TG);
    }

    strata::kernels::vhead_bitplane_destroy(bh);
    ck(cudaStreamDestroy(s), "destroy stream");
    ck(cudaFree(d_w), "free"); ck(cudaFree(d_x), "free"); ck(cudaFree(d_xq), "free");
    ck(cudaFree(d_logits), "free"); ck(cudaFree(d_bp_logits), "free");
    ck(cudaFree(d_out), "free"); ck(cudaFree(d_ref_out), "free"); ck(cudaFree(d_hist), "free");
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0 || !strata::kernels::vhead_bitplane_supported()) {
        std::printf("vhead_bitplane_test: sm_70 or sm_80+ CUDA GPU required (skipped)\n");
        return 77;
    }
    const bool full = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const int n_out = full ? 248320 : 16384;
    test_type(8, "Q8_0", 2560, n_out, full);
    test_type(14, "Q6_K", 2560, n_out, full);
    std::printf("vhead_bitplane_test: ok\n");
    return 0;
}
