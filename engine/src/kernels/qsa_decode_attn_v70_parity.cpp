// src/kernels/qsa_decode_attn_v70_parity.cpp - PR 3.3: Volta sm_70 mma.m8n8k4 + 128-bit KV decode attention parity,
// CUDA Graph replay test, and microbenchmark (decode attention + kv_stream_resolve) on all 4 KV formats (fp16, int8,
// q4_0, k8v4).  GPU, synthetic, no model.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

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
        std::fprintf(stderr, "qsa_decode_attn_v70_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return p;
}
template <typename T> T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> T* halloc(size_t n) {
    void* h = nullptr;
    void* d = nullptr;
    ck(cudaHostAlloc(&h, n * sizeof(T) + 256, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(h, 0, n * sizeof(T) + 256);
    ck(cudaHostGetDevicePointer(&d, h, 0), "cudaHostGetDevicePointer");
    return (T*) d;
}

uint16_t f16_value(uint32_t r) { return (uint16_t) ((r & 0x8000u) | ((11u + (r >> 16) % 5u) << 10) | (r & 0x3FFu)); }
uint16_t f16_scale_q8(uint32_t r) { return (uint16_t) ((9u << 10) | (r & 0x3FFu)); }
// Q4_0 scales can be negative (d = max_val / -8); exercise both signs!
uint16_t f16_scale_q4_signed(uint32_t r) { return (uint16_t) ((r & 0x8000u) | (13u << 10) | (r & 0x3FFu)); }

constexpr int kFmtF16 = 0, kFmtInt8 = 1, kFmtQ4 = 2, kFmtK8V4 = 3;
const char* fmt_name(int f) { return f == kFmtF16 ? "fp16" : f == kFmtInt8 ? "int8" : f == kFmtQ4 ? "q4_0" : "k8v4"; }

struct DevicePools {
    uint16_t *k16 = nullptr, *v16 = nullptr, *k8s = nullptr, *v8s = nullptr;
    int8_t *k8 = nullptr, *v8 = nullptr;
    uint8_t *k4 = nullptr, *v4 = nullptr;
    k::QsaAttnPools view(int fmt, const int32_t* table) const {
        k::QsaAttnPools p;
        p.page_table = table;
        if (fmt == kFmtF16) { p.k_pool = k16; p.v_pool = v16; }
        else if (fmt == kFmtInt8) { p.k_q = k8; p.v_q = v8; p.k_scale = k8s; p.v_scale = v8s; }
        else if (fmt == kFmtQ4) { p.k_q4 = k4; p.v_q4 = v4; }
        else { p.k_q = k8; p.k_scale = k8s; p.v_q4 = v4; }
        return p;
    }
};

DevicePools make_pools(int64_t rows, int64_t D, std::mt19937& rng) {
    DevicePools p;
    {
        std::vector<uint16_t> h((size_t) (rows * D));
        for (auto& x : h) x = f16_value(rng());
        p.k16 = upload(h);
        for (auto& x : h) x = f16_value(rng());
        p.v16 = upload(h);
    }
    {
        std::vector<int8_t> c((size_t) (rows * D));
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        p.k8 = upload(c);
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        p.v8 = upload(c);
        std::vector<uint16_t> sc((size_t) (rows * (D / k::KV_Q8_GROUP)));
        for (auto& x : sc) x = f16_scale_q8(rng());
        p.k8s = upload(sc);
        for (auto& x : sc) x = f16_scale_q8(rng());
        p.v8s = upload(sc);
    }
    {
        const size_t blocks = (size_t) (rows * (D / k::QK4_0));
        std::vector<uint8_t> b(blocks * sizeof(k::block_q4_0));
        for (int side = 0; side < 2; ++side) {
            for (size_t i = 0; i < blocks; ++i) {
                k::block_q4_0 blk;
                blk.d = (i % 97u == 0u) ? 0u : f16_scale_q4_signed(rng());
                for (auto& q : blk.qs) q = (uint8_t) (rng() & 0xFFu);
                std::memcpy(b.data() + i * sizeof(k::block_q4_0), &blk, sizeof(blk));
            }
            (side == 0 ? p.k4 : p.v4) = upload(b);
        }
    }
    return p;
}

constexpr int64_t kCells = 40960;
constexpr int64_t kHole = 8192;

void make_sel(int32_t* ids, int64_t cap, int width, int shape, std::mt19937& rng) {
    const int64_t w = std::max<int64_t>(0, std::min<int64_t>(width, cap));
    if (shape == 0) {
        int64_t start = (rng() % 2 == 0) ? std::max<int64_t>(0, kHole - 64 * (int64_t) (rng() % 6))
                                         : (int64_t) (rng() % (uint32_t) (kCells - std::max<int64_t>(w, 1)));
        if (start + w > kCells) start = kCells - w;
        for (int64_t j = 0; j < cap; ++j) ids[j] = (int32_t) ((start + j) % kCells);
    } else if (shape == 1) {
        // Ascending page4 blocks (like QSA top-k)
        const int64_t pages = kCells / 4;
        std::vector<int32_t> blks((size_t) ((w + 3) / 4));
        for (auto& b : blks) b = (int32_t) (rng() % (uint32_t) pages);
        std::sort(blks.begin(), blks.end());
        int64_t pos = 0;
        for (int32_t b : blks) {
            for (int r = 0; r < 4 && pos < w; ++r) ids[pos++] = b * 4 + r;
        }
        for (; pos < cap; ++pos) ids[pos] = 0;
    } else {
        for (int64_t j = 0; j < cap; ++j) ids[j] = (int32_t) (kHole + (j % 256));   // all-masked hole!
    }
}

struct ErrStats {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
    long long nonfinite = 0;
};

ErrStats compare_vecs(const std::vector<float>& ref, const std::vector<float>& got) {
    ErrStats e;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) { ++e.nonfinite; continue; }
        const double d = (double) got[i] - (double) ref[i];
        e.max_abs = std::max(e.max_abs, std::fabs(d));
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    e.rel_l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    return e;
}

void bench_kv_stream(const k::QsaShapes& s, std::mt19937& rng) {
    std::printf("\n--kv_stream_resolve bench (CUDA events, 100 iters, n_q=5, cap=2051):\n");
    const int64_t n_blocks = 75000, n_slots = 8192, cap = 2051, n_q = 5;
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "e0");
    ck(cudaEventCreate(&e1), "e1");
    for (int fmt : {k::kKvHybrid, k::kKvInt8}) {
        const char* name = fmt == k::kKvHybrid ? "k8v4" : "int8";
        const size_t rows_p = (size_t) s.n_head_kv * s.page_size;
        const size_t d_rows = (size_t) n_slots * rows_p;
        const size_t h_rows = (size_t) n_blocks * rows_p;
        k::QsaAttnPools slots{};
        k::KvHostPools host{};
        if (fmt == k::kKvHybrid) {
            const size_t q4 = k::kv_q4_bytes_per_head((int) s.head_dim);
            slots.k_q = dalloc<int8_t>(d_rows * s.head_dim);
            slots.k_scale = dalloc<uint16_t>(d_rows * 4);
            slots.v_q4 = dalloc<uint8_t>(d_rows * q4);
            host.k_q = halloc<int8_t>(h_rows * s.head_dim);
            host.k_scale = halloc<uint16_t>(h_rows * 4);
            host.v_q4 = halloc<uint8_t>(h_rows * q4);
        } else {
            slots.k_q = dalloc<int8_t>(d_rows * s.head_dim);
            slots.v_q = dalloc<int8_t>(d_rows * s.head_dim);
            slots.k_scale = dalloc<uint16_t>(d_rows * 4);
            slots.v_scale = dalloc<uint16_t>(d_rows * 4);
            host.k_q = halloc<int8_t>(h_rows * s.head_dim);
            host.v_q = halloc<int8_t>(h_rows * s.head_dim);
            host.k_scale = halloc<uint16_t>(h_rows * 4);
            host.v_scale = halloc<uint16_t>(h_rows * 4);
        }
        k::KvStreamMap m{};
        m.page_table = dalloc<int32_t>(n_blocks);
        m.slot_block = dalloc<int32_t>(n_slots);
        m.slot_stamp = dalloc<int32_t>(n_slots);
        m.slot_ref = dalloc<int32_t>(n_slots);
        m.miss_block = dalloc<int32_t>(n_slots);
        m.miss_slot = dalloc<int32_t>(n_slots);
        m.ctl = dalloc<int32_t>(k::kKvCtlInts);
        m.n_blocks = n_blocks;
        m.n_slots = n_slots;
        slots.page_table = m.page_table;

        std::vector<int32_t> hst((size_t) n_q * k::kStepCount, 0);
        for (int q = 0; q < n_q; ++q) hst[(size_t) q * k::kStepCount + k::kStepWidth] = (int32_t) cap;
        int32_t* steps = upload(hst);
        int32_t* ids = dalloc<int32_t>((size_t) n_q * cap);

        for (int target_misses : {0, 4, 16, 64, 256}) {
            // Prepare 2 alternating id buffers so each call hits (512 - target_misses) and misses target_misses!
            std::vector<int32_t> hids0((size_t) n_q * cap), hids1((size_t) n_q * cap);
            for (int q = 0; q < n_q; ++q) {
                for (int b = 0; b < 512; ++b) {
                    const int blk0 = b < target_misses ? (10000 + b) : b;
                    const int blk1 = b < target_misses ? (20000 + b) : b;
                    for (int r = 0; r < 4; ++r) {
                        const int idx = b * 4 + r;
                        if (idx < cap) {
                            hids0[(size_t) q * cap + idx] = blk0 * 4 + r;
                            hids1[(size_t) q * cap + idx] = blk1 * 4 + r;
                        }
                    }
                }
                for (int idx = 2048; idx < cap; ++idx) {
                    hids0[(size_t) q * cap + idx] = 512 * 4 + (idx - 2048);
                    hids1[(size_t) q * cap + idx] = 512 * 4 + (idx - 2048);
                }
            }
            int32_t* d_ids0 = upload(hids0);
            int32_t* d_ids1 = upload(hids1);
            // Warm resident set for target_misses == 0; for target_misses > 0, reset before each iter?
            // Even cleaner: fill n_slots with blocks 0..512, then alternate fresh blocks each iter!
            const int iters = 60;
            std::vector<int32_t*> id_bufs((size_t) iters);
            for (int it = 0; it < iters; ++it) {
                if (target_misses == 0) {
                    id_bufs[it] = d_ids0;
                } else {
                    std::vector<int32_t> h((size_t) n_q * cap);
                    const int base_blk = 1000 + it * target_misses;
                    for (int q = 0; q < n_q; ++q) {
                        for (int b = 0; b < 512; ++b) {
                            const int blk = b < target_misses ? (base_blk + b) : b;
                            for (int r = 0; r < 4; ++r)
                                if (b * 4 + r < cap) h[(size_t) q * cap + b * 4 + r] = blk * 4 + r;
                        }
                        for (int idx = 2048; idx < cap; ++idx) h[(size_t) q * cap + idx] = 512 * 4 + (idx - 2048);
                    }
                    id_bufs[it] = upload(h);
                }
            }
            k::kv_stream_reset(m, nullptr);
            k::kv_stream_resolve(m, slots, host, fmt, d_ids0, steps, n_q, cap, s, nullptr);
            ck(cudaDeviceSynchronize(), "kv warm");
            const k::KvStreamCounters c0 = k::kv_stream_counters(m);
            ck(cudaEventRecord(e0, nullptr), "e0");
            for (int it = 0; it < iters; ++it)
                k::kv_stream_resolve(m, slots, host, fmt, id_bufs[it], steps, n_q, cap, s, nullptr);
            ck(cudaEventRecord(e1, nullptr), "e1");
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0.0f;
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            const k::KvStreamCounters c1 = k::kv_stream_counters(m);
            const double avg_miss = (double) (c1.misses - c0.misses) / iters;
            const double us = 1000.0 * ms / iters;
            const double gb_s = avg_miss > 0.0 ? (avg_miss * (double) k::kv_block_bytes(s, fmt)) / (us * 1e3) : 0.0;
            std::printf("  %-5s target_miss=%3d (actual %.1f): %7.2f us/call  (eff %.2f GB/s)\n",
                        name, target_misses, avg_miss, us, gb_s);
            if (target_misses > 0) for (auto* p : id_bufs) cudaFree(p);
            cudaFree(d_ids0);
            cudaFree(d_ids1);
        }
        cudaFree(steps);
        cudaFree(ids);
        cudaFree(m.page_table); cudaFree(m.slot_block); cudaFree(m.slot_stamp);
        cudaFree(m.slot_ref); cudaFree(m.miss_block); cudaFree(m.miss_slot); cudaFree(m.ctl);
        (void) rng;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    bool do_bench = false;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        else if (std::strcmp(argv[i], "--quick") == 0) quick = true;
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no GPU");
        return 77;
    }
    ck(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (prop.major != 7 || prop.minor != 0) {
        std::printf("SKIP: device %d is %s (%d.%d), not sm_70\n", device, prop.name, prop.major, prop.minor);
        return 77;
    }
    const char* v70_env = std::getenv("STRATA_ATTN_V70");
    const int v70_mode = (v70_env && *v70_env) ? std::atoi(v70_env) : 2;
    std::printf("qsa_decode_attn_v70_parity on device %d (%s, %d.%d) STRATA_ATTN_V70=%d\n",
                device, prop.name, prop.major, prop.minor, v70_mode);

    std::mt19937 rng(20261124u);
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t PS = s.page_size, pages = kCells / PS, H = s.n_head_kv, D = s.head_dim, NH = s.n_head;

    std::vector<int32_t> tab((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
    std::shuffle(tab.begin(), tab.end(), rng);
    for (int64_t i = 0; i < pages; ++i) if (rng() % 32u == 0) tab[(size_t) i] = -1;
    for (int64_t i = kHole / PS; i < (kHole + 256) / PS; ++i) tab[(size_t) i] = -1;
    int32_t* d_table = upload(tab);
    DevicePools pools = make_pools(pages * H * PS, D, rng);

    const int64_t max_q = 6, max_cap = 32768;
    float* d_q = dalloc<float>((size_t) (max_q * NH * D));
    int32_t* d_ids = dalloc<int32_t>((size_t) (max_q * max_cap));
    int32_t* d_steps = dalloc<int32_t>((size_t) (max_q * k::kStepCount));
    const size_t scr_f = (size_t) k::qsa_decode_attn_scratch_floats(max_cap, s);
    float* d_scr = dalloc<float>(scr_f * (size_t) max_q);
    float* d_ref = dalloc<float>((size_t) (max_q * NH * D));
    float* d_got = dalloc<float>((size_t) (max_q * NH * D));

    // Accuracy tolerance: mode 2 (hi+lo) achieves FP32-level (< 1e-5); mode 1 (1-pass FP16 Q/P) < 5e-3 max_abs, < 1e-3 rel_l2
    const double tol_abs = (v70_mode >= 2) ? 1e-5 : 5e-3;
    const double tol_rel = (v70_mode >= 2) ? 5e-6 : 1e-3;

    const int32_t widths[] = {0, 1, 3, 63, 64, 65, 255, 2051, 16384, 32768};
    const int32_t quick_widths[] = {131};
    int total_cases = 0, failed = 0;
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const std::vector<int64_t> caps = quick ? std::vector<int64_t>{2051} : std::vector<int64_t>{2051, 32768};
    const std::vector<int> nqs = quick ? std::vector<int>{2} : std::vector<int>{1, 2, 5, 6};

    for (int fmt = 0; fmt < 4; ++fmt) {
        double worst_abs = 0.0, worst_rel = 0.0;
        int fmt_cases = 0, fmt_fail = 0;
        for (int64_t cap : caps) {
            for (int n_q : nqs) {
                const int nw = quick ? 1 : (int) (sizeof(widths) / sizeof(widths[0]));
                const int n_extra = quick ? 0 : 2;
                for (int wc = 0; wc < nw + n_extra; ++wc) {
                    const int32_t w_tab = (wc < nw) ? (quick ? quick_widths[wc] : widths[wc]) : 0;
                    if (wc < nw && w_tab > cap) continue;
                    std::vector<float> hq((size_t) (n_q * NH * D));
                    if (!quick && wc == nw) {
                        std::fill(hq.begin(), hq.end(), 0.0f);   // zero-Q edge case
                    } else {
                        for (auto& x : hq) x = nd(rng);
                    }
                    std::vector<int32_t> hid((size_t) (n_q * cap), 0);
                    std::vector<int32_t> hst((size_t) (n_q * k::kStepCount), 0);
                    for (int qi = 0; qi < n_q; ++qi) {
                        int w = (wc < nw)
                                    ? w_tab
                                    : (wc == nw ? (int) std::min<int64_t>(cap, 2051)
                                                : (int) (rng() % (uint32_t) (cap + 1)));
                        int shape = (!quick && wc == nw + 1 && qi == 0) ? 2 : (int) (rng() % 2u);
                        make_sel(hid.data() + (size_t) qi * cap, cap, w, shape, rng);
                        hst[(size_t) qi * k::kStepCount + k::kStepWidth] = w;
                    }
                    ck(cudaMemcpy(d_q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
                    ck(cudaMemcpy(d_ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "ids");
                    ck(cudaMemcpy(d_steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");

                    // Reference: FP32 attn_chunk_kernel + attn_merge_kernel
                    ck(cudaMemset(d_scr, 0xFF, scr_f * (size_t) n_q * 4), "nan scr");
                    k::qsa_decode_attn_chunks_only(d_q, pools.view(fmt, d_table), d_ids, d_steps, cap, s, d_scr, n_q, 0, nullptr);
                    k::qsa_decode_attn_merge_only(d_steps, cap, s, d_scr, d_ref, n_q, 0, nullptr);

                    // Under test: production entry point
                    ck(cudaMemset(d_scr, 0xFF, scr_f * (size_t) n_q * 4), "nan scr2");
                    if (n_q == 1 && (wc & 1))
                        k::qsa_decode_attn_step(d_q, pools.view(fmt, d_table), d_ids, d_steps, cap, s, d_scr, d_got, nullptr);
                    else
                        k::qsa_decode_attn_batch(d_q, pools.view(fmt, d_table), d_ids, d_steps, cap, s, d_scr, d_got, n_q, nullptr);
                    ck(cudaDeviceSynchronize(), "sync");

                    const size_t n = (size_t) (n_q * NH * D);
                    std::vector<float> a(n), b(n);
                    ck(cudaMemcpy(a.data(), d_ref, n * 4, cudaMemcpyDeviceToHost), "ref");
                    ck(cudaMemcpy(b.data(), d_got, n * 4, cudaMemcpyDeviceToHost), "got");
                    const ErrStats es = compare_vecs(a, b);
                    worst_abs = std::max(worst_abs, es.max_abs);
                    worst_rel = std::max(worst_rel, es.rel_l2);
                    ++total_cases; ++fmt_cases;
                    if (es.nonfinite > 0 || es.max_abs > tol_abs || es.rel_l2 > tol_rel) {
                        ++failed; ++fmt_fail;
                        if (fmt_fail <= 3)
                            std::printf("  FAIL %s cap=%lld n_q=%d wc=%d: max_abs=%.3e rel_l2=%.3e nonfin=%lld\n",
                                        fmt_name(fmt), (long long) cap, n_q, wc, es.max_abs, es.rel_l2, es.nonfinite);
                    }
                }
            }
        }
        std::printf("  [%s] %d cases: max_abs=%.3e rel_l2=%.3e -> %s\n",
                    fmt_name(fmt), fmt_cases, worst_abs, worst_rel, fmt_fail ? "FAIL" : "PASS");
    }

    // CUDA Graph capture + replay across dynamic widths (k8v4, cap=2051, n_q=5)
    {
        cudaStream_t cs;
        ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "cs");
        const int n_q = 5;
        const int64_t cap = 2051;
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "cap begin");
        k::qsa_decode_attn_batch(d_q, pools.view(kFmtK8V4, d_table), d_ids, d_steps, cap, s, d_scr, d_got, n_q, cs);
        ck(cudaStreamEndCapture(cs, &g), "cap end");
        ck(cudaGraphInstantiate(&ge, g, 0), "inst");
        cudaGraphDestroy(g);

        double g_max = 0.0, g_rel = 0.0;
        int g_fail = 0;
        const std::vector<int> g_widths = quick ? std::vector<int>{65, 137} : std::vector<int>{0, 1, 64, 513, 2051, 137, 2051, 3};
        for (int w : g_widths) {
            std::vector<float> hq((size_t) (n_q * NH * D));
            for (auto& x : hq) x = nd(rng);
            std::vector<int32_t> hid((size_t) (n_q * cap), 0);
            std::vector<int32_t> hst((size_t) (n_q * k::kStepCount), 0);
            for (int qi = 0; qi < n_q; ++qi) {
                make_sel(hid.data() + (size_t) qi * cap, cap, w, 1, rng);
                hst[(size_t) qi * k::kStepCount + k::kStepWidth] = w;
            }
            ck(cudaMemcpy(d_q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "gq");
            ck(cudaMemcpy(d_ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "gids");
            ck(cudaMemcpy(d_steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "gsteps");
            k::qsa_decode_attn_chunks_only(d_q, pools.view(kFmtK8V4, d_table), d_ids, d_steps, cap, s, d_scr, n_q, 0, cs);
            k::qsa_decode_attn_merge_only(d_steps, cap, s, d_scr, d_ref, n_q, 0, cs);
            ck(cudaGraphLaunch(ge, cs), "glaunch");
            ck(cudaStreamSynchronize(cs), "gsync");
            const size_t n = (size_t) (n_q * NH * D);
            std::vector<float> a(n), b(n);
            ck(cudaMemcpy(a.data(), d_ref, n * 4, cudaMemcpyDeviceToHost), "ga");
            ck(cudaMemcpy(b.data(), d_got, n * 4, cudaMemcpyDeviceToHost), "gb");
            const ErrStats es = compare_vecs(a, b);
            g_max = std::max(g_max, es.max_abs);
            g_rel = std::max(g_rel, es.rel_l2);
            if (es.nonfinite > 0 || es.max_abs > tol_abs || es.rel_l2 > tol_rel) ++g_fail;
        }
        std::printf("  [cuda-graph k8v4 n_q=5] %zu dynamic replays: max_abs=%.3e rel_l2=%.3e -> %s\n",
                    g_widths.size(), g_max, g_rel, g_fail ? "FAIL" : "PASS");
        failed += g_fail;
        cudaGraphExecDestroy(ge);
        cudaStreamDestroy(cs);
    }

    if (do_bench) {
        std::printf("\n--attn bench (CUDA events, 200 iters, chunk+merge: V70 prod vs Daily pre75):\n");
        std::printf("%-6s %6s %4s %6s | %9s %9s %8s\n", "fmt", "cap", "n_q", "width", "pre75 us", "v70 us", "speedup");
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "e0");
        ck(cudaEventCreate(&e1), "e1");
        const int iters = 200;
        for (int fmt : {kFmtK8V4, kFmtInt8, kFmtQ4, kFmtF16}) {
            for (auto [cap, width] : std::vector<std::pair<int64_t, int>>{{2051, 2051}, {32768, 32768}}) {
                if (fmt != kFmtInt8 && fmt != kFmtK8V4 && cap == 32768) continue;
                for (int n_q : {1, 2, 4, 5, 6}) {
                    std::vector<float> hq((size_t) (n_q * NH * D));
                    for (auto& x : hq) x = nd(rng);
                    std::vector<int32_t> hid((size_t) (n_q * cap), 0);
                    std::vector<int32_t> hst((size_t) (n_q * k::kStepCount), 0);
                    for (int qi = 0; qi < n_q; ++qi) {
                        make_sel(hid.data() + (size_t) qi * cap, cap, width, cap == 32768 ? 0 : 1, rng);
                        hst[(size_t) qi * k::kStepCount + k::kStepWidth] = width;
                    }
                    ck(cudaMemcpy(d_q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "bq");
                    ck(cudaMemcpy(d_ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "bids");
                    ck(cudaMemcpy(d_steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "bsteps");
                    const auto pv = pools.view(fmt, d_table);

                    auto run_pre75 = [&] {
                        k::qsa_decode_attn_chunks_only(d_q, pv, d_ids, d_steps, cap, s, d_scr, n_q, 1, nullptr);
                        k::qsa_decode_attn_merge_only(d_steps, cap, s, d_scr, d_ref, n_q, 1, nullptr);
                    };
                    auto run_v70 = [&] {
                        k::qsa_decode_attn_batch(d_q, pv, d_ids, d_steps, cap, s, d_scr, d_got, n_q, nullptr);
                    };
                    auto time_us = [&](auto&& fn) {
                        for (int i = 0; i < 30; ++i) fn();
                        ck(cudaDeviceSynchronize(), "warm");
                        ck(cudaEventRecord(e0, nullptr), "e0");
                        for (int i = 0; i < iters; ++i) fn();
                        ck(cudaEventRecord(e1, nullptr), "e1");
                        ck(cudaEventSynchronize(e1), "sync");
                        float ms = 0.0f;
                        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                        return 1000.0 * ms / iters;
                    };
                    const double t_pre = time_us(run_pre75);
                    const double t_v70 = time_us(run_v70);
                    std::printf("%-6s %6lld %4d %6d | %9.2f %9.2f %7.2fx\n",
                                fmt_name(fmt), (long long) cap, n_q, width, t_pre, t_v70, t_pre / t_v70);
                }
            }
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        bench_kv_stream(s, rng);
    }

    std::printf("qsa_decode_attn_v70_parity: %d cases, %d failed -> %s\n", total_cases, failed, failed ? "FAIL" : "PASS");
    return failed == 0 ? 0 : 1;
}
