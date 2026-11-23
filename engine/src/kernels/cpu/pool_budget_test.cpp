// CPU-only tests for dynamic ExpertPool worker budgets via shared memfd (PR 1.1).
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace cpu = strata::kernels::cpu;

namespace {

int g_checks = 0;
void check(bool ok, const char* msg) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1);
    else unsetenv(k);
#endif
}

struct RangeCtx {
    uint64_t* out = nullptr;
    uint64_t salt = 0;
    int busy_iters = 0;
};

void range_task(void* p, int idx) {
    auto* ctx = static_cast<RangeCtx*>(p);
    uint64_t v = (uint64_t) idx * 0x9e3779b97f4a7c15ull ^ ctx->salt;
    for (int k = 0; k < ctx->busy_iters; ++k) {
        v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ull;
    }
    ctx->out[idx] = v;
}

#if defined(__linux__)
std::map<std::string, uint64_t> read_pool_thread_ticks() {
    std::map<std::string, uint64_t> ticks;
    DIR* dir = ::opendir("/proc/self/task");
    if (!dir) return ticks;
    while (struct dirent* ent = ::readdir(dir)) {
        if (ent->d_name[0] == '.') continue;
        std::string base = std::string("/proc/self/task/") + ent->d_name;
        char comm[64] = {};
        if (FILE* fc = std::fopen((base + "/comm").c_str(), "r")) {
            if (std::fgets(comm, sizeof comm, fc)) {
                size_t len = std::strlen(comm);
                while (len > 0 && (comm[len - 1] == '\n' || comm[len - 1] == '\r')) comm[--len] = '\0';
            }
            std::fclose(fc);
        }
        if (std::strncmp(comm, "strata-pool-", 12) != 0) continue;
        if (FILE* fs = std::fopen((base + "/stat").c_str(), "r")) {
            char buf[1024] = {};
            const size_t n = std::fread(buf, 1, sizeof(buf) - 1, fs);
            std::fclose(fs);
            buf[n] = '\0';
            const char* rp = std::strrchr(buf, ')');
            if (rp && rp[1] == ' ') {
                unsigned long long utime = 0, stime = 0;
                const char* cur = rp + 2;
                int field = 3;
                while (*cur) {
                    while (*cur == ' ') ++cur;
                    if (!*cur) break;
                    char* end = nullptr;
                    if (field == 14) utime = std::strtoull(cur, &end, 10);
                    else if (field == 15) { stime = std::strtoull(cur, &end, 10); break; }
                    else { while (*cur && *cur != ' ') ++cur; }
                    if (field == 14 && end) cur = end;
                    ++field;
                }
                ticks[comm] = (uint64_t) (utime + stime);
            }
        }
    }
    ::closedir(dir);
    return ticks;
}
#endif

}  // namespace

int main() {
    // 1. Basic PoolBudgetSlot clamping, telemetry, and memfd mapping
    {
        cpu::PoolBudgetSlot slot;
        check(slot.target_workers.load() == 0, "default target_workers is 0");
        cpu::ExpertPool pool(8, /*pin=*/false, /*host_works=*/true, cpu::PoolAffinity::All, 0, &slot);
        check(pool.workers() == 8, "pool has 8 workers");
        check(pool.active_workers() == 8, "target 0 starts full pool");
        check(slot.applied_workers.load() == 8, "applied_workers initialized to 8");
        check(slot.changes.load() == 0, "initial changes is 0");

        std::vector<uint64_t> out(32, 0), ref(32, 0);
        RangeCtx ctx{out.data(), 123, 0};
        RangeCtx ref_ctx{ref.data(), 123, 0};
        for (int i = 0; i < 32; ++i) range_task(&ref_ctx, i);

        pool.run_range(32, range_task, &ctx);
        check(out == ref, "full pool run_range output matches");
        check(slot.changes.load() == 0, "no changes when target remains 0");

        slot.target_workers.store(3, std::memory_order_relaxed);
        pool.run_range(32, range_task, &ctx);
        check(out == ref && pool.active_workers() == 3, "shrunk to 3 workers");
        check(slot.applied_workers.load() == 3 && slot.changes.load() == 1, "telemetry recorded shrink to 3");

        slot.target_workers.store(999, std::memory_order_relaxed);
        pool.run_range(32, range_task, &ctx);
        check(out == ref && pool.active_workers() == 8, "clamped above n_ to 8");
        check(slot.applied_workers.load() == 8 && slot.changes.load() == 2, "telemetry recorded grow to 8");

        // Shrink down to 1 and destroy pool cleanly (must wake all 7 budget-parked workers)
        slot.target_workers.store(1, std::memory_order_relaxed);
        pool.run_range(32, range_task, &ctx);
        check(out == ref && pool.active_workers() == 1 && slot.applied_workers.load() == 1, "shrunk to 1 before dtor");
    }

#if defined(__linux__)
    // 2. Real Linux memfd_create + /proc/self/fd/<fd> + PoolBudgetMapping + error validation
    {
        const int fd = ::memfd_create("strata-pool-budget-test", MFD_CLOEXEC);
        check(fd >= 0, "memfd_create succeeds");
        check(::ftruncate(fd, 4096) == 0, "ftruncate 4096");
        void* p = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        check(p != MAP_FAILED, "parent mmap succeeds");
        std::memset(p, 0, 4096);
        auto* parent_slots = reinterpret_cast<cpu::PoolBudgetSlot*>(p);
        parent_slots[1].target_workers.store(4, std::memory_order_relaxed);

        const std::string fd_path = "/proc/self/fd/" + std::to_string(fd);
        std::string err;
        cpu::PoolBudgetMapping bad_slot;
        check(!bad_slot.open(fd_path, 64, err) && bad_slot.slot() == nullptr, "rejects out-of-range slot");
        cpu::PoolBudgetMapping bad_path;
        check(!bad_path.open("/nonexistent/strata/budget", 0, err) && bad_path.slot() == nullptr, "rejects missing path");

        cpu::PoolBudgetMapping map;
        check(map.open(fd_path, 1, err) && map.slot() != nullptr, "PoolBudgetMapping opens slot 1 via memfd");
        {
            cpu::ExpertPool pool(8, false, true, cpu::PoolAffinity::All, 0, map.slot());
            check(pool.active_workers() == 4 && parent_slots[1].applied_workers.load() == 4, "honors pre-set target=4");
            std::vector<uint64_t> out(24, 0);
            RangeCtx ctx{out.data(), 77, 0};
            pool.run_range(24, range_task, &ctx);
            parent_slots[1].target_workers.store(8, std::memory_order_release);
            pool.run_range(24, range_task, &ctx);
            check(pool.active_workers() == 8 && parent_slots[1].applied_workers.load() == 8, "reflects grow via memfd");
            check(parent_slots[1].changes.load() == 1, "parent observes transition count");
        }
        map.close();
        ::munmap(p, 4096);
        ::close(fd);
    }
#endif

    // 3. Chaotic mid-flight budget stress: 30 workers, 10,000 batches + short spin (exercises both sleep_cv and budget_cv)
    {
        set_env("STRATA_POOL_SPIN_US", "100");
        cpu::PoolBudgetSlot slot;
        slot.target_workers.store(15, std::memory_order_relaxed);
        cpu::ExpertPool pool(30, /*pin=*/false, /*host_works=*/true, cpu::PoolAffinity::All, 0, &slot);

        constexpr int kBatches = 10000;
        constexpr int kTasks = 48;
        std::vector<uint64_t> out(kTasks), ref(kTasks);
        std::atomic<bool> stop_mutator{false};
        std::thread mutator([&] {
            std::mt19937 rng(42);
            const uint32_t choices[] = {1, 4, 15, 29, 30};
            while (!stop_mutator.load(std::memory_order_relaxed)) {
                slot.target_workers.store(choices[rng() % 5], std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        });

        for (int b = 0; b < kBatches; ++b) {
            const uint64_t salt = 0x10000000ull + (uint64_t) b;
            RangeCtx ctx{out.data(), salt, 4};
            RangeCtx rctx{ref.data(), salt, 4};
            std::fill(out.begin(), out.end(), 0);
            for (int i = 0; i < kTasks; ++i) range_task(&rctx, i);
            pool.run_range(kTasks, range_task, &ctx);
            if (out != ref) {
                stop_mutator.store(true);
                mutator.join();
                check(false, "chaotic run_range output mismatch");
            }
        }

        // Also stress run_split_multi_native under the same chaotic mutator when AVX1 is available
        if (cpu::cpu_avx1_ok()) {
            cpu::NativeFmt f;
            std::string err;
            check(cpu::native_fmt(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, cpu::H, cpu::FF, f, err), "native_fmt Q4_0");
            std::vector<uint8_t> blob(f.bytes);
            std::vector<float> weights((size_t) cpu::H * cpu::FF);
            for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.02f * std::sin((float) i * 0.13f);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data(), 0, cpu::FF, cpu::H, nullptr);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
            std::vector<std::vector<uint8_t>> act(2, std::vector<uint8_t>(cpu::kNativeActBytes));
            std::vector<float> x(cpu::H);
            for (int t = 0; t < 2; ++t) {
                for (int i = 0; i < cpu::H; ++i) x[i] = std::sin((float) (i + 7 * t) * 0.041f);
                cpu::native_quant_act(f, x.data(), act[t].data());
            }
            constexpr int kExp = 6;
            std::vector<float> eout((size_t) kExp * cpu::MAXT * cpu::H, 0.0f);
            std::vector<cpu::ExpertJobMulti> jobs(kExp);
            for (int e = 0; e < kExp; ++e) {
                jobs[e].blob = blob.data();
                jobs[e].nt = (e % 2) + 1;
                for (int t = 0; t < jobs[e].nt; ++t) {
                    jobs[e].nact[t] = act[t].data();
                    jobs[e].out[t] = eout.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
                }
            }
            pool.run_split_multi_native(f, jobs.data(), kExp);
            const std::vector<float> eref = eout;
            for (int b = 0; b < 300; ++b) {
                std::fill(eout.begin(), eout.end(), 0.0f);
                pool.run_split_multi_native(f, jobs.data(), kExp);
                if (std::memcmp(eout.data(), eref.data(), eout.size() * sizeof(float)) != 0) {
                    stop_mutator.store(true);
                    mutator.join();
                    check(false, "chaotic run_split_multi_native output mismatch");
                }
            }
        }

        stop_mutator.store(true);
        mutator.join();
        check(slot.changes.load() > 10, "mutator exercised budget transitions under stress");
        slot.target_workers.store(1, std::memory_order_relaxed);
        RangeCtx shrink_ctx{out.data(), 999, 0};
        pool.run_range(kTasks, range_task, &shrink_ctx);
        set_env("STRATA_POOL_SPIN_US", nullptr);
    }

#if defined(__linux__)
    // 4. Zero-CPU dormant verification: workers >= active_limit_ sleep in budget_cv_ with ~0% CPU
    {
        cpu::PoolBudgetSlot slot;
        slot.target_workers.store(2, std::memory_order_relaxed);
        cpu::ExpertPool pool(16, /*pin=*/false, /*host_works=*/true, cpu::PoolAffinity::All, 0, &slot);

        std::vector<uint64_t> out(8);
        RangeCtx ctx{out.data(), 555, 2000};
        for (int i = 0; i < 50; ++i) pool.run_range(8, range_task, &ctx);

        const uint64_t changes_before = slot.changes.load();
        const auto t_before = read_pool_thread_ticks();
        const auto wall0 = std::chrono::steady_clock::now();
        for (int b = 0; b < 2000; ++b) {
            ctx.salt = (uint64_t) b;
            pool.run_range(8, range_task, &ctx);
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
        const auto t_after = read_pool_thread_ticks();
        check(slot.changes.load() == changes_before, "no spurious budget transitions during steady run");

        uint64_t active_ticks = 0, dormant_ticks = 0;
        for (int i = 0; i < 16; ++i) {
            const std::string name = "strata-pool-" + std::to_string(i);
            const auto it0 = t_before.find(name);
            const auto it1 = t_after.find(name);
            if (it0 == t_before.end() || it1 == t_after.end()) continue;
            const uint64_t dt = it1->second >= it0->second ? (it1->second - it0->second) : 0;
            if (i < 2) active_ticks += dt;
            else dormant_ticks += dt;
        }
        std::printf("pool_budget_test: steady 2/16 workers over 2000 batches (%.1f ms): active_ticks=%llu dormant_ticks=%llu\n",
                    wall_ms, (unsigned long long) active_ticks, (unsigned long long) dormant_ticks);
        check(!t_before.empty(), "observed strata-pool-* threads in /proc/self/task");
        check(dormant_ticks <= 2, "dormant workers consume ~0% CPU");
    }
#endif

    std::printf("pool_budget_test: %d checks passed\n", g_checks);
    return 0;
}
