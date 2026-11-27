// src/platform/direct_file.cpp - see include/strata/platform/direct_file.hpp.
#include "strata/platform/direct_file.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/io_uring.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#endif
#endif

namespace strata::platform {

double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

void* DirectFile::alloc_aligned(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, alignment(), bytes) == 0 ? p : nullptr;
#endif
}

void DirectFile::free_aligned(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

namespace {
/// A queued read: `submit` only queues it; the pool's threads issue it (perf-review C-4 / F-2).  One thread
/// issuing every read was the limit of the n-gram table's prompt reads: ~12 us of kernel time per read, ~80K
/// reads/s, while an NVMe drive serves several times that at depth.
struct Pending {
    uint64_t offset;
    void* buffer;
    uint32_t length;
    uint64_t tag;
};

/// The number of issuing threads: STRATA_IO_THREADS, else 4 (Windows: overlapped submits) / 16 (Linux: each
/// thread does one blocking pread, so the thread count is the queue depth).
int io_threads(int dflt) {
    const char* v = std::getenv("STRATA_IO_THREADS");
    const int n = v ? std::atoi(v) : dflt;
    return std::clamp(n, 1, 64);
}
}  // namespace

#if defined(_WIN32)
// ------------------------------------------------------------------------------------------------ Windows
namespace {
/// One in-flight request. The OVERLAPPED must be the first member so a completion packet's OVERLAPPED*
/// converts back to the request.
struct Req {
    OVERLAPPED ov;
    uint64_t tag;
};
}  // namespace

struct DirectFile::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE port = nullptr;
    uint64_t size = 0;
    std::mutex mu;                      // everything below
    std::deque<Req*> free_reqs;
    std::vector<Req*> all_reqs;
    std::deque<Completion> immediate;   // requests that completed without a port packet (EOF) or failed
    std::deque<Pending> queue;          // submitted, not yet issued
    std::condition_variable cv;
    bool stop = false;
    std::vector<std::thread> pool;

    Req* take() {
        if (free_reqs.empty()) {
            Req* r = new Req();
            all_reqs.push_back(r);
            return r;
        }
        Req* r = free_reqs.front();
        free_reqs.pop_front();
        return r;
    }

    void issue(const Pending& p) {
        Req* r;
        {
            std::lock_guard<std::mutex> lk(mu);
            r = take();
        }
        std::memset(&r->ov, 0, sizeof r->ov);
        r->ov.Offset = (DWORD) (p.offset & 0xFFFFFFFFull);
        r->ov.OffsetHigh = (DWORD) (p.offset >> 32);
        r->tag = p.tag;
        if (!ReadFile(file, p.buffer, p.length, nullptr, &r->ov)) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) {
                std::lock_guard<std::mutex> lk(mu);
                free_reqs.push_back(r);
                // at or past end of file: a zero-byte completion; anything else: a failed one
                immediate.push_back(Completion{p.tag, 0, e == ERROR_HANDLE_EOF});
                PostQueuedCompletionStatus(port, 0, 0, nullptr);   // wake a waiter to collect it
            }
        }
    }

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            issue(p);
            lk.lock();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() {
    close();
    for (Req* r : impl_->all_reqs) delete r;
    delete impl_;
}

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath((size_t) (wlen > 0 ? wlen : 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    // NO_BUFFERING: the cache manager never holds these pages, so the table cannot grow into RAM.
    // RANDOM_ACCESS: no read-ahead. OVERLAPPED: many reads in flight, completed through the port below.
    impl_->file = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) {
        err = "DirectFile: cannot open " + path + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(impl_->file, &sz)) {
        err = "DirectFile: cannot size " + path;
        close();
        return false;
    }
    impl_->size = (uint64_t) sz.QuadPart;
    impl_->port = CreateIoCompletionPort(impl_->file, nullptr, 0, 1);
    if (impl_->port == nullptr) {
        err = "DirectFile: CreateIoCompletionPort failed (error " + std::to_string(GetLastError()) + ")";
        close();
        return false;
    }
    // Completions of reads that finish synchronously are still queued to the port, so every issued read
    // produces exactly one packet; `wait` is the only completion path.
    const int n = io_threads(4);
    for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { strata::aux_cpus::pin_current_thread(); impl_->worker(); });
    return true;
}

void DirectFile::close() {
    impl_->stop_pool();
    if (impl_->port != nullptr) CloseHandle(impl_->port);
    if (impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file);
    impl_->port = nullptr;
    impl_->file = INVALID_HANDLE_VALUE;
    impl_->size = 0;
    impl_->immediate.clear();
    impl_->queue.clear();
}

bool DirectFile::is_open() const { return impl_->file != INVALID_HANDLE_VALUE; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
    }
    impl_->cv.notify_one();
    return true;
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        while (n < max && !impl_->immediate.empty()) {
            out[n++] = impl_->immediate.front();
            impl_->immediate.pop_front();
        }
    }
    if (n == max || !is_open()) return n;
    OVERLAPPED_ENTRY entries[64];
    const ULONG want = (ULONG) (max - n < 64 ? max - n : 64);
    ULONG got = 0;
    const DWORD t = timeout_ms < 0 ? INFINITE : (DWORD) timeout_ms;
    if (!GetQueuedCompletionStatusEx(impl_->port, entries, want, &got, n > 0 ? 0 : t, FALSE)) return n;
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (ULONG i = 0; i < got; ++i) {
        if (entries[i].lpOverlapped == nullptr) {          // a wake() packet (or an immediate completion's nudge)
            while (n < max && !impl_->immediate.empty()) {
                out[n++] = impl_->immediate.front();
                impl_->immediate.pop_front();
            }
            if (n < max) out[n++] = Completion{WAKE_TAG, 0, true};
            continue;
        }
        Req* r = (Req*) entries[i].lpOverlapped;
        const uint32_t status = (uint32_t) r->ov.Internal;   // an NTSTATUS
        Completion c;
        c.tag = r->tag;
        c.bytes = entries[i].dwNumberOfBytesTransferred;
        // STATUS_END_OF_FILE (0xC0000011) is a legal short read at the table's last page.
        c.ok = status == 0 || status == 0xC0000011u;
        out[n++] = c;
        impl_->free_reqs.push_back(r);
    }
    return n;
}

void DirectFile::flush() {}
bool DirectFile::using_uring() const { return false; }

void DirectFile::wake() {
    if (impl_->port != nullptr) PostQueuedCompletionStatus(impl_->port, 0, 0, nullptr);
}

#else
// ------------------------------------------------------------------------------------------------ POSIX
// PR 6.3a: On Linux, prefer raw io_uring batch-submission (zero liburing dependency, kill-switch
// STRATA_PLE_URING=0) over the 16-thread blocking pread pool. Falls back cleanly to pread pool on
// non-Linux POSIX or if io_uring_setup is unavailable.
#if defined(__linux__) && defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter)
#define STRATA_HAS_IO_URING 1
namespace {
bool uring_enabled_by_env() {
    const char* v = std::getenv("STRATA_PLE_URING");
    return v == nullptr || std::atoi(v) != 0;
}
int sys_io_uring_setup(unsigned entries, struct io_uring_params* p) {
    return (int) ::syscall(__NR_io_uring_setup, entries, p);
}
int sys_io_uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags, const void* arg, size_t argsz) {
    return (int) ::syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, arg, argsz);
}
}  // namespace
#else
#define STRATA_HAS_IO_URING 0
#endif

struct DirectFile::Impl {
    int fd = -1;
    uint64_t size = 0;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::deque<Pending> queue;
    std::deque<Completion> done;
    bool stop = false;
    std::vector<std::thread> pool;

#if STRATA_HAS_IO_URING
    bool use_uring = false;
    int ring_fd = -1;
    int wake_fd = -1;
    uint64_t wake_buf = 0;
    std::atomic<bool> wake_signaled{false};
    bool wake_armed = false;
    bool single_mmap = false;
    void* sq_ring_ptr = MAP_FAILED;
    size_t sq_ring_sz = 0;
    void* cq_ring_ptr = MAP_FAILED;
    size_t cq_ring_sz = 0;
    struct io_uring_sqe* sqes = nullptr;
    size_t sqes_sz = 0;
    uint32_t* k_sq_head = nullptr;
    uint32_t* k_sq_tail = nullptr;
    uint32_t* k_sq_mask = nullptr;
    uint32_t* k_sq_array = nullptr;
    uint32_t sq_entries = 0;
    uint32_t sq_local_tail = 0;
    uint32_t sq_pending = 0;
    uint32_t* k_cq_head = nullptr;
    uint32_t* k_cq_tail = nullptr;
    uint32_t* k_cq_mask = nullptr;
    struct io_uring_cqe* k_cqes = nullptr;
    uint64_t uring_inflight = 0;   // user reads submitted (or queued in SQ) not yet reaped
    bool ext_arg_ok = false;

    void close_uring() {
        if (use_uring) {
            // Flush any queued SQEs and drain every in-flight user read before returning so no
            // kernel DMA can touch caller buffers after DirectFile::close().
            flush_uring_locked();
            while (uring_inflight > 0) {
                Completion tmp[64];
                const int n = wait_uring(tmp, 64, -1);
                if (n <= 0) break;
            }
        }
        if (sqes != nullptr && sqes != MAP_FAILED) ::munmap(sqes, sqes_sz);
        sqes = nullptr; sqes_sz = 0;
        if (!single_mmap && cq_ring_ptr != MAP_FAILED) ::munmap(cq_ring_ptr, cq_ring_sz);
        cq_ring_ptr = MAP_FAILED; cq_ring_sz = 0;
        if (sq_ring_ptr != MAP_FAILED) ::munmap(sq_ring_ptr, sq_ring_sz);
        sq_ring_ptr = MAP_FAILED; sq_ring_sz = 0;
        if (ring_fd >= 0) ::close(ring_fd);
        ring_fd = -1;
        if (wake_fd >= 0) ::close(wake_fd);
        wake_fd = -1;
        use_uring = false;
        wake_armed = false;
        wake_signaled.store(false, std::memory_order_relaxed);
        single_mmap = false;
        sq_entries = sq_local_tail = sq_pending = 0;
        uring_inflight = 0;
        k_sq_head = k_sq_tail = k_sq_mask = k_sq_array = nullptr;
        k_cq_head = k_cq_tail = k_cq_mask = nullptr;
        k_cqes = nullptr;
    }

    bool open_uring() {
        if (!uring_enabled_by_env()) return false;
        struct io_uring_params p {};
        const int rfd = sys_io_uring_setup(512, &p);
        if (rfd < 0) return false;
        ring_fd = rfd;
        single_mmap = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
        ext_arg_ok = (p.features & IORING_FEAT_EXT_ARG) != 0;
        sq_ring_sz = p.sq_off.array + (size_t) p.sq_entries * sizeof(uint32_t);
        cq_ring_sz = p.cq_off.cqes + (size_t) p.cq_entries * sizeof(struct io_uring_cqe);
        if (single_mmap) {
            if (cq_ring_sz > sq_ring_sz) sq_ring_sz = cq_ring_sz;
            cq_ring_sz = sq_ring_sz;
        }
        sq_ring_ptr = ::mmap(nullptr, sq_ring_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
        if (sq_ring_ptr == MAP_FAILED) { close_uring(); return false; }
        if (single_mmap) {
            cq_ring_ptr = sq_ring_ptr;
        } else {
            cq_ring_ptr = ::mmap(nullptr, cq_ring_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
            if (cq_ring_ptr == MAP_FAILED) { close_uring(); return false; }
        }
        sqes_sz = (size_t) p.sq_entries * sizeof(struct io_uring_sqe);
        void* sp = ::mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQES);
        if (sp == MAP_FAILED) { close_uring(); return false; }
        sqes = static_cast<struct io_uring_sqe*>(sp);
        uint8_t* sqb = static_cast<uint8_t*>(sq_ring_ptr);
        uint8_t* cqb = static_cast<uint8_t*>(cq_ring_ptr);
        k_sq_head  = reinterpret_cast<uint32_t*>(sqb + p.sq_off.head);
        k_sq_tail  = reinterpret_cast<uint32_t*>(sqb + p.sq_off.tail);
        k_sq_mask  = reinterpret_cast<uint32_t*>(sqb + p.sq_off.ring_mask);
        k_sq_array = reinterpret_cast<uint32_t*>(sqb + p.sq_off.array);
        k_cq_head  = reinterpret_cast<uint32_t*>(cqb + p.cq_off.head);
        k_cq_tail  = reinterpret_cast<uint32_t*>(cqb + p.cq_off.tail);
        k_cq_mask  = reinterpret_cast<uint32_t*>(cqb + p.cq_off.ring_mask);
        k_cqes     = reinterpret_cast<struct io_uring_cqe*>(cqb + p.cq_off.cqes);
        sq_entries = p.sq_entries;
        sq_local_tail = *k_sq_tail;
        sq_pending = 0;
        uring_inflight = 0;
        wake_fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (wake_fd < 0) { close_uring(); return false; }
        wake_armed = false;
        wake_signaled.store(false, std::memory_order_relaxed);
        use_uring = true;
        arm_wake_locked();
        flush_uring_locked();
        return true;
    }

    bool push_sqe_locked(const struct io_uring_sqe& sqe) {
        const uint32_t head = __atomic_load_n(k_sq_head, __ATOMIC_ACQUIRE);
        if ((uint32_t) (sq_local_tail - head) >= sq_entries) {
            if (!flush_uring_locked()) return false;
            const uint32_t head2 = __atomic_load_n(k_sq_head, __ATOMIC_ACQUIRE);
            if ((uint32_t) (sq_local_tail - head2) >= sq_entries) return false;
        }
        const uint32_t idx = sq_local_tail & *k_sq_mask;
        sqes[idx] = sqe;
        k_sq_array[idx] = idx;
        ++sq_local_tail;
        ++sq_pending;
        return true;
    }

    void arm_wake_locked() {
        if (wake_armed || wake_fd < 0) return;
        struct io_uring_sqe sqe {};
        sqe.opcode = IORING_OP_READ;
        sqe.fd = wake_fd;
        sqe.off = 0;
        sqe.addr = (uint64_t) (uintptr_t) &wake_buf;
        sqe.len = sizeof(wake_buf);
        sqe.user_data = DirectFile::WAKE_TAG;
        if (push_sqe_locked(sqe)) wake_armed = true;
    }

    bool flush_uring_locked() {
        if (!use_uring) return false;
        if (!wake_armed) arm_wake_locked();
        if (sq_pending == 0) return true;
        __atomic_store_n(k_sq_tail, sq_local_tail, __ATOMIC_RELEASE);
        while (sq_pending > 0) {
            const int r = sys_io_uring_enter(ring_fd, sq_pending, 0, 0, nullptr, 0);
            if (r > 0) {
                sq_pending -= (uint32_t) r;
            } else if (r < 0 && errno == EINTR) {
                continue;
            } else {
                // EBUSY/EAGAIN when CQ overflows: stop submitting and let wait() reap
                return r >= 0;
            }
        }
        return true;
    }

    int reap_cq_locked(Completion* out, int max) {
        uint32_t head = *k_cq_head;
        const uint32_t tail = __atomic_load_n(k_cq_tail, __ATOMIC_ACQUIRE);
        const uint32_t mask = *k_cq_mask;
        int n = 0;
        while (n < max && head != tail) {
            const struct io_uring_cqe& cqe = k_cqes[head & mask];
            ++head;
            if (cqe.user_data == DirectFile::WAKE_TAG) {
                wake_armed = false;
                wake_signaled.store(false, std::memory_order_release);
                out[n++] = Completion{DirectFile::WAKE_TAG, 0, true};
                continue;
            }
            if (uring_inflight > 0) --uring_inflight;
            out[n++] = Completion{cqe.user_data, cqe.res >= 0 ? (uint32_t) cqe.res : 0u, cqe.res >= 0};
        }
        if (n > 0) __atomic_store_n(k_cq_head, head, __ATOMIC_RELEASE);
        return n;
    }

    int wait_uring(Completion* out, int max, int timeout_ms) {
        if (max <= 0 || !use_uring) return 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            flush_uring_locked();
            const int got = reap_cq_locked(out, max);
            if (got > 0 || timeout_ms == 0) return got;
        }
        // Block outside `mu` so concurrent submit() cannot deadlock (wake() uses lockless eventfd).
        for (;;) {
            int r = 0;
            if (timeout_ms < 0) {
                r = sys_io_uring_enter(ring_fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr, 0);
            } else if (ext_arg_ok) {
                struct __kernel_timespec ts {};
                ts.tv_sec = timeout_ms / 1000;
                ts.tv_nsec = (long long) (timeout_ms % 1000) * 1000000LL;
                struct io_uring_getevents_arg arg {};
                arg.ts = (uint64_t) (uintptr_t) &ts;
                r = sys_io_uring_enter(ring_fd, 0, 1, IORING_ENTER_GETEVENTS | IORING_ENTER_EXT_ARG, &arg, sizeof(arg));
            } else {
                r = sys_io_uring_enter(ring_fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr, 0);
            }
            if (r < 0 && errno == EINTR) continue;
            break;
        }
        std::lock_guard<std::mutex> lk(mu);
        return reap_cq_locked(out, max);
    }
#endif

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_work.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            const ssize_t got = pread(fd, p.buffer, p.length, (off_t) p.offset);
            lk.lock();
            done.push_back(Completion{p.tag, got < 0 ? 0u : (uint32_t) got, got >= 0});
            cv_done.notify_all();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_work.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() { close(); delete impl_; }

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
#if defined(__APPLE__)   // macOS has no O_DIRECT: F_NOCACHE keeps the reads out of the unified buffer cache
    impl_->fd = ::open(path.c_str(), O_RDONLY);
    if (impl_->fd >= 0 && fcntl(impl_->fd, F_NOCACHE, 1) != 0) { ::close(impl_->fd); impl_->fd = -1; }
#else
    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
#endif
    if (impl_->fd < 0) { err = "DirectFile: cannot open " + path; return false; }
    struct stat st;
    if (fstat(impl_->fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    impl_->size = (uint64_t) st.st_size;
#if STRATA_HAS_IO_URING
    if (impl_->open_uring()) return true;
#endif
    const int n = io_threads(16);
    for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { strata::aux_cpus::pin_current_thread(); impl_->worker(); });
    return true;
}

void DirectFile::close() {
#if STRATA_HAS_IO_URING
    if (impl_->use_uring) impl_->close_uring();
#endif
    impl_->stop_pool();
    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
    impl_->size = 0;
    impl_->done.clear();
    impl_->queue.clear();
}

bool DirectFile::is_open() const { return impl_->fd >= 0; }
uint64_t DirectFile::size() const { return impl_->size; }
bool DirectFile::using_uring() const {
#if STRATA_HAS_IO_URING
    return impl_->use_uring;
#else
    return false;
#endif
}

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
#if STRATA_HAS_IO_URING
    if (impl_->use_uring) {
        std::lock_guard<std::mutex> lk(impl_->mu);
        if (!impl_->wake_armed) impl_->arm_wake_locked();
        // Reserve 1 slot for wake rearm; flush if SQ is nearly full.
        if (impl_->sq_pending + 2 >= impl_->sq_entries) impl_->flush_uring_locked();
        struct io_uring_sqe sqe {};
        sqe.opcode = IORING_OP_READ;
        sqe.fd = impl_->fd;
        sqe.off = offset;
        sqe.addr = (uint64_t) (uintptr_t) buffer;
        sqe.len = length;
        sqe.user_data = tag;
        if (!impl_->push_sqe_locked(sqe)) {
            err = "DirectFile: io_uring submission queue full";
            return false;
        }
        ++impl_->uring_inflight;
        return true;
    }
#endif
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
    }
    impl_->cv_work.notify_one();
    return true;
}

void DirectFile::flush() {
#if STRATA_HAS_IO_URING
    if (impl_->use_uring) {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->flush_uring_locked();
    }
#endif
}

void DirectFile::wake() {
#if STRATA_HAS_IO_URING
    if (impl_->use_uring) {
        if (!impl_->wake_signaled.exchange(true, std::memory_order_acq_rel) && impl_->wake_fd >= 0) {
            const uint64_t one = 1;
            const ssize_t r = ::write(impl_->wake_fd, &one, sizeof(one));
            (void) r;
        }
        return;
    }
#endif
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->done.push_back(Completion{WAKE_TAG, 0, true});
    impl_->cv_done.notify_all();
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
#if STRATA_HAS_IO_URING
    if (impl_->use_uring) return impl_->wait_uring(out, max, timeout_ms);
#endif
    std::unique_lock<std::mutex> lk(impl_->mu);
    if (impl_->done.empty() && timeout_ms != 0) {
        auto ready = [&] { return !impl_->done.empty(); };
        if (timeout_ms < 0) impl_->cv_done.wait(lk, ready);
        else impl_->cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    }
    int n = 0;
    while (n < max && !impl_->done.empty()) {
        out[n++] = impl_->done.front();
        impl_->done.pop_front();
    }
    return n;
}
#endif

}  // namespace strata::platform
