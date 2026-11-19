// CPU-only tests for the conversation disk tier's two-folder mode (the replica handoff built on #5835): a writer
// engine's folder read by a peer engine without a restart, the leases that keep a snapshot from being removed under
// a reader, and the read-only guarantees of the peer.  POSIX only: on Windows the peer mode is refused by design.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<int32_t> tokens(size_t n, int32_t base) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = base + int32_t(i % 1000);
    return v;
}

ConversationCheckpoint checkpoint(const std::vector<int32_t>& ids, size_t n, uint8_t seed) {
    ConversationCheckpoint c;
    c.ids.assign(ids.begin(), ids.begin() + (std::ptrdiff_t) n);
    c.gdn.assign(4099, seed);
    c.ple.assign(77, uint8_t(seed + 1));
    c.tails.assign(301, uint8_t(seed + 2));
    c.dead.assign(64, uint8_t(seed + 3));
    c.block_pos.assign(8, uint8_t(seed + 4));
    c.used = seed;
    return c;
}

SavedConversation stage_image(const std::vector<int32_t>& ids, size_t at, int64_t lo, int64_t hi, uint8_t seed) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = lo; s.layer_hi = hi;
    s.live = checkpoint(ids, ids.size(), seed);
    s.checkpoints.push_back(checkpoint(ids, at, uint8_t(seed + 10)));
    ConversationKv kv;
    kv.format = 2; kv.cells = (int64_t) ids.size(); kv.heads = 2; kv.head_dim = 256; kv.page_size = 64; kv.idx_dim = 128;
    kv.k = pattern(65536 + seed, seed);
    kv.v = pattern(4096, uint8_t(seed + 1));
    s.kv.push_back(std::move(kv));
    return s;
}

SavedConversation split_image(const std::vector<int32_t>& ids, size_t at, uint8_t seed) {
    SavedConversation s = stage_image(ids, at, 0, 19, seed);
    s.stage_images.push_back(stage_image(ids, at, 19, 48, uint8_t(seed + 50)));
    return s;
}

ConversationDiskLayout split_layout() {
    ConversationDiskLayout l;
    l.stages = {{0, 19}, {19, 48}};
    return l;
}

std::vector<int32_t> prompt_of(const std::vector<int32_t>& ids, size_t extra) {
    std::vector<int32_t> p = ids;
    for (size_t i = 0; i < extra; ++i) p.push_back(7);
    return p;
}

std::string snapshot_path(const std::string& folder, uint64_t id, const char* suffix) {
    char name[32];
    std::snprintf(name, sizeof name, "%016llx%s", (unsigned long long) id, suffix);
    return (fs::path(folder) / name).string();
}

/// Every regular file's name and size in a folder: what a read-only peer must not change.
std::vector<std::pair<std::string, uintmax_t>> listing(const std::string& folder) {
    std::vector<std::pair<std::string, uintmax_t>> out;
    std::error_code ec;
    for (const auto& d : fs::directory_iterator(folder, ec)) {
        std::error_code e2;
        if (!d.is_regular_file(e2)) continue;
        out.emplace_back(d.path().filename().string(), d.file_size(e2));
    }
    std::sort(out.begin(), out.end());
    return out;
}

/// A held shared lease, as a reader process would have it (the same process is enough: flock is per open file).
int take_shared(const std::string& path) {
#ifndef _WIN32
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    if (::flock(fd, LOCK_SH | LOCK_NB) != 0) { ::close(fd); return -1; }
    return fd;
#else
    (void) path;
    return -1;
#endif
}

void release_fd(int fd) {
#ifndef _WIN32
    if (fd >= 0) ::close(fd);
#else
    (void) fd;
#endif
}

#ifndef _WIN32
/// A child process that holds the shared lease on `path` until it is killed; true once it holds it.
bool child_holds(const std::string& path, pid_t& pid) {
    int p[2];
    if (::pipe(p) != 0) return false;
    pid = ::fork();
    if (pid == 0) {
        ::close(p[0]);
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0 || ::flock(fd, LOCK_SH) != 0) ::_exit(3);
        const char c = 'x';
        if (::write(p[1], &c, 1) != 1) ::_exit(4);
        for (;;) ::pause();   // until SIGKILL
    }
    ::close(p[1]);
    char c = 0;
    const ssize_t got = ::read(p[0], &c, 1);
    ::close(p[0]);
    if (got == 1) return true;
    ::kill(pid, SIGKILL);
    (void) ::waitpid(pid, nullptr, 0);
    return false;
}
#endif
} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / ("strata-convdisk-peer-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const SessionFileIdentity id{0xaaaabbbbccccddddull, 0x1111222233334444ull};
    std::string error;

    ConversationDiskOptions writer_opt;
    writer_opt.dir = (root / "r0").string();
    writer_opt.owner = "r0";
    writer_opt.budget_bytes = 64ull << 20;
    writer_opt.refresh_ms = 60000;   // the throttle is tested below: an automatic re-scan is not due for a minute
    writer_opt.durable = false;

    ConversationDiskOptions peer_opt = writer_opt;
    peer_opt.owner = "r1";

    const auto a_ids = tokens(5000, 1000), b_ids = tokens(3000, 5000), c_ids = tokens(4000, 9000);
    const SavedConversation a = split_image(a_ids, 4000, 1), b = split_image(b_ids, 2000, 2), c = split_image(c_ids, 3000, 7);

    {   // a peer of a folder that is not there: refused, and no directory is created
        ConversationDiskOptions missing = peer_opt;
        missing.dir = (root / "nope").string();
        ConversationDiskTier t;
        check(!t.open_peer(missing, id, split_layout(), error), "a peer of a folder that is not there is refused");
        check(!fs::exists(missing.dir), "the peer did not create the folder");
    }

    auto writer = std::make_unique<ConversationDiskTier>();
    check(writer->open(writer_opt, id, split_layout(), error), "the writer opens its folder");
    ConversationDiskTier peer;
    check(peer.open_peer(peer_opt, id, split_layout(), error), "a peer opens the writer's folder read-only");
    check(peer.is_peer() && peer.mode() == ConversationDiskTier::Mode::peer, "the peer knows its mode");
    check(peer.size() == 0, "the peer sees nothing yet");
    uint64_t peer_write = 0;
    check(!peer.spill(a, peer_write, error), "a peer never writes");
    check(!peer.drop(1), "a peer never removes");
    check(peer.clear() == 0, "a peer's clear removes nothing");

    uint64_t ida = 0, idb = 0, idc = 0;
    check(writer->spill(a, ida, error), "the writer spills a conversation");
    ConversationDiskTier::RefreshReport rr;
    check(peer.refresh(error, /*force=*/false, &rr) && peer.size() == 0,
          "an automatic refresh inside refresh_ms is throttled");
    check(peer.refresh(error, /*force=*/true, &rr) && rr.added == 1 && peer.size() == 1,
          "a forced refresh finds the snapshot without a restart");
    auto m = peer.best(prompt_of(a_ids, 3), {}, true);
    check(m.id == ida && m.tokens == 5000 && m.live, "the peer matches the deepest point");
    check(m.generation == peer.generation(ida) && m.generation != 0, "the peer's match carries the generation");

    SavedConversation back;
    SessionStatus st;
    check(peer.load(ida, {}, back, error, &st, m.generation), "a peer reads the snapshot back");
    check(back.disk_id == 0, "a peer restore is not a local file: the next park writes a local copy");
    check(back.live.ids == a.live.ids && back.stage_images.size() == 1, "the peer's image is the writer's");
    check(back.kv[0].k == a.kv[0].k && back.stage_images[0].kv[0].v == a.stage_images[0].kv[0].v,
          "the peer reads every stage's K/V bytes");
    check(peer.reads() == 1, "the peer counted its read");

    {   // a generation that changed between the match and the read: one refresh and a fallback, never a wrong state
        SavedConversation wrong;
        SessionStatus st2;
        check(!peer.load(ida, {}, wrong, error, &st2, m.generation + 100), "a stale generation is refused");
        check(st2.error == SessionError::invalid, "and reported as invalid");
        check(wrong.live.ids.empty(), "the caller's image is untouched on failure");
        check(peer.contains(ida), "the snapshot is kept: only the match was stale");
        check(peer.load(ida, {}, wrong, error, &st2, peer.generation(ida)),
              "the retry with the current generation reads");
    }

    check(writer->spill(b, idb, error), "the writer spills a second conversation");
    check(peer.refresh(error, /*force=*/true, &rr) && rr.added == 1 && peer.size() == 2,
          "the peer sees the second one");

    {   // read-only, verified from the outside: not one file added, removed or rewritten by scans and reads
        const auto before = listing(writer->folder());
        for (int i = 0; i < 3; ++i) {
            check(peer.refresh(error, i == 0, nullptr), "a peer's refresh succeeds");
            SavedConversation image;
            (void) peer.load(ida, {}, image, error, nullptr, peer.generation(ida));
        }
        const auto after = listing(writer->folder());
        check(before == after, "a peer's scans and reads change nothing in the writer's folder");
    }

    {   // a damaged index on the peer's side: ignored, never removed, and the entry is not offered
        const std::string idx = snapshot_path(writer->folder(), idb, ".idx");
        check(fs::exists(idx), "the second conversation's index is there");
        {
            std::fstream f(idx, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(60);
            f.put('\x7f');
        }
        check(peer.refresh(error, /*force=*/true, &rr), "a peer's refresh survives a damaged index");
        check(rr.damaged == 1, "the damaged index is counted, not repaired");
        check(fs::exists(idx), "the peer did not remove the damaged file");
        check(peer.size() == 1 && !peer.contains(idb), "the damaged conversation is not offered");
        // the writer, opening its own folder, drops the damaged entry (its folder, its cleanup)
        writer.reset();
        writer = std::make_unique<ConversationDiskTier>();
        ConversationDiskTier::OpenReport rep;
        check(writer->open(writer_opt, id, split_layout(), error, &rep) && rep.dropped == 1,
              "the writer drops the damaged entry");
    }

    {   // the writer cannot remove a snapshot a reader holds: the eviction is skipped until the lease is let go
        check(writer->spill(c, idc, error), "the writer spills a third conversation");
        const std::string lease = snapshot_path(writer->folder(), idc, ".lock");
        check(fs::exists(lease), "the snapshot has its lease file");
        const int held = take_shared(lease);
        check(held >= 0, "a reader's shared lease is taken");
        check(!writer->drop(idc), "the writer's drop is refused while the lease is held");
        check(writer->contains(idc) && fs::exists(lease), "nothing was removed");
        check(writer->eviction_skipped() >= 1 && writer->lease_conflicts() >= 1, "the conflict is counted");
        release_fd(held);
        check(writer->drop(idc), "after the lease is released the writer removes it");
        check(!writer->contains(idc), "the snapshot is gone");
    }

#ifndef _WIN32
    {   // the lease is the kernel's: a killed reader's lock is released, and its snapshot becomes removable
        uint64_t idd = 0;
        const SavedConversation d = split_image(tokens(2000, 7777), 1000, 11);
        check(writer->spill(d, idd, error), "spill for the kill test");
        const std::string lease = snapshot_path(writer->folder(), idd, ".lock");
        pid_t pid = 0;
        check(child_holds(lease, pid), "a child process holds the shared lease");
        check(!writer->drop(idd), "the writer cannot remove it while the child lives");
        check(::kill(pid, SIGKILL) == 0, "the child is killed");
        int status = 0;
        check(::waitpid(pid, &status, 0) == pid, "the child is reaped");
        check(writer->drop(idd), "SIGKILL released the lease: the writer removes the snapshot");
    }
#endif

    {   // another stage layout is another leaf folder: the peer sees nothing of these files
        ConversationDiskTier other;
        ConversationDiskLayout one;
        one.stages = {{0, 48}};
        std::string e2;
        check(other.open_peer(peer_opt, id, one, e2), "a peer with another layout opens");
        check(other.size() == 0, "and sees none of another carve's files");
    }

    {   // the writer's lifetime lock is released with the tier: another engine may own the folder afterwards
        writer.reset();
        ConversationDiskTier again;
        check(again.open(writer_opt, id, split_layout(), error), "the folder opens once the previous writer is gone");
    }

    std::error_code ec;
    fs::remove_all(root, ec);
    std::printf("conversation_disk_peer_test: %d checks passed\n", checks);
    return 0;
}
