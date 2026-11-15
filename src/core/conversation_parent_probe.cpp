// The C++ side of the parent parking store's integration test (serve/test_parent_parking_integration.py): it connects
// to the descriptor the Python server hands it, publishes a fixed conversation image, or finds and reads one back, and
// prints one line the test parses.  Not a test itself - the real store is the other side - so it stays out of ctest.
#include "strata/core/conversation_parent.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

// The same image the test's reader expects: fixed bytes, so a round trip through the store can be compared exactly.
SavedConversation sample() {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0;
    s.layer_hi = 48;
    s.cvec = false;
    s.live.ids = {1000, 1001, 1002, 1003};
    s.live.imgs = {{2, 0x1234567890abcdefull}};
    s.live.gdn = std::vector<uint8_t>(4099, 7);
    s.live.ple = std::vector<uint8_t>(77, 8);
    s.live.tails = std::vector<uint8_t>(301, 9);
    s.live.dead = std::vector<uint8_t>(64, 10);
    s.live.block_pos = std::vector<uint8_t>(8, 11);
    s.live.used = 42;
    ConversationCheckpoint c;
    c.ids = {1000, 1001};
    c.imgs = {{2, 0x1234567890abcdefull}};
    c.gdn = std::vector<uint8_t>(4099, 12);
    c.ple = std::vector<uint8_t>(77, 13);
    c.tails = std::vector<uint8_t>(301, 14);
    c.dead = std::vector<uint8_t>(64, 15);
    c.block_pos = std::vector<uint8_t>(8, 16);
    c.used = 41;
    s.checkpoints.push_back(c);
    ConversationKv kv;
    kv.format = 2;
    kv.cells = 1024;
    kv.heads = 2;
    kv.head_dim = 256;
    kv.page_size = 64;
    kv.pooled_rows = 9;
    kv.idx_dim = 128;
    kv.k = pattern(4096, 1);
    kv.v = pattern(1024, 2);
    kv.k_scale = pattern(512, 3);
    kv.v_scale = pattern(512, 4);
    kv.pooled = pattern(9 * 128 * 4, 5);
    s.kv.push_back(std::move(kv));
    return s;
}

std::vector<SessionKvSource> sources_of(const SavedConversation& image) {
    std::vector<SessionKvSource> out;
    for (const auto& kv : image.kv) {
        SessionKvSource s;
        s.format = kv.format;
        s.cells = kv.cells;
        s.heads = kv.heads;
        s.head_dim = kv.head_dim;
        s.page_size = kv.page_size;
        s.pooled_rows = kv.pooled_rows;
        s.idx_dim = kv.idx_dim;
        const std::array<const ConversationBuffer*, 5> parts = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
        for (size_t i = 0; i < 5; ++i) s.sizes[i] = parts[i]->size();
        s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
            return part < 5 && parts[part]->read(dst, offset, n);
        };
        s.error = [] { return std::string("probe source"); };
        out.push_back(std::move(s));
    }
    return out;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i];
        const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim || !(x.k == y.k) ||
            !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) || !(x.pooled == y.pooled))
            return false;
    }
    return true;
}

int fail(const std::string& why) {
    std::printf("FAIL %s\n", why.c_str());
    return 1;
}
} // namespace

int main(int argc, char** argv) {
    int fd = -1, timeout_ms = 2000;
    std::string owner, mode;
    ParentIdentity identity;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--fd") fd = std::atoi(next().c_str());
        else if (a == "--owner") owner = next();
        else if (a == "--mode") mode = next();
        else if (a == "--model") identity.model = std::strtoull(next().c_str(), nullptr, 0);
        else if (a == "--config") identity.config = std::strtoull(next().c_str(), nullptr, 0);
        else if (a == "--layout") identity.layout_key = std::strtoull(next().c_str(), nullptr, 0);
        else if (a == "--timeout-ms") timeout_ms = std::atoi(next().c_str());
        else if (a == "--stage") {
            const std::string v = next();
            const size_t colon = v.find(':');
            if (colon == std::string::npos) return fail("--stage needs lo:hi");
            identity.stages.emplace_back(std::strtoll(v.substr(0, colon).c_str(), nullptr, 10),
                                         std::strtoll(v.substr(colon + 1).c_str(), nullptr, 10));
        } else return fail("unknown argument " + a);
    }
    if (fd < 0) return fail("--fd is required");
    if (mode != "publish" && mode != "read") return fail("--mode must be publish or read");
    identity.owner = owner;

    ConversationParentClient client;
    std::string error;
    if (!client.connect(fd, identity, timeout_ms, error)) return fail("connect: " + error);

    // the engine asks for the store's counters right after HELLO (its start line): the real store's answer must parse
    {
        ParentStats stats;
        if (!client.stats(stats, error)) return fail("stats: " + error);
        std::printf("STATS entries=%zu entry_bytes=%llu epoch=%llu longest=%lld\n", stats.entries,
                    (unsigned long long) stats.entry_bytes, (unsigned long long) stats.store_epoch,
                    (long long) stats.longest_tokens);
        if (!client.ping(error)) return fail("ping: " + error);
    }
    const SavedConversation original = sample();
    const auto sources = sources_of(original);
    auto meta = original;
    meta.kv.clear();
    const SessionFileIdentity sid {identity.model, identity.config};

    if (mode == "publish") {
        uint64_t measured = 0;
        if (!session_file_measure(meta, sources, measured, error)) return fail("measure: " + error);
        ParentPending pending;
        bool dedup = false;
        const std::string publish_id = parent_publish_id();
        if (!client.begin(publish_id, parent_points(original), false, {measured}, pending, dedup, error))
            return fail("begin: " + error);
        if (dedup) {
            std::printf("DEDUP\n");
            return 0;
        }
        size_t written = 0;
        SessionWriteOptions wo;
        wo.durable = false;
        SessionStatus st;
        if (!session_file_write_fd(pending.fd(0), meta, sources, sid, written, error, wo, &st))
            return fail("write: " + error);
        if (written != measured) return fail("the written size differs from the measured one");
        pending.close_fds();
        uint64_t id = 0, generation = 0;
        if (!client.commit(publish_id, pending, id, generation, error)) return fail("commit: " + error);
        std::printf("PUBLISHED %llu %llu %llu\n", (unsigned long long) id, (unsigned long long) generation,
                    (unsigned long long) measured);
        return 0;
    }

    // read: find the deepest resume point the store holds for this prompt, acquire it and read it back
    uint64_t cookie = 0;
    std::vector<int64_t> lengths;
    if (!client.match_begin(8, false, original.live.imgs, {}, cookie, lengths, error))
        return fail("match_begin: " + error);
    if (lengths.empty()) return fail("the store offers no resume point");
    std::optional<ParentCandidate> candidate;
    if (!client.match_end(cookie, parent_hashes(original.live.ids, lengths), candidate, error))
        return fail("match_end: " + error);
    if (!candidate) return fail("the store found no candidate");
    ParentLease lease;
    if (!client.acquire(*candidate, lease, error)) return fail("acquire: " + error);
    SavedConversation back;
    std::vector<SessionReadLimits> limits(1);
    SessionStatus st;
    if (!parent_load(lease, limits, back, error, &st)) return fail("load: " + error);
    if (!same(original, back)) return fail("the image read back differs from the one published");
    lease.close_fds();
    if (!lease.release("ok", error)) return fail("release: " + error);
    std::printf("READ %lld %s %zu\n", (long long) candidate->tokens, candidate->owner.c_str(), back.bytes());
    return 0;
}
