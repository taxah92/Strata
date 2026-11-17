// CPU-only tests for the parent parking store's client (include/strata/core/conversation_parent.hpp).
// A fake broker in this process speaks the frozen protocol v1 over real SOCK_SEQPACKET socketpairs and hands out real
// memfds, so the framing, the SCM_RIGHTS passing and the session-file round trip are exercised without the Python
// server, a model or a card.  The wire format is .scratch/handoff-notes/parent-parking/contract.md.
#include "strata/core/conversation_parent.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

#ifndef _WIN32
constexpr size_t kFrame = 36;
constexpr char kMagic[8] = {'S', 'T', 'R', 'P', 'A', 'R', 'K', '1'};
constexpr uint64_t kPayloadSeed = 0x5354525041524B31ull;   // "STRPARK1", the project's big-endian reading
enum : uint16_t { kHello = 1, kAdvice = 2, kBegin = 3, kCommit = 4, kAbort = 5, kMatchBegin = 6, kMatchEnd = 7,
                  kAcquire = 8, kRelease = 9, kClear = 10, kStats = 11, kPing = 12, kError = 255 };

// ---- frame I/O, the broker's side of the frozen format
bool send_frame(int fd, uint16_t op, uint64_t request_id, const std::string& payload, const std::vector<int>& fds) {
    uint8_t h[kFrame];
    std::memcpy(h, kMagic, 8);
    const uint16_t version = 1;
    const uint32_t flags = 0, n = (uint32_t) payload.size();
    std::memcpy(h + 8, &version, 2);
    std::memcpy(h + 10, &op, 2);
    std::memcpy(h + 12, &flags, 4);
    std::memcpy(h + 16, &n, 4);
    std::memcpy(h + 20, &request_id, 8);
    const uint64_t hash = session_hash64(payload.data(), payload.size(), kPayloadSeed);
    std::memcpy(h + 28, &hash, 8);
    std::vector<uint8_t> frame(h, h + kFrame);
    frame.insert(frame.end(), payload.begin(), payload.end());
    struct iovec iov {frame.data(), frame.size()};
    struct msghdr msg {};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    std::vector<char> control(CMSG_SPACE(sizeof(int) * (fds.empty() ? 1 : fds.size())));
    if (!fds.empty()) {
        msg.msg_control = control.data();
        msg.msg_controllen = CMSG_SPACE(sizeof(int) * fds.size());
        struct cmsghdr* c = CMSG_FIRSTHDR(&msg);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(sizeof(int) * fds.size());
        std::memcpy(CMSG_DATA(c), fds.data(), sizeof(int) * fds.size());
    }
    return ::sendmsg(fd, &msg, 0) == (ssize_t) frame.size();
}

bool recv_frame(int fd, uint16_t& op, uint64_t& request_id, std::string& payload, std::vector<int>& fds) {
    std::vector<uint8_t> buf(1u << 20);
    std::vector<char> control(CMSG_SPACE(sizeof(int) * 64));
    struct iovec iov {buf.data(), buf.size()};
    struct msghdr msg {};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.data();
    msg.msg_controllen = control.size();
    const ssize_t n = ::recvmsg(fd, &msg, 0);
    if (n < (ssize_t) kFrame) return false;
    if (std::memcmp(buf.data(), kMagic, 8) != 0) return false;
    uint16_t version = 0;
    uint32_t flags = 0, bytes = 0;
    uint64_t hash = 0;
    std::memcpy(&version, buf.data() + 8, 2);
    std::memcpy(&op, buf.data() + 10, 2);
    std::memcpy(&flags, buf.data() + 12, 4);
    std::memcpy(&bytes, buf.data() + 16, 4);
    std::memcpy(&request_id, buf.data() + 20, 8);
    std::memcpy(&hash, buf.data() + 28, 8);
    if (version != 1 || flags != 0 || (size_t) n != kFrame + bytes) return false;
    payload.assign((const char*) buf.data() + kFrame, bytes);
    if (hash != session_hash64(payload.data(), payload.size(), kPayloadSeed)) return false;
    fds.clear();
    for (struct cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c))
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
            const size_t count = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
            const int* got = (const int*) CMSG_DATA(c);
            for (size_t i = 0; i < count; ++i) fds.push_back(got[i]);
        }
    return true;
}

// ---- a tiny JSON reader for the broker's side: the fields the client sends, nothing else
std::string field_string(const std::string& json, const char* key) {
    const std::string needle = std::string("\"") + key + "\":\"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) return {};
    const size_t from = at + needle.size();
    const size_t to = json.find('"', from);
    return to == std::string::npos ? std::string() : json.substr(from, to - from);
}
bool field_u64(const std::string& json, const char* key, uint64_t& out) {
    const std::string needle = std::string("\"") + key + "\":";
    const size_t at = json.find(needle);
    if (at == std::string::npos) return false;
    const char* p = json.c_str() + at + needle.size();
    char* end = nullptr;
    const unsigned long long v = std::strtoull(p, &end, 10);
    if (end == p) return false;
    out = (uint64_t) v;
    return true;
}
std::vector<uint64_t> field_u64_array(const std::string& json, const char* key) {
    std::vector<uint64_t> out;
    const std::string needle = std::string("\"") + key + "\":[";
    const size_t at = json.find(needle);
    if (at == std::string::npos) return out;
    const char* p = json.c_str() + at + needle.size();
    while (*p != ']' && *p != '\0') {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(p, &end, 10);
        if (end == p) break;
        out.push_back((uint64_t) v);
        p = end;
        if (*p == ',') ++p;
    }
    return out;
}
// The client's {"tokens":N,"hash":N} pairs (prefix_hashes in PARK_ADVICE and PARK_BEGIN); empty when the key is absent.
std::vector<std::pair<int64_t, uint64_t>> field_hash_array(const std::string& json, const char* key) {
    std::vector<std::pair<int64_t, uint64_t>> out;
    const std::string needle = std::string("\"") + key + "\":[";
    const size_t at = json.find(needle);
    if (at == std::string::npos) return out;
    const char* p = json.c_str() + at + needle.size();
    while (*p != ']' && *p != '\0') {
        if (*p == '{') {
            long long tokens = 0;
            unsigned long long hash = 0;
            if (std::sscanf(p, "{\"tokens\":%lld,\"hash\":%llu}", &tokens, &hash) == 2)
                out.emplace_back((int64_t) tokens, (uint64_t) hash);
        }
        while (*p != '\0' && *p != '}' && *p != ']') ++p;
        if (*p == '}') ++p;
        if (*p == ',') ++p;
    }
    return out;
}

int memfd_of(size_t size) {
    const int fd = ::memfd_create("strata-parent-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) return -1;
    if (size && ::ftruncate(fd, (off_t) size) != 0) { ::close(fd); return -1; }
    return fd;
}

// ---- the fake broker: one thread, every channel, the protocol's state in plain members
struct Broker {
    std::vector<int> channels;
    std::thread thread;
    std::atomic<bool> stop {false};
    std::atomic<int> served {0};
    std::string failure;   // the first protocol violation the broker saw (a test fails on it)

    // behaviour knobs, set before the thread starts
    bool hello_refuses = false;        // answer HELLO with ERROR identity
    bool close_after_hello = false;    // close the channel right after the HELLO answer
    bool bad_answer_hash = false;      // corrupt the payload hash of the next answer
    bool begin_without_fd = false;     // answer PARK_BEGIN with no descriptor
    bool commit_without_answer = false;// drop the PARK_COMMIT answer (the client times out)
    bool acquire_stale = false;        // answer ACQUIRE with ERROR stale_generation
    bool match_empty = false;          // answer MATCH_BEGIN with no lengths
    bool advice_prefixes = false;      // answer PARK_ADVICE with prefix_lengths [2]
    bool advice_bad_prefixes = false;  // answer PARK_ADVICE with unsorted prefix_lengths [3,2]

    // state
    uint64_t epoch = 1;
    uint64_t next_id = 100, next_generation = 7, next_pending = 50, next_lease = 900, next_cookie = 5;
    uint64_t pending_id = 0;
    std::vector<int> pending_fds;
    std::vector<uint64_t> pending_bytes;
    std::vector<std::pair<int64_t, uint64_t>> last_advice_hashes;
    std::vector<std::pair<int64_t, uint64_t>> last_begin_hashes;
    uint64_t committed_id = 0, committed_generation = 0;
    std::vector<int> committed_fds;
    std::vector<uint64_t> committed_bytes;
    int64_t committed_tokens = 0;
    bool committed_live = false;
    std::string committed_owner = "r0";
    int releases = 0;
    std::string last_release_result;
    int clears = 0;
    int aborts = 0;
    int eofs = 0;

    ~Broker() {
        stop = true;
        if (thread.joinable()) thread.join();
        for (int fd : channels) ::close(fd);
        for (int fd : pending_fds) ::close(fd);
        for (int fd : committed_fds) ::close(fd);
    }

    void start() { thread = std::thread([this] { run(); }); }

    void run() {
        while (!stop) {
            std::vector<struct pollfd> pfds;
            for (int fd : channels) pfds.push_back({fd, POLLIN, 0});
            if (pfds.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
            const int n = ::poll(pfds.data(), pfds.size(), 20);
            if (n <= 0) continue;
            for (size_t i = 0; i < pfds.size(); ++i) {
                if (!(pfds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
                uint16_t op = 0;
                uint64_t request_id = 0;
                std::string payload;
                std::vector<int> fds;
                if (!recv_frame(channels[i], op, request_id, payload, fds)) {
                    ++eofs;
                    ::close(channels[i]);
                    channels.erase(channels.begin() + (std::ptrdiff_t) i);
                    break;
                }
                for (int fd : fds) ::close(fd);   // the client sends none
                ++served;
                handle(channels[i], op, request_id, payload);
            }
        }
    }

    void answer(int fd, uint16_t op, uint64_t request_id, const std::string& payload,
                const std::vector<int>& fds = {}) {
        if (bad_answer_hash) {
            bad_answer_hash = false;
            // a frame whose payload hash does not match: the client must refuse it and degrade
            uint8_t h[kFrame];
            std::memcpy(h, kMagic, 8);
            const uint16_t version = 1;
            const uint32_t flags = 0, n = (uint32_t) payload.size();
            std::memcpy(h + 8, &version, 2);
            std::memcpy(h + 10, &op, 2);
            std::memcpy(h + 12, &flags, 4);
            std::memcpy(h + 16, &n, 4);
            std::memcpy(h + 20, &request_id, 8);
            const uint64_t wrong = 0x1234567890abcdefull;
            std::memcpy(h + 28, &wrong, 8);
            std::vector<uint8_t> frame(h, h + kFrame);
            frame.insert(frame.end(), payload.begin(), payload.end());
            (void) ::send(fd, frame.data(), frame.size(), 0);
            return;
        }
        (void) send_frame(fd, op, request_id, payload, fds);
    }

    void error(int fd, uint64_t request_id, const char* code, const char* message) {
        answer(fd, kError, request_id,
               std::string("{\"code\":\"") + code + "\",\"message\":\"" + message + "\",\"retryable\":false}");
    }

    void handle(int fd, uint16_t op, uint64_t request_id, const std::string& payload) {
        switch (op) {
            case kHello:
                if (hello_refuses) { error(fd, request_id, "identity", "another identity"); return; }
                answer(fd, kHello, request_id,
                       "{\"store_epoch\":" + std::to_string(epoch) + ",\"entries\":0,\"bytes\":0,"
                       "\"limits\":{\"max_payload\":1048576,\"max_points\":4096,\"max_stages\":64,\"max_fds\":64,"
                       "\"max_exclude_ids\":1}}");
                if (close_after_hello) {
                    ::close(fd);
                    for (size_t i = 0; i < channels.size(); ++i)
                        if (channels[i] == fd) { channels.erase(channels.begin() + (std::ptrdiff_t) i); break; }
                }
                return;
            case kAdvice: {
                last_advice_hashes = field_hash_array(payload, "prefix_hashes");
                std::string body = "{\"holding_id\":0,\"parked_prefix_tokens\":0,\"entries\":0,\"bytes\":0,\"epoch\":" +
                                   std::to_string(epoch);
                if (advice_prefixes) body += ",\"prefix_lengths\":[2]";
                if (advice_bad_prefixes) body += ",\"prefix_lengths\":[3,2]";
                body += "}";
                answer(fd, kAdvice, request_id, body);
                return;
            }
            case kBegin: {
                last_begin_hashes = field_hash_array(payload, "prefix_hashes");
                pending_bytes = field_u64_array(payload, "stage_bytes");
                pending_id = next_pending++;
                for (int old : pending_fds) ::close(old);
                pending_fds.clear();
                std::vector<int> send;
                for (uint64_t n : pending_bytes) {
                    const int m = memfd_of((size_t) n);
                    if (m < 0) { error(fd, request_id, "internal", "no memfd"); return; }
                    pending_fds.push_back(m);
                    send.push_back(m);
                }
                std::string body = "{\"pending_id\":" + std::to_string(pending_id) + ",\"stage_bytes\":[";
                for (size_t i = 0; i < pending_bytes.size(); ++i)
                    body += (i ? "," : "") + std::to_string(pending_bytes[i]);
                body += "],\"dedup\":false,\"holding_id\":0}";
                answer(fd, kBegin, request_id, body, begin_without_fd ? std::vector<int>{} : send);
                return;
            }
            case kCommit: {
                if (commit_without_answer) { commit_without_answer = false; return; }
                uint64_t written = 0;
                (void) field_u64(payload, "pending_id", written);
                if (written != pending_id) { error(fd, request_id, "bad_request", "unknown pending"); return; }
                // the store's own check: a regular file of the exact size, then the four seals
                for (size_t i = 0; i < pending_fds.size(); ++i) {
                    struct stat st {};
                    if (::fstat(pending_fds[i], &st) != 0 || !S_ISREG(st.st_mode) ||
                        (uint64_t) st.st_size != pending_bytes[i]) {
                        error(fd, request_id, "bad_request", "bad stage size");
                        return;
                    }
                    if (::fcntl(pending_fds[i], F_ADD_SEALS,
                                F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0) {
                        error(fd, request_id, "internal", "seal failed");
                        return;
                    }
                }
                for (int old : committed_fds) ::close(old);
                committed_fds = pending_fds;
                committed_bytes = pending_bytes;
                pending_fds.clear();
                pending_bytes.clear();
                committed_id = next_id++;
                committed_generation = next_generation++;
                committed_tokens = 4;   // the sample image's live length
                committed_live = true;
                answer(fd, kCommit, request_id,
                       "{\"id\":" + std::to_string(committed_id) + ",\"generation\":" +
                           std::to_string(committed_generation) + ",\"entry_bytes\":1,\"dedup\":false,"
                           "\"entries\":1,\"bytes\":1,\"evictions\":0,\"superseded\":0}");
                return;
            }
            case kAbort:
                ++aborts;
                for (int old : pending_fds) ::close(old);
                pending_fds.clear();
                pending_bytes.clear();
                pending_id = 0;
                answer(fd, kAbort, request_id, "{\"gone\":true}");
                return;
            case kMatchBegin: {
                std::string body = "{\"cookie\":" + std::to_string(next_cookie++) + ",\"lengths\":[";
                if (!match_empty && committed_id != 0) body += std::to_string(committed_tokens);
                body += "]}";
                answer(fd, kMatchBegin, request_id, body);
                return;
            }
            case kMatchEnd: {
                if (committed_id == 0) { answer(fd, kMatchEnd, request_id, "{\"candidate\":null}"); return; }
                std::string body = "{\"candidate\":{\"id\":" + std::to_string(committed_id) + ",\"generation\":" +
                                   std::to_string(committed_generation) + ",\"tokens\":" +
                                   std::to_string(committed_tokens) + ",\"live\":" +
                                   (committed_live ? "true" : "false") + ",\"borrow\":false,\"stage_bytes\":[";
                for (size_t i = 0; i < committed_bytes.size(); ++i)
                    body += (i ? "," : "") + std::to_string(committed_bytes[i]);
                body += "],\"read_charge_bytes\":1,\"owner\":\"" + committed_owner + "\"}}";
                answer(fd, kMatchEnd, request_id, body);
                return;
            }
            case kAcquire: {
                if (acquire_stale) { error(fd, request_id, "stale_generation", "the entry changed"); return; }
                if (committed_id == 0) { error(fd, request_id, "gone", "no such entry"); return; }
                std::string body = "{\"lease_id\":" + std::to_string(next_lease++) + ",\"id\":" +
                                   std::to_string(committed_id) + ",\"generation\":" +
                                   std::to_string(committed_generation) + ",\"tokens\":" +
                                   std::to_string(committed_tokens) + ",\"live\":" +
                                   (committed_live ? "true" : "false") + ",\"borrow\":false,\"stage_bytes\":[";
                for (size_t i = 0; i < committed_bytes.size(); ++i)
                    body += (i ? "," : "") + std::to_string(committed_bytes[i]);
                body += "],\"read_charge_bytes\":1,\"owner\":\"" + committed_owner + "\"}";
                answer(fd, kAcquire, request_id, body, committed_fds);
                return;
            }
            case kRelease:
                ++releases;
                last_release_result = field_string(payload, "result");
                answer(fd, kRelease, request_id, "{\"gone\":false}");
                return;
            case kClear:
                ++clears;
                for (int old : committed_fds) ::close(old);
                committed_fds.clear();
                committed_bytes.clear();
                committed_id = 0;
                ++epoch;
                answer(fd, kClear, request_id,
                       "{\"cleared\":1,\"epoch\":" + std::to_string(epoch) + "}");
                return;
            case kStats:
                answer(fd, kStats, request_id,
                       "{\"enabled\":true,\"capacity_bytes\":1024,\"charged_bytes\":0,\"entries\":0,\"entry_bytes\":0,"
                       "\"pending\":0,\"pending_bytes\":0,\"leases\":0,\"restore_reserved_bytes\":0,"
                       "\"retired_lease_bytes\":0,\"published\":0,\"published_bytes\":0,\"deduplicated\":0,"
                       "\"reads\":0,\"peer_reads\":0,\"evictions\":0,\"eviction_skipped_leased\":0,\"superseded\":0,"
                       "\"clears\":0,\"damaged\":0,\"capacity_refusals\":0,\"budget_violations\":0,\"store_epoch\":" +
                           std::to_string(epoch) + ",\"longest_tokens\":0,\"replicas\":[]}");
                return;
            case kPing:
                answer(fd, kPing, request_id, "{\"epoch\":" + std::to_string(epoch) + "}");
                return;
            default:
                failure = "unknown op " + std::to_string(op);
                error(fd, request_id, "protocol", "unknown op");
                return;
        }
    }
};

// A socketpair with the broker on one end and the client on the other.
int connect_pair(Broker& broker) {
    int sv[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv) != 0) return -1;
    broker.channels.push_back(sv[0]);
    return sv[1];
}

ParentIdentity identity(const char* owner = "r0") {
    ParentIdentity id;
    id.model = 0x1111222233334444ull;
    id.config = 0x5555666677778888ull;
    id.layout_key = 0x9999aaaabbbbccccull;
    id.stages = {{0, 48}};
    id.owner = owner;
    return id;
}

// A small conversation image and its streamed K/V sources, the shape the engine writes.
ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

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
        s.error = [] { return std::string("test source"); };
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

// Waits until the broker has served `n` requests (the client is synchronous, so this is only a safety net).
void settle(Broker& broker, int n) {
    for (int i = 0; i < 2000 && broker.served.load() < n; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
#endif
} // namespace

int main() {
#ifndef _WIN32
    const SavedConversation original = sample();
    const auto sources = sources_of(original);
    auto meta = original;
    meta.kv.clear();
    const SessionFileIdentity sid {identity().model, identity().config};

    // ---- HELLO, and an identity the store refuses (a start error, never a silent loss of the feature)
    {
        Broker broker;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(fd >= 0, "hello: socketpair");
        check(client.connect(fd, identity(), 2000, error), "hello: connect");
        check(client.state() == ConversationParentClient::State::connected, "hello: connected");
        check(client.epoch() == 1, "hello: the store's epoch");
        check(client.owner() == "r0", "hello: the owner");
        client.close();
        check(client.state() == ConversationParentClient::State::off, "hello: closed");
    }
    {
        Broker broker;
        broker.hello_refuses = true;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(!client.connect(fd, identity(), 2000, error), "hello: a refused identity fails the start");
        check(error.find("identity") != std::string::npos, "hello: the refusal names the identity");
        check(client.state() == ConversationParentClient::State::off, "hello: a refused start leaves the client off");
    }

    // ---- publish: measure, PARK_BEGIN hands out memfds, the session file is written into them, COMMIT seals them,
    // ACQUIRE hands a sealed one back and the image round-trips
    {
        Broker broker;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "publish: connect");
        uint64_t measured = 0;
        check(session_file_measure(meta, sources, measured, error), "publish: measure");
        const std::vector<ParentPoint> points = parent_points(original);
        check(points.size() == 2 && points[0].live && !points[1].live, "publish: the live point comes first");
        check(points[0].tokens == 4 && points[1].tokens == 2, "publish: the points' lengths");
        check(points[0].hash == parent_ids_hash(original.live.ids.data(), original.live.ids.size()),
              "publish: the point's hash is the ids hash");
        ParentAdvice advice;
        check(client.advice(points, false, advice, error) && advice.holding_id == 0, "publish: advice");
        ParentPending pending;
        bool dedup = false;
        check(client.begin(parent_publish_id(), points, false, {measured}, pending, dedup, error), "publish: begin");
        check(!dedup && pending.valid() && pending.stages() == 1, "publish: a pending with one descriptor");
        check(pending.stage_bytes().size() == 1 && pending.stage_bytes()[0] == measured,
              "publish: the store sized the descriptor to the measured bytes");
        size_t written = 0;
        SessionWriteOptions wo;
        wo.durable = false;
        SessionStatus st;
        check(session_file_write_fd(pending.fd(0), meta, sources, sid, written, error, wo, &st),
              "publish: the session file written into the store's memfd");
        check(written == measured, "publish: the written size is the measured one");
        pending.close_fds();
        uint64_t id = 0, generation = 0;
        check(client.commit("0123456789abcdef0123456789abcdef", pending, id, generation, error), "publish: commit");
        check(id != 0 && generation != 0, "publish: the store's id and generation");
        // find it again and read it back
        uint64_t cookie = 0;
        std::vector<int64_t> lengths;
        check(client.match_begin(8, false, {}, {}, cookie, lengths, error), "publish: match_begin");
        check(lengths.size() == 1 && lengths[0] == 4, "publish: the store offers the committed point");
        std::optional<ParentCandidate> candidate;
        check(client.match_end(cookie, parent_hashes(original.live.ids, lengths), candidate, error),
              "publish: match_end");
        check(candidate.has_value() && candidate->id == id && candidate->tokens == 4 && candidate->live,
              "publish: the candidate is the committed entry");
        check(candidate->owner == "r0", "publish: the candidate names its owner");
        ParentLease lease;
        check(client.acquire(*candidate, lease, error), "publish: acquire");
        check(lease.valid() && lease.stages() == 1, "publish: a lease with one sealed descriptor");
        SavedConversation back;
        std::vector<SessionReadLimits> limits(1);
        check(parent_load(lease, limits, back, error, &st), "publish: load");
        check(same(original, back), "publish: the image round-trips through the store's memfd");
        lease.close_fds();
        check(lease.release("ok", error), "publish: release");
        check(broker.releases == 1 && broker.last_release_result == "ok", "publish: the store saw the release");
        check(broker.failure.empty(), "publish: no protocol violation");
    }

    // ---- a lease that is dropped without a release releases with "io" (the entry stays for another try)
    {
        Broker broker;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "lease: connect");
        const std::vector<ParentPoint> points = parent_points(original);
        uint64_t measured = 0;
        check(session_file_measure(meta, sources, measured, error), "lease: measure");
        ParentPending pending;
        bool dedup = false;
        check(client.begin(parent_publish_id(), points, false, {measured}, pending, dedup, error), "lease: begin");
        size_t written = 0;
        SessionWriteOptions wo;
        wo.durable = false;
        SessionStatus st;
        check(session_file_write_fd(pending.fd(0), meta, sources, sid, written, error, wo, &st), "lease: write");
        pending.close_fds();
        uint64_t id = 0, generation = 0;
        check(client.commit("fedcba9876543210fedcba9876543210", pending, id, generation, error), "lease: commit");
        ParentCandidate candidate;
        candidate.id = id;
        candidate.generation = generation;
        candidate.tokens = 4;
        candidate.live = true;
        candidate.stage_bytes = {measured};
        {
            ParentLease lease;
            check(client.acquire(candidate, lease, error), "lease: acquire");
        }   // destroyed without a release
        settle(broker, 4);
        check(broker.releases == 1 && broker.last_release_result == "io",
              "lease: a dropped lease releases with io, so the entry is kept");
    }

    // ---- a stale ACQUIRE is reported with its code, so the caller can make one fresh attempt
    {
        Broker broker;
        broker.acquire_stale = true;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "stale: connect");
        ParentCandidate candidate;
        candidate.id = 1;
        candidate.generation = 1;
        candidate.tokens = 4;
        candidate.stage_bytes = {16};
        ParentLease lease;
        check(!client.acquire(candidate, lease, error), "stale: acquire fails");
        check(client.last_error_code() == "stale_generation", "stale: the code says the generation moved");
        check(client.state() == ConversationParentClient::State::connected, "stale: a refusal is not a transport error");
    }

    // ---- a transport error degrades the client, and every later call fails cleanly (the request falls back to prefill)
    {
        Broker broker;
        broker.close_after_hello = true;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "degraded: connect");
        settle(broker, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ParentAdvice advice;
        const bool ok = client.advice(parent_points(original), false, advice, error);
        check(!ok, "degraded: the call after the channel closed fails");
        check(client.degraded(), "degraded: the client says so");
        check(!client.advice(parent_points(original), false, advice, error), "degraded: and every later call too");
        check(!error.empty(), "degraded: with a reason");
    }

    // ---- a corrupted answer is refused, not parsed
    {
        Broker broker;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "hash: connect");
        broker.bad_answer_hash = true;   // the next answer (the advice) is corrupted
        ParentAdvice advice;
        check(!client.advice(parent_points(original), false, advice, error), "hash: a bad payload hash is refused");
        check(client.degraded(), "hash: and degrades the channel");
    }

    // ---- PARK_ADVICE's prefix_lengths: the client parses them, sends the hashes back in the second advice and in begin
    {
        Broker broker;
        broker.advice_prefixes = true;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "prefix: connect");
        const std::vector<ParentPoint> points = parent_points(original);
        const std::vector<std::pair<int64_t, uint64_t>> hashes = {{2, 0x1234ull}};
        ParentAdvice advice;
        check(client.advice(points, false, advice, error, hashes), "prefix: advice with the hashes");
        check(advice.prefix_lengths.size() == 1 && advice.prefix_lengths[0] == 2,
              "prefix: the store's prefix lengths parse");
        check(broker.last_advice_hashes == hashes, "prefix: the advice carried the hashes");
        uint64_t measured = 0;
        check(session_file_measure(meta, sources, measured, error), "prefix: measure");
        const std::string pid = parent_publish_id();
        ParentPending pending;
        bool dedup = false;
        check(client.begin(pid, points, false, {measured}, pending, dedup, error, hashes),
              "prefix: begin with the hashes");
        check(broker.last_begin_hashes == hashes, "prefix: the begin carried the hashes");
        check(client.abort(pid, pending.id(), error), "prefix: abort");
    }

    // ---- prefix_lengths an old store does not send read as none; unsorted ones are refused
    {
        Broker broker;   // the answer carries no prefix_lengths
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "noprefix: connect");
        ParentAdvice advice;
        check(client.advice(parent_points(original), false, advice, error), "noprefix: advice");
        check(advice.prefix_lengths.empty(), "noprefix: an absent key is an empty list");
    }
    {
        Broker broker;
        broker.advice_bad_prefixes = true;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "badprefix: connect");
        ParentAdvice advice;
        check(!client.advice(parent_points(original), false, advice, error),
              "badprefix: unsorted prefix lengths are refused");
        check(client.degraded(), "badprefix: and degrade the channel");
    }

    // ---- a PARK_BEGIN answer without its descriptor is a protocol violation
    {
        Broker broker;
        broker.begin_without_fd = true;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "fd: connect");
        ParentPending pending;
        bool dedup = false;
        check(!client.begin(parent_publish_id(), parent_points(original), false, {1024}, pending, dedup, error),
              "fd: a begin answer without its descriptor is refused");
        check(client.degraded(), "fd: and degrades the channel");
    }

    // ---- a lost COMMIT answer is retried once (the store is idempotent by publish_id), and the retry succeeds
    {
        Broker broker;
        broker.commit_without_answer = true;   // the first answer is dropped
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "lost: connect");
        ParentPending pending;
        bool dedup = false;
        check(client.begin(parent_publish_id(), parent_points(original), false, {1024}, pending, dedup, error),
              "lost: begin");
        uint64_t id = 0, generation = 0;
        check(client.commit("00112233445566778899aabbccddeeff", pending, id, generation, error),
              "lost: a lost commit answer is retried and succeeds");
        check(!client.degraded(), "lost: the retry leaves the channel connected");
        check(id != 0 && generation != 0, "lost: the retry's answer is the store's");
    }

    // ---- CLEAR and STATS
    {
        Broker broker;
        broker.start();
        ConversationParentClient client;
        std::string error;
        const int fd = connect_pair(broker);
        check(client.connect(fd, identity(), 2000, error), "clear: connect");
        size_t cleared = 0;
        check(client.clear("nonfinite", cleared, error) && cleared == 1, "clear: the store cleared its entries");
        check(client.epoch() == 2, "clear: the epoch moved");
        ParentStats stats;
        check(client.stats(stats, error), "stats: read");
        check(stats.enabled && stats.store_epoch == 2, "stats: the snapshot");
        check(client.ping(error), "ping");
    }

    // ---- two clients share one store: one publishes, the other reads the same sealed snapshot
    {
        Broker broker;
        broker.start();
        ConversationParentClient a, b;
        std::string error;
        const int fa = connect_pair(broker);
        const int fb = connect_pair(broker);
        check(a.connect(fa, identity("r0"), 2000, error), "share: r0 connects");
        check(b.connect(fb, identity("r1"), 2000, error), "share: r1 connects");
        uint64_t measured = 0;
        check(session_file_measure(meta, sources, measured, error), "share: measure");
        ParentPending pending;
        bool dedup = false;
        check(a.begin(parent_publish_id(), parent_points(original), false, {measured}, pending, dedup, error),
              "share: r0 begins");
        size_t written = 0;
        SessionWriteOptions wo;
        wo.durable = false;
        SessionStatus st;
        check(session_file_write_fd(pending.fd(0), meta, sources, sid, written, error, wo, &st), "share: r0 writes");
        pending.close_fds();
        uint64_t id = 0, generation = 0;
        check(a.commit("aaaabbbbccccddddeeeeffff00001111", pending, id, generation, error), "share: r0 commits");
        uint64_t cookie = 0;
        std::vector<int64_t> lengths;
        check(b.match_begin(8, false, {}, {}, cookie, lengths, error), "share: r1 matches");
        std::optional<ParentCandidate> candidate;
        check(b.match_end(cookie, parent_hashes(original.live.ids, lengths), candidate, error), "share: r1 finds it");
        check(candidate.has_value() && candidate->id == id, "share: the same entry");
        ParentLease lease;
        check(b.acquire(*candidate, lease, error), "share: r1 acquires");
        SavedConversation back;
        std::vector<SessionReadLimits> limits(1);
        check(parent_load(lease, limits, back, error, &st) && same(original, back),
              "share: r1 reads the snapshot r0 published");
        lease.close_fds();
        check(lease.release("ok", error), "share: r1 releases");
        // r0's own entry is still there: a whole restore does not take it
        uint64_t cookie2 = 0;
        std::vector<int64_t> lengths2;
        check(a.match_begin(8, false, {}, {}, cookie2, lengths2, error) && lengths2.size() == 1,
              "share: the entry stays after a read");
    }

    // ---- SIGKILL of a client leaves the broker's state clean (it sees EOF and drops the channel)
    {
        Broker broker;
        broker.start();
        int sv[2] = {-1, -1};
        check(::socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv) == 0, "kill: socketpair");
        broker.channels.push_back(sv[0]);
        const pid_t pid = ::fork();
        check(pid >= 0, "kill: fork");
        if (pid == 0) {
            ConversationParentClient client;
            std::string error;
            if (!client.connect(sv[1], identity("r9"), 2000, error)) ::_exit(2);
            ::pause();   // wait for the signal
            ::_exit(0);
        }
        ::close(sv[1]);
        settle(broker, 1);
        check(broker.served.load() == 1, "kill: the child said HELLO");
        ::kill(pid, SIGKILL);
        int status = 0;
        ::waitpid(pid, &status, 0);
        for (int i = 0; i < 2000 && broker.eofs == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(broker.eofs == 1, "kill: the broker saw the channel's EOF");
        check(broker.failure.empty(), "kill: no protocol violation");
    }
#endif

    std::printf("conversation_parent_test: %d checks passed\n", checks);
    return 0;
}
