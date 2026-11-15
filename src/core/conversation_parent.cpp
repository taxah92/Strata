// The parent parking store's client: the wire format of contract.md (protocol v1) over the SOCK_SEQPACKET channel the
// engine inherits, and the free functions the engine needs around it.  No CUDA, no model: the tests build without a
// card.
//
// The client is single-threaded by design: the engine calls it from its request loop, one request is outstanding at a
// time, and every exchange is bounded by timeout_ms.  A transport failure (EOF, a timeout, a short or corrupt frame, a
// protocol violation) degrades the client for good: the socket is closed, every later call returns false with the
// recorded error, and the engine falls back to reading the prompt.  The store cleans up this channel's pending
// publications and leases on EOF; committed entries stay.
//
// Errors are strings of the form "parent parking: <code>: <message>".  <code> is the protocol's code for a store
// refusal ("capacity", "gone", "stale_generation", ...), or "transport", "protocol" or "bad_request" for a local
// failure.  last_error_code() carries the store's code alone, which is what the engine needs to tell a stale match
// from a real refusal.
#include "strata/core/conversation_parent.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <process.h>
#endif

namespace strata::core {
namespace {

// ---- the frozen protocol v1 (contract.md §1, §2)
constexpr char kMagic[8] = {'S', 'T', 'R', 'P', 'A', 'R', 'K', '1'};
constexpr uint16_t kVersion = 1;
constexpr size_t kHeaderBytes = 36;
constexpr size_t kMaxPayload = 1048576;
constexpr size_t kMaxPoints = 4096;
constexpr size_t kMaxStages = 64;
constexpr size_t kMaxFds = 64;
constexpr size_t kMaxExcludeIds = 1;
constexpr size_t kMaxOwner = 64;
constexpr uint64_t kPayloadSeed = 0x5354525041524B31ull;   // "STRPARK1" read as a big-endian u64
constexpr uint64_t kIdsSeed = 0x5354524944533031ull;       // "STRIDS01", conversation_disk_ids_hash's seed

enum : uint16_t {
    kOpHello = 1,
    kOpParkAdvice = 2,
    kOpParkBegin = 3,
    kOpParkCommit = 4,
    kOpParkAbort = 5,
    kOpMatchBegin = 6,
    kOpMatchEnd = 7,
    kOpAcquire = 8,
    kOpRelease = 9,
    kOpClear = 10,
    kOpStats = 11,
    kOpPing = 12,
    kOpError = 255,
};

int64_t current_pid() {
#ifdef _WIN32
    return (int64_t) ::_getpid();
#else
    return (int64_t) ::getpid();
#endif
}

// ---- JSON: the payloads are small objects of integers, strings, booleans and arrays.  A dependency is not worth it
// for this, and a strict parser is what "fail closed" needs: integers only, no duplicate keys, no trailing bytes, a
// bounded depth.
struct Json {
    enum class Type { null_value, boolean, integer, string, array, object };
    Type type = Type::null_value;
    bool flag = false;
    bool negative = false;
    uint64_t magnitude = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> members;

    const Json* find(const char* key) const {
        for (const auto& m : members)
            if (m.first == key) return &m.second;
        return nullptr;
    }
    bool as_u64(uint64_t& out) const {
        if (type != Type::integer || negative) return false;
        out = magnitude;
        return true;
    }
    bool as_i64(int64_t& out) const {
        if (type != Type::integer) return false;
        if (!negative) {
            if (magnitude > (uint64_t) std::numeric_limits<int64_t>::max()) return false;
            out = (int64_t) magnitude;
            return true;
        }
        const uint64_t limit = (uint64_t) std::numeric_limits<int64_t>::max() + 1;
        if (magnitude > limit) return false;
        out = magnitude == limit ? std::numeric_limits<int64_t>::min() : -(int64_t) magnitude;
        return true;
    }
    bool get_u64(const char* key, uint64_t& out) const {
        const Json* v = find(key);
        return v && v->as_u64(out);
    }
    bool get_i64(const char* key, int64_t& out) const {
        const Json* v = find(key);
        return v && v->as_i64(out);
    }
    bool get_bool(const char* key, bool& out) const {
        const Json* v = find(key);
        if (!v || v->type != Type::boolean) return false;
        out = v->flag;
        return true;
    }
    bool get_string(const char* key, std::string& out) const {
        const Json* v = find(key);
        if (!v || v->type != Type::string) return false;
        out = v->text;
        return true;
    }
    bool get_array(const char* key, const std::vector<Json>*& out) const {
        const Json* v = find(key);
        if (!v || v->type != Type::array) return false;
        out = &v->items;
        return true;
    }
    bool get_object(const char* key, const Json*& out) const {
        const Json* v = find(key);
        if (!v || v->type != Type::object) return false;
        out = v;
        return true;
    }
};

class JsonParser {
public:
    JsonParser(const char* begin, size_t n) : p_(begin), end_(begin + n) {}

    bool parse(Json& out, std::string& error) {
        skip();
        if (!value(out, 0)) {
            error = why_;
            return false;
        }
        skip();
        if (p_ != end_) {
            error = "trailing bytes after the JSON value";
            return false;
        }
        return true;
    }

private:
    bool fail(const char* why) {
        if (why_.empty()) why_ = why;
        return false;
    }
    void skip() {
        while (p_ != end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }
    bool value(Json& out, int depth) {
        if (depth > 32) return fail("the JSON is nested too deeply");
        skip();
        if (p_ == end_) return fail("the JSON ends early");
        switch (*p_) {
            case '{': return object(out, depth);
            case '[': return array(out, depth);
            case '"':
                out.type = Json::Type::string;
                return string(out.text);
            case 't': return literal("true", Json::Type::boolean, true, out);
            case 'f': return literal("false", Json::Type::boolean, false, out);
            case 'n': return literal("null", Json::Type::null_value, false, out);
            default: return number(out);
        }
    }
    bool literal(const char* word, Json::Type type, bool flag, Json& out) {
        const size_t n = std::strlen(word);
        if ((size_t) (end_ - p_) < n || std::memcmp(p_, word, n) != 0) return fail("an unknown literal");
        p_ += n;
        out.type = type;
        out.flag = flag;
        return true;
    }
    bool number(Json& out) {
        out.type = Json::Type::integer;
        out.negative = false;
        out.magnitude = 0;
        if (p_ != end_ && *p_ == '-') {
            out.negative = true;
            ++p_;
        }
        if (p_ == end_ || *p_ < '0' || *p_ > '9') return fail("expected an integer");
        if (*p_ == '0') {
            ++p_;
            if (p_ != end_ && *p_ >= '0' && *p_ <= '9') return fail("a leading zero");
        } else {
            while (p_ != end_ && *p_ >= '0' && *p_ <= '9') {
                const uint64_t digit = (uint64_t) (*p_ - '0');
                if (out.magnitude > (UINT64_MAX - digit) / 10) return fail("an integer is too large");
                out.magnitude = out.magnitude * 10 + digit;
                ++p_;
            }
        }
        if (p_ != end_ && (*p_ == '.' || *p_ == 'e' || *p_ == 'E')) return fail("only integers are supported");
        if (out.negative && out.magnitude > (uint64_t) std::numeric_limits<int64_t>::max() + 1)
            return fail("an integer is too small");
        return true;
    }
    bool hex4(uint32_t& out) {
        if (end_ - p_ < 4) return fail("a short \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else return fail("a bad \\u escape");
        }
        out = v;
        return true;
    }
    static void utf8(uint32_t cp, std::string& out) {
        if (cp < 0x80) {
            out.push_back((char) cp);
        } else if (cp < 0x800) {
            out.push_back((char) (0xC0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char) (0xE0 | (cp >> 12)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char) (0xF0 | (cp >> 18)));
            out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }
    bool string(std::string& out) {
        if (p_ == end_ || *p_ != '"') return fail("expected a string");
        ++p_;
        out.clear();
        while (p_ != end_) {
            const unsigned char c = (unsigned char) *p_++;
            if (c == '"') return true;
            if (c < 0x20) return fail("a control character in a string");
            if (c != '\\') {
                out.push_back((char) c);
                continue;
            }
            if (p_ == end_) return fail("the string ends early");
            const char e = *p_++;
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (end_ - p_ < 6 || p_[0] != '\\' || p_[1] != 'u') return fail("a lone surrogate");
                        p_ += 2;
                        uint32_t low = 0;
                        if (!hex4(low)) return false;
                        if (low < 0xDC00 || low > 0xDFFF) return fail("a lone surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("a lone surrogate");
                    }
                    utf8(cp, out);
                    break;
                }
                default: return fail("an unknown escape");
            }
        }
        return fail("the string ends early");
    }
    bool object(Json& out, int depth) {
        out.type = Json::Type::object;
        ++p_;   // '{'
        skip();
        if (p_ != end_ && *p_ == '}') {
            ++p_;
            return true;
        }
        for (;;) {
            skip();
            std::string key;
            if (!string(key)) return false;
            skip();
            if (p_ == end_ || *p_ != ':') return fail("expected ':'");
            ++p_;
            Json v;
            if (!value(v, depth + 1)) return false;
            for (const auto& m : out.members)
                if (m.first == key) return fail("a duplicate key");
            out.members.emplace_back(std::move(key), std::move(v));
            skip();
            if (p_ == end_) return fail("the object ends early");
            if (*p_ == ',') {
                ++p_;
                continue;
            }
            if (*p_ == '}') {
                ++p_;
                return true;
            }
            return fail("expected ',' or '}'");
        }
    }
    bool array(Json& out, int depth) {
        out.type = Json::Type::array;
        ++p_;   // '['
        skip();
        if (p_ != end_ && *p_ == ']') {
            ++p_;
            return true;
        }
        for (;;) {
            Json v;
            if (!value(v, depth + 1)) return false;
            out.items.push_back(std::move(v));
            skip();
            if (p_ == end_) return fail("the array ends early");
            if (*p_ == ',') {
                ++p_;
                continue;
            }
            if (*p_ == ']') {
                ++p_;
                return true;
            }
            return fail("expected ',' or ']'");
        }
    }

    const char* p_;
    const char* end_;
    std::string why_;
};

bool parse_json(const std::string& text, Json& out, std::string& why) {
    JsonParser parser(text.data(), text.size());
    return parser.parse(out, why);
}

class JsonOut {
public:
    std::string text;

    void object_begin() {
        comma();
        text += '{';
        first_.push_back(true);
    }
    void object_end() {
        text += '}';
        first_.pop_back();
    }
    void array_begin() {
        comma();
        text += '[';
        first_.push_back(true);
    }
    void array_end() {
        text += ']';
        first_.pop_back();
    }
    void key(const char* k) {
        comma();
        quote(k);
        text += ':';
        after_key_ = true;
    }
    void u64(uint64_t v) {
        comma();
        text += std::to_string(v);
    }
    void i64(int64_t v) {
        comma();
        text += std::to_string(v);
    }
    void boolean(bool v) {
        comma();
        text += v ? "true" : "false";
    }
    void null() {
        comma();
        text += "null";
    }
    void string(const std::string& v) {
        comma();
        quote(v);
    }

private:
    void comma() {
        if (after_key_) {
            after_key_ = false;
            return;
        }
        if (first_.empty()) return;
        if (!first_.back()) text += ',';
        first_.back() = false;
    }
    void quote(const std::string& v) {
        text += '"';
        for (const char c : v) {
            const unsigned char u = (unsigned char) c;
            switch (u) {
                case '"': text += "\\\""; break;
                case '\\': text += "\\\\"; break;
                case '\b': text += "\\b"; break;
                case '\f': text += "\\f"; break;
                case '\n': text += "\\n"; break;
                case '\r': text += "\\r"; break;
                case '\t': text += "\\t"; break;
                default:
                    if (u < 0x20) {
                        static const char* hex = "0123456789abcdef";
                        text += "\\u00";
                        text += hex[u >> 4];
                        text += hex[u & 0xF];
                    } else {
                        text += c;
                    }
            }
        }
        text += '"';
    }
    std::vector<bool> first_;
    bool after_key_ = false;
};

// ---- the checks both sides owe the contract (§1, §8)
bool valid_owner(const std::string& owner) {
    if (owner.empty() || owner.size() > kMaxOwner) return false;
    for (const char c : owner) {
        const unsigned char u = (unsigned char) c;
        if (u < 0x20 || u > 0x7E) return false;
    }
    return true;
}

bool valid_publish_id(const std::string& id) {
    if (id.size() != 32) return false;
    for (const char c : id)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

bool valid_result(const char* result) {
    if (!result) return false;
    const std::string r(result);
    return r == "ok" || r == "invalid" || r == "memory" || r == "io" || r == "mismatch";
}

bool valid_reason(const char* reason) {
    if (!reason) return false;
    const std::string r(reason);
    return r == "nonfinite" || r == "unload" || r == "test";
}

bool checked_sum(const std::vector<uint64_t>& values, uint64_t& out) {
    uint64_t n = 0;
    for (uint64_t v : values) {
        if (v > UINT64_MAX - n) return false;
        n += v;
    }
    out = n;
    return true;
}

void put_points(JsonOut& j, const std::vector<ParentPoint>& points) {
    j.array_begin();
    for (const auto& p : points) {
        j.object_begin();
        j.key("tokens");
        j.i64(p.tokens);
        j.key("hash");
        j.u64(p.hash);
        j.key("imgs");
        j.array_begin();
        for (const auto& im : p.imgs) {
            j.object_begin();
            j.key("start");
            j.i64(im.start);
            j.key("hash");
            j.u64(im.hash);
            j.object_end();
        }
        j.array_end();
        j.key("live");
        j.boolean(p.live);
        j.key("pinned");
        j.boolean(p.pinned);
        j.object_end();
    }
    j.array_end();
}

bool get_candidate(const Json& v, ParentCandidate& out, std::string& why) {
    if (v.type != Json::Type::object) {
        why = "the candidate is not an object";
        return false;
    }
    if (!v.get_u64("id", out.id) || !v.get_u64("generation", out.generation) || !v.get_i64("tokens", out.tokens) ||
        !v.get_bool("live", out.live) || !v.get_bool("borrow", out.borrow) ||
        !v.get_u64("read_charge_bytes", out.read_charge_bytes) || !v.get_string("owner", out.owner)) {
        why = "the candidate is missing a field";
        return false;
    }
    const std::vector<Json>* sizes = nullptr;
    if (!v.get_array("stage_bytes", sizes)) {
        why = "the candidate has no stage sizes";
        return false;
    }
    out.stage_bytes.clear();
    for (const auto& s : *sizes) {
        uint64_t n = 0;
        if (!s.as_u64(n)) {
            why = "a stage size is not an integer";
            return false;
        }
        out.stage_bytes.push_back(n);
    }
    return true;
}

// ---- the lease registry.  The frozen header's parent_load takes only the lease, but a session file is bound to the
// store's identity (the model and config fingerprints) and session_file_read_fd checks it.  The client records the
// identity of every lease it hands out, keyed by the store's lease id, and parent_load looks it up.  The record also
// names the client, so a lease that outlives its client is dropped locally instead of dereferencing a dead pointer.
struct LeaseRecord {
    ConversationParentClient* client = nullptr;
    ParentIdentity identity;
};

std::mutex& lease_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<uint64_t, LeaseRecord>& lease_records() {
    static std::unordered_map<uint64_t, LeaseRecord> m;
    return m;
}

void remember_lease(uint64_t lease_id, ConversationParentClient* client, const ParentIdentity& identity) {
    std::lock_guard<std::mutex> lock(lease_mutex());
    lease_records()[lease_id] = LeaseRecord{client, identity};
}

bool find_lease(uint64_t lease_id, LeaseRecord& out) {
    std::lock_guard<std::mutex> lock(lease_mutex());
    const auto it = lease_records().find(lease_id);
    if (it == lease_records().end()) return false;
    out = it->second;
    return true;
}

void forget_lease(uint64_t lease_id) {
    std::lock_guard<std::mutex> lock(lease_mutex());
    lease_records().erase(lease_id);
}

void forget_client_leases(ConversationParentClient* client) {
    std::lock_guard<std::mutex> lock(lease_mutex());
    for (auto it = lease_records().begin(); it != lease_records().end();) {
        if (it->second.client == client) it = lease_records().erase(it);
        else ++it;
    }
}

#ifndef _WIN32
// ---- the frame on the wire (contract.md §2.1)
int64_t now_ms() {
    return (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void close_all(std::vector<int>& fds) {
    for (int fd : fds)
        if (fd >= 0) ::close(fd);
    fds.clear();
}

void frame_header(uint8_t* h, uint16_t op, uint32_t payload_bytes, uint64_t request_id, uint64_t payload_hash) {
    std::memcpy(h, kMagic, 8);
    const uint16_t version = kVersion;
    const uint32_t flags = 0;
    std::memcpy(h + 8, &version, 2);
    std::memcpy(h + 10, &op, 2);
    std::memcpy(h + 12, &flags, 4);
    std::memcpy(h + 16, &payload_bytes, 4);
    std::memcpy(h + 20, &request_id, 8);
    std::memcpy(h + 28, &payload_hash, 8);
}

enum class Io { ok, timeout, closed, failed };

// The receive buffer is one frame's worth and is reused: a control exchange is small, and a 1 MiB allocation per
// request would be waste.
uint8_t* receive_buffer() {
    static thread_local std::unique_ptr<uint8_t[]> buffer;
    if (!buffer) buffer.reset(new (std::nothrow) uint8_t[kHeaderBytes + kMaxPayload]);
    return buffer.get();
}

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

Io send_frame(int fd, const uint8_t* header, const std::string& payload, int64_t deadline, std::string& why) {
    struct iovec iov[2];
    iov[0].iov_base = const_cast<uint8_t*>(header);
    iov[0].iov_len = kHeaderBytes;
    iov[1].iov_base = const_cast<char*>(payload.data());
    iov[1].iov_len = payload.size();
    struct msghdr msg {};
    msg.msg_iov = iov;
    msg.msg_iovlen = payload.empty() ? 1 : 2;
    for (;;) {
        const ssize_t n = ::sendmsg(fd, &msg, kSendFlags);
        if (n == (ssize_t) (kHeaderBytes + payload.size())) return Io::ok;
        if (n >= 0) {
            why = "a short send";
            return Io::failed;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            const int64_t left = deadline - now_ms();
            if (left <= 0) {
                why = "the store did not take the request in time";
                return Io::timeout;
            }
            struct pollfd pfd { fd, POLLOUT, 0 };
            const int r = ::poll(&pfd, 1, (int) std::min<int64_t>(left, 60000));
            if (r < 0 && errno != EINTR) {
                why = std::strerror(errno);
                return Io::failed;
            }
            continue;
        }
        if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN) {
            why = "the store closed the channel";
            return Io::closed;
        }
        why = std::strerror(errno);
        return Io::failed;
    }
}

struct Frame {
    uint8_t header[kHeaderBytes];
    std::string payload;
    std::vector<int> fds;
};

Io recv_frame(int fd, int64_t deadline, Frame& frame, std::string& why) {
    uint8_t* buffer = receive_buffer();
    if (!buffer) {
        why = "out of memory for the receive buffer";
        return Io::failed;
    }
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * kMaxFds)];
    for (;;) {
        const int64_t left = deadline - now_ms();
        if (left <= 0) {
            why = "the store did not answer in time";
            return Io::timeout;
        }
        struct pollfd pfd { fd, POLLIN, 0 };
        const int r = ::poll(&pfd, 1, (int) std::min<int64_t>(left, 60000));
        if (r < 0) {
            if (errno == EINTR) continue;
            why = std::strerror(errno);
            return Io::failed;
        }
        if (r == 0) continue;   // the deadline is recomputed above
        struct iovec iov { buffer, kHeaderBytes + kMaxPayload };
        struct msghdr msg {};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        const ssize_t n = ::recvmsg(fd, &msg, 0);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            if (errno == ECONNRESET || errno == ENOTCONN) {
                why = "the store closed the channel";
                return Io::closed;
            }
            why = std::strerror(errno);
            return Io::failed;
        }
        if (n == 0) {
            why = "the store closed the channel";
            return Io::closed;
        }
        if (msg.msg_flags & MSG_TRUNC) {
            why = "the frame is larger than the protocol allows";
            return Io::failed;
        }
        if ((size_t) n < kHeaderBytes) {
            why = "a frame shorter than its header";
            return Io::failed;
        }
        std::memcpy(frame.header, buffer, kHeaderBytes);
        frame.payload.assign((const char*) buffer + kHeaderBytes, (size_t) n - kHeaderBytes);
        frame.fds.clear();
        for (struct cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
            if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) continue;
            const size_t bytes = c->cmsg_len > CMSG_LEN(0) ? (size_t) c->cmsg_len - CMSG_LEN(0) : 0;
            const size_t count = bytes / sizeof(int);
            const int* received = (const int*) (void*) CMSG_DATA(c);
            for (size_t i = 0; i < count; ++i) frame.fds.push_back(received[i]);
        }
        if (msg.msg_flags & MSG_CTRUNC) {
            close_all(frame.fds);
            why = "the frame's descriptors were truncated";
            return Io::failed;
        }
        if (frame.fds.size() > kMaxFds) {
            close_all(frame.fds);
            why = "too many descriptors in one frame";
            return Io::failed;
        }
        return Io::ok;
    }
}

// A response that arrives after its request was given up on is stale: the next request must not read it as its own.
void drain_frames(int fd) {
    uint8_t buffer[64];
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * kMaxFds)];
    for (;;) {
        struct iovec iov { buffer, sizeof(buffer) };
        struct msghdr msg {};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        const ssize_t n = ::recvmsg(fd, &msg, MSG_DONTWAIT);
        if (n < 0) {
            if (errno == EINTR) continue;
            return;   // EAGAIN, or a broken channel the next send reports
        }
        if (n == 0) return;
        for (struct cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
            if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) continue;
            const size_t bytes = c->cmsg_len > CMSG_LEN(0) ? (size_t) c->cmsg_len - CMSG_LEN(0) : 0;
            const size_t count = bytes / sizeof(int);
            const int* received = (const int*) (void*) CMSG_DATA(c);
            for (size_t i = 0; i < count; ++i) ::close(received[i]);
        }
    }
}
#endif   // _WIN32

} // namespace

// ---- the free functions the engine needs around the client

std::vector<ParentPoint> parent_points(const SavedConversation& image) {
    std::vector<ParentPoint> out;
    out.reserve(1 + image.checkpoints.size());
    auto add = [&out](const ConversationCheckpoint& c, bool live) {
        ParentPoint p;
        p.tokens = (int64_t) c.ids.size();
        p.hash = parent_ids_hash(c.ids.data(), c.ids.size());
        p.imgs = c.imgs;
        p.live = live;
        p.pinned = c.pinned;
        out.push_back(std::move(p));
    };
    add(image.live, true);
    for (const auto& c : image.checkpoints) add(c, false);
    return out;
}

uint64_t parent_ids_hash(const int32_t* ids, size_t n) {
    return session_hash64(ids, n * sizeof(int32_t), kIdsSeed);
}

std::vector<std::pair<int64_t, uint64_t>> parent_hashes(const std::vector<int32_t>& ids,
                                                        const std::vector<int64_t>& lengths) {
    // The store sends the lengths sorted and unique; sort a copy anyway so a caller's order cannot make the walk go
    // backwards, and hash each distinct length once.
    std::vector<int64_t> wanted;
    wanted.reserve(lengths.size());
    for (int64_t L : lengths)
        if (L > 0 && (uint64_t) L <= ids.size()) wanted.push_back(L);
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    std::vector<std::pair<int64_t, uint64_t>> at;
    at.reserve(wanted.size());
    SessionHasher h(kIdsSeed);
    size_t done = 0;
    for (int64_t L : wanted) {
        h.update(ids.data() + done, ((size_t) L - done) * sizeof(int32_t));
        done = (size_t) L;
        at.emplace_back(L, h.digest());
    }
    // the caller's order, with the lengths that cannot be hashed left out
    std::vector<std::pair<int64_t, uint64_t>> out;
    out.reserve(lengths.size());
    for (int64_t L : lengths) {
        const auto it = std::lower_bound(at.begin(), at.end(), std::make_pair(L, uint64_t(0)),
                                         [](const auto& a, const auto& b) { return a.first < b.first; });
        if (it != at.end() && it->first == L) out.push_back(*it);
    }
    return out;
}

bool parent_load(ParentLease& lease, const std::vector<SessionReadLimits>& limits, SavedConversation& image,
                 std::string& error, SessionStatus* status) {
    SessionStatus st;
    st.error = SessionError::invalid;
    auto done = [&](bool ok) {
        if (status) *status = st;
        return ok;
    };
    if (!lease.valid()) {
        error = "parent parking: no lease to load";
        return done(false);
    }
    LeaseRecord record;
    if (!find_lease(lease.lease_id(), record)) {
        error = "parent parking: the lease is not this client's";
        return done(false);
    }
    const size_t stages = lease.stages();
    if (stages == 0 || stages != record.identity.stages.size()) {
        error = "parent parking: the lease has " + std::to_string(stages) + " stages, the store's identity has " +
                std::to_string(record.identity.stages.size());
        return done(false);
    }
    const SessionFileIdentity id{record.identity.model, record.identity.config};
    SavedConversation loaded;
    SessionReadLimits defaults;
    for (size_t k = 0; k < stages; ++k) {
        const int fd = lease.fd(k);
        if (fd < 0) {
            error = "parent parking: the lease has no descriptor for stage " + std::to_string(k);
            lease.close_fds();
            return done(false);
        }
        SavedConversation stage;
        size_t bytes = 0;
        const SessionReadLimits& stage_limits = k < limits.size() ? limits[k] : defaults;
        if (!session_file_read_fd(fd, id, stage, bytes, error, stage_limits, &st)) {
            lease.close_fds();
            return done(false);
        }
        if (k == 0) loaded = std::move(stage);
        else loaded.stage_images.push_back(std::move(stage));
    }
    lease.close_fds();
    image = std::move(loaded);
    st.error = SessionError::none;
    return done(true);
}

std::string parent_publish_id() {
    static std::mutex mutex;
    static std::mt19937_64 rng = [] {
        std::mt19937_64 engine;
        try {
            std::random_device rd;
            std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
            engine.seed(seq);
        } catch (...) {
            // no entropy source: the clock, the process and the engine's own address are enough for an id that only
            // has to be unique within one store's lifetime, not unpredictable
            const uint64_t t = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
            std::seed_seq seq{t, (uint64_t) (uintptr_t) &engine, (uint64_t) current_pid()};
            engine.seed(seq);
        }
        return engine;
    }();
    uint64_t a = 0, b = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        a = rng();
        b = rng();
    }
    static const char* hex = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
        out[(size_t) i] = hex[(a >> (60 - 4 * i)) & 0xF];
        out[(size_t) (16 + i)] = hex[(b >> (60 - 4 * i)) & 0xF];
    }
    return out;
}

// ---- ParentPending

ParentPending::ParentPending(ParentPending&& other) noexcept
    : pending_id_(other.pending_id_), fds_(std::move(other.fds_)), stage_bytes_(std::move(other.stage_bytes_)) {
    other.pending_id_ = 0;
    other.fds_.clear();
    other.stage_bytes_.clear();
}

ParentPending& ParentPending::operator=(ParentPending&& other) noexcept {
    if (this == &other) return *this;
    close_fds();
    pending_id_ = other.pending_id_;
    fds_ = std::move(other.fds_);
    stage_bytes_ = std::move(other.stage_bytes_);
    other.pending_id_ = 0;
    other.fds_.clear();
    other.stage_bytes_.clear();
    return *this;
}

void ParentPending::close_fds() {
#ifndef _WIN32
    for (int fd : fds_)
        if (fd >= 0) ::close(fd);
#endif
    fds_.clear();
}

void ParentPending::reset() {
    close_fds();
    pending_id_ = 0;
    stage_bytes_.clear();
}

// ---- ParentLease

ParentLease::ParentLease(ParentLease&& other) noexcept
    : client_(other.client_), lease_id_(other.lease_id_), candidate_(std::move(other.candidate_)),
      fds_(std::move(other.fds_)) {
    other.client_ = nullptr;
    other.lease_id_ = 0;
    other.candidate_ = {};
    other.fds_.clear();
}

ParentLease& ParentLease::operator=(ParentLease&& other) noexcept {
    if (this == &other) return *this;
    if (lease_id_ != 0) {
        // a move into a live lease ends the old one the way the destructor would; the store keeps the entry for
        // another try
        try {
            std::string ignored;
            release("io", ignored);
        } catch (...) {
            lease_id_ = 0;
            close_fds();
        }
    }
    client_ = other.client_;
    lease_id_ = other.lease_id_;
    candidate_ = std::move(other.candidate_);
    fds_ = std::move(other.fds_);
    other.client_ = nullptr;
    other.lease_id_ = 0;
    other.candidate_ = {};
    other.fds_.clear();
    return *this;
}

ParentLease::~ParentLease() {
    if (lease_id_ == 0) return;
    try {
        std::string ignored;
        release("io", ignored);
    } catch (...) {
        lease_id_ = 0;
        close_fds();
    }
}

void ParentLease::close_fds() {
#ifndef _WIN32
    for (int fd : fds_)
        if (fd >= 0) ::close(fd);
#endif
    fds_.clear();
}

bool ParentLease::release(const char* result, std::string& error) {
    if (lease_id_ == 0) return true;   // idempotent: already released
    const uint64_t id = lease_id_;
    LeaseRecord record;
    bool ok = false;
    if (find_lease(id, record) && record.client) {
        ok = record.client->release(id, result, error);
    } else {
        error = "parent parking: the channel is gone";
    }
    lease_id_ = 0;
    candidate_ = {};
    close_fds();
    forget_lease(id);
    return ok;
}

// ---- ConversationParentClient

ConversationParentClient::~ConversationParentClient() {
    close();
    forget_client_leases(this);
}

void ConversationParentClient::close() {
    if (fd_ >= 0) {
#ifndef _WIN32
        ::close(fd_);
#endif
        fd_ = -1;
    }
    state_ = State::off;
}

void ConversationParentClient::degrade(const std::string& why) {
    last_error_ = why;
    state_ = State::degraded;
    if (fd_ >= 0) {
#ifndef _WIN32
        ::close(fd_);
#endif
        fd_ = -1;
    }
}

bool ConversationParentClient::connect(int fd, const ParentIdentity& identity, int timeout_ms, std::string& error) {
    close();
    error.clear();
    if (fd < 0) {
        error = "parent parking: no descriptor to connect";
        return false;
    }
    if (!valid_owner(identity.owner)) {
        error = "parent parking: bad_request: the owner must be 1..64 printable ASCII characters";
        return false;
    }
    if (identity.stages.empty() || identity.stages.size() > kMaxStages) {
        error = "parent parking: bad_request: the stage count is outside 1..64";
        return false;
    }
    for (const auto& stage : identity.stages) {
        if (stage.first >= stage.second) {
            error = "parent parking: bad_request: a stage's layer range is empty";
            return false;
        }
    }
    fd_ = fd;
    identity_ = identity;
    timeout_ms_ = timeout_ms > 0 ? timeout_ms : 2000;
    epoch_ = 0;
    next_request_id_ = 1;
    last_error_.clear();
    last_error_code_.clear();
#ifndef _WIN32
    const int flags = ::fcntl(fd_, F_GETFL);
    if (flags < 0 || ::fcntl(fd_, F_SETFL, flags | O_NONBLOCK) != 0) {
        error = std::string("parent parking: the descriptor: ") + std::strerror(errno);
        close();
        return false;
    }
#ifdef SO_NOSIGPIPE
    const int one = 1;
    (void) ::setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
#endif
    state_ = State::connected;
    JsonOut j;
    j.object_begin();
    j.key("owner");
    j.string(identity_.owner);
    j.key("protocol");
    j.u64(kVersion);
    j.key("model");
    j.u64(identity_.model);
    j.key("config");
    j.u64(identity_.config);
    j.key("layout_key");
    j.u64(identity_.layout_key);
    j.key("stages");
    j.array_begin();
    for (const auto& stage : identity_.stages) {
        j.array_begin();
        j.i64(stage.first);
        j.i64(stage.second);
        j.array_end();
    }
    j.array_end();
    j.key("pid");
    j.i64(current_pid());
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpHello, j.text, response, fds, error)) {
        close();
        return false;
    }
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        error = "parent parking: protocol: the HELLO answer is not JSON: " + why;
        close();
        return false;
    }
    uint64_t store_epoch = 0;
    if (!r.get_u64("store_epoch", store_epoch)) {
        error = "parent parking: protocol: the HELLO answer has no store_epoch";
        close();
        return false;
    }
    epoch_ = store_epoch;
    return true;
}

bool ConversationParentClient::request(uint16_t op, const std::string& payload, std::string& response,
                                       std::vector<int>& fds, std::string& error) {
    fds.clear();
    response.clear();
    last_error_code_.clear();
#ifdef _WIN32
    (void) op;
    (void) payload;
    error = "parent parking: the parent parking store is not supported on this platform";
    return false;
#else
    if (state_ != State::connected) {
        error = last_error_.empty() ? "parent parking: the channel is not connected" : last_error_;
        return false;
    }
    if (payload.size() > kMaxPayload) {
        error = "parent parking: bad_request: the payload is larger than the protocol allows";
        return false;
    }
    const bool takes_fds = op == kOpParkBegin || op == kOpAcquire;
    // A lost COMMIT answer is the one failure worth repeating: the store is idempotent by publish_id, so the second
    // COMMIT either publishes or returns the first answer.  Every other op is not repeated.
    const int attempts = op == kOpParkCommit ? 2 : 1;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        drain_frames(fd_);
        const uint64_t request_id = next_request_id_++;
        uint8_t header[kHeaderBytes];
        frame_header(header, op, (uint32_t) payload.size(), request_id,
                     session_hash64(payload.data(), payload.size(), kPayloadSeed));
        const int64_t deadline = now_ms() + (timeout_ms_ > 0 ? timeout_ms_ : 1);
        std::string why;
        if (send_frame(fd_, header, payload, deadline, why) != Io::ok) {
            degrade("parent parking: transport: " + why);
            error = last_error_;
            return false;
        }
        Frame frame;
        const Io got = recv_frame(fd_, deadline, frame, why);
        if (got != Io::ok) {
            if (got == Io::timeout && attempt + 1 < attempts) continue;
            degrade("parent parking: transport: " + why);
            error = last_error_;
            return false;
        }
        if (std::memcmp(frame.header, kMagic, 8) != 0) {
            close_all(frame.fds);
            degrade("parent parking: protocol: the answer's frame magic is wrong");
            error = last_error_;
            return false;
        }
        uint16_t version = 0, answer_op = 0;
        uint32_t flags = 0, payload_bytes = 0;
        uint64_t answer_id = 0, payload_hash = 0;
        std::memcpy(&version, frame.header + 8, 2);
        std::memcpy(&answer_op, frame.header + 10, 2);
        std::memcpy(&flags, frame.header + 12, 4);
        std::memcpy(&payload_bytes, frame.header + 16, 4);
        std::memcpy(&answer_id, frame.header + 20, 8);
        std::memcpy(&payload_hash, frame.header + 28, 8);
        if (version != kVersion || flags != 0) {
            close_all(frame.fds);
            degrade("parent parking: protocol: the answer's frame header is not protocol v1");
            error = last_error_;
            return false;
        }
        if (payload_bytes != frame.payload.size()) {
            close_all(frame.fds);
            degrade("parent parking: protocol: the answer's payload size does not match its frame");
            error = last_error_;
            return false;
        }
        if (answer_id != request_id) {
            close_all(frame.fds);
            degrade("parent parking: protocol: the answer belongs to another request");
            error = last_error_;
            return false;
        }
        if (payload_hash != session_hash64(frame.payload.data(), frame.payload.size(), kPayloadSeed)) {
            close_all(frame.fds);
            degrade("parent parking: transport: the answer's payload hash does not match");
            error = last_error_;
            return false;
        }
        if (!takes_fds && !frame.fds.empty()) {
            close_all(frame.fds);
            degrade("parent parking: protocol: the answer carried descriptors it must not");
            error = last_error_;
            return false;
        }
        if (answer_op == kOpError) {
            close_all(frame.fds);
            Json parsed;
            std::string why_json;
            if (!parse_json(frame.payload, parsed, why_json)) {
                degrade("parent parking: protocol: the error answer is not JSON: " + why_json);
                error = last_error_;
                return false;
            }
            std::string code, message;
            if (!parsed.get_string("code", code) || !parsed.get_string("message", message)) {
                degrade("parent parking: protocol: the error answer is missing a field");
                error = last_error_;
                return false;
            }
            last_error_code_ = code;
            error = "parent parking: " + code + (message.empty() ? std::string() : ": " + message);
            // the store closes the channel after these two: there is nothing left to talk to
            if (code == "bad_request" || code == "protocol") degrade(error);
            return false;
        }
        if (answer_op != op) {
            close_all(frame.fds);
            degrade("parent parking: protocol: the answer's op does not match the request");
            error = last_error_;
            return false;
        }
        response = std::move(frame.payload);
        fds = std::move(frame.fds);
        return true;
    }
    error = last_error_.empty() ? "parent parking: transport: the request was not answered" : last_error_;
    return false;
#endif
}

bool ConversationParentClient::advice(const std::vector<ParentPoint>& points, bool cvec, ParentAdvice& out,
                                      std::string& error) {
    out = {};
    if (points.size() > kMaxPoints) {
        error = "parent parking: bad_request: too many resume points";
        return false;
    }
    JsonOut j;
    j.object_begin();
    j.key("cvec");
    j.boolean(cvec);
    j.key("points");
    put_points(j, points);
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpParkAdvice, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        degrade("parent parking: protocol: the advice answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    ParentAdvice parsed;
    if (!r.get_u64("holding_id", parsed.holding_id) ||
        !r.get_i64("parked_prefix_tokens", parsed.parked_prefix_tokens)) {
        degrade("parent parking: protocol: the advice answer is missing a field");
        error = last_error_;
        return false;
    }
    uint64_t epoch = 0;
    if (r.get_u64("epoch", epoch)) epoch_ = epoch;
    out = parsed;
    return true;
}

bool ConversationParentClient::begin(const std::string& publish_id, const std::vector<ParentPoint>& points, bool cvec,
                                     const std::vector<uint64_t>& stage_bytes, ParentPending& out, bool& dedup,
                                     std::string& error) {
    out.reset();
    dedup = false;
    if (!valid_publish_id(publish_id)) {
        error = "parent parking: bad_request: the publish id must be 32 lowercase hex characters";
        return false;
    }
    if (stage_bytes.empty() || stage_bytes.size() > kMaxStages) {
        error = "parent parking: bad_request: the stage count is outside 1..64";
        return false;
    }
    if (stage_bytes.size() != identity_.stages.size()) {
        error = "parent parking: bad_request: " + std::to_string(stage_bytes.size()) + " stage sizes for " +
                std::to_string(identity_.stages.size()) + " stages";
        return false;
    }
    if (points.size() > kMaxPoints) {
        error = "parent parking: bad_request: too many resume points";
        return false;
    }
    uint64_t total = 0;
    if (!checked_sum(stage_bytes, total)) {
        error = "parent parking: bad_request: the stage sizes overflow";
        return false;
    }
    JsonOut j;
    j.object_begin();
    j.key("publish_id");
    j.string(publish_id);
    j.key("owner");
    j.string(identity_.owner);
    j.key("model");
    j.u64(identity_.model);
    j.key("config");
    j.u64(identity_.config);
    j.key("layout_key");
    j.u64(identity_.layout_key);
    j.key("stages");
    j.array_begin();
    for (const auto& stage : identity_.stages) {
        j.array_begin();
        j.i64(stage.first);
        j.i64(stage.second);
        j.array_end();
    }
    j.array_end();
    j.key("cvec");
    j.boolean(cvec);
    j.key("points");
    put_points(j, points);
    j.key("stage_bytes");
    j.array_begin();
    for (uint64_t n : stage_bytes) j.u64(n);
    j.array_end();
    j.key("epoch");
    j.u64(epoch_);
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpParkBegin, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        close_all(fds);
        degrade("parent parking: protocol: the begin answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    uint64_t pending_id = 0, holding_id = 0;
    bool is_dedup = false;
    const std::vector<Json>* echoed = nullptr;
    if (!r.get_u64("pending_id", pending_id) || !r.get_bool("dedup", is_dedup) ||
        !r.get_u64("holding_id", holding_id) || !r.get_array("stage_bytes", echoed)) {
        close_all(fds);
        degrade("parent parking: protocol: the begin answer is missing a field");
        error = last_error_;
        return false;
    }
    (void) holding_id;   // informational: the dedup answer's entry, which the caller does not need here
    if (is_dedup) {
        if (!fds.empty()) {
            close_all(fds);
            degrade("parent parking: protocol: a dedup answer carried descriptors");
            error = last_error_;
            return false;
        }
        if (pending_id != 0) {
            degrade("parent parking: protocol: a dedup answer has a pending id");
            error = last_error_;
            return false;
        }
        dedup = true;
        return true;
    }
    if (pending_id == 0) {
        close_all(fds);
        degrade("parent parking: protocol: a begin answer has no pending id");
        error = last_error_;
        return false;
    }
    if (fds.size() != stage_bytes.size()) {
        const size_t got = fds.size();
        close_all(fds);
        degrade("parent parking: protocol: the store sent " + std::to_string(got) + " descriptors for " +
                std::to_string(stage_bytes.size()) + " stages");
        error = last_error_;
        return false;
    }
    std::vector<uint64_t> sizes;
    sizes.reserve(echoed->size());
    for (const auto& v : *echoed) {
        uint64_t n = 0;
        if (!v.as_u64(n)) {
            close_all(fds);
            degrade("parent parking: protocol: a stage size is not an integer");
            error = last_error_;
            return false;
        }
        sizes.push_back(n);
    }
    if (sizes != stage_bytes) {
        close_all(fds);
        degrade("parent parking: protocol: the store echoed other stage sizes");
        error = last_error_;
        return false;
    }
#ifndef _WIN32
    for (size_t k = 0; k < fds.size(); ++k) {
        struct stat st {};
        if (::fstat(fds[k], &st) != 0 || !S_ISREG(st.st_mode) || (uint64_t) st.st_size != stage_bytes[k] ||
            st.st_uid != ::getuid()) {
            close_all(fds);
            degrade("parent parking: protocol: the store's writable descriptor " + std::to_string(k) +
                    " is not a regular file of the asked size");
            error = last_error_;
            return false;
        }
    }
#endif
    out.pending_id_ = pending_id;
    out.fds_ = std::move(fds);
    out.stage_bytes_ = stage_bytes;
    return true;
}

bool ConversationParentClient::commit(const std::string& publish_id, const ParentPending& pending, uint64_t& id,
                                      uint64_t& generation, std::string& error) {
    id = 0;
    generation = 0;
    if (!valid_publish_id(publish_id)) {
        error = "parent parking: bad_request: the publish id must be 32 lowercase hex characters";
        return false;
    }
    if (!pending.valid()) {
        error = "parent parking: bad_request: no pending publication";
        return false;
    }
    JsonOut j;
    j.object_begin();
    j.key("publish_id");
    j.string(publish_id);
    j.key("pending_id");
    j.u64(pending.id());
    j.key("written_bytes");
    j.array_begin();
    for (uint64_t n : pending.stage_bytes()) j.u64(n);
    j.array_end();
    j.key("epoch");
    j.u64(epoch_);
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpParkCommit, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        degrade("parent parking: protocol: the commit answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    uint64_t entry_bytes = 0;
    bool dedup = false;
    if (!r.get_u64("id", id) || !r.get_u64("generation", generation) || !r.get_u64("entry_bytes", entry_bytes) ||
        !r.get_bool("dedup", dedup)) {
        degrade("parent parking: protocol: the commit answer is missing a field");
        error = last_error_;
        return false;
    }
    (void) entry_bytes;
    (void) dedup;
    if (id == 0 || generation == 0) {
        degrade("parent parking: protocol: the commit answer has a zero id or generation");
        error = last_error_;
        return false;
    }
    return true;
}

bool ConversationParentClient::abort(const std::string& publish_id, uint64_t pending_id, std::string& error) {
    if (!valid_publish_id(publish_id)) {
        error = "parent parking: bad_request: the publish id must be 32 lowercase hex characters";
        return false;
    }
    if (pending_id == 0) return true;   // nothing was created: there is nothing to abort
    JsonOut j;
    j.object_begin();
    j.key("publish_id");
    j.string(publish_id);
    j.key("pending_id");
    j.u64(pending_id);
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpParkAbort, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        degrade("parent parking: protocol: the abort answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    bool gone = false;
    if (!r.get_bool("gone", gone)) {
        degrade("parent parking: protocol: the abort answer has no gone");
        error = last_error_;
        return false;
    }
    (void) gone;   // idempotent either way
    return true;
}

bool ConversationParentClient::match_begin(int64_t prompt_tokens, bool cvec,
                                           const std::vector<ConversationImageKey>& images,
                                           const std::vector<uint64_t>& exclude_ids, uint64_t& cookie,
                                           std::vector<int64_t>& lengths, std::string& error) {
    cookie = 0;
    lengths.clear();
    if (prompt_tokens < 0) {
        error = "parent parking: bad_request: a negative prompt length";
        return false;
    }
    if (exclude_ids.size() > kMaxExcludeIds) {
        error = "parent parking: bad_request: more than one excluded id";
        return false;
    }
    JsonOut j;
    j.object_begin();
    j.key("prompt_tokens");
    j.i64(prompt_tokens);
    j.key("cvec");
    j.boolean(cvec);
    j.key("images");
    j.array_begin();
    for (const auto& im : images) {
        j.object_begin();
        j.key("start");
        j.i64(im.start);
        j.key("hash");
        j.u64(im.hash);
        j.object_end();
    }
    j.array_end();
    j.key("exclude_ids");
    j.array_begin();
    for (uint64_t id : exclude_ids) j.u64(id);
    j.array_end();
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpMatchBegin, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        degrade("parent parking: protocol: the match answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    const std::vector<Json>* items = nullptr;
    if (!r.get_u64("cookie", cookie) || !r.get_array("lengths", items)) {
        degrade("parent parking: protocol: the match answer is missing a field");
        error = last_error_;
        return false;
    }
    if (cookie == 0) {
        degrade("parent parking: protocol: the match answer has a zero cookie");
        error = last_error_;
        return false;
    }
    int64_t previous = 0;
    for (const auto& v : *items) {
        int64_t length = 0;
        if (!v.as_i64(length)) {
            degrade("parent parking: protocol: a match length is not an integer");
            error = last_error_;
            return false;
        }
        if (length <= previous || length < 1 || length > prompt_tokens - 1) {
            degrade("parent parking: protocol: the match lengths are not sorted, unique and inside the prompt");
            error = last_error_;
            return false;
        }
        previous = length;
        lengths.push_back(length);
    }
    return true;
}

bool ConversationParentClient::match_end(uint64_t cookie, const std::vector<std::pair<int64_t, uint64_t>>& hashes,
                                         std::optional<ParentCandidate>& out, std::string& error) {
    out.reset();
    if (cookie == 0) {
        error = "parent parking: bad_request: a zero match cookie";
        return false;
    }
    JsonOut j;
    j.object_begin();
    j.key("cookie");
    j.u64(cookie);
    j.key("hashes");
    j.array_begin();
    for (const auto& h : hashes) {
        j.object_begin();
        j.key("tokens");
        j.i64(h.first);
        j.key("hash");
        j.u64(h.second);
        j.object_end();
    }
    j.array_end();
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpMatchEnd, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        degrade("parent parking: protocol: the match end answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    const Json* candidate = r.find("candidate");
    if (!candidate) {
        degrade("parent parking: protocol: the match end answer has no candidate");
        error = last_error_;
        return false;
    }
    if (candidate->type == Json::Type::null_value) return true;
    ParentCandidate parsed;
    if (!get_candidate(*candidate, parsed, why)) {
        degrade("parent parking: protocol: " + why);
        error = last_error_;
        return false;
    }
    if (parsed.id == 0 || parsed.generation == 0 || !valid_owner(parsed.owner) ||
        parsed.stage_bytes.size() != identity_.stages.size()) {
        degrade("parent parking: protocol: the candidate is not a usable entry");
        error = last_error_;
        return false;
    }
    for (uint64_t n : parsed.stage_bytes)
        if (n == 0) {
            degrade("parent parking: protocol: the candidate has an empty stage");
            error = last_error_;
            return false;
        }
    out = std::move(parsed);
    return true;
}

bool ConversationParentClient::acquire(const ParentCandidate& candidate, ParentLease& out, std::string& error) {
    if (candidate.id == 0 || candidate.generation == 0) {
        error = "parent parking: bad_request: the candidate has no id or generation";
        return false;
    }
    if (candidate.stage_bytes.size() != identity_.stages.size()) {
        error = "parent parking: bad_request: the candidate's stage count is not the store's";
        return false;
    }
    JsonOut j;
    j.object_begin();
    j.key("id");
    j.u64(candidate.id);
    j.key("generation");
    j.u64(candidate.generation);
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpAcquire, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        close_all(fds);
        degrade("parent parking: protocol: the acquire answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    uint64_t lease_id = 0, id = 0, generation = 0;
    ParentCandidate parsed;
    if (!r.get_u64("lease_id", lease_id) || !r.get_u64("id", id) || !r.get_u64("generation", generation) ||
        !get_candidate(r, parsed, why)) {
        close_all(fds);
        degrade("parent parking: protocol: the acquire answer is missing a field");
        error = last_error_;
        return false;
    }
    if (lease_id == 0 || id != candidate.id || generation != candidate.generation ||
        parsed.stage_bytes != candidate.stage_bytes || !valid_owner(parsed.owner)) {
        close_all(fds);
        degrade("parent parking: protocol: the acquire answer is not for the asked entry");
        error = last_error_;
        return false;
    }
    if (fds.size() != parsed.stage_bytes.size()) {
        const size_t got = fds.size();
        close_all(fds);
        degrade("parent parking: protocol: the store sent " + std::to_string(got) + " descriptors for " +
                std::to_string(parsed.stage_bytes.size()) + " stages");
        error = last_error_;
        return false;
    }
#ifndef _WIN32
    for (size_t k = 0; k < fds.size(); ++k) {
        struct stat st {};
        if (::fstat(fds[k], &st) != 0 || !S_ISREG(st.st_mode) || (uint64_t) st.st_size != parsed.stage_bytes[k] ||
            st.st_uid != ::getuid()) {
            close_all(fds);
            degrade("parent parking: protocol: the store's sealed descriptor " + std::to_string(k) +
                    " is not a regular file of the asked size");
            error = last_error_;
            return false;
        }
    }
#endif
    out.client_ = this;
    out.lease_id_ = lease_id;
    out.candidate_ = std::move(parsed);
    out.fds_ = std::move(fds);
    remember_lease(lease_id, this, identity_);
    return true;
}

bool ConversationParentClient::release(uint64_t lease_id, const char* result, std::string& error) {
    if (lease_id == 0) {
        error = "parent parking: bad_request: a zero lease id";
        return false;
    }
    if (!valid_result(result)) {
        error = "parent parking: bad_request: an unknown release result";
        return false;
    }
    JsonOut j;
    j.object_begin();
    j.key("lease_id");
    j.u64(lease_id);
    j.key("result");
    j.string(result);
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpRelease, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        degrade("parent parking: protocol: the release answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    bool gone = false;
    if (!r.get_bool("gone", gone)) {
        degrade("parent parking: protocol: the release answer has no gone");
        error = last_error_;
        return false;
    }
    (void) gone;   // idempotent either way
    return true;
}

bool ConversationParentClient::clear(const char* reason, size_t& cleared, std::string& error) {
    cleared = 0;
    if (!valid_reason(reason)) {
        error = "parent parking: bad_request: an unknown clear reason";
        return false;
    }
    JsonOut j;
    j.object_begin();
    j.key("reason");
    j.string(reason);
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpClear, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        degrade("parent parking: protocol: the clear answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    uint64_t count = 0, epoch = 0;
    if (!r.get_u64("cleared", count) || !r.get_u64("epoch", epoch) ||
        count > (uint64_t) std::numeric_limits<size_t>::max()) {
        degrade("parent parking: protocol: the clear answer is missing a field");
        error = last_error_;
        return false;
    }
    epoch_ = epoch;
    cleared = (size_t) count;
    return true;
}

bool ConversationParentClient::stats(ParentStats& out, std::string& error) {
    out = {};
    JsonOut j;
    j.object_begin();
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpStats, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        degrade("parent parking: protocol: the stats answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    ParentStats s;
    auto need_u64 = [&](const char* key, uint64_t& value) {
        if (!r.get_u64(key, value)) {
            why = std::string("the stats have no ") + key;
            return false;
        }
        return true;
    };
    auto need_size = [&](const char* key, size_t& value) {
        uint64_t n = 0;
        if (!r.get_u64(key, n) || n > (uint64_t) std::numeric_limits<size_t>::max()) {
            why = std::string("the stats have no ") + key;
            return false;
        }
        value = (size_t) n;
        return true;
    };
    if (!r.get_bool("enabled", s.enabled) || !need_u64("capacity_bytes", s.capacity_bytes) ||
        !need_u64("charged_bytes", s.charged_bytes) || !need_size("entries", s.entries) ||
        !need_u64("entry_bytes", s.entry_bytes) || !need_size("pending", s.pending) ||
        !need_u64("pending_bytes", s.pending_bytes) || !need_size("leases", s.leases) ||
        !need_u64("restore_reserved_bytes", s.restore_reserved_bytes) ||
        !need_u64("retired_lease_bytes", s.retired_lease_bytes) || !need_size("published", s.published) ||
        !need_u64("published_bytes", s.published_bytes) || !need_size("deduplicated", s.deduplicated) ||
        !need_size("reads", s.reads) || !need_size("peer_reads", s.peer_reads) ||
        !need_size("evictions", s.evictions) || !need_size("eviction_skipped_leased", s.eviction_skipped_leased) ||
        !need_size("superseded", s.superseded) || !need_size("clears", s.clears) || !need_size("damaged", s.damaged) ||
        !need_size("capacity_refusals", s.capacity_refusals) ||
        !need_size("budget_violations", s.budget_violations) || !need_u64("store_epoch", s.store_epoch) ||
        !r.get_i64("longest_tokens", s.longest_tokens)) {
        degrade("parent parking: protocol: " + why);
        error = last_error_;
        return false;
    }
    epoch_ = s.store_epoch;
    out = s;
    return true;
}

bool ConversationParentClient::ping(std::string& error) {
    JsonOut j;
    j.object_begin();
    j.object_end();
    std::string response;
    std::vector<int> fds;
    if (!request(kOpPing, j.text, response, fds, error)) return false;
    Json r;
    std::string why;
    if (!parse_json(response, r, why)) {
        degrade("parent parking: protocol: the ping answer is not JSON: " + why);
        error = last_error_;
        return false;
    }
    uint64_t epoch = 0;
    if (!r.get_u64("epoch", epoch)) {
        degrade("parent parking: protocol: the ping answer has no epoch");
        error = last_error_;
        return false;
    }
    epoch_ = epoch;
    return true;
}

} // namespace strata::core
