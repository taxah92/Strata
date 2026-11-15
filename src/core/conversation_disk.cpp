// The conversation cache's disk tier (include/strata/core/conversation_disk.hpp, docs/CONVERSATION_CACHE_DISK.md).
// Stage files are session files (conversation_file.cpp); this file adds the per-conversation index file, the folder
// scan at start-up, prefix matching by token hashes, least-recently-used retention, the two folder modes (a writer
// owns its folder, a peer reads another engine's) and the cross-process leases (POSIX flock on <id>.lock).
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <string>
#include <system_error>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

static_assert(std::endian::native == std::endian::little, "conversation disk index files are little-endian only");

namespace strata::core {
namespace {
namespace fs = std::filesystem;

constexpr char kIndexMagic[8] = {'S', 'T', 'R', 'C', 'D', 'X', '\x01', '\0'};
constexpr uint32_t kIndexVersion = 2;      // v2: the owner's name and the content generation (a peer needs both)
constexpr uint32_t kIndexVersionMin = 1;   // a v1 index (the first #5835 build) still reads: no owner, generation 0
constexpr size_t kMaxOwner = 64;
constexpr uint64_t kIndexSeed = 0x5354524344583031ull;   // "STRCDX01"
constexpr uint64_t kIdsSeed = 0x5354524944533031ull;     // "STRIDS01"
constexpr uint64_t kLayoutSeed = 0x53545243444c3031ull;  // "STRCDL01"
constexpr size_t kIndexMaxBytes = 64u << 20;             // far above any real index (a few KiB)
constexpr uint64_t kMaxPoints = 4096, kMaxImages = 1u << 20, kMaxStages = 64;
constexpr char kWriterLock[] = "writer.lock";

int64_t monotonic_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t random_id() {
    static std::mt19937_64 rng{[] {
        std::random_device rd;
        return ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    }()};
    uint64_t v = 0;
    while (v == 0 || v == UINT64_MAX) v = rng();   // 0 is "none", UINT64_MAX the caller's "failed" mark
    return v;
}

std::string hex16(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}

bool parse_hex16(const std::string& s, uint64_t& v) {
    if (s.size() != 16) return false;
    v = 0;
    for (char c : s) {
        int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (d < 0) return false;
        v = (v << 4) | (uint64_t) d;
    }
    return v != 0;
}

// A held POSIX flock: the writer's lifetime lock on the folder (exclusive, taken at open) and a snapshot's lease
// (shared for a read, exclusive non-blocking for a removal).  On Windows there is no inter-process lease: the writer
// keeps its folder private by configuration only, and a peer is refused before it gets here.
class HeldLock {
public:
    HeldLock() = default;
    HeldLock(const HeldLock&) = delete;
    HeldLock& operator=(const HeldLock&) = delete;
    ~HeldLock() { release(); }

    /// Takes the lock.  `create`: the writer's own files (a lease file is created when missing); a peer never creates.
    /// A busy non-blocking lock returns false with `busy` set; a missing file (a peer's snapshot gone) returns false
    /// with `busy` clear.  False always: `error` says why.
    bool take(const fs::path& path, bool exclusive, bool create, bool& busy, std::string& error) {
        release();
        busy = false;
#ifndef _WIN32
        const int flags = (create ? O_RDWR | O_CREAT : O_RDONLY) | O_CLOEXEC;
        const int fd = ::open(path.c_str(), flags, 0600);
        if (fd < 0) {
            error = std::strerror(errno);
            return false;
        }
        if (::flock(fd, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) != 0) {
            const int e = errno;
            ::close(fd);
            busy = e == EWOULDBLOCK || e == EAGAIN;
            error = std::strerror(e);
            return false;
        }
        fd_ = fd;
        return true;
#else
        (void) path; (void) exclusive; (void) create;
        error = "no inter-process leases on this platform";
        return false;
#endif
    }
    /// Keeps the lock without the destructor: the writer's lifetime lock lives in ConversationDiskTier.
    int detach() {
        const int fd = fd_;
        fd_ = -1;
        return fd;
    }
    void release() {
#ifndef _WIN32
        if (fd_ >= 0) ::close(fd_);
#endif
        fd_ = -1;
    }
    bool held() const { return fd_ >= 0; }

private:
    int fd_ = -1;
};

// ---- the index file: little-endian, fixed-width
//   v2: magic[8] version:u32 reserved:u32 model:u64 config:u64 layout:u64 owner_len:u32 reserved:u32 owner[owner_len]
//       id:u64 generation:u64 used:u64 cvec:u64
//   v1: magic[8] version:u32 reserved:u32 model:u64 config:u64 layout:u64 id:u64 used:u64 cvec:u64
//   both: points:u64 { tokens:i64 hash:u64 live:u64 imgs:u64 { start:i64 hash:u64 }* }*
//         files:u64 { bytes:u64 }*
//         hash:u64 (session_hash64 of every byte before it, seed kIndexSeed)
struct Writer {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) { const uint8_t* q = (const uint8_t*) p; b.insert(b.end(), q, q + n); }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
};

struct Reader {
    const std::vector<uint8_t>& b;
    size_t at = 0;
    bool u32(uint32_t& v) {
        if (b.size() - at < 4) return false;
        std::memcpy(&v, b.data() + at, 4);
        at += 4;
        return true;
    }
    bool u64(uint64_t& v) {
        if (b.size() - at < 8) return false;
        std::memcpy(&v, b.data() + at, 8);
        at += 8;
        return true;
    }
    bool i64(int64_t& v) { uint64_t u = 0; if (!u64(u)) return false; v = (int64_t) u; return true; }
};

bool read_small(const fs::path& p, std::vector<uint8_t>& out) {
    std::error_code ec;
    const auto size = fs::file_size(p, ec);
    if (ec || size > kIndexMaxBytes) return false;
    std::FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) return false;
    out.resize((size_t) size);
    const bool ok = std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

// temporary name beside `path`, flushed, renamed over it, folder flushed (POSIX)
bool write_small(const fs::path& path, const std::vector<uint8_t>& data, bool durable, std::string& error) {
    const fs::path tmp = path.parent_path() / ("." + path.filename().string() + "." + hex16(random_id()) + ".tmp");
    std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
    if (!f) { error = "conversation disk: cannot create " + tmp.string() + ": " + std::strerror(errno); return false; }
    bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size() && std::fflush(f) == 0;
#ifndef _WIN32
    if (ok && durable) ok = ::fsync(::fileno(f)) == 0;
#endif
    const int saved = errno;
    ok = std::fclose(f) == 0 && ok;
    std::error_code ec;
    if (!ok) {
        error = "conversation disk: cannot write " + tmp.string() + ": " + std::strerror(saved);
        fs::remove(tmp, ec);
        return false;
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        error = "conversation disk: cannot rename " + tmp.string() + ": " + ec.message();
        fs::remove(tmp, ec);
        return false;
    }
#ifndef _WIN32
    if (durable) {
        const int d = ::open(path.parent_path().string().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (d >= 0) { (void) ::fsync(d); ::close(d); }   // EINVAL on some filesystems: not a failure (as for SAVE)
    }
#endif
    return true;
}

// The pictures below `n` tokens, in prompt order (what conversation_prefix compares a checkpoint's against).
std::vector<ConversationImageKey> images_below(const std::vector<ConversationImageKey>& images, int64_t n) {
    std::vector<ConversationImageKey> out;
    for (const auto& im : images) if (im.start < n) out.push_back(im);
    return out;
}

ConversationDiskPoint point_of(const ConversationCheckpoint& c, bool live) {
    ConversationDiskPoint p;
    p.tokens = (int64_t) c.ids.size();
    p.hash = conversation_disk_ids_hash(c.ids.data(), c.ids.size());
    p.imgs = c.imgs;
    p.live = live;
    return p;
}

const ConversationDiskPoint* deepest_checkpoint(const std::vector<ConversationDiskPoint>& points) {
    const ConversationDiskPoint* d = nullptr;
    for (const auto& p : points)
        if (!p.live && (!d || p.tokens > d->tokens)) d = &p;
    return d;
}

bool same_point(const ConversationDiskPoint& a, const ConversationDiskPoint& b) {
    return a.tokens == b.tokens && a.hash == b.hash && a.imgs == b.imgs;
}

const ConversationDiskTier::Entry* find_in(const std::vector<ConversationDiskTier::Entry>& v, uint64_t id) {
    for (const auto& e : v) if (e.id == id) return &e;
    return nullptr;
}

bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}
} // namespace

uint64_t ConversationDiskLayout::key() const {
    SessionHasher h(kLayoutSeed);
    const uint64_t n = stages.size();
    h.update(&n, 8);
    for (const auto& s : stages) { h.update(&s.first, 8); h.update(&s.second, 8); }
    return h.digest();
}

uint64_t conversation_disk_ids_hash(const int32_t* ids, size_t n) {
    return session_hash64(ids, n * sizeof(int32_t), kIdsSeed);
}

std::vector<ConversationDiskPoint> conversation_disk_points(const SavedConversation& image) {
    std::vector<ConversationDiskPoint> out;
    out.reserve(1 + image.checkpoints.size());
    out.push_back(point_of(image.live, true));
    for (const auto& c : image.checkpoints) out.push_back(point_of(c, false));
    return out;
}

ConversationDiskTier::~ConversationDiskTier() {
#ifndef _WIN32
    if (lock_fd_ >= 0) ::close(lock_fd_);   // the writer's lifetime lock: the folder is free again
    lock_fd_ = -1;
#endif
}

std::string ConversationDiskTier::path_of(uint64_t id, const char* suffix) const {
    return (fs::path(folder_) / (hex16(id) + suffix)).string();
}

std::string ConversationDiskTier::stage_path(uint64_t id, size_t stage) const {
    return path_of(id, (".s" + std::to_string(stage) + ".strses").c_str());
}

std::string ConversationDiskTier::lock_path(uint64_t id) const { return path_of(id, ".lock"); }

const ConversationDiskTier::Entry* ConversationDiskTier::find(uint64_t id) const {
    for (const auto& e : entries_) if (e.id == id) return &e;
    return nullptr;
}

uint64_t ConversationDiskTier::generation(uint64_t id) const {
    const Entry* e = find(id);
    return e != nullptr ? e->generation : 0;
}

ConversationDiskTier::Entry* ConversationDiskTier::find_mut(uint64_t id) {
    for (auto& e : entries_) if (e.id == id) return &e;
    return nullptr;
}

void ConversationDiskTier::note_used(Entry& e) {
    e.used = ++clock_;
    // keep entries_ least recently used first (a peer's index stamps come from its owner; the local bump wins)
    std::stable_sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
}

bool ConversationDiskTier::write_index(const Entry& e, std::string& error) const {
    Writer w;
    w.raw(kIndexMagic, 8);
    w.u32(kIndexVersion);
    w.u32(0);
    w.u64(identity_.model); w.u64(identity_.config); w.u64(layout_.key());
    const size_t olen = std::min(owner_.size(), kMaxOwner);
    w.u32((uint32_t) olen);
    w.u32(0);
    if (olen != 0) w.raw(owner_.data(), olen);
    w.u64(e.id); w.u64(e.generation); w.u64(e.used); w.u64(e.cvec ? 1 : 0);
    w.u64(e.points.size());
    for (const auto& p : e.points) {
        w.i64(p.tokens); w.u64(p.hash); w.u64(p.live ? 1 : 0);
        w.u64(p.imgs.size());
        for (const auto& im : p.imgs) { w.i64(im.start); w.u64(im.hash); }
    }
    w.u64(e.files.size());
    for (uint64_t f : e.files) w.u64(f);
    w.u64(session_hash64(w.b.data(), w.b.size(), kIndexSeed));
    return write_small(path_of(e.id, ".idx"), w.b, options_.durable, error);
}

void ConversationDiskTier::remove_files(uint64_t id, size_t stages) const {
    std::error_code ec;
    fs::remove(path_of(id, ".idx"), ec);   // the index first: a crash after it leaves orphans, never a broken entry
    for (size_t k = 0; k < stages; ++k) fs::remove(stage_path(id, k), ec);
    fs::remove(lock_path(id), ec);         // (the caller holds the exclusive lease, so no reader is inside)
}

bool ConversationDiskTier::open(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
                                const ConversationDiskLayout& layout, std::string& error, OpenReport* report) {
    open_ = false;
    entries_.clear();
    bytes_ = clock_ = 0;
    try {
        open_ = open_impl(options, identity, layout, /*peer=*/false, error, report);
    } catch (const std::exception& e) {   // a filesystem_error from a directory walk, a bad_alloc
        error = std::string("conversation disk: ") + e.what();
        entries_.clear();
        bytes_ = 0;
        open_ = false;
    }
    if (!open_) mode_ = Mode::writer;
    return open_;
}

bool ConversationDiskTier::open_peer(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
                                     const ConversationDiskLayout& layout, std::string& error, OpenReport* report) {
    open_ = false;
    entries_.clear();
    bytes_ = clock_ = 0;
    try {
        open_ = open_impl(options, identity, layout, /*peer=*/true, error, report);
    } catch (const std::exception& e) {
        error = std::string("conversation disk (peer): ") + e.what();
        entries_.clear();
        bytes_ = 0;
        open_ = false;
    }
    if (!open_) mode_ = Mode::writer;
    return open_;
}

bool ConversationDiskTier::open_impl(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
                                     const ConversationDiskLayout& layout, bool peer, std::string& error,
                                     OpenReport* report) {
    OpenReport rep;
    if (options.dir.empty()) { error = "conversation disk: no folder"; return false; }
    if (layout.stages.empty() || layout.stages.size() > kMaxStages) { error = "conversation disk: invalid stage layout"; return false; }
    if (!peer && options.budget_bytes == 0) { error = "conversation disk: a zero byte budget"; return false; }
#ifdef _WIN32
    if (peer) {   // no inter-process leases: a peer folder could be changed under the read
        error = "conversation disk: the peer mode is POSIX only";
        return false;
    }
#endif
    options_ = options;
    identity_ = identity;
    layout_ = layout;
    mode_ = peer ? Mode::peer : Mode::writer;
    writer_dir_ = options.dir;
    owner_ = !options.owner.empty() ? options.owner : fs::path(options.dir).filename().string();
    const std::string leaf = hex16(identity.model) + "-" + hex16(identity.config) + "-" + hex16(layout.key());
    folder_ = (fs::path(options.dir) / leaf).string();
    std::error_code ec;
    if (peer) {
        if (!fs::is_directory(options.dir, ec)) {
            error = "conversation disk: the peer folder is not there: " + options.dir;
            return false;
        }
        open_ = true;
        leaf_seen_ = fs::is_directory(folder_, ec);
        if (leaf_seen_) {
            const auto t = fs::last_write_time(folder_, ec);
            if (!ec) leaf_mtime_ = (int64_t) t.time_since_epoch().count();
        }
        last_scan_ms_ = monotonic_ms();
        RefreshReport rr;
        scan(nullptr, &rr, /*peer=*/true, /*first=*/true);
        if (report) *report = rep;
        return true;
    }
    fs::create_directories(writer_dir_, ec);
    if (ec || !fs::is_directory(writer_dir_, ec)) {
        error = "conversation disk: cannot create " + writer_dir_ + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    fs::permissions(writer_dir_, fs::perms::owner_all, fs::perm_options::replace, ec);   // 0700: the user's own state
    {
        // one writer per folder for its whole lifetime: a second engine is refused instead of cleaning up under the first
        HeldLock lifetime;
        bool busy = false;
        std::string le;
        if (!lifetime.take(fs::path(writer_dir_) / kWriterLock, /*exclusive=*/true, /*create=*/true, busy, le)) {
            error = busy ? ("conversation disk: another engine owns " + writer_dir_ + " (writer.lock is held)")
                         : ("conversation disk: cannot take " + writer_dir_ + "/writer.lock: " + le);
            return false;
        }
        lock_fd_ = lifetime.detach();
    }
#endif
    fs::create_directories(folder_, ec);
    if (ec || !fs::is_directory(folder_, ec)) {
        error = "conversation disk: cannot create " + folder_ + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    fs::permissions(folder_, fs::perms::owner_all, fs::perm_options::replace, ec);   // 0700: the user's own state
#endif
    // the other identities' and layouts' folders: reported, never touched (another engine may use them)
    for (const auto& d : fs::directory_iterator(options.dir, ec)) {
        std::error_code e2;
        if (!d.is_directory(e2) || d.path().filename().string() == leaf) continue;
        ++rep.other_folders;
        for (const auto& f : fs::recursive_directory_iterator(d.path(), fs::directory_options::skip_permission_denied, e2))
            if (f.is_regular_file(e2)) {
                const auto size = f.file_size(e2);
                if (!e2) rep.other_bytes += size;
            }
    }
    scan(&rep, nullptr, /*peer=*/false, /*first=*/true);
    open_ = true;
    std::string room_error;
    if (!make_room(0, room_error)) {   // (cannot fail with nothing incoming; a smaller budget evicts)
        error = room_error;
        open_ = false;
        return false;
    }
    if (report) *report = rep;
    return true;
}

bool ConversationDiskTier::scan(OpenReport* open, RefreshReport* refresh, bool peer, bool first) {
    std::error_code ec;
    const bool present = fs::is_directory(folder_, ec);
    if (!present) {
        entries_.clear();
        bytes_ = 0;
        return true;
    }
    std::vector<Entry> found;
    std::vector<std::string> stray;
    // pass 1: the index files
    for (const auto& d : fs::directory_iterator(folder_, ec)) {
        std::error_code e2;
        if (!d.is_regular_file(e2)) continue;
        const std::string name = d.path().filename().string();
        if (!name.empty() && name[0] == '.') {
            if (!peer && ends_with(name, ".tmp")) stray.push_back(d.path().string());   // an unfinished write's temporary file
            continue;
        }
        if (name.size() != 20 || name.compare(16, 4, ".idx") != 0) continue;
        uint64_t id = 0;
        if (!parse_hex16(name.substr(0, 16), id)) {
            if (!peer) stray.push_back(d.path().string());
            continue;
        }
        Entry e;
        if (!parse_index(d.path().string(), id, e)) {
            if (peer) {   // not this runtime's, or damaged mid-write: ignored, never removed, tried again on the next refresh
                if (refresh) ++refresh->damaged;
                continue;
            }
            // the folder is this identity's and layout's alone: a damaged index's files go
            remove_files(id, layout_.stages.size());
            if (open) ++open->dropped;
            continue;
        }
        if (const Entry* old = find_in(entries_, id)) {   // this run's marks survive a re-scan
            if (old->failed) e.failed = true;
            if (peer) e.used = std::max(e.used, old->used);   // the local LRU stamp of a peer is its own
        }
        clock_ = std::max(clock_, e.used);
        found.push_back(std::move(e));
    }
    // pass 2: stage and lease files no index names (a write that did not finish).  A peer removes nothing at all.
    if (!peer) {
        for (const auto& d : fs::directory_iterator(folder_, ec)) {
            std::error_code e2;
            if (!d.is_regular_file(e2)) continue;
            const std::string name = d.path().filename().string();
            uint64_t id = 0;
            const bool stage = name.size() >= 16 + 10 && name.compare(16, 2, ".s") == 0 && parse_hex16(name.substr(0, 16), id);
            const bool lease = name.size() == 21 && name.compare(16, 5, ".lock") == 0 && parse_hex16(name.substr(0, 16), id);
            if ((stage || lease) && find_in(found, id) == nullptr) stray.push_back(d.path().string());
        }
        for (const auto& p : stray) { std::error_code e4; if (fs::remove(p, e4) && open) ++open->orphans; }
    }
    if (peer) {
        if (refresh) {
            for (const auto& e : found) if (find_in(entries_, e.id) == nullptr) ++refresh->added;
            for (const auto& e : entries_) if (find_in(found, e.id) == nullptr) ++refresh->gone;
        }
    } else if (first && open) {
        open->kept = found.size();
    }
    bytes_ = 0;
    for (const auto& e : found) bytes_ += e.bytes;
    entries_ = std::move(found);
    std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
    return true;
}

bool ConversationDiskTier::refresh(std::string& error, bool force, RefreshReport* report) {
    if (!open_) return false;
    if (mode_ != Mode::peer) { error.clear(); return true; }
    const int64_t now = monotonic_ms();
    if (!force && options_.refresh_ms > 0 && last_scan_ms_ != 0 && now - last_scan_ms_ < options_.refresh_ms) return true;
    std::error_code ec;
    const bool present = fs::is_directory(folder_, ec);
    int64_t mt = 0;
    if (present) {
        const auto t = fs::last_write_time(folder_, ec);
        if (!ec) mt = (int64_t) t.time_since_epoch().count();
    }
    if (!force && present == leaf_seen_ && mt == leaf_mtime_) {
        last_scan_ms_ = now;   // nothing changed since the last scan
        return true;
    }
    last_scan_ms_ = now;
    RefreshReport rr;
    if (present) {
        ++refreshes_;
        scan(nullptr, &rr, /*peer=*/true, /*first=*/false);
    } else if (leaf_seen_) {   // the writer's folder went away (it restarted with another layout, or was cleaned)
        entries_.clear();
        bytes_ = 0;
    }
    leaf_seen_ = present;
    leaf_mtime_ = mt;
    if (report) *report = rr;
    error.clear();
    return true;
}

bool ConversationDiskTier::parse_index(const std::string& path, uint64_t id, Entry& e) const {
    std::vector<uint8_t> b;
    if (!read_small(path, b) || b.size() < 16 + 8 * 8) return false;
    uint64_t stored = 0;
    std::memcpy(&stored, b.data() + b.size() - 8, 8);
    if (std::memcmp(b.data(), kIndexMagic, 8) != 0 || stored != session_hash64(b.data(), b.size() - 8, kIndexSeed))
        return false;
    uint32_t version = 0;
    std::memcpy(&version, b.data() + 8, 4);
    if (version < kIndexVersionMin || version > kIndexVersion) return false;
    Reader r{b, 16};
    uint64_t model = 0, config = 0, lkey = 0, rid = 0, used = 0, cvec = 0, np = 0, nf = 0, generation = 0;
    std::string owner;
    if (!r.u64(model) || !r.u64(config) || !r.u64(lkey)) return false;
    if (version >= 2) {
        uint32_t olen = 0, reserved = 0;
        if (!r.u32(olen) || !r.u32(reserved) || olen > kMaxOwner || olen > b.size() - r.at) return false;
        owner.assign((const char*) b.data() + r.at, olen);
        r.at += olen;
        if (!r.u64(rid) || !r.u64(generation)) return false;
    } else if (!r.u64(rid)) {
        return false;
    }
    if (!r.u64(used) || !r.u64(cvec) || model != identity_.model || config != identity_.config ||
        lkey != layout_.key() || rid != id || cvec > 1 || !r.u64(np) || np < 1 || np > kMaxPoints)
        return false;
    e.id = id;
    e.generation = generation;
    e.used = used;
    e.cvec = cvec == 1;
    for (uint64_t i = 0; i < np; ++i) {
        ConversationDiskPoint p;
        uint64_t live = 0, ni = 0;
        if (!r.i64(p.tokens) || !r.u64(p.hash) || !r.u64(live) || live > 1 || !r.u64(ni) || ni > kMaxImages ||
            p.tokens <= 0 || ni > (b.size() - r.at) / 16)
            return false;
        p.live = live == 1;
        p.imgs.resize((size_t) ni);
        for (auto& k : p.imgs) if (!r.i64(k.start) || !r.u64(k.hash)) return false;
        e.points.push_back(std::move(p));
    }
    if (!r.u64(nf) || nf != layout_.stages.size()) return false;
    for (uint64_t k = 0; k < nf; ++k) {
        uint64_t size = 0;
        std::error_code ec;
        if (!r.u64(size)) return false;
        const auto have = fs::file_size(stage_path(id, (size_t) k), ec);
        if (ec || have != size) return false;   // a stage file missing or cut short
        e.files.push_back(size);
        e.bytes += size;
    }
    (void) owner;   // (informational: the leaf folder already names the identity, and the lease names the snapshot)
    return r.at + 8 == b.size() && e.points.front().live;
}

bool ConversationDiskTier::make_room(uint64_t incoming, std::string& error, uint64_t keep) {
    if (incoming > options_.budget_bytes) {
        error = "conversation disk: the conversation (" + std::to_string(incoming >> 20) +
                " MiB) is larger than the disk budget";
        return false;
    }
    auto over = [&] {
        return bytes_ > options_.budget_bytes - incoming ||
               entries_.size() + (incoming ? 1 : 0) > options_.max_entries;
    };
    if (!over()) { overshoot_ = false; return true; }
    // the least recently used goes first; the conversation being read (load) and `keep` (just written) stay.  A
    // victim another process is reading (its shared lease is held) is skipped: the next one goes, and when every
    // victim is taken the budget runs over for now (the next write after a reader leaves picks it up).
    std::vector<uint64_t> order(entries_.size());
    std::iota(order.begin(), order.end(), (uint64_t) 0);
    std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) { return entries_[a].used < entries_[b].used; });
    std::vector<uint64_t> victims;
    victims.reserve(order.size());
    for (uint64_t i : order) {
        const Entry& e = entries_[i];
        if (e.id != loading_ && e.id != keep) victims.push_back(e.id);
    }
    for (uint64_t id : victims) {
        if (!over()) break;
        if (drop(id)) ++evictions_;
    }
    overshoot_ = over();
    return true;
}

ConversationDiskTier::Match ConversationDiskTier::best_ids(const std::vector<int32_t>& prompt,
                                                           const std::vector<ConversationImageKey>& images,
                                                           bool cvec) const {
    Match best;
    if (!open_ || prompt.size() < 2) return best;
    // the lengths worth hashing: deeper than nothing, short of the prompt's last token, pictures as the prompt's
    std::vector<int64_t> lens;
    for (const auto& e : entries_) {
        if (e.cvec != cvec || e.failed) continue;
        for (const auto& p : e.points)
            if (p.tokens > 0 && (size_t) p.tokens < prompt.size() && images_below(images, p.tokens) == p.imgs)
                lens.push_back(p.tokens);
    }
    if (lens.empty()) return best;
    std::sort(lens.begin(), lens.end());
    lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
    // one streaming pass over the prompt: the digest at each length equals the one-shot hash of that prefix
    std::vector<std::pair<int64_t, uint64_t>> at;
    at.reserve(lens.size());
    SessionHasher h(kIdsSeed);
    size_t done = 0;
    for (int64_t L : lens) {
        h.update(prompt.data() + done, ((size_t) L - done) * sizeof(int32_t));
        done = (size_t) L;
        at.emplace_back(L, h.digest());
    }
    auto hash_at = [&](int64_t L, uint64_t& out) {
        const auto it = std::lower_bound(at.begin(), at.end(), std::make_pair(L, uint64_t(0)),
                                         [](const auto& a, const auto& b) { return a.first < b.first; });
        if (it == at.end() || it->first != L) return false;
        out = it->second;
        return true;
    };
    // most recently used last: >= lets a later (more recent) entry win a tie
    for (const auto& e : entries_) {
        if (e.cvec != cvec || e.failed) continue;
        for (const auto& p : e.points) {
            uint64_t hp = 0;
            if (p.tokens < best.tokens || !hash_at(p.tokens, hp) || hp != p.hash) continue;
            if (images_below(images, p.tokens) != p.imgs) continue;
            if (p.tokens > best.tokens || (p.tokens == best.tokens && (p.live || !best.live)))
                best = {e.id, p.tokens, p.live, e.generation};
        }
    }
    return best;
}

uint64_t ConversationDiskTier::holding(const SavedConversation& image) const {
    if (!open_ || mode_ == Mode::peer) return 0;
    const auto points = conversation_disk_points(image);
    for (const auto& e : entries_)
        if (e.cvec == image.cvec && e.points.size() == points.size() &&
            std::equal(points.begin(), points.end(), e.points.begin(), same_point))
            return e.id;
    return 0;
}

size_t ConversationDiskTier::drop_superseded(const SavedConversation& image, uint64_t keep) {
    if (!open_ || mode_ == Mode::peer) return 0;
    const auto points = conversation_disk_points(image);
    std::vector<uint64_t> gone;
    for (const auto& e : entries_) {
        if (e.cvec != image.cvec || e.id == loading_ || e.id == keep) continue;
        const ConversationDiskPoint* deepest = deepest_checkpoint(e.points);
        if (!deepest) continue;
        for (const auto& p : points)
            if (same_point(p, *deepest)) { gone.push_back(e.id); break; }
    }
    size_t n = 0;
    for (uint64_t id : gone)
        if (drop(id)) ++n;   // (a copy another process is reading stays: the match takes the deepest one anyway)
    return n;
}

void ConversationDiskTier::mark_failed(uint64_t id) {
    if (Entry* e = find_mut(id)) e->failed = true;
}

bool ConversationDiskTier::drop(uint64_t id) {
    if (mode_ == Mode::peer) return false;
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].id != id) continue;
        HeldLock lease;
        bool busy = false;
        std::string le;
        if (!lease.take(lock_path(id), /*exclusive=*/true, /*create=*/false, busy, le)) {
            if (busy || fs::exists(lock_path(id))) {   // a reader is inside (or the lease cannot be taken): skip it
                ++lease_conflicts_;
                ++eviction_skipped_;
                return false;
            }
            // no lease file (an entry written by a build before the leases): nothing can hold it
        }
        remove_files(id, entries_[i].files.size());
        bytes_ -= std::min(bytes_, entries_[i].bytes);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
        return true;
    }
    return false;
}

size_t ConversationDiskTier::clear() {
    if (mode_ == Mode::peer) return 0;
    const size_t n = entries_.size();
    // the index files go first: nothing is matched from this moment on, whatever the leases allow below
    for (const auto& e : entries_) {
        std::error_code ec;
        fs::remove(path_of(e.id, ".idx"), ec);
    }
    for (const auto& e : entries_) {
        HeldLock lease;
        bool busy = false;
        std::string le;
        if (!lease.take(lock_path(e.id), /*exclusive=*/true, /*create=*/false, busy, le) &&
            (busy || fs::exists(lock_path(e.id)))) {
            // a reader is inside: its stage files stay (leftovers the next writer open removes), its lease is not broken
            ++lease_conflicts_;
            ++eviction_skipped_;
            continue;
        }
        for (size_t k = 0; k < e.files.size(); ++k) {
            std::error_code ec;
            fs::remove(stage_path(e.id, k), ec);
        }
        {
            std::error_code ec;
            fs::remove(lock_path(e.id), ec);
        }
    }
    entries_.clear();
    bytes_ = 0;
    return n;
}

bool ConversationDiskTier::spill(const SavedConversation& image, uint64_t& id, std::string& error,
                                 SessionStatus* status, const SessionProgress& progress, const SessionPhase& phase) {
    SessionStatus st;
    auto fail = [&](const std::string& why, SessionError kind = SessionError::invalid) {
        error = why;
        st.error = kind;
        if (status) *status = st;
        return false;
    };
    id = 0;
    if (!open_) return fail("conversation disk: off");
    if (mode_ == Mode::peer) return fail("conversation disk: a peer folder is read-only");
    const size_t n_files = 1 + image.stage_images.size();
    if (n_files != layout_.stages.size()) return fail("conversation disk: the image has another stage count");
    for (size_t k = 0; k < n_files; ++k) {
        const SavedConversation& part = k == 0 ? image : image.stage_images[k - 1];
        if (part.layer_lo != layout_.stages[k].first || part.layer_hi != layout_.stages[k].second)
            return fail("conversation disk: the image has another layer range");
        if (!part.stage_images.empty() && k != 0) return fail("conversation disk: nested stage images");
    }
    if (image.live.ids.empty()) return fail("conversation disk: an empty conversation");
    if (options_.max_snapshot_bytes != 0 && (uint64_t) image.bytes() > options_.max_snapshot_bytes)
        return fail("conversation disk: the conversation (" + std::to_string((uint64_t) image.bytes() >> 20) +
                    " MiB) is larger than the configured maximum (" + std::to_string(options_.max_snapshot_bytes >> 20) +
                    " MiB)", SessionError::invalid);
    if (const uint64_t kept = holding(image)) {   // already there (written when idle, or read back unchanged)
        id = kept;
        if (Entry* e = find_mut(kept)) {
            e->used = ++clock_;                // the new LRU stamp (the generation stays: the bytes did not change)
            std::string ignored;
            (void) write_index(*e, ignored);
            std::stable_sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
        }
        if (status) *status = st;
        return true;
    }
    // nothing is deleted before the new copy is committed (a failed write keeps the older copies): the budget may
    // run over by this one conversation until make_room below
    if ((uint64_t) image.bytes() > options_.budget_bytes) {
        if (!make_room(image.bytes(), error)) { st.error = SessionError::storage; if (status) *status = st; return false; }
    }
    Entry e;
    e.id = random_id();
    while (find(e.id) != nullptr) e.id = random_id();
    e.generation = ++generation_;
    e.cvec = image.cvec;
    e.points = conversation_disk_points(image);
    e.used = ++clock_;
    SessionWriteOptions wo;
    wo.min_free_bytes = options_.min_free_bytes;
    wo.durable = options_.durable;
    wo.progress = progress;
    wo.phase = phase;
    for (size_t k = 0; k < n_files; ++k) {
        const SavedConversation& part = k == 0 ? image : image.stage_images[k - 1];
        size_t written = 0;
        // stage 0's stage_images are not part of format v1's payload: each stage is its own file
        if (!session_file_write(stage_path(e.id, k), part, identity_, written, error, wo, &st)) {
            remove_files(e.id, n_files);
            if (status) *status = st;
            return false;
        }
        e.files.push_back(written);
        e.bytes += written;
    }
    {   // the lease file exists before the index commits the set: a reader takes it before it reads any stage file
        const std::string lp = lock_path(e.id);
        std::FILE* f = std::fopen(lp.c_str(), "ab");
        if (f == nullptr) {
            const std::string why = "conversation disk: cannot create " + lp + ": " + std::strerror(errno);
            remove_files(e.id, n_files);
            return fail(why, SessionError::io);
        }
        std::fclose(f);
    }
    if (!write_index(e, error)) {
        remove_files(e.id, n_files);
        return fail(error, SessionError::io);
    }
    bytes_ += e.bytes;
    id = e.id;
    entries_.push_back(std::move(e));
    ++writes_;
    drop_superseded(image, id);
    std::string ignored;
    (void) make_room(0, ignored, id);
    if (status) *status = st;
    return true;
}

namespace {
// The stage images have to hold the conversation the index describes: the same resume points, the same layer ranges,
// and (for a split) the same tokens in every stage's checkpoints.
bool parts_match(const std::vector<SavedConversation>& parts, const std::vector<ConversationDiskPoint>& points,
                 bool cvec, const ConversationDiskLayout& layout) {
    if (parts.empty()) return false;
    const auto got = conversation_disk_points(parts[0]);
    if (parts[0].cvec != cvec || got.size() != points.size() ||
        !std::equal(got.begin(), got.end(), points.begin(), same_point))
        return false;
    for (size_t k = 0; k < parts.size(); ++k)
        if (parts[k].layer_lo != layout.stages[k].first || parts[k].layer_hi != layout.stages[k].second) return false;
    for (size_t k = 1; k < parts.size(); ++k) {
        if (parts[k].live.ids != parts[0].live.ids) return false;
        if (parts[k].checkpoints.size() != parts[0].checkpoints.size()) return false;
        for (size_t c = 0; c < parts[0].checkpoints.size(); ++c)
            if (parts[k].checkpoints[c].ids != parts[0].checkpoints[c].ids) return false;
    }
    return true;
}
} // namespace

bool ConversationDiskTier::load(uint64_t id, const std::vector<SessionReadLimits>& limits, SavedConversation& out,
                                std::string& error, SessionStatus* status, uint64_t expect_generation) {
    SessionStatus st;
    if (!open_) {
        error = "conversation disk: off";
        st.error = SessionError::invalid;
        if (status) *status = st;
        return false;
    }
    if (options_.max_snapshot_bytes != 0) {
        const Entry* e = find(id);
        if (e != nullptr && e->bytes > options_.max_snapshot_bytes) {
            error = "conversation disk: the conversation (" + std::to_string(e->bytes >> 20) +
                    " MiB) is larger than the configured maximum";
            st.error = SessionError::invalid;
            if (status) *status = st;
            return false;
        }
    }
    auto read_parts = [&](size_t n_files, std::vector<SavedConversation>& parts) -> bool {
        parts.clear();
        parts.resize(n_files);
        for (size_t k = 0; k < n_files; ++k) {
            size_t read = 0;
            const SessionReadLimits none;
            if (!session_file_read(stage_path(id, k), identity_, parts[k], read, error,
                                   k < limits.size() ? limits[k] : none, &st))
                return false;
        }
        return true;
    };
    if (mode_ == Mode::peer) {
        uint64_t want = expect_generation;
        for (int attempt = 0; attempt < 2; ++attempt) {
            Entry* e = find_mut(id);
            if (e == nullptr || e->failed) {
                error = "conversation disk: no such conversation on the peer";
                st.error = SessionError::invalid;
                if (status) *status = st;
                return false;
            }
            const uint64_t gen = want != 0 ? want : e->generation;
            const std::vector<uint64_t> files = e->files;
            // the RAM preflight runs BEFORE the lease is taken (the caller's gate evicts parked conversations, which
            // can take a moment; a reader's lease must not be held for that, and a refusal never takes it at all)
            for (size_t k = 0; k < files.size(); ++k) {
                const SessionReadLimits& lim = k < limits.size() ? limits[k] : SessionReadLimits{};
                if (!lim.admit) continue;
                std::string why;
                if (!lim.admit(session_read_peak_bytes(files[k]), why)) {
                    error = "conversation disk: " + why;
                    st.error = SessionError::memory;
                    if (status) *status = st;
                    return false;
                }
            }
            HeldLock lease;
            bool busy = false;
            std::string le;
            if (!lease.take(lock_path(id), /*exclusive=*/false, /*create=*/false, busy, le)) {
                // gone (the writer removed it), or busy with another reader: one forced refresh and retry, then give up
                if (attempt == 0) { std::string ignored; refresh(ignored, /*force=*/true); continue; }
                error = "conversation disk: the peer conversation is not readable (" + le + ")";
                st.error = SessionError::invalid;
                if (status) *status = st;
                return false;
            }
            Entry fresh;
            if (!parse_index(path_of(id, ".idx"), id, fresh)) {
                if (attempt == 0) { std::string ignored; refresh(ignored, /*force=*/true); continue; }
                error = "conversation disk: the peer conversation changed while it was matched";
                st.error = SessionError::invalid;
                if (status) *status = st;
                return false;
            }
            if (fresh.generation != gen) {
                if (attempt == 0) { std::string ignored; refresh(ignored, /*force=*/true); continue; }
                error = "conversation disk: the peer conversation was replaced while it was matched";
                st.error = SessionError::invalid;
                if (status) *status = st;
                return false;
            }
            std::vector<SavedConversation> parts;
            if (!read_parts(fresh.files.size(), parts)) {
                if (st.error == SessionError::invalid) mark_failed(id);
                if (status) *status = st;
                return false;
            }
            if (!parts_match(parts, fresh.points, fresh.cvec, layout_)) {
                mark_failed(id);   // the peer wrote it, but not for this runtime: keep the files, do not offer them again
                error = "conversation disk: the peer files do not hold the conversation their index names";
                st.error = SessionError::invalid;
                if (status) *status = st;
                return false;
            }
            SavedConversation image = std::move(parts[0]);
            for (size_t k = 1; k < parts.size(); ++k) image.stage_images.push_back(std::move(parts[k]));
            image.disk_id = 0;   // another engine's file: parking this conversation here writes the local copy
            out = std::move(image);
            if (Entry* after = find_mut(id)) note_used(*after);   // the local LRU only: a peer never writes the index
            ++reads_;
            st.error = SessionError::none;
            if (status) *status = st;
            return true;
        }
        return false;
    }
    // ---- writer
    Entry* e = find_mut(id);
    if (e == nullptr) {
        error = "conversation disk: no such conversation";
        st.error = SessionError::invalid;
        if (status) *status = st;
        return false;
    }
    const size_t n_files = e->files.size();
    // while its files are read, the RAM preflight may evict parked conversations to the disk (a write that may
    // evict here too): this one is not the victim
    struct Loading {
        uint64_t& slot;
        ~Loading() { slot = 0; }
    } loading{loading_};
    loading_ = id;
    std::vector<SavedConversation> parts;
    if (!read_parts(n_files, parts)) {
        if (st.error == SessionError::invalid) mark_failed(id);   // this run cannot use it: keep, do not offer again
        if (status) *status = st;
        return false;
    }
    // (a write during the reads may have moved the entries)
    e = find_mut(id);
    if (e == nullptr) {
        error = "conversation disk: the conversation was dropped while it was read";
        st.error = SessionError::io;
        if (status) *status = st;
        return false;
    }
    if (!parts_match(parts, e->points, e->cvec, layout_)) {
        drop(id);
        error = "conversation disk: the files do not hold the conversation their index names";
        st.error = SessionError::invalid;
        if (status) *status = st;
        return false;
    }
    SavedConversation image = std::move(parts[0]);
    for (size_t k = 1; k < parts.size(); ++k) image.stage_images.push_back(std::move(parts[k]));
    image.disk_id = id;
    out = std::move(image);
    e->used = ++clock_;               // the new LRU stamp; a failure of the index write only loses the order after a restart
    std::string ignored;
    (void) write_index(*e, ignored);
    std::stable_sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
    ++reads_;
    st.error = SessionError::none;
    if (status) *status = st;
    return true;
}

} // namespace strata::core
