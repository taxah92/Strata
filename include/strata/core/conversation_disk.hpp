// The conversation cache's disk tier (--conversation-cache-dir): parked conversations kept as files on a local drive
// (an NVMe SSD), so that RAM holds only the conversations in use while tens of them stay restorable - and survive an
// engine restart.  docs/CONVERSATION_CACHE_DISK.md has the design; in short:
//
//   * one conversation = one file per layer-split stage (session file format v1, conversation_file.hpp: aligned
//     16 MiB blocks, O_DIRECT where the filesystem takes it, payload hash, atomic rename) plus a small index file
//     (`<id>.idx`) written LAST: it commits the set.  Stage files without an index are a write that did not finish
//     and are removed when the writer opens the folder.  A `<id>.lock` file carries the cross-process lease.
//   * the files live in a subfolder named after the session identity (model and config fingerprints) and the stage
//     layout (each stage's layer range), so another model, another engine version or another --layer-split carve
//     never sees them.  Other subfolders are left alone and only reported.
//   * the index file carries what a request is matched by - every resume point's length, the hash of its tokens and
//     its pictures - so the tier keeps a few hundred bytes per conversation in RAM and matches a prompt without
//     reading a stage file.  A hash match is confirmed after the load (the loaded tokens are compared again by the
//     caller's ordinary validation), so a collision costs one wasted read, never a wrong state.
//   * least recently used conversations leave when the byte budget or the entry cap is reached.
//
// Two modes (--replica-handoff: one folder per engine, every engine sees the others read-only):
//
//   * writer: the engine owns the folder for its whole lifetime (`writer.lock`, an exclusive flock taken at open:
//     a second writer of the same folder is refused instead of cleaning up under the first).  It creates, evicts and
//     removes only its own files; every removal of a snapshot first takes that snapshot's exclusive lease, and a
//     snapshot another process is reading is skipped (the budget may run over until it is free).
//   * peer: read-only view of another engine's writer folder.  It never creates a directory or a file, never
//     chmods, never removes anything (not even a crash leftover) and never writes an index: a damaged or foreign
//     file is ignored and tried again on the next refresh.  It re-scans the folder when its modification time
//     changed (and on demand after a match miss), and every read takes the snapshot's shared lease, re-reads the
//     index under it and compares the generation it matched, so a snapshot replaced or removed in between costs one
//     refresh and a fallback, never a wrong state.
//
// Pure host code (no CUDA), single-threaded: the engine calls it from its request loop only.  The hashes detect
// accidental corruption; they do not authenticate a file: keep the folder private to the engine's user.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The engine's stage layout: [layer_lo, layer_hi) of stage 0 (the main session), then of each later stage.  A
/// conversation's files are usable only by an engine with the same layout (each stage file holds one carve).
struct ConversationDiskLayout {
    std::vector<std::pair<int64_t, int64_t>> stages;
    uint64_t key() const;
};

struct ConversationDiskOptions {
    std::string dir;                 ///< the writer folder this tier uses; a peer reads the other engine's writer folder
    std::string owner;               ///< the writer's name, written into the index and the logs (default: the folder's name)
    uint64_t budget_bytes = 0;       ///< writer: --conversation-cache-disk-gib, the most bytes of files kept
    size_t max_entries = 256;        ///< the most conversations kept, whatever their size
    uint64_t min_free_bytes = 0;     ///< free-space preflight of every stage file (--session-min-free-mib)
    uint64_t max_snapshot_bytes = 0; ///< --conversation-cache-max-snapshot-mib: a larger conversation is refused, 0 = no limit
    int64_t refresh_ms = 250;        ///< peer: the least interval between two automatic re-scans (an on-demand one ignores it)
    bool durable = true;             ///< flush each file and the folder (crash safety); off only in tests
};

/// One point a request can resume from: the conversation's live end or one of its checkpoints.
struct ConversationDiskPoint {
    int64_t tokens = 0;
    uint64_t hash = 0;                          ///< conversation_disk_ids_hash of the first `tokens` ids
    std::vector<ConversationImageKey> imgs;     ///< the pictures below `tokens`, as the checkpoint keeps them
    bool live = false;
    bool operator==(const ConversationDiskPoint&) const = default;
};

uint64_t conversation_disk_ids_hash(const int32_t* ids, size_t n);
/// The live point first, then every checkpoint's, in the image's order.
std::vector<ConversationDiskPoint> conversation_disk_points(const SavedConversation& image);

/// The loaded image is the same conversation as the request body up to `tokens`: every id of that resume point equals
/// the body's (not only the 64-bit hash, which is a search key), the pictures below it match and the control-vector
/// mode matches.  A mismatch is a fallback, never a restore.  No filesystem access (the caller's request body and the
/// loaded image only).
template<class Token>
bool conversation_disk_body_extends(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                                    bool cvec, const SavedConversation& image, int64_t tokens, std::string& why) {
    if (tokens <= 0 || (size_t) tokens > prompt.size()) { why = "the matched length is not inside the prompt"; return false; }
    if (image.cvec != cvec) { why = "the control-vector mode differs"; return false; }
    const ConversationCheckpoint* at = nullptr;
    if ((int64_t) image.live.ids.size() == tokens) at = &image.live;
    else
        for (const auto& c : image.checkpoints)
            if ((int64_t) c.ids.size() == tokens) { at = &c; break; }
    if (at == nullptr) { why = "the loaded image has no such resume point"; return false; }
    for (int64_t i = 0; i < tokens; ++i)
        if ((int64_t) prompt[(size_t) i] != (int64_t) at->ids[(size_t) i]) {
            why = "the prompt differs from the loaded image inside the matched prefix";
            return false;
        }
    std::vector<ConversationImageKey> below;
    for (const auto& im : images) if (im.start < tokens) below.push_back(im);
    if (below != at->imgs) { why = "the pictures below the match differ"; return false; }
    return true;
}

class ConversationDiskTier {
public:
    enum class Mode { writer, peer };
    ConversationDiskTier() = default;
    ConversationDiskTier(const ConversationDiskTier&) = delete;   // it owns the writer's lifetime lock file descriptor
    ConversationDiskTier& operator=(const ConversationDiskTier&) = delete;

    struct Match {
        uint64_t id = 0;       ///< 0: none
        int64_t tokens = 0;
        bool live = false;
        uint64_t generation = 0;   ///< the content version the match was made against (compare it under the lease)
    };
    struct Entry {
        uint64_t id = 0, used = 0, generation = 0, bytes = 0;
        bool cvec = true;
        std::vector<ConversationDiskPoint> points;
        std::vector<uint64_t> files;   ///< each stage file's size, stage 0 first
        bool failed = false;           ///< this run could not restore it: not offered again until a restart (kept)
    };
    struct OpenReport {
        size_t kept = 0, dropped = 0, orphans = 0;   ///< index files taken, refused and stray files removed
        size_t other_folders = 0;                    ///< subfolders of other identities or layouts, left alone
        uint64_t other_bytes = 0;
    };
    struct RefreshReport {
        size_t added = 0, gone = 0, damaged = 0;   ///< entries new to the local index, disappeared, unreadable (kept on disk)
    };

    /// Writer mode: creates (or reopens) the tier's subfolder, takes the folder's lifetime lock and indexes the
    /// conversations already there.  False: the tier stays off and `error` says why (the engine runs on without it).
    bool open(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
              const ConversationDiskLayout& layout, std::string& error, OpenReport* report = nullptr);
    /// Peer mode: a read-only view of another engine's writer folder.  Creates nothing; a folder that is not there
    /// (yet) is reported through `error` and leaves the tier off.  False: the tier stays off.
    bool open_peer(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
                   const ConversationDiskLayout& layout, std::string& error, OpenReport* report = nullptr);
    ~ConversationDiskTier();

    Mode mode() const { return mode_; }
    bool is_peer() const { return mode_ == Mode::peer; }
    bool enabled() const { return open_; }
    const std::string& owner() const { return owner_; }
    const std::string& folder() const { return folder_; }
    /// Peer: re-scan the folder now if its modification time changed (or `force`), never throttled when forced.
    /// Writer: a no-op that returns true.  False only for a peer that is off or a folder it could not stat.
    bool refresh(std::string& error, bool force = false, RefreshReport* report = nullptr);

    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    size_t writes() const { return writes_; }
    size_t reads() const { return reads_; }
    size_t evictions() const { return evictions_; }
    size_t eviction_skipped() const { return eviction_skipped_; }   ///< a victim another process had leased (writer)
    size_t lease_conflicts() const { return lease_conflicts_; }     ///< exclusive leases that were busy (writer)
    size_t refreshes() const { return refreshes_; }                 ///< peer re-scans performed
    bool overshoot() const { return overshoot_; }                   ///< the last budget pass kept more bytes than allowed
    bool contains(uint64_t id) const { return find(id) != nullptr; }
    const Entry* find(uint64_t id) const;
    uint64_t generation(uint64_t id) const;

    /// The deepest point of any kept conversation that `prompt` starts with (same rule as conversation_prefix: the
    /// prompt's last token is never covered, pictures must match in order).  Ties prefer the most recently used.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::vector<int32_t> ids(prompt.size());
        for (size_t i = 0; i < prompt.size(); ++i) ids[i] = (int32_t) prompt[i];
        return best_ids(ids, images, cvec);
    }
    Match best_ids(const std::vector<int32_t>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const;

    /// A kept conversation with exactly this image's resume points and control-vector mode (nothing to write).
    uint64_t holding(const SavedConversation& image) const;
    /// Writer: writes `image` (its stage images included) as one conversation: the conversations it supersedes go
    /// first (drop_superseded), then the least recently used until it fits.  `id` is the new (or the identical kept)
    /// conversation.  On failure nothing new is left on the disk and `status->error` says what kind of failure.
    bool spill(const SavedConversation& image, uint64_t& id, std::string& error, SessionStatus* status = nullptr,
               const SessionProgress& progress = {}, const SessionPhase& phase = {});
    /// Reads conversation `id` into `out` (stage images included).  Writer: out.disk_id = id.  Peer: the shared lease
    /// is taken first, then the index is re-read and `expect_generation` (when not 0) compared; a mismatch costs one
    /// forced refresh and one retry, then a normal failure.  `limits[k]` bounds stage k's file
    /// (conversation_session_read_limits; a stage past the list reads with defaults).  A file that is corrupt or not
    /// this runtime's drops the conversation (writer) or marks it failed (peer); a RAM or I/O refusal keeps it.
    /// `out` is unchanged on failure.
    bool load(uint64_t id, const std::vector<SessionReadLimits>& limits, SavedConversation& out, std::string& error,
              SessionStatus* status = nullptr, uint64_t expect_generation = 0);
    /// Writer: removes conversation `id` (its index file first, then its lease file, then its stage files) unless
    /// another process holds its lease - then nothing is removed and false is returned (the eviction is skipped).
    bool drop(uint64_t id);
    /// The kept conversations `image` supersedes (ConversationCache::drop_superseded's rule: their deepest checkpoint
    /// is one of the image's points), removed.  Returns how many.
    size_t drop_superseded(const SavedConversation& image, uint64_t keep = 0);
    /// This run cannot restore `id` (read limits, snapshot validation): it is not offered again, its files stay
    /// (another start-up with other settings may restore it; the budget's LRU removes it in time).
    void mark_failed(uint64_t id);
    /// Writer: every kept conversation removed (#879: after non-finite logits any of them may carry the poison).  The
    /// index files go first (nothing is matched from then on); a snapshot another process is reading keeps its stage
    /// files (they become leftovers the next writer open removes), its lease is not violated.
    size_t clear();

private:
    bool open_impl(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
                   const ConversationDiskLayout& layout, bool peer, std::string& error, OpenReport* report);
    bool scan(OpenReport* open, RefreshReport* refresh, bool peer, bool first);
    bool parse_index(const std::string& path, uint64_t id, Entry& e) const;
    bool write_index(const Entry& e, std::string& error) const;
    bool make_room(uint64_t incoming, std::string& error, uint64_t keep = 0);
    void remove_files(uint64_t id, size_t stages) const;
    std::string path_of(uint64_t id, const char* suffix) const;
    std::string stage_path(uint64_t id, size_t stage) const;
    std::string lock_path(uint64_t id) const;
    Entry* find_mut(uint64_t id);
    void note_used(Entry& e);

    bool open_ = false;
    Mode mode_ = Mode::writer;
    ConversationDiskOptions options_;
    SessionFileIdentity identity_;
    ConversationDiskLayout layout_;
    std::string owner_;
    std::string folder_;               ///< the leaf folder (the identity's and the layout's)
    std::string writer_dir_;           ///< the folder --conversation-cache-dir names (writer.lock lives there)
    std::vector<Entry> entries_;
    uint64_t bytes_ = 0, eviction_skipped_ = 0, lease_conflicts_ = 0, refreshes_ = 0;
    size_t writes_ = 0, reads_ = 0, evictions_ = 0;
    uint64_t clock_ = 0;               ///< writer: the LRU stamp source
    uint64_t loading_ = 0;             ///< writer: the conversation its own load is reading (not a victim)
    uint64_t generation_ = 0;          ///< writer: the next content version
    bool overshoot_ = false;
    bool leaf_seen_ = false;           ///< peer: the leaf folder existed at the last scan
    int64_t leaf_mtime_ = 0;           ///< peer: the leaf folder's modification time at the last scan
    int64_t last_scan_ms_ = 0;         ///< peer: monotonic ms of the last scan (the refresh_ms throttle)
    int lock_fd_ = -1;                 ///< writer: the folder's lifetime lock (POSIX); -1 = not taken (Windows, off)
};

} // namespace strata::core
