// The parent parking store's client.  The Python server (serve/parent_parking.py) owns ONE RAM store of parked
// conversations for every replica of the engine, and each engine talks to it over a SOCK_SEQPACKET socketpair it
// inherits at start (--conversation-parent-parking-fd).  The store owns the index, the LRU and the memfd snapshots;
// this client only speaks the protocol: it asks for a memfd per stage, writes the session file into it with
// session_file_write_fd, and reads a sealed one back with session_file_read_fd.  No CUDA here, so the unit and
// multiprocess tests build without a model or a card.
//
// The wire format is frozen in .scratch/handoff-notes/parent-parking/contract.md (protocol v1): a 36-byte
// little-endian frame, a UTF-8 JSON payload of at most 1 MiB, and SCM_RIGHTS descriptors only in the PARK_BEGIN and
// ACQUIRE responses.  One request is outstanding per channel; every exchange is bounded by `timeout_ms`.
//
// A transport failure does not kill the engine: the client becomes degraded, the request that hit it falls back to
// reading the prompt, and the next real restart of the replica gets a fresh socketpair and the same store.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

// What the store is bound to: the identity a session file carries, the stage layout, and the replica's name.  The
// first replica's HELLO fixes the identity of the whole store; another replica with other values is a start error.
struct ParentIdentity {
    uint64_t model = 0, config = 0, layout_key = 0;
    std::vector<std::pair<int64_t, int64_t>> stages;   // [layer_lo, layer_hi) of stage 0, then of each later stage
    std::string owner;                                 // "r0".."rN": diagnostics and counters, not access control
};

// One resume point of a conversation, as the store matches it: the live end or a checkpoint.  `pinned` is run-time
// metadata (a session file does not carry it) and decides whether the store may borrow the entry.
struct ParentPoint {
    int64_t tokens = 0;
    uint64_t hash = 0;
    std::vector<ConversationImageKey> imgs;
    bool live = false;
    bool pinned = false;
};

// The store's answer to "does anything already hold this conversation": the identical entry (dedup: nothing to write)
// and the longest parked entry this conversation's outgoing state supersedes (ConversationCache::parked_prefix_tokens'
// rule: parking now would only rewrite what a turn from now drops).  prefix_lengths are the deepest checkpoints of
// parked entries the outgoing state does not hold but may supersede by a prefix hash (seed STRIDS01): the engine
// hashes its ids at exactly those lengths and sends them back as prefix_hashes.
struct ParentAdvice {
    uint64_t holding_id = 0;
    int64_t parked_prefix_tokens = 0;
    std::vector<int64_t> prefix_lengths;
};

// A candidate the store offers for a prompt: what to acquire, and how much RAM the restore will be charged.
struct ParentCandidate {
    uint64_t id = 0, generation = 0;
    int64_t tokens = 0;
    bool live = false, borrow = false;
    std::vector<uint64_t> stage_bytes;
    uint64_t read_charge_bytes = 0;
    std::string owner;   // the replica that published it ("r0".."rN"): the log says whose conversation this is
};

// The store's counters (the plan's §14.2 snapshot).  The engine reads them for its own log line and for
// STRATA_KV_GROW_HOLD, which needs the longest resume point the store holds.
struct ParentStats {
    bool enabled = false;
    uint64_t capacity_bytes = 0, charged_bytes = 0, entry_bytes = 0, pending_bytes = 0;
    uint64_t restore_reserved_bytes = 0, retired_lease_bytes = 0, published_bytes = 0, store_epoch = 0;
    size_t entries = 0, pending = 0, leases = 0, published = 0, deduplicated = 0, reads = 0, peer_reads = 0;
    size_t evictions = 0, eviction_skipped_leased = 0, superseded = 0, clears = 0, damaged = 0;
    size_t capacity_refusals = 0, budget_violations = 0;
    int64_t longest_tokens = 0;   // the deepest resume point any committed entry holds
};

class ConversationParentClient;

// A pending publication: the store's memfds, one per stage, writable until the commit seals them.  The destructor
// closes this side's copies (the store keeps its own); the pending itself is dropped by PARK_ABORT or by the channel's
// EOF, so a caller that gives up must still abort it.
class ParentPending {
public:
    ParentPending() = default;
    ParentPending(ParentPending&& other) noexcept;
    ParentPending& operator=(ParentPending&& other) noexcept;
    ParentPending(const ParentPending&) = delete;
    ParentPending& operator=(const ParentPending&) = delete;
    ~ParentPending() { close_fds(); }

    bool valid() const { return pending_id_ != 0; }
    uint64_t id() const { return pending_id_; }
    size_t stages() const { return fds_.size(); }
    int fd(size_t stage) const { return stage < fds_.size() ? fds_[stage] : -1; }
    const std::vector<uint64_t>& stage_bytes() const { return stage_bytes_; }
    // After the last stage is written: the store seals the memfds on COMMIT, and a writable copy left open would only
    // keep the write side alive.
    void close_fds();
    void reset();

private:
    friend class ConversationParentClient;
    uint64_t pending_id_ = 0;
    std::vector<int> fds_;
    std::vector<uint64_t> stage_bytes_;
};

// A lease on a committed entry: the sealed stage fds, and the store's promise to keep the bytes (and to count the
// restore's RAM) until it is released.  The caller holds it until the materialized image is destroyed, so the budget
// never undercounts the heap image while the next conversation parks.
class ParentLease {
public:
    ParentLease() = default;
    ParentLease(ParentLease&& other) noexcept;
    ParentLease& operator=(ParentLease&& other) noexcept;
    ParentLease(const ParentLease&) = delete;
    ParentLease& operator=(const ParentLease&) = delete;
    ~ParentLease();   // releases with "io" when the caller did not: the entry stays for another try

    bool valid() const { return lease_id_ != 0; }
    uint64_t lease_id() const { return lease_id_; }
    const ParentCandidate& candidate() const { return candidate_; }
    size_t stages() const { return fds_.size(); }
    int fd(size_t stage) const { return stage < fds_.size() ? fds_[stage] : -1; }
    void close_fds();
    // "ok" | "invalid" | "memory" | "io" | "mismatch": invalid and mismatch quarantine the entry for every replica,
    // memory and io keep it for another try.  Idempotent; false only when the channel is gone (the lease is then
    // dropped locally and the store cleans it up on EOF).
    bool release(const char* result, std::string& error);

private:
    friend class ConversationParentClient;
    ConversationParentClient* client_ = nullptr;
    uint64_t lease_id_ = 0;
    ParentCandidate candidate_;
    std::vector<int> fds_;
};

class ConversationParentClient {
public:
    enum class State { off, connected, degraded };

    ConversationParentClient() = default;
    ~ConversationParentClient();
    ConversationParentClient(const ConversationParentClient&) = delete;
    ConversationParentClient& operator=(const ConversationParentClient&) = delete;

    // Takes the inherited descriptor and says HELLO.  False means the store refused the identity or the protocol: the
    // engine must not start (a start error, never a silent loss of the feature).  `timeout_ms` bounds every control
    // exchange, not the K/V transfer through the memfds.
    bool connect(int fd, const ParentIdentity& identity, int timeout_ms, std::string& error);
    void close();
    State state() const { return state_; }
    bool enabled() const { return state_ != State::off; }
    bool degraded() const { return state_ == State::degraded; }
    const ParentIdentity& identity() const { return identity_; }
    const std::string& owner() const { return identity_.owner; }
    uint64_t epoch() const { return epoch_; }
    int timeout_ms() const { return timeout_ms_; }
    // The last failure's protocol code ("gone", "stale_generation", "capacity", "stale_epoch", ...), empty when the
    // last call succeeded or the failure was local (a timeout, a closed channel).  The caller needs it to tell a stale
    // match (worth one fresh attempt) from a real refusal.
    const std::string& last_error_code() const { return last_error_code_; }

    // PARK_ADVICE: the identical entry already in the store (dedup) and the longest entry this conversation supersedes.
    // `prefix_hashes` (optional) are this conversation's ids hashed at the answer's prefix_lengths: the store counts
    // them into parked_prefix_tokens.
    bool advice(const std::vector<ParentPoint>& points, bool cvec, ParentAdvice& out, std::string& error,
                const std::vector<std::pair<int64_t, uint64_t>>& prefix_hashes = {});
    // PARK_BEGIN: the store's memfds for this snapshot, one per stage, each sized to `stage_bytes[k]`.  A dedup answer
    // leaves `out` invalid and sets `dedup` (nothing to write, no commit).  `prefix_hashes` (optional) let COMMIT drop
    // an entry whose deepest checkpoint the new snapshot supersedes by a prefix hash.
    bool begin(const std::string& publish_id, const std::vector<ParentPoint>& points, bool cvec,
               const std::vector<uint64_t>& stage_bytes, ParentPending& out, bool& dedup, std::string& error,
               const std::vector<std::pair<int64_t, uint64_t>>& prefix_hashes = {});
    // PARK_COMMIT: seals the memfds and publishes.  Idempotent by publish_id (a lost ACK costs nothing).
    bool commit(const std::string& publish_id, const ParentPending& pending, uint64_t& id, uint64_t& generation,
                std::string& error);
    // PARK_ABORT: drops the pending.  Idempotent.
    bool abort(const std::string& publish_id, uint64_t pending_id, std::string& error);

    // MATCH_BEGIN: the resume lengths worth hashing (sorted, unique, below the prompt's last token).
    bool match_begin(int64_t prompt_tokens, bool cvec, const std::vector<ConversationImageKey>& images,
                     const std::vector<uint64_t>& exclude_ids, uint64_t& cookie, std::vector<int64_t>& lengths,
                     std::string& error);
    // MATCH_END: the deepest candidate whose hash matched, or none.
    bool match_end(uint64_t cookie, const std::vector<std::pair<int64_t, uint64_t>>& hashes,
                   std::optional<ParentCandidate>& out, std::string& error);
    // ACQUIRE: the sealed stage fds and a lease that keeps the bytes (and the RAM charge) until it is released.
    bool acquire(const ParentCandidate& candidate, ParentLease& out, std::string& error);

    // RELEASE: ends a lease.  "invalid"/"mismatch" quarantine the entry for every replica.
    bool release(uint64_t lease_id, const char* result, std::string& error);
    // CLEAR: every committed entry of every replica (#879 after non-finite logits, and unload).
    bool clear(const char* reason, size_t& cleared, std::string& error);
    bool stats(ParentStats& out, std::string& error);
    bool ping(std::string& error);

private:
    friend class ParentLease;
    bool request(uint16_t op, const std::string& payload, std::string& response, std::vector<int>& fds,
                 std::string& error);
    void degrade(const std::string& why);

    int fd_ = -1;
    State state_ = State::off;
    ParentIdentity identity_;
    int timeout_ms_ = 2000;
    uint64_t epoch_ = 0;
    uint64_t next_request_id_ = 1;
    std::string last_error_;
    std::string last_error_code_;
};

// The resume points of `image`: the live end first, then every checkpoint in the image's order.
std::vector<ParentPoint> parent_points(const SavedConversation& image);
// The hash the store matches a resume point by: conversation_disk_ids_hash's function (seed "STRIDS01").
uint64_t parent_ids_hash(const int32_t* ids, size_t n);
// The hashes MATCH_BEGIN's lengths need, in one pass over `ids` (the walk ConversationDiskTier::best_ids does).
std::vector<std::pair<int64_t, uint64_t>> parent_hashes(const std::vector<int32_t>& ids,
                                                        const std::vector<int64_t>& lengths);
// Reads every stage of `lease` into `image`: stage 0 is the image itself, the later stages its stage_images (the shape
// conversation_snapshot_restore expects).  The fds are closed after the read; the lease stays until the caller
// releases it, after the GPU restore.
bool parent_load(ParentLease& lease, const std::vector<SessionReadLimits>& limits, SavedConversation& image,
                 std::string& error, SessionStatus* status = nullptr);
// A fresh publish id: 32 lowercase hex characters.
std::string parent_publish_id();

} // namespace strata::core
