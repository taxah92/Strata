# Conversation cache on disk (`--conversation-cache-dir`)

Status: **built and unit-tested on the CPU; compiled for CUDA but not yet run on a GPU in this tree.** The code
is in `include/strata/core/conversation_disk.hpp`, `src/core/conversation_disk.cpp`, the hooks in
`include/strata/core/conversation_cache.hpp` and the `--serve` loop of `src/program/generate.cpp`; tests in
`src/core/conversation_disk_test.cpp` (the tier, its index and its LRU) and `src/core/conversation_disk_peer_test.cpp`
(two engines' folders, the leases, a killed reader). The speeds quoted below were measured by the pull request's
author, not on this tree; this tree's evidence so far is `ctest -R conversation` (7/7 passed) and a compile of the
CUDA engine. Two engines on one host can share their folders read-only (`--conversation-cache-peer-dir`, "Two
engines: a writer and its peers") - that part is new here and has not been run on real cards either.

The code was written against Strata main `13eedf55` on top of the conversation-disk work of PR #5835
(`39daf8fd`), whose authorship the commits keep.

## Why

`--conversation-cache-mib` parks the conversations a request switches away from in host RAM. On
Qwen3.8-Flash-Next (IQ3_S, `--kv int8`, `--mtp`) a parked conversation costs about **19 KB per token plus
about 0.5 GB** (running state and checkpoints): 80K tokens are 2.1 GB, 114K tokens 2.7 GB (measured from
the `parked ... snapshot_bytes=` log lines). On a 62 GB host that also holds 40 GB of expert weights,
RAM fits one or two such conversations. An agent workload with more than that many long sessions
alternates and every switch reads 30-120K tokens of prompt again: 30-70 s.

A local NVMe drive reads 2.5-7 GB/s. Keeping the parked images there makes a switch back cost about one
second of file reading plus the ~0.2 s RAM-to-GPU restore that the cache already does, and lets tens of
conversations stay restorable. Because the files outlive the process, conversations also **survive an
engine restart** (same model, same settings, same layer split).

## What a parked conversation holds

`SavedConversation` (`conversation_cache.hpp`), captured by `conversation_snapshot_save`:

- `live`: the running state at the conversation's end - the GDN recurrences and conv histories of the
  linear-attention layers, each QSA layer's indexer tail (`tails`, `dead`, `block_pos`) and the PLE
  history - plus its token ids and picture keys.
- `checkpoints`: the same running state at earlier points of the chain (the system-prompt root, the last
  turn boundary, every `--prompt-cache-every` tokens), ~118 MB each on this model.
- `kv`: the positional state - each full-attention (QSA) layer's K/V pages in identity layout with its
  scales and pooled indexer rows, then the MTP draft layer's K/V.
- With a layer split: `stage_images`, one image per later stage holding **that stage's** layers
  (`layer_lo`, `layer_hi`), its own running state and checkpoints and its K/V. The drafter's K/V is
  saved once, with the image of the stage whose GPU runs the drafter.

Everything is plain host memory (`ConversationBuffer`, 16 MiB segments). Nothing in a parked image
refers to device addresses.

## Why the file SAVE / RESTORE refused a layer split

The manual session files (`POST /slots/0?action=save|restore`, engine `SAVE`/`RESTORE`) refuse
`!stages.empty() || multi_gpu`. Format v1 (`conversation_file.cpp`) serializes one image: the payload
has one layer range, and `stage_images` is not part of it; a checkpoint that still carries `stage_parts`
is refused outright. The SAVE path also streams the K/V straight from the main session's pools
(`conversation_snapshot_sources`) and the RESTORE path writes back into the main session only.

The disk tier does not need a new payload format: a parked split image is already **one v1-compatible
image per stage** (the checkpoints were split apart into the stages by
`conversation_checkpoints_split` before the capture). So each stage image is written as its own v1
session file, and a small index file binds them into one conversation. Reading returns the same
`SavedConversation` with `stage_images` filled in, and the existing request path validates and
restores it stage by stage, as it does for one taken from RAM.

## Design

### Files

```
<--conversation-cache-dir>/          the writer folder (one engine owns it for its lifetime)
    writer.lock                      exclusive flock: a second writer of the same folder is refused
    <model fp>-<config fp>-<layout key>/
        <id>.s0.strses               stage 0 (the main session; the drafter's K/V when it runs there)
        <id>.s1.strses               stage 1 ... one per stage, session file format v1
        <id>.lock                    the snapshot's lease (shared for a read, exclusive for a removal)
        <id>.idx                     the index: written last, it commits the conversation
```

- **Subfolder per identity and layout.** The model fingerprint and the config fingerprint are the ones the
  session files use (`session_identity` in `generate.cpp`: sampled model files by role; rope, K/V format,
  engine version, control vector, arithmetic switches). The layout key hashes each stage's layer range.
  Another model, engine version or `--layer-split` carve gets its own folder and never parses foreign
  files. Other folders are reported at start-up and left alone.
- **Stage files** are written by `session_file_write`: aligned 16 MiB blocks, `O_DIRECT` where the
  filesystem accepts it (no page-cache pollution of the mmap'd model), a payload hash, a hidden temporary
  name renamed over the target, file and folder flush, a free-space preflight
  (`--session-min-free-mib`). They are read by `session_file_read` with every count checked against the
  runtime's limits before allocation (`conversation_session_read_limits`) and a RAM preflight.
- **Index file** (`STRCDX` v1, little-endian): identity, layout key, id, LRU stamp, control-vector mode,
  every resume point (`tokens`, a 64-bit hash of its token ids, its picture keys, live or checkpoint) and
  each stage file's size, then a hash of all of it. It is written last with the same temp + flush +
  rename + folder flush sequence.

### Two engines: a writer and its peers

One replica owns one folder ("writer mode"): it creates the subfolder, removes crash leftovers, keeps the
least recently used entries and is the only process that writes, renames or unlinks anything there. The
ownership is a file, not a convention: `<dir>/writer.lock` is an exclusive `flock` held for the engine's whole
lifetime, and a second engine that opens the same folder fails at start-up with a message naming the lock.
That protects a restarting engine from cleaning up a folder another live engine is writing.

`--conversation-cache-peer-dir PATH` (repeatable) adds a **read-only** view of another engine's writer folder -
this is what `serve/replicas.py` sets up for replicas on the same host:

- A peer creates no directory and no file, chmods nothing, removes nothing - not a crash leftover, not a
  damaged index, not an orphan - and never writes an index. A file it cannot parse is skipped and tried again
  on the next re-scan.
- It re-scans when the leaf folder's modification time changed (a `stat`; `--conversation-cache-refresh-ms`
  bounds how often) and on demand: after a match miss the request path forces one re-scan, because the other
  replica publishes when it parks.
- The index (v2; v1 still reads) names the owner and carries a content generation. A snapshot is referenced as
  `(owner, id, generation)`, never as a bare id, so two owners' random ids cannot be confused.
- Every read holds the snapshot's `<id>.lock` shared (`flock(LOCK_SH)`) while it reads every stage file and
  validates the image, then re-reads the index and compares the generation it matched. A snapshot that was
  replaced or removed in between costs one forced re-scan and a fallback, never a wrong state.
- A writer that supersedes a copy, evicts for the budget or clears after non-finite logits takes that
  snapshot's lease exclusive and non-blocking, and **skips** a snapshot another process is reading: the next
  least recently used one goes instead, and when every victim is taken the byte budget runs over for a while
  (counted, never a deletion under a reader). A killed reader's lock is released by the kernel.
- The RAM preflight of a peer read (what the image needs, and the floor) runs **before** the lease is taken.
  The restore then passes the ordinary request path: the outgoing conversation is parked, the image is
  validated with `conversation_snapshot_validate` per stage, and after it `conversation_disk_body_extends`
  compares the request body with the loaded image's matched resume point token by token - the index hash is a
  search key, not proof that two conversations sharing a system prompt are the same one.
- The store picks the deepest parked conversation across the live session, RAM, the engine's own folder and
  its peers (the same `--conversation-cache-disk-min-gain` threshold); a tie prefers the live point, then the
  engine's own folder. LRU stays per owner: the sum of the replicas' budgets is the store's ceiling, and two
  copies of one conversation on two owners may coexist until their owners evict them - the deepest exact
  prefix wins, so it is only wasted space, not a wrong restore. One alternative source is tried after a
  failed read; nothing more.

The mode is POSIX only: on Windows a peer folder is refused at start-up, while the local writer keeps working
as it did (no inter-process lease there - the folder is private by configuration).

### Two replicas on one host: one run-config block

The `--replicas` mode builds the folders from one block in the run config instead of duplicating `args` per engine
(`serve/server.py`):

```json
"replicas": 2,
"replica_handoff": {
  "enabled": true,
  "dir": "/mnt/nvme/strata-handoff",
  "spill": "on-park",
  "min_tokens": 8192,
  "min_gain": 4096,
  "disk_gib_per_replica": 16,
  "max_snapshot_mib": 8192,
  "min_free_mib": 8192,
  "refresh_ms": 250
}
```

Replica `i` gets `--conversation-cache-dir <dir>/r<i>`, one `--conversation-cache-peer-dir <dir>/r<j>` per neighbour,
`--conversation-cache-owner r<i>` and the budget/threshold flags from the block; the directories are created 0700
before the engines start, and the server prints one line that says the handoff is on. The start is refused with a
message (not a silent fallback) when there are fewer than two replicas, when batch slots or `--parallel` above 1
are configured, on a non-POSIX host, when the engines' own `args` already name a `--conversation-cache-dir`, or when
two replicas would share one folder.

These numbers are starting stand values, not safe production values: the store must be an NVMe (a tmpfs would keep
a second RAM copy of every image), the per-replica budget is a ceiling rather than a reservation, and `min_tokens` /
`min_gain` are for the A/B to tune - a short request should not pay for a handoff.

### Spill-on-par

`--conversation-cache-spill-on-park` (opt-in) writes a conversation to the disk folder **at the park itself**,
not only when RAM evicts it, when the idle writer gets to it or when it cannot fit the RAM budget. The
handoff case is "A was parked when this engine switched to C, and A's next turn is routed to the other card":
a tier that only writes on eviction may have nothing to offer at that moment. The write is synchronous, on the
request loop between requests, the background checkpoint copies are joined first (as a session SAVE does) and
the published files are immutable; `holding()` means an image the tier already holds is not written twice. It
also covers a park the RAM cache refuses on the physical-RAM floor: the disk copy is written before the RAM
put, so the conversation is not lost and the next turn reads it back.

### Matching

The tier keeps only the index in RAM (a few hundred bytes per conversation). `best()` takes the lengths
of all points whose pictures match the prompt's, hashes the prompt once in a streaming pass, and returns
the deepest point whose hash matches - the rule of `conversation_prefix` (the last prompt token is never
covered; pictures must match in order; the control-vector mode must match). A hash match is confirmed
after the read: the loaded live ids are compared with the index, and the image then passes the ordinary
`conversation_snapshot_validate` of every stage before any device write. A collision or a stale file
costs one wasted read and falls back to reading the prompt; it can never restore a wrong state.

In the request path (`generate.cpp`, after `conversations.best`), a disk match is used when it reaches at
least `--conversation-cache-disk-min-gain` (default 4096) tokens deeper than everything else on offer
(the live session, an idle batch slot, a RAM-parked conversation). It is loaded into an
`incoming` image and from there on follows the existing "take" path: the outgoing conversation is parked
with the incoming image counted as `held`, the image is restored stage by stage, its K/V is retained for
the next park. Disk matches are never borrows: the files stay on disk and are not changed by a read.

### When it writes (disk wear)

- **On the park itself** with `--conversation-cache-spill-on-park` (off by default; see above).
- **On eviction.** `ConversationCache::evict_oldest()` calls the tier's hook before it drops an entry
  (eviction for room or for the physical-RAM floor; not `take`, `clear` or `drop_superseded`). The hook
  writes the image unless the tier already holds it.
- **When idle.** With `--conversation-cache-disk-idle-s N` (default 20), the serve loop's `next_line`
  waits for a request at most N seconds; on timeout it writes the oldest RAM-parked conversation the
  tier does not hold yet, marks it (`SavedConversation::disk_id`) and waits again. This is what makes
  conversations survive a restart, and it moves the write cost out of the next switch. A request that
  arrives during a write waits for that one write (about a second for 2-3 GB). `0` writes on eviction only.
- **No duplicate writes.** `holding()` recognizes an image identical to a kept conversation (same points,
  same mode); `drop_superseded()` removes the copies a newer turn of the same conversation replaces
  (the rule of `ConversationCache::drop_superseded`). So a conversation alternating with others costs
  one write per park at most; a burst of switches inside the idle window writes only what RAM evicts.

### Retention

LRU by the stamp in the index (rewritten on every read, persisted across restarts), bounded by
`--conversation-cache-disk-gib` (default 64) and 256 conversations. The conversation being read is
never the victim. After non-finite logits (#879) every conversation on disk is removed, like the RAM ones.

### Concurrency and crash safety

- Everything runs on the engine's request thread: the idle writer runs only while no request, batch
  window or pipelined group is in flight (`next_line` is reached only then); an eviction write runs
  inside the park that caused it; a read runs before the outgoing conversation is parked. No device
  state is read or written by the tier, so it cannot race the GPU.
- A write is announced to the watchdog like a SAVE (`progress_allow` with `session_phase_limit_s` for
  the flushes, a beat per 16 MiB block).
- A crash between stage files and the index leaves stage files without an index; a crash during a file
  leaves a hidden `.tmp`. `open()` removes both. A damaged or foreign index drops its conversation.
- A stage file that fails its hash or the runtime's limits on read drops the conversation; a RAM or I/O
  refusal keeps it.

### RAM

RAM then only needs the active conversation(s): e.g. `--conversation-cache-mib 3072
--conversation-cache-slots 1` keeps one parked image in RAM (the one most likely to come back next) and
everything else on disk. A read needs the image's size in RAM above `--conversation-cache-min-free-mib`;
the read's RAM preflight evicts RAM-parked conversations (to disk) first when it has to.

## Flags

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--conversation-cache-dir PATH` | off | the folder (local SSD; the tier makes a 0700 subfolder) |
| `--conversation-cache-disk-gib N` | 64 | byte budget of the files |
| `--conversation-cache-disk-idle-s N` | 20 | write parked conversations after N s without a request (0 = on eviction only) |
| `--conversation-cache-disk-min-gain N` | 4096 | read from disk only for at least N more reused tokens |
| `--conversation-cache-peer-dir PATH` | none | another engine's `--conversation-cache-dir`, read only (repeatable) |
| `--conversation-cache-owner NAME` | folder name | this engine's name in the index and the logs |
| `--conversation-cache-spill-on-park` | off | write every parked conversation to the disk at the park |
| `--conversation-cache-max-snapshot-mib N` | 8192 | the largest conversation written or read (0 = no limit) |
| `--conversation-cache-refresh-ms N` | 250 | how often a peer re-scans the other engine's folder |

The tier needs the RAM cache on (`--conversation-cache-mib`, `--conversation-cache-slots`,
`--prompt-cache`). With `--layer-split auto` the carve depends on the free VRAM at start-up; for the
files to survive restarts, give the split explicitly (`--layer-split 19`), or a changed carve starts an
empty folder.

## Log lines

```
strata serve: conversation disk: <folder> (owner=r0): N conversations kept from earlier runs, X of Y GiB; ...
strata serve: conversation disk: peer <dir> (owner=r1): N conversations, X of Y GiB, read-only
strata serve: conversation disk: wrote N tokens (evicted|idle|direct|on-park) in T ms, R GB/s; entries=E bytes=B evictions=V
strata serve: conversation disk: read N tokens (M reused, live|checkpoint[, peer r1]) in T ms, R GB/s; entries=E bytes=B
strata serve: conversation disk: peer r1 refreshed: A new, G gone, D damaged
strata serve: conversation disk: stats (<tag>) wrote=W read=R evictions=V skipped=S lease_conflicts=L overshoot=O ...
strata serve: conversation disk: not written (...) / cannot read a conversation (...)
```

The server's Monitor (`ConvCacheLog`) counts them as `disk_writes`, `disk_reads`, `disk_entries`,
`disk_bytes`.

## Open points

- Reads are synchronous `pread`-style 16 MiB blocks followed by a full hash and then the existing
  host-to-device restore. Overlapping the file read with the upload (pinned double buffer per stage)
  would hide most of the restore time; not done in this draft.
- The idle writer copies nothing but holds the engine thread for one write; a request arriving then waits.
- The manual SAVE/RESTORE still refuses a layer split; it could reuse the per-stage file set.
- The idle writer writes whole conversations; a conversation that grows by a few hundred tokens per turn
  is rewritten whole each idle period (there is no delta format).
- A failed write backs off for 30 s; there is no per-conversation back-off, no file-age cap and no "keep the
  last three copies" pruning - `drop_superseded` and the LRU are the whole retention policy.
- The peer then writes its own copy when it parses the restored conversation again; nothing migrates the old
  copy to the new owner, so a conversation that moves between replicas leaves one copy behind until its
  owner's LRU takes it.
- Two engines' leases are `flock` on a local filesystem; a network folder (NFS/CIFS) is not a supported
  store: the lock semantics and `O_DIRECT` differ there.
- `used` (the LRU stamp) is per owner and not comparable between owners; the tie-break across folders is
  "own before a peer", then the order of `--conversation-cache-peer-dir`.
