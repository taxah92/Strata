# Parent parking: one RAM store in the server for every replica (`replica_parent_parking`)

Status: **built, unit-tested on the CPU, integration-tested against the real Python store, and run on the live stand
(two Tesla V100: a conversation parked on one card was restored on the other).** The code is in `serve/parent_parking.py` (the store),
`include/strata/core/conversation_parent.hpp` and `src/core/conversation_parent.cpp` (the engine's client), the
`--serve` loop of `src/program/generate.cpp`, and the fd variants of the session file in
`include/strata/core/conversation_file.hpp`. Tests: `src/core/conversation_file_test.cpp` (the fd API),
`src/core/conversation_parent_test.cpp` (the client against a fake broker), `serve/test_parent_parking.py` (the store
and its protocol), `serve/test_parent_parking_integration.py` (the real store against the real client), and
`serve/test_parent_parking_server.py` (the run config, the per-spawn channel, the lifecycle and the metrics).

The design and its acceptance criteria are in the plan
`archive/plans/strata-parent-process-parking-implementation-2026-11-15.md` (InfraHub repository). This page is the
runbook and the limits.

## Why

`--conversation-cache-mib` parks the conversations a request switches away from in the engine's own host RAM. With
data-parallel replicas every engine keeps its own copy, so two replicas need two budgets (2 × 20 GiB on the
production host) and a conversation parked on one card is invisible to the other: the pull scheduler moves a request
to the free card and that card reads the whole prompt again (measured on the disk handoff: 28K tokens, 20.9 s against
10.1 s with a peer restore).

The disk tier (`--conversation-cache-dir`, [CONVERSATION_CACHE_DISK.md](CONVERSATION_CACHE_DISK.md)) solves the
cross-replica part by writing the snapshot to an NVMe folder. This mode solves it in RAM instead: **one store in the
server process, one copy of each parked conversation for the whole server, no file anywhere.**

## What it is

The Python server owns a `ParentParkingStore`: the index, the global LRU, the RAM budget and one memfd per stage of
each parked conversation. Every engine talks to it over a `SOCK_SEQPACKET` socketpair it inherits at start
(`--conversation-parent-parking-fd`), and the store hands descriptors over with `SCM_RIGHTS`.

A park is a streamed write, not a heap image:

1. The engine measures each stage's session file (`session_file_measure`) from the active session's own K/V
   (`conversation_snapshot_sources`), so no full host copy of the K/V is made.
2. `PARK_BEGIN` asks the store for one memfd per stage, sized to the measured bytes; the store reserves the bytes in
   its budget first.
3. The engine writes format v1 straight into those memfds (`session_file_write_fd`) and closes its copies.
4. `PARK_COMMIT` makes the store seal them (`F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL`) and publish the
   entry. Until the commit the entry is invisible to every reader.

A restore is three short exchanges and one descriptor handover:

1. `MATCH_BEGIN` returns the resume lengths worth hashing (below the prompt's last token, pictures and control-vector
   mode already filtered).
2. The engine hashes its prompt once at exactly those lengths and sends them in `MATCH_END`; the store answers the
   deepest candidate.
3. `ACQUIRE` re-checks the entry's generation, reserves the restore's RAM and hands over the sealed memfds. The engine
   reads them with `session_file_read_fd` and restores the image the ordinary way.

The lease is held until the materialized image is destroyed, after the GPU restore, so the store's budget counts the
heap image while the next conversation parks.

### Superseded, and the prefix hashes

Publishing (COMMIT) drops every older snapshot of the same conversation whose deepest checkpoint the new one still
holds: that snapshot only adds the tail the client rewrote.  A checkpoint counts as held in two ways:

- it is one of the new snapshot's own points (verbatim, as before), or
- its `(tokens, hash)` pair is in the `prefix_hashes` the client sent: the outgoing ids' hash at exactly that prefix
  length (seed `STRIDS01`, the same function `MATCH_END` uses), with the same pictures below the checkpoint
  (`_images_below`).

The second way covers the case the verbatim check misses: a harness evicted the checkpoint from the `--prompt-cache`
ring between two publishes of one linear conversation, so the new snapshot no longer names it.  The pictures below the
checkpoint must match too, so two branches that parted before the checkpoint (or the same tokens with other images)
are never dropped.

`PARK_ADVICE` drives this in one pass.  Besides `parked_prefix_tokens` the answer carries `prefix_lengths`: the sorted
unique lengths of the deepest checkpoints the outgoing state does not hold (at most 4096, none past the outgoing live
point, pictures below matching).  The engine hashes its ids at exactly those lengths and sends them as the optional
`prefix_hashes` of `PARK_BEGIN` (and, when `min_gain` is on and an advised length is close enough to cancel the park,
of a second `PARK_ADVICE`, which re-counts `parked_prefix_tokens` before the measure).  When `prefix_lengths` is empty
no hashing happens at all.

The keys are optional on both sides and the order of deployment does not matter.  An old client never sends
`prefix_hashes` and ignores the extra answer key: it parks exactly as before.  An old store never answers
`prefix_lengths`: the client reads that as none and sends no hashes.  A malformed `prefix_hashes` list is a
`bad_request` that leaves the channel open — the keys are a supersede hint, not part of the snapshot's identity, so a
retried BEGIN with the same `publish_id` never fails on them.

A superseded entry with an active lease is **retired, not freed**: it leaves the index but its bytes stay charged
(`retired_lease_bytes`) until the last reader releases the lease.  This is deliberate and unlike eviction, which keeps
a leased entry in the index.

## What it replaces

The local heap cache is **replaced, not duplicated**: in this mode the engine creates its `ConversationCache` with a
zero budget, `park_source` delegates to the parent path before any heap capture, `conversations.put` and
`conversations.retain` never run, and `ConversationKvReuse` is not created. The K/V always comes from the active
session, streamed.

`STRATA_PARK_ASYNC` and `STRATA_PARK_VRAM_MIB` do not apply: the store's memfds are written synchronously. The engine
says so once, before `READY`, rather than ignoring them silently.

## RAM accounting

The store's budget (`ram_gib`) covers the whole parking subsystem, not the model and not the active K/V:

```
charged_bytes = committed stage bytes
              + pending stage bytes
              + bytes of entries evicted while a reader still holds them
              + restore reservations (what a reader's parsed image will take)
```

A restore reservation is `session_read_peak_bytes` of each stage, taken before the descriptors are handed over and
released only when the materialized image is gone. A victim with an active lease is never chosen, and an eviction plan
is modelled whole before anything is deleted: when the target cannot be reached, nothing is evicted and the park is
skipped. `budget_violations` in the metrics is the invariant `charged_bytes <= capacity_bytes` and is always 0.

`ram_gib: 40` therefore replaces the previous 2 × 20 GiB, including the transient restore images — at the price of the
store evicting more old entries before a restore than a per-replica budget would.

## Configuration

```json
"replicas": [{"gpus": [0], "priority": 10}, {"gpus": [1], "priority": 0}],
"replica_pull": true,
"replica_prefer_idle_s": 45,
"replica_parent_parking": {
  "enabled": true,
  "ram_gib": 40,
  "max_entries": 20,
  "min_tokens": 8192,
  "min_gain": 4096,
  "ipc_timeout_ms": 2000
}
```

| Key | Meaning |
|---|---|
| `enabled` | the block is off without it; `false` returns the old behaviour on the same binary |
| `ram_gib` | **required**: the one hard budget for the whole store, restore reservations included |
| `max_entries` | optional, default `10 × replicas` |
| `min_tokens` | optional, default 0: park only conversations of at least this many tokens (independent of `--conversation-cache-min-tokens`) |
| `min_gain` | default 4096: restore from the store only for at least this many more reused tokens than the live session or a batch slot offers |
| `ipc_timeout_ms` | default 2000: the bound on every control exchange, not on the K/V transfer |
| `replicas[].priority` / `replica_prefer` | optional per-replica routing priority (higher wins equal-cost ties and receives idle migrations); `replica_prefer` accepts a replica index or ordered list |
| `replica_prefer_idle_s` | optional idle threshold in seconds (`0..3600`): when a higher-priority free replica has been idle for at least this long, the pull scheduler sends `PARK` to an idle lower-priority replica holding a live conversation and moves the request onto the higher-priority replica (with safe fallback back to the source replica if `PARK` fails on capacity) |

With `replica_parent_parking` enabled, the router tracks each replica's current live conversation separately from conversations already displaced into the shared RAM store: once a conversation is parked in the shared store, all replicas tie on restore cost and higher-priority replicas win immediately without waiting for `replica_prefer_idle_s` or sending `PARK`.

The server writes the engine flags itself (`--conversation-parent-parking-fd`, `-owner`, `-min-tokens`, `-min-gain`,
`-timeout-ms`); they are not meant to be written by hand. The engine refuses a start it cannot
honour: no `--serve`, a `--conversation-cache-dir`/`--conversation-cache-peer-dir`, `--prompt-cache 0`, a missing
owner, or a layer split on one GPU.

The start is refused, never silently downgraded, when: there are fewer than two replicas, `replica_pull` is off, batch
slots or `--parallel` above 1 are on, the all-cards split is on, the platform is not Linux (or has no
`memfd_create`/seals/`SCM_RIGHTS`), `replica_handoff` is on at the same time, the engine args already carry a
`--conversation-cache-dir`/`--conversation-cache-peer-dir` or a parent flag, the block has an unknown key or a
non-integer value, a budget is zero or negative, the replicas' model/config/layout
fingerprints differ, or an explicit `--conversation-cache-mib 0`/`--conversation-cache-slots 0`/`--prompt-cache 0`
makes parking meaningless.

The one start line says what was built:

```text
[strata] replica parent parking: one RAM store for 2 replicas, 40 GiB, 20 entries, publish on park,
         snapshots survive replica restart but not server restart
```

## Metrics

`/metrics` and `/v1/status` gain `conversation_parent_parking` (only when the mode is on): the store's counters
(`capacity_bytes`, `max_entries`, `charged_bytes`, `entries`, `entry_bytes`, `pending`, `leases`, `restore_reserved_bytes`,
`retired_lease_bytes`, `published`, `deduplicated`, `reads`, `peer_reads`, `evictions`, `superseded`, `clears`,
`damaged`, `capacity_refusals`, `budget_violations`, `store_epoch`, `longest_tokens`) and one row per replica
(`state`, `published`, `reads`, `peer_reads`, `fallbacks`, `ipc_errors`). The global gauges are the store's own; they
are never summed over replicas.

The engine's own log lines carry the events (timings, token counts, reasons — never token ids or prompt text):

```text
strata serve: conversation parent: connected owner=r0; entries=E bytes=B epoch=Q
strata serve: conversation parent: published N tokens (on-park) in T ms, R GB/s; entries=E bytes=B dedup=D
strata serve: conversation parent: not published (on-park, N tokens): REASON [KIND]
strata serve: conversation parent: read N tokens (R reused, live|checkpoint, owner rX) in T ms, R GB/s
strata serve: conversation parent: cannot read (N tokens): REASON [KIND]
strata serve: conversation parent: borrowed N tokens (owner rX) in T ms
strata serve: conversation parent: cleared N entries (nonfinite); epoch=Q
strata serve: conversation parent: degraded: REASON; prompt fallback until this replica restarts
```

## Limits

- **No persistence across a server restart.** The snapshots are memfds owned by the server process; when it ends they
  are gone. A *replica* restart keeps them (the store outlives the engine), which is the case the mode exists for.
- **Not with the disk handoff.** `replica_handoff` and `replica_parent_parking` are two mechanisms for the same job and
  cannot be on at the same time; there is no automatic fallback from one to the other.
- **Not with batch slots.** The MVP parks one request per replica; `--parallel`/`--batch`/`--slots` above 1 are a start
  error.
- **Restore is not zero-copy.** The session file is materialized in the reader's RAM and then copied to the GPU, as the
  heap path does. The store's budget counts that transient image.
- **The async park switches do not apply** (`STRATA_PARK_ASYNC`, `STRATA_PARK_VRAM_MIB`): the memfd write is
  synchronous.
- **A degraded channel is not repaired in place.** A transport error makes that replica read prompts until it is
  restarted; the store's committed entries stay available to the other replicas.
- **`#879` clears the whole store**, not only this replica's entries: after non-finite logits a poisoned state may have
  migrated between replicas through a cross-restore.

## Rollback

One line: set `"enabled": false` in the `replica_parent_parking` block and restart the service. The engines then park
in their own heap again, exactly as before; nothing has to be migrated between storage media.

## Tests and what was run

| What | Result |
|---|---|
| `ctest` in the conversation build (`-DSTRATA_BUILD_CONVERSATION_TESTS=ON`) | 87/87 passed, 4 skipped (the skips are pre-existing) |
| `pytest serve/test_parent_parking.py` | 32 passed |
| `pytest serve/test_parent_parking_integration.py` (the real store against the real client) | 5 passed |
| `pytest serve/` (the whole server suite) | see the stage journal |
| CUDA sm_89 build (`strata`, `libstrata_kq.so`) | compiled |
| CUDA sm_70 build (container, CUDA 12.9.1, `-DSTRATA_EXPERIMENTAL_SM60=ON`) | compiled |
| HIP and SYCL builds | **not run**: no ROCm and no `icpx` on the development host |
| A GPU run of park/local restore/peer restore | **passed** on the live stand (two Tesla V100, Qwen3.8-Flash-Next): a 28 485-token conversation was parked in 400 ms (707 MB, 1.77 GB/s), restored on the SAME replica in 528 ms (28 181 of 28 485 tokens reused) and restored on the OTHER replica in 528 ms; the store's `peer_reads` counted it. CLEAR through `unload`, the fallback after it, SIGKILL of one replica (the committed entry stayed), `budget_violations=0`, and `diskstats` +0.01 MiB over a park+restore cycle were checked the same day |
| The three-arm A/B (baseline / disk A / parent) | **not run** |
| A byte-exact comparison with a reference (`--adapt-swaps 0`) | **not run**: the stand ran with the ordinary settings, where the baseline itself is not deterministic (#1937) |
| Borrow and an eviction while a restored image is held, on the live stand | **not run**: covered by the store's unit tests only |
| Three or more replicas on the live stand | **not run**: the stand had two cards |
