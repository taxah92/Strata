"""Data-parallel replicas: N independent engines, each on its own group of cards, behind one server.

A model that fits two cards (IQ3_S on two R9700s) runs better as two 2-card engines than as one 4-card layer split:
the split runs one pipeline whose stages wait on each other, two replicas run two pipelines.  The server keeps its one
HTTP front, its one Service and its one set of endpoints; `ReplicaEngine` stands in for the engine and spreads the
requests over the replicas.

Opt-in, in the run config:

    "replicas": 2                                     the "gpu" list cut into 2 equal groups ([0,1] and [2,3])
    "replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}]  the groups named; an entry may add "env" and "args"

or `serve/server.py --replicas 2`.  Nothing changes without it.

Routing.  Each request goes to the replica with the lowest cost, cost = prompt tokens the replica would have to read
again (it holds the start of a conversation in its cache) + LOAD_TOKENS for every request already running there.  A
conversation's next turn therefore goes back to its replica, until that replica is so much busier than another that
reading the prompt again is cheaper than waiting.  What each replica holds is tracked as hashes of 64-token blocks of
the prompts and answers it served (the engine itself checks the real prefix, as it always does).

Host RAM.  Every engine keeps its own copy of the experts in RAM (about 47 GiB for Qwen3.8-Flash-Next IQ3_S), so two
replicas need two copies and four need four.  "replica_shared_arena": true (or STRATA_REPLICA_SHARED_ARENA=1; opt-in)
backs the experts' arena of all replicas with ONE shared memory object (the engine's --shared-expert-arena, MAP_SHARED):
the experts are in RAM once, whatever the number of replicas.  On Linux the object is a memfd the server creates
(os.memfd_create): it is not a file on a mount, so the size of /dev/shm does not limit it; the engines open it as
/proc/<server pid>/fd/<n>.  On Windows it is a named section (CreateFileMapping) the engines open by name.  Without memfd (an old kernel, a sandbox that refuses it) a file in /dev/shm is used instead.
The pack hash in the header refuses another model; the first engine to start reads the experts into it and the others
wait for it and skip the read (a state word in the header: nobody / a pid reading / filled; an engine whose reader died
takes the read over); the server closes the memfd (or unlinks the file) when it ends - an engine that still maps it
keeps its pages, the kernel frees them with the last mapping - and a /dev/shm file whose server is gone is removed at
the next start.  See shared_arena_plan.

CPU cores.  On Linux the replicas get disjoint physical cores (serve/cpu_affinity.py, from PR #1436 by agorevski), so
the host threads of the engines (the expert workers) do not compete.  "replica_cpus": false turns it off.

Pull scheduler (opt-in).  The routing above gives a request a replica at once, even when every replica is busy: the
request is pinned to a card before that card has a free slot, and the other cards may sit idle while it waits.  With
"replica_pull": true (and no batch slots, which the MVP does not pull into) a request takes a free replica at once, or
else waits in ONE queue shared by every replica.  When a slot frees, the waiting request that would re-read the least
of its prompt goes into it (Router.matched); an older request goes first after "replica_starve_s".  Default off: the
route() and the order of the old path stay as they were.
"""
from __future__ import annotations

import collections
import contextlib
import copy
import math
import os
import sys
import threading
import time

BLOCK = 64                          # tokens per hashed block of a prompt
LOAD_TOKENS = 3000                  # the cost of one request already running on a replica, in prompt tokens
INDEX_MAX = 262144                  # block hashes kept (about 17M tokens of prompts)
HEAL_COOLDOWN_S = 60.0              # a replica that died is started again at most this often
PULL_STARVE_S = 60.0                # after this long the oldest waiting request goes first, whatever its cache says
PULL_POLL_S = 1.0                   # a waiting request checks its cancel, its deadline and the replicas this often
PULL_BEAT_S = 5.0                   # ... and sends a None heartbeat this often (<= the server's QUEUE_BEAT_S = 10)


class ReplicaConfigError(ValueError):
    pass


def gpu_groups(cfg: dict) -> list[list[int]]:
    """The cards of each replica, from the config's "replicas" (see the module doc); [] when there are none or one."""
    spec = cfg.get("replicas")
    if spec is None or spec is False:
        return []
    cards = _cards(cfg.get("gpu"))
    if isinstance(spec, bool) or (not isinstance(spec, (int, list))):
        raise ReplicaConfigError(f'"replicas" must be a number or a list of {{"gpus": [...]}}, not {spec!r}')
    if isinstance(spec, int):
        if spec <= 1:
            return []
        if not cards:
            raise ReplicaConfigError(f'"replicas": {spec} needs the "gpu" list to cut into groups (e.g. "gpu": [0, 1, 2, 3])')
        if len(cards) % spec:
            raise ReplicaConfigError(f'"replicas": {spec} does not divide the {len(cards)} cards {cards} evenly: name the '
                                     f'groups ("replicas": [{{"gpus": [0, 1]}}, ...]) or change the count')
        n = len(cards) // spec
        return [cards[i * n:(i + 1) * n] for i in range(spec)]
    groups = []
    for e in spec:
        g = e.get("gpus") if isinstance(e, dict) else e
        groups.append(_cards(g))
    if any(not g for g in groups):
        raise ReplicaConfigError('every replica needs its cards: {"gpus": [0, 1]}')
    flat = [c for g in groups for c in g]
    if len(set(flat)) != len(flat):
        raise ReplicaConfigError(f"a card is in two replicas: {groups}")
    if len(groups) < 2:
        return []
    return groups


def _cards(g) -> list[int]:
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    try:
        return [int(str(x).strip()) for x in items if str(x).strip() != ""]
    except ValueError:
        raise ReplicaConfigError(f"cards must be numbers, not {g!r}") from None


PullSettings = collections.namedtuple("PullSettings",
                                      "enabled starve_s queue_timeout_s max_queue prefer_idle_s priorities",
                                      defaults=(None, None, None, None))


class PullRequest:
    """One request in the one shared pending queue.  Its fields change only under ReplicaEngine.cv."""
    __slots__ = ("seq", "ids", "arrive", "cancel", "deadline", "replica", "matched", "migrate_from", "reason")

    def __init__(self):
        self.seq = 0                    # the queue's own order: tie-break, and "the oldest" is the first one
        self.ids = []                   # the prompt, as the request made it (the server does not change it)
        self.arrive = 0.0               # time.monotonic() when it was put in the queue
        self.cancel = None              # the request's own Event: a client that left sets it
        self.deadline = None            # time.monotonic() after which waiting is over (queue_timeout), or None
        self.replica = None             # the replica it was given; None while it still waits
        self.matched = None             # Router.matched(ids) of the last selection round (the reason in the view)
        self.migrate_from = None        # lower-priority free replica to PARK before starting on self.replica
        self.reason = None              # assignment reason ("prefer-idle", "starved", etc.)


def pull_choose(pending, free, now: float, starve_s: float, matched_of, *,
                priorities=None, idle_since=None, prefer_idle_s: float | None = None,
                live_matched_of=None):
    """(the waiting request, the replica, was it the starvation rule) or None: the pull order as one pure function,
    so it can be tested without threads.  `pending` is in arrival order, `free` the living replicas with
    inflight[k] == 0.  Cost = the prompt tokens the replica would read again (len(ids) - Router.matched(ids)[k]) -
    the first term of route()'s cost, as a free replica has no running request to pay LOAD_TOKENS for.

    When `prefer_idle_s` and `priorities` are provided, a free higher-priority replica that has been idle for at least
    `prefer_idle_s` seconds can pull a live conversation off a free lower-priority replica (`entry.migrate_from`)."""
    if not free or not pending:
        return None
    prio = (lambda k: (-float(priorities[k]), k)) if priorities is not None else (lambda k: (0.0, k))
    oldest = next(iter(pending.values()))
    if starve_s is not None and now - oldest.arrive >= starve_s:       # the oldest goes first, whatever its cache says
        got = matched_of(oldest.ids)
        oldest.matched = got
        k = min(free, key=lambda k: (len(oldest.ids) - got[k], prio(k)))
        oldest.migrate_from = None
        oldest.reason = "starved"
        return oldest, k, True
    can_prefer = prefer_idle_s is not None and idle_since is not None and priorities is not None
    idle_pref = [k for k in free if can_prefer and now - idle_since[k] >= prefer_idle_s]
    best = None
    for e in pending.values():                  # arrival order; a smaller seq wins an equal cost
        got = matched_of(e.ids)                 # the card's map is read AT THE CHOICE, not when the request arrived
        live_got = live_matched_of(e.ids) if live_matched_of is not None else got
        e.matched = got
        base = len(e.ids)
        best_src_for = {}
        if idle_pref:
            live_srcs = [s for s in free if live_got[s] > 0]
            if live_srcs:
                for kp in idle_pref:
                    cands = [s for s in live_srcs
                             if s != kp and float(priorities[kp]) > float(priorities[s]) and live_got[s] > got[kp]]
                    if cands:
                        best_src_for[kp] = max(cands, key=lambda s: (live_got[s], float(priorities[s]), -s))
        migrated_srcs = set(best_src_for.values())
        for k in free:
            if k in best_src_for:
                k_src = best_src_for[k]
                re_read = base - live_got[k_src]
                mig_rank = 1
                mig_src = k_src
            elif k in migrated_srcs:
                re_read = base - got[k]
                mig_rank = 2
                mig_src = None
            else:
                re_read = base - got[k]
                mig_rank = 0
                mig_src = None
            cand = (re_read, mig_rank, e.seq, prio(k))
            if best is None or cand < best[0]:
                best = (cand, e, k, mig_src)
    if best is None:
        return None
    _, chosen_e, chosen_k, chosen_src = best
    chosen_e.migrate_from = chosen_src
    chosen_e.reason = "prefer-idle" if chosen_src is not None else None
    return chosen_e, chosen_k, False


def _requested_slots(cfg: dict) -> int:
    """How many batch slots the run config asks each engine for, as serve/server.py reads it ("parallel", --batch,
    --slots; up to 1 means the engine takes one request at a time)."""
    requested = cfg.get("parallel")
    args = [str(a) for a in (cfg.get("args") or [])]
    if isinstance(cfg.get("replicas"), list):
        for e in cfg["replicas"]:
            if isinstance(e, dict):
                args += [str(a) for a in (e.get("args") or [])]
    if requested is None:
        for flag in ("--batch", "--slots"):
            if flag in args and args.index(flag) + 1 < len(args):
                try:
                    requested = int(args[args.index(flag) + 1])
                except ValueError:
                    pass
            for a in args:
                if a.startswith(flag + "="):
                    try:
                        requested = int(a.split("=", 1)[1])
                    except ValueError:
                        pass
    return requested if isinstance(requested, int) and not isinstance(requested, bool) and requested > 0 else 1


def prefer_idle_config(cfg: dict) -> float | None:
    """Opt-in idle threshold (seconds) after which a higher-priority free replica pulls a live conversation off
    a lower-priority free replica via PARK; None when disabled."""
    val = cfg.get("replica_prefer_idle_s")
    if val is None or val is False:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not 0.0 <= val <= 3600.0:
        raise ReplicaConfigError(f'"replica_prefer_idle_s" must be seconds in 0..3600, not {val!r}')
    return float(val)


def replica_priorities(cfg: dict, n: int | None = None) -> list[float]:
    """Per-replica routing priorities (higher = preferred when free).  Reads per-entry "priority" in "replicas",
    optional top-level "replica_prefer" (replica index or list of indices), or defaults r0 above others when
    "replica_prefer_idle_s" is configured without explicit priorities."""
    groups = gpu_groups(cfg)
    if n is None:
        n = len(groups)
    if n < 2:
        return [0.0] * max(0, n)
    spec = cfg.get("replicas")
    priorities = [0.0] * n
    has_explicit = False
    if isinstance(spec, list):
        for i, e in enumerate(spec[:n]):
            if isinstance(e, dict) and e.get("priority") is not None:
                p = e["priority"]
                if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
                    raise ReplicaConfigError(f'replicas[{i}]."priority" must be a finite number, not {p!r}')
                priorities[i] = float(p)
                has_explicit = True
    prefer = cfg.get("replica_prefer")
    if prefer is not None:
        if isinstance(prefer, bool) or not isinstance(prefer, (int, list)):
            raise ReplicaConfigError(f'"replica_prefer" must be a replica index or list of indices, not {prefer!r}')
        order = [prefer] if isinstance(prefer, int) else list(prefer)
        seen_p = set()
        for idx in order:
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < n:
                raise ReplicaConfigError(f'"replica_prefer" index {idx!r} out of range (0..{n - 1})')
            if idx in seen_p:
                raise ReplicaConfigError(f'"replica_prefer" has duplicate index {idx}')
            seen_p.add(idx)
        if not has_explicit and order:
            for rank, idx in enumerate(order):
                priorities[idx] = float(len(order) - rank)
            has_explicit = True
    if not has_explicit and prefer_idle_config(cfg) is not None:
        priorities[0] = 1.0
    return priorities


def pull_config(cfg: dict) -> PullSettings:
    """Opt-in pull scheduler: the run config's ("replica_pull", "replica_starve_s", "replica_queue_timeout_s",
    "replica_max_queue", "replica_prefer_idle_s", "replica_prefer" / replicas[].priority) as a PullSettings, for
    serve/server.py's start_replicas.  Raises ReplicaConfigError on a bad value, before any engine has been started."""
    want = cfg.get("replica_pull", False)
    if isinstance(want, bool) and not want:
        return PullSettings(False, PULL_STARVE_S)
    if not isinstance(want, bool):
        raise ReplicaConfigError(f'"replica_pull" must be true or false, not {want!r}')
    if not gpu_groups(cfg):
        raise ReplicaConfigError('"replica_pull" needs "replicas": 2 or more data-parallel engines')
    if cfg.get("split_serve"):
        raise ReplicaConfigError('"replica_pull" is for "replicas"; split serving has its own hand-over')
    if _requested_slots(cfg) > 1:
        raise ReplicaConfigError('"replica_pull" does not support batch slots ("parallel"/--batch/--slots > 1): '
                                 'run the replicas one request at a time')
    starve = cfg.get("replica_starve_s", PULL_STARVE_S)
    if isinstance(starve, bool) or not isinstance(starve, (int, float)) or not math.isfinite(starve) \
            or not 0.0 <= starve <= 3600.0:
        raise ReplicaConfigError(f'"replica_starve_s" must be seconds in 0..3600, not {starve!r}')
    timeout = cfg.get("replica_queue_timeout_s")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                or not math.isfinite(timeout) or not 0.0 < timeout <= 3600.0):
        raise ReplicaConfigError('"replica_queue_timeout_s" must be seconds in 0..3600, '
                                 f'not {timeout!r}')
    max_queue = cfg.get("replica_max_queue")
    if max_queue is not None and (isinstance(max_queue, bool) or not isinstance(max_queue, int)
                                  or not 0 <= max_queue <= 4096):
        raise ReplicaConfigError(f'"replica_max_queue" must be a whole number in 0..4096, not {max_queue!r}')
    prefer_idle = prefer_idle_config(cfg)
    priorities = replica_priorities(cfg)
    return PullSettings(True, float(starve), float(timeout) if timeout is not None else None, max_queue,
                        prefer_idle, priorities)


ARENA_PREFIX = "strata-replicas-"
_MEMFDS: dict = {}                   # the memfds this server created: path (/proc/<pid>/fd/<n>) -> fd
ARENA_MIN_RAM = 8 << 30              # a memfd arena: at least this much free RAM to try (the engine reserves the real size)
ARENA_MIN_FREE = 8 << 30             # /dev/shm must have at least this much free to try (the engine reserves the real size)
ARENA_INCOMPATIBLE = ("--mmap-experts", "--resident-experts", "--shared-expert-arena", "--resident-budget-gib")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_arenas(shm: str = "/dev/shm", uid: int | None = None) -> list[str]:
    """Remove the arena files of servers that no longer run (a killed server cannot unlink its own).  A file of a living
    server, or of another user, is never touched; a mapping an engine still holds stays valid either way."""
    if uid is None:
        uid = os.getuid() if hasattr(os, "getuid") else 0
    gone = []
    try:
        names = os.listdir(shm)
    except OSError:
        return gone
    head = f"{ARENA_PREFIX}{uid}-"
    for n in names:
        if not (n.startswith(head) and n.endswith(".bin")):
            continue
        try:
            pid = int(n[len(head):-4])
        except ValueError:
            continue
        if _pid_alive(pid):
            continue
        try:
            os.unlink(os.path.join(shm, n))
            gone.append(n)
        except OSError:
            pass
    return gone


def _free(path: str) -> int:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def _mem_available() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def shared_arena_plan(cfg: dict, n_replicas: int, *, shm: str = "/dev/shm", platform: str | None = None,
                      environ=None, free_bytes=None, use_memfd: bool | None = None) -> tuple[str | None, str]:
    """(the file for --shared-expert-arena or None, a line for the log).  Opt-in: "replica_shared_arena": true (or
    STRATA_REPLICA_SHARED_ARENA=1).  Only on Linux (the engine's MAP_SHARED backing), for Qwen's arena (not
    the mapped or budgeted RAM modes, which have no arena), and with room in /dev/shm.  The engine checks the real
    size and reserves it before it loads (a tmpfs that is too small is an error at start, not a bus error later)."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    want = cfg.get("replica_shared_arena")
    if want is None:
        want = str(environ.get("STRATA_REPLICA_SHARED_ARENA", "")).strip().lower() in ("1", "true", "on", "yes")
    if not want or n_replicas < 2:
        return None, ""
    if platform.startswith("win"):
        # a named pagefile-backed section the engines open by name (src/core/pinned.cu); it lives while any engine holds it
        name = f"Local{chr(92)}strata-replicas-{os.getpid()}"
        return f"win:{name}", (f"one shared expert arena for {n_replicas} replicas (a Windows section, {name}): "
                               "the experts are in RAM once")
    if not platform.startswith("linux"):
        return None, "shared expert arena: Linux and Windows only; each replica keeps its own copy of the experts in RAM"
    args = [str(a) for a in cfg.get("args") or []]
    if isinstance(cfg.get("replicas"), list):
        for e in cfg["replicas"]:
            if isinstance(e, dict):
                args += [str(a) for a in e.get("args") or []]
    if any(a in args for a in ARENA_INCOMPATIBLE):
        return None, "shared expert arena: this model/config has no private expert arena to share; each replica as before"
    if use_memfd is None:
        use_memfd = hasattr(os, "memfd_create") and str(cfg.get("replica_shared_arena")).lower() != "devshm"
    if use_memfd:
        avail = free_bytes if free_bytes is not None else _mem_available()
        if avail < ARENA_MIN_RAM:
            return None, (f"shared expert arena: only {avail / 2**30:.1f} GiB of RAM is free; each replica keeps its own "
                          "copy of the experts in RAM")
        try:
            fd = os.memfd_create("strata-replicas-arena", os.MFD_CLOEXEC)
        except OSError:
            fd = -1
        if fd >= 0:
            path = f"/proc/{os.getpid()}/fd/{fd}"
            _MEMFDS[path] = fd
            warn = ""
            try:                 # the arena once plus about 7 GB of private RAM per engine (measured); a warning, never a refusal
                from strata_control import replica_profile as _rp
                q = _rp.quant_of(cfg)
                if q:
                    need = _rp.ARENA_GB[q] + n_replicas * _rp.PER_REPLICA_GB
                    if avail / 2**30 < need + 4:
                        warn = (f" WARNING: about {need:.0f} GB are needed for {n_replicas} engines and {avail / 2**30:.0f} GB "
                                "are free: the last engine may not fit (fewer replicas, or the layer split, is safer)")
            except Exception:    # noqa: BLE001
                pass
            return path, (f"one shared expert arena for {n_replicas} replicas (memory object of the server, not limited by "
                          f"the size of /dev/shm): the experts are in RAM once.{warn}")
    if not (os.path.isdir(shm) and os.access(shm, os.W_OK)):
        return None, f"shared expert arena: {shm} is not usable; each replica keeps its own copy of the experts in RAM"
    free = free_bytes if free_bytes is not None else _free(shm)
    if free < ARENA_MIN_FREE:
        return None, (f"shared expert arena: {shm} has only {free / 2**30:.1f} GiB free (a tmpfs is half the RAM by "
                      "default); each replica keeps its own copy of the experts in RAM")
    sweep_stale_arenas(shm)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    path = os.path.join(shm, f"{ARENA_PREFIX}{uid}-{os.getpid()}.bin")
    return path, f"one shared expert arena for {n_replicas} replicas ({path}): the experts are in RAM once"


def remove_arena(path: str | None) -> None:
    """Close the memfd / unlink the arena file (the server's end, or a start that fell back): engines that map it keep
    their pages."""
    if not path or path.startswith("win:"):      # (a Windows section goes with its last handle)
        return
    fd = _MEMFDS.pop(path, None)
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _suffixed(path, i: int):
    if not isinstance(path, str) or not path:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.r{i}{ext}"


def replica_cfg(cfg: dict, i: int, gpus: list[int]) -> dict:
    """The config of replica i: its own cards, log and learned-profile file; the entry's "env" and "args" added."""
    c = copy.deepcopy(cfg)
    spec = cfg.get("replicas")
    entry = spec[i] if isinstance(spec, list) and isinstance(spec[i], dict) else {}
    c.pop("replicas", None)
    c["gpu"] = gpus if len(gpus) > 1 else gpus[0]
    c.pop("hip_ordinal", None)                 # a one-card config's Windows ordinal: not for a group
    c["log"] = _suffixed(cfg.get("log"), i)
    if cfg.get("expert_profile_save"):
        c["expert_profile_save"] = _suffixed(cfg["expert_profile_save"], i)
    if entry.get("env"):
        c["env"] = {**(c.get("env") or {}), **{str(k): str(v) for k, v in entry["env"].items()}}
    if entry.get("args"):
        c["args"] = list(c["args"]) + [str(x) for x in entry["args"]]
    return c


def cpu_sets(n: int, enabled: bool = True):
    """n disjoint sets of CPUs (whole physical cores, SMT siblings together), or None (off, not Linux, or the topology
    cannot be read: the engines then share the cores as they would without replicas)."""
    if not enabled or n < 2 or not hasattr(os, "sched_getaffinity"):
        return None
    try:
        from serve.cpu_affinity import partition_cpu_sets
        return partition_cpu_sets(n, set(os.sched_getaffinity(0)))
    except (ValueError, OSError, ImportError) as e:
        print(f"[strata] replicas: no separate CPU cores ({e})", flush=True)
        return None


class Router:
    """Which replica holds the start of a prompt: chained hashes of its 64-token blocks, each with the set of replicas
    that hold it (bits 0..n-1).  When `shared_parking=True`, bits 0..n-1 track each replica's current LIVE GPU KV only,
    while bit `n` (`SHARED_BIT`) tracks prefixes displaced into the shared ParentParkingStore (available to all)."""

    def __init__(self, n: int, cap: int = INDEX_MAX, *, shared_parking: bool = False, shared_min_tokens: int = 0):
        self.n, self.cap = n, cap
        self.shared_parking = bool(shared_parking)
        self.shared_min_tokens = max(0, int(shared_min_tokens))
        self.blocks: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.live_chains: list[list[int]] = [[] for _ in range(n)]
        self.lock = threading.Lock()

    @staticmethod
    def chain(ids) -> list[int]:
        out, h = [], 0
        for k in range(len(ids) // BLOCK):
            h = hash((h, tuple(ids[k * BLOCK:(k + 1) * BLOCK])))
            out.append(h)
        return out

    def _matched_locked(self, ids) -> tuple[list[int], int]:
        chains = self.chain(ids)
        live = [0] * self.n
        shared = 0
        alive_live = (1 << self.n) - 1
        shared_bit = 1 << self.n
        shared_alive = self.shared_parking
        for k, h in enumerate(chains):
            m = self.blocks.get(h, 0)
            m_live = m & alive_live
            if m_live:
                alive_live = m_live
                for r in range(self.n):
                    if m_live >> r & 1:
                        live[r] = (k + 1) * BLOCK
            else:
                alive_live = 0
            if shared_alive and (m & shared_bit):
                shared = (k + 1) * BLOCK
            else:
                shared_alive = False
            if not alive_live and not shared_alive:
                break
        return live, shared

    def live_matched(self, ids) -> list[int]:
        """Per replica: how many leading tokens of `ids` it currently holds in its live cache."""
        with self.lock:
            live, _ = self._matched_locked(ids)
            return live

    def matched(self, ids) -> list[int]:
        """Per replica: how many leading tokens of `ids` it holds or can restore from shared parent parking."""
        with self.lock:
            live, shared = self._matched_locked(ids)
        if not self.shared_parking or shared <= 0:
            return live
        shared_score = max(0, shared - BLOCK)
        return [max(x, shared_score) for x in live]

    def note(self, r: int, ids) -> None:
        cur = self.chain(ids)
        with self.lock:
            if self.shared_parking:
                prev = self.live_chains[r]
                if prev and (len(prev) > len(cur) or cur[:len(prev)] != prev):
                    cur_set = set(cur)
                    keep_bit = ~(1 << r)
                    share = (1 << self.n) if (len(prev) * BLOCK >= self.shared_min_tokens) else 0
                    for h in prev:
                        if h not in self.blocks:
                            continue
                        m = self.blocks[h]
                        if h not in cur_set:
                            m &= keep_bit
                        m |= share
                        if m:
                            self.blocks[h] = m
                        else:
                            del self.blocks[h]
                self.live_chains[r] = cur
            r_bit = 1 << r
            for h in cur:
                self.blocks[h] = self.blocks.pop(h, 0) | r_bit
            while len(self.blocks) > self.cap:
                self.blocks.popitem(last=False)

    def demote_live(self, r: int, *, parked: bool = True) -> None:
        """Replica r handed off its live conversation (e.g. via PARK): drop its live bit and optionally mark
        those blocks as available in the shared store."""
        with self.lock:
            prev = self.live_chains[r]
            self.live_chains[r] = []
            if not prev:
                return
            keep_bit = ~(1 << r)
            share = (1 << self.n) if (self.shared_parking and parked and len(prev) * BLOCK >= self.shared_min_tokens) else 0
            for h in prev:
                if h not in self.blocks:
                    continue
                m = (self.blocks[h] & keep_bit) | share
                if m:
                    self.blocks[h] = m
                else:
                    del self.blocks[h]

    def forget(self, r: int) -> None:
        """Replica r lost its cache (it restarted)."""
        with self.lock:
            self.live_chains[r] = []
            bit = ~(1 << r)
            for h in [h for h, m in self.blocks.items() if m >> r & 1]:
                m = self.blocks[h] & bit
                if m:
                    self.blocks[h] = m
                else:
                    del self.blocks[h]

    def forget_shared(self) -> None:
        """Clear shared parent-parking bits (on store unload or #879 nonfinite clear)."""
        if not self.shared_parking:
            return
        with self.lock:
            bit = ~(1 << self.n)
            for h in [h for h, m in self.blocks.items() if m >> self.n & 1]:
                m = self.blocks[h] & bit
                if m:
                    self.blocks[h] = m
                else:
                    del self.blocks[h]


class ReplicaEngine:
    """Stands in for one engine (the interface Service uses), over several."""

    NOT_HERE = {"session_file", "ctl", "slot_cv"}       # not offered: one conversation file / control line
    PROPAGATE = {"silence_s", "gpu_busy"}

    def __init__(self, engines: list, load_tokens: int = LOAD_TOKENS, *, pull: bool = False,
                 starve_s: float = PULL_STARVE_S, poll_s: float = PULL_POLL_S, beat_s: float = PULL_BEAT_S,
                 queue_timeout_s: float | None = None, max_queue: int | None = None,
                 prefer_idle_s: float | None = None, priorities: list[float] | None = None,
                 parent_parking=None):
        if len(engines) < 2:
            raise ValueError("ReplicaEngine needs two or more engines")
        if not isinstance(pull, bool):
            raise ReplicaConfigError(f'"replica_pull" must be true or false, not {pull!r}')
        if prefer_idle_s is not None:
            if isinstance(prefer_idle_s, bool) or not isinstance(prefer_idle_s, (int, float)) \
                    or not math.isfinite(prefer_idle_s) or not 0.0 <= prefer_idle_s <= 3600.0:
                raise ReplicaConfigError(f'"replica_prefer_idle_s" must be seconds in 0..3600, not {prefer_idle_s!r}')
        if priorities is not None:
            if len(priorities) != len(engines):
                raise ReplicaConfigError(
                    f"priorities length ({len(priorities)}) does not match replicas ({len(engines)})")
            for i, p in enumerate(priorities):
                if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
                    raise ReplicaConfigError(f'replicas[{i}]."priority" must be a finite number, not {p!r}')
        if pull:                                    # the MVP pulls one request into one engine, never into a slot
            slots = [int(getattr(e, "batch", 0) or 0) for e in engines]
            if any(slots):
                k = next(k for k, n in enumerate(slots) if n)
                raise ReplicaConfigError(
                    f'"replica_pull" does not support batch slots (replica {k} reports {slots[k]}): run the '
                    'replicas one request at a time ("parallel": 1, no --batch/--slots)')
            if isinstance(starve_s, bool) or not isinstance(starve_s, (int, float)) \
                    or not math.isfinite(starve_s) or not 0.0 <= starve_s <= 3600.0:
                raise ReplicaConfigError(f'"replica_starve_s" must be seconds in 0..3600, not {starve_s!r}')
            for name, value in (("replica_pull_poll_s", poll_s), ("replica_pull_beat_s", beat_s)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not math.isfinite(value) or value <= 0:
                    raise ReplicaConfigError(f'{name} must be a positive number of seconds, not {value!r}')
            if queue_timeout_s is not None and (isinstance(queue_timeout_s, bool)
                                                or not isinstance(queue_timeout_s, (int, float))
                                                or not math.isfinite(queue_timeout_s) or queue_timeout_s <= 0):
                raise ReplicaConfigError('"replica_queue_timeout_s" must be a positive number of seconds, '
                                         f"not {queue_timeout_s!r}")
            if max_queue is not None and (isinstance(max_queue, bool) or not isinstance(max_queue, int)
                                          or not 0 <= max_queue <= 4096):
                raise ReplicaConfigError(f'"replica_max_queue" must be a whole number in 0..4096, not {max_queue!r}')
        d = self.__dict__
        d["engines"] = list(engines)
        park_cfg = getattr(parent_parking, "_config", None)
        shared_min = getattr(park_cfg, "min_tokens", 0) or 0
        d["router"] = Router(len(engines), shared_parking=parent_parking is not None, shared_min_tokens=shared_min)
        d["load_tokens"] = load_tokens
        d["inflight"] = [0] * len(engines)
        d["run_locks"] = [threading.Lock() for _ in engines]    # an engine without batch slots: one request at a time
        d["lock"] = threading.Lock()
        d["cv"] = threading.Condition(d["lock"])    # the scheduler's condition: the same lock, so inflight and the
        d["pull_enabled"] = pull                    # pending queue always agree (route()/finally use self.lock too)
        d["pull_starve_s"] = float(starve_s)
        d["pull_poll_s"] = float(poll_s)
        d["pull_beat_s"] = float(beat_s)
        d["queue_timeout_s"] = float(queue_timeout_s) if queue_timeout_s is not None else None
        d["max_queue"] = max_queue
        d["prefer_idle_s"] = float(prefer_idle_s) if prefer_idle_s is not None else None
        if priorities is not None:
            d["priorities"] = [float(p) for p in priorities]
        elif prefer_idle_s is not None:
            d["priorities"] = [1.0 if k == 0 else 0.0 for k in range(len(engines))]
        else:
            d["priorities"] = [0.0] * len(engines)
        d["idle_since"] = [time.monotonic()] * len(engines)
        d["pull_seq"] = 0
        d["pending"] = collections.OrderedDict()    # seq -> PullRequest, order = arrival: the oldest is the first
        d["pull_stats"] = {"submitted": 0, "assigned": 0, "starved": 0, "cancelled": 0, "closed": 0,
                           "timeouts": 0, "no_replica": 0, "rejected": 0, "migrated": 0, "migrate_fallbacks": 0}
        d["last_pick"] = None                   # {"replica", "reason", "at"} of the last assignment (for /metrics)
        d["tl"] = threading.local()
        d["healing"] = {}
        d["heal_at"] = [0.0] * len(engines)
        d["routed"] = [0] * len(engines)
        d["gen_seen"] = [getattr(e, "gen", 0) for e in engines]
        d["paused"] = {}                            # replica index -> owner thread ident while paused
        d["pause_depth"] = {}                       # replica index -> re-entrant pause depth
        d["reserves"] = [None] * len(engines)       # last VRAM reserve asked per replica (None = start's)
        d["reserves_asked"] = [False] * len(engines)
        # the server's one parent parking store (serve/parent_parking.py, plan §12.3): in __dict__, never through
        # __getattr__, so the store is this object's own and not one engine's.  None when the mode is off.
        d["parent_parking"] = parent_parking

    # ---- attribute plumbing
    @property
    def current(self):
        return getattr(self.tl, "cur", None) or self.engines[0]

    def __getattr__(self, name):
        if name in self.NOT_HERE or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.current, name)

    def __setattr__(self, name, value):
        if name in self.PROPAGATE:
            for e in self.engines:
                setattr(e, name, value)
        else:
            self.__dict__[name] = value

    @property
    def last(self):
        """The DONE figures of the calling thread's own request (#2892).  An unbatched engine has one shared `last`
        that the next request overwrites the moment the replica's lock is free, so generate() copies the figures of
        this caller's DONE while it still owns the replica; until the thread has had one, the engine's own object."""
        tl = self.tl
        if not hasattr(tl, "last"):
            tl.last = self.current.last
        return tl.last

    @last.setter
    def last(self, value):
        self.current.last = value
        self.tl.last = value

    @property
    def batch(self) -> int:
        """Parallel slots, in total: the Service lets this many requests run at once (it holds no FIFO).  An engine
        without slots counts one."""
        return sum(max(1, int(getattr(e, "batch", 0) or 0)) for e in self.engines)

    @property
    def waiting(self) -> int:
        return sum(int(getattr(e, "waiting", 0) or 0) for e in self.engines)

    @property
    def max_context(self) -> int:
        v = [e.max_context for e in self.engines if getattr(e, "max_context", 0) and e.alive()]
        return min(v) if v else 0

    @property
    def known_ctx(self) -> int:
        v = [getattr(e, "known_ctx", 0) for e in self.engines]
        return min(x for x in v if x) if any(v) else 0

    @property
    def unloaded(self) -> bool:
        return all(getattr(e, "unloaded", False) for e in self.engines)

    @property
    def starting(self) -> bool:
        return not self.alive() and any(getattr(e, "starting", False) for e in self.engines)

    @property
    def info(self) -> dict:
        i = dict(self.engines[0].info)
        i["replicas"] = len(self.engines)
        i["replicas_alive"] = sum(1 for e in self.engines if e.alive())
        return i

    @info.setter
    def info(self, value):
        pass

    @property
    def spawn(self):
        return self.engines[0].spawn

    @property
    def log_path(self):
        return self.engines[0].log_path

    @property
    def progress(self):
        if getattr(self.tl, "pending", False):         # a request still waiting for a replica shows no other's progress
            return None
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.progress
        return next((e.progress for e in self.engines if getattr(e, "progress", None)), None)

    @property
    def prefill_tok_s_mean(self):
        if getattr(self.tl, "pending", False):
            return None
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.prefill_tok_s_mean
        return next((e.prefill_tok_s_mean for e in self.engines if getattr(e, "prefill_tok_s_mean", None)), None)

    def alive(self) -> bool:
        return any(e.alive() for e in self.engines)

    def exit_code(self):
        return next((c for c in (e.exit_code() for e in self.engines if not e.alive()) if c is not None), None)

    def death_note(self) -> str:
        return next((e.death_note() for e in self.engines if not e.alive()), "")

    def slots_view(self) -> list[dict]:
        out, base = [], 0
        for k, e in enumerate(self.engines):
            n = int(getattr(e, "batch", 0) or 0)
            view = e.slots_view() if n and hasattr(e, "slots_view") else \
                [{"slot": 0, "state": "busy" if self.inflight[k] else "idle", "held_tokens": 0}]
            for s in view:
                out.append({**s, "slot": base + s["slot"], "replica": k})
            base += max(1, n)
        return out

    def replica_view(self) -> list[dict]:
        """For /metrics: each replica's state, requests running, and requests routed to it so far.  With the pull
        scheduler on, each row also carries the one shared queue (depth, the oldest request's age, the replica and the
        reason of the last assignment): the server's /metrics needs no new field to show it.  With the server's parent
        parking store on, each row also carries its channel's state (connected, degraded or closed)."""
        rows = [{"replica": k, "alive": e.alive(), "running": self.inflight[k], "routed": self.routed[k],
                 "batch": int(getattr(e, "batch", 0) or 0)} for k, e in enumerate(self.engines)]
        store = self.__dict__.get("parent_parking")
        if store is not None:
            for row in rows:
                row["parent_parking"] = store.replica_state(f"r{row['replica']}")
        if self.pull_enabled:
            queue = self.pull_view()
            for row in rows:
                row["queue"] = {"depth": queue["pending"], "oldest_s": queue["oldest_pending_s"],
                                "picked": queue["picked"], "reason": queue["reason"],
                                "max": queue["max_queue"], "timeout_s": queue["queue_timeout_s"],
                                "migrated": queue["migrated"], "migrate_fallbacks": queue["migrate_fallbacks"],
                                "prefer_idle_s": queue["prefer_idle_s"]}
        return rows

    def pull_view(self) -> dict:
        """The one pending queue for /metrics (it is global: the queue is shared by every replica, so nothing here is
        read from a single engine's log).  oldest_pending_s is None when nothing waits (null, not 0)."""
        now = time.monotonic()
        with self.cv:
            oldest = next(iter(self.pending.values()), None)
            stats = dict(self.pull_stats)
            depth = len(self.pending)
            pick = self.last_pick
        return {"enabled": self.pull_enabled, "pending": depth,
                "oldest_pending_s": round(now - oldest.arrive, 3) if oldest is not None else None,
                "starve_s": self.pull_starve_s, "poll_s": self.pull_poll_s, "beat_s": self.pull_beat_s,
                "queue_timeout_s": self.queue_timeout_s, "max_queue": self.max_queue,
                "prefer_idle_s": self.prefer_idle_s,
                "picked": pick["replica"] if pick else None, "reason": pick["reason"] if pick else None,
                **stats}

    def vram_view(self) -> list[dict]:
        """Per-replica VRAM and expert-cache readings for /v1/status and /metrics."""
        rows = []
        for k, e in enumerate(self.engines):
            info = getattr(e, "info", None) or {}
            vram = dict(info.get("vram") or {})
            for key in ("expert_slots", "expert_cache_mib", "expert_cache_full_mib", "vram_free_mib"):
                if key not in vram and key in info:
                    vram[key] = info[key]
            row = {"replica": k, "alive": e.alive(), **vram,
                   "elastic": bool(info.get("vram_elastic"))}
            if "reserve_mib" not in row and self.reserves_asked[k]:
                row["reserve_mib"] = self.reserves[k]
            rows.append(row)
        return rows

    def vram_summary(self) -> dict:
        """Aggregated VRAM summary for /v1/status across replicas, preserving per-replica details."""
        rows = self.vram_view()
        out: dict = {
            "elastic": bool(rows) and all(r.get("elastic") for r in rows),
            "replicas": rows,
        }
        for key in ("expert_slots", "expert_cache_mib", "expert_cache_full_mib", "vram_free_mib"):
            vals = [r[key] for r in rows if isinstance(r.get(key), int)]
            if len(vals) == len(rows):
                out[key] = sum(vals)
        return out

    def normalize_replicas(self, replica=None) -> list[int]:
        """Validate and expand a replica selector (None -> all, int -> [int], list/tuple -> unique ordered list)."""
        n = len(self.engines)
        if replica is None:
            return list(range(n))
        if isinstance(replica, bool):
            raise ValueError("replica: an integer index or a non-empty list of indices")
        if isinstance(replica, int):
            items = [replica]
        elif isinstance(replica, (list, tuple)):
            if not replica:
                raise ValueError("replica: list must not be empty")
            items = list(replica)
        else:
            raise ValueError("replica: an integer index or a non-empty list of indices")
        for x in items:
            if isinstance(x, bool) or not isinstance(x, int) or not (0 <= x < n):
                raise ValueError(f"replica {x!r} out of range (0..{n - 1})")
        return list(dict.fromkeys(items))

    @contextlib.contextmanager
    def pause(self, replicas=None, wait_s: float = 300.0):
        """Pause routing to `replicas` and wait until any running request on them finishes.

        Re-entrant for the same thread (e.g. a GPU vision turn holding pause on replica 0 while shrinking its
        expert cache).  Only parallel=1 replicas (no batch slots) can be paused cleanly without slot arbitration.
        """
        from serve.server import ModelBusy
        targets = self.normalize_replicas(replicas)
        for k in targets:
            slots = int(getattr(self.engines[k], "batch", 0) or 0)
            if slots > 0:
                raise ValueError(
                    f"per-replica pause/resize is not supported with batch slots (replica {k} reports {slots}): "
                    'use "parallel": 1 replicas'
                )
        deadline = time.monotonic() + max(0.0, float(wait_s))
        tid = threading.get_ident()
        claimed: list[int] = []
        try:
            with self.cv:
                while any(self.paused.get(k) not in (None, tid) for k in targets):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise ModelBusy("another pause or VRAM resize is already in progress on the target replica")
                    self.cv.wait(timeout=left)
                for k in targets:
                    self.paused[k] = tid
                    self.pause_depth[k] = self.pause_depth.get(k, 0) + 1
                    claimed.append(k)
                while any(self.inflight[k] > 0 for k in targets):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        busy = [k for k in targets if self.inflight[k] > 0]
                        raise ModelBusy(
                            f"replica{'s' if len(busy) > 1 else ''} {', '.join(str(k) for k in busy)} "
                            "still busy; try again when the running request finishes"
                        )
                    self.cv.wait(timeout=left)
            yield targets
        finally:
            if claimed:
                with self.cv:
                    for k in claimed:
                        depth = self.pause_depth.get(k, 1) - 1
                        if depth <= 0:
                            self.pause_depth.pop(k, None)
                            self.paused.pop(k, None)
                        else:
                            self.pause_depth[k] = depth
                    if self.pull_enabled:
                        self._pull_pump_locked(time.monotonic())
                    self.cv.notify_all()

    def _reapply_reserve(self, k: int) -> None:
        if not self.reserves_asked[k] or self.reserves[k] is None:
            return
        e = self.engines[k]
        if not hasattr(e, "vram") or not e.alive():
            return
        try:
            e.vram(self.reserves[k])
        except Exception as err:                 # noqa: BLE001
            print(f"[strata] replica {k}: the VRAM reserve ({self.reserves[k]} MiB) was not applied: {err}", flush=True)

    def vram(self, reserve_mib: int | None | dict[int, int | None], replica=None, wait_s: float = 300.0) -> dict:
        """Apply a VRAM reserve to selected replicas (None = all), pausing and draining them first.

        If one replica fails after earlier ones succeeded, already-resized replicas are rolled back to their
        previous reserves before raising.
        """
        from serve.server import EngineDied
        targets = self.normalize_replicas(replica)
        for k in targets:
            if not hasattr(self.engines[k], "vram"):
                raise ValueError(f"replica {k} cannot resize its VRAM use")

        def target_reserve(k: int) -> int | None:
            return reserve_mib[k] if isinstance(reserve_mib, dict) else reserve_mib

        with self.pause(targets, wait_s=wait_s):
            if not any(e.alive() for e in self.engines):
                rows = []
                for k in targets:
                    r_k = target_reserve(k)
                    self.reserves[k] = r_k
                    self.reserves_asked[k] = True
                    rows.append({"replica": k, "status": "not loaded", "reserve_mib": r_k,
                                 "note": "applied when the model loads"})
                if len(targets) == 1:
                    return {**rows[0], "replicas": rows}
                return {"status": "not loaded",
                        "reserve_mib": None if isinstance(reserve_mib, dict) else reserve_mib,
                        "replicas": rows}

            rows = []
            applied: list[tuple[int, int | None, bool]] = []
            try:
                for k in targets:
                    r_k = target_reserve(k)
                    e = self.engines[k]
                    if not e.alive():
                        if getattr(e, "unloaded", False):
                            prev = (k, self.reserves[k], self.reserves_asked[k])
                            self.reserves[k] = r_k
                            self.reserves_asked[k] = True
                            applied.append(prev)
                            rows.append({"replica": k, "status": "not loaded", "reserve_mib": r_k,
                                         "note": "applied when the model loads"})
                            continue
                        raise EngineDied(f"replica {k} is not running")
                    prev = (k, self.reserves[k], self.reserves_asked[k])
                    out = e.vram(r_k)
                    self.reserves[k] = r_k
                    self.reserves_asked[k] = True
                    applied.append(prev)
                    rows.append({"replica": k, "status": "ok", **out})
            except Exception:
                for pk, prev_res, prev_asked in reversed(applied):
                    try:
                        if self.engines[pk].alive():
                            self.engines[pk].vram(prev_res if prev_asked else None)
                    except Exception as rb_err:  # noqa: BLE001
                        print(f"[strata] replica {pk}: rollback of VRAM reserve failed: {rb_err}", flush=True)
                    self.reserves[pk] = prev_res
                    self.reserves_asked[pk] = prev_asked
                raise

        status = "ok" if any(r.get("status") == "ok" for r in rows) else "not loaded"
        if len(targets) == 1:
            return {**rows[0], "replicas": rows}
        return {"status": status, "replicas": rows}

    # ---- life cycle
    def restart(self, tries: int = 3):
        dead = [k for k, e in enumerate(self.engines) if not e.alive()]
        errors: dict[int, BaseException] = {}

        def one(k):
            try:
                self.engines[k].restart(tries)
                self.router.forget(k)
                self._reapply_reserve(k)
                with self.cv:
                    self.idle_since[k] = time.monotonic()
            except BaseException as e:           # noqa: BLE001
                errors[k] = e
        ts = [threading.Thread(target=one, args=(k,), daemon=True) for k in dead]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errors and not self.alive():
            raise next(iter(errors.values()))
        for k, e in errors.items():
            print(f"[strata] replica {k} did not start ({e}); the others serve", flush=True)

    def unload(self):
        try:
            for e in self.engines:
                e.unload()
            for k in range(len(self.engines)):
                self.router.forget(k)
            self.router.forget_shared()
        finally:
            store = self.__dict__.get("parent_parking")
            if store is not None:                   # plan §12.3: the engines are down, so the store's RAM goes too
                store.clear("unload")               # (even when an engine would not unload: the RAM is the point)

    def close(self):
        err = None
        for e in self.engines:
            try:
                e.close()
            except BaseException as x:           # noqa: BLE001
                err = err or x
        remove_arena(self.__dict__.get("arena_file"))   # (an engine that still maps it keeps its pages)
        store = self.__dict__.get("parent_parking")
        if store is not None:                       # plan §12.3: the engines first, then the broker and its memfds
            try:
                store.close()
            except BaseException as x:           # noqa: BLE001
                err = err or x
        if err:
            raise err

    # ---- routing
    def route(self, ids) -> int:
        """The replica for this prompt (and its running count raised): lowest cost among the living unpaused ones."""
        while True:
            got = self.router.matched(ids)
            with self.cv:
                alive = [k for k, e in enumerate(self.engines) if e.alive()]
                if not alive:
                    return -1
                avail = [k for k in alive if k not in self.paused]
                if not avail:
                    self.cv.wait(timeout=0.5)
                    continue
                cost = {k: (len(ids) - got[k]) + self.load_tokens * self.inflight[k] for k in avail}
                # the least loaded wins a tie of cost (equal cards, nothing cached): spread, do not pile on replica 0
                k = min(avail, key=lambda k: (cost[k], self.inflight[k], -self.priorities[k], self.routed[k]))
                self.inflight[k] += 1
                self.routed[k] += 1
            if len(alive) < len(self.engines):
                self.heal()
            return k

    def heal(self) -> None:
        """Start a replica that died again, in the background (the others keep serving)."""
        now = time.monotonic()
        for k, e in enumerate(self.engines):
            if e.alive() or getattr(e, "unloaded", False) or getattr(e, "starting", False) or self.healing.get(k):
                continue
            if now < self.heal_at[k]:
                continue
            self.heal_at[k] = now + HEAL_COOLDOWN_S
            self.healing[k] = True

            def run(k=k):
                try:
                    print(f"[strata] replica {k} had stopped; starting it again", flush=True)
                    self.engines[k].restart()
                    self.router.forget(k)
                    self._reapply_reserve(k)
                    with self.cv:
                        self.idle_since[k] = time.monotonic()
                    print(f"[strata] replica {k} is running again", flush=True)
                except BaseException as x:       # noqa: BLE001
                    print(f"[strata] replica {k} did not start again: {x}", flush=True)
                finally:
                    self.healing[k] = False
                    with self.cv:
                        self.cv.notify_all()
            threading.Thread(target=run, daemon=True).start()

    # ---- pull scheduler (opt-in: "replica_pull": true; see pull_config and the module doc)
    def _pull_wait(self, ids, cancel):
        """The replica for this prompt under the pull scheduler, or None heartbeats while the request waits - a
        generator, so the heartbeat goes out on the request's own turn and the server's keep-alive is unchanged.
        A free replica takes it at once (the pump); otherwise it waits in the one shared queue, and every slot that
        frees is given to the waiting request that would re-read the least of its prompt (or to the oldest one, after
        replica_starve_s).  Returns `(replica, migrate_from)`, or `(-1, None)` when the client left (its queue entry
        removed).  The assignment and inflight + 1 are one step under self.cv; generation happens outside it."""
        from serve.server import EngineDied, QueueWaitExpired
        now = time.monotonic()
        entry = PullRequest()
        with self.cv:
            self.pull_seq += 1
            entry.seq, entry.ids, entry.arrive = self.pull_seq, ids, now
            entry.cancel = cancel
            entry.deadline = getattr(cancel, "_strata_queue_deadline", None) if cancel is not None else None
            if self.queue_timeout_s is not None:
                own = now + self.queue_timeout_s
                entry.deadline = own if entry.deadline is None else min(entry.deadline, own)
            self.pending[entry.seq] = entry
            self.pull_stats["submitted"] += 1
            self.tl.pending = True
            self._pull_pump_locked(now)             # a free replica takes the request at once
            if entry.replica is None and self.max_queue is not None and len(self.pending) > self.max_queue:
                self._pull_drop_locked(entry, "rejected")       # (a free card still serves: this caps waiting only)
                self.tl.pending = False
                raise QueueWaitExpired(f"the replica queue is full (replica_max_queue={self.max_queue}); retry later")
            self.cv.notify_all()
        heal = self._pull_some_dead()
        beat = time.monotonic()
        taken = False
        mig_src = None
        try:
            while True:
                with self.cv:
                    if entry.replica is not None:   # assigned: this request owns a slot from here
                        k = entry.replica
                        mig_src = entry.migrate_from
                        break
                    now = time.monotonic()
                    if cancel is not None and cancel.is_set():
                        self._pull_drop_locked(entry, "cancelled")
                        return -1, None
                    if entry.deadline is not None and now >= entry.deadline:
                        self._pull_drop_locked(entry, "timeouts")
                        raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
                    if not any(e.alive() for e in self.engines) and not self._pull_starting():
                        # nothing is alive and nothing is being started: there is nothing to wait for
                        self._pull_drop_locked(entry, "no_replica")
                        raise EngineDied("no replica is running; this request was not sent")
                    self._pull_pump_locked(now)     # a healed replica, or a request that has waited long enough
                    if entry.replica is not None:
                        k = entry.replica
                        mig_src = entry.migrate_from
                        break
                    left = None if entry.deadline is None else max(0.0, entry.deadline - now)
                    self.cv.wait(timeout=min(self.pull_poll_s, left) if left is not None else self.pull_poll_s)
                if heal:                            # outside the lock: heal() itself decides, and never blocks
                    self.heal()
                    heal = self._pull_some_dead()
                if time.monotonic() - beat >= self.pull_beat_s:
                    beat = time.monotonic()
                    yield None                      # the None the server turns into a keep-alive
            taken = True
            return k, mig_src
        finally:
            with self.cv:
                if not taken:
                    if entry.replica is not None:   # a slot was given but this waiter leaves before using it
                        self.inflight[entry.replica] -= 1
                        if self.inflight[entry.replica] == 0:
                            self.idle_since[entry.replica] = time.monotonic()
                        if entry.migrate_from is not None:
                            self.inflight[entry.migrate_from] -= 1
                            if self.inflight[entry.migrate_from] == 0:
                                self.idle_since[entry.migrate_from] = time.monotonic()
                            entry.migrate_from = None
                        self.pull_stats["closed"] += 1
                        entry.replica = None
                        self._pull_pump_locked(time.monotonic())
                    else:
                        self._pull_drop_locked(entry, "closed")
                self.cv.notify_all()
            self.tl.pending = False

    def _pull_pump_locked(self, now: float) -> None:
        """Give a free replica to a waiting request, as many times as it can, under self.cv.  Called when a request
        joins the queue, when a slot frees, and on every poll tick of every waiter (so a healed replica is used
        without a new request)."""
        if not self.pending:
            return
        free = [k for k, e in enumerate(self.engines)
                if e.alive() and not self.inflight[k] and k not in self.paused]
        while free:
            pick = pull_choose(self.pending, free, now, self.pull_starve_s, self.router.matched,
                               priorities=self.priorities, idle_since=self.idle_since,
                               prefer_idle_s=self.prefer_idle_s, live_matched_of=self.router.live_matched)
            if pick is None:
                break
            entry, k, starved = pick
            mig_src = entry.migrate_from
            if mig_src is not None and (mig_src == k or mig_src not in free):
                mig_src = None
                entry.migrate_from = None
            self.pending.pop(entry.seq, None)       # the assignment and inflight + 1 are one step under self.cv
            entry.replica = k
            self.inflight[k] += 1
            self.routed[k] += 1
            self.pull_stats["assigned"] += 1
            if starved:
                self.pull_stats["starved"] += 1
            if mig_src is not None:
                self.inflight[mig_src] += 1
                self.pull_stats["migrated"] += 1
                free.remove(mig_src)
            reason = ("starved" if starved else
                      ("prefer-idle" if mig_src is not None else
                       ("reuse" if entry.matched and entry.matched[k] else "min-reread")))
            self.last_pick = {"replica": k,
                              "reason": reason,
                              **({"from": mig_src} if mig_src is not None else {}),
                              "at": time.monotonic()}
            free.remove(k)
        self.cv.notify_all()

    def _release_migration_src(self, src: int) -> None:
        with self.cv:
            self.inflight[src] -= 1
            if self.inflight[src] == 0:
                self.idle_since[src] = time.monotonic()
            self._pull_pump_locked(time.monotonic())
            self.cv.notify_all()

    def _complete_migration(self, dst: int, src: int, ids, cancel) -> int:
        """Ask the lower-priority idle replica `src` to PARK its live conversation so `dst` can restore it;
        returns `dst` on success (or harmless no-op) and falls back to `src` if PARK failed on a long live session."""
        from serve.server import EngineDied
        src_lock = self.run_locks[src]
        acquired = False
        try:
            while not acquired:
                acquired = src_lock.acquire(timeout=0.25)
                if not acquired and cancel is not None and cancel.is_set():
                    with self.cv:
                        for rep in (dst, src):
                            self.inflight[rep] -= 1
                            if self.inflight[rep] == 0:
                                self.idle_since[rep] = time.monotonic()
                        self.routed[dst] = max(0, self.routed[dst] - 1)
                        self.pull_stats["cancelled"] += 1
                        self._pull_pump_locked(time.monotonic())
                        self.cv.notify_all()
                    return -1
            park_fn = getattr(self.engines[src], "park", None)
            if park_fn is None:
                res = {"status": "published"}
            else:
                try:
                    res = park_fn()
                except (ValueError, EngineDied, OSError) as exc:
                    res = {"status": "error", "error": str(exc)}
        except BaseException:
            if acquired:
                src_lock.release()
                acquired = False
            with self.cv:
                for rep in (dst, src):
                    self.inflight[rep] -= 1
                    if self.inflight[rep] == 0:
                        self.idle_since[rep] = time.monotonic()
                self._pull_pump_locked(time.monotonic())
                self.cv.notify_all()
            raise
        if acquired:
            src_lock.release()

        status = res.get("status") if isinstance(res, dict) else None
        if status in ("published", "dedup", "minimum", "empty", "off"):
            self.router.demote_live(src, parked=(status in ("published", "dedup")))
            self._release_migration_src(src)
            return dst

        src_alive = self.engines[src].alive()
        live_src = self.router.live_matched(ids)[src]
        dst_got = self.router.matched(ids)[dst]
        if src_alive and (live_src - dst_got) >= max(BLOCK, self.router.shared_min_tokens):
            with self.cv:
                self.inflight[dst] -= 1
                if self.inflight[dst] == 0:
                    self.idle_since[dst] = time.monotonic()
                self.routed[dst] = max(0, self.routed[dst] - 1)
                self.routed[src] += 1
                self.pull_stats["migrate_fallbacks"] += 1
                self.last_pick = {"replica": src, "reason": "migrate-fallback", "at": time.monotonic()}
                self._pull_pump_locked(time.monotonic())
                self.cv.notify_all()
            return src

        self._release_migration_src(src)
        if not src_alive:
            self.heal()
        return dst

    def _pull_drop_locked(self, entry: PullRequest, reason: str) -> None:
        """Take this request out of the queue, if it is still in it; idempotent, so a request that was already
        assigned is never dropped twice.  Called under self.cv."""
        if self.pending.pop(entry.seq, None) is not None:
            self.pull_stats[reason] += 1

    def _pull_starting(self) -> bool:
        """Some replica is being started again (heal/restart): "every replica is dead" then means wait, not fail."""
        return any(getattr(e, "starting", False) for e in self.engines) or any(self.healing.values())

    def _pull_some_dead(self) -> bool:
        """A replica that died could be started again: the wait loop calls heal() outside the lock."""
        return any(not e.alive() and not getattr(e, "unloaded", False) and not getattr(e, "starting", False)
                   for e in self.engines)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        # imported here: serve.server imports this module
        from serve.server import EngineDied, EngineNonFinite
        if self.pull_enabled:                                # opt-in: wait in the shared queue for a free replica
            k, mig_src = yield from self._pull_wait(ids, cancel)
            if k < 0:                                        # the client left before a replica was free: nothing to do
                return
            if mig_src is not None:
                k = self._complete_migration(k, mig_src, ids, cancel)
                if k < 0:
                    return
        else:                                                # the old order: a replica at once, then run_locks
            k = self.route(ids)
            if k < 0:
                raise EngineDied("no replica is running; this request was not sent")
        e = self.engines[k]
        self.tl.cur = e
        _ = self.last                                         # the caller's figures exist before this request (#2892)
        produced: list[int] = []
        held = False
        gen = None
        native_before = None
        nonfinite = False
        try:
            selected = getattr(cancel, "_strata_replica_selected", None)
            if selected is not None:
                selected(k)
            if self.pull_enabled:                            # the queue is over: the terminal row becomes "reading"
                if cancel is not None:
                    setattr(cancel, "_strata_replica", k)
                started = getattr(cancel, "_strata_queue_started", None)
                if started is not None:
                    started()
                if getattr(cancel, "_strata_queue_deadline", None) is not None:
                    cancel._strata_queue_deadline = None
            self.router.note(k, ids)
            if not int(getattr(e, "batch", 0) or 0):          # no slots: this replica takes one request at a time
                lock = self.run_locks[k]
                while not lock.acquire(timeout=0.25):
                    if cancel is not None and cancel.is_set():
                        return
                held = True
            native_before = e.last
            if embeddings is not None:
                gen = e.generate(ids, max_new, sampling, cancel, embeddings=embeddings)
            else:
                gen = e.generate(ids, max_new, sampling, cancel)
            for t in gen:
                if t is not None:
                    produced.append(t)
                yield t
        except EngineNonFinite:
            nonfinite = True
            raise
        finally:
            # #2892: the replica's pipe belongs to this request until its native exchange is over.  The child
            # generator sends STOP and reads to its DONE when it is closed; that has to happen HERE, before the lock is
            # given back - left to the garbage collector it ran after the next request had sent its GEN, and the two
            # read each other's tokens and DONE.  A close that fails (the engine never answered its STOP) ends the
            # replica: it is not handed on (heal() starts it again), and the error is not lost.
            try:
                if gen is not None:
                    try:
                        gen.close()
                    except BaseException:                     # noqa: BLE001 - the pipe cannot be trusted any more
                        self.quarantine(k)
                        raise
            finally:
                try:
                    # the figures of the DONE this caller's request read, copied while it still owns the replica; with
                    # no new DONE the caller's earlier object stays, so Service does not record an old result as new
                    if gen is not None and e.last is not native_before:
                        self.tl.last = dict(e.last)
                finally:
                    if self.pull_enabled:
                        # the card's map is updated BEFORE the slot is handed on: the next assignment (the pump
                        # below) must see the fresh map, and the freed control line before it can run (design §5.5)
                        if nonfinite:                   # #879: the engine dropped every cached conversation
                            self.router.forget(k)
                            if self.__dict__.get("parent_parking") is not None:
                                self.router.forget_shared()
                        elif produced:
                            self.router.note(k, list(ids) + produced)
                        if held:
                            self.run_locks[k].release()
                        with self.cv:
                            self.inflight[k] -= 1
                            if self.inflight[k] == 0:
                                self.idle_since[k] = time.monotonic()
                            self._pull_pump_locked(time.monotonic())
                            self.cv.notify_all()
                    else:
                        if held:
                            self.run_locks[k].release()
                        with self.cv:
                            self.inflight[k] -= 1
                            if self.inflight[k] == 0:
                                self.idle_since[k] = time.monotonic()
                            self.cv.notify_all()
                        if nonfinite:                   # #879: the engine dropped every cached conversation
                            self.router.forget(k)
                            if self.__dict__.get("parent_parking") is not None:
                                self.router.forget_shared()
                        elif produced:
                            self.router.note(k, list(ids) + produced)

    def quarantine(self, k: int) -> None:
        """A replica whose pipe could not be drained is out of the routing until it has been started again: its process
        is ended (nothing it still prints can reach a later request) and heal() restarts it."""
        e = self.engines[k]
        try:
            e.ended = True
        except AttributeError:
            pass
        proc = getattr(e, "proc", None)
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
        except (OSError, AttributeError, ValueError):
            pass
        self.router.forget(k)
