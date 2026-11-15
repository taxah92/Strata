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


PullSettings = collections.namedtuple("PullSettings", "enabled starve_s queue_timeout_s max_queue",
                                      defaults=(None, None))


class PullRequest:
    """One request in the one shared pending queue.  Its fields change only under ReplicaEngine.cv."""
    __slots__ = ("seq", "ids", "arrive", "cancel", "deadline", "replica", "matched")

    def __init__(self):
        self.seq = 0                    # the queue's own order: tie-break, and "the oldest" is the first one
        self.ids = []                   # the prompt, as the request made it (the server does not change it)
        self.arrive = 0.0               # time.monotonic() when it was put in the queue
        self.cancel = None              # the request's own Event: a client that left sets it
        self.deadline = None            # time.monotonic() after which waiting is over (queue_timeout), or None
        self.replica = None             # the replica it was given; None while it still waits
        self.matched = None             # Router.matched(ids) of the last selection round (the reason in the view)


def pull_choose(pending, free, now: float, starve_s: float, matched_of):
    """(the waiting request, the replica, was it the starvation rule) or None: the pull order as one pure function,
    so it can be tested without threads.  `pending` is in arrival order, `free` the living replicas with
    inflight[k] == 0.  Cost = the prompt tokens the replica would read again (len(ids) - Router.matched(ids)[k]) -
    the first term of route()'s cost, as a free replica has no running request to pay LOAD_TOKENS for."""
    if not free or not pending:
        return None
    oldest = next(iter(pending.values()))
    if starve_s is not None and now - oldest.arrive >= starve_s:       # the oldest goes first, whatever its cache says
        got = matched_of(oldest.ids)
        oldest.matched = got
        k = min(free, key=lambda k: (len(oldest.ids) - got[k], k))
        return oldest, k, True
    best = None
    for e in pending.values():                  # arrival order; a smaller seq wins an equal cost
        got = matched_of(e.ids)                 # the card's map is read AT THE CHOICE, not when the request arrived
        e.matched = got
        base = len(e.ids)
        for k in free:
            cand = (base - got[k], e.seq, k)    # re-read, then age, then the lower replica index
            if best is None or cand < best[0]:
                best = (cand, e, k)
    if best is None:
        return None
    return best[1], best[2], False


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


def pull_config(cfg: dict) -> PullSettings:
    """Opt-in pull scheduler: the run config's ("replica_pull", "replica_starve_s", "replica_queue_timeout_s",
    "replica_max_queue") as a PullConfig, for serve/server.py's start_replicas:
    ReplicaEngine(engines, load_tokens=..., pull=pc.enabled, starve_s=pc.starve_s,
                  queue_timeout_s=pc.queue_timeout_s, max_queue=pc.max_queue).  Raises ReplicaConfigError on a bad
    value, before any engine has been started.  queue_timeout_s and max_queue are the contract's optional
    scheduler-level limits (None: the Service's own queue_timeout/max_queue admission is the only limit)."""
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
    return PullSettings(True, float(starve), float(timeout) if timeout is not None else None, max_queue)


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
    that served it (a bit mask)."""

    def __init__(self, n: int, cap: int = INDEX_MAX):
        self.n, self.cap = n, cap
        self.blocks: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def chain(ids) -> list[int]:
        out, h = [], 0
        for k in range(len(ids) // BLOCK):
            h = hash((h, tuple(ids[k * BLOCK:(k + 1) * BLOCK])))
            out.append(h)
        return out

    def matched(self, ids) -> list[int]:
        """Per replica: how many leading tokens of `ids` it holds (a multiple of BLOCK)."""
        got = [0] * self.n
        alive = (1 << self.n) - 1
        with self.lock:
            for k, h in enumerate(self.chain(ids)):
                m = self.blocks.get(h, 0) & alive
                if not m:
                    break
                alive = m
                for r in range(self.n):
                    if m >> r & 1:
                        got[r] = (k + 1) * BLOCK
        return got

    def note(self, r: int, ids) -> None:
        with self.lock:
            for h in self.chain(ids):
                self.blocks[h] = self.blocks.pop(h, 0) | (1 << r)
            while len(self.blocks) > self.cap:
                self.blocks.popitem(last=False)

    def forget(self, r: int) -> None:
        """Replica r lost its cache (it restarted)."""
        with self.lock:
            bit = ~(1 << r)
            for h in [h for h, m in self.blocks.items() if m >> r & 1]:
                m = self.blocks[h] & bit
                if m:
                    self.blocks[h] = m
                else:
                    del self.blocks[h]


class ReplicaEngine:
    """Stands in for one engine (the interface Service uses), over several."""

    NOT_HERE = {"session_file", "vram", "ctl", "slot_cv"}       # not offered: one conversation file / VRAM resize / control line
    PROPAGATE = {"silence_s", "gpu_busy"}

    def __init__(self, engines: list, load_tokens: int = LOAD_TOKENS, *, pull: bool = False,
                 starve_s: float = PULL_STARVE_S, poll_s: float = PULL_POLL_S, beat_s: float = PULL_BEAT_S,
                 queue_timeout_s: float | None = None, max_queue: int | None = None):
        if len(engines) < 2:
            raise ValueError("ReplicaEngine needs two or more engines")
        if not isinstance(pull, bool):
            raise ReplicaConfigError(f'"replica_pull" must be true or false, not {pull!r}')
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
        d["router"] = Router(len(engines))
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
        d["pull_seq"] = 0
        d["pending"] = collections.OrderedDict()    # seq -> PullRequest, order = arrival: the oldest is the first
        d["pull_stats"] = {"submitted": 0, "assigned": 0, "starved": 0, "cancelled": 0, "closed": 0,
                           "timeouts": 0, "no_replica": 0, "rejected": 0}
        d["last_pick"] = None                   # {"replica", "reason", "at"} of the last assignment (for /metrics)
        d["tl"] = threading.local()
        d["healing"] = {}
        d["heal_at"] = [0.0] * len(engines)
        d["routed"] = [0] * len(engines)
        d["gen_seen"] = [getattr(e, "gen", 0) for e in engines]

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
        reason of the last assignment): the server's /metrics needs no new field to show it."""
        rows = [{"replica": k, "alive": e.alive(), "running": self.inflight[k], "routed": self.routed[k],
                 "batch": int(getattr(e, "batch", 0) or 0)} for k, e in enumerate(self.engines)]
        if self.pull_enabled:
            queue = self.pull_view()
            for row in rows:
                row["queue"] = {"depth": queue["pending"], "oldest_s": queue["oldest_pending_s"],
                                "picked": queue["picked"], "reason": queue["reason"],
                                "max": queue["max_queue"], "timeout_s": queue["queue_timeout_s"]}
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
                "picked": pick["replica"] if pick else None, "reason": pick["reason"] if pick else None,
                **stats}

    # ---- life cycle
    def restart(self, tries: int = 3):
        dead = [k for k, e in enumerate(self.engines) if not e.alive()]
        errors: dict[int, BaseException] = {}

        def one(k):
            try:
                self.engines[k].restart(tries)
                self.router.forget(k)
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
        for e in self.engines:
            e.unload()
        for k in range(len(self.engines)):
            self.router.forget(k)

    def close(self):
        err = None
        for e in self.engines:
            try:
                e.close()
            except BaseException as x:           # noqa: BLE001
                err = err or x
        remove_arena(self.__dict__.get("arena_file"))   # (an engine that still maps it keeps its pages)
        if err:
            raise err

    # ---- routing
    def route(self, ids) -> int:
        """The replica for this prompt (and its running count raised): lowest cost among the living ones."""
        got = self.router.matched(ids)
        with self.lock:
            alive = [k for k, e in enumerate(self.engines) if e.alive()]
            if not alive:
                return -1
            cost = {k: (len(ids) - got[k]) + self.load_tokens * self.inflight[k] for k in alive}
            # the least loaded wins a tie of cost (equal cards, nothing cached): spread, do not pile on replica 0
            k = min(alive, key=lambda k: (cost[k], self.inflight[k], self.routed[k]))
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
                    print(f"[strata] replica {k} is running again", flush=True)
                except BaseException as x:       # noqa: BLE001
                    print(f"[strata] replica {k} did not start again: {x}", flush=True)
                finally:
                    self.healing[k] = False
            threading.Thread(target=run, daemon=True).start()

    # ---- pull scheduler (opt-in: "replica_pull": true; see pull_config and the module doc)
    def _pull_wait(self, ids, cancel):
        """The replica for this prompt under the pull scheduler, or None heartbeats while the request waits - a
        generator, so the heartbeat goes out on the request's own turn and the server's keep-alive is unchanged.
        A free replica takes it at once (the pump); otherwise it waits in the one shared queue, and every slot that
        frees is given to the waiting request that would re-read the least of its prompt (or to the oldest one, after
        replica_starve_s).  Returns the replica index, or -1 when the client left (its queue entry removed).  The
        assignment and inflight + 1 are one step under self.cv; generation happens outside it."""
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
        try:
            while True:
                with self.cv:
                    if entry.replica is not None:   # assigned: this request owns a slot from here
                        k = entry.replica
                        break
                    now = time.monotonic()
                    if cancel is not None and cancel.is_set():
                        self._pull_drop_locked(entry, "cancelled")
                        return -1
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
            return k
        finally:
            with self.cv:
                if not taken:
                    if entry.replica is not None:   # a slot was given but this waiter leaves before using it
                        self.inflight[entry.replica] -= 1
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
        free = [k for k, e in enumerate(self.engines) if e.alive() and not self.inflight[k]]
        while free:
            pick = pull_choose(self.pending, free, now, self.pull_starve_s, self.router.matched)
            if pick is None:
                break
            entry, k, starved = pick
            self.pending.pop(entry.seq, None)       # the assignment and inflight + 1 are one step under self.cv
            entry.replica = k
            self.inflight[k] += 1
            self.routed[k] += 1
            self.pull_stats["assigned"] += 1
            if starved:
                self.pull_stats["starved"] += 1
            self.last_pick = {"replica": k,
                              "reason": "starved" if starved else
                                        ("reuse" if entry.matched and entry.matched[k] else "min-reread"),
                              "at": time.monotonic()}
            free.remove(k)
        self.cv.notify_all()

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
            k = yield from self._pull_wait(ids, cancel)
            if k < 0:                                        # the client left before a replica was free: nothing to do
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
                        elif produced:
                            self.router.note(k, list(ids) + produced)
                        if held:
                            self.run_locks[k].release()
                        with self.cv:
                            self.inflight[k] -= 1
                            self._pull_pump_locked(time.monotonic())
                    else:
                        if held:
                            self.run_locks[k].release()
                        with self.lock:
                            self.inflight[k] -= 1
                        if nonfinite:                   # #879: the engine dropped every cached conversation
                            self.router.forget(k)
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
