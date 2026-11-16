"""Parent parking: one RAM store in the server owns every replica's parked conversations.

A parked conversation is a session v1 image (the same bytes the disk tier writes) that a replica would otherwise keep
in its own heap.  With two or more replicas that means two copies of the same conversation, and a conversation parked
on r0 cannot be restored on r1 at all.  This module is the parent side of the replacement: the Python server keeps the
images in memfds it creates, and the engines stream a snapshot into a memfd the parent hands them and read one back
from a sealed memfd the parent hands them.  No K/V crosses the socket: only metadata, hashes and lengths do.

The store is the only owner of the index, the global LRU and the memfds.  One daemon thread (the broker) serves every
replica's socketpair with `selectors`; only that thread changes entries, pending publishes, matches, leases and the
LRU.  `open_channel` may be called from several threads at once (parallel StrataEngine starts) and only guards the
registration queue with a lock.  `snapshot()` copies scalars under the same short lock; it never runs a callback and
never touches a memfd.

The wire format is frozen in `.scratch/handoff-notes/parent-parking/contract.md` (protocol v1): a 36-byte little-endian
header (`STRPARK1`, version, op, flags, payload size, request id, payload hash) and a UTF-8 JSON payload, over
`AF_UNIX`/`SOCK_SEQPACKET`; `SCM_RIGHTS` only in the answers to `PARK_BEGIN` (writable stage fds) and `ACQUIRE`
(sealed stage fds).  The payload hash is the project's own `session_hash64` (src/core/conversation_file.cpp) with the
seed `STRPARK1` read as a big-endian u64 (the project's convention).

RAM accounting is the point of the whole exercise.  `charged_bytes` is the visible committed snapshots plus pending
publishes plus snapshots evicted while a reader still holds a lease plus the transient restore reservations; it never
exceeds `ram_gib`.  An eviction plan is modelled whole before anything is deleted, and a snapshot with an active lease
is never chosen as a victim.

Opt-in, in the run config:

    "replica_parent_parking": {
      "enabled": true,
      "ram_gib": 40,              the one hard budget, restore reservations included
      "max_entries": 20,          optional, default 10 * replicas
      "min_tokens": 8192,         optional, default 0
      "min_gain": 4096,           optional, default 4096
      "ipc_timeout_ms": 2000      optional, default 2000
    }

`read_parent_parking` checks the block at start (a block that cannot be used is a start error, never a silent
fallback); `enabled: false` or no block returns None and nothing here runs.  The rest of the start checks of the plan
(two or more replicas, `replica_pull`, no batch slots, no `replica_handoff`, Linux) belong to the server's start
validation, not to this module.
"""
from __future__ import annotations

import atexit
import collections
import dataclasses
import fcntl
import json
import os
import random
import selectors
import socket
import stat
import struct
import threading
import time

# ---- protocol v1 (contract §1, §2)
MAGIC = b"STRPARK1"
VERSION = 1
HEADER = struct.Struct("<8sHHIIQQ")          # 36 bytes: magic, version, op, flags, payload_bytes, request_id, hash
MAX_PAYLOAD = 1 << 20
MAX_POINTS = 4096
MAX_STAGES = 64
MAX_FDS = 64
MAX_EXCLUDE_IDS = 1
MAX_OWNER = 64
PUBLISH_ID_LEN = 32
HASH_SEED = 0x5354525041524B31              # "STRPARK1" read as a big-endian u64 (the project's convention)
IDS_SEED = 0x5354524944533031               # "STRIDS01": the point hash the engines send
LAYOUT_SEED = 0x53545243444c3031             # "STRCDL01": ConversationDiskLayout::key()

OP_HELLO = 1
OP_PARK_ADVICE = 2
OP_PARK_BEGIN = 3
OP_PARK_COMMIT = 4
OP_PARK_ABORT = 5
OP_MATCH_BEGIN = 6
OP_MATCH_END = 7
OP_ACQUIRE = 8
OP_RELEASE = 9
OP_CLEAR = 10
OP_STATS = 11
OP_PING = 12
OP_ERROR = 255

LIMITS = {"max_payload": MAX_PAYLOAD, "max_points": MAX_POINTS, "max_stages": MAX_STAGES, "max_fds": MAX_FDS,
          "max_exclude_ids": MAX_EXCLUDE_IDS}
RELEASE_RESULTS = ("ok", "invalid", "memory", "io", "mismatch")
CLEAR_REASONS = ("nonfinite", "unload", "test")
RETRYABLE = {"bad_request": False, "protocol": False, "identity": False, "capacity": True,
             "pinned_working_set": True, "stale_epoch": False, "gone": False, "stale_generation": False,
             "internal": True}

# ---- session file sizes (src/core/conversation_file.cpp)
READ_BLOCK = 16 << 20                       # kBlock: the read buffer and the segment directory's unit
READ_PAGE = 4096                            # a page per buffer segment
READ_OVERHEAD = 1 << 20                     # the small vectors
MEMFD_NAME = "strata-parent-parking"
SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
SOCKET_BUFFER = 4 << 20                     # a 1 MiB frame must fit in one SOCK_SEQPACKET message
COMMITTED_MEMORY = 4096                     # publish ids remembered for a lost COMMIT answer
BROKER_JOIN_S = 10.0
COMMAND_TIMEOUT_S = 30.0

# ---- session_hash64 (src/core/conversation_file.cpp SessionHasher)
P1 = 0x9E3779B185EBCA87
P2 = 0xC2B2AE3D27D4EB4F
P3 = 0x165667B19E3779F9
P4 = 0x85EBCA77C2B2AE63
P5 = 0x27D4EB2F165667C5
MASK64 = (1 << 64) - 1


def _rotl(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (64 - bits))) & MASK64


def _round1(acc: int, word: int) -> int:
    return (_rotl((acc + word * P2) & MASK64, 31) * P1) & MASK64


def session_hash64(data: bytes, seed: int = 0) -> int:
    """The project's non-cryptographic 64-bit hash: xxHash64-style rounds, not the standard XXH64 stream.  Both sides
    of the protocol must agree byte for byte, so this is a faithful copy of SessionHasher, not a new hash."""
    lane = [(seed + P1 + P2) & MASK64, (seed + P2) & MASK64, seed & MASK64, (seed - P1) & MASK64]
    total = len(data)
    offset = 0
    while offset + 32 <= total:
        for i in range(4):
            lane[i] = _round1(lane[i], int.from_bytes(data[offset + 8 * i:offset + 8 * i + 8], "little"))
        offset += 32
    pending = data[offset:]
    h = (_rotl(lane[0], 1) + _rotl(lane[1], 7) + _rotl(lane[2], 12) + _rotl(lane[3], 18)) & MASK64
    for value in lane:
        h = ((h ^ _round1(0, value)) * P1 + P4) & MASK64
    h = (h + total) & MASK64
    i = 0
    while i + 8 <= len(pending):
        h = (h ^ _round1(0, int.from_bytes(pending[i:i + 8], "little"))) & MASK64
        h = (_rotl(h, 27) * P1 + P4) & MASK64
        i += 8
    while i < len(pending):
        h = (h ^ (pending[i] * P5)) & MASK64
        h = (_rotl(h, 11) * P1) & MASK64
        i += 1
    h ^= h >> 33
    h = (h * P2) & MASK64
    h ^= h >> 29
    h = (h * P3) & MASK64
    h ^= h >> 32
    return h


def layout_key(stages) -> int:
    """`ConversationDiskLayout::key()`: the stage layout fingerprint the engines send in HELLO.  The server builds the
    store's identity with it so a replica with another layer split is refused at HELLO."""
    pairs = [(int(lo), int(hi)) for lo, hi in stages]
    data = struct.pack("<Q", len(pairs)) + b"".join(struct.pack("<qq", lo, hi) for lo, hi in pairs)
    return session_hash64(data, LAYOUT_SEED)


def session_read_peak_bytes(file_bytes: int) -> int:
    """What a read of a session file of `file_bytes` holds at its peak: the parsed image, the 16 MiB read buffer, a
    page per buffer segment and the small vectors (src/core/conversation_file.cpp session_read_peak_bytes)."""
    segments = file_bytes // READ_BLOCK + 1
    return file_bytes + READ_BLOCK + segments * READ_PAGE + READ_OVERHEAD


# ---- run config (plan §13.1, §13.3)
PARENT_PARKING_KEYS = ("enabled", "ram_gib", "max_entries", "min_tokens", "min_gain", "ipc_timeout_ms")
PARENT_PARKING_MIN_GAIN = 4096
PARENT_PARKING_IPC_TIMEOUT_MS = 2000


class ParentParkingConfigError(ValueError):
    """The run config's "replica_parent_parking" block cannot be used as written: the start says why (never a silent
    fallback to no parent parking)."""


@dataclasses.dataclass(frozen=True)
class ParentParkingConfig:
    """The checked "replica_parent_parking" block with its defaults filled in."""
    ram_gib: int
    max_entries: int
    min_tokens: int = 0
    min_gain: int = PARENT_PARKING_MIN_GAIN
    ipc_timeout_ms: int = PARENT_PARKING_IPC_TIMEOUT_MS
    enabled: bool = True

    @property
    def capacity_bytes(self) -> int:
        return self.ram_gib << 30

    @property
    def ipc_timeout_s(self) -> float:
        return self.ipc_timeout_ms / 1000.0


def _config_whole(block: dict, key: str, *, lo: int, default=None):
    """The block's whole number `key` (>= lo), `default` when it is absent; anything else is a start error."""
    value = block.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < lo:
        raise ParentParkingConfigError(f'replica_parent_parking: "{key}" must be a whole number >= {lo}, '
                                       f'not {value!r}')
    return value


def _config_replicas(cfg: dict) -> int:
    """How many replicas this start has, for the default max_entries (10 * replicas).  Parent parking needs two or
    more; a config that does not say is read as the minimum."""
    spec = cfg.get("replicas")
    if isinstance(spec, bool):
        return 2
    if isinstance(spec, int):
        return max(2, spec)
    if isinstance(spec, list):
        return max(2, len(spec))
    return 2


def read_parent_parking(cfg: dict, replicas: int | None = None) -> ParentParkingConfig | None:
    """The checked run config's "replica_parent_parking" block with its defaults filled in; None when it is off (no
    block, or "enabled": false - the default).  Raises ParentParkingConfigError for a block that cannot be used.

    `replicas` is the number of replicas this start has (the server passes len(groups)); without it the count is read
    from the config's "replicas" and defaults to the two parent parking needs."""
    block = cfg.get("replica_parent_parking")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ParentParkingConfigError(f'"replica_parent_parking" must be an object (a block of keys), not {block!r}')
    enabled = block.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ParentParkingConfigError(f'replica_parent_parking: "enabled" must be true or false, not {enabled!r}')
    if not enabled:                                     # off by default: the rest of the block is not read
        return None
    unknown = sorted(key for key in block if key not in PARENT_PARKING_KEYS)
    if unknown:
        raise ParentParkingConfigError('replica_parent_parking: unknown key(s) ' +
                                       ", ".join(repr(key) for key in unknown) + "; the block takes " +
                                       ", ".join(PARENT_PARKING_KEYS))
    ram_gib = block.get("ram_gib")
    if ram_gib is None:
        raise ParentParkingConfigError('replica_parent_parking: "enabled": true needs "ram_gib", the one hard budget '
                                       'of the store (restore reservations included; 40 in the production config)')
    ram_gib = _config_whole(block, "ram_gib", lo=1)
    max_entries = _config_whole(block, "max_entries", lo=1)
    min_tokens = _config_whole(block, "min_tokens", lo=0, default=0)
    min_gain = _config_whole(block, "min_gain", lo=0, default=PARENT_PARKING_MIN_GAIN)
    ipc_timeout_ms = _config_whole(block, "ipc_timeout_ms", lo=1, default=PARENT_PARKING_IPC_TIMEOUT_MS)
    if max_entries is None:
        max_entries = 10 * (replicas if replicas is not None else _config_replicas(cfg))
    return ParentParkingConfig(ram_gib=ram_gib, max_entries=max_entries, min_tokens=min_tokens, min_gain=min_gain,
                               ipc_timeout_ms=ipc_timeout_ms)


# ---- identity (plan §5)
@dataclasses.dataclass(frozen=True)
class ParentParkingIdentity:
    """What every replica's snapshots are bound to: the model/config fingerprints and the stage layout.  The first
    HELLO fixes it for the whole store; a replica with other values is refused at start."""
    model: int
    config: int
    layout_key: int
    stages: tuple

    def as_tuple(self) -> tuple:
        return (self.model, self.config, self.layout_key, tuple(self.stages))


def _identity_int(value, name: str, *, lo: int = 0, hi: int = MASK64) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"parent parking identity: {name} must be a whole number in {lo}..{hi}, not {value!r}")
    return value


def _identity_stages(value) -> tuple:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_STAGES:
        raise ValueError(f"parent parking identity: stages must be a list of 1..{MAX_STAGES} (lo, hi) pairs")
    out = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"parent parking identity: a stage must be a (lo, hi) pair, not {pair!r}")
        lo = _identity_int(pair[0], "stage lo", lo=-(1 << 63), hi=(1 << 63) - 1)
        hi = _identity_int(pair[1], "stage hi", lo=-(1 << 63), hi=(1 << 63) - 1)
        if lo > hi:
            raise ValueError(f"parent parking identity: stage {lo}..{hi} is empty")
        out.append((lo, hi))
    return tuple(out)


def _identity_of(identity) -> ParentParkingIdentity | None:
    """A ParentParkingIdentity from the object the server passes (a dataclass, a dict or any object with the four
    fields); None means the first HELLO fixes it."""
    if identity is None:
        return None
    if isinstance(identity, ParentParkingIdentity):
        return identity
    if isinstance(identity, dict):
        get = identity.get
    else:
        def get(key):
            return getattr(identity, key, None)
    return ParentParkingIdentity(_identity_int(get("model"), "model"),
                                 _identity_int(get("config"), "config"),
                                 _identity_int(get("layout_key"), "layout_key"),
                                 _identity_stages(get("stages")))


# ---- errors and records
class ParentParkingClosed(RuntimeError):
    """The store was closed; no new channel can be opened."""


class _RequestError(Exception):
    """One refused request: the code and message of the ERROR frame, and whether the channel closes after it."""

    def __init__(self, code: str, message: str, *, close: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.close = close

    def payload(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": RETRYABLE.get(self.code, False)}


@dataclasses.dataclass(frozen=True)
class _Point:
    """One point of a snapshot: the live state (live=True, first) or a checkpoint, as the engine hashes it."""
    tokens: int
    hash: int
    imgs: tuple
    live: bool
    pinned: bool


@dataclasses.dataclass
class _Entry:
    """One committed snapshot: the index record and the sealed stage memfds the parent owns."""
    id: int
    generation: int
    publish_id: str
    owner: str
    model: int
    config: int
    layout_key: int
    stages: tuple
    cvec: bool
    points: tuple
    stage_bytes: tuple
    fds: list
    bytes: int
    used: int
    pinned: bool
    leases: set
    meta: tuple


@dataclasses.dataclass
class _Pending:
    """A publish between PARK_BEGIN and PARK_COMMIT: charged, invisible to MATCH, with its writable stage memfds."""
    id: int
    publish_id: str
    conn: "_Connection"
    owner: str
    model: int
    config: int
    layout_key: int
    stages: tuple
    cvec: bool
    points: tuple
    stage_bytes: tuple
    fds: list
    bytes: int
    epoch: int
    meta: tuple


@dataclasses.dataclass
class _Match:
    """A MATCH_BEGIN waiting for its MATCH_END: the lengths worth hashing and the entries that offered them."""
    cookie: int
    conn: "_Connection"
    prompt_tokens: int
    cvec: bool
    images: tuple
    lengths: tuple
    candidates: frozenset
    expiry: float


@dataclasses.dataclass
class _Lease:
    """One reader's hold on a committed snapshot: the restore reservation lives until RELEASE or EOF."""
    id: int
    entry_id: int
    generation: int
    conn: "_Connection"
    owner: str
    stage_bytes: tuple
    read_charge: int


@dataclasses.dataclass
class _Retired:
    """A snapshot removed from the index while a reader still holds it: its bytes stay charged until the last lease."""
    entry_id: int
    bytes: int
    leases: set


class _OwnerStats:
    """Per-replica counters for the snapshot: what this owner published and read, and its protocol failures."""
    __slots__ = ("published", "reads", "peer_reads", "fallbacks", "ipc_errors")

    def __init__(self):
        self.published = 0
        self.reads = 0
        self.peer_reads = 0
        self.fallbacks = 0
        self.ipc_errors = 0


class _Connection:
    """One replica's channel: its socket, its HELLO, and the pending publishes, leases and match it owns."""
    __slots__ = ("conn_id", "sock", "owner", "hello", "last_request_id", "pending", "leases", "match",
                 "last_candidate")

    def __init__(self, conn_id: int, sock: socket.socket, owner: str):
        self.conn_id = conn_id
        self.sock = sock
        self.owner = owner
        self.hello = False
        self.last_request_id = 0
        self.pending = set()
        self.leases = set()
        self.match = None
        self.last_candidate = None


class ParentChannel:
    """The server's end of one replica's channel: what open_channel returns beside the child fd.  The broker owns the
    socket; this handle only reports the channel's state and can close it (the engine's EOF does the same)."""
    __slots__ = ("owner", "_store", "_conn_id")

    def __init__(self, store: "ParentParkingStore", owner: str, conn_id: int):
        self.owner = owner
        self._store = store
        self._conn_id = conn_id

    @property
    def state(self) -> str:
        return self._store._channel_state(self._conn_id)

    def close(self) -> None:
        self._store._close_channel(self._conn_id)

    def __repr__(self) -> str:
        return f"ParentChannel(owner={self.owner!r}, state={self.state!r})"


# ---- JSON and field checks (contract §3, §8)
class _DuplicateKey(ValueError):
    pass


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


def _reject_constant(text):
    raise ValueError(f"not a JSON number: {text}")


def _bad(message: str, *, close: bool = True) -> _RequestError:
    return _RequestError("bad_request", message, close=close)


def _parse_json(payload: bytes) -> dict:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise _bad("a payload must not start with a byte order mark")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise _bad("a payload must be UTF-8") from None
    try:
        value = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except (ValueError, _DuplicateKey):
        raise _bad("a payload must be one JSON object") from None
    if not isinstance(value, dict):
        raise _bad("a payload must be one JSON object")
    return value


def _keys(obj: dict, required, optional=()) -> None:
    unknown = sorted(key for key in obj if key not in required and key not in optional)
    if unknown:
        raise _bad("unknown key(s) " + ", ".join(repr(key) for key in unknown))
    missing = [key for key in required if key not in obj]
    if missing:
        raise _bad("missing key(s) " + ", ".join(repr(key) for key in missing))


def _whole(value, name: str, *, lo: int = 0, hi: int = MASK64) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise _bad(f'"{name}" must be a whole number in {lo}..{hi}')
    return value


def _int(obj: dict, key: str, *, lo: int = 0, hi: int = MASK64) -> int:
    return _whole(obj[key], key, lo=lo, hi=hi)


def _u64(obj: dict, key: str, *, lo: int = 0) -> int:
    return _whole(obj[key], key, lo=lo, hi=MASK64)


def _i64(obj: dict, key: str, *, lo: int = 0) -> int:
    return _whole(obj[key], key, lo=lo, hi=(1 << 63) - 1)


def _bool(obj: dict, key: str) -> bool:
    value = obj[key]
    if not isinstance(value, bool):
        raise _bad(f'"{key}" must be true or false')
    return value


def _text(value, name: str, *, max_len: int | None = None) -> str:
    if not isinstance(value, str):
        raise _bad(f'"{name}" must be a string')
    if max_len is not None and len(value) > max_len:
        raise _bad(f'"{name}" is longer than {max_len} characters')
    if not value.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise _bad(f'"{name}" must be printable ASCII')
    return value


def _owner_name(value) -> str:
    """An owner name: 1..64 printable ASCII characters ("r0".."rN" in the server)."""
    if not isinstance(value, str) or not value or len(value) > MAX_OWNER:
        raise ValueError(f"an owner must be 1..{MAX_OWNER} characters")
    if not value.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("an owner must be printable ASCII")
    return value


def _owner_field(obj: dict, key: str = "owner") -> str:
    try:
        return _owner_name(obj[key])
    except ValueError as exc:
        raise _bad(str(exc)) from None


def _publish_id(value) -> str:
    if not isinstance(value, str) or len(value) != PUBLISH_ID_LEN or any(c not in "0123456789abcdef" for c in value):
        raise _bad(f'"publish_id" must be exactly {PUBLISH_ID_LEN} lowercase hex characters')
    return value


def _stages(value) -> tuple:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_STAGES:
        raise _bad(f'"stages" must be a list of 1..{MAX_STAGES} (lo, hi) pairs')
    out = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise _bad('a stage must be a [lo, hi] pair')
        lo = _whole(pair[0], "stage lo", lo=-(1 << 63), hi=(1 << 63) - 1)
        hi = _whole(pair[1], "stage hi", lo=-(1 << 63), hi=(1 << 63) - 1)
        if lo > hi:
            raise _bad(f"stage {lo}..{hi} is empty")
        out.append((lo, hi))
    return tuple(out)


def _images(value) -> tuple:
    if not isinstance(value, list):
        raise _bad('"images" must be a list')
    out = []
    for item in value:
        if not isinstance(item, dict):
            raise _bad("an image must be an object")
        _keys(item, ("start", "hash"))
        out.append((_i64(item, "start"), _u64(item, "hash")))
    return tuple(out)


def _points(value) -> tuple:
    if not isinstance(value, list) or len(value) > MAX_POINTS:
        raise _bad(f'"points" must be a list of at most {MAX_POINTS} points')
    out = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise _bad("a point must be an object")
        _keys(item, ("tokens", "hash", "imgs", "live", "pinned"))
        live = _bool(item, "live")
        if live and index != 0:
            raise _bad("only the first point may be the live state")
        out.append(_Point(_i64(item, "tokens", lo=1), _u64(item, "hash"), _images(item["imgs"]), live,
                          _bool(item, "pinned")))
    return tuple(out)


def _u64_list(value, name: str, *, max_len: int, lo: int = 0) -> list:
    if not isinstance(value, list) or len(value) > max_len:
        raise _bad(f'"{name}" must be a list of at most {max_len} whole numbers')
    return [_whole(item, name, lo=lo, hi=MASK64) for item in value]


def _hashes(value) -> list:
    if not isinstance(value, list):
        raise _bad('"hashes" must be a list')
    out = []
    for item in value:
        if not isinstance(item, dict):
            raise _bad("a hash must be an object")
        _keys(item, ("tokens", "hash"))
        out.append((_i64(item, "tokens", lo=1), _u64(item, "hash")))
    return out


def _images_below(images: tuple, tokens: int) -> tuple:
    return tuple(image for image in images if image[0] < tokens)


def _same_point(a: _Point, b: _Point) -> bool:
    """The disk tier's same_point: tokens, hash and pictures.  live/pinned are runtime metadata, not identity."""
    return a.tokens == b.tokens and a.hash == b.hash and a.imgs == b.imgs


def _deepest_checkpoint(points: tuple) -> _Point | None:
    deepest = None
    for point in points:
        if point.live:
            continue
        if deepest is None or point.tokens > deepest.tokens:
            deepest = point
    return deepest


def _deepest_point(entry: _Entry) -> _Point | None:
    """The point an ACQUIRE without a remembered MATCH_END restores: the deepest one, live on a tie."""
    best = None
    for point in entry.points:
        if best is None or point.tokens > best.tokens or (point.tokens == best.tokens and point.live):
            best = point
    return best


def _read_charge(stage_bytes) -> int:
    return sum(session_read_peak_bytes(size) for size in stage_bytes)


def _replica_number(owner: str, fallback: int) -> int:
    if len(owner) > 1 and owner[0] == "r" and owner[1:].isdigit():
        return int(owner[1:])
    return fallback


def _tune(sock: socket.socket) -> None:
    """A 1 MiB frame must fit one SOCK_SEQPACKET message: the sender's SO_SNDBUF bounds it."""
    for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, option, SOCKET_BUFFER)
        except OSError:
            pass


def _send_timeout(sock: socket.socket, seconds: float) -> None:
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO,
                        struct.pack("@ll", int(seconds), int((seconds % 1.0) * 1_000_000)))
    except OSError:
        pass


# ---- the store
class ParentParkingStore:
    """The parent-side store: the index, the global LRU, the memfds and the one broker thread that serves every
    replica's socketpair.  `open_channel` gives the server a child fd to pass to an engine; `snapshot` is what
    /metrics reads; `clear` is the global clear after #879 or unload; `close` frees every fd and socket."""

    def __init__(self, config: ParentParkingConfig, identity=None, *, capacity_bytes: int | None = None):
        if not isinstance(config, ParentParkingConfig):
            raise TypeError("ParentParkingStore needs a ParentParkingConfig")
        if not (hasattr(os, "memfd_create") and hasattr(fcntl, "F_ADD_SEALS") and hasattr(socket, "send_fds")):
            raise ParentParkingConfigError("parent parking needs Linux memfd_create, seals and SCM_RIGHTS")
        self._config = config
        self._identity = _identity_of(identity)
        self._capacity = int(capacity_bytes) if capacity_bytes is not None else config.capacity_bytes
        if self._capacity <= 0:
            raise ValueError("parent parking capacity must be positive")
        self._lock = threading.Lock()
        self._closed = False
        self._sel = None
        self._thread = None
        # index and accounting (only the broker changes these)
        self._entries: dict[int, _Entry] = {}
        self._lru: collections.OrderedDict = collections.OrderedDict()
        self._pending: dict[int, _Pending] = {}
        self._pending_by_publish: dict[str, int] = {}
        self._committed_by_publish: collections.OrderedDict = collections.OrderedDict()
        self._matches: dict[int, _Match] = {}
        self._leases: dict[int, _Lease] = {}
        self._retired: dict[int, _Retired] = {}
        self._connections: dict[int, _Connection] = {}
        self._owners: dict[str, _Connection] = {}
        self._owner_state: dict[str, str] = {}
        self._owner_stats: dict[str, _OwnerStats] = {}
        self._owner_index: dict[str, int] = {}
        self._used_ids: set = set()
        self._entry_bytes = 0
        self._pending_bytes = 0
        self._retired_bytes = 0
        self._restore_reserved = 0
        self._published = 0
        self._published_bytes = 0
        self._deduplicated = 0
        self._reads = 0
        self._peer_reads = 0
        self._evictions = 0
        self._eviction_skipped_leased = 0
        self._superseded = 0
        self._clears = 0
        self._damaged = 0
        self._capacity_refusals = 0
        self._store_epoch = 1
        self._generation = 0
        self._clock = 0
        self._next_pending_id = 0
        self._next_lease_id = 0
        self._next_cookie = 0
        self._next_conn = 0
        # thread plumbing
        self._incoming: list = []
        self._commands: list = []
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        os.set_blocking(self._wake_w, False)
        self._thread = threading.Thread(target=self._run, name="strata-parent-parking", daemon=True)
        self._thread.start()
        atexit.register(self._atexit_close)

    # ---- public API (what serve/server.py and serve/replicas.py call)
    def open_channel(self, owner: str) -> tuple[int, ParentChannel]:
        """A new channel for one replica: (the child fd to pass to the engine, the server's handle).  Called before
        every spawn, so a channel that already exists for this owner is replaced (its pending and leases are
        cancelled as on EOF; committed snapshots stay)."""
        owner = _owner_name(owner)
        with self._lock:
            if self._closed:
                raise ParentParkingClosed("the parent parking store is closed")
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET, 0)
            try:
                _tune(parent)
                _tune(child)
                _send_timeout(parent, max(2.0, self._config.ipc_timeout_s))
                conn_id = self._next_conn
                self._next_conn += 1
                self._incoming.append((parent, owner, conn_id))
                self._wake()
            except BaseException:
                parent.close()
                child.close()
                raise
        return child.detach(), ParentChannel(self, owner, conn_id)

    def snapshot(self) -> dict:
        """The store's counters and per-replica state, exactly the keys of plan §14.2 (what /metrics and /v1/status
        add as "conversation_parent_parking")."""
        with self._lock:
            return self._snapshot_locked()

    def clear(self, reason: str) -> int:
        """Remove every committed snapshot of every owner, cancel every pending publish and bump the store epoch;
        readers that already hold a lease finish their restore on the old fds.  Returns how many entries went."""
        if reason not in CLEAR_REASONS:
            raise ValueError(f"unknown clear reason {reason!r}")
        if self._closed:
            return 0
        return int(self._post(lambda: self._clear(reason)) or 0)

    def close(self) -> None:
        """Stop the broker and free every socket, memfd and wake pipe.  Idempotent; atexit calls it as a safety net."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._wake()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=BROKER_JOIN_S)
            if thread.is_alive():                       # a peer that stopped reading: unblock the broker's send
                for conn in list(self._connections.values()):
                    try:
                        conn.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                thread.join(timeout=5.0)
        for fd in (self._wake_r, self._wake_w):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._wake_r = self._wake_w = -1
        try:
            atexit.unregister(self._atexit_close)
        except Exception:                               # noqa: BLE001 - unregistering is best effort
            pass

    def replica_state(self, owner: str) -> str:
        """connected (a channel is up), degraded (its channel went away; a restart gets a new one) or closed."""
        with self._lock:
            return self._owner_state.get(owner, "closed")

    # ---- thread plumbing
    def _atexit_close(self) -> None:
        try:
            self.close()
        except Exception:                               # noqa: BLE001 - the interpreter is going away
            pass

    def _wake(self) -> None:
        try:
            os.write(self._wake_w, b"x")
        except OSError:
            pass

    def _drain_wake(self) -> None:
        while True:
            try:
                if not os.read(self._wake_r, 4096):
                    return
            except OSError:
                return

    def _post(self, fn):
        """Run `fn` in the broker thread (the only writer of the index) and return its result."""
        if threading.current_thread() is self._thread:
            raise RuntimeError("parent parking: the broker cannot wait for itself")
        if self._closed:
            return None
        event = threading.Event()
        box: list = []
        with self._lock:
            if self._closed:
                return None
            self._commands.append((fn, box, event))
            self._wake()
        if not event.wait(timeout=COMMAND_TIMEOUT_S):
            raise RuntimeError("parent parking: the broker did not answer")
        if not box:
            return None
        if isinstance(box[0], BaseException):
            raise box[0]
        return box[0]

    def _run(self) -> None:
        sel = selectors.DefaultSelector()
        self._sel = sel
        try:
            sel.register(self._wake_r, selectors.EVENT_READ, None)
            while True:
                with self._lock:
                    incoming, self._incoming = self._incoming, []
                    commands, self._commands = self._commands, []
                    closed = self._closed
                if closed:                              # a waiting clear() must not hang on a stopping broker
                    for _fn, box, event in commands:
                        box.append(None)
                        event.set()
                    break
                for parent, owner, conn_id in incoming:
                    try:
                        with self._lock:
                            self._register(sel, parent, owner, conn_id)
                    except Exception:                   # noqa: BLE001 - one bad channel must not stop the broker
                        try:
                            parent.close()
                        except OSError:
                            pass
                for fn, box, event in commands:
                    try:
                        with self._lock:
                            box.append(fn())
                    except BaseException as exc:        # noqa: BLE001 - report it to the waiting thread
                        box.append(exc)
                    finally:
                        event.set()
                timeout = self._select_timeout()
                try:
                    events = sel.select(timeout)
                except OSError:
                    break
                for key, _mask in events:
                    if key.fileobj is self._wake_r:
                        self._drain_wake()
                    else:
                        self._serve(key.data)
                with self._lock:
                    self._expire_matches()
        finally:
            with self._lock:
                self._shutdown()
            self._sel = None

    def _select_timeout(self):
        with self._lock:
            if not self._matches:
                return None
            now = time.monotonic()
            return max(0.0, min(match.expiry - now for match in self._matches.values()))

    def _expire_matches(self) -> None:
        if not self._matches:
            return
        now = time.monotonic()
        for cookie, match in list(self._matches.items()):
            if match.expiry <= now:
                self._matches.pop(cookie, None)
                if match.conn.match == cookie:
                    match.conn.match = None

    # ---- channel lifecycle (broker thread, lock held)
    def _register(self, sel, sock: socket.socket, owner: str, conn_id: int) -> None:
        old = self._owners.get(owner)
        if old is not None:
            self._drop(old, "replaced")
        conn = _Connection(conn_id, sock, owner)
        self._connections[conn_id] = conn
        self._owners[owner] = conn
        self._owner_state[owner] = "connected"
        self._owner_stats.setdefault(owner, _OwnerStats())
        if owner not in self._owner_index:
            self._owner_index[owner] = _replica_number(owner, len(self._owner_index))
        sel.register(sock, selectors.EVENT_READ, conn)

    def _drop(self, conn: _Connection, reason: str) -> None:
        """EOF, a transport error or a replacement: cancel this channel's pending and leases, keep committed entries."""
        if self._connections.get(conn.conn_id) is not conn:
            return
        try:
            if self._sel is not None:
                self._sel.unregister(conn.sock)
        except (KeyError, ValueError, OSError):
            pass
        # Drain unread rx frames before close() so Linux unix_release_sock does
        # not raise ECONNRESET on the peer socket ahead of an already-queued OP_ERROR.
        # Note: Python stdlib socket.recv_fds ignores its `flags` argument, so switch
        # the closing socket to non-blocking before draining.
        try:
            conn.sock.setblocking(False)
            for _ in range(64):
                try:
                    data, fds, _flags, _addr = socket.recv_fds(
                        conn.sock, HEADER.size + MAX_PAYLOAD + 1, MAX_FDS
                    )
                except (BlockingIOError, InterruptedError, OSError):
                    break
                for fd in fds:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                if not data:
                    break
        except OSError:
            pass
        try:
            conn.sock.close()
        except OSError:
            pass
        for pending_id in list(conn.pending):
            pending = self._pending.get(pending_id)
            if pending is not None:
                self._cancel_pending(pending)
        for lease_id in list(conn.leases):
            lease = self._leases.get(lease_id)
            if lease is not None:
                self._release_lease(lease, None)
        if conn.match is not None:
            self._matches.pop(conn.match, None)
            conn.match = None
        self._connections.pop(conn.conn_id, None)
        if conn.owner is not None and self._owners.get(conn.owner) is conn:
            del self._owners[conn.owner]
            self._owner_state[conn.owner] = "closed" if self._closed else "degraded"

    def _channel_state(self, conn_id: int) -> str:
        with self._lock:
            if conn_id in self._connections:
                return "connected"
            return "closed" if self._closed else "degraded"

    def _close_channel(self, conn_id: int) -> None:
        if self._closed:
            return
        self._post(lambda: self._drop_id(conn_id))

    def _drop_id(self, conn_id: int) -> None:
        conn = self._connections.get(conn_id)
        if conn is not None:
            self._drop(conn, "closed")

    # ---- one request (broker thread)
    def _serve(self, conn: _Connection) -> None:
        try:
            data, fds, flags, _addr = socket.recv_fds(conn.sock, HEADER.size + MAX_PAYLOAD + 1, MAX_FDS)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            with self._lock:
                self._drop(conn, "error")
            return
        if not data:
            with self._lock:
                self._drop(conn, "eof")
            return
        request_id = 0
        if len(data) >= HEADER.size:
            request_id = HEADER.unpack_from(data)[5]
        close_after = False
        out_fds: list = []
        with self._lock:
            try:
                if flags & socket.MSG_CTRUNC:
                    raise _bad("too many file descriptors in one message")
                op, payload, out_fds = self._handle_frame(conn, data, fds)
            except _RequestError as exc:
                op, payload, out_fds = OP_ERROR, exc.payload(), []
                close_after = exc.close
                if conn.owner is not None and (exc.close or exc.code == "protocol"):
                    stats = self._owner_stats.get(conn.owner)
                    if stats is not None:
                        stats.ipc_errors += 1
            except Exception:                           # noqa: BLE001 - a store bug must not kill the broker
                op, payload, out_fds = OP_ERROR, {"code": "internal", "message": "parent parking store error",
                                                  "retryable": True}, []
        for fd in fds:                                  # a request never carries fds; close whatever came
            try:
                os.close(fd)
            except OSError:
                pass
        if not close_after and self._pending_input(conn):
            second = self._peek_request_id(conn)
            op, payload, out_fds = OP_ERROR, {"code": "bad_request",
                                              "message": "a second request before the answer",
                                              "retryable": False}, []
            request_id = second or request_id
            close_after = True
        if not self._send(conn, op, request_id, payload, out_fds) or close_after:
            with self._lock:
                self._drop(conn, "closed")

    def _pending_input(self, conn: _Connection) -> bool:
        """A frame already queued before this answer was sent: the client broke the one-outstanding rule."""
        try:
            return bool(conn.sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT))
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return False

    def _peek_request_id(self, conn: _Connection) -> int:
        try:
            head = conn.sock.recv(HEADER.size, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        except (BlockingIOError, InterruptedError, OSError):
            return 0
        if len(head) < HEADER.size:
            return 0
        try:
            return HEADER.unpack_from(head)[5]
        except struct.error:
            return 0

    def _send(self, conn: _Connection, op: int, request_id: int, payload: dict, fds) -> bool:
        try:
            body = json.dumps(payload, separators=(",", ":")).encode("ascii")
        except (TypeError, ValueError):
            return False
        header = HEADER.pack(MAGIC, VERSION, op, 0, len(body), request_id, session_hash64(body, HASH_SEED))
        data = header + body
        try:
            if fds:
                sent = socket.send_fds(conn.sock, [data], list(fds))
            else:
                sent = conn.sock.send(data)
        except OSError:
            return False
        return sent == len(data)

    def _handle_frame(self, conn: _Connection, data: bytes, fds) -> tuple:
        if len(data) < HEADER.size:
            raise _bad("a frame is shorter than its header")
        magic, version, op, flags, payload_bytes, request_id, payload_hash = HEADER.unpack_from(data)
        if magic != MAGIC:
            raise _bad("bad frame magic")
        if version != VERSION:
            raise _RequestError("protocol", "unknown protocol version", close=True)
        if flags != 0:
            raise _bad("nonzero frame flags")
        if payload_bytes > MAX_PAYLOAD:
            raise _bad("a payload larger than the protocol allows")
        if len(data) != HEADER.size + payload_bytes:
            raise _bad("the frame length does not match payload_bytes")
        if request_id == 0:
            raise _bad("a zero request_id")
        if request_id <= conn.last_request_id:
            raise _bad("request_id is not increasing")
        payload = data[HEADER.size:]
        if session_hash64(payload, HASH_SEED) != payload_hash:
            raise _bad("payload hash mismatch")
        if fds:
            raise _bad("a request carries no file descriptors")
        conn.last_request_id = request_id
        handler = _HANDLERS.get(op)
        if handler is None:
            raise _RequestError("protocol", "unknown operation", close=True)
        return handler(self, conn, _parse_json(payload))

    def _need_hello(self, conn: _Connection) -> None:
        if not conn.hello:
            raise _RequestError("protocol", "HELLO first")

    # ---- operations (contract §4)
    def _op_hello(self, conn: _Connection, req: dict) -> tuple:
        _keys(req, ("owner", "protocol", "model", "config", "layout_key", "stages", "pid"))
        owner = _owner_field(req)
        protocol = _int(req, "protocol", hi=0xFFFFFFFF)
        if protocol != VERSION:
            raise _RequestError("protocol", "unknown protocol version", close=True)
        identity = ParentParkingIdentity(_u64(req, "model"), _u64(req, "config"), _u64(req, "layout_key"),
                                         _stages(req["stages"]))
        _int(req, "pid", lo=-(1 << 63), hi=(1 << 63) - 1)
        if self._identity is None:
            self._identity = identity
        elif identity != self._identity:
            raise _RequestError("identity", "identity does not match the store", close=True)
        if conn.owner is not None and conn.owner != owner:
            raise _RequestError("identity", "the channel already has another owner", close=True)
        other = self._owners.get(owner)
        if other is not None and other is not conn:
            raise _RequestError("identity", "owner already has a channel", close=True)
        conn.owner = owner
        conn.hello = True
        self._owners[owner] = conn
        self._owner_state[owner] = "connected"
        self._owner_stats.setdefault(owner, _OwnerStats())
        if owner not in self._owner_index:
            self._owner_index[owner] = _replica_number(owner, len(self._owner_index))
        return OP_HELLO, {"store_epoch": self._store_epoch, "entries": len(self._entries),
                          "bytes": self._entry_bytes, "limits": dict(LIMITS)}, []

    def _op_advice(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ("cvec", "points"))
        cvec = _bool(req, "cvec")
        points = _points(req["points"])
        holding = self._holding(cvec, points)
        if holding is not None:
            self._touch(holding)
        return OP_PARK_ADVICE, {"holding_id": holding.id if holding is not None else 0,
                                "parked_prefix_tokens": self._parked_prefix(cvec, points),
                                "entries": len(self._entries), "bytes": self._entry_bytes,
                                "epoch": self._store_epoch}, []

    def _op_begin(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ("publish_id", "owner", "model", "config", "layout_key", "stages", "cvec", "points",
                    "stage_bytes", "epoch"))
        publish_id = _publish_id(req["publish_id"])
        owner = _owner_field(req)
        if owner != conn.owner:
            raise _bad("owner does not match the channel")
        model = _u64(req, "model")
        config = _u64(req, "config")
        layout_key = _u64(req, "layout_key")
        stages = _stages(req["stages"])
        if (model, config, layout_key, stages) != self._identity.as_tuple():
            raise _RequestError("identity", "identity does not match the store")
        cvec = _bool(req, "cvec")
        points = _points(req["points"])
        stage_bytes = _u64_list(req["stage_bytes"], "stage_bytes", max_len=MAX_STAGES)
        if len(stage_bytes) != len(stages):
            raise _bad("stage_bytes must name one size per stage")
        if sum(stage_bytes) > MASK64:
            raise _bad("the stage sizes overflow")
        epoch = _u64(req, "epoch")
        if epoch != self._store_epoch:
            raise _RequestError("stale_epoch", "the store has another epoch")
        meta = (owner, model, config, layout_key, stages, cvec, points, tuple(stage_bytes))
        pending_id = self._pending_by_publish.get(publish_id)
        if pending_id is not None:
            pending = self._pending.get(pending_id)
            if pending is None or pending.conn is not conn or pending.meta != meta:
                raise _bad("publish_id reused with other metadata", close=False)
            return OP_PARK_BEGIN, {"pending_id": pending.id, "stage_bytes": list(pending.stage_bytes),
                                   "dedup": False, "holding_id": 0}, list(pending.fds)
        committed = self._committed_by_publish.get(publish_id)
        if committed is not None:
            entry = self._entries.get(committed[0])
            if entry is not None and entry.meta != meta:
                raise _bad("publish_id already committed with other metadata", close=False)
            return OP_PARK_BEGIN, {"pending_id": 0, "stage_bytes": [], "dedup": True,
                                   "holding_id": committed[0] if entry is not None else 0}, []
        holding = self._holding(cvec, points)
        if holding is not None:
            self._touch(holding)
            self._deduplicated += 1
            return OP_PARK_BEGIN, {"pending_id": 0, "stage_bytes": [], "dedup": True, "holding_id": holding.id}, []
        incoming = sum(stage_bytes)
        self._evict(self._eviction_plan(incoming, slots=1))
        fds: list = []
        try:
            for size in stage_bytes:
                fd = os.memfd_create(MEMFD_NAME, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
                try:
                    os.ftruncate(fd, size)
                except OSError:
                    os.close(fd)
                    raise
                fds.append(fd)
        except OSError:
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise _RequestError("internal", "cannot create a stage snapshot") from None
        self._next_pending_id += 1
        pending = _Pending(self._next_pending_id, publish_id, conn, owner, model, config, layout_key, stages, cvec,
                           points, tuple(stage_bytes), fds, incoming, epoch, meta)
        self._pending[pending.id] = pending
        self._pending_by_publish[publish_id] = pending.id
        self._pending_bytes += incoming
        conn.pending.add(pending.id)
        return OP_PARK_BEGIN, {"pending_id": pending.id, "stage_bytes": list(pending.stage_bytes),
                               "dedup": False, "holding_id": 0}, list(pending.fds)

    def _op_commit(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ("publish_id", "pending_id", "written_bytes", "epoch"))
        publish_id = _publish_id(req["publish_id"])
        pending_id = _u64(req, "pending_id", lo=1)
        written = _u64_list(req["written_bytes"], "written_bytes", max_len=MAX_STAGES)
        epoch = _u64(req, "epoch")
        if epoch != self._store_epoch:
            pending = self._pending.get(pending_id)
            if pending is not None and pending.conn is conn and pending.publish_id == publish_id:
                self._cancel_pending(pending)
            raise _RequestError("stale_epoch", "the store has another epoch")
        committed = self._committed_by_publish.get(publish_id)
        if committed is not None:
            # the answer is repeated only while the entry it names is still there: an entry evicted or cleared since
            # must not be reported as published (the engine would log a park that no reader can find)
            if committed[0] in self._entries:
                return OP_PARK_COMMIT, {"id": committed[0], "generation": committed[1], "entry_bytes": committed[2],
                                        "dedup": False, "entries": len(self._entries), "bytes": self._entry_bytes,
                                        "evictions": 0, "superseded": 0}, []
            self._committed_by_publish.pop(publish_id, None)
            raise _RequestError("gone", "the entry this publish id named is gone")
        pending = self._pending.get(pending_id)
        if pending is None or pending.conn is not conn or pending.publish_id != publish_id:
            raise _RequestError("gone", "no such pending publish")
        if len(written) != len(pending.stage_bytes) or any(a != b for a, b in zip(written, pending.stage_bytes)):
            self._cancel_pending(pending)
            raise _bad("written_bytes do not match the reserved sizes", close=False)
        uid = os.getuid() if hasattr(os, "getuid") else 0
        for fd, size in zip(pending.fds, pending.stage_bytes):
            try:
                info = os.fstat(fd)
            except OSError:
                self._cancel_pending(pending)
                raise _bad("a stage fd is not open", close=False) from None
            if not stat.S_ISREG(info.st_mode) or info.st_size != size or info.st_uid != uid:
                self._cancel_pending(pending)
                raise _bad("a stage fd is not the reserved regular file", close=False)
        try:
            for fd in pending.fds:
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
        except OSError:
            self._cancel_pending(pending)
            raise _RequestError("internal", "cannot seal a stage snapshot") from None
        entry = _Entry(0, 0, publish_id, pending.owner, pending.model, pending.config, pending.layout_key,
                       pending.stages, pending.cvec, pending.points, pending.stage_bytes, pending.fds,
                       pending.bytes, 0, any(point.pinned for point in pending.points), set(), pending.meta)
        entry.id = self._new_id()
        pending.fds = []                                # the entry owns the memfds from here
        self._pending.pop(pending.id, None)
        self._pending_by_publish.pop(publish_id, None)
        self._pending_bytes -= pending.bytes
        conn.pending.discard(pending.id)
        self._generation += 1
        entry.generation = self._generation
        self._clock += 1
        entry.used = self._clock
        self._entries[entry.id] = entry
        self._lru[entry.id] = None
        self._entry_bytes += entry.bytes
        self._published += 1
        self._published_bytes += entry.bytes
        stats = self._owner_stats.get(entry.owner)
        if stats is not None:
            stats.published += 1
        self._committed_by_publish[publish_id] = (entry.id, entry.generation, entry.bytes)
        while len(self._committed_by_publish) > COMMITTED_MEMORY:
            self._committed_by_publish.popitem(last=False)
        superseded = self._drop_superseded(entry)
        return OP_PARK_COMMIT, {"id": entry.id, "generation": entry.generation, "entry_bytes": entry.bytes,
                                "dedup": False, "entries": len(self._entries), "bytes": self._entry_bytes,
                                "evictions": 0, "superseded": superseded}, []

    def _op_abort(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ("publish_id", "pending_id"))
        publish_id = _publish_id(req["publish_id"])
        pending_id = _u64(req, "pending_id", lo=1)
        pending = self._pending.get(pending_id)
        if pending is None or pending.conn is not conn or pending.publish_id != publish_id:
            return OP_PARK_ABORT, {"gone": True}, []
        self._cancel_pending(pending)
        return OP_PARK_ABORT, {"gone": False}, []

    def _op_match_begin(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ("prompt_tokens", "cvec", "images", "exclude_ids"))
        prompt_tokens = _i64(req, "prompt_tokens")
        cvec = _bool(req, "cvec")
        images = _images(req["images"])
        exclude = set(_u64_list(req["exclude_ids"], "exclude_ids", max_len=MAX_EXCLUDE_IDS, lo=1))
        lengths: set = set()
        candidates: set = set()
        for entry in self._entries.values():
            if entry.cvec != cvec or entry.id in exclude:
                continue
            for point in entry.points:
                if 1 <= point.tokens <= prompt_tokens - 1 and _images_below(images, point.tokens) == point.imgs:
                    lengths.add(point.tokens)
                    candidates.add(entry.id)
        if conn.match is not None:
            self._matches.pop(conn.match, None)
            conn.match = None
        conn.last_candidate = None
        self._next_cookie += 1
        ordered = tuple(sorted(lengths))
        self._matches[self._next_cookie] = _Match(self._next_cookie, conn, prompt_tokens, cvec, images, ordered,
                                                  frozenset(candidates),
                                                  time.monotonic() + 4.0 * self._config.ipc_timeout_s)
        conn.match = self._next_cookie
        return OP_MATCH_BEGIN, {"cookie": self._next_cookie, "lengths": list(ordered)}, []

    def _op_match_end(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ("cookie", "hashes"))
        cookie = _u64(req, "cookie", lo=1)
        hashes = _hashes(req["hashes"])
        match = self._matches.get(cookie)
        if match is None or match.conn is not conn or match.expiry <= time.monotonic():
            if match is not None:
                self._matches.pop(cookie, None)
                if match.conn.match == cookie:
                    match.conn.match = None
            raise _RequestError("gone", "no such match")
        self._matches.pop(cookie, None)
        conn.match = None
        if len(hashes) != len(match.lengths) or sorted(tokens for tokens, _hash in hashes) != list(match.lengths):
            raise _bad("hashes do not cover the lengths", close=False)
        hash_at = dict(hashes)
        best = None
        for entry_id in match.candidates:
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            for point in entry.points:
                if point.tokens not in hash_at or hash_at[point.tokens] != point.hash:
                    continue
                if _images_below(match.images, point.tokens) != point.imgs:
                    continue
                if best is None or _better_point(point, entry, best[0], best[1]):
                    best = (point, entry)
        if best is None:
            return OP_MATCH_END, {"candidate": None}, []
        point, entry = best
        borrow = self._borrows(entry, point.tokens, point.live)
        charge = _read_charge(entry.stage_bytes)
        conn.last_candidate = (entry.id, entry.generation, point.tokens, point.live, borrow)
        return OP_MATCH_END, {"candidate": {"id": entry.id, "generation": entry.generation, "tokens": point.tokens,
                                            "live": point.live, "borrow": borrow,
                                            "stage_bytes": list(entry.stage_bytes),
                                            "read_charge_bytes": charge, "owner": entry.owner}}, []

    def _op_acquire(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ("id", "generation"))
        entry_id = _u64(req, "id", lo=1)
        generation = _u64(req, "generation", lo=1)
        try:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise _RequestError("gone", "no such entry")
            if entry.generation != generation:
                raise _RequestError("stale_generation", "the entry has another generation")
            candidate = conn.last_candidate
            if candidate is not None and candidate[0] == entry_id and candidate[1] == generation:
                tokens, live, borrow = candidate[2], candidate[3], candidate[4]
            else:
                point = _deepest_point(entry)
                tokens = point.tokens if point is not None else 0
                live = point.live if point is not None else False
                borrow = self._borrows(entry, tokens, live)
            charge = _read_charge(entry.stage_bytes)
            self._evict(self._eviction_plan(charge, protect=(entry_id,)))
            self._next_lease_id += 1
            lease = _Lease(self._next_lease_id, entry_id, generation, conn, conn.owner, entry.stage_bytes, charge)
            self._leases[lease.id] = lease
            conn.leases.add(lease.id)
            entry.leases.add(lease.id)
            self._restore_reserved += charge
            self._touch(entry)
            self._reads += 1
            stats = self._owner_stats.get(conn.owner)
            if stats is not None:
                stats.reads += 1
                if entry.owner != conn.owner:
                    stats.peer_reads += 1
            if entry.owner != conn.owner:
                self._peer_reads += 1
            return OP_ACQUIRE, {"lease_id": lease.id, "id": entry.id, "generation": entry.generation,
                                "tokens": tokens, "live": live, "borrow": borrow,
                                "stage_bytes": list(entry.stage_bytes),
                                "read_charge_bytes": charge, "owner": entry.owner}, list(entry.fds)
        except _RequestError:
            stats = self._owner_stats.get(conn.owner)
            if stats is not None:
                stats.fallbacks += 1
            raise

    def _op_release(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ("lease_id", "result"))
        lease_id = _u64(req, "lease_id", lo=1)
        result = _text(req["result"], "result")
        if result not in RELEASE_RESULTS:
            raise _bad("unknown release result", close=False)
        lease = self._leases.get(lease_id)
        if lease is None or lease.conn is not conn:
            return OP_RELEASE, {"gone": True}, []
        self._release_lease(lease, result)
        return OP_RELEASE, {"gone": False}, []

    def _op_clear(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ("reason",))
        reason = _text(req["reason"], "reason")
        if reason not in CLEAR_REASONS:
            raise _bad("unknown clear reason", close=False)
        return OP_CLEAR, {"cleared": self._clear(reason), "epoch": self._store_epoch}, []

    def _op_stats(self, conn: _Connection, req: dict) -> tuple:
        self._need_hello(conn)
        _keys(req, ())
        return OP_STATS, self._snapshot_locked(), []

    def _op_ping(self, conn: _Connection, req: dict) -> tuple:
        _keys(req, ())
        return OP_PING, {"epoch": self._store_epoch}, []

    # ---- index helpers (broker thread, lock held)
    def _new_id(self) -> int:
        while True:
            value = random.SystemRandom().getrandbits(63) | 1
            if value not in self._used_ids:
                self._used_ids.add(value)
                return value

    def _touch(self, entry: _Entry) -> None:
        self._clock += 1
        entry.used = self._clock
        if entry.id in self._lru:
            self._lru.move_to_end(entry.id)

    def _holding(self, cvec: bool, points: tuple) -> _Entry | None:
        """The committed snapshot with exactly this cvec and this ordered point list.  The whole point is compared,
        live and pinned included: two publishes are the same snapshot only when the engine sent the same points."""
        for entry in self._entries.values():
            if entry.cvec == cvec and entry.points == points:
                return entry
        return None

    def _parked_prefix(self, cvec: bool, points: tuple) -> int:
        """The deepest checkpoint of a committed snapshot that this outgoing state holds (parked_prefix_tokens)."""
        live = next((point for point in points if point.live), None)
        checkpoints = [point for point in points if not point.live]

        def held(point: _Point) -> bool:
            if live is not None and _same_point(point, live):
                return True
            return any(_same_point(point, other) for other in checkpoints)

        longest = 0
        for entry in self._entries.values():
            if entry.cvec != cvec:
                continue
            deepest = _deepest_checkpoint(entry.points)
            if deepest is None or deepest.tokens <= 0:
                continue
            if held(deepest):
                longest = max(longest, deepest.tokens)
        return longest

    def _borrows(self, entry: _Entry, tokens: int, live: bool) -> bool:
        """ConversationCache::borrows: not live, shallower than the deepest checkpoint, no pinned checkpoint at or
        below the match."""
        if tokens <= 0 or live:
            return False
        deepest = 0
        for point in entry.points:
            if point.live:
                continue
            if point.pinned and point.tokens <= tokens:
                return False
            deepest = max(deepest, point.tokens)
        return tokens < deepest

    def _drop_superseded(self, new_entry: _Entry) -> int:
        """ConversationDiskTier::drop_superseded: an older snapshot whose deepest checkpoint the new one still holds
        only adds the tail the client rewrote, so it goes (a leased one is retired, not freed)."""
        live = next((point for point in new_entry.points if point.live), None)
        checkpoints = [point for point in new_entry.points if not point.live]

        def held(point: _Point) -> bool:
            if live is not None and _same_point(point, live):
                return True
            return any(_same_point(point, other) for other in checkpoints)

        dropped = 0
        for entry in list(self._entries.values()):
            if entry.id == new_entry.id or entry.cvec != new_entry.cvec:
                continue
            deepest = _deepest_checkpoint(entry.points)
            if deepest is None or deepest.tokens <= 0:
                continue
            if held(deepest):
                self._remove_entry(entry)
                dropped += 1
        self._superseded += dropped
        return dropped

    def _remove_entry(self, entry: _Entry) -> None:
        """Take a snapshot out of the index.  A leased one keeps its bytes charged (retired) until the last lease."""
        self._entries.pop(entry.id, None)
        self._lru.pop(entry.id, None)
        self._entry_bytes -= entry.bytes
        if entry.leases:
            self._retired[entry.id] = _Retired(entry.id, entry.bytes, set(entry.leases))
            self._retired_bytes += entry.bytes
        for fd in entry.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        entry.fds = []

    def _cancel_pending(self, pending: _Pending) -> None:
        self._pending.pop(pending.id, None)
        self._pending_by_publish.pop(pending.publish_id, None)
        self._pending_bytes -= pending.bytes
        pending.conn.pending.discard(pending.id)
        for fd in pending.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        pending.fds = []

    def _release_lease(self, lease: _Lease, result: str | None) -> None:
        self._leases.pop(lease.id, None)
        lease.conn.leases.discard(lease.id)
        self._restore_reserved -= lease.read_charge
        entry = self._entries.get(lease.entry_id)
        if entry is not None:
            entry.leases.discard(lease.id)
        retired = self._retired.get(lease.entry_id)
        if retired is not None:
            retired.leases.discard(lease.id)
            if not retired.leases:
                self._retired.pop(lease.entry_id, None)
                self._retired_bytes -= retired.bytes
        if result in ("invalid", "mismatch"):
            self._damaged += 1
            if entry is not None:
                self._remove_entry(entry)

    def _evict(self, plan: list) -> None:
        for entry_id in plan:
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            self._remove_entry(entry)
            self._evictions += 1

    def _charged(self) -> int:
        return self._entry_bytes + self._pending_bytes + self._retired_bytes + self._restore_reserved

    def _eviction_plan(self, incoming_bytes: int, *, slots: int = 0, protect=()) -> list:
        """The whole eviction plan for `incoming_bytes` (and `slots` new entries), oldest first.  Nothing is deleted
        here: a plan that cannot reach the target raises capacity/pinned_working_set and changes nothing."""
        need_bytes = max(0, self._charged() + incoming_bytes - self._capacity)
        need_slots = max(0, len(self._entries) + len(self._pending) + slots - self._config.max_entries)
        if need_bytes == 0 and need_slots == 0:
            return []
        freed = 0
        plan: list = []
        skipped_leased = 0
        skipped_pinned = 0
        for entry_id in list(self._lru):
            if freed >= need_bytes and len(plan) >= need_slots:
                break
            if entry_id in protect:
                continue
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            if entry.leases:
                skipped_leased += 1
                continue
            if entry.pinned:
                skipped_pinned += 1
                continue
            plan.append(entry_id)
            freed += entry.bytes
        if freed < need_bytes or len(plan) < need_slots:
            self._capacity_refusals += 1
            if not plan and (skipped_leased or skipped_pinned):
                raise _RequestError("pinned_working_set", "everything that could be evicted is pinned or leased")
            raise _RequestError("capacity", "the budget cannot hold the snapshot even after eviction")
        self._eviction_skipped_leased += skipped_leased
        return plan

    def _clear(self, reason: str) -> int:
        cleared = 0
        for entry in list(self._entries.values()):
            self._remove_entry(entry)
            cleared += 1
        for pending in list(self._pending.values()):
            self._cancel_pending(pending)
        for cookie, match in list(self._matches.items()):
            self._matches.pop(cookie, None)
            if match.conn.match == cookie:
                match.conn.match = None
        # a publish id must not answer for an entry this clear removed: a repeated COMMIT after it has to say gone
        self._committed_by_publish.clear()
        self._store_epoch += 1
        self._clears += 1
        return cleared

    def _snapshot_locked(self) -> dict:
        charged = self._charged()
        longest = 0
        for entry in self._entries.values():
            for point in entry.points:
                if point.tokens > longest:
                    longest = point.tokens
        replicas = []
        for owner in sorted(self._owner_state, key=lambda name: self._owner_index.get(name, 0)):
            stats = self._owner_stats.get(owner)
            replicas.append({"replica": self._owner_index.get(owner, 0), "state": self._owner_state[owner],
                             "published": stats.published if stats is not None else 0,
                             "reads": stats.reads if stats is not None else 0,
                             "peer_reads": stats.peer_reads if stats is not None else 0,
                             "fallbacks": stats.fallbacks if stats is not None else 0,
                             "ipc_errors": stats.ipc_errors if stats is not None else 0})
        return {"enabled": True, "capacity_bytes": self._capacity, "max_entries": self._config.max_entries,
                "charged_bytes": charged,
                "entries": len(self._entries), "entry_bytes": self._entry_bytes,
                "pending": len(self._pending), "pending_bytes": self._pending_bytes,
                "leases": len(self._leases), "restore_reserved_bytes": self._restore_reserved,
                "retired_lease_bytes": self._retired_bytes,
                "published": self._published, "published_bytes": self._published_bytes,
                "deduplicated": self._deduplicated, "reads": self._reads, "peer_reads": self._peer_reads,
                "evictions": self._evictions, "eviction_skipped_leased": self._eviction_skipped_leased,
                "superseded": self._superseded, "clears": self._clears, "damaged": self._damaged,
                "capacity_refusals": self._capacity_refusals,
                "budget_violations": int(charged > self._capacity),
                "store_epoch": self._store_epoch, "longest_tokens": longest, "replicas": replicas}

    def _shutdown(self) -> None:
        for parent, _owner, _conn_id in self._incoming:     # opened just before close(): never registered
            try:
                parent.close()
            except OSError:
                pass
        self._incoming.clear()
        for conn in list(self._connections.values()):
            self._drop(conn, "closed")
        for entry in list(self._entries.values()):
            for fd in entry.fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            entry.fds = []
        for pending in list(self._pending.values()):
            for fd in pending.fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            pending.fds = []
        self._entries.clear()
        self._lru.clear()
        self._pending.clear()
        self._pending_by_publish.clear()
        self._retired.clear()
        self._leases.clear()
        self._matches.clear()
        self._entry_bytes = self._pending_bytes = self._retired_bytes = self._restore_reserved = 0
        for owner in self._owner_state:
            self._owner_state[owner] = "closed"
        if self._sel is not None:
            try:
                self._sel.close()
            except OSError:
                pass

    # ---- tests and diagnostics
    def _all_fds(self) -> list:
        """Every fd the store owns right now (the tests check that close() frees them all)."""
        with self._lock:
            fds: list = []
            for entry in self._entries.values():
                fds.extend(entry.fds)
            for pending in self._pending.values():
                fds.extend(pending.fds)
            for conn in self._connections.values():
                try:
                    fds.append(conn.sock.fileno())
                except OSError:
                    pass
            fds.extend(fd for fd in (self._wake_r, self._wake_w) if fd >= 0)
            return fds


def _better_point(point: _Point, entry: _Entry, best: _Point, best_entry: _Entry) -> bool:
    """Deeper wins; on a tie live wins; then the more recently used snapshot."""
    if point.tokens != best.tokens:
        return point.tokens > best.tokens
    if point.live != best.live:
        return point.live
    return entry.used > best_entry.used


_HANDLERS = {
    OP_HELLO: ParentParkingStore._op_hello,
    OP_PARK_ADVICE: ParentParkingStore._op_advice,
    OP_PARK_BEGIN: ParentParkingStore._op_begin,
    OP_PARK_COMMIT: ParentParkingStore._op_commit,
    OP_PARK_ABORT: ParentParkingStore._op_abort,
    OP_MATCH_BEGIN: ParentParkingStore._op_match_begin,
    OP_MATCH_END: ParentParkingStore._op_match_end,
    OP_ACQUIRE: ParentParkingStore._op_acquire,
    OP_RELEASE: ParentParkingStore._op_release,
    OP_CLEAR: ParentParkingStore._op_clear,
    OP_STATS: ParentParkingStore._op_stats,
    OP_PING: ParentParkingStore._op_ping,
}

__all__ = [
    "CLEAR_REASONS", "HASH_SEED", "HEADER", "IDS_SEED", "LAYOUT_SEED", "LIMITS", "MAGIC", "MAX_EXCLUDE_IDS",
    "MAX_FDS", "MAX_OWNER", "MAX_PAYLOAD", "MAX_POINTS", "MAX_STAGES", "MEMFD_NAME", "OP_ACQUIRE", "OP_CLEAR",
    "OP_ERROR", "OP_HELLO", "OP_MATCH_BEGIN", "OP_MATCH_END", "OP_PARK_ABORT", "OP_PARK_ADVICE", "OP_PARK_BEGIN",
    "OP_PARK_COMMIT", "OP_PING", "OP_RELEASE", "OP_STATS", "ParentChannel", "ParentParkingClosed",
    "ParentParkingConfig", "ParentParkingConfigError", "ParentParkingIdentity", "ParentParkingStore",
    "RELEASE_RESULTS", "VERSION", "layout_key", "read_parent_parking", "session_hash64",
    "session_read_peak_bytes",
]
