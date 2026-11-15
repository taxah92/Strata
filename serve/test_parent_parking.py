"""Stage E2 (Python): the parent parking store and protocol v1, over real socketpairs and real memfds.

No transport is mocked: every test opens a real `AF_UNIX`/`SOCK_SEQPACKET` socketpair through
`ParentParkingStore.open_channel`, sends real 36-byte frames with real `SCM_RIGHTS` file descriptors, and the store
creates real memfds with real seals.  No model, no GPU and no engine are needed.

The tests follow plan §16.2 one for one (the names are the plan's), plus the run-config checks of plan §13.1/§13.3 and
one test that `open_channel` is safe from several threads at once (parallel StrataEngine starts).
"""
import errno
import fcntl
import json
import os
import socket
import struct
import threading
import time
import unittest

from serve import parent_parking as pp

IDENTITY = pp.ParentParkingIdentity(model=7, config=9, layout_key=pp.layout_key([(0, 31)]), stages=((0, 31),))
STAGES = [[0, 31]]
HASH_LIVE, HASH_CP = 0x1111, 0x2222
POINTS_A = ({"tokens": 100, "hash": HASH_LIVE, "imgs": [], "live": True, "pinned": False},
            {"tokens": 50, "hash": HASH_CP, "imgs": [], "live": False, "pinned": False})
POINTS_B = ({"tokens": 200, "hash": 0x3333, "imgs": [], "live": True, "pinned": False},
            {"tokens": 150, "hash": 0x4444, "imgs": [], "live": False, "pinned": False})


def point(tokens, hash, imgs=(), live=False, pinned=False):
    """One point of a snapshot, as the engine sends it."""
    return {"tokens": tokens, "hash": hash, "imgs": [{"start": start, "hash": h} for start, h in imgs],
            "live": live, "pinned": pinned}


def new_publish_id():
    return os.urandom(16).hex()


class Client:
    """One replica's end of a channel: real frames over the real socketpair, no transport mocks."""

    def __init__(self, store, owner="r0", hello=True):
        fd, self.channel = store.open_channel(owner)
        self._init(fd, owner, hello)

    @classmethod
    def from_fd(cls, fd, owner, hello=True):
        client = cls.__new__(cls)
        client._init(fd, owner, hello)
        return client

    def _init(self, fd, owner, hello):
        self.owner = owner
        self.fd = fd
        self.sock = socket.socket(fileno=fd)
        self.request_id = 0
        self.epoch = 1
        if hello:
            self.hello()

    # ---- frames
    def send_frame(self, op, payload, *, fds=(), request_id=None, magic=pp.MAGIC, version=pp.VERSION, flags=0,
                   payload_hash=None, payload_bytes=None):
        if request_id is None:
            self.request_id += 1
            request_id = self.request_id
        body = payload if isinstance(payload, bytes) else json.dumps(payload, separators=(",", ":")).encode()
        header = pp.HEADER.pack(magic, version, op, flags, len(body) if payload_bytes is None else payload_bytes,
                                request_id,
                                pp.session_hash64(body, pp.HASH_SEED) if payload_hash is None else payload_hash)
        data = header + body
        if fds:
            socket.send_fds(self.sock, [data], list(fds))
        else:
            self.sock.send(data)
        return request_id

    def recv_frame(self):
        try:
            data, fds, flags, _addr = socket.recv_fds(self.sock, pp.HEADER.size + pp.MAX_PAYLOAD + 1, pp.MAX_FDS)
        except OSError:                                 # the store closed the channel with data still queued: a reset
            return None, None, []
        if not data:
            return None, None, []
        _magic, _version, op, _flags, size, _request_id, _hash = pp.HEADER.unpack_from(data)
        return op, json.loads(data[pp.HEADER.size:pp.HEADER.size + size]), fds

    def call(self, op, payload, *, fds=()):
        self.send_frame(op, payload, fds=fds)
        return self.recv_frame()

    def call_ok(self, op, payload, *, fds=()):
        got, body, out = self.call(op, payload, fds=fds)
        if got != op:
            raise AssertionError(f"expected op {op}, got {got}: {body}")
        return body, out

    # ---- operations
    def hello(self, **changes):
        payload = {"owner": self.owner, "protocol": 1, "model": IDENTITY.model, "config": IDENTITY.config,
                   "layout_key": IDENTITY.layout_key, "stages": [list(stage) for stage in IDENTITY.stages],
                   "pid": os.getpid()}
        payload.update(changes)
        body, _ = self.call_ok(pp.OP_HELLO, payload)
        self.epoch = body["store_epoch"]
        return body

    def ping(self):
        body, _ = self.call_ok(pp.OP_PING, {})
        self.epoch = body["epoch"]
        return body

    def advice(self, points, cvec=False):
        body, _ = self.call_ok(pp.OP_PARK_ADVICE, {"cvec": cvec, "points": list(points)})
        return body

    def begin_payload(self, points, sizes, *, publish_id=None, cvec=False, epoch=None, owner=None):
        return {"publish_id": publish_id or new_publish_id(), "owner": owner or self.owner,
                "model": IDENTITY.model, "config": IDENTITY.config, "layout_key": IDENTITY.layout_key,
                "stages": [list(stage) for stage in IDENTITY.stages], "cvec": cvec, "points": list(points),
                "stage_bytes": list(sizes), "epoch": self.epoch if epoch is None else epoch}

    def begin(self, points, sizes, *, publish_id=None, cvec=False, epoch=None):
        publish_id = publish_id or new_publish_id()
        body, fds = self.call_ok(pp.OP_PARK_BEGIN,
                                 self.begin_payload(points, sizes, publish_id=publish_id, cvec=cvec, epoch=epoch))
        return publish_id, body, fds

    def commit(self, publish_id, pending_id, sizes, *, epoch=None):
        body, _ = self.call_ok(pp.OP_PARK_COMMIT,
                               {"publish_id": publish_id, "pending_id": pending_id,
                                "written_bytes": list(sizes), "epoch": self.epoch if epoch is None else epoch})
        return body

    def publish(self, points, sizes, *, publish_id=None, cvec=False, data=None, write=True):
        publish_id, begin, fds = self.begin(points, sizes, publish_id=publish_id, cvec=cvec)
        if begin["dedup"]:
            for fd in fds:
                os.close(fd)
            return publish_id, begin
        if write:
            for fd, size in zip(fds, sizes):
                if data is not None:
                    os.pwrite(fd, data[:size], 0)
                elif size:
                    os.pwrite(fd, b"\x5a" * size, 0)
        for fd in fds:
            os.close(fd)
        return publish_id, self.commit(publish_id, begin["pending_id"], sizes)

    def abort(self, publish_id, pending_id):
        body, _ = self.call_ok(pp.OP_PARK_ABORT, {"publish_id": publish_id, "pending_id": pending_id})
        return body

    def match_begin(self, prompt_tokens, *, images=(), cvec=False, exclude_ids=()):
        body, _ = self.call_ok(pp.OP_MATCH_BEGIN,
                               {"prompt_tokens": prompt_tokens, "cvec": cvec,
                                "images": [{"start": start, "hash": h} for start, h in images],
                                "exclude_ids": list(exclude_ids)})
        return body

    def match_end(self, cookie, hashes):
        body, _ = self.call_ok(pp.OP_MATCH_END,
                               {"cookie": cookie, "hashes": [{"tokens": tokens, "hash": h} for tokens, h in hashes]})
        return body

    def match_at(self, prompt_tokens, wanted, *, images=(), cvec=False, exclude_ids=()):
        """MATCH_BEGIN then MATCH_END with `wanted`'s hashes and 0 for the lengths the test does not care about."""
        begin = self.match_begin(prompt_tokens, images=images, cvec=cvec, exclude_ids=exclude_ids)
        hashes = [(tokens, wanted.get(tokens, 0)) for tokens in begin["lengths"]]
        return self.match_end(begin["cookie"], hashes)

    def acquire(self, entry_id, generation):
        return self.call_ok(pp.OP_ACQUIRE, {"id": entry_id, "generation": generation})

    def release(self, lease_id, result="ok"):
        body, _ = self.call_ok(pp.OP_RELEASE, {"lease_id": lease_id, "result": result})
        return body

    def clear(self, reason="test"):
        body, _ = self.call_ok(pp.OP_CLEAR, {"reason": reason})
        self.epoch = body["epoch"]
        return body

    def stats(self):
        body, _ = self.call_ok(pp.OP_STATS, {})
        return body

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class StoreCase(unittest.TestCase):
    """A store per test, its clients closed first and the store last."""

    def setUp(self):
        self.store = None
        self.clients = []

    def tearDown(self):
        for client in self.clients:
            client.close()
        if self.store is not None:
            self.store.close()

    def make_store(self, *, ram_gib=1, max_entries=10, max_snapshot_mib=64, ipc_timeout_ms=2000,
                   capacity_bytes=None, identity=IDENTITY):
        config = pp.ParentParkingConfig(ram_gib=ram_gib, max_entries=max_entries,
                                        max_snapshot_mib=max_snapshot_mib, ipc_timeout_ms=ipc_timeout_ms)
        self.store = pp.ParentParkingStore(config, identity, capacity_bytes=capacity_bytes)
        return self.store

    def client(self, owner="r0", hello=True):
        client = Client(self.store, owner, hello=hello)
        self.clients.append(client)
        return client

    def wait_state(self, owner, state, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.store.replica_state(owner) == state:
                return
            time.sleep(0.005)
        self.fail(f"{owner} did not become {state}: {self.store.replica_state(owner)}")

    def assert_error(self, client, op, payload, code):
        got, body, _fds = client.call(op, payload)
        self.assertEqual(got, pp.OP_ERROR, body)
        self.assertEqual(body["code"], code, body)
        return body


class ParentParkingProtocol(StoreCase):
    """Plan §16.2: the store and protocol v1 over real socketpairs and real memfds."""

    def test_two_replicas_publish_and_acquire_each_other(self):
        self.make_store(capacity_bytes=64 << 20)
        r0, r1 = self.client("r0"), self.client("r1")
        data = b"a snapshot r0 parked" * 20
        _publish_id, entry = r0.publish(POINTS_A, [len(data)], data=data)
        self.assertEqual(entry["entries"], 1)
        candidate = r1.match_at(101, {100: HASH_LIVE, 50: HASH_CP})["candidate"]
        self.assertEqual((candidate["id"], candidate["tokens"], candidate["live"], candidate["owner"]),
                         (entry["id"], 100, True, "r0"))
        lease, fds = r1.acquire(entry["id"], entry["generation"])
        self.assertEqual(os.pread(fds[0], len(data), 0), data)
        self.assertEqual((lease["tokens"], lease["live"], lease["borrow"], lease["owner"]),
                         (100, True, False, "r0"))
        self.assertEqual(r1.release(lease["lease_id"])["gone"], False)
        os.close(fds[0])
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["reads"], stats["peer_reads"]), (1, 1, 1))
        self.assertEqual(stats["replicas"][1]["peer_reads"], 1)

    def test_pending_is_invisible_and_abort_frees_charge(self):
        self.make_store()
        r0 = self.client("r0")
        publish_id, begin, fds = r0.begin(POINTS_A, [1000])
        stats = self.store.snapshot()
        self.assertEqual((stats["pending"], stats["pending_bytes"], stats["entries"]), (1, 1000, 0))
        self.assertEqual(stats["charged_bytes"], 1000)
        self.assertEqual(r0.match_begin(101)["lengths"], [])            # pending is not in MATCH
        self.assertEqual(r0.advice(POINTS_A)["holding_id"], 0)          # ... nor a dedup holding
        for fd in fds:
            os.close(fd)
        self.assertEqual(r0.abort(publish_id, begin["pending_id"])["gone"], False)
        stats = self.store.snapshot()
        self.assertEqual((stats["pending"], stats["pending_bytes"], stats["charged_bytes"]), (0, 0, 0))

    def test_commit_seals_snapshot(self):
        self.make_store()
        r0 = self.client("r0")
        publish_id, begin, fds = r0.begin(POINTS_A, [64])
        os.pwrite(fds[0], b"x" * 64, 0)
        r0.commit(publish_id, begin["pending_id"], [64])
        with self.assertRaises(OSError) as caught:
            os.write(fds[0], b"y")
        self.assertEqual(caught.exception.errno, errno.EPERM)
        with self.assertRaises(OSError):
            os.ftruncate(fds[0], 128)
        with self.assertRaises(OSError):
            fcntl.fcntl(fds[0], fcntl.F_ADD_SEALS, fcntl.F_SEAL_SEAL)
        os.close(fds[0])

    def test_reader_fd_survives_eviction(self):
        self.make_store(capacity_bytes=64 << 20)
        r0, r1 = self.client("r0"), self.client("r1")
        data = b"the reader keeps its pages" * 40
        _publish_id, entry = r0.publish(POINTS_A, [len(data)], data=data)
        lease, fds = r1.acquire(entry["id"], entry["generation"])
        r0.clear("test")                                                # the parent drops and closes its own fds
        self.assertEqual(self.store.snapshot()["entries"], 0)
        self.assertEqual(os.pread(fds[0], len(data), 0), data)
        r1.release(lease["lease_id"])
        os.close(fds[0])

    def test_restart_preserves_committed_but_aborts_pending_and_releases_leases(self):
        self.make_store(capacity_bytes=64 << 20)
        r0, r1 = self.client("r0"), self.client("r1")
        _publish_id, entry = r0.publish(POINTS_A, [1000])
        lease, fds = r0.acquire(entry["id"], entry["generation"])
        os.close(fds[0])
        _pending_id, _begin, pending_fds = r0.begin(POINTS_B, [1000])
        for fd in pending_fds:
            os.close(fd)
        r0.close()                                                      # the engine died: EOF on its channel
        self.wait_state("r0", "degraded")
        stats = self.store.snapshot()
        self.assertEqual((stats["pending"], stats["leases"], stats["entries"]), (0, 0, 1))
        candidate = r1.match_at(101, {100: HASH_LIVE, 50: HASH_CP})["candidate"]
        self.assertEqual(candidate["id"], entry["id"])                  # committed survives the restart
        r0b = self.client("r0")                                         # a restart gets a new channel
        self.assertEqual(r0b.stats()["replicas"][0]["state"], "connected")
        lease, fds = r0b.acquire(entry["id"], entry["generation"])
        os.close(fds[0])
        r0b.release(lease["lease_id"])

    def test_server_close_frees_all_fds(self):
        self.make_store(capacity_bytes=64 << 20)
        r0 = self.client("r0")
        _publish_id, entry = r0.publish(POINTS_A, [1000])
        lease, fds = r0.acquire(entry["id"], entry["generation"])
        os.close(fds[0])
        _pending_id, _begin, pending_fds = r0.begin(POINTS_B, [1000])
        for fd in pending_fds:
            os.close(fd)
        owned = self.store._all_fds()
        self.assertTrue(owned)
        self.store.close()
        for fd in owned:
            with self.assertRaises(OSError):
                os.fstat(fd)
        self.assertEqual(self.store.replica_state("r0"), "closed")
        self.assertEqual(self.store.snapshot()["entries"], 0)

    def test_duplicate_publish_id_begin_commit_abort_are_idempotent(self):
        self.make_store()
        r0 = self.client("r0")
        publish_id = "b" * 32
        _pid, begin1, fds1 = r0.begin(POINTS_A, [100], publish_id=publish_id)
        _pid, begin2, fds2 = r0.begin(POINTS_A, [100], publish_id=publish_id)
        self.assertEqual(begin1["pending_id"], begin2["pending_id"])
        self.assertEqual(os.fstat(fds1[0]).st_ino, os.fstat(fds2[0]).st_ino)   # the same memfd, sent again
        os.pwrite(fds1[0], b"x" * 100, 0)
        for fd in fds1 + fds2:
            os.close(fd)
        entry = r0.commit(publish_id, begin1["pending_id"], [100])
        again = r0.commit(publish_id, begin1["pending_id"], [100])
        self.assertEqual((again["id"], again["generation"], again["entry_bytes"]),
                         (entry["id"], entry["generation"], entry["entry_bytes"]))
        _pid, begin3, fds3 = r0.begin(POINTS_A, [100], publish_id=publish_id)
        self.assertTrue(begin3["dedup"])                                # BEGIN after COMMIT is a dedup answer
        self.assertEqual(begin3["holding_id"], entry["id"])
        self.assertEqual(fds3, [])
        other = "c" * 32
        _pid, begin4, fds4 = r0.begin(POINTS_B, [100], publish_id=other)
        for fd in fds4:
            os.close(fd)
        self.assertEqual(r0.abort(other, begin4["pending_id"])["gone"], False)
        self.assertEqual(r0.abort(other, begin4["pending_id"])["gone"], True)
        self.assertEqual(r0.abort(publish_id, begin1["pending_id"])["gone"], True)

    def test_conflicting_publish_id_is_rejected(self):
        self.make_store()
        r0 = self.client("r0")
        publish_id = "d" * 32
        _pid, begin, fds = r0.begin(POINTS_A, [100], publish_id=publish_id)
        for fd in fds:
            os.close(fd)
        self.assert_error(r0, pp.OP_PARK_BEGIN, r0.begin_payload(POINTS_B, [100], publish_id=publish_id),
                          "bad_request")
        self.assertEqual(r0.ping()["epoch"], 1)                         # the channel stays open
        self.assertEqual(r0.abort(publish_id, begin["pending_id"])["gone"], False)

    def test_match_lengths_one_pass_and_tie_rules(self):
        self.make_store()
        r0 = self.client("r0")
        _pid, e1 = r0.publish([point(100, 0x100, live=True), point(60, 0x60), point(50, HASH_CP)], [100])
        _pid, e2 = r0.publish([point(100, 0x100, live=True), point(70, 0x70), point(50, HASH_CP)], [100])
        begin = r0.match_begin(100)                                     # the prompt's last token is excluded
        self.assertEqual(begin["lengths"], [50, 60, 70])
        self.assertEqual(begin["lengths"], sorted(set(begin["lengths"])))
        begin = r0.match_begin(101)
        self.assertEqual(begin["lengths"], [50, 60, 70, 100])
        candidate = r0.match_end(begin["cookie"], [(50, HASH_CP), (60, 0x60), (70, 0x70), (100, 0x100)])["candidate"]
        self.assertEqual((candidate["id"], candidate["tokens"], candidate["live"]), (e2["id"], 100, True))
        candidate = r0.match_at(101, {50: HASH_CP})["candidate"]        # tie at 50: the later used wins
        self.assertEqual((candidate["id"], candidate["tokens"], candidate["live"]), (e2["id"], 50, False))
        _pid, e3 = r0.publish([point(50, HASH_CP, live=True)], [100])
        candidate = r0.match_at(101, {50: HASH_CP})["candidate"]        # tie at 50: live wins
        self.assertEqual((candidate["id"], candidate["tokens"], candidate["live"]), (e3["id"], 50, True))

    def test_match_images_cvec_and_exclude(self):
        self.make_store()
        r0 = self.client("r0")
        _pid, e1 = r0.publish([point(10, 0x10, imgs=[(2, 0x22)], live=True)], [100])
        _pid, _e2 = r0.publish([point(10, 0x10, imgs=[(2, 0x22)], live=True)], [100], cvec=True)
        self.assertEqual(r0.match_begin(20, cvec=False)["lengths"], [])                 # the picture is missing
        self.assertEqual(r0.match_begin(20, cvec=False, images=[(2, 0x22)])["lengths"], [10])
        self.assertEqual(r0.match_begin(20, cvec=True, images=[(2, 0x22)])["lengths"], [10])
        self.assertEqual(r0.match_begin(20, cvec=False, images=[(2, 0x22), (15, 0x99)])["lengths"], [10])
        self.assertEqual(r0.match_begin(20, cvec=False, images=[(2, 0x22), (5, 0x99)])["lengths"], [])
        self.assertEqual(r0.match_begin(20, cvec=False, images=[(2, 0x22)], exclude_ids=[e1["id"]])["lengths"], [])

    def test_acquire_rechecks_generation_and_one_retry(self):
        self.make_store(capacity_bytes=64 << 20)
        r0 = self.client("r0")
        _pid, entry = r0.publish(POINTS_A, [100])
        candidate = r0.match_at(101, {100: HASH_LIVE, 50: HASH_CP})["candidate"]
        self.assert_error(r0, pp.OP_ACQUIRE, {"id": candidate["id"], "generation": candidate["generation"] + 1},
                          "stale_generation")
        begin = r0.match_begin(101, exclude_ids=[candidate["id"]])      # the one retry the contract allows
        self.assertEqual(begin["lengths"], [])
        self.assertIsNone(r0.match_end(begin["cookie"], [])["candidate"])
        r0.clear("test")
        self.assert_error(r0, pp.OP_ACQUIRE, {"id": candidate["id"], "generation": candidate["generation"]}, "gone")

    def test_global_dedup_and_superseded(self):
        self.make_store()
        r0, r1 = self.client("r0"), self.client("r1")
        _pid, e1 = r0.publish(POINTS_A, [100])
        _pid, e2 = r1.publish(POINTS_A, [100])                          # the same snapshot from another replica
        self.assertTrue(e2["dedup"])
        self.assertEqual(e2["holding_id"], e1["id"])
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["deduplicated"]), (1, 1))
        advice = r1.advice(POINTS_A)
        self.assertEqual((advice["holding_id"], advice["parked_prefix_tokens"]), (e1["id"], 50))
        _pid, _e3 = r1.publish([point(120, 0x120, live=True), point(100, 0x100), point(50, HASH_CP)], [100])
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["superseded"]), (1, 1))
        self.assertEqual(stats["longest_tokens"], 120)

    def test_borrow_rule_and_pinned_donor(self):
        self.make_store()
        r0 = self.client("r0")
        _pid, _e1 = r0.publish([point(100, 0x100, live=True), point(60, 0x60), point(30, 0x30)], [100])
        candidate = r0.match_at(101, {30: 0x30})["candidate"]
        self.assertEqual((candidate["tokens"], candidate["live"], candidate["borrow"]), (30, False, True))
        candidate = r0.match_at(101, {60: 0x60})["candidate"]
        self.assertEqual((candidate["tokens"], candidate["borrow"]), (60, False))
        candidate = r0.match_at(101, {100: 0x100})["candidate"]
        self.assertEqual((candidate["tokens"], candidate["live"], candidate["borrow"]), (100, True, False))
        _pid, _e2 = r0.publish([point(100, 0x100, live=True), point(60, 0x60),
                                point(30, 0x30, pinned=True)], [100])
        candidate = r0.match_at(101, {30: 0x30})["candidate"]           # a pinned prefix is taken, not borrowed
        self.assertEqual((candidate["tokens"], candidate["borrow"]), (30, False))

    def test_lru_and_atomic_eviction_plan(self):
        self.make_store(capacity_bytes=1000)
        r0 = self.client("r0")
        _pid, _a = r0.publish([point(10, 0x10, live=True)], [400])
        _pid, _b = r0.publish([point(20, 0x20, live=True)], [400])
        self.assertEqual(self.store.snapshot()["charged_bytes"], 800)
        _pid, _c = r0.publish([point(30, 0x30, live=True)], [400])
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["evictions"]), (2, 1))
        self.assertEqual(r0.match_begin(11)["lengths"], [])             # the oldest (10 tokens) went
        self.assertEqual(r0.match_begin(31)["lengths"], [20, 30])
        self.assert_error(r0, pp.OP_PARK_BEGIN, r0.begin_payload([point(40, 0x40, live=True)], [2000]), "capacity")
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["capacity_refusals"]), (2, 1))
        self.assertEqual(r0.match_begin(31)["lengths"], [20, 30])       # nothing was deleted by the failed plan
        self.make_store(capacity_bytes=1000)
        r0 = self.client("r0")
        r0.publish([point(10, 0x10, live=True, pinned=True)], [400])
        r0.publish([point(20, 0x20, live=True, pinned=True)], [400])
        self.assert_error(r0, pp.OP_PARK_BEGIN, r0.begin_payload([point(30, 0x30, live=True)], [400]),
                          "pinned_working_set")

    def test_hard_capacity_includes_pending_restore_and_retired_leases(self):
        self.make_store(capacity_bytes=64 << 20)
        r0 = self.client("r0")
        r0.publish([point(10, 0x10, live=True)], [20 << 20], write=False)
        _pid, begin, fds = r0.begin([point(20, 0x20, live=True)], [50 << 20])
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["pending"], stats["evictions"]), (0, 1, 1))
        for fd in fds:
            os.close(fd)
        r0.abort(_pid, begin["pending_id"])
        _pid, entry = r0.publish([point(30, 0x30, live=True)], [20 << 20], write=False)
        lease, lease_fds = r0.acquire(entry["id"], entry["generation"])
        os.close(lease_fds[0])
        self.assertEqual(self.store.snapshot()["restore_reserved_bytes"], pp.session_read_peak_bytes(20 << 20))
        self.assert_error(r0, pp.OP_PARK_BEGIN, r0.begin_payload([point(40, 0x40, live=True)], [20 << 20]),
                          "pinned_working_set")                     # the only snapshot is leased
        r0.clear("test")
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["retired_lease_bytes"]), (0, 20 << 20))
        self.assert_error(r0, pp.OP_PARK_BEGIN, r0.begin_payload([point(50, 0x50, live=True)], [20 << 20]),
                          "capacity")                               # retired bytes are still charged
        r0.release(lease["lease_id"])
        stats = self.store.snapshot()
        self.assertEqual((stats["retired_lease_bytes"], stats["restore_reserved_bytes"]), (0, 0))
        r0.publish([point(50, 0x50, live=True)], [20 << 20], write=False)
        self.assertEqual(self.store.snapshot()["entries"], 1)

    def test_restore_reservation_lives_until_release(self):
        self.make_store(capacity_bytes=64 << 20)
        r0 = self.client("r0")
        _pid, entry = r0.publish(POINTS_A, [1000])
        lease, fds = r0.acquire(entry["id"], entry["generation"])
        os.close(fds[0])
        charge = pp.session_read_peak_bytes(1000)
        stats = self.store.snapshot()
        self.assertEqual((stats["restore_reserved_bytes"], stats["charged_bytes"]), (charge, 1000 + charge))
        r0.release(lease["lease_id"])
        self.assertEqual(self.store.snapshot()["restore_reserved_bytes"], 0)
        lease, fds = r0.acquire(entry["id"], entry["generation"])
        os.close(fds[0])
        r0.close()                                                      # EOF releases the lease too
        self.wait_state("r0", "degraded")
        self.assertEqual(self.store.snapshot()["restore_reserved_bytes"], 0)

    def test_clear_epoch_rejects_stale_commit_and_clears_all_owners(self):
        self.make_store()
        r0, r1 = self.client("r0"), self.client("r1")
        r0.publish([point(10, 0x10, live=True)], [100])
        r1.publish([point(20, 0x20, live=True)], [100])
        publish_id, begin, fds = r0.begin([point(30, 0x30, live=True)], [100])
        for fd in fds:
            os.close(fd)
        old_epoch = r0.epoch
        cleared = r1.clear("nonfinite")
        self.assertEqual((cleared["cleared"], cleared["epoch"]), (2, old_epoch + 1))
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["pending"], stats["clears"]), (0, 0, 1))
        self.assert_error(r0, pp.OP_PARK_COMMIT,
                          {"publish_id": publish_id, "pending_id": begin["pending_id"], "written_bytes": [100],
                           "epoch": old_epoch}, "stale_epoch")

    def test_invalid_release_quarantines_globally_memory_release_does_not(self):
        self.make_store(capacity_bytes=64 << 20)
        r0, r1 = self.client("r0"), self.client("r1")
        _pid, entry = r0.publish([point(10, 0x10, live=True)], [100])
        lease, fds = r1.acquire(entry["id"], entry["generation"])
        os.close(fds[0])
        r1.release(lease["lease_id"], "invalid")
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["damaged"]), (0, 1))
        self.assertEqual(r1.match_begin(11)["lengths"], [])
        _pid, entry = r0.publish([point(20, 0x20, live=True)], [100])
        lease, fds = r1.acquire(entry["id"], entry["generation"])
        os.close(fds[0])
        r1.release(lease["lease_id"], "memory")
        stats = self.store.snapshot()
        self.assertEqual((stats["entries"], stats["damaged"]), (1, 1))
        self.assertEqual(r1.match_begin(21)["lengths"], [20])

    def test_identity_layout_stage_count_size_and_payload_limits(self):
        self.make_store(max_snapshot_mib=1)
        wrong = self.client("r0", hello=False)
        self.assert_error(wrong, pp.OP_HELLO,
                          {"owner": "r0", "protocol": 1, "model": 8, "config": IDENTITY.config,
                           "layout_key": IDENTITY.layout_key, "stages": STAGES, "pid": os.getpid()}, "identity")
        self.assertIsNone(wrong.recv_frame()[0])                        # the channel closed
        r1 = self.client("r1")
        payload = r1.begin_payload([point(10, 0x10, live=True)], [100])
        payload["layout_key"] = 12345
        self.assert_error(r1, pp.OP_PARK_BEGIN, payload, "identity")
        r2 = self.client("r2")
        self.assert_error(r2, pp.OP_PARK_BEGIN, r2.begin_payload([point(10, 0x10, live=True)], [100, 200]),
                          "bad_request")
        r3 = self.client("r3")
        self.assert_error(r3, pp.OP_PARK_BEGIN, r3.begin_payload([point(10, 0x10, live=True)], [(1 << 20) + 1]),
                          "bad_request")
        r4 = self.client("r4")
        self.assert_error(r4, pp.OP_PARK_BEGIN,
                          r4.begin_payload([point(i + 1, i) for i in range(pp.MAX_POINTS + 1)], [100]),
                          "bad_request")
        r5 = self.client("r5")
        r5.send_frame(pp.OP_PING, b"x" * (pp.MAX_PAYLOAD + 1))
        got, body, _fds = r5.recv_frame()
        self.assertEqual((got, body["code"]), (pp.OP_ERROR, "bad_request"))
        self.assertIsNone(r5.recv_frame()[0])

    def test_bad_magic_hash_request_id_ancillary_and_second_outstanding(self):
        self.make_store()
        bad_magic = self.client("r0", hello=False)
        bad_magic.send_frame(pp.OP_PING, {}, magic=b"BADMAGIC")
        got, body, _fds = bad_magic.recv_frame()
        self.assertEqual((got, body["code"]), (pp.OP_ERROR, "bad_request"))
        self.assertIsNone(bad_magic.recv_frame()[0])
        bad_hash = self.client("r1", hello=False)
        bad_hash.send_frame(pp.OP_PING, {}, payload_hash=1)
        got, body, _fds = bad_hash.recv_frame()
        self.assertEqual((got, body["code"]), (pp.OP_ERROR, "bad_request"))
        self.assertIsNone(bad_hash.recv_frame()[0])
        zero_id = self.client("r2", hello=False)
        zero_id.send_frame(pp.OP_PING, {}, request_id=0)
        got, body, _fds = zero_id.recv_frame()
        self.assertEqual((got, body["code"]), (pp.OP_ERROR, "bad_request"))
        self.assertIsNone(zero_id.recv_frame()[0])
        ancillary = self.client("r3", hello=False)
        fd = os.memfd_create("strata-test-ancillary", os.MFD_CLOEXEC)
        try:
            ancillary.send_frame(pp.OP_PING, {}, fds=[fd])
            got, body, _fds = ancillary.recv_frame()
            self.assertEqual((got, body["code"]), (pp.OP_ERROR, "bad_request"))
            self.assertIsNone(ancillary.recv_frame()[0])
        finally:
            os.close(fd)
        second = self.client("r4", hello=False)
        second.send_frame(pp.OP_PING, {}, request_id=1)
        second.send_frame(pp.OP_PING, {}, request_id=2)
        got, body, _fds = second.recv_frame()
        self.assertEqual((got, body["code"]), (pp.OP_ERROR, "bad_request"))
        self.assertIsNone(second.recv_frame()[0])

    def test_socket_timeout_closes_channel_and_cleans_connection_state(self):
        self.make_store(capacity_bytes=64 << 20, ipc_timeout_ms=20)
        r0 = self.client("r0")
        _pid, entry = r0.publish(POINTS_A, [100])
        lease, fds = r0.acquire(entry["id"], entry["generation"])
        os.close(fds[0])
        _pending_id, _begin, pending_fds = r0.begin(POINTS_B, [100])
        for fd in pending_fds:
            os.close(fd)
        r0.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        r0.close()                                                      # the client timed out and reset its socket
        self.wait_state("r0", "degraded")
        stats = self.store.snapshot()
        self.assertEqual((stats["pending"], stats["leases"], stats["entries"]), (0, 0, 1))
        r0b = self.client("r0")
        lease, fds = r0b.acquire(entry["id"], entry["generation"])
        os.close(fds[0])
        r0b.release(lease["lease_id"])
        begin = r0b.match_begin(101)                                    # a cookie lives 4 * ipc_timeout_ms
        time.sleep(0.12)
        self.assert_error(r0b, pp.OP_MATCH_END, {"cookie": begin["cookie"], "hashes": []}, "gone")

    def test_default_off_creates_no_thread_socket_or_memfd(self):
        threads_before = {thread.name for thread in threading.enumerate()}
        fds_before = len(os.listdir("/proc/self/fd"))
        self.assertIsNone(pp.read_parent_parking({}))
        self.assertIsNone(pp.read_parent_parking({"replica_parent_parking": {"enabled": False}}))
        self.assertIsNone(pp.read_parent_parking({"replica_parent_parking": {"enabled": False, "ram_gib": 40}}))
        self.assertEqual({thread.name for thread in threading.enumerate()} - threads_before, set())
        self.assertEqual(len(os.listdir("/proc/self/fd")), fds_before)
        self.assertNotIn("strata-parent-parking", {thread.name for thread in threading.enumerate()})


class ParentParkingConfigTests(unittest.TestCase):
    """Plan §13.1/§13.3: the run config's "replica_parent_parking" block is checked at start, never guessed."""

    def test_off_by_default_and_when_disabled(self):
        self.assertIsNone(pp.read_parent_parking({}))
        self.assertIsNone(pp.read_parent_parking({"replica_parent_parking": {"enabled": False}}))
        self.assertIsNone(pp.read_parent_parking({"replica_parent_parking": {"enabled": False, "ram_gib": 40}}))

    def test_the_block_must_be_an_object(self):
        with self.assertRaises(pp.ParentParkingConfigError):
            pp.read_parent_parking({"replica_parent_parking": [1, 2]})

    def test_enabled_must_be_a_bool(self):
        with self.assertRaises(pp.ParentParkingConfigError):
            pp.read_parent_parking({"replica_parent_parking": {"enabled": "yes", "ram_gib": 40}})

    def test_unknown_keys_are_refused(self):
        with self.assertRaises(pp.ParentParkingConfigError) as caught:
            pp.read_parent_parking({"replica_parent_parking": {"enabled": True, "ram_gib": 40, "unknown": 1}})
        self.assertIn("unknown key", str(caught.exception))

    def test_ram_gib_is_required_and_positive(self):
        with self.assertRaises(pp.ParentParkingConfigError) as caught:
            pp.read_parent_parking({"replica_parent_parking": {"enabled": True}})
        self.assertIn("ram_gib", str(caught.exception))
        for value in (0, -1, True, "40", 1.5):
            with self.subTest(value=value), self.assertRaises(pp.ParentParkingConfigError):
                pp.read_parent_parking({"replica_parent_parking": {"enabled": True, "ram_gib": value}})

    def test_the_fields_are_checked(self):
        for block, expected in (
                ({"max_entries": 0}, "max_entries"),
                ({"max_entries": True}, "max_entries"),
                ({"min_tokens": -1}, "min_tokens"),
                ({"min_gain": "4096"}, "min_gain"),
                ({"max_snapshot_mib": 0}, "max_snapshot_mib"),
                ({"ipc_timeout_ms": 0}, "ipc_timeout_ms"),
                ({"ipc_timeout_ms": 1.5}, "ipc_timeout_ms")):
            with self.subTest(block=block), self.assertRaises(pp.ParentParkingConfigError) as caught:
                pp.read_parent_parking({"replica_parent_parking": {"enabled": True, "ram_gib": 40, **block}})
            self.assertIn(expected, str(caught.exception))

    def test_max_snapshot_cannot_exceed_ram_gib(self):
        with self.assertRaises(pp.ParentParkingConfigError) as caught:
            pp.read_parent_parking({"replica_parent_parking": {"enabled": True, "ram_gib": 1,
                                                               "max_snapshot_mib": 1025}})
        self.assertIn("max_snapshot_mib", str(caught.exception))
        self.assertIsNotNone(pp.read_parent_parking({"replica_parent_parking": {"enabled": True, "ram_gib": 1,
                                                                                "max_snapshot_mib": 1024}}))

    def test_the_defaults(self):
        config = pp.read_parent_parking({"replicas": 3,
                                         "replica_parent_parking": {"enabled": True, "ram_gib": 40}})
        self.assertEqual((config.max_entries, config.min_tokens, config.min_gain, config.max_snapshot_mib,
                          config.ipc_timeout_ms), (30, 0, 4096, 8192, 2000))
        self.assertEqual(config.capacity_bytes, 40 << 30)
        self.assertEqual(config.max_snapshot_bytes, 8192 << 20)
        self.assertTrue(config.enabled)

    def test_the_replicas_argument_wins(self):
        config = pp.read_parent_parking({"replicas": 2, "replica_parent_parking": {"enabled": True, "ram_gib": 40}},
                                        replicas=4)
        self.assertEqual(config.max_entries, 40)


class ParentParkingThreads(unittest.TestCase):
    """open_channel is called from parallel StrataEngine starts: every channel must be its own."""

    def test_open_channel_from_several_threads(self):
        store = pp.ParentParkingStore(pp.ParentParkingConfig(ram_gib=1, max_entries=10, max_snapshot_mib=64),
                                      IDENTITY)
        clients = []
        try:
            results, errors = [], []

            def start(index):
                try:
                    results.append((index, store.open_channel(f"r{index}")))
                except Exception as exc:                # noqa: BLE001 - reported to the test thread
                    errors.append(exc)

            threads = [threading.Thread(target=start, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 8)
            self.assertEqual(len({fd for _index, (fd, _channel) in results}), 8)
            for index, (fd, channel) in results:
                client = Client.from_fd(fd, f"r{index}")
                clients.append(client)
                rows = {row["replica"]: row for row in client.stats()["replicas"]}
                self.assertEqual(rows[index]["state"], "connected")
                self.assertEqual(channel.state, "connected")
        finally:
            for client in clients:
                client.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
