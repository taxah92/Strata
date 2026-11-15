"""Stage 5 (Python): the run config's "replica_parent_parking" block, the per-spawn channel, the store's lifecycle and
the parent parking metrics.  No native model or GPU is required: the engines are fakes and the store is the real one
(serve/parent_parking.py), so the socketpairs, the broker thread and the fd passing are exercised for real.

The config and the flags are plan §13.1/§13.3: a block that cannot be used is a start error, never a silent fallback to
the local heap cache.  The lifecycle is plan §12.2/§12.3: ONE store for the whole server, a new channel for every
spawn, the store cleared on unload and closed after the engines.
"""
import contextlib
import io
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve import parent_parking as pp
from serve import replicas as rp
from serve.frontend import ChatTemplate
from serve.replicas import ReplicaEngine
from serve.server import (ByteTokenizer, EngineDied, MockEngine, ParentParkingConfigError, Service, StrataEngine,
                          parent_parking_args, parent_parking_note, parent_parking_plan, start_replicas)
from serve.test_parent_parking import IDENTITY, POINTS_A, POINTS_B, Client

TEMPLATE = ChatTemplate(Path(__file__).parent / "chat_template.jinja")
GROUPS2, GROUPS3 = [[0, 1], [2, 3]], [[0], [1], [2]]
PULL_ON = rp.PullSettings(True, 60.0, None, None)
PULL_OFF = rp.PullSettings(False, 60.0, None, None)


def block(**changes):
    """The plan's start block (§13.1), with any field changed for one test."""
    b = {"enabled": True, "ram_gib": 40, "max_entries": 20, "min_tokens": 8192, "min_gain": 4096,
         "max_snapshot_mib": 8192, "ipc_timeout_ms": 2000}
    b.update(changes)
    return b


def config(b=None, **kwargs):
    cfg = {"args": [], **kwargs}
    if b is not None:
        cfg["replica_parent_parking"] = b
    return cfg


class ParentParkingConfigTests(unittest.TestCase):
    """Plan §13.1/§13.3: the block is read and checked at start, and every refusal names what to fix."""

    def plan(self, b=None, groups=GROUPS2, **kwargs):
        return parent_parking_plan(config(b, **kwargs), groups, pull=PULL_ON)

    def problem(self, b=None, groups=GROUPS2, posix=True, **kwargs):
        with self.assertRaises(ParentParkingConfigError) as caught:
            parent_parking_plan(config(b, **kwargs), groups, posix=posix, pull=PULL_ON)
        return str(caught.exception)

    def test_off_by_default_and_when_disabled(self):
        self.assertIsNone(parent_parking_plan({"args": []}, GROUPS2))
        self.assertIsNone(self.plan({"enabled": False}))
        self.assertIsNone(self.plan({"enabled": False, "ram_gib": 40}))     # a disabled block is not read further

    def test_the_block_must_be_an_object(self):
        self.assertIn("object", self.problem([1, 2]))

    def test_enabled_must_be_a_bool(self):
        self.assertIn('"enabled"', self.problem({"enabled": "yes", "ram_gib": 40}))

    def test_ram_gib_is_required_and_positive(self):
        self.assertIn("ram_gib", self.problem({"enabled": True}))
        for value in (0, -1, True, "40", 1.5):
            with self.subTest(value=value):
                self.assertIn("ram_gib", self.problem({"enabled": True, "ram_gib": value}))

    def test_unknown_keys_are_refused(self):
        self.assertIn("unknown key", self.problem(block(unknown_key=1)))

    def test_the_fields_are_checked(self):
        for changes, expected in (({"max_entries": 0}, "max_entries"),
                                  ({"max_entries": True}, "max_entries"),
                                  ({"min_tokens": -1}, "min_tokens"),
                                  ({"min_gain": "4096"}, "min_gain"),
                                  ({"max_snapshot_mib": 0}, "max_snapshot_mib"),
                                  ({"ipc_timeout_ms": 0}, "ipc_timeout_ms"),
                                  ({"ipc_timeout_ms": 1.5}, "ipc_timeout_ms")):
            with self.subTest(changes=changes):
                self.assertIn(expected, self.problem(block(**changes)))

    def test_max_snapshot_cannot_exceed_ram_gib(self):
        self.assertIn("max_snapshot_mib", self.problem(block(ram_gib=1, max_snapshot_mib=1025)))
        self.assertIsNotNone(self.plan(block(ram_gib=1, max_snapshot_mib=1024)))

    def test_the_defaults_are_the_plans_start_values(self):
        plan = self.plan({"enabled": True, "ram_gib": 40})
        self.assertEqual((plan.max_entries, plan.min_tokens, plan.min_gain, plan.max_snapshot_mib,
                          plan.ipc_timeout_ms), (20, 0, 4096, 8192, 2000))
        self.assertEqual(plan.capacity_bytes, 40 << 30)
        self.assertTrue(plan.enabled)

    def test_the_default_max_entries_is_ten_per_replica(self):
        self.assertEqual(self.plan({"enabled": True, "ram_gib": 40}, groups=GROUPS3).max_entries, 30)

    def test_two_replicas_are_needed(self):
        self.assertIn('"replicas" 2 or more', self.problem(block(), groups=[]))
        self.assertIn("this start has 1", self.problem(block(), groups=[[0]]))

    def test_replica_pull_is_required(self):
        with self.assertRaises(ParentParkingConfigError) as caught:
            parent_parking_plan(config(block()), GROUPS2, pull=PULL_OFF)
        self.assertIn("replica_pull", str(caught.exception))

    def test_no_batch_slots(self):
        self.assertIn('"parallel": 2', self.problem(block(), parallel=2))
        self.assertIn("--batch 2", self.problem(block(), args=["--batch", "2"]))
        self.assertIn("--slots 2", self.problem(block(), args=["--slots", "2"]))
        self.assertIn("--parallel 2", self.problem(block(), args=["--parallel", "2"]))
        self.assertIsNotNone(self.plan(block(), parallel=1))                 # one request each is what it runs

    def test_split_serving_is_refused(self):
        self.assertIn("split", self.problem(block(), split_serve={"prefill_gpu": 0, "decode_gpu": 1}))

    def test_posix_only(self):
        self.assertIn("POSIX", self.problem(block(), posix=False))

    def test_replica_handoff_at_the_same_time_is_refused(self):
        cfg = config(block())
        cfg["replica_handoff"] = {"enabled": True, "dir": "/srv/strata-handoff", "disk_gib_per_replica": 16}
        with self.assertRaises(ParentParkingConfigError) as caught:
            parent_parking_plan(cfg, GROUPS2, pull=PULL_ON)
        self.assertIn("replica_handoff", str(caught.exception))

    def test_a_hand_written_cache_directory_is_refused(self):
        for flag in ("--conversation-cache-dir", "--conversation-cache-peer-dir"):
            with self.subTest(flag=flag):
                self.assertIn(flag, self.problem(block(), args=[flag, "/somewhere/else"]))

    def test_a_hand_written_parent_flag_is_refused(self):
        for flag in server.PARENT_PARKING_ENGINE_FLAGS:
            with self.subTest(flag=flag):
                self.assertIn(flag, self.problem(block(), args=[flag, "1"]))

    def test_a_hand_written_parent_flag_in_user_args_is_refused(self):
        cfg = config(block())
        cfg["user_args"] = ["--conversation-parent-parking-fd", "7"]
        with self.assertRaises(ParentParkingConfigError) as caught:
            parent_parking_plan(cfg, GROUPS2, pull=PULL_ON)
        self.assertIn("--conversation-parent-parking-fd", str(caught.exception))

    def test_a_zero_cache_setting_is_refused(self):
        for flag in ("--conversation-cache-mib", "--conversation-cache-slots", "--prompt-cache"):
            with self.subTest(flag=flag):
                self.assertIn(flag, self.problem(block(), args=[flag, "0"]))
        self.assertIsNotNone(self.plan(block(), args=["--conversation-cache-mib", "2048"]))

    def test_the_engine_flags_are_the_thresholds(self):
        plan = self.plan(block())
        self.assertEqual(parent_parking_args(plan),
                         ["--conversation-parent-parking-min-tokens", "8192",
                          "--conversation-parent-parking-min-gain", "4096",
                          "--conversation-parent-parking-max-snapshot-mib", "8192",
                          "--conversation-parent-parking-timeout-ms", "2000"])

    def test_the_start_line(self):
        self.assertEqual(parent_parking_note(self.plan(block()), 2),
                         "replica parent parking: one RAM store for 2 replicas, 40 GiB, 20 entries, publish on park, "
                         "snapshots survive replica restart but not server restart")


class FakeEngineProcess:
    """A stand-in for the engine process: it answers READY (or the lines it is given) and keeps a duplicate of the
    channel fd it was handed, so the store's channel stays up after StrataEngine closes its own copy."""

    def __init__(self, child_fd=None, stdout="READY 4096\n"):
        self.stdout = io.StringIO(stdout)
        self.stdin = io.StringIO()
        self.returncode = None
        self.child_fd = os.dup(child_fd) if child_fd is not None else None
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def terminate(self):
        self.returncode = -15

    def close_channel(self):
        if self.child_fd is not None:
            os.close(self.child_fd)
            self.child_fd = None


class ChannelCase(unittest.TestCase):
    """A real store and fake engine processes: the channel is a real socketpair and the fd passing is real."""

    def setUp(self):
        self.store = pp.ParentParkingStore(pp.ParentParkingConfig(ram_gib=1, max_entries=10, max_snapshot_mib=64),
                                           IDENTITY)
        self.addCleanup(self.store.close)
        self.record = []

    def tearDown(self):
        for row in self.record:
            row["proc"].close_channel()

    def popen(self, stdout="READY 4096\n"):
        """A popen that records the start and hands the fake process a duplicate of the channel fd."""
        def popen(what, argv, **kw):
            child = kw.get("pass_fds") or ()
            proc = FakeEngineProcess(child[0] if child else None, stdout)
            self.record.append({"args": list(argv), "kwargs": kw, "proc": proc})
            return proc
        return popen

    def start(self, owner="r0", args=None, stdout="READY 4096\n", **kwargs):
        with mock.patch.object(server, "popen", self.popen(stdout)), \
                mock.patch.object(server, "end_leftover_engines"):
            engine = StrataEngine("fake-engine", list(args or []), parent_parking=self.store, parent_owner=owner,
                                  **kwargs)
        self.addCleanup(engine.close)
        return engine

    def wait_state(self, owner, state, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.store.replica_state(owner) == state:
                return
            time.sleep(0.01)
        self.fail(f"{owner} is {self.store.replica_state(owner)!r}, not {state!r}")


class ParentParkingChannel(ChannelCase):
    """Plan §12.2: a new channel before EVERY popen, the fd in pass_fds, the parent's copy closed, and the store kept
    through every retry and restart path."""

    def test_the_first_start_opens_a_channel_and_passes_the_fd(self):
        engine = self.start()
        self.assertEqual(len(self.record), 1)
        args, kw = self.record[0]["args"], self.record[0]["kwargs"]
        child = kw["pass_fds"][0]
        self.assertEqual(args[args.index("--conversation-parent-parking-fd") + 1], str(child))
        self.assertEqual(args[args.index("--conversation-parent-parking-owner") + 1], "r0")
        with self.assertRaises(OSError):
            os.fstat(child)                              # the parent's copy is closed after the spawn
        self.wait_state("r0", "connected")
        self.assertEqual(engine.spawn[1], [])            # the command a restart starts again has no fd in it

    def test_every_restart_opens_a_new_channel(self):
        calls = []
        real = self.store.open_channel
        self.store.open_channel = lambda owner: (calls.append(owner), real(owner))[1]
        with mock.patch.object(server, "popen", self.popen()), \
                mock.patch.object(server, "end_leftover_engines"):
            engine = StrataEngine("fake-engine", [], parent_parking=self.store, parent_owner="r0")
            self.addCleanup(engine.close)
            self.assertEqual(calls, ["r0"])
            engine.restart()
        self.assertEqual(calls, ["r0", "r0"])
        self.assertEqual(len(self.record), 2)
        self.assertNotEqual(self.record[0]["proc"].child_fd, self.record[1]["proc"].child_fd)
        for row in self.record:
            self.assertEqual(row["args"].count("--conversation-parent-parking-fd"), 1)

    def test_a_lazy_start_opens_no_channel_until_the_first_load(self):
        with mock.patch.object(server, "popen", self.popen()), \
                mock.patch.object(server, "end_leftover_engines"):
            engine = StrataEngine("fake-engine", [], lazy=True, parent_parking=self.store, parent_owner="r0")
            self.addCleanup(engine.close)
            self.assertEqual(self.record, [])
            self.assertEqual(self.store.replica_state("r0"), "closed")
            engine.restart()
        self.assertEqual(len(self.record), 1)
        self.assertIn("--conversation-parent-parking-fd", self.record[0]["args"])

    def retry_popen(self, procs):
        """A popen that hands out `procs` in order, each with a duplicate of its start's channel fd."""
        def popen(what, argv, **kw):
            proc = procs[len(self.record)]
            child = kw.get("pass_fds") or ()
            if child:
                proc.child_fd = os.dup(child[0])
            self.record.append({"args": list(argv), "kwargs": kw, "proc": proc})
            return proc
        return popen

    def test_a_retry_start_gets_its_own_channel(self):
        procs = [FakeEngineProcess(None, ""), FakeEngineProcess(None, "READY 4096\n")]
        with mock.patch.object(server, "popen", self.retry_popen(procs)), \
                mock.patch.object(server, "end_leftover_engines"):
            engine = StrataEngine("fake-engine", [], parent_parking=self.store, parent_owner="r0",
                                  retry_args=["--retry"])
            self.addCleanup(engine.close)
        self.assertEqual(len(self.record), 2)
        self.assertIn("--retry", self.record[1]["args"])
        self.assertEqual(self.record[1]["args"].count("--conversation-parent-parking-fd"), 1)
        self.assertEqual(engine.spawn[1], ["--retry"])   # the arguments that worked are kept for a later restart

    def test_a_nodraft_retry_gets_its_own_channel(self):
        procs = [FakeEngineProcess(None, ""), FakeEngineProcess(None, "READY 4096\n")]
        with mock.patch.object(server, "popen", self.retry_popen(procs)), \
                mock.patch.object(server, "end_leftover_engines"), \
                mock.patch.object(server.vram_fit, "engine_stop_reason", return_value="expert cache: no room"):
            engine = StrataEngine("fake-engine", [], parent_parking=self.store, parent_owner="r0",
                                  retry_vram_args=["--nodraft"])
            self.addCleanup(engine.close)
        self.assertEqual(len(self.record), 2)
        self.assertIn("--nodraft", self.record[1]["args"])
        self.assertEqual(self.record[1]["args"].count("--conversation-parent-parking-fd"), 1)
        self.assertEqual(engine.spawn[1], ["--nodraft"])

    def test_a_failed_ready_is_a_start_error_and_the_channel_degrades(self):
        def popen(what, argv, **kw):
            proc = FakeEngineProcess(None, "ERR the server's parking store refused this engine: identity does not "
                                           "match the store\n")
            self.record.append({"args": list(argv), "kwargs": kw, "proc": proc})
            return proc
        with mock.patch.object(server, "popen", popen), \
                mock.patch.object(server, "end_leftover_engines"):
            with self.assertRaises(EngineDied) as caught:
                StrataEngine("fake-engine", [], parent_parking=self.store, parent_owner="r0")
        self.assertIn("refused", str(caught.exception))
        self.assertIn("identity", str(caught.exception))
        child = self.record[0]["kwargs"]["pass_fds"][0]
        with self.assertRaises(OSError):
            os.fstat(child)                              # the failed start leaves no fd behind
        self.wait_state("r0", "degraded")

    def test_a_runtime_eof_only_degrades_and_does_not_end_the_engine(self):
        engine = self.start()
        self.wait_state("r0", "connected")
        proc = self.record[0]["proc"]
        proc.close_channel()                             # the engine's client went away
        self.wait_state("r0", "degraded")
        self.assertFalse(proc.killed)
        self.assertIs(engine.proc, proc)

    def test_parallel_starts_do_not_mix_owners_or_channels(self):
        engines = {}

        def start(owner):
            engines[owner] = StrataEngine("fake-engine", [], parent_parking=self.store, parent_owner=owner)

        with mock.patch.object(server, "popen", self.popen()), \
                mock.patch.object(server, "end_leftover_engines"):
            threads = [threading.Thread(target=start, args=(f"r{i}",)) for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        for engine in engines.values():
            self.addCleanup(engine.close)
        self.assertEqual(len(self.record), 2)
        owners = [row["args"][row["args"].index("--conversation-parent-parking-owner") + 1] for row in self.record]
        self.assertEqual(sorted(owners), ["r0", "r1"])
        fds = [row["kwargs"]["pass_fds"][0] for row in self.record]
        self.assertEqual(len(set(fds)), 2)
        for owner in ("r0", "r1"):
            self.wait_state(owner, "connected")


class FakeEngine(MockEngine):
    """A MockEngine with the attributes a replica engine has: its own log, its INFO facts, and the lifecycle calls
    counted (serve/test_replica_handoff.py's FakeEngine, plus close/unload)."""

    def __init__(self, tokenizer, log_path=None):
        super().__init__(tokenizer, "Hi.", max_context=4096)
        self.info = {}
        self.log_path, self.log_start = log_path, 0
        self.closed = 0
        self.unloaded = 0

    def alive(self):
        return True

    def close(self):
        self.closed += 1

    def unload(self):
        self.unloaded += 1

    def restart(self, tries=3):
        pass


class ReplicaLifecycle(unittest.TestCase):
    """Plan §12.3: restart keeps the store, unload clears it, close ends the engines before the broker, and the view
    carries each replica's channel state."""

    def setUp(self):
        self.store = pp.ParentParkingStore(pp.ParentParkingConfig(ram_gib=1, max_entries=10, max_snapshot_mib=64),
                                           IDENTITY)
        self.addCleanup(self.store.close)
        self.client = Client(self.store, "r0")
        self.addCleanup(self.client.close)
        self.client.publish(POINTS_A, [4096])

    def engine(self, **kwargs):
        tok = ByteTokenizer()
        self.e0, self.e1 = FakeEngine(tok), FakeEngine(tok)
        return ReplicaEngine([self.e0, self.e1], **kwargs)

    def test_restart_keeps_the_parked_conversations(self):
        eng = self.engine(parent_parking=self.store)
        eng.restart()
        snapshot = self.store.snapshot()
        self.assertEqual((snapshot["entries"], snapshot["clears"]), (1, 0))

    def test_unload_unloads_the_engines_then_clears_the_store(self):
        eng = self.engine(parent_parking=self.store)
        eng.unload()
        self.assertEqual((self.e0.unloaded, self.e1.unloaded), (1, 1))
        snapshot = self.store.snapshot()
        self.assertEqual((snapshot["entries"], snapshot["clears"]), (0, 1))

    def test_unload_clears_the_store_even_when_an_engine_will_not_unload(self):
        eng = self.engine(parent_parking=self.store)

        def stuck():
            raise RuntimeError("still releasing GPU/RAM")

        self.e0.unload = stuck
        with self.assertRaises(RuntimeError):
            eng.unload()
        self.assertEqual(self.store.snapshot()["entries"], 0)     # the RAM is the point of an unload

    def test_close_closes_the_engines_then_the_broker(self):
        eng = self.engine(parent_parking=self.store)
        order = []
        self.e0.close = lambda: order.append("r0")
        self.e1.close = lambda: order.append("r1")
        real = self.store.close
        self.store.close = lambda: (order.append("store"), real())
        eng.close()
        self.assertEqual(order, ["r0", "r1", "store"])
        with self.assertRaises(pp.ParentParkingClosed):
            self.store.open_channel("r0")

    def test_replica_view_carries_each_channels_state(self):
        eng = self.engine(parent_parking=self.store)
        self.assertEqual([row["parent_parking"] for row in eng.replica_view()], ["connected", "closed"])
        self.client.close()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and eng.replica_view()[0]["parent_parking"] != "degraded":
            time.sleep(0.01)
        self.assertEqual(eng.replica_view()[0]["parent_parking"], "degraded")

    def test_without_the_store_the_view_keeps_its_old_fields(self):
        eng = self.engine()
        self.assertEqual(set(eng.replica_view()[0]), {"replica", "alive", "running", "routed", "batch"})


class ParentParkingMetrics(unittest.TestCase):
    """Plan §14.2: /metrics and /v1/status add the store's snapshot only when the mode is on, /metrics/prometheus adds
    the global gauges and counters, and the default-off answers keep exactly their old fields."""

    def setUp(self):
        self.store = pp.ParentParkingStore(pp.ParentParkingConfig(ram_gib=1, max_entries=10, max_snapshot_mib=64),
                                           IDENTITY)
        self.addCleanup(self.store.close)
        self.clients = [Client(self.store, "r0"), Client(self.store, "r1")]
        for client in self.clients:
            self.addCleanup(client.close)
        self.clients[0].publish(POINTS_A, [4096])
        self.clients[1].publish(POINTS_B, [4096])

    def service(self, store=None):
        tok = ByteTokenizer()
        return Service(ReplicaEngine([FakeEngine(tok), FakeEngine(tok)], parent_parking=store), tok, TEMPLATE)

    def test_metrics_carry_the_stores_snapshot(self):
        svc = self.service(self.store)
        snapshot = self.store.snapshot()
        self.assertEqual(svc.metrics()["conversation_parent_parking"], snapshot)
        self.assertEqual((snapshot["entries"], snapshot["max_entries"], snapshot["published"], snapshot["budget_violations"]),
                         (2, 10, 2, 0))
        # the global figures are the store's own, and each replica's events stay in its own row
        self.assertEqual([row["replica"] for row in snapshot["replicas"]], [0, 1])
        self.assertEqual([row["published"] for row in snapshot["replicas"]], [1, 1])
        self.assertEqual([row["state"] for row in snapshot["replicas"]], ["connected", "connected"])

    def test_status_carries_the_stores_snapshot(self):
        svc = self.service(self.store)
        self.assertEqual(svc.v1_status()["conversation_parent_parking"], self.store.snapshot())

    def test_prometheus_has_the_global_gauges_and_no_labels(self):
        svc = self.service(self.store)
        values = {}
        for line in svc.prometheus().splitlines():
            if line.startswith("strata:conversation_parent_"):
                name, _, value = line.partition(" ")
                values[name] = value
        self.assertEqual(values["strata:conversation_parent_capacity_bytes"], str(1 << 30))
        self.assertEqual(values["strata:conversation_parent_entries"], "2")
        self.assertEqual(values["strata:conversation_parent_published_total"], "2")
        self.assertEqual(values["strata:conversation_parent_budget_violations"], "0")
        self.assertEqual(values["strata:conversation_parent_store_epoch"], "1")
        self.assertNotIn("{", "".join(values))           # no owner or entry-id labels: they would be high cardinality

    def test_the_default_off_answers_gain_no_field(self):
        svc = self.service()
        self.assertNotIn("conversation_parent_parking", svc.metrics())
        self.assertNotIn("conversation_parent_parking", svc.v1_status())
        self.assertNotIn("conversation_parent", svc.prometheus())


class ParentParkingStart(unittest.TestCase):
    """start_replicas: ONE store for the whole server, passed to every engine, and the start line of plan §13.3."""

    class StartedEngine:
        def __init__(self, args, **kwargs):
            self.args, self.kwargs = list(args), kwargs

        def alive(self):
            return True

    def start(self, cfg, groups, record, *, retry_seen=None, **kwargs):
        def fake_engine(exe, args, **kwargs):
            record.append((list(args), kwargs))
            return self.StartedEngine(args, **kwargs)

        def fake_mtp(cfg, exe, extra=None):
            if retry_seen is not None:
                retry_seen.append(list(cfg["args"]))
            return None

        output = io.StringIO()
        with mock.patch.object(server, "StrataEngine", side_effect=fake_engine), \
                mock.patch.object(server, "engine_args", side_effect=lambda cfg, exe=None, **kw: list(cfg["args"])), \
                mock.patch.object(server, "child_env", return_value={}), \
                mock.patch.object(server, "mtp_retry_args", side_effect=fake_mtp), \
                mock.patch.object(server, "nodraft_retry_args", return_value=None), \
                mock.patch.object(server, "warn_budget_over_ram"), \
                mock.patch.object(rp, "cpu_sets", return_value=None), \
                mock.patch.object(rp, "shared_arena_plan", return_value=(None, "")), \
                mock.patch.object(rp, "ReplicaEngine", return_value=object()), \
                contextlib.redirect_stdout(output):
            server.start_replicas(cfg, groups, "exe", [], False, offline=True, **kwargs)
        return output.getvalue()

    def test_one_store_for_every_replica_and_the_start_line(self):
        record = []
        out = self.start(config(block()), GROUPS2, record, pull=PULL_ON)
        store = record[0][1]["parent_parking"]
        self.addCleanup(store.close)
        self.assertTrue(all(kw["parent_parking"] is store for _args, kw in record))
        self.assertEqual([kw["parent_owner"] for _args, kw in record], ["r0", "r1"])
        for args, _kw in record:
            self.assertEqual(args[args.index("--conversation-parent-parking-min-tokens") + 1], "8192")
            self.assertEqual(args[args.index("--conversation-parent-parking-timeout-ms") + 1], "2000")
        self.assertIn("[strata] replica parent parking: one RAM store for 2 replicas, 40 GiB, 20 entries, publish on "
                      "park, snapshots survive replica restart but not server restart", out)

    def test_three_replicas_share_the_one_store(self):
        record = []
        out = self.start(config(block()), GROUPS3, record, pull=PULL_ON)
        store = record[0][1]["parent_parking"]
        self.addCleanup(store.close)
        self.assertEqual(len(record), 3)
        self.assertTrue(all(kw["parent_parking"] is store for _args, kw in record))
        self.assertEqual([kw["parent_owner"] for _args, kw in record], ["r0", "r1", "r2"])
        self.assertIn("one RAM store for 3 replicas", out)

    def test_the_retry_arguments_carry_the_thresholds(self):
        record, retry_seen = [], []
        self.start(config(block()), GROUPS2, record, retry_seen=retry_seen, pull=PULL_ON)
        store = record[0][1]["parent_parking"]
        self.addCleanup(store.close)
        self.assertEqual(len(retry_seen), 2)
        for args in retry_seen:
            self.assertIn("--conversation-parent-parking-min-tokens", args)
            self.assertIn("--conversation-parent-parking-timeout-ms", args)

    def test_no_block_means_no_store_and_no_new_flags(self):
        record = []
        out = self.start({"args": []}, GROUPS2, record)
        for args, kw in record:
            self.assertIsNone(kw.get("parent_parking"))
            self.assertIsNone(kw.get("parent_owner"))
            self.assertFalse([a for a in args if a.startswith("--conversation-parent-parking-")])
        self.assertNotIn("replica parent parking", out)

    def test_an_unusable_block_stops_the_start(self):
        with self.assertRaises(ParentParkingConfigError):
            server.start_replicas(config(block(), parallel=2), GROUPS2, "exe", [], False, pull=PULL_ON)


if __name__ == "__main__":
    unittest.main()
