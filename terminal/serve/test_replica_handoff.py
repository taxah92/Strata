"""Stage 3 (Python): the run config's "replica_handoff" block, the per-replica writer/peer directories, and the
per-replica conversation-cache metrics.  No native model or GPU is required (the logs are files, the engines are fakes).

The config and the flags are the frozen handoff contract's (§6 and §8 of the design notes): a block that cannot be
used is a start error, never a silent fallback to no handoff, and a replica's figures are never another replica's.
"""
import contextlib
import io
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.replicas import ReplicaEngine
from serve.server import (ByteTokenizer, ConvCacheLog, MockEngine, ReplicaHandoffConfigError, Service,
                          ensure_replica_handoff_dirs, replica_handoff_args, replica_handoff_plan,
                          replica_queue_view)

TEMPLATE = ChatTemplate(Path(__file__).parent / "chat_template.jinja")
GROUPS2, GROUPS3 = [[0, 1], [2, 3]], [[0], [1], [2]]


def handoff(**changes):
    """The contract's start block (§6), with any field changed for one test."""
    block = {"enabled": True, "dir": "/srv/strata-handoff", "spill": "on-park", "min_tokens": 8192,
             "min_gain": 4096, "disk_gib_per_replica": 16, "max_snapshot_mib": 8192, "min_free_mib": 8192,
             "refresh_ms": 250}
    block.update(changes)
    return block


def config(block=None, **kwargs):
    cfg = {"args": [], **kwargs}
    if block is not None:
        cfg["replica_handoff"] = block
    return cfg


class HandoffConfig(unittest.TestCase):
    """The block is read and checked at start: every refusal names what to fix."""

    def plan(self, block, groups=GROUPS2, **cfg_keys):
        return replica_handoff_plan(config(block, **cfg_keys), groups, posix=True)

    def problem(self, block, groups=GROUPS2, posix=True, **cfg_keys):
        with self.assertRaises(ReplicaHandoffConfigError) as e:
            replica_handoff_plan(config(block, **cfg_keys), groups, posix=posix)
        return str(e.exception)

    def test_off_by_default_and_when_disabled(self):
        self.assertIsNone(replica_handoff_plan(config(), GROUPS2))
        self.assertIsNone(self.plan({"enabled": False}))
        self.assertIsNone(self.plan({"enabled": False, "dir": "/x"}))       # a disabled block is not read further

    def test_the_block_must_be_an_object(self):
        self.assertIn("object", self.problem([1, 2]))
        self.assertIn('"enabled"', self.problem({"enabled": "yes", "dir": "/x"}))

    def test_enabled_needs_a_dir(self):
        self.assertIn('"dir"', self.problem({"enabled": True}))
        self.assertIn('"dir"', self.problem({"enabled": True, "dir": "  "}))
        self.assertIn("absolute", self.problem({"enabled": True, "dir": "relative/store"}))

    def test_enabled_needs_the_disk_budget(self):
        block = handoff()
        del block["disk_gib_per_replica"]
        self.assertIn("disk_gib_per_replica", self.problem(block))

    def test_two_replicas_are_needed(self):
        self.assertIn('"replicas" 2 or more', self.problem(handoff(), groups=[]))
        self.assertIn("this start has 0", self.problem(handoff(), groups=[]))
        self.assertIn("this start has 1", self.problem(handoff(), groups=[[0]]))

    def test_no_batch_slots(self):
        full = handoff()
        self.assertIn('"parallel": 2', self.problem(full, parallel=2))
        self.assertIn("--batch 2", self.problem(full, args=["--batch", "2"]))
        self.assertIn("--slots 2", self.problem(full, args=["--slots", "2"]))
        self.assertIn("--parallel 2", self.problem(full, args=["--parallel", "2"]))
        per_replica = {"replicas": [{"gpus": [0], "args": ["--batch", "2"]}, {"gpus": [1]}]}
        with self.assertRaises(ReplicaHandoffConfigError) as e:
            replica_handoff_plan(config(full, **per_replica), groups=[[0], [1]], posix=True)
        self.assertIn("--batch 2", str(e.exception))
        self.assertIsNotNone(self.plan(full, parallel=1))                     # one request each is what it runs

    def test_posix_only(self):
        self.assertIn("POSIX", self.problem(handoff(), posix=False))

    def test_a_hand_written_writer_directory_is_refused(self):
        message = self.problem(handoff(), args=["--conversation-cache-dir", "/somewhere/else"])
        self.assertIn("--conversation-cache-dir", message)
        self.assertIn("0, 1", message)                      # the same option for every replica: they would share it
        cfg = config(handoff(), replicas=[{"gpus": [0], "args": ["--conversation-cache-dir", "/mine"]},
                                          {"gpus": [1]}])
        with self.assertRaises(ReplicaHandoffConfigError) as e:
            replica_handoff_plan(cfg, [[0], [1]], posix=True)
        self.assertIn("replica 0", str(e.exception))
        message = self.problem(handoff(), args=["--conversation-cache-dir=/somewhere/else"])
        self.assertIn("--conversation-cache-dir", message)

    def test_a_hand_written_peer_directory_is_refused(self):
        cfg = config(handoff(), args=[], replicas=[{"gpus": [0], "args": ["--conversation-cache-peer-dir", "/peer"]},
                                                   {"gpus": [1]}])
        with self.assertRaises(ReplicaHandoffConfigError) as e:
            replica_handoff_plan(cfg, [[0], [1]], posix=True)
        self.assertIn("--conversation-cache-peer-dir", str(e.exception))
        self.assertIn("replica 0", str(e.exception))

    def test_the_same_writer_directory_on_two_replicas_is_refused(self):
        cfg = config(handoff(), replicas=[{"gpus": [0], "args": ["--conversation-cache-dir", "/same"]},
                                          {"gpus": [1], "args": ["--conversation-cache-dir", "/same"]}])
        with self.assertRaises(ReplicaHandoffConfigError) as e:
            replica_handoff_plan(cfg, [[0], [1]], posix=True)
        message = str(e.exception)
        self.assertIn("/same", message)
        self.assertIn("0, 1", message)
        self.assertIn("own", message)

    def test_a_shared_person_option_counts_too(self):
        cfg = config(handoff(), args=[])
        cfg["user_args"] = ["--conversation-cache-dir", "/live"]
        with self.assertRaises(ReplicaHandoffConfigError) as e:
            replica_handoff_plan(cfg, GROUPS2, posix=True)
        self.assertIn("0, 1", str(e.exception))

    def test_the_fields_are_checked(self):
        for block, expected in (
                (handoff(spill="eager"), '"spill"'),
                (handoff(unknown_key=1), "unknown key"),
                (handoff(min_tokens=-1), '"min_tokens"'),
                (handoff(min_gain="4096"), '"min_gain"'),
                (handoff(max_snapshot_mib=0), '"max_snapshot_mib"'),
                (handoff(min_free_mib=-1), '"min_free_mib"'),
                (handoff(refresh_ms=1.5), '"refresh_ms"'),
                (handoff(disk_gib_per_replica=0), '"disk_gib_per_replica"')):
            with self.subTest(block=block):
                self.assertIn(expected, self.problem(block))

    def test_the_defaults_are_the_prototypes_start_values(self):
        block = {"enabled": True, "dir": "/srv/strata-handoff", "disk_gib_per_replica": 16}
        plan = self.plan(block)
        self.assertEqual((plan["min_gain"], plan["refresh_ms"], plan["spill"]), (4096, 250, None))
        self.assertIsNone(plan["min_tokens"])
        self.assertIsNone(plan["max_snapshot_mib"])


class HandoffArgs(unittest.TestCase):
    """start_replicas gives every replica its own writer directory and one read-only peer directory per neighbour."""

    def plan(self, groups=GROUPS2, **changes):
        return replica_handoff_plan(config(handoff(**changes)), groups, posix=True)

    def test_two_replicas(self):
        plan = self.plan()
        root = plan["dir"]
        self.assertEqual(replica_handoff_args(plan, 0, 2),
                         ["--conversation-cache-dir", os.path.join(root, "r0"),
                          "--conversation-cache-peer-dir", os.path.join(root, "r1"),
                          "--conversation-cache-owner", "r0",
                          "--conversation-cache-disk-gib", "16",
                          "--conversation-cache-disk-min-gain", "4096",
                          "--conversation-cache-refresh-ms", "250",
                          "--conversation-cache-spill-on-park",
                          "--conversation-cache-min-tokens", "8192",
                          "--conversation-cache-max-snapshot-mib", "8192",
                          "--conversation-cache-min-free-mib", "8192"])
        self.assertEqual(replica_handoff_args(plan, 1, 2)[:4],
                         ["--conversation-cache-dir", os.path.join(root, "r1"),
                          "--conversation-cache-peer-dir", os.path.join(root, "r0")])

    def test_three_replicas_have_two_peers_each(self):
        plan = self.plan(GROUPS3)
        root = plan["dir"]
        for i in range(3):
            args = replica_handoff_args(plan, i, 3)
            peers = [args[k + 1] for k, a in enumerate(args) if a == "--conversation-cache-peer-dir"]
            self.assertEqual(peers, [os.path.join(root, f"r{j}") for j in range(3) if j != i])
            self.assertEqual(args.count("--conversation-cache-dir"), 1)
            self.assertEqual(args[args.index("--conversation-cache-dir") + 1], os.path.join(root, f"r{i}"))
            self.assertEqual(args[args.index("--conversation-cache-owner") + 1], f"r{i}")
            self.assertNotIn(os.path.join(root, f"r{i}"), peers)

    def test_the_optional_flags_are_left_out_when_not_set(self):
        plan = self.plan(min_tokens=None, max_snapshot_mib=None, min_free_mib=None, spill=None)
        args = replica_handoff_args(plan, 0, 2)
        for flag in ("--conversation-cache-spill-on-park", "--conversation-cache-min-tokens",
                     "--conversation-cache-max-snapshot-mib", "--conversation-cache-min-free-mib"):
            self.assertNotIn(flag, args)
        self.assertIn("--conversation-cache-disk-gib", args)


class HandoffDirs(unittest.TestCase):
    def test_each_replica_directory_is_created_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "store")
            plan = {"dir": root, "replicas": 3}
            paths = ensure_replica_handoff_dirs(plan)
            self.assertEqual(paths, [os.path.join(root, f"r{i}") for i in range(3)])
            self.assertEqual(stat.S_IMODE(os.stat(root).st_mode), 0o700)
            for path in paths:
                self.assertTrue(os.path.isdir(path))
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o700)

    def test_an_existing_directory_is_made_private_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "store")
            os.makedirs(os.path.join(root, "r0"))
            os.chmod(os.path.join(root, "r0"), 0o755)
            ensure_replica_handoff_dirs({"dir": root, "replicas": 2})
            self.assertEqual(stat.S_IMODE(os.stat(os.path.join(root, "r0")).st_mode), 0o700)

    def test_an_unusable_path_is_a_start_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = os.path.join(tmp, "file")
            Path(blocker).write_text("not a directory")
            with self.assertRaises(ReplicaHandoffConfigError) as e:
                ensure_replica_handoff_dirs({"dir": blocker, "replicas": 2})
            self.assertIn("cannot create", str(e.exception))


class HandoffStart(unittest.TestCase):
    """start_replicas itself: the handoff flags go into the engines' arguments and the store is made before they run."""

    class FakeEngine:
        def __init__(self, args, **kwargs):
            self.args, self.kwargs = list(args), kwargs

        def alive(self):
            return True

    def start(self, cfg, groups, record):
        import serve.server as server
        from serve import replicas as rp

        def fake_engine(exe, args, **kwargs):
            record.append(list(args))
            return self.FakeEngine(args, **kwargs)

        output = io.StringIO()
        with mock.patch.object(server, "StrataEngine", side_effect=fake_engine), \
                mock.patch.object(server, "engine_args", side_effect=lambda cfg, exe=None, **kw: list(cfg["args"])), \
                mock.patch.object(server, "child_env", return_value={}), \
                mock.patch.object(server, "mtp_retry_args", return_value=None), \
                mock.patch.object(server, "nodraft_retry_args", return_value=None), \
                mock.patch.object(server, "warn_budget_over_ram"), \
                mock.patch.object(rp, "cpu_sets", return_value=None), \
                mock.patch.object(rp, "shared_arena_plan", return_value=(None, "")), \
                mock.patch.object(rp, "ReplicaEngine", return_value=object()), \
                contextlib.redirect_stdout(output):
            server.start_replicas(cfg, groups, "exe", [], False, offline=True)
        return output.getvalue()

    def test_handoff_flags_per_replica_and_the_private_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "store")
            cfg = {"args": [], "replica_handoff": handoff(dir=root)}
            record = []
            out = self.start(cfg, [[0], [1]], record)
            self.assertEqual(len(record), 2)
            for i, args in enumerate(record):
                self.assertEqual(args[args.index("--conversation-cache-owner") + 1], f"r{i}")
                self.assertEqual(args[args.index("--conversation-cache-dir") + 1], os.path.join(root, f"r{i}"))
                self.assertIn(os.path.join(root, f"r{1 - i}"), args)
                self.assertIn("--conversation-cache-spill-on-park", args)
                self.assertEqual(args.count("--conversation-cache-peer-dir"), 1)
            for i in range(2):
                self.assertEqual(stat.S_IMODE(os.stat(os.path.join(root, f"r{i}")).st_mode), 0o700)
            self.assertIn("[strata] replica handoff:", out)
            self.assertIn(f"{root} (r0..r1), 16 GiB per replica", out)
            self.assertIn("spill on park", out)

    def test_no_block_means_no_new_flags_and_no_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = []
            out = self.start({"args": []}, [[0], [1]], record)
            for args in record:
                self.assertFalse([a for a in args if a.startswith("--conversation-cache-")], args)
            self.assertNotIn("replica handoff", out)
            self.assertEqual(os.listdir(tmp), [])

    def test_an_unusable_block_stops_the_start(self):
        import serve.server as server
        with self.assertRaises(ReplicaHandoffConfigError):
            server.start_replicas({"args": [], "parallel": 2, "replica_handoff": handoff()}, GROUPS2, "exe", [], False)


class FakeEngine(MockEngine):
    """A MockEngine with the attributes a replica engine has: its own log and its INFO facts."""

    def __init__(self, tokenizer, log_path, parked_mib=4096):
        super().__init__(tokenizer, "Hi.", max_context=4096)
        self.info = {"conversation_cache_mib": parked_mib, "conversation_cache_slots": 1}
        self.log_path, self.log_start = str(log_path), 0

    def alive(self):
        return True


class PerReplicaMetrics(unittest.TestCase):
    """One ConvCacheLog per replica: /metrics and /v1/status add up the replicas without mixing replica 0 into the sum."""

    R0 = ("strata serve: conversation disk: /srv/handoff/r0 (owner=r0): 0 conversations kept from earlier runs, "
          "0.00 of 16 GiB; 0 dropped, 0 stray files removed; 0 other folders (0.00 GiB) left alone\n"
          "strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
          "snapshot_bytes=104857600 reused_kv_bytes=0\n"
          "strata serve: conversation disk: wrote 8192 tokens (on-park) in 12.5 ms, 2.30 GB/s; entries=3 "
          "bytes=734003200 evictions=0\n"
          "strata serve: conversation disk: read 12000 tokens (11800 reused, live, peer r1) in 40.0 ms, 2.30 GB/s; "
          "entries=1 bytes=100000000\n"
          "strata serve: conversation disk: stats (on-park) wrote=1 read=1 evictions=0 skipped=2 lease_conflicts=1 "
          "overshoot=0 entries=3 bytes=734003200 peer_read=1 peer_refresh=3 peer_entries=5 peer_bytes=1000000000\n")
    R1 = ("strata serve: conversation disk: /srv/handoff/r1 (owner=r1): 0 conversations kept from earlier runs, "
          "0.00 of 16 GiB; 0 dropped, 0 stray files removed; 0 other folders (0.00 GiB) left alone\n"
          "strata serve: conversation cache: parked 7000 tokens in 20.0 ms; parked=2 bytes=209715200 evictions=0 "
          "snapshot_bytes=209715200 reused_kv_bytes=0\n"
          "strata serve: conversation disk: read 80000 tokens (79000 reused, live) in 800.0 ms, 2.60 GB/s; "
          "entries=2 bytes=100000000\n"
          "strata serve: conversation disk: stats (read) wrote=0 read=1 evictions=0 skipped=0 lease_conflicts=0 "
          "overshoot=0 entries=2 bytes=100000000 peer_read=0 peer_refresh=1 peer_entries=0 peer_bytes=0 "
          "peer_miss=1\n")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logs = [Path(self.tmp.name) / "r0.log", Path(self.tmp.name) / "r1.log"]
        self.logs[0].write_text(self.R0, encoding="utf-8")
        self.logs[1].write_text(self.R1, encoding="utf-8")
        tok = ByteTokenizer()
        self.engine = ReplicaEngine([FakeEngine(tok, self.logs[0]), FakeEngine(tok, self.logs[1])])
        self.svc = Service(self.engine, tok, TEMPLATE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_replica_has_its_own_reader(self):
        c = ConvCacheLog()
        self.assertEqual(c.poll(str(self.logs[0]), 0)["parked"], 1)
        self.assertEqual(c.poll(str(self.logs[1]), 0)["parked"], 2)     # another log: its own figures, not r0's
        self.assertEqual(c.poll(str(self.logs[0]), 0)["parked"], 1)

    def test_metrics_are_per_replica_and_summed(self):
        cache = self.svc.metrics()["conversation_cache"]
        self.assertEqual([r["replica"] for r in cache["replicas"]], [0, 1])
        self.assertEqual((cache["replicas"][0]["parked"], cache["replicas"][1]["parked"]), (1, 2))
        self.assertEqual(cache["parked"], 1)                           # the old fields stay replica 0's
        self.assertEqual(cache["total"]["parked"], 3)                  # the sum is its own object
        self.assertNotEqual(cache["parked"], cache["total"]["parked"])
        self.assertEqual((cache["replicas"][0]["spills"], cache["replicas"][1]["spills"]), (1, 0))
        self.assertEqual(cache["total"]["spills"], 1)
        self.assertEqual(cache["replicas"][0]["peer_reads"], 1)
        self.assertEqual(cache["replicas"][0]["lease_conflicts"], 1)
        self.assertEqual(cache["total"]["peer_reads"], 1)
        self.assertEqual(cache["total"]["lease_conflicts"], 1)
        self.assertEqual(cache["total"]["evictions_skipped"], 2)
        self.assertEqual(cache["total"]["peer_refreshes"], 4)
        self.assertEqual(cache["total"]["peer_entries"], 5)
        self.assertEqual(cache["replicas"][1]["peer_misses"], 1)
        self.assertEqual(cache["total"]["restore_tokens"], 80000)      # replica 1's own folder
        self.assertEqual(cache["total"]["restore_reused"], 79000)
        self.assertEqual(cache["total"]["restore_read"], 1000)
        self.assertEqual(cache["total"]["peer_restore_tokens"], 12000)  # replica 0's peer read
        self.assertEqual(cache["total"]["peer_restore_reused"], 11800)
        self.assertEqual(cache["total"]["disk_write_causes"], {"on-park": 1})

    def test_metrics_put_each_reader_next_to_its_replica(self):
        m = self.svc.metrics()
        rows = {row["replica"]: row for row in m["live"]["replicas"]}
        self.assertEqual(rows[0]["cache"]["parked"], 1)
        self.assertEqual(rows[1]["cache"]["parked"], 2)
        self.assertEqual(rows[1]["cache"]["restore_tokens"], 80000)
        self.assertEqual(rows[0]["cache"]["peer_restore_tokens"], 12000)
        self.assertIsNone(m["live"]["replica_queue"]["depth"])         # the pull scheduler is off: no queue

    def test_status_is_per_replica_and_summed(self):
        status = self.svc.v1_status()
        self.assertEqual([r["replica"] for r in status["replicas"]], [0, 1])
        self.assertEqual([r["replica"] for r in status["conversation_cache"]["replicas"]], [0, 1])
        self.assertEqual(status["conversation_cache"]["total"]["parked"], 3)
        self.assertEqual(status["conversation_cache"]["replicas"][0]["parked"], 1)
        self.assertEqual(status["conversation_cache"]["total"]["peer_restore_tokens"], 12000)
        self.assertIsNone(status["replica_queue"]["depth"])

    def test_a_new_line_reaches_only_its_own_replica(self):
        self.svc.metrics()                                            # reads both logs up to now
        with open(self.logs[1], "a", encoding="utf-8") as f:
            f.write("strata serve: conversation cache: parked 9000 tokens in 3.0 ms; parked=3 bytes=300 evictions=0\n")
        cache = self.svc.metrics()["conversation_cache"]
        self.assertEqual((cache["replicas"][0]["parked"], cache["replicas"][1]["parked"]), (1, 3))
        self.assertEqual(cache["total"]["parked"], 4)
        self.assertEqual(cache["parked"], 1)                           # r0 unchanged

    def test_a_single_engine_keeps_the_old_card(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hi.", max_context=4096), tok, TEMPLATE)
        cache = svc.metrics()["conversation_cache"]
        self.assertNotIn("replicas", cache)
        self.assertNotIn("total", cache)
        self.assertFalse([key for key in cache if key.startswith(("spill", "peer_", "restore_", "lease_",
                                                                 "disk_evictions", "disk_write_failed"))], cache)
        self.assertNotIn("conversation_cache", svc.v1_status())


class ReplicaQueue(unittest.TestCase):
    """The pull scheduler's ONE pending queue (serve/replicas.py replica_view): reported once, not summed over replicas."""

    QUEUE = {"depth": 3, "oldest_s": 12.5, "max": 16, "timeout_s": 60.0, "picked": 1, "reason": "reuse",
             "assigned": 7, "starved": 1, "cancelled": 2, "timed_out": 0, "rejected": 3}

    def test_the_shared_queue_is_reported_once(self):
        rows = [{"replica": 0, "alive": True, "queue": self.QUEUE},
                {"replica": 1, "alive": False, "queue": self.QUEUE}]
        view = replica_queue_view(rows)
        self.assertEqual((view["depth"], view["oldest_s"], view["picked"], view["reason"]), (3, 12.5, 1, "reuse"))
        self.assertEqual((view["assigned"], view["starved"], view["cancelled"], view["rejected"]), (7, 1, 2, 3))
        self.assertEqual(view["reachable"], 1)                       # (the rows' alive, not a queue figure)

    def test_without_the_scheduler_there_is_no_queue(self):
        view = replica_queue_view([{"replica": 0, "alive": True, "running": 0, "routed": 0}])
        self.assertIsNone(view["depth"])
        self.assertIsNone(view["oldest_s"])


class EngineLogFormats(unittest.TestCase):
    """The engine log lines the per-replica reader counts (stage 3): the exact formats the C++ side prints.  The
    older #5835 lines without the peer marker keep their meaning, so an engine of the previous build still reports."""

    LINES = (
        "strata serve: conversation disk: /srv/handoff/r0 (owner=r0): 2 conversations kept from earlier runs, "
        "1.20 of 16 GiB; 0 dropped, 1 stray files removed; 1 other folders (0.50 GiB) left alone\n"
        "strata serve: conversation disk: peer /srv/handoff/r1 (owner=r1): 5 conversations, 2.10 GiB, read-only\n"
        "strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
        "snapshot_bytes=104857600 reused_kv_bytes=0\n"
        "strata serve: conversation disk: wrote 8192 tokens (on-park) in 12.5 ms, 2.30 GB/s; entries=3 "
        "bytes=734003200 evictions=0\n"
        "strata serve: conversation disk: not written (on-park, 9000 tokens): min_tokens [none]\n"
        "strata serve: conversation disk: read 80000 tokens (79000 reused, live) in 800.0 ms, 2.60 GB/s; "
        "entries=3 bytes=734003200\n"
        "strata serve: conversation disk: read 12000 tokens (11800 reused, live, peer r1) in 40.0 ms, 2.30 GB/s; "
        "entries=5 bytes=500000000\n"
        "strata serve: conversation disk: cannot read a conversation (12000 tokens, peer): ram floor [memory]\n"
        "strata serve: conversation disk: peer r1 refreshed: 3 new, 1 gone, 0 damaged\n"
        "strata serve: conversation disk: stats (on-park) wrote=1 read=2 evictions=0 skipped=1 lease_conflicts=2 "
        "overshoot=0 entries=5 bytes=500000000 peer_read=1 peer_refresh=2 peer_entries=3 peer_bytes=734003200 "
        "spill_bytes=734003200 peer_miss=2\n"
        "strata serve: something else entirely: 42\n")

    def test_every_counted_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "r0.log"
            log.write_text(self.LINES, encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
        self.assertEqual((st["parks"], st["restores"]), (1, 0))
        self.assertEqual((st["disk_owner"], st["peer_tiers"], st["peer_last_owner"]), ("r0", 1, "r1"))
        self.assertEqual((st["disk_writes"], st["disk_reads"], st["disk_evictions"]), (1, 2, 0))
        self.assertEqual((st["disk_entries"], st["disk_bytes"]), (5, 500000000))
        self.assertEqual((st["spills"], st["spill_ms"], st["spill_bytes"]), (1, 12.5, 734003200))
        self.assertEqual((st["disk_write_failed"], st["spill_failed"]), (1, 1))
        self.assertEqual(st["disk_write_failed_reasons"], {"min_tokens": 1})
        self.assertEqual(st["disk_write_causes"], {"on-park": 1})
        self.assertEqual((st["restore_ms"], st["restore_tokens"], st["restore_reused"], st["restore_read"]),
                         (800.0, 80000, 79000, 1000))                # the read of this replica's own folder
        self.assertEqual((st["peer_reads"], st["peer_restore_ms"], st["peer_restore_tokens"],
                          st["peer_restore_reused"], st["peer_restore_read"]),
                         (1, 40.0, 12000, 11800, 200))               # the read of a neighbour's snapshot
        self.assertEqual((st["peer_misses"], st["peer_fallbacks"]), (2, 1))
        self.assertEqual(st["peer_fallback_reasons"], {"ram_floor": 1})
        self.assertEqual((st["peer_refreshes"], st["peer_new_entries"], st["peer_gone_entries"],
                          st["peer_damaged_entries"]), (2, 3, 1, 0))
        self.assertEqual((st["peer_entries"], st["peer_bytes"]), (3, 734003200))
        self.assertEqual((st["lease_conflicts"], st["evictions_skipped"], st["budget_overshoot"]), (2, 1, 0))
        self.assertEqual(st["last_stats_tag"], "on-park")

    def test_the_older_5835_lines_keep_their_meaning(self):
        # before stage 3 the lines had a plain "(evicted)" / "(M reused, live)" and no peer marker, and no stats line
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "r0.log"
            log.write_text(
                "strata serve: conversation disk: wrote 80000 tokens (evicted) in 900.0 ms, 2.30 GB/s; "
                "entries=3 bytes=6000000000 evictions=0\n"
                "strata serve: conversation disk: read 80000 tokens (79000 reused, live) in 800.0 ms, 2.60 GB/s; "
                "entries=3 bytes=6000000000\n", encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
        self.assertEqual((st["disk_writes"], st["disk_reads"]), (1, 1))
        self.assertEqual((st["spills"], st["peer_reads"]), (0, 0))
        self.assertEqual((st["restore_tokens"], st["restore_reused"]), (80000, 79000))

    def test_the_last_stats_line_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "r0.log"
            log.write_text(
                "strata serve: conversation disk: stats (spill) wrote=1 read=0 evictions=0 skipped=0 "
                "lease_conflicts=0 overshoot=0 entries=1 bytes=1000 peer_read=0 peer_refresh=0 peer_entries=0 "
                "peer_bytes=0\n"
                "strata serve: conversation disk: stats (read) wrote=2 read=3 evictions=1 skipped=1 "
                "lease_conflicts=1 overshoot=1 entries=4 bytes=4000 peer_read=2 peer_refresh=1 peer_entries=3 "
                "peer_bytes=3000\n", encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
        self.assertEqual((st["disk_writes"], st["disk_reads"], st["disk_evictions"]), (2, 3, 1))
        self.assertEqual((st["evictions_skipped"], st["lease_conflicts"], st["budget_overshoot"]), (1, 1, 1))
        self.assertEqual((st["peer_reads"], st["peer_refreshes"], st["peer_entries"], st["peer_bytes"]),
                         (2, 1, 3, 3000))                          # never added up: 1+2 would be 3 peer reads

    def test_a_cut_line_is_read_once_it_is_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "r0.log"
            log.write_text("strata serve: conversation disk: wrote 8192 tokens (on-park) in 12.5 ms, 2.30 GB/s; "
                           "entries=7 bytes=734003200 evic", encoding="utf-8")
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), 0)["spills"], 0)         # no whole line yet
            with open(log, "a", encoding="utf-8") as f:
                f.write("tions=0\n")
            self.assertEqual(c.poll(str(log), 0)["spills"], 1)

    def test_a_restarted_engine_does_not_carry_its_figures_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "r0.log"
            log.write_text(self.LINES, encoding="utf-8")
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), 0)["spills"], 1)
            self.assertEqual(c.poll(str(log), log.stat().st_size)["spills"], 0)   # the engine started again
            self.assertEqual(c.poll(None, None)["spills"], 0)


if __name__ == "__main__":
    unittest.main()
