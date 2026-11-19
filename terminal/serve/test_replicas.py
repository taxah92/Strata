"""Data-parallel replicas (serve/replicas.py): the config, the router, and the real Service over HTTP with two fake
engines (the fake of test_parallel.py) behind a ReplicaEngine."""
import collections
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from types import SimpleNamespace
from serve.frontend import ChatTemplate
from serve.replicas import (BLOCK, PullRequest, ReplicaConfigError, ReplicaEngine, Router, cpu_sets, gpu_groups,
                            prefer_idle_config, pull_choose, pull_config, remove_arena, replica_cfg,
                            replica_priorities, shared_arena_plan, sweep_stale_arenas)
from serve.server import (ByteTokenizer, EngineDied, QueueWaitExpired, Service, StrataEngine, serve)
from serve.test_parallel import FAKE_BATCH


class Config(unittest.TestCase):
    def test_a_number_cuts_the_gpu_list(self):
        self.assertEqual(gpu_groups({"gpu": [0, 1, 2, 3], "replicas": 2}), [[0, 1], [2, 3]])
        self.assertEqual(gpu_groups({"gpu": "0,1,2,3", "replicas": 4}), [[0], [1], [2], [3]])
        self.assertEqual(gpu_groups({"gpu": [0, 1], "replicas": 1}), [])
        self.assertEqual(gpu_groups({"gpu": [0, 1]}), [])

    def test_named_groups(self):
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}, {"gpus": [2]}]}), [[0, 1], [2]])
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}]}), [])

    def test_mistakes_are_said(self):
        for cfg in ({"gpu": [0, 1, 2], "replicas": 2}, {"replicas": 2}, {"replicas": "2"},
                    {"replicas": [{"gpus": [0, 1]}, {"gpus": [1, 2]}]}, {"replicas": [{"gpus": []}, {"gpus": [1]}]}):
            with self.assertRaises(ReplicaConfigError, msg=cfg):
                gpu_groups(cfg)

    def test_replica_cfg(self):
        cfg = {"gpu": [0, 1, 2, 3], "args": ["--x"], "log": "/l/strata.log", "hip_ordinal": 1, "env": {"A": "1"},
               "expert_profile_save": "/p/prof.bin",
               "replicas": [{"gpus": [0, 1], "env": {"B": 2}}, {"gpus": [2, 3], "args": ["--y"]}]}
        a, b = replica_cfg(cfg, 0, [0, 1]), replica_cfg(cfg, 1, [2, 3])
        self.assertEqual((a["gpu"], b["gpu"]), ([0, 1], [2, 3]))
        self.assertEqual((a["log"], b["log"]), ("/l/strata.r0.log", "/l/strata.r1.log"))
        self.assertEqual(a["env"], {"A": "1", "B": "2"})
        self.assertEqual(b["args"], ["--x", "--y"])
        self.assertNotIn("replicas", a)
        self.assertNotIn("hip_ordinal", a)
        self.assertEqual(a["expert_profile_save"], "/p/prof.r0.bin")
        self.assertEqual(cfg["args"], ["--x"])                       # the original is untouched
        self.assertEqual(replica_cfg({"gpu": [0, 1], "args": []}, 0, [3])["gpu"], 3)

    def test_cpu_sets_off(self):
        self.assertIsNone(cpu_sets(2, enabled=False))
        self.assertIsNone(cpu_sets(1))


class Routing(unittest.TestCase):
    def test_matched_prefix_per_replica(self):
        r = Router(3)
        a = list(range(BLOCK * 4))
        r.note(0, a)
        r.note(1, a[:BLOCK * 2])
        self.assertEqual(r.matched(a + [9] * 70), [BLOCK * 4, BLOCK * 2, 0])
        self.assertEqual(r.matched([7] * 200), [0, 0, 0])
        r.forget(0)
        self.assertEqual(r.matched(a), [0, BLOCK * 2, 0])

    def test_the_index_is_bounded(self):
        r = Router(2, cap=10)
        for i in range(50):
            r.note(i % 2, [i] * (BLOCK * 2))
        self.assertLessEqual(len(r.blocks), 10)


class FakeEngine:
    def __init__(self, n=4):
        self.batch, self.up = n, True
        self.max_context = 4096
        self.info = {}
        self.last = {}

    def alive(self):
        return self.up

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        yield 7


class RouteChoice(unittest.TestCase):
    def test_least_loaded_then_affinity(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        picks = [e.route([i] * 100) for i in range(4)]             # nothing cached: spread
        self.assertEqual(sorted(picks), [0, 0, 1, 1])
        ids = list(range(BLOCK * 8))
        e.inflight = [0, 0]
        k = e.route(ids)
        e.router.note(k, ids)
        e.inflight = [0, 0]
        self.assertEqual(e.route(ids + [5] * 10), k)                # the conversation goes back
        e.inflight = [0, 0]
        e.inflight[k] = 1                                           # 512 tokens held < 3000 per running request
        self.assertNotEqual(e.route(ids + [5] * 10), k)

    def test_a_long_prefix_outweighs_load(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        ids = list(range(BLOCK * 200))                              # 12,800 tokens held on replica 0
        e.router.note(0, ids)
        e.inflight = [2, 0]
        self.assertEqual(e.route(ids + [1]), 0)
        e.inflight = [5, 0]
        self.assertEqual(e.route(ids + [1]), 1)

    def test_a_dead_replica_is_skipped(self):
        a, b = FakeEngine(), FakeEngine()
        e = ReplicaEngine([a, b])
        e.heal = lambda: None
        ids = list(range(BLOCK * 3))
        e.router.note(0, ids)
        a.up = False
        self.assertEqual(e.route(ids), 1)
        b.up = False
        self.assertEqual(e.route(ids), -1)

    def test_nonfinite_forgets_the_replica(self):
        """#879: after DONE ... nonfinite the engine holds no conversation: the router forgets that replica."""
        from serve.server import EngineNonFinite

        class NonFinite(FakeEngine):
            def generate(self, ids, max_new, sampling, cancel):
                yield 7
                raise EngineNonFinite("the engine's logits became non-finite (#879)")

        e = ReplicaEngine([NonFinite(), NonFinite()])
        other, ids = [9] * (BLOCK * 4), list(range(BLOCK * 3))
        e.router.note(0, other)
        e.router.note(1, other)
        k = 0
        e.inflight[k] += 1                                          # as route() counts it
        with mock.patch.object(e, "route", return_value=k):
            with self.assertRaises(EngineNonFinite):
                list(e.generate(ids, 10, {}, threading.Event()))
        self.assertEqual(e.router.matched(ids)[k], 0)
        self.assertEqual(e.router.matched(other)[k], 0)             # its older conversations too
        self.assertEqual(e.router.matched(other)[1 - k], len(other))
        self.assertEqual(e.inflight[k], 0)

    def test_batch_is_the_sum(self):
        self.assertEqual(ReplicaEngine([FakeEngine(4), FakeEngine(4)]).batch, 8)
        self.assertEqual(ReplicaEngine([FakeEngine(0), FakeEngine(0)]).batch, 2)

    def test_selected_replica_is_reported_for_solo_requests_too(self):
        engine = ReplicaEngine([FakeEngine(), FakeEngine()])
        engine.inflight[0] = 1
        cancel = threading.Event()
        selected = []
        cancel._strata_replica_selected = selected.append
        self.assertEqual(list(engine.generate([1, 2], 1, {}, cancel)), [7])
        self.assertEqual(selected, [1])


class GateEngine:
    """A replica without batch slots for the pull scheduler tests: the tokens it yields, the requests it served,
    whether it is alive and the figures its last request left - enough to see which replica took which request, with
    no engine and no GPU."""

    def __init__(self, name: str = "r", out=(7,)):
        self.name, self.batch, self.up = name, 0, True
        self.max_context, self.info, self.log_path = 4096, {}, None
        self.last = {}
        self.out = list(out)
        self.calls = []
        self.restarts = 0

    def alive(self):
        return self.up

    def restart(self, tries: int = 3):
        self.restarts += 1
        self.up = True

    def death_note(self):
        return "the test replica ended"

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append(list(ids))
        self.last = {"replica": self.name, "generated": len(self.calls)}
        for t in self.out:
            yield t


class PullChoice(unittest.TestCase):
    """The opt-in pull scheduler (replica_pull): a free replica takes a request at once, otherwise the request waits in
    the one shared queue; the least re-read takes a free slot, the oldest one after replica_starve_s; and every way a
    waiting request can leave (cancel, disconnect, queue_timeout, max_queue, death) is known to the queue."""

    def engine(self, **kw):
        kw.setdefault("poll_s", 0.01)               # (the tests do not wait the real 1 s / 5 s of the defaults)
        kw.setdefault("beat_s", 0.02)
        self.engines = [GateEngine("r0"), GateEngine("r1")]
        return ReplicaEngine(self.engines, pull=True, **kw)

    @staticmethod
    def request(engine, ids, seen=None):
        cancel = threading.Event()
        if seen is not None:
            cancel._strata_replica_selected = seen.append
        return engine.generate(list(ids), 16, {}, cancel), cancel

    def hold(self, engine, ids):
        """A request running on a replica: it holds that replica's slot until the test closes it."""
        g, _ = self.request(engine, ids)
        self.assertEqual(next(g), 7)
        return g

    def token(self, g, limit=200):
        """Advance a request to its token through the None heartbeats it sends while it waits."""
        for _ in range(limit):
            t = next(g)
            if t is not None:
                return t
        self.fail("the request never got a token")

    def test_a_free_replica_takes_a_new_request_without_waiting(self):
        e = self.engine()
        seen = []
        g0, _ = self.request(e, [1] * 40, seen)
        self.assertEqual(next(g0), 7)              # no None: it never waited in the queue
        self.assertEqual(dict(e.pending), {})
        self.assertEqual(e.routed[0], 1)           # the tie of two idle cards goes to the lower index
        g1, _ = self.request(e, [2] * 40, seen)
        self.assertEqual(next(g1), 7)
        self.assertEqual(seen, [0, 1])             # the idle cards are used in turn
        self.assertEqual(e.inflight, [1, 1])
        g0.close()
        g1.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_a_waiting_request_is_not_pinned_to_the_busy_replica(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)                # holds replica 0
        g1 = self.hold(e, [2] * 40)                # holds replica 1
        ids = [3] * (BLOCK * 2)
        e.router.note(0, ids)                      # replica 0 holds its prefix: pinning would wait for card 0
        seen = []
        waiting, _ = self.request(e, ids, seen)
        self.assertIsNone(next(waiting))           # both cards are busy: the request waits, a heartbeat goes out
        self.assertEqual(len(e.pending), 1)
        self.assertEqual(e.inflight, [1, 1])
        g1.close()                                 # card 1 frees first
        self.assertEqual(self.token(waiting), 7)
        self.assertEqual(seen, [1])                # it took the idle card, not its conversation's busy one
        g0.close()
        waiting.close()
        self.assertEqual((e.inflight, e.pending), ([0, 0], {}))

    def test_the_request_that_reads_least_again_goes_to_the_freed_replica(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        reuse = [7] * (BLOCK * 4)
        plain = [8] * (BLOCK * 4)
        g_plain, _ = self.request(e, plain)
        self.assertIsNone(next(g_plain))           # arrives first, reads everything on card 0
        g_reuse, _ = self.request(e, reuse)
        self.assertIsNone(next(g_reuse))
        self.assertEqual(len(e.pending), 2)
        e.router.note(0, reuse)                     # the map changes WHILE both wait: the choice must see it

        g0.close()                                 # card 0 frees: the request the card knows takes it
        self.assertEqual(self.token(g_reuse), 7)
        self.assertIsNone(next(g_plain))           # the older one waits: its prompt is not on that card
        self.assertEqual(e.pull_stats["assigned"], 3)
        self.assertEqual(e.replica_view()[0]["queue"]["reason"], "reuse")
        g1.close()
        self.assertEqual(self.token(g_plain), 7)
        g_plain.close()
        g_reuse.close()
        self.assertEqual((e.inflight, e.pending), ([0, 0], {}))

    def test_equal_cost_goes_to_the_earlier_request(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        seen = []
        g_old, _ = self.request(e, [3] * (BLOCK * 2), seen)
        self.assertIsNone(next(g_old))
        g_new, _ = self.request(e, [4] * (BLOCK * 2))
        self.assertIsNone(next(g_new))
        g0.close()
        self.assertEqual(self.token(g_old), 7)     # equal re-reads: the earlier one goes first
        self.assertEqual(seen, [0])
        self.assertIsNone(next(g_new))
        g1.close()
        self.assertEqual(self.token(g_new), 7)
        g_old.close()
        g_new.close()
        self.assertEqual((e.inflight, e.pending), ([0, 0], {}))

    def test_the_oldest_request_goes_first_after_the_starve_threshold(self):
        e = self.engine(starve_s=0.05)
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        seen = []
        g_old, _ = self.request(e, [3] * (BLOCK * 2), seen)
        self.assertIsNone(next(g_old))             # no cache anywhere
        later = [7] * (BLOCK * 2)
        e.router.note(0, later)                    # the later one would read none of its prompt again on card 0
        g_new, _ = self.request(e, later)
        self.assertIsNone(next(g_new))
        time.sleep(0.06)
        g0.close()                                 # affinity says `later`, starvation says the oldest
        self.assertEqual(self.token(g_old), 7)
        self.assertEqual(seen, [0])
        self.assertEqual(e.replica_view()[0]["queue"]["reason"], "starved")
        self.assertEqual(e.pull_stats["starved"], 1)
        g1.close()
        self.assertEqual(self.token(g_new), 7)
        g_old.close()
        g_new.close()
        self.assertEqual((e.inflight, e.pending), ([0, 0], {}))

    def test_two_frees_assign_two_requests_and_never_one_slot_twice(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        seen = []
        ga, _ = self.request(e, [3] * (BLOCK * 2), seen)
        self.assertIsNone(next(ga))
        gb, _ = self.request(e, [4] * (BLOCK * 2), seen)
        self.assertIsNone(next(gb))
        barrier = threading.Barrier(2)

        def free(g):
            barrier.wait()
            g.close()
        threads = [threading.Thread(target=free, args=(g,)) for g in (g0, g1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(dict(e.pending), {})      # both waiters were given a card, one each
        self.assertEqual(e.inflight, [1, 1])
        self.assertEqual(e.pull_stats["assigned"], 4)
        self.assertEqual(self.token(ga), 7)
        self.assertEqual(self.token(gb), 7)
        self.assertEqual(set(seen), {0, 1})
        self.assertEqual(sorted(len(eng.calls) for eng in self.engines), [2, 2])   # nothing was served twice
        ga.close()
        gb.close()
        self.assertEqual((e.inflight, e.routed), ([0, 0], [2, 2]))

    def test_a_slot_is_released_when_the_request_ends_or_is_closed(self):
        e = self.engine()
        g = self.hold(e, [1] * 40)                 # replica 0
        self.assertEqual(e.inflight, [1, 0])
        g.close()                                  # closed in the middle of its generation
        self.assertEqual((e.inflight, e.pending), ([0, 0], {}))
        seen = []
        again, _ = self.request(e, [2] * 40, seen)
        self.assertEqual(next(again), 7)           # the freed card takes the next request at once
        self.assertEqual(seen, [0])
        again.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_a_slot_given_to_a_client_that_left_is_given_back(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        waiting, _ = self.request(e, [3] * (BLOCK * 2))
        self.assertIsNone(next(waiting))           # suspended at its heartbeat, still in the queue
        g0.close()                                 # the pump gives the waiting request replica 0 ...
        self.assertEqual((e.inflight, dict(e.pending)), ([1, 1], {}))
        waiting.close()                            # ... but the client left before using it: the slot comes back
        self.assertEqual((e.inflight, e.pull_stats["closed"]), ([0, 1], 1))
        g1.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_generation_happens_outside_the_scheduler_lock_and_inside_run_locks(self):
        e = self.engine()
        g = self.hold(e, [1] * 40)                 # replica 0 is running this request
        self.assertTrue(e.run_locks[0].locked())   # the control line is protected, as before (run_locks[k] stays)
        self.assertTrue(e.lock.acquire(timeout=1.0))   # but the scheduler lock is free while it generates
        e.lock.release()
        g.close()
        self.assertFalse(e.run_locks[0].locked())
        self.assertEqual(e.inflight, [0, 0])

    def test_a_dead_replica_is_never_chosen(self):
        e = self.engine()
        e.heal = lambda: None                      # (this test: the dead card stays dead)
        g0 = self.hold(e, [1] * 40)                # card 0 is busy, card 1 is dead
        self.engines[1].up = False
        seen = []
        waiting, _ = self.request(e, [3] * (BLOCK * 2), seen)
        self.assertIsNone(next(waiting))
        self.assertEqual(len(e.pending), 1)
        g0.close()                                 # the living card frees; the dead one is not offered the request
        self.assertEqual(self.token(waiting), 7)
        self.assertEqual(seen, [0])
        self.assertEqual(self.engines[1].calls, [])
        waiting.close()
        self.assertEqual((e.inflight, e.pending), ([0, 0], {}))

    def test_all_replicas_dead_end_the_pending_request(self):
        e = self.engine()
        e.heal = lambda: None
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        waiting, _ = self.request(e, [3] * (BLOCK * 2))
        self.assertIsNone(next(waiting))
        self.engines[0].up = False
        self.engines[1].up = False
        with self.assertRaises(EngineDied):
            next(waiting)
        self.assertEqual((dict(e.pending), e.pull_stats["no_replica"]), ({}, 1))   # no eternal pending request
        g0.close()
        g1.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_a_healed_replica_picks_up_a_waiting_request(self):
        e = self.engine()
        e.heal = lambda: None                      # (the heal is imitated: the card comes back by itself)
        g0 = self.hold(e, [1] * 40)
        self.engines[1].up = False
        seen = []
        waiting, _ = self.request(e, [3] * (BLOCK * 2), seen)
        self.assertIsNone(next(waiting))
        self.engines[1].up = True
        self.assertEqual(self.token(waiting), 7)   # the poll tick sees the living card and gives it the request
        self.assertEqual(seen, [1])
        self.assertEqual((dict(e.pending), e.inflight), ({}, [1, 1]))
        waiting.close()
        g0.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_heal_starting_a_replica_picks_up_the_waiting_request(self):
        e = self.engine()                          # real heal(): GateEngine.restart() brings the card back
        g0 = self.hold(e, [1] * 40)                # card 0 serves and stays busy
        self.engines[1].up = False                 # card 1 is down
        self.engines[0].calls, self.engines[1].calls = [], []
        seen = []
        waiting, _ = self.request(e, [3] * (BLOCK * 2), seen)
        self.assertEqual(self.token(waiting), 7)   # the wait loop called heal(), card 1 came back and took it
        self.assertEqual(seen, [1])
        self.assertTrue(self.engines[1].alive())
        self.assertEqual(self.engines[1].restarts, 1)
        self.assertEqual(dict(e.pending), {})
        waiting.close()
        g0.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_a_replica_that_does_not_come_back_ends_the_pending_request(self):
        e = self.engine()
        e.engines[0].restart = mock.Mock(side_effect=RuntimeError("this replica will not start"))
        e.engines[0].up = False
        g1 = self.hold(e, [2] * 40)                # only card 1 is alive, so it serves
        waiting, _ = self.request(e, [3] * (BLOCK * 2))
        self.assertIsNone(next(waiting))
        e.engines[1].up = False                    # now every replica is dead and one heal attempt fails
        with self.assertRaises(EngineDied):
            for _ in range(300):
                next(waiting)
        self.assertEqual((dict(e.pending), e.pull_stats["no_replica"]), ({}, 1))
        g1.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_cancel_removes_only_its_own_pending_request(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        ids_a, ids_b = [3] * (BLOCK * 2), [4] * (BLOCK * 2)
        ga, cancel_a = self.request(e, ids_a)
        self.assertIsNone(next(ga))
        seen = []
        gb, cancel_b = self.request(e, ids_b, seen)
        self.assertIsNone(next(gb))
        self.assertEqual(len(e.pending), 2)
        cancel_a.set()
        self.assertEqual(list(ga), [])             # a client that left before assignment: nothing generated
        self.assertEqual([p.ids for p in e.pending.values()], [ids_b])   # its own entry only
        self.assertEqual((e.inflight, e.pull_stats["cancelled"]), ([1, 1], 1))   # no slot leaked, none taken
        g0.close()
        self.assertEqual(self.token(gb), 7)        # the other one still gets its card
        self.assertEqual(seen, [0])
        self.assertEqual(e.engines[0].calls[-1], ids_b)
        g1.close()
        gb.close()
        self.assertEqual((e.inflight, e.pending), ([0, 0], {}))

    def test_a_client_that_leaves_removes_its_waiting_request(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        ids = [3] * (BLOCK * 2)
        waiting, _ = self.request(e, ids)
        self.assertIsNone(next(waiting))
        self.assertEqual(len(e.pending), 1)
        waiting.close()                            # GeneratorExit: the server closes a request of a client that left
        self.assertEqual(dict(e.pending), {})
        self.assertEqual((e.inflight, e.pull_stats["closed"]), ([1, 1], 1))   # its own entry only, no slot leaked
        self.assertNotIn(ids, self.engines[0].calls + self.engines[1].calls)
        g0.close()
        g1.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_queue_deadline_while_pending_raises_queue_wait_expired(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        ga, cancel_a = self.request(e, [3] * (BLOCK * 2))
        cancel_a._strata_queue_deadline = time.monotonic() + 0.1
        self.assertIsNone(next(ga))
        gb, _ = self.request(e, [4] * (BLOCK * 2))
        self.assertIsNone(next(gb))
        with self.assertRaises(QueueWaitExpired):
            self.token(ga)
        self.assertEqual([p.ids for p in e.pending.values()], [[4] * (BLOCK * 2)])   # only its own entry went
        self.assertEqual((e.inflight, e.pull_stats["timeouts"]), ([1, 1], 1))
        g0.close()
        g1.close()
        gb.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_queue_timeout_from_the_config_removes_the_own_pending(self):
        e = self.engine(queue_timeout_s=0.05)
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        ga, _ = self.request(e, [3] * (BLOCK * 2))
        self.assertIsNone(next(ga))
        with self.assertRaises(QueueWaitExpired):
            self.token(ga)
        self.assertEqual((dict(e.pending), e.inflight, e.pull_stats["timeouts"]), ({}, [1, 1], 1))
        g0.close()
        g1.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_max_queue_is_a_cap_and_a_full_queue_is_rejected(self):
        e = self.engine(max_queue=1)
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        ga, _ = self.request(e, [3] * (BLOCK * 2))
        self.assertIsNone(next(ga))                # the one place of the queue
        self.assertEqual(len(e.pending), 1)
        gb, _ = self.request(e, [4] * (BLOCK * 2))
        with self.assertRaises(QueueWaitExpired) as caught:
            next(gb)
        self.assertIn("replica_max_queue=1", str(caught.exception))
        self.assertEqual((e.pull_stats["rejected"], len(e.pending), e.inflight), (1, 1, [1, 1]))
        g0.close()
        self.assertEqual(self.token(ga), 7)        # the queued one still gets its card
        g1.close()
        ga.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_max_queue_zero_means_no_waiting_but_a_free_card_still_serves(self):
        e = self.engine(max_queue=0)
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        gb, _ = self.request(e, [4] * (BLOCK * 2))
        with self.assertRaises(QueueWaitExpired):
            next(gb)                               # no waiting allowed
        self.assertEqual((dict(e.pending), e.inflight), ({}, [1, 1]))
        g1.close()
        g = self.hold(e, [4] * (BLOCK * 2))        # a free card takes a request as before, without the queue
        self.assertEqual(dict(e.pending), {})
        g.close()
        g0.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_heartbeats_are_yielded_while_pending(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        waiting, _ = self.request(e, [3] * (BLOCK * 2))
        self.assertIsNone(next(waiting))           # the keep-alive the server turns into a "ping"
        self.assertIsNone(next(waiting))           # and it keeps coming while the request waits
        self.assertEqual(len(e.pending), 1)
        g0.close()
        self.assertEqual(self.token(waiting), 7)
        waiting.close()
        g1.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_last_produced_and_router_note_belong_to_the_assigned_replica(self):
        e = self.engine()
        a = [1] * (BLOCK * 2 - 1)                  # with the produced token it is exactly two blocks
        seen = []
        g, _ = self.request(e, a, seen)
        self.assertEqual(next(g), 7)
        g.close()
        self.assertEqual(seen, [0])
        self.assertEqual(e.last["replica"], "r0")  # this thread's figures are the replica that served it
        self.assertEqual(e.router.matched(a + [7]), [BLOCK * 2, 0])   # note(ids + produced), on replica 0 only
        hold = self.hold(e, [9] * 40)              # keeps replica 0 busy, so the next request must take replica 1
        b = [2] * BLOCK
        seen = []
        g, _ = self.request(e, b, seen)
        self.assertEqual(next(g), 7)
        g.close()
        self.assertEqual(seen, [1])
        self.assertEqual(e.last["replica"], "r1")  # this thread's figures followed the request to its replica
        self.assertEqual(e.router.matched(b + [7]), [0, BLOCK])
        hold.close()
        self.assertEqual(e.router.matched(a + [7]), [BLOCK * 2, 0])   # the first request's note stayed on its card
        self.assertEqual((e.routed, e.inflight), ([2, 1], [0, 0]))

    def test_the_selected_replica_is_reported_at_assignment(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        seen = []
        waiting, cancel = self.request(e, [3] * (BLOCK * 2), seen)
        cancel._strata_queue_deadline = time.monotonic() + 60
        self.assertIsNone(next(waiting))
        self.assertEqual(seen, [])                 # nothing is reported while it waits
        self.assertIsNone(getattr(cancel, "_strata_replica", None))
        g1.close()
        self.assertEqual(self.token(waiting), 7)
        self.assertEqual(seen, [1])
        self.assertEqual(cancel._strata_replica, 1)
        self.assertIsNone(cancel._strata_queue_deadline)   # the queue is over: the deadline is cleared
        waiting.close()
        g0.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_pull_view_is_global_and_replica_view_per_replica(self):
        e = self.engine()
        g0 = self.hold(e, [1] * 40)
        g1 = self.hold(e, [2] * 40)
        waiting, _ = self.request(e, [3] * (BLOCK * 2))
        self.assertIsNone(next(waiting))
        q = e.pull_view()
        self.assertEqual((q["enabled"], q["pending"]), (True, 1))
        self.assertIsNotNone(q["oldest_pending_s"])
        self.assertGreater(q["oldest_pending_s"], 0.0)
        rows = e.replica_view()
        self.assertEqual([r["replica"] for r in rows], [0, 1])
        self.assertEqual([r["running"] for r in rows], [1, 1])
        self.assertEqual([r["routed"] for r in rows], [1, 1])
        self.assertEqual(rows[0]["queue"]["depth"], 1)          # (contract §7: the queue is visible in the view)
        self.assertEqual(rows[1]["queue"], rows[0]["queue"])
        g0.close()
        self.assertEqual(self.token(waiting), 7)
        self.assertEqual(e.pull_stats["assigned"], 3)
        waiting.close()
        g1.close()
        self.assertEqual(e.inflight, [0, 0])

    def test_pull_choose_is_the_order_without_threads(self):
        def matched(ids):
            return ([BLOCK * 2, 0] if ids[0] == 1 else [0, BLOCK])
        a, b = PullRequest(), PullRequest()
        a.seq, a.ids, a.arrive = 1, [2] * (BLOCK * 2), 100.0
        b.seq, b.ids, b.arrive = 2, [1] * (BLOCK * 2), 101.0
        pending = collections.OrderedDict((p.seq, p) for p in (a, b))
        self.assertEqual(pull_choose(pending, [], 102.0, 60.0, matched), None)          # nothing free
        self.assertEqual(pull_choose(collections.OrderedDict(), [0], 102.0, 60.0, matched), None)
        got = pull_choose(pending, [0], 102.0, 60.0, matched)                          # b reads none of it again
        self.assertEqual((got[0].seq, got[1], got[2]), (2, 0, False))
        got = pull_choose(pending, [0], 160.5, 60.0, matched)                          # a is now the oldest
        self.assertEqual((got[0].seq, got[1], got[2]), (1, 0, True))
        got = pull_choose(pending, [0, 1], 102.0, 60.0, matched)                       # free cards, one request
        self.assertEqual((got[0].seq, got[1], got[2]), (2, 0, False))
        got = pull_choose(pending, [0], 102.0, 0.0, matched)                           # starve_s = 0: the oldest.
        self.assertEqual((got[0].seq, got[1], got[2]), (1, 0, True))
        self.assertEqual((a.matched, b.matched), ([0, BLOCK], [BLOCK * 2, 0]))         # the last round is kept

    def test_batch_slots_are_refused_at_start(self):
        with self.assertRaises(ReplicaConfigError) as caught:
            ReplicaEngine([FakeEngine(4), FakeEngine(4)], pull=True)
        self.assertIn("replica_pull", str(caught.exception))
        self.assertIn("batch", str(caught.exception))
        self.assertTrue(ReplicaEngine([FakeEngine(0), FakeEngine(0)], pull=True).pull_enabled)

    def test_the_default_engine_keeps_the_old_path(self):
        e = ReplicaEngine([FakeEngine(0), FakeEngine(0)])
        self.assertFalse(e.pull_enabled)
        self.assertEqual(dict(e.pending), {})
        self.assertFalse(e.pull_view()["enabled"])
        self.assertEqual(set(e.replica_view()[0]), {"replica", "alive", "running", "routed", "batch"})
        self.assertEqual([e.route([1] * 100) for _ in range(4)], [0, 1, 0, 1])   # the old least-loaded order
        e.inflight = [0, 0]
        with mock.patch.object(e, "route", wraps=e.route) as routed:
            g = e.generate([1] * 100, 10, {}, threading.Event())
            self.assertEqual(next(g), 7)
            g.close()
        self.assertEqual(routed.call_count, 1)     # the old generate() still goes through route()


class PullConfig(unittest.TestCase):
    def test_off_by_default(self):
        pc = pull_config({})
        self.assertEqual((pc.enabled, pc.starve_s), (False, 60.0))
        self.assertEqual(pull_config({"replica_pull": False}).enabled, False)

    def test_the_keys_are_validated(self):
        for cfg in ({"replica_pull": 1}, {"replica_pull": "yes"}, {"replica_pull": None},
                    {"replica_pull": True, "replicas": 2, "gpu": [0, 1], "replica_starve_s": -1},
                    {"replica_pull": True, "replicas": 2, "gpu": [0, 1], "replica_starve_s": True},
                    {"replica_pull": True, "replicas": 2, "gpu": [0, 1], "replica_starve_s": "60"},
                    {"replica_pull": True, "replicas": 2, "gpu": [0, 1], "replica_queue_timeout_s": -1},
                    {"replica_pull": True, "replicas": 2, "gpu": [0, 1], "replica_max_queue": -1}):
            with self.assertRaises(ReplicaConfigError, msg=cfg):
                pull_config(cfg)
        pc = pull_config({"replica_pull": True, "replicas": 2, "gpu": [0, 1], "replica_starve_s": 0,
                          "replica_queue_timeout_s": 30, "replica_max_queue": 2})
        self.assertEqual((pc.enabled, pc.starve_s, pc.queue_timeout_s, pc.max_queue), (True, 0.0, 30.0, 2))
        with self.assertRaises(ReplicaConfigError):
            ReplicaEngine([GateEngine(), GateEngine()], pull=1)

    def test_pull_needs_replicas_and_no_split(self):
        for cfg in ({"replica_pull": True},
                    {"replica_pull": True, "replicas": 1, "gpu": [0, 1]},
                    {"replica_pull": True, "replicas": 2, "gpu": [0, 1], "split_serve": {"prefill_gpu": 0}}):
            with self.assertRaises(ReplicaConfigError, msg=cfg):
                pull_config(cfg)
        self.assertTrue(pull_config({"replica_pull": True, "replicas": 2, "gpu": [0, 1]}).enabled)

    def test_pull_refuses_batch_slots(self):
        for cfg in ({"replica_pull": True, "replicas": 2, "gpu": [0, 1], "parallel": 2},
                    {"replica_pull": True, "replicas": 2, "gpu": [0, 1], "args": ["--batch", "2"]},
                    {"replica_pull": True, "replicas": 2, "gpu": [0, 1], "args": ["--slots=4"]}):
            with self.assertRaises(ReplicaConfigError, msg=cfg):
                pull_config(cfg)
        pull_config({"replica_pull": True, "replicas": 2, "gpu": [0, 1], "parallel": 1})

    def test_the_engine_starts_from_the_run_config(self):
        pc = pull_config({"replica_pull": True, "replicas": 2, "gpu": [0, 1], "replica_starve_s": 5,
                          "replica_queue_timeout_s": 1.5, "replica_max_queue": 2})
        e = ReplicaEngine([GateEngine(), GateEngine()], pull=pc.enabled, starve_s=pc.starve_s,
                          queue_timeout_s=pc.queue_timeout_s, max_queue=pc.max_queue)
        self.assertTrue(e.pull_enabled)
        self.assertEqual((e.pull_starve_s, e.queue_timeout_s, e.max_queue), (5.0, 1.5, 2))


class PullService(unittest.TestCase):
    """The pull scheduler behind the real Service (no HTTP): its keep-alives, its queue timeout with the existing
    429 contract, the admission limit, the terminal row, a death while waiting, and the queue in /metrics' view."""

    def setUp(self):
        self.engine = ReplicaEngine([GateEngine("r0"), GateEngine("r1")], pull=True, poll_s=0.01, beat_s=0.02)
        self.tok = ByteTokenizer()
        self.svc = Service(self.engine, self.tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))

    def hold(self, name="x"):
        g = self.engine.generate(list(name.encode()), 16, {}, threading.Event())
        self.assertEqual(next(g), 7)
        return g

    def run_request(self):
        cancel = threading.Event()
        return self.svc.run(self.tok.encode("hi"), False, None, 8, {}, cancel), cancel

    def test_service_keepalives_while_a_request_waits_for_a_replica(self):
        g0 = self.hold()
        g1 = self.hold()
        run, cancel = self.run_request()
        self.assertEqual(next(run), ("ping", None))            # the Service turned the None into its keep-alive
        row = self.svc.terminal_requests[cancel._strata_request_id]
        self.assertEqual((row["state"], row.get("replica")), ("waiting", None))
        self.assertIsNone(row.get("first_engine_token_s"))
        g0.close()
        kinds = []
        for kind, _ in run:
            kinds.append(kind)
            if kind == "done":
                break
        self.assertIn("event", kinds)                          # the answer came once a replica was free
        g1.close()
        self.assertEqual(self.engine.inflight, [0, 0])

    def test_service_reports_queue_wait_timeout_as_the_existing_error(self):
        self.svc.queue_timeout_s = 0.1
        g0 = self.hold()
        g1 = self.hold()
        run, cancel = self.run_request()
        self.assertEqual(next(run), ("ping", None))
        with self.assertRaises(QueueWaitExpired):
            list(run)
        self.assertEqual(self.svc.queue_timeouts, 1)
        self.assertEqual(dict(self.engine.pending), {})
        self.assertEqual(self.engine.inflight, [1, 1])
        g0.close()
        g1.close()
        self.assertEqual(self.engine.inflight, [0, 0])

    def test_max_queue_is_still_the_admission_limit(self):
        self.svc.max_queue = 0
        self.assertEqual(self.svc.admission_limit(), 2)        # max(1, engine.batch) + max_queue, as before
        self.assertTrue(self.svc.reserve_inference())
        self.assertTrue(self.svc.reserve_inference())
        self.assertFalse(self.svc.reserve_inference())
        self.assertEqual(self.svc.admission_rejected, 1)
        self.svc.release_inference()
        self.svc.release_inference()
        self.assertEqual(self.svc.admission_reserved, 0)

    def test_the_terminal_row_names_the_replica_only_after_assignment(self):
        g0 = self.hold()
        g1 = self.hold()
        run, cancel = self.run_request()
        self.assertEqual(next(run), ("ping", None))
        request_id = cancel._strata_request_id
        self.assertNotIn("replica", self.svc.terminal_requests[request_id])
        g1.close()
        for kind, _ in run:
            if kind != "ping":
                break
        row = self.svc.terminal_requests[request_id]
        self.assertEqual((row["state"], row["replica"]), ("reading", 1))
        for _ in run:
            pass
        g0.close()
        self.assertEqual(self.engine.inflight, [0, 0])

    def test_a_replica_death_while_waiting_ends_with_an_engine_died(self):
        g0 = self.hold()
        g1 = self.hold()
        run, cancel = self.run_request()
        self.assertEqual(next(run), ("ping", None))
        self.engine.engines[0].up = False
        self.engine.engines[1].up = False
        with self.assertRaises(EngineDied):
            list(run)
        self.assertEqual((dict(self.engine.pending), self.engine.pull_stats["no_replica"]), ({}, 1))
        g0.close()
        g1.close()
        self.assertEqual(self.engine.inflight, [0, 0])

    def test_the_queue_reaches_metrics_through_the_existing_view(self):
        g0 = self.hold()
        g1 = self.hold()
        m = self.svc.metrics()
        rows = m["live"]["replicas"]
        self.assertEqual([r["replica"] for r in rows], [0, 1])
        self.assertEqual(rows[0]["queue"]["depth"], 0)         # (no new field is needed on the server's side)
        self.assertIsNone(rows[0]["queue"]["oldest_s"])
        self.assertEqual(m["live"]["queued"], 0)
        self.assertEqual(self.svc.v1_status()["concurrency"]["serving"], 2)   # the engine still counts as parallel
        g0.close()
        g1.close()


class ReusedRouting(unittest.TestCase):
    """live.reused is read from the replica that serves the request, never from a finished request's stale value."""

    def test_the_request_replica_is_read(self):
        a, b = FakeEngine(), FakeEngine()
        a.reused, b.reused = 0, 1234
        e = ReplicaEngine([a, b])
        e.tl.cur = b
        self.assertEqual(e.reused, 1234)
        e.tl.cur = a
        self.assertEqual(e.reused, 0)                               # its own request holds no prefix: 0, not b's

    def test_the_status_thread_reads_the_replica_with_a_request_in_flight(self):
        a, b = FakeEngine(), FakeEngine()
        a.reused, b.reused = 1234, 500                              # a: finished earlier (stale); b: reading now
        e = ReplicaEngine([a, b])
        e.inflight = [0, 1]
        self.assertEqual(e.reused, 500)
        b.reused = 0                                                # b's request resumes nothing
        self.assertEqual(e.reused, 0)
        e.inflight = [0, 0]
        self.assertEqual(e.reused, 0)

    def test_progress_is_read_from_the_replica_in_flight(self):
        a, b = FakeEngine(), FakeEngine()
        a.progress = (12000, 12000)                                 # a: finished earlier (stale)
        b.progress = None                                           # b: reading now, no chunk yet
        e = ReplicaEngine([a, b])
        e.inflight = [0, 1]
        self.assertIsNone(e.progress)                               # strict 0 before the first confirmed chunk
        b.progress = (10, 100)
        self.assertEqual(e.progress, (10, 100))
        e.inflight = [0, 0]
        self.assertIsNone(e.progress)

    def test_prefill_rate_is_read_from_the_replica_in_flight(self):
        a, b = FakeEngine(), FakeEngine()
        a.prefill_tok_s_mean = 4000.0                               # a: finished earlier (stale)
        b.prefill_tok_s_mean = None
        e = ReplicaEngine([a, b])
        e.inflight = [0, 1]
        self.assertIsNone(e.prefill_tok_s_mean)
        b.prefill_tok_s_mean = 123.4
        self.assertEqual(e.prefill_tok_s_mean, 123.4)
        e.inflight = [0, 0]
        self.assertIsNone(e.prefill_tok_s_mean)


class ReplicaService(unittest.TestCase):
    def start(self, slots=4, n=2, pull=False):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        real = server.subprocess.Popen
        self.logs = [Path(self.tmp.name) / f"r{i}.log" for i in range(n)]
        engines = []
        for i in range(n):
            extra = (["--batch", str(slots)] if slots else []) + ["--log", str(self.logs[i])]
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engines.append(StrataEngine("strata", extra))
        self.engine = ReplicaEngine(engines, **(dict(pull=True, poll_s=0.01, beat_s=0.02) if pull else {}))
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def post(self, messages, max_tokens=64):
        body = {"messages": messages, "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def chat(self, text, max_tokens=64):
        return self.post([{"role": "user", "content": text}], max_tokens)

    def served(self):
        return [sum(1 for l in p.read_text().splitlines() if l.split()[0] in ("GEN", "BGEN")) if p.exists() else 0
                for p in self.logs]

    def burst(self, prompts):
        out = [None] * len(prompts)

        def one(i):
            out[i] = self.chat(prompts[i])
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return out

    def test_requests_spread_over_both_and_answers_are_the_solo_answer(self):
        self.start()
        solo = self.chat("hello there")
        before = self.served()
        out = self.burst([f"request number {i} " + "x" * 300 for i in range(6)])
        got = [b - a for a, b in zip(before, self.served())]
        self.assertTrue(all(g >= 2 for g in got), got)
        self.assertEqual(set(out), {solo})

    def test_metrics_and_status_see_all_slots(self):
        self.start()
        self.assertEqual(self.svc.v1_status()["concurrency"]["serving"], 8)
        m = self.svc.metrics()
        self.assertEqual(len(m["live"]["slots"]), 8)
        self.assertEqual([r["replica"] for r in m["live"]["replicas"]], [0, 1])
        self.assertEqual(m["engine"]["replicas"], 2)

    def test_terminal_telemetry_correlates_second_replica_without_global_sample_claim(self):
        self.start()
        self.svc.replica_gpu_groups = [[0, 1], [2, 3]]
        second = self.engine.engines[1]
        request_id = "abcdef012345"
        second.slot_live[1] = {"request_id": request_id, "state": "reading", "prompt_tokens": 100,
                               "prompt_read": 12, "generated": 0, "started": time.time(), "first_token": None}
        first = self.engine.engines[0]
        first.terminal_batch_sample = {"windows": 2, "processed_rows": 8, "active_rows": 4,
                                       "elapsed_ms": 1000, "observed_monotonic": time.monotonic()}
        second.terminal_batch_sample = {"windows": 2, "processed_rows": 6, "active_rows": 4,
                                        "elapsed_ms": 1000, "observed_monotonic": time.monotonic()}
        first.terminal_stage_sample = {"elapsed_us": 1_000_000, "in_flight_us": [800_000, 600_000],
                                       "observed_monotonic": time.monotonic()}
        second.terminal_stage_sample = {"elapsed_us": 1_000_000, "in_flight_us": [700_000, 500_000],
                                        "observed_monotonic": time.monotonic()}
        self.svc.backend = "cuda"
        for child, cards in ((first, [0, 1]), (second, [2, 3])):
            child.spawn = (*child.spawn[:4], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                              "CUDA_VISIBLE_DEVICES": ",".join(map(str, cards))},
                           *child.spawn[5:])
            child.terminal_stages = [
                {"stage": 0, "layer_start": 0, "layer_end": 19, "engine_ordinal": 0},
                {"stage": 1, "layer_start": 20, "layer_end": 39, "engine_ordinal": 1}]
        with self.svc.status_lock:
            self.svc.terminal_requests[request_id] = {"id": request_id, "state": "waiting",
                                                       "queued_at": time.time(), "admitted_at": time.time(),
                                                       "prompt_tokens": 100}
        with urllib.request.urlopen(self.base + "/v1/telemetry", timeout=10) as response:
            snapshot = json.load(response)
        slot = snapshot["work"]["slots"][5]
        self.assertEqual((slot["slot"], slot["replica"], slot["group"], slot["request_id"]),
                         (5, 1, 1, request_id))
        request = snapshot["work"]["requests"][0]
        self.assertEqual((request["slot"], request["replica"], request["group"], request["state"]),
                         (5, 1, 1, "reading"))
        self.assertEqual(snapshot["work"]["groups"], 2)
        self.assertIsNone(snapshot["work"]["effective_rows_window"])
        self.assertEqual([(row["replica"], row["batch_sample_rows"], row["batch_sample_active_rows"])
                          for row in snapshot["work"]["replica_batches"]], [(0, 8, 4), (1, 6, 4)])
        self.assertEqual([(row["replica"], row["stage_in_flight_us"])
                          for row in snapshot["work"]["replica_stage_samples"]],
                         [(0, [800_000, 600_000]), (1, [700_000, 500_000])])
        self.assertNotIn("stage_in_flight_us", snapshot["work"])
        self.assertEqual(snapshot["topology"]["stages"], [])
        self.assertEqual(snapshot["topology"]["replica_stages"][1]["stages"][1]["device_index"], 3)
        self.assertEqual(snapshot["topology"]["replica_devices"],
                         [{"replica": 0, "devices": [0, 1]}, {"replica": 1, "devices": [2, 3]}])
        second.spawn = (*second.spawn[:4], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                            "CUDA_VISIBLE_DEVICES": "9,3"}, *second.spawn[5:])
        with urllib.request.urlopen(self.base + "/v1/telemetry", timeout=10) as response:
            mismatch = json.load(response)
        self.assertIsNone(mismatch["topology"]["replica_stages"][1]["stages"][0]["device_index"])
        self.assertEqual(mismatch["topology"]["replica_stages"][1]["stages"][1]["device_index"], 3)

    def test_engines_without_slots_run_one_each_at_once(self):
        self.start(slots=0)
        out = self.burst(["a" * 40, "b" * 40])
        self.assertEqual(self.served(), [1, 1])
        self.assertEqual(len(set(out)), 1)

    def test_a_burst_over_replica_pull_uses_both_replicas(self):
        self.start(slots=0, pull=True)                     # two engines, one request each at a time, one shared queue
        solo = self.chat("hello there")
        before = self.served()
        out = self.burst([f"request number {i} " + "x" * 300 for i in range(4)])
        got = [b - a for a, b in zip(before, self.served())]
        self.assertTrue(all(g >= 1 for g in got), got)     # the queued requests waited for a card, then ran
        self.assertEqual(sum(got), 4)                      # every request ran exactly once
        self.assertEqual(set(out), {solo})

    def test_a_conversation_goes_back_to_its_replica(self):
        self.start()
        first = "begin " + "lorem ipsum " * 40
        reply = self.chat(first)
        k = self.served().index(1)
        self.post([{"role": "user", "content": first}, {"role": "assistant", "content": reply},
                   {"role": "user", "content": "and then?"}], 16)
        self.assertEqual(self.served()[k], 2)

    def test_a_replica_that_died_is_skipped(self):
        self.start()
        self.engine.heal = lambda: None
        self.engine.engines[0].proc.kill()
        self.engine.engines[0].proc.wait()
        self.assertTrue(self.engine.alive())
        for i in range(3):
            self.assertIn("ok, done.", self.chat(f"after the loss {i}"))
        self.assertEqual(self.served()[1], 3)


class SharedArena(unittest.TestCase):
    """The opt-in shared expert arena of the replicas (shared_arena_plan): when it is used, when it is not, what a stale
    file means."""
    CFG = {"replica_shared_arena": True, "args": ["--pack", "/p"], "gpu": [0, 1]}

    def plan(self, cfg=None, n=2, tmp=None, **kw):
        return shared_arena_plan(cfg or self.CFG, n, shm=tmp, platform=kw.pop("platform", "linux"),
                                 environ=kw.pop("environ", {}), free_bytes=kw.pop("free_bytes", 100 << 30),
                                 use_memfd=kw.pop("use_memfd", False), **kw)

    def test_off_by_default_and_for_one_engine(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.plan({"args": []}, tmp=d), (None, ""))
            self.assertEqual(self.plan(n=1, tmp=d), (None, ""))
            self.assertEqual(self.plan({"args": []}, tmp=d, environ={"STRATA_REPLICA_SHARED_ARENA": "0"}), (None, ""))

    def test_on_by_config_or_environment(self):
        with tempfile.TemporaryDirectory() as d:
            path, note = self.plan(tmp=d)
            self.assertTrue(path and path.startswith(d) and path.endswith(".bin"), path)
            self.assertIn("once", note)
            path2, _ = self.plan({"args": []}, tmp=d, environ={"STRATA_REPLICA_SHARED_ARENA": "1"})
            self.assertTrue(path2)
            self.assertEqual(path, path2)      # (one server: one file)

    def test_windows_gets_a_named_section(self):
        with tempfile.TemporaryDirectory() as d:
            path, note = self.plan(tmp=d, platform="win32")
            self.assertTrue(path and path.startswith("win:Local") and "strata-replicas-" in path, path)
            self.assertIn("Windows section", note)
            remove_arena(path)                 # (nothing to remove: harmless)

    def test_only_where_there_is_an_arena_to_share(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self.plan(tmp=d, platform="darwin")[0])
            for flag in ("--mmap-experts", "--resident-experts", "--resident-budget-gib", "--shared-expert-arena"):
                self.assertIsNone(self.plan({**self.CFG, "args": ["--pack", "/p", flag]}, tmp=d)[0], flag)
            self.assertIsNone(self.plan({**self.CFG, "replicas": [{"gpus": [0], "args": ["--mmap-experts"]}, {"gpus": [1]}]}, tmp=d)[0])

    def test_no_room_or_no_shm_says_so_and_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            path, note = self.plan(tmp=d, free_bytes=2 << 30)
            self.assertIsNone(path)
            self.assertIn("2.0 GiB free", note)
            path, note = self.plan(tmp=d + "/nope")
            self.assertIsNone(path)
            self.assertIn("not usable", note)

    def test_a_stale_file_goes_a_live_one_stays(self):
        with tempfile.TemporaryDirectory() as d:
            uid = __import__("os").getuid() if hasattr(__import__("os"), "getuid") else 0
            dead = Path(d) / f"strata-replicas-{uid}-2999999.bin"
            live = Path(d) / f"strata-replicas-{uid}-{__import__('os').getpid()}.bin"
            other = Path(d) / f"strata-replicas-{uid + 1}-2999999.bin"
            junk = Path(d) / "strata-replicas-x.bin"
            for f in (dead, live, other, junk):
                f.write_bytes(b"x")
            with mock.patch("serve.replicas._pid_alive", side_effect=lambda pid: pid != 2999999):
                gone = sweep_stale_arenas(d, uid)
            self.assertEqual(gone, [dead.name])
            self.assertFalse(dead.exists())
            self.assertTrue(live.exists() and other.exists() and junk.exists())

    @unittest.skipUnless(hasattr(__import__("os"), "memfd_create") and sys.platform.startswith("linux"), "Linux memfd")
    def test_a_memfd_is_not_limited_by_dev_shm_and_is_closed_at_the_end(self):
        import os
        path, note = shared_arena_plan(self.CFG, 2, platform="linux", environ={}, free_bytes=100 << 30, use_memfd=True)
        self.assertTrue(path and path.startswith(f"/proc/{os.getpid()}/fd/"), path)
        fd = int(path.rsplit("/", 1)[1])
        os.ftruncate(fd, 1 << 20)
        self.assertEqual(os.fstat(fd).st_size, 1 << 20)
        with open(path, "r+b") as other:          # (how an engine opens it)
            other.seek(100)
            other.write(b"x")
        remove_arena(path)
        with self.assertRaises(OSError):
            os.fstat(fd)
        remove_arena(path)                         # (twice: harmless)

    @unittest.skipUnless(hasattr(__import__("os"), "memfd_create") and sys.platform.startswith("linux"), "Linux memfd")
    def test_a_memfd_arena_warns_when_the_engines_will_not_all_fit(self):
        cfg = {**self.CFG, "model_name": "qwen3.8-flash-next-iq3_s"}
        path, note = shared_arena_plan(cfg, 2, platform="linux", environ={}, free_bytes=57 << 30, use_memfd=True)
        try:
            self.assertTrue(path)
            self.assertIn("WARNING", note)
            self.assertIn("may not fit", note)
        finally:
            remove_arena(path)
        path, note = shared_arena_plan(cfg, 2, platform="linux", environ={}, free_bytes=120 << 30, use_memfd=True)
        try:
            self.assertNotIn("WARNING", note)
        finally:
            remove_arena(path)

    def test_a_memfd_needs_free_ram(self):
        path, note = shared_arena_plan(self.CFG, 2, platform="linux", environ={}, free_bytes=2 << 30, use_memfd=True)
        self.assertIsNone(path)
        self.assertIn("2.0 GiB of RAM", note)

    def test_removing_twice_or_nothing_is_harmless(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.bin"
            f.write_bytes(b"x")
            remove_arena(str(f))
            remove_arena(str(f))
            remove_arena(None)
            self.assertFalse(f.exists())

    def test_the_replica_engine_unlinks_the_file_at_close(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.bin"
            f.write_bytes(b"x")

            class E:
                def close(self):
                    pass
            r = ReplicaEngine([E(), E()])
            r.__dict__["arena_file"] = str(f)
            r.close()
            self.assertFalse(f.exists())


class PriorityAndIdleHandoff(unittest.TestCase):
    """Priority routing, shared-store vs live affinity in Router, and coordinated PARK handoff (r1 -> r0)."""

    def test_priority_and_prefer_idle_config_validation(self):
        self.assertIsNone(prefer_idle_config({}))
        self.assertIsNone(prefer_idle_config({"replica_prefer_idle_s": False}))
        self.assertEqual(prefer_idle_config({"replica_prefer_idle_s": 45}), 45.0)
        for bad in (-1, True, "45", float("nan"), 3601):
            with self.assertRaises(ReplicaConfigError, msg=bad):
                prefer_idle_config({"replica_prefer_idle_s": bad})

        cfg = {"replicas": [{"gpus": [0], "priority": 10}, {"gpus": [1], "priority": 0}]}
        self.assertEqual(replica_priorities(cfg), [10.0, 0.0])
        self.assertEqual(replica_priorities({"gpu": [0, 1], "replicas": 2, "replica_prefer_idle_s": 45}), [1.0, 0.0])
        self.assertEqual(replica_priorities({"gpu": [0, 1], "replicas": 2, "replica_prefer": 1}), [0.0, 1.0])
        self.assertEqual(replica_priorities({"gpu": [0, 1, 2], "replicas": 3, "replica_prefer": [2, 0]}), [1.0, 0.0, 2.0])

        for bad_cfg in ({"replicas": [{"gpus": [0], "priority": True}, {"gpus": [1]}]},
                        {"gpu": [0, 1], "replicas": 2, "replica_prefer": 2},
                        {"gpu": [0, 1], "replicas": 2, "replica_prefer": [0, 0]},
                        {"gpu": [0, 1], "replicas": 2, "replica_prefer": True}):
            with self.assertRaises(ReplicaConfigError, msg=bad_cfg):
                replica_priorities(bad_cfg)

        pc = pull_config({"replica_pull": True, "replicas": [{"gpus": [0], "priority": 5}, {"gpus": [1]}],
                          "replica_prefer_idle_s": 30})
        self.assertEqual((pc.prefer_idle_s, pc.priorities), (30.0, [5.0, 0.0]))

        with self.assertRaises(ReplicaConfigError):
            ReplicaEngine([GateEngine("r0"), GateEngine("r1")], pull=True, priorities=[1.0])
        with self.assertRaises(ReplicaConfigError):
            ReplicaEngine([GateEngine("r0"), GateEngine("r1")], pull=True, prefer_idle_s=-1)

    def test_router_distinguishes_live_from_shared_parking(self):
        r = Router(2, shared_parking=True, shared_min_tokens=BLOCK * 2)
        chat_a = [1] * (BLOCK * 4)
        chat_b = [2] * (BLOCK * 4)
        short = [3] * BLOCK
        r.note(1, chat_a)
        self.assertEqual(r.live_matched(chat_a), [0, BLOCK * 4])
        self.assertEqual(r.matched(chat_a), [0, BLOCK * 4])

        # Switching away on r1 moves chat_a (>= shared_min_tokens) into the shared bit and clears r1's live bit.
        r.note(1, chat_b)
        self.assertEqual(r.live_matched(chat_a), [0, 0])
        self.assertEqual(r.matched(chat_a), [BLOCK * 3, BLOCK * 3])
        self.assertEqual(r.live_matched(chat_b), [0, BLOCK * 4])

        # A conversation below shared_min_tokens is not marked shared on switch.
        r.note(0, short)
        r.note(0, chat_a)
        self.assertEqual(r.live_matched(short), [0, 0])
        self.assertEqual(r.matched(short), [0, 0])

        # demote_live(parked=True) drops r1's live ownership and keeps shared visibility; forget_shared clears it.
        r.demote_live(1, parked=True)
        self.assertEqual(r.live_matched(chat_b), [0, 0])
        self.assertEqual(r.matched(chat_b), [BLOCK * 3, BLOCK * 3])
        r.forget_shared()
        self.assertEqual(r.matched(chat_b), [0, 0])

    def test_pull_choose_prefer_idle_and_native_priority(self):
        chat_r1 = [1] * (BLOCK * 4)
        chat_r0 = [2] * (BLOCK * 4)

        def matched(ids):
            return [0, BLOCK * 4] if ids[:4] == chat_r1[:4] else [BLOCK * 4, 0]

        def live_matched(ids):
            return matched(ids)

        req_r1 = PullRequest()
        req_r1.seq, req_r1.ids, req_r1.arrive = 1, chat_r1 + [9] * 10, 100.0
        pending = collections.OrderedDict([(req_r1.seq, req_r1)])

        # r0 idle 10s (< 45s): keep chat_r1 on r1 without migrating.
        pick = pull_choose(pending, [0, 1], 110.0, 60.0, matched,
                           priorities=[10.0, 0.0], idle_since=[100.0, 105.0], prefer_idle_s=45.0,
                           live_matched_of=live_matched)
        self.assertEqual((pick[0].seq, pick[1], pick[2], pick[0].migrate_from), (1, 1, False, None))

        # r0 idle 45s (>= 45s): migrate chat_r1 from free r1 -> r0.
        pick = pull_choose(pending, [0, 1], 145.0, 60.0, matched,
                           priorities=[10.0, 0.0], idle_since=[100.0, 140.0], prefer_idle_s=45.0,
                           live_matched_of=live_matched)
        self.assertEqual((pick[0].seq, pick[1], pick[2], pick[0].migrate_from, pick[0].reason),
                         (1, 0, False, 1, "prefer-idle"))

        # Native request for r0 wins over migrating a request from r1 when both are waiting.
        req_r0 = PullRequest()
        req_r0.seq, req_r0.ids, req_r0.arrive = 2, chat_r0 + [9] * 10, 101.0
        pending[req_r0.seq] = req_r0
        pick = pull_choose(pending, [0, 1], 145.0, 60.0, matched,
                           priorities=[10.0, 0.0], idle_since=[100.0, 140.0], prefer_idle_s=45.0,
                           live_matched_of=live_matched)
        self.assertEqual((pick[0].seq, pick[1], pick[0].migrate_from), (2, 0, None))

    def _parking_engine(self, park_fn=None, *, prefer_idle_s=0.05):
        class ParkableGateEngine(GateEngine):
            def __init__(self, name, park_cb=None):
                super().__init__(name)
                self.park_cb = park_cb
                self.parks = 0

            def park(self):
                self.parks += 1
                if self.park_cb is not None:
                    return self.park_cb()
                return {"status": "published", "tokens": 256, "bytes": 1024, "ms": 1.0}

        e0 = ParkableGateEngine("r0")
        e1 = ParkableGateEngine("r1", park_fn)
        fake_store = SimpleNamespace(_config=SimpleNamespace(min_tokens=BLOCK * 2),
                                     replica_state=lambda owner: {"state": "connected"})
        eng = ReplicaEngine([e0, e1], pull=True, poll_s=0.01, beat_s=0.02,
                            prefer_idle_s=prefer_idle_s, priorities=[10.0, 0.0], parent_parking=fake_store)
        return eng, e0, e1

    def test_replica_engine_migrates_live_conversation_via_park_and_keeps_on_r0(self):
        eng, e0, e1 = self._parking_engine(prefer_idle_s=0.04)
        chat = [1] * (BLOCK * 4)

        # Hold r0 busy so chat lands on r1 first.
        g0 = eng.generate([9] * (BLOCK * 2), 16, {}, threading.Event())
        self.assertEqual(next(g0), 7)
        list(eng.generate(chat, 16, {}, threading.Event()))
        self.assertEqual((len(e0.calls), len(e1.calls), e1.parks), (1, 1, 0))
        g0.close()

        # Before prefer_idle_s: next turn stays on r1 without PARK.
        list(eng.generate(chat + [7, 2], 16, {}, threading.Event()))
        self.assertEqual((len(e0.calls), len(e1.calls), e1.parks), (1, 2, 0))

        # Let r0 sit idle >= prefer_idle_s: next turn triggers PARK on r1, migrates to r0, and releases r1.
        time.sleep(0.05)
        seen = []
        cancel = threading.Event()
        cancel._strata_replica_selected = seen.append
        list(eng.generate(chat + [7, 2, 7, 3], 16, {}, cancel))
        self.assertEqual(seen, [0])
        self.assertEqual((len(e0.calls), len(e1.calls), e1.parks), (2, 2, 1))
        self.assertEqual((eng.inflight, eng.pull_stats["migrated"]), ([0, 0], 1))
        self.assertEqual(eng.replica_view()[0]["queue"]["reason"], "prefer-idle")
        self.assertEqual(eng.router.live_matched(chat + [7, 2, 7, 3]), [BLOCK * 4, 0])

        # Subsequent turn naturally stays on r0 with zero further PARKs.
        list(eng.generate(chat + [7, 2, 7, 3, 7, 4], 16, {}, threading.Event()))
        self.assertEqual((len(e0.calls), len(e1.calls), e1.parks), (3, 2, 1))

    def test_replica_engine_falls_back_to_r1_when_park_refused_for_capacity(self):
        eng, e0, e1 = self._parking_engine(
            park_fn=lambda: {"status": "capacity", "tokens": 256, "bytes": 0, "ms": 0.0},
            prefer_idle_s=0.04,
        )
        chat = [5] * (BLOCK * 4)
        g0 = eng.generate([9] * (BLOCK * 2), 16, {}, threading.Event())
        self.assertEqual(next(g0), 7)
        list(eng.generate(chat, 16, {}, threading.Event()))
        g0.close()

        time.sleep(0.05)
        seen = []
        cancel = threading.Event()
        cancel._strata_replica_selected = seen.append
        list(eng.generate(chat + [7, 1], 16, {}, cancel))
        self.assertEqual(seen, [1])
        self.assertEqual((len(e0.calls), len(e1.calls), e1.parks), (1, 2, 1))
        self.assertEqual((eng.inflight, eng.pull_stats["migrated"], eng.pull_stats["migrate_fallbacks"]),
                         ([0, 0], 1, 1))
        self.assertEqual(eng.last_pick["reason"], "migrate-fallback")

    def test_displaced_conversation_routes_to_priority_r0_immediately(self):
        eng, e0, e1 = self._parking_engine(prefer_idle_s=60.0)
        chat_a = [3] * (BLOCK * 4)
        chat_b = [4] * (BLOCK * 4)

        g0 = eng.generate([9] * (BLOCK * 2), 16, {}, threading.Event())
        self.assertEqual(next(g0), 7)
        list(eng.generate(chat_a, 16, {}, threading.Event()))     # chat_a runs on r1
        list(eng.generate(chat_b, 16, {}, threading.Event()))     # chat_b displaces chat_a into shared parking
        g0.close()                                                # r0 frees just now (idle ~0s << 60s)

        seen = []
        cancel = threading.Event()
        cancel._strata_replica_selected = seen.append
        list(eng.generate(chat_a + [7, 1], 16, {}, cancel))       # chat_a is in shared store -> immediately picks r0!
        self.assertEqual(seen, [0])
        self.assertEqual((len(e0.calls), len(e1.calls), e1.parks), (2, 2, 0))
        self.assertEqual(eng.pull_stats["migrated"], 0)

    def test_strata_engine_park_command_protocol(self):
        import serve.server as server
        fake_script = (
            "import sys\n"
            "print('READY 4096 stop', flush=True)\n"
            "for line in sys.stdin:\n"
            "    cmd = line.strip()\n"
            "    if cmd == 'PARK':\n"
            "        if getattr(sys, '_fail_next', False):\n"
            "            print('ERR PARK: not while batch slots are decoding', flush=True)\n"
            "        else:\n"
            "            print('PARK status=published tokens=12345 bytes=67890 ms=12.3', flush=True)\n"
            "            sys._fail_next = True\n"
            "    elif cmd == 'QUIT':\n"
            "        break\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fake_park.py"
            p.write_text(fake_script, encoding="utf-8")
            real_popen = server.subprocess.Popen
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real_popen([sys.executable, str(p)], **kw)):
                eng = StrataEngine("strata", ["--log", str(Path(d) / "fake.log")])
            try:
                res = eng.park()
                self.assertEqual(res, {"status": "published", "tokens": 12345, "bytes": 67890, "ms": 12.3})
                with self.assertRaises(ValueError) as caught:
                    eng.park()
                self.assertIn("batch slots", str(caught.exception))
            finally:
                eng.close()


class ReplicaVramResize(unittest.TestCase):
    """Per-replica POST /v1/vram: targeting, status/metrics views, pause/drain, rollback, and error codes."""

    def _make_engine_cls(self):
        class _VramGate(GateEngine):
            def __init__(self, tok, *, elastic=True, batch=0, vram_free=1500):
                super().__init__(tok)
                self.batch = batch
                self.elastic = elastic
                self.reserves = []
                self.full_mib = 18000
                self.unloaded = False
                self.info = {
                    "expert_slots": 900,
                    "expert_cache_mib": self.full_mib,
                    "vram_free_mib": vram_free,
                    "vram_elastic": int(elastic),
                }

            def unload(self):
                self.unloaded = True
                self.up = False

            def restart(self, tries=3):
                self.unloaded = False
                super().restart(tries)

            def vram(self, reserve_mib, timeout=120.0):
                if not self.elastic:
                    raise ValueError("this engine cannot resize its VRAM use (needs --vram-elastic)")
                self.reserves.append(reserve_mib)
                eff = 2048 if reserve_mib is None else reserve_mib
                cache = max(0, self.full_mib - max(0, eff - 1500))
                slots = cache // 20
                out = {
                    "expert_slots": slots,
                    "expert_cache_mib": cache,
                    "expert_cache_full_mib": self.full_mib,
                    "vram_free_mib": eff,
                }
                self.info.update({k: out[k] for k in ("expert_slots", "expert_cache_mib", "vram_free_mib")})
                self.info["vram"] = dict(out)
                return out

        return _VramGate

    def _start(self, *, pull=False, elastic=(True, True), batch=(0, 0)):
        cls = self._make_engine_cls()
        self.children = [cls(f"r{i}", elastic=elastic[i], batch=batch[i]) for i in range(len(elastic))]
        self.engine = ReplicaEngine(
            self.children,
            **({"pull": True, "poll_s": 0.01, "beat_s": 0.02} if pull else {}),
        )
        self.tok = ByteTokenizer()
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()

    def _req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read().decode())

    def test_targeted_and_global_resize_and_status_metrics_views(self):
        self._start()
        code, body = self._req("POST", "/v1/vram", {"reserve_mib": 2500, "replica": 0})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["replica"], 0)
        self.assertEqual(self.children[0].reserves, [2500])
        self.assertEqual(self.children[1].reserves, [])

        code, body = self._req("POST", "/v1/vram", {"reserve_mib": 3200, "replica": [1]})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["replica"], 1)
        self.assertEqual(self.children[0].reserves, [2500])
        self.assertEqual(self.children[1].reserves, [3200])

        _, st = self._req("GET", "/v1/status")
        self.assertTrue(st["vram"]["elastic"])
        self.assertEqual(len(st["vram"]["replicas"]), 2)
        self.assertEqual(st["vram"]["replicas"][0]["vram_free_mib"], 2500)
        self.assertEqual(st["vram"]["replicas"][1]["vram_free_mib"], 3200)
        self.assertEqual(st["vram"]["replicas"][0]["reserve_mib"], 2500)
        self.assertEqual(st["vram"]["replicas"][1]["reserve_mib"], 3200)
        self.assertEqual(st["vram"]["vram_free_mib"], 5700)
        # existing replica_view row shape stays unchanged
        self.assertEqual(set(st["replicas"][0].keys()), {"replica", "alive", "running", "routed", "batch"})

        _, m = self._req("GET", "/metrics")
        self.assertEqual(len(m["engine"]["vram_replicas"]), 2)
        self.assertEqual(m["engine"]["vram_replicas"][0]["vram_free_mib"], 2500)
        self.assertEqual(m["engine"]["vram_replicas"][1]["vram_free_mib"], 3200)

        code, body = self._req("POST", "/v1/vram", {"reserve_mib": 1000})
        self.assertEqual(code, 200, body)
        self.assertEqual([r["replica"] for r in body["replicas"]], [0, 1])
        self.assertEqual(self.children[0].reserves[-1], 1000)
        self.assertEqual(self.children[1].reserves[-1], 1000)

        code, body = self._req("POST", "/v1/vram", {"reserve_mib": None, "replica": 0})
        self.assertEqual(code, 200, body)
        self.assertIsNone(self.children[0].reserves[-1])
        self.assertEqual(self.children[1].reserves[-1], 1000)

    def test_invalid_replica_selector_and_solo_rejected_with_400(self):
        self._start()
        for bad in (2, -1, [], True, "0", [0, 5]):
            code, body = self._req("POST", "/v1/vram", {"reserve_mib": 1000, "replica": bad})
            self.assertEqual(code, 400, (bad, body))
        self.assertEqual(self.children[0].reserves, [])
        self.assertEqual(self.children[1].reserves, [])

        cls = self._make_engine_cls()
        solo_svc = Service(cls("solo"), self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        with self.assertRaises(ValueError):
            solo_svc.vram(1000, replica=0)

    def test_pause_drains_target_and_blocks_new_routes_in_classic_and_pull_modes(self):
        for pull in (False, True):
            with self.subTest(pull=pull):
                self._start(pull=pull)
                # Hold replica 0 busy with an active request
                if pull:
                    g0 = self.engine.generate([1, 2], 16, {}, threading.Event())
                    for tok in g0:
                        if tok is not None:
                            self.assertEqual(tok, 7)
                            break
                else:
                    g0 = self.engine.generate([1, 2], 16, {}, threading.Event())
                    self.assertEqual(next(g0), 7)

                # Replica 1 resizes immediately without waiting for replica 0
                code, _ = self._req("POST", "/v1/vram", {"reserve_mib": 2100, "replica": 1})
                self.assertEqual(code, 200)

                # Busy timeout -> 409 ModelBusy while replica 0 still runs
                self.svc.vram_wait_s = 0.08
                code, _ = self._req("POST", "/v1/vram", {"reserve_mib": 2600, "replica": 0})
                self.assertEqual(code, 409)
                self.svc.vram_wait_s = 10.0

                # Start resize on replica 0 in background: while waiting for g0 to finish, new requests go ONLY to 1
                res = {}

                def do_resize():
                    res["out"] = self._req("POST", "/v1/vram", {"reserve_mib": 2600, "replica": 0})

                t = threading.Thread(target=do_resize, daemon=True)
                t.start()
                deadline = time.monotonic() + 2.0
                while 0 not in self.engine.paused and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIn(0, self.engine.paused)

                # Route a concurrent request: must go to replica 1 while 0 is paused
                routed_k = None
                g1 = self.engine.generate([3, 4], 16, {}, threading.Event())
                for tok in g1:
                    if tok is not None:
                        routed_k = tok
                        break
                self.assertIsNotNone(routed_k)
                self.assertEqual(self.engine.inflight[1], 1)
                g1.close()

                # Releasing g0 lets the resize finish
                g0.close()
                t.join(timeout=3.0)
                self.assertFalse(t.is_alive())
                self.assertEqual(res["out"][0], 200)
                self.assertEqual(self.children[0].reserves[-1], 2600)

                # Pausing ALL replicas delays route()/_pull_wait() until unpaused instead of failing EngineDied
                picked = []

                def wait_next():
                    if pull:
                        gx = self.engine.generate([5, 6], 16, {}, threading.Event())
                        for tx in gx:
                            if tx is not None:
                                picked.append(tx)
                                break
                        gx.close()
                    else:
                        picked.append(self.engine.route([5, 6]))

                with self.engine.pause([0, 1]):
                    rt = threading.Thread(target=wait_next, daemon=True)
                    rt.start()
                    time.sleep(0.05)
                    self.assertEqual(picked, [])
                rt.join(timeout=2.0)
                self.assertEqual(len(picked), 1)
                if pull:
                    self.assertEqual(picked[0], 7)
                    self.assertEqual(self.engine.inflight, [0, 0])
                else:
                    self.assertIn(picked[0], (0, 1))
                    with self.engine.cv:
                        self.engine.inflight[picked[0]] -= 1
                        self.engine.cv.notify_all()
                self.tearDown()
                self.httpd = None

    def test_batch_slots_rejected_and_rollback_on_mixed_elastic_failure(self):
        self._start(batch=(2, 0))
        code, body = self._req("POST", "/v1/vram", {"reserve_mib": 2200, "replica": 0})
        self.assertEqual(code, 400, body)
        self.tearDown()
        self.httpd = None

        self._start(elastic=(True, False))
        _, st = self._req("GET", "/v1/status")
        self.assertFalse(st["vram"]["elastic"])
        self.assertTrue(st["vram"]["replicas"][0]["elastic"])
        self.assertFalse(st["vram"]["replicas"][1]["elastic"])

        # Establish prior explicit reserve on replica 0
        code, _ = self._req("POST", "/v1/vram", {"reserve_mib": 1900, "replica": 0})
        self.assertEqual(code, 200)
        # Global resize hits failure on replica 1 -> rolls replica 0 back to 1900
        code, body = self._req("POST", "/v1/vram", {"reserve_mib": 3000})
        self.assertEqual(code, 400, body)
        self.assertEqual(self.children[0].reserves, [1900, 3000, 1900])
        self.assertEqual(self.svc.vram_reserve_by_replica[0], 1900)
        self.assertNotIn(1, self.svc.vram_reserve_by_replica)

    def test_unloaded_per_replica_reserve_reapplied_only_to_target_on_restart(self):
        self._start()
        self.svc.unload()
        code, body = self._req("POST", "/v1/vram", {"reserve_mib": 2400, "replica": 1})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["status"], "not loaded")
        self.assertEqual(self.children[0].reserves, [])
        self.assertEqual(self.children[1].reserves, [])
        self.svc.ensure_loaded()
        self.assertEqual(self.children[0].reserves, [])
        self.assertEqual(self.children[1].reserves, [2400])


if __name__ == "__main__":
    unittest.main()
