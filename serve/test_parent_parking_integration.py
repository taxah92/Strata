"""The parent parking store against the real C++ client: the Python store on one end of a real socketpair, the engine's
own client (src/core/conversation_parent.cpp, built as build/conv/conversation_parent_probe) on the other.

This is the one test that proves the two sides agree on the wire: the frozen protocol v1 (36-byte frame, JSON payload,
SCM_RIGHTS descriptors), the memfd the store hands out, the session file the engine writes into it, the seals the store
puts on it, and the image read back through a second replica's channel.  It needs the conversation build
(-DSTRATA_BUILD_CONVERSATION_TESTS=ON); without it the test is skipped, never silently passed.
"""
import os
import subprocess
import unittest
from pathlib import Path

from serve.parent_parking import ParentParkingConfig, ParentParkingStore

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "build" / "conv" / "conversation_parent_probe"
IDENTITY = {"model": 0x1111222233334444, "config": 0x5555666677778888, "layout_key": 0x9999AAAABBBBCCCC,
            "stages": [(0, 48)]}


def config(**changes):
    block = {"ram_gib": 1, "max_entries": 4, "min_tokens": 0, "min_gain": 1, "max_snapshot_mib": 64,
             "ipc_timeout_ms": 2000}
    block.update(changes)
    return ParentParkingConfig(**block)


def probe_args(fd, owner, mode):
    return [str(PROBE), "--fd", str(fd), "--owner", owner, "--mode", mode,
            "--model", hex(IDENTITY["model"]), "--config", hex(IDENTITY["config"]),
            "--layout", hex(IDENTITY["layout_key"]), "--stage", "0:48"]


@unittest.skipUnless(PROBE.exists(), f"needs the conversation build ({PROBE})")
class ParentParkingIntegration(unittest.TestCase):
    def setUp(self):
        self.store = ParentParkingStore(config(), IDENTITY)
        self.addCleanup(self.store.close)

    def run_probe(self, fd, owner, mode):
        """Runs the C++ client with the store's end of the channel inherited, and returns its one output line."""
        proc = subprocess.run(probe_args(fd, owner, mode), pass_fds=(fd,), capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, f"the probe failed: {proc.stdout}{proc.stderr}")
        lines = proc.stdout.strip().splitlines()
        # the probe asks for the store's counters right after HELLO, exactly as the engine does, and says so on its
        # own line; the answer the test parses is the last one
        self.assertTrue(any(ln.startswith("STATS entries=") for ln in lines), proc.stdout)
        return lines[-1]

    def test_one_replica_publishes_and_reads_its_own_snapshot(self):
        fd, channel = self.store.open_channel("r0")
        try:
            published = self.run_probe(fd, "r0", "publish")
        finally:
            os.close(fd)
        self.assertTrue(published.startswith("PUBLISHED "), published)
        entry_id, generation, size = (int(x) for x in published.split()[1:4])
        self.assertGreater(entry_id, 0)
        self.assertGreater(generation, 0)
        self.assertGreater(size, 0)
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["entries"], 1)
        self.assertEqual(snapshot["entry_bytes"], size)
        self.assertEqual(snapshot["charged_bytes"], size)
        self.assertEqual(snapshot["budget_violations"], 0)

        fd, channel = self.store.open_channel("r0")
        try:
            read = self.run_probe(fd, "r0", "read")
        finally:
            os.close(fd)
        self.assertTrue(read.startswith("READ 4 r0 "), read)
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["reads"], 1)
        self.assertEqual(snapshot["leases"], 0)          # the probe released it
        self.assertEqual(snapshot["restore_reserved_bytes"], 0)
        self.assertEqual(snapshot["entries"], 1)         # a whole restore does not take the entry

    def test_a_second_replica_reads_what_the_first_published(self):
        fd, channel = self.store.open_channel("r0")
        try:
            published = self.run_probe(fd, "r0", "publish")
        finally:
            os.close(fd)
        self.assertTrue(published.startswith("PUBLISHED "), published)

        fd, channel = self.store.open_channel("r1")
        try:
            read = self.run_probe(fd, "r1", "read")
        finally:
            os.close(fd)
        self.assertTrue(read.startswith("READ 4 r0 "), read)   # the owner is the replica that published it
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["reads"], 1)
        self.assertEqual(snapshot["peer_reads"], 1)
        self.assertEqual(snapshot["entries"], 1)

    def test_the_same_snapshot_published_twice_is_one_entry(self):
        fd, channel = self.store.open_channel("r0")
        try:
            self.run_probe(fd, "r0", "publish")
        finally:
            os.close(fd)
        fd, channel = self.store.open_channel("r1")
        try:
            second = self.run_probe(fd, "r1", "publish")
        finally:
            os.close(fd)
        self.assertEqual(second, "DEDUP")
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["entries"], 1)
        self.assertEqual(snapshot["deduplicated"], 1)

    def test_a_reader_that_dies_leaves_the_entry_and_frees_its_lease(self):
        fd, channel = self.store.open_channel("r0")
        try:
            self.run_probe(fd, "r0", "publish")
        finally:
            os.close(fd)
        # a channel that says HELLO and then dies: the store must drop its connection state, not its entries
        fd, channel = self.store.open_channel("r1")
        proc = subprocess.Popen(probe_args(fd, "r1", "read"), pass_fds=(fd,), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        os.close(fd)
        proc.kill()
        proc.wait(timeout=30)
        for _ in range(200):
            if self.store.replica_state("r1") in ("degraded", "closed"):
                break
            import time
            time.sleep(0.01)
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["entries"], 1)
        self.assertEqual(snapshot["leases"], 0)
        self.assertEqual(snapshot["restore_reserved_bytes"], 0)

    def test_clear_empties_the_store_for_every_owner(self):
        fd, channel = self.store.open_channel("r0")
        try:
            self.run_probe(fd, "r0", "publish")
        finally:
            os.close(fd)
        self.assertEqual(self.store.snapshot()["entries"], 1)
        self.assertEqual(self.store.clear("nonfinite"), 1)
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["entries"], 0)
        self.assertEqual(snapshot["charged_bytes"], 0)
        self.assertEqual(snapshot["store_epoch"], 2)
        self.assertEqual(snapshot["clears"], 1)


if __name__ == "__main__":
    unittest.main()
