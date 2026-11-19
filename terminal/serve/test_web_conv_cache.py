"""Monitor's Conversation cache card (serve/web/app.js): parent RAM parking, cross-replica transfers, legacy fallback."""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class ConvCacheCardInNode(unittest.TestCase):
    def test_parent_parking_and_handoff_rendering(self):
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("test_web_conv_cache.js"))],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
