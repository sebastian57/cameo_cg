"""Tests for post-label trajectory cleanup."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class CleanupTests(unittest.TestCase):
    def test_cleanup_removes_in_state_trajectories_but_keeps_inputs(self):
        from sampling.collect_meanforce import cleanup_state_trajectories

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state_000000"
            state.mkdir()
            (state / "seed.gro").write_text("seed")
            (state / "production.mdp").write_text("mdp")
            (state / "biased.trr").write_bytes(b"trajectory")
            (state / "unbiased_forces.trr").symlink_to("biased.trr")

            removed = cleanup_state_trajectories(state)

            self.assertEqual(removed, len(b"trajectory"))
            self.assertFalse((state / "biased.trr").exists())
            self.assertFalse((state / "unbiased_forces.trr").exists())
            self.assertTrue((state / "seed.gro").exists())
            self.assertTrue((state / "production.mdp").exists())

    def test_cleanup_refuses_trajectory_symlink_leaving_external_file_intact(self):
        from sampling.collect_meanforce import cleanup_state_trajectories

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state_000000"
            state.mkdir()
            external = root / "external.trr"
            external.write_bytes(b"must remain")
            (state / "biased.trr").symlink_to(external)

            with self.assertRaises(ValueError):
                cleanup_state_trajectories(state)

            self.assertTrue(external.exists())
            self.assertTrue((state / "biased.trr").is_symlink())


if __name__ == "__main__":
    unittest.main()
