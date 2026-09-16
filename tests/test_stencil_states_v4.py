"""Tests for exact-anchor stencil expansion inputs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class ExactAnchorSelectionTests(unittest.TestCase):
    def test_preserves_saved_anchor_order_and_source_indices(self):
        from sampling.build_stencil_states_v4 import load_exact_anchor_selection

        ref = np.zeros((8, 2, 3), np.float64)
        pool = np.ones((10, 2, 3), np.float64)
        saved = np.array([
            [ref[3], pool[5]],
        ], dtype=np.float64).reshape(2, 2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anchors.npz"
            np.savez(path, R=saved, anchor_source=np.array([0, 1], np.int8),
                     anchor_index=np.array([3, 5], np.int64),
                     anchor=np.arange(2), direction=np.full(2, -1, np.int8))
            R, source, index, report = load_exact_anchor_selection(path, ref, pool)

        np.testing.assert_array_equal(R, saved)
        np.testing.assert_array_equal(source, [0, 1])
        np.testing.assert_array_equal(index, [3, 5])
        self.assertEqual(report["n_from_reference"], 1)
        self.assertEqual(report["n_from_pool"], 1)


if __name__ == "__main__":
    unittest.main()
