"""Tests for v5.1 pool seed lookup compatibility."""
from __future__ import annotations

import unittest
from pathlib import Path


class PoolSeedLookupTests(unittest.TestCase):
    def test_reference_mdp_rewrite_uses_base_timestep_and_independent_state_start(self):
        from sampling.build_stencil_campaign_v4 import rewrite_production_mdp

        base = [
            "integrator = md",
            "dt = 0.002",
            "nsteps = 250000",
            "nstxout = 100",
            "nstvout = 100",
            "nstfout = 100",
            "nstxout-compressed = 1000",
            "continuation = yes",
            "gen_vel = no",
            "gen_seed = -1",
        ]
        out = rewrite_production_mdp(base, ps_per_state=3.8, output_ps=0.2,
                                     gen_seed=1234)

        self.assertIn("nsteps                  = 1900 ; 3.8 ps", out)
        self.assertIn("nstxout                 = 100", out)
        self.assertIn("nstvout                 = 100", out)
        self.assertIn("nstfout                 = 100", out)
        self.assertTrue(any(line.startswith("nstxout-compressed") and line.split("=", 1)[1].strip() == "1000" for line in out))
        self.assertIn("continuation            = no", out)
        self.assertIn("gen_vel                 = yes", out)
        self.assertIn("gen_temp                = 298", out)
        self.assertIn("gen_seed                = 1234", out)
        self.assertTrue(any(line.startswith("freezegrps              = CGbeads") for line in out))
        self.assertIn("freezedim               = Y Y Y", out)

    def test_seed_gro_writer_preserves_reference_atom_names(self):
        from sampling.build_stencil_campaign_v4 import format_seed_gro

        template = [
            "reference",
            "2",
            "    1ACE   HH31    1   0.000   0.000   0.000  1.0  2.0  3.0",
            "    1ACE     O    2   0.100   0.100   0.100  4.0  5.0  6.0",
            "   1.0   1.0   1.0",
        ]
        out = format_seed_gro(template, [[1.23456, 2.34567, 3.45678],
                                         [0.2, 0.3, 0.4]], [2.0, 2.0, 2.0])

        self.assertIn("HH31", out[2])
        self.assertIn("    O", out[3])
        self.assertEqual(out[2][20:44], "   1.235   2.346   3.457")
        self.assertEqual(out[3][20:44], "   0.200   0.300   0.400")
        self.assertNotIn("1.0  2.0  3.0", out[2])

        triclinic = format_seed_gro(template, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0],
                                     [0.5, 0.25, 3.0]])
        self.assertEqual([float(x) for x in triclinic[-1].split()],
                         [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.25])

    def test_case_template_and_frame_offset(self):
        from sampling.build_stencil_campaign_v4 import pool_case_path, pool_source_frame

        root = Path("/campaign")
        self.assertEqual(pool_case_path(root, "case_{index:03d}", 3),
                         root / "case_003")
        self.assertEqual(pool_case_path(root, "replica_{index:02d}", 3),
                         root / "replica_03")
        self.assertEqual(pool_source_frame(37, 100), 137)

    def test_pool_case_frame_grouping_initializes_new_cases(self):
        from sampling.build_stencil_campaign_v4 import pool_case_frame_groups

        groups = pool_case_frame_groups(
            pool_slots=[0, 1, 2],
            anchor_index=[12, 2, 19],
            pool_starts=[0, 10, 20])
        self.assertEqual(groups, {0: [(2, 1)], 1: [(2, 0), (9, 2)]})

    def test_state_manifest_entry_rounds_realized_target(self):
        from sampling.build_stencil_campaign_v4 import state_manifest_entry

        entry = state_manifest_entry(
            state=7, anchor=2, direction=3,
            target=[[1.23456, 2.34567, 3.45678]],
            anchor_source=1, anchor_index=19, multiplier=-1.0, kind=2)
        self.assertEqual(entry["state"], 7)
        self.assertEqual(entry["anchor"], 2)
        self.assertEqual(entry["target"], [[1.235, 2.346, 3.457]])
        self.assertEqual(entry["anchor_source"], 1)
        self.assertEqual(entry["anchor_index"], 19)
        self.assertEqual(entry["multiplier"], -1.0)
        self.assertEqual(entry["kind"], 2)

    def test_state_records_are_initialized_before_expansion(self):
        source = Path(__file__).resolve().parents[1] / "sampling/build_stencil_campaign_v4.py"
        text = source.read_text()
        self.assertLess(text.index("state_records = []"), text.index("for k in order:"))


if __name__ == "__main__":
    unittest.main()
