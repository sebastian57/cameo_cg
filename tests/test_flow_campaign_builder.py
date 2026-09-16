"""Regression tests for generated flow sampling campaigns."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


class FlowCampaignBuilderTests(unittest.TestCase):
    def test_group_filename_matches_submit_script_width(self):
        from sampling.build_flow_md_campaign import main

        repo = Path(__file__).resolve().parents[1]
        out = Path(tempfile.mkdtemp()) / "campaign"
        argv = [
            "build_flow_md_campaign",
            "--bias-npz",
            str(repo.parent / "SAMPLING/tica_regional_weighting/results/ala2_bb6_reference/"
                "smooth_reference_bias_lambda0p25.npz"),
            "--reference",
            str(repo / "local_work/input_data/ala2_cg_backbone_cb_6bead_200k.npz"),
            "--flow",
            str(repo / "local_work/flow_sweep_final/flow_small_seed0.npz"),
            "--mdp",
            str(repo / "sampling/campaigns/production_298K_dt1fs.mdp"),
            "--topology",
            str(repo / "local_work/aa_reference/ala2_constrained/topol.top"),
            "--structures",
            str(repo / "local_work/aa_reference/start_frames_bb6"),
            "--outdir",
            str(out),
            "--n-replicas",
            "1",
            "--grid-points",
            "1",
            "--arms",
            "flow",
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            main()
        finally:
            sys.argv = old_argv

        self.assertTrue((out / "flow/run_group_0000.sh").exists())
        self.assertFalse((out / "flow/run_group_000.sh").exists())


if __name__ == "__main__":
    unittest.main()
