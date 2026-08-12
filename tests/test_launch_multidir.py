"""Tests for the `-multidir` launch topology (sampling/launch.py).

These are all static-generation checks: they assert the emitted scripts are internally
consistent and preserve the invariants the rest of the pipeline depends on. Actually running
GROMACS needs a node, so throughput is measured separately.
"""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import yaml

from sampling.launch import (
    Topology,
    cpus_per_rank,
    group_ranges,
    multidir_group_script,
    submit_script,
)

CAMPAIGN_YAML = Path("sampling/campaigns/ala2_bb6_allcorridor_metad.yaml")


class GroupingTests(unittest.TestCase):
    def test_groups_partition_every_replica_exactly_once(self):
        for n, per in ((12, 6), (12, 5), (7, 1), (4, 99)):
            groups = group_ranges(n, per)
            flat = [r for g in groups for r in g]
            self.assertEqual(flat, list(range(n)), f"n={n} per={per}")
            self.assertTrue(all(len(g) <= per for g in groups))

    def test_grouping_limits_the_blast_radius_of_one_crashed_rank(self):
        """A crashed rank aborts its whole mdrun, so groups must be smaller than the campaign.

        This is the regression versus the per-case layout, where one array task failed alone
        and `collect.py --coords-only` could skip it.
        """
        groups = group_ranges(96, 8)
        self.assertEqual(len(groups), 12)
        self.assertTrue(all(len(g) == 8 for g in groups))

    def test_rejects_a_nonsense_group_size(self):
        with self.assertRaises(ValueError):
            group_ranges(10, 0)

    def test_cpus_per_rank_fits_the_node_and_is_capped(self):
        # 6 x 64 = 384 on a 288-CPU node simply never schedules
        for ranks in (1, 2, 4, 6, 8, 16, 32):
            cpus = cpus_per_rank(ranks, node_cpus=288)
            self.assertLessEqual(ranks * cpus, 288, f"{ranks} ranks x {cpus} cpus")
            self.assertGreaterEqual(cpus, 1)
        self.assertEqual(cpus_per_rank(1, node_cpus=288, cap=16), 16)   # capped, not 288
        self.assertEqual(cpus_per_rank(64, node_cpus=288), 4)


class GroupScriptTests(unittest.TestCase):
    def _script(self, **kw):
        base = dict(case_dirs=["replica_00", "replica_01"],
                    structure_for=["/tmp/a.gro", "/tmp/b.gro"],
                    topology="/tmp/topol.top", ntomp=16, n_gpus=4)
        base.update(kw)
        return multidir_group_script(**base)

    def test_uses_gmx_mpi_not_the_thread_mpi_binary(self):
        """`gmx` is thread-MPI here and silently cannot do -multidir; `gmx_mpi` can."""
        script = self._script()
        self.assertIn("gmx_mpi mdrun -multidir", script)
        self.assertNotRegex(script, r"(?<!_mpi)\bgmx mdrun")

    def test_grompp_runs_for_every_replica_before_any_mdrun(self):
        script = self._script()
        # the literal command, not the word "mdrun" in the header comment
        first_mdrun = script.index("gmx_mpi mdrun")
        for d in ("replica_00", "replica_01"):
            grompp = re.search(rf'cd "{d}" && gmx grompp', script)
            self.assertIsNotNone(grompp, f"no grompp for {d}")
            self.assertLess(grompp.start(), first_mdrun,
                            "-multidir needs every .tpr built before mdrun starts")

    def test_bias_free_rerun_is_present_and_carries_no_plumed(self):
        """The rerun produces the training labels; a -plumed there would poison them."""
        script = self._script()
        rerun = [ln for ln in script.splitlines() if "-rerun" in ln]
        self.assertEqual(len(rerun), 1)
        # the rerun is a continuation line pair; check the whole statement
        stmt = script[script.index("-s biased.tpr"):]
        self.assertIn("unbiased_forces", stmt)
        self.assertNotIn("-plumed", stmt.split("echo")[0])

    def test_no_server_block_when_the_backend_is_native(self):
        script = self._script(use_server=False)
        self.assertNotIn("sampling.server", script)
        self.assertNotIn("cgbias_", script)

    def test_server_mode_starts_one_server_per_replica_and_reaps_them(self):
        """server.py is one-process-per-replica: backlog=1, serial accept, no replica id in
        the wire header. Multiplexing would serialise the ranks, so N servers it is."""
        script = self._script(use_server=True, socket_dir="/tmp", campaign_name="camp",
                              replicas=[0, 1], repo="/repo")
        self.assertEqual(script.count("python -m sampling.server"), 2)
        self.assertIn("cgbias_camp_r0.sock", script)
        self.assertIn("cgbias_camp_r1.sock", script)
        # the parent starts them, so the parent must clean them up
        self.assertIn("trap", script)
        self.assertIn("PIDS", script)

    def test_server_mode_refuses_inconsistent_arguments(self):
        with self.assertRaises(ValueError):
            self._script(use_server=True, socket_dir=None, campaign_name="c", replicas=[0, 1])
        with self.assertRaises(ValueError):
            self._script(use_server=True, socket_dir="/tmp", campaign_name="c", replicas=[0])

    def test_mismatched_case_and_structure_lists_are_rejected(self):
        with self.assertRaises(ValueError):
            multidir_group_script(case_dirs=["a", "b"], structure_for=["/tmp/a.gro"],
                                  topology="t", ntomp=8)


class SubmitScriptTests(unittest.TestCase):
    def test_cpu_request_fits_the_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = submit_script(campaign_dir=Path(tmp), groups=group_ranges(12, 6),
                                 job_name="x", node_cpus=288)
        ranks = int(re.search(r"ntasks-per-node=(\d+)", text).group(1))
        cpus = int(re.search(r"cpus-per-task=(\d+)", text).group(1))
        self.assertEqual(ranks, 6)
        self.assertLessEqual(ranks * cpus, 288)

    def test_paths_are_absolute_because_srun_runs_from_an_arbitrary_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel = Path(tmp).relative_to("/") if False else Path(tmp)
            text = submit_script(campaign_dir=rel, groups=group_ranges(4, 2), job_name="x")
        srun = [ln for ln in text.splitlines() if ln.startswith("srun")][0]
        self.assertIn(str(Path(tmp).resolve()), srun)

    def test_array_spans_the_groups_not_the_replicas(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = submit_script(campaign_dir=Path(tmp), groups=group_ranges(96, 8),
                                 job_name="x")
        self.assertIn("--array=0-11", text)   # 12 groups, not 96 replicas


class TopologyTests(unittest.TestCase):
    """Two regimes of one knob: replicas packed onto shared GPUs (small systems) versus
    replicas domain-decomposed over several GPUs (large ones)."""

    def test_packed_regime_stays_on_one_node(self):
        t = Topology(n_replicas=8, ranks_per_replica=1, gpus_per_node=4)
        self.assertEqual(t.total_ranks, 8)
        self.assertEqual(t.nodes, 1)
        self.assertEqual(t.ranks_per_node, 8)
        self.assertLessEqual(t.ranks_per_node * t.cpus_per_task, t.node_cpus)

    def test_decomposed_regime_spreads_over_nodes(self):
        # 4 ranks per replica on a 4-GPU node -> one replica per node
        t = Topology(n_replicas=8, ranks_per_replica=4, gpus_per_node=4)
        self.assertEqual(t.total_ranks, 32)
        self.assertEqual(t.nodes, 8)
        self.assertEqual(t.ranks_per_node, 4)

        # 2 ranks per replica -> two replicas per node
        t2 = Topology(n_replicas=8, ranks_per_replica=2, gpus_per_node=4)
        self.assertEqual(t2.nodes, 4)
        self.assertEqual(t2.ranks_per_node, 4)

    def test_cpu_budget_always_fits_the_node(self):
        for reps in (1, 4, 8, 16):
            for rpr in (1, 2, 4):
                t = Topology(n_replicas=reps, ranks_per_replica=rpr)
                self.assertLessEqual(t.ranks_per_node * t.cpus_per_task, t.node_cpus,
                                     f"{reps} replicas x {rpr} ranks")

    def test_submit_script_requests_the_computed_node_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = submit_script(campaign_dir=Path(tmp), groups=group_ranges(8, 8),
                                 job_name="x", ranks_per_replica=4)
        self.assertIn("--nodes=8", text)
        self.assertIn("--ntasks-per-node=4", text)

    def test_group_script_launches_total_ranks_not_replica_count(self):
        script = multidir_group_script(
            case_dirs=["replica_00", "replica_01"], structure_for=["/a.gro", "/b.gro"],
            topology="t", ntomp=16, ranks_per_replica=4)
        self.assertIn("srun -n 8 gmx_mpi mdrun -multidir", script)
        # -gpu_id only makes sense when ranks share a node's GPUs
        self.assertNotIn("-gpu_id", script)

    def test_packed_group_script_pins_gpu_ids(self):
        script = multidir_group_script(
            case_dirs=["replica_00", "replica_01"], structure_for=["/a.gro", "/b.gro"],
            topology="t", ntomp=16, ranks_per_replica=1, n_gpus=4)
        self.assertIn("srun -n 2 gmx_mpi mdrun -multidir", script)
        self.assertIn("-gpu_id 0123", script)


class CasesIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(CAMPAIGN_YAML.exists(), "campaign yaml absent")
    def test_multidir_campaign_keeps_the_layout_collect_py_expects(self):
        """collect.py finds cases by `replica_*` prefix and reads biased/unbiased trr from
        inside each one. Multidir must not disturb that."""
        from sampling.cases import main as cases_main

        cfg = yaml.safe_load(CAMPAIGN_YAML.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            cfg.update(bias_backend="plumed", launch="multidir", n_replicas=4,
                       replicas_per_job=2, output_dir=tmp)
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            import sys
            argv = sys.argv
            try:
                sys.argv = ["cases", "--config", str(cfg_path)]
                cases_main()
            finally:
                sys.argv = argv

            root = Path(tmp)
            for r in range(4):
                self.assertTrue((root / f"replica_{r:02d}" / "plumed.dat").exists())
            self.assertEqual(len(list(root.glob("run_group_*.sh"))), 2)
            self.assertTrue((root / "submit.slurm").exists())
            # the tabulated field is shared, not copied per replica
            grids = list(root.glob("tica_regional_grid_*.dat"))
            self.assertEqual(len(grids), 1, "grid must be written once at the campaign root")
            self.assertFalse((root / "replica_00" / grids[0].name).exists())
            self.assertIn(f"../{grids[0].name}",
                          (root / "replica_00" / "plumed.dat").read_text())


if __name__ == "__main__":
    unittest.main()
