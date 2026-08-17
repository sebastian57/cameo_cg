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

    def test_biased_phase_uses_gmx_mpi_not_the_thread_mpi_binary(self):
        """`gmx` is thread-MPI here and silently cannot do -multidir; `gmx_mpi` can.

        Scoped to the BIASED phase only: the rerun deliberately uses plain `gmx`, because
        `-rerun` does not support multi-simulation at all (see GromacsMultidirConstraintTests).
        """
        script = self._script()
        biased = script[script.index("2. biased production"):script.index("3. bias-free")]
        self.assertIn("gmx_mpi mdrun -multidir", biased)
        self.assertNotRegex(biased, r"(?<!_mpi)\bgmx mdrun")

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

    def test_paths_are_absolute_because_the_batch_job_runs_from_an_arbitrary_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = submit_script(campaign_dir=Path(tmp), groups=group_ranges(4, 2),
                                 job_name="x")
        # the launch line is `bash <abs>/run_group_NNN.sh` -- it was `srun ...` until the
        # nested-srun deadlock of 2026-08-12, so select it by content, not by command
        launch = [ln for ln in text.splitlines()
                  if "run_group_" in ln and not ln.lstrip().startswith("#")][0]
        self.assertIn(str(Path(tmp).resolve()), launch)

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
        # one GPU per rank: `-gpu_id 0123` with 2 ranks dies with "task assignment failed"
        self.assertIn("-gpu_id 01", script)
        self.assertNotIn("-gpu_id 0123", script)


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


class NestedSrunRegressionTests(unittest.TestCase):
    """Jobs 1324907-1324910 and 1324913 (2026-08-12) burned their whole walltime on
    `srun: Job step creation temporarily disabled` and produced nothing. Cause: the batch
    script wrapped the group script in an outer `srun`, whose step held the allocation, so
    the group script's own `srun -n N gmx_mpi mdrun -multidir` could never start."""

    def test_submit_script_does_not_wrap_the_group_script_in_srun(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = submit_script(campaign_dir=Path(tmp), groups=group_ranges(4, 2),
                                 job_name="x")
        launch = [ln for ln in text.splitlines()
                  if "run_group_" in ln and not ln.lstrip().startswith("#")]
        self.assertEqual(len(launch), 1, launch)
        self.assertNotRegex(launch[0], r"^\s*srun\b",
                            "outer srun around the group script deadlocks the inner one")
        self.assertRegex(launch[0], r"^\s*bash\b")

    def test_group_script_still_uses_srun_for_mdrun(self):
        """The inner srun is the one that must survive -- it is what spans the ranks."""
        script = multidir_group_script(case_dirs=["replica_00", "replica_01"],
                                       structure_for=["/a.gro", "/b.gro"],
                                       topology="t", ntomp=16)
        self.assertIn("srun -n 2 gmx_mpi mdrun -multidir", script)


class GromacsMultidirConstraintTests(unittest.TestCase):
    """Three constraints learned by running (2026-08-13, jobs 1342172-1342186). All three
    produced a job that consumed its allocation and wrote no usable output."""

    def _script(self, n, **kw):
        base = dict(case_dirs=[f"replica_{i:02d}" for i in range(n)],
                    structure_for=[f"/s{i}.gro" for i in range(n)],
                    topology="/t.top", ntomp=16, n_gpus=4)
        base.update(kw)
        return multidir_group_script(**base)

    def test_single_replica_group_does_not_use_multidir(self):
        """`-multidir` with one directory: 'The single simulation case is not supported'."""
        s = self._script(1)
        cmds = [ln for ln in s.splitlines()
                if "mdrun" in ln and not ln.lstrip().startswith("#")]
        self.assertTrue(cmds)
        for ln in cmds:
            self.assertNotIn("-multidir", ln, ln)
        self.assertIn("gmx mdrun -deffnm biased", s)

    def test_rerun_is_never_multidir(self):
        """'Multiple simulations not supported by rerun' -- rerun.cpp:258. The biased mdrun
        completes and reports Performance, THEN the job dies, so this is easy to miss."""
        for n in (2, 4, 8):
            s = self._script(n)
            rerun_stmt = s[s.index("bias-free force rerun"):]
            self.assertIn("-rerun biased.trr", rerun_stmt)
            self.assertNotIn("-multidir", rerun_stmt, f"n={n}")
            self.assertIn("gmx mdrun -s biased.tpr", rerun_stmt)

    def test_rerun_runs_replicas_concurrently_and_checks_every_exit(self):
        s = self._script(4)
        tail = s[s.index("bias-free force rerun"):]
        self.assertIn("pids+=($!)", tail)
        self.assertIn('wait "$p" || fail=1', tail)
        self.assertIn("CUDA_VISIBLE_DEVICES=$(( i % 4 ))", tail)

    def test_gpu_id_never_lists_more_gpus_than_ranks(self):
        """2 ranks against `-gpu_id 0123` dies with a bare 'task assignment failed'."""
        import re
        for n, expect in ((2, "01"), (4, "0123"), (8, "0123")):
            m = re.search(r"-gpu_id (\d+)", self._script(n))
            self.assertIsNotNone(m, f"n={n}")
            self.assertEqual(m.group(1), expect, f"n={n}")


class PlumedOutputNamingTests(unittest.TestCase):
    """PLUMED suffixes every output with the replica index under -multidir (colvar.0.dat,
    colvar.1.dat, ...). Downstream readers use the plain name -- build_harvest_campaign.py
    opens `<case>/colvar.dat` literally -- so the index is stripped after the biased run.
    Measured on job 1342295."""

    def _script(self, n):
        return multidir_group_script(
            case_dirs=[f"case_{i:03d}" for i in range(n)],
            structure_for=["seed.gro"] * n, topology="/t.top", ntomp=8, n_gpus=4)

    def test_index_is_stripped_for_multi_replica_groups(self):
        s = self._script(4)
        self.assertIn("strip PLUMED's per-replica index", s)
        self.assertIn('mv -f "$f"', s)
        self.assertIn('mv -f "$d/HILLS.$i" "$d/HILLS"', s)
        # must happen after the biased run and before anything reads the files
        self.assertLess(s.index("gmx_mpi mdrun -multidir"), s.index("strip PLUMED"))

    def test_no_rename_for_a_single_simulation(self):
        """Without -multidir PLUMED writes the plain name; renaming would be wrong."""
        s = self._script(1)
        self.assertNotIn('mv -f "$f"', s)
        self.assertIn("does not suffix", s)
