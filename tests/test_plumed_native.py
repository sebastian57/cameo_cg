"""Equivalence tests: generated PLUMED input vs the Python bias implementations.

The Python biases in `sampling/biases/` are the ORACLE. They already have finite-difference
force tests in `test_sampling_integrated.py`, so if the generated `plumed.dat` reproduces them
the native route is correct by transitivity.

Everything here runs through `plumed driver`, which evaluates a plumed.dat on an xyz
trajectory offline -- no GROMACS, no GPU, no bias server. Tests skip if PLUMED is absent.

TWO TRAPS THESE TESTS EXIST TO PIN, both found while writing the generator:

1. `plumed driver --dump-forces` writes forces in PLUMED's INTERNAL energy unit (kJ/mol) even
   when the input declares `UNITS ENERGY=kcal/mol`. UNITS governs input parsing and PRINT
   output, not the force dump. Forces must be divided by 4.184 before comparison. Missing
   this looks exactly like a 4x wrong bias.
2. `tica_energy_gradient` already includes the harmonic walls, so tabulating it into the
   EXTERNAL grid AND emitting UPPER/LOWER_WALLS double-counts them. `write_external_grid`
   zeroes the wall stiffness for tabulation; `test_walls_are_not_double_counted` is the guard.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from utils.jax_setup import apply_jax_compat_shims  # noqa: F401  (import-order parity)

from sampling.biases.tica_regional import SmoothTICABias
from sampling.mapping import get_mapping
from sampling.plumed_native import (
    _row_energy_gradient,
    _without_walls,
    external_block,
    tica_cv_block,
    walls_block,
    write_external_grid,
)

KJ_PER_KCAL = 4.184
GMX_MODULES = "Stages/2025 GCC/13.3.0 ParaStationMPI/5.11.0-1 GROMACS/2024.3-PLUMED-2.9.3"
BIAS_NPZ = Path("/e/project1/cameo/schmidt36/SAMPLING/tica_regional_weighting/results/"
                "ala2_bb6_reference/smooth_allcorridor_attractor_A2.npz")
REFERENCE = Path("/e/project1/cameo/schmidt36/cameo_cg/local_work/input_data/"
                 "ala2_cg_backbone_cb_6bead_200k.npz")


def _have_plumed() -> bool:
    if shutil.which("plumed"):
        return True
    probe = subprocess.run(
        ["bash", "-lc", f"module --force purge >/dev/null 2>&1; "
                        f"module load {GMX_MODULES} >/dev/null 2>&1; command -v plumed"],
        capture_output=True, text=True)
    return probe.returncode == 0 and bool(probe.stdout.strip())


def _run_driver(workdir: Path, plumed_file: str, xyz: str = "frames.xyz",
                dump_forces: str | None = None) -> None:
    cmd = (f"plumed driver --ixyz {xyz} --plumed {plumed_file} --length-units A")
    if dump_forces:
        cmd += f" --dump-forces {dump_forces}"
    full = (f"module --force purge >/dev/null 2>&1; module load {GMX_MODULES} >/dev/null 2>&1; "
            f"cd {workdir} && {cmd}")
    proc = subprocess.run(["bash", "-lc", full], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"plumed driver failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def _write_xyz(path: Path, R: np.ndarray) -> None:
    """xyz with a box on the comment line -- the driver refuses the file without one."""
    with path.open("w") as fh:
        for frame in R:
            fh.write(f"{len(frame)}\n1000.0 1000.0 1000.0\n")
            for p in frame:
                fh.write(f"C {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def _read_dumped_forces(path: Path, n_atoms: int) -> np.ndarray:
    """`--dump-forces` is xyz-shaped: count line, virial line, then `X fx fy fz` per atom."""
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    out, i = [], 0
    while i < len(rows):
        count = int(rows[i][0])
        i += 2  # skip the count and the virial line
        out.append([[float(v) for v in rows[i + k][1:4]] for k in range(count)])
        i += count
    arr = np.asarray(out)
    assert arr.shape[1] == n_atoms, f"expected {n_atoms} atoms, parsed {arr.shape[1]}"
    return arr


class _BeadMapping:
    """PLUMED addressing the 6 CG beads directly as atoms 1..6.

    The driver runs on a standalone bead-only xyz, whereas a campaign's plumed.dat addresses
    the AA atom numbers. Only the atom NUMBERS differ; the generated structure is identical,
    which is what these tests are checking.
    """

    aa_atom_indices_1based = tuple(range(1, 7))

    def __init__(self):
        self.cvs = get_mapping("ala2_backbone_cb_6").cvs

    def plumed_atom_selection(self) -> str:
        return "1-6"


@unittest.skipUnless(_have_plumed(), "plumed not available")
@unittest.skipUnless(BIAS_NPZ.exists() and REFERENCE.exists(), "bias/reference artifacts absent")
class PlumedNativeEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bias = SmoothTICABias.load(BIAS_NPZ)
        cls.mapping = _BeadMapping()
        with np.load(REFERENCE) as data:
            cls.R = np.asarray(data["R"][:2000:100], dtype=np.float64)
        cls.tmp = Path(os.environ.get("CAMEO_TEST_TMP",
                                      "/e/project1/cameo/schmidt36/cameo_cg/local_work/"
                                      "plumed_check/pytest"))
        cls.tmp.mkdir(parents=True, exist_ok=True)
        _write_xyz(cls.tmp / "frames.xyz", cls.R)

    def test_tica_cvs_match_the_python_projection_exactly(self):
        """COMBINE over DISTANCE is the same function as (d - mean) @ coefficients."""
        (self.tmp / "cv_only.dat").write_text(
            "UNITS LENGTH=A ENERGY=kcal/mol\n"
            + tica_cv_block(self.bias, self.mapping)
            + "PRINT ARG=tic1,tic2 FILE=cv_only_out.dat STRIDE=1\n")
        _run_driver(self.tmp, "cv_only.dat")
        plumed = np.loadtxt(self.tmp / "cv_only_out.dat")[:, 1:3]
        python = self.bias.projection.transform(self.R)
        # cv output is printed to 6 decimals, so this is exact to the file's precision
        np.testing.assert_allclose(plumed, python, atol=2e-6)

    def test_analytic_restraint_forces_match_after_the_kJ_conversion(self):
        """Pins trap #1: dump-forces is kJ/mol/A even under UNITS ENERGY=kcal/mol.

        Uses a plain RESTRAINT so there is no grid interpolation anywhere -- any discrepancy
        here is the CV chain rule or the units, nothing else.
        """
        (self.tmp / "rest.dat").write_text(
            "UNITS LENGTH=A ENERGY=kcal/mol\n"
            + tica_cv_block(self.bias, self.mapping)
            + "r1: RESTRAINT ARG=tic1 AT=0.0 KAPPA=1.0\n"
              "PRINT ARG=tic1,r1.bias FILE=rest_out.dat STRIDE=1\n")
        _run_driver(self.tmp, "rest.dat", dump_forces="rest_forces.dat")
        forces = _read_dumped_forces(self.tmp / "rest_forces.dat", 6) / KJ_PER_KCAL

        expected = np.empty_like(forces)
        for k, frame in enumerate(self.R):
            z, jac = self.bias.projection.value_and_jacobian(frame)
            expected[k] = -np.einsum("k,kij->ij", np.array([z[0], 0.0]), jac)
        np.testing.assert_allclose(forces, expected, atol=1e-6)

    def test_external_grid_reproduces_energy_and_forces(self):
        grid = self.tmp / "grid.dat"
        write_external_grid(self.bias, grid, n_points=(401, 401), pad=0.15, label="treg")
        (self.tmp / "full.dat").write_text(
            "UNITS LENGTH=A ENERGY=kcal/mol\n"
            + tica_cv_block(self.bias, self.mapping)
            + external_block(grid) + walls_block(self.bias)
            + "PRINT ARG=tic1,tic2,treg.bias,twall_lo.bias,twall_hi.bias "
              "FILE=full_out.dat STRIDE=1\n")
        _run_driver(self.tmp, "full.dat", dump_forces="full_forces.dat")

        cv = np.loadtxt(self.tmp / "full_out.dat")
        energy_plumed = cv[:, 3] + cv[:, 4] + cv[:, 5]
        forces_plumed = _read_dumped_forces(self.tmp / "full_forces.dat", 6) / KJ_PER_KCAL

        energy_py = np.empty(len(self.R))
        forces_py = np.empty_like(forces_plumed)
        with np.errstate(divide="ignore"):
            for k, frame in enumerate(self.R):
                energy_py[k], forces_py[k], _ = self.bias.evaluate_A(frame)

        # Tolerances are the measured grid-interpolation error at 401x401, not guesses:
        # max|dE| 5.2e-3 kcal/mol and max|dF| ~2% of max|F|.
        self.assertLess(np.abs(energy_py - energy_plumed).max(), 2.0e-2)
        self.assertLess(np.abs(forces_py - forces_plumed).max(),
                        0.05 * np.abs(forces_py).max())

    def test_walls_are_not_double_counted(self):
        """Pins trap #2.

        `tica_energy_gradient` adds the walls itself, so the tabulated grid must exclude them
        or every frame outside `bounds` gets them twice. Before the fix, the single test frame
        outside bounds had 25x the force error of the 19 inside.
        """
        stripped = _without_walls(self.bias)
        self.assertTrue(np.all(stripped.wall_k_kcal_mol == 0.0))
        # the original must not be mutated -- it is shared with the Python bias route
        self.assertTrue(np.any(self.bias.wall_k_kcal_mol > 0.0))
        # the attractor payload is set via object.__setattr__, not a dataclass field, so a
        # dataclasses.replace() clone would silently drop it and switch energy branch
        self.assertIsNotNone(getattr(stripped, "attractor_weights", None))

        outside = np.array([self.bias.bounds[0, 1] + 0.5, self.bias.bounds[1, 1] + 0.5])
        with np.errstate(divide="ignore"):
            e_full, _ = self.bias.tica_energy_gradient(outside)
            e_bare, _ = stripped.tica_energy_gradient(outside)
        self.assertGreater(e_full - e_bare, 0.0, "wall term should be positive outside bounds")

    def test_grid_header_matches_the_data_and_covers_the_thermal_excursion(self):
        """Two grid invariants, both of which produced silent wrong answers when broken.

        1. `nbins` must equal the ACTUAL axis length - 1. When the sigma-padding grew the axes
           from 401 to 625 points while the header still said 400, PLUMED reinterpreted the
           file on the wrong stride and the force error jumped 0.027 -> 0.489 with no error
           message.
        2. The grid must extend at least `pad_sigma` thermal excursions `sqrt(kT/wall_k)` past
           the walls, or a normal trajectory walks off it and EXTERNAL aborts mid-run. The old
           fixed 15%-of-range pad gave less than ONE sigma along tic1.
        """
        grid = self.tmp / "hdr.dat"
        xs, ys = write_external_grid(self.bias, grid, n_points=(401, 401),
                                     pad=0.15, pad_sigma=4.0, label="treg")
        header = {}
        for line in grid.read_text().splitlines():
            if not line.startswith("#!"):
                break
            parts = line.split()
            if len(parts) == 4 and parts[1] == "SET":
                header[parts[2]] = parts[3]

        self.assertEqual(int(header["nbins_tic1"]), len(xs) - 1)
        self.assertEqual(int(header["nbins_tic2"]), len(ys) - 1)
        self.assertAlmostEqual(float(header["min_tic1"]), xs[0], places=6)
        self.assertAlmostEqual(float(header["max_tic2"]), ys[-1], places=6)

        n_rows = sum(1 for ln in grid.read_text().splitlines() if not ln.startswith("#"))
        self.assertEqual(n_rows, len(xs) * len(ys))

        sigma = np.sqrt(float(self.bias.kbt_kcal_mol)
                        / np.asarray(self.bias.wall_k_kcal_mol, dtype=float))
        for axis, values in enumerate((xs, ys)):
            lo, hi = self.bias.bounds[axis]
            self.assertLessEqual(values[0], lo - 4.0 * sigma[axis] + 1e-9)
            self.assertGreaterEqual(values[-1], hi + 4.0 * sigma[axis] - 1e-9)

    def test_vectorised_tabulation_matches_the_scalar_oracle(self):
        """The grid writer uses a vectorised KDE; it must equal tica_energy_gradient."""
        stripped = _without_walls(self.bias)
        xs = np.linspace(float(self.bias.bounds[0, 0]), float(self.bias.bounds[0, 1]), 40)
        y = float(np.mean(self.bias.bounds[1]))
        with np.errstate(divide="ignore"):
            energy_vec, grad_vec = _row_energy_gradient(stripped, xs, y)
            for i, x in enumerate(xs):
                energy_ref, grad_ref = stripped.tica_energy_gradient(np.array([x, y]))
                self.assertAlmostEqual(float(energy_vec[i]), float(energy_ref), places=12)
                np.testing.assert_allclose(grad_vec[i], grad_ref, atol=1e-12)


class BiasBackendSelectionTests(unittest.TestCase):
    """`bias_backend` must refuse loudly rather than silently dropping a bias it cannot express."""

    def test_native_backend_rejects_terms_with_no_plumed_equivalent(self):
        from sampling.cases import NATIVE_BIAS_TYPES, native_backend_supported

        self.assertEqual(native_backend_supported(
            {"biases": [{"type": "tica_regional"}, {"type": "tica_metad"}]}), (True, []))

        ok, blockers = native_backend_supported(
            {"biases": [{"type": "tica_metad"}, {"type": "mlcg_teacher"}]})
        self.assertFalse(ok)
        self.assertEqual(blockers, ["mlcg_teacher"])

        # chi has no PLUMED CV -- see the module docstring for why it is not ported
        ok, blockers = native_backend_supported(
            {"biases": [{"type": "local_inversion_umbrella"}]})
        self.assertFalse(ok)
        self.assertEqual(blockers, ["local_inversion_umbrella"])

        # a disabled term must not block the native route
        self.assertTrue(native_backend_supported(
            {"biases": [{"type": "tica_metad"},
                        {"type": "mlcg_teacher", "enabled": False}]})[0])

        self.assertNotIn("local_inversion_umbrella", NATIVE_BIAS_TYPES)
        self.assertNotIn("mlcg_teacher", NATIVE_BIAS_TYPES)

    @unittest.skipUnless(BIAS_NPZ.exists(), "bias artifact absent")
    def test_native_case_emits_no_server_config_and_no_plugin_load(self):
        import tempfile

        import yaml

        from sampling.cases import build_case

        cfg = yaml.safe_load(Path("sampling/campaigns/ala2_bb6_allcorridor_metad.yaml").read_text())
        cfg["bias_backend"] = "plumed"
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "replica_00"
            build_case(case, 0, cfg)
            plumed = (case / "plumed.dat").read_text()
            run = (case / "run_case.sh").read_text()

            self.assertFalse((case / "server.yaml").exists(),
                             "native route must not emit a bias-server config")
            self.assertNotIn("CGBias.so", plumed)
            self.assertNotIn("CG_BIAS", plumed)
            self.assertNotIn("sampling.server", run)
            # and it must still produce the mandatory bias-free rerun
            self.assertIn("unbiased_forces", run)
            self.assertIn("UNITS LENGTH=A ENERGY=kcal/mol", plumed)
            self.assertIn("METAD", plumed)
            self.assertIn("EXTERNAL", plumed)

    def test_python_backend_is_the_default_and_still_emits_the_server(self):
        import tempfile

        import yaml

        from sampling.cases import build_case

        cfg = yaml.safe_load(Path("sampling/campaigns/ala2_bb6_allcorridor_metad.yaml").read_text())
        cfg.pop("bias_backend", None)
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "replica_00"
            build_case(case, 0, cfg)
            self.assertTrue((case / "server.yaml").exists())
            self.assertIn("CG_BIAS", (case / "plumed.dat").read_text())


if __name__ == "__main__":
    unittest.main()
