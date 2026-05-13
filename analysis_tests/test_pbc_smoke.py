"""Smoke test: verify PBC training runs end-to-end without errors.

Creates a tiny synthetic NPZ with a 'box' key and runs 2 epochs of
train.main().  Asserts that the run completes without exceptions and
that output files (params pickle + MLIR) are produced.

Run directly:
    cd cameo_cg && python analysis_tests/test_pbc_smoke.py

Or via pytest:
    pytest analysis_tests/test_pbc_smoke.py -v
"""

import pathlib
import subprocess
import sys
import tempfile

import numpy as np

# Ensure project root is on path when run directly.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPTIMIZER_BLOCK = """\
optimizer:
  grad_clip: 2.0
  adabelief:
    lr: 0.001
    peak_lr: 0.003
    end_lr: 0.001
    warmup_steps: 2
    decay_steps: 10
    beta1: 0.95
    beta2: 0.999
    eps: 1.0e-8
    grad_clip: 5.0
    weight_decay: 0.0
"""

_ALLEGRO_BLOCK = """\
  allegro:
    num_layers: 1
    max_ell: 1
    mlp_n_hidden: 16
    mlp_n_layers: 1
    embed_n_hidden: [16]
    n_radial_basis: 4
    avg_num_neighbors: 5.0
"""

_TRAINING_BLOCK = """\
training:
  stages:
    - optimizer: adabelief
      epochs: 2
  val_fraction: 0.2
  batch_per_device: 2
  batch_cache: 2
  checkpoint_freq: 0
  gammas:
    F: 1.0
    U: 0.0
"""


def _make_synthetic_pbc_npz(path: pathlib.Path, N: int = 20, T: int = 12,
                              box=(50.0, 50.0, 50.0)) -> None:
    """Write a tiny PBC dataset: random coords wrapped into [0, box], random forces."""
    rng = np.random.RandomState(0)
    box_arr = np.array(box, dtype=np.float32)
    R = (rng.uniform(0.0, 1.0, (T, N, 3)) * box_arr[None, None, :]).astype(np.float32)
    F = rng.randn(T, N, 3).astype(np.float32)
    mask = np.ones((T, N), dtype=np.float32)
    species = np.zeros((T, N), dtype=np.int32)
    np.savez(str(path), R=R, F=F, mask=mask, species=species, box=box_arr)


def _make_pbc_config(npz_path: pathlib.Path, export_dir: pathlib.Path) -> str:
    return (
        f"seed: 0\n"
        f"model_context: test\n"
        f"model_id: pbc_smoke\n\n"
        f"paths:\n  output_dir: {export_dir}\n\n"
        f"data:\n  path: {npz_path}\n  val_fraction: 0.2\n  max_frames: ~\n\n"
        f"model:\n"
        f"  ml_model: allegro_cueq_fast\n"
        f"  pbc: true\n"
        f"  cutoff: 8.0\n"
        f"  dr_threshold: 0.5\n"
        f"  neighbor_list_format: dense\n"
        f"  use_priors: false\n"
        + _ALLEGRO_BLOCK
        + "\n"
        + _TRAINING_BLOCK
        + "\n"
        + _OPTIMIZER_BLOCK
    )


def _make_free_config(npz_path: pathlib.Path, export_dir: pathlib.Path) -> str:
    return (
        f"seed: 0\n"
        f"model_context: test\n"
        f"model_id: free_smoke\n\n"
        f"paths:\n  output_dir: {export_dir}\n\n"
        f"data:\n  path: {npz_path}\n  val_fraction: 0.2\n  max_frames: ~\n\n"
        f"model:\n"
        f"  ml_model: allegro_cueq_fast\n"
        f"  pbc: false\n"
        f"  cutoff: 8.0\n"
        f"  dr_threshold: 0.5\n"
        f"  neighbor_list_format: dense\n"
        f"  use_priors: false\n"
        + _ALLEGRO_BLOCK
        + "\n"
        + _TRAINING_BLOCK
        + "\n"
        + _OPTIMIZER_BLOCK
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pbc_smoke_run():
    """2-epoch PBC training run; verify no exceptions and output files exist."""
    with tempfile.TemporaryDirectory(prefix="pbc_smoke_") as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        npz_path = tmpdir / "pbc_data.npz"
        config_path = tmpdir / "pbc_config.yaml"
        export_dir = tmpdir / "export"
        export_dir.mkdir()

        _make_synthetic_pbc_npz(npz_path)
        config_path.write_text(_make_pbc_config(npz_path, export_dir))

        from scripts.train import main
        main(str(config_path), job_id="pbc_smoke_test")

        params_files = list(export_dir.rglob("*_params.pkl"))
        assert params_files, f"No params pickle found under {export_dir}"
        mlir_files = list(export_dir.rglob("*.mlir"))
        assert mlir_files, f"No MLIR export found under {export_dir}"

        print(f"  params : {params_files[0].name}")
        print(f"  mlir   : {mlir_files[0].name}")


def test_free_smoke_run():
    """Regression: pbc=false path must still work after the PBC changes.

    Runs in a subprocess to avoid chemtrain global symbol state collisions
    when calling main() more than once in the same process.
    """
    with tempfile.TemporaryDirectory(prefix="free_smoke_") as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        npz_path = tmpdir / "free_data.npz"
        config_path = tmpdir / "free_config.yaml"
        export_dir = tmpdir / "export"
        export_dir.mkdir()

        _make_synthetic_pbc_npz(npz_path)
        config_path.write_text(_make_free_config(npz_path, export_dir))

        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, '{_REPO_ROOT}'); "
             f"from scripts.train import main; main('{config_path}', job_id='free_smoke_test')"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr[-3000:])
            raise AssertionError(f"free-space smoke run failed (rc={result.returncode})")

        params_files = list(export_dir.rglob("*_params.pkl"))
        assert params_files, f"No params pickle found under {export_dir}"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running PBC smoke test …")
    test_pbc_smoke_run()
    print("  PASSED: PBC run\n")

    print("Running free-space regression smoke test …")
    test_free_smoke_run()
    print("  PASSED: free-space run\n")

    print("All PBC smoke tests PASSED.")
