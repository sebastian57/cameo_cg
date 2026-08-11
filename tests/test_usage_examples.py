from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent



def _yaml(name: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / name).read_text())


def _has_path(mapping: dict, dotted: str) -> bool:
    value = mapping
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def test_base_config_is_the_stable_training_registry():
    cfg = _yaml("base_config.yaml")
    required = {
        "run.description", "paths.output_dir", "paths.checkpoint_dir",
        "data.crossfit", "data.static_neighbors", "data.tile_target_edges",
        "model.output_mode", "model.direct_force", "model.allegro",
        "model.allegro_cueq", "model.mace", "model.painn",
        "model.robustness_gate", "model.local_extrapolation_gate",
        "model.edge_distance_gate", "export.enabled",
        "training.force_labels", "training.msam", "training.swa",
        "training.dsm", "training.hvp", "training.safety_regularization",
        "training.prior_residual", "training.noised_residual_training",
        "training.relative_entropy", "training.profiling", "ensemble.enabled",
    }
    assert not sorted(path for path in required if not _has_path(cfg, path))


def test_example_training_stays_a_small_disabled_optional_fm_starter():
    cfg = _yaml("example_training.yaml")
    assert cfg["training"]["gammas"] == {"F": 1.0, "U": 0.0}
    assert cfg["training"]["msam"]["enabled"] is False
    assert cfg["training"]["swa"]["enabled"] is False
    assert cfg["training"]["relative_entropy"]["enabled"] is False
    assert cfg["ensemble"]["enabled"] is False
    assert len((PROJECT_ROOT / "configs" / "example_training.yaml").read_text().splitlines()) < 230


def test_example_md_is_the_stable_md_registry_without_runtime_keys():
    md = _yaml("example_md.yaml")["md"]
    required = {
        "training_config_path", "params_path", "dataset_path", "frame_indices",
        "cell_list", "disable_cell_list", "scan_chunk_size", "integrator",
        "tau", "equilibrate", "zero_com_velocity", "stability_abort",
        "bias", "override_use_priors", "prior_only", "robustness_gate",
        "local_extrapolation_gate", "edge_distance_gate", "h_constraints",
        "force_decomp", "force_decomp_every", "observables_filename",
        "continuous_output", "dump_for_ovito",
    }
    assert not sorted(required - set(md))
    assert "_partial_output_path" not in md


def test_example_md_is_a_safe_smoke_template():
    md = yaml.safe_load((PROJECT_ROOT / "configs" / "example_md.yaml").read_text())["md"]
    assert md["dt"] == 1.0
    assert md["n_steps"] <= 10_000
    assert md["rescale_initial_temperature"] is True
    assert md["initial_temperature_scale"] == 1.0
    assert md["continuous_output"] is True
    assert md["stability_abort"] == {
        "min_pair_distance_A": 0.7,
        "max_force_kcal_per_mol_A": 10_000.0,
        "max_temperature_K": 3_000.0,
    }


def test_lammps_data_generator_is_parameterized(tmp_path: Path):
    dataset = tmp_path / "input.npz"
    output = tmp_path / "frame.data"
    np.savez(
        dataset,
        R=np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [2.0, 3.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        resname=np.asarray(["ALA", "GLY", "ALA"]),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "md_setup" / "lmp_input_gen.py"),
            "--dataset",
            str(dataset),
            "--frame",
            "1",
            "--output",
            str(output),
            "--padding",
            "5",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    generated = output.read_text()
    assert "3 atoms" in generated
    assert "2 atom types" in generated
    assert "2.00000000 1.00000000 0.00000000" in generated


def test_lammps_example_uses_runtime_variables():
    input_text = (PROJECT_ROOT / "md_setup" / "inp_lammps_trained.in").read_text()
    submit_text = (PROJECT_ROOT / "md_setup" / "submit_lammps_chemtrain.sh").read_text()
    for variable in ("data_file", "model_file", "temperature", "run_steps"):
        assert "${" + variable + "}" in input_text
        assert "-var " + variable in submit_text
    assert "4zohB01" not in input_text
    assert "/p/project1" not in submit_text
