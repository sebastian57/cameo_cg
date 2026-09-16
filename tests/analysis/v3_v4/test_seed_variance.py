import importlib

import numpy as np
import pytest


seedvar = importlib.import_module("05_seed_variance")


def test_delta_is_alphaR_minus_beta_energy_offset():
    energies = np.array([1.0, 3.0, 2.0, 5.0])
    masks = {
        "beta": np.array([True, False, True, False]),
        "alphaR": np.array([False, True, False, True]),
    }
    assert seedvar.delta_from_energies(energies, np.zeros(4), masks)["delta"] == pytest.approx(2.5)


def test_seed_manifest_preserves_model_names():
    manifest = [{"name": "seed_a"}, {"name": "seed_b"}]
    reference = np.zeros((4, 1, 3))
    result = seedvar.evaluate_seed_manifest(
        manifest,
        reference,
        n_frames=4,
        n_boot=20,
        seed=8,
        energy_evaluator=lambda entry, R: np.arange(len(R), dtype=float)
        + (0.0 if entry["name"] == "seed_a" else 1.0),
        baseline_energies=np.zeros(4),
    )
    assert [row["name"] for row in result["models"]] == ["seed_a", "seed_b"]
