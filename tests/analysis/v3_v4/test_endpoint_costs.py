import importlib

import numpy as np
import pytest


endpoint = importlib.import_module("01_endpoint_costs")


def _complete_path():
    states = {
        "state": np.array([0, 1, 2]),
        "R": np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]]),
        "anchor": np.array([0, 0, 0]),
        "direction": np.array([-1, 0, 0]),
        "multiplier": np.array([0.0, 1.0, -1.0]),
        "eps_state": np.array([0.0, 1.0, 1.0]),
        "lam_state": np.array([10.0, 10.0, 10.0]),
        "kind": np.array([0, 1, 1]),
    }
    labels = {
        "state": np.array([0, 1, 2]),
        "R": states["R"].copy(),
        "F": np.array([[[-2.0, 0.0, 0.0]], [[-2.0, 0.0, 0.0]], [[4.4, 0.0, 0.0]]]),
        "SE": np.full((3, 1, 3), 0.1),
    }
    return states, labels


def test_endpoint_summary_reports_linear_asymmetry():
    states, labels = _complete_path()
    result = endpoint.summarize_endpoint_costs(states, labels, n_boot=100, seed=3)
    assert result["n_complete_pairs"] == 1
    assert result["median_asymmetry_kcal_mol"] == pytest.approx(0.8, abs=0.05)


def test_endpoint_summary_excludes_incomplete_pairs():
    states, labels = _complete_path()
    labels = {key: value[[0, 1]] for key, value in labels.items()}
    result = endpoint.summarize_endpoint_costs(states, labels, n_boot=50, seed=3)
    assert result["n_complete_pairs"] == 0
    assert result["n_incomplete_pairs"] == 1
