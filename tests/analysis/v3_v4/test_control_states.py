import importlib

import numpy as np


controls = importlib.import_module("03_control_states")


def test_control_metadata_preserves_anchor_provenance():
    states = {
        "R": np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        "anchor": np.array([0, 0]),
        "direction": np.array([-1, 0]),
        "multiplier": np.array([0.0, 1.0]),
        "anchor_source": np.array([0, 0]),
        "anchor_index": np.array([17, 17]),
    }
    out = controls.build_control_metadata(states, np.array([0]), [0.25, 0.5], [0.04])
    assert {"control_kind", "mode_sign", "path_fraction", "fixed_eps_A"} <= set(out)
    assert np.all(out["anchor_index"] == 17)
    assert np.any(out["mode_sign"] > 0)
    assert np.any(out["mode_sign"] < 0)
