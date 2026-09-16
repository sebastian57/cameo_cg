import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.physics.stencil_hessian import (
    build_stencil_pairs,
    compute_measured_hvp,
    evaluate_model_hvp,
    parse_model_spec,
    summarize_hvp,
)
from analysis.sampling.error_map import _jsonable_path
from analysis.sampling.gradient_recovery_error import _jsonable_path as _gradient_jsonable_path


def _synthetic_stencil():
    anchor = np.repeat([101, 203], 5)
    direction = np.tile([-1, 0, 0, 1, 1], 2)
    multiplier = np.tile([0.0, 1.0, -1.0, 1.0, -1.0], 2)
    meanforce = {
        "R": np.zeros((10, 2, 3), dtype=np.float32),
        "F": np.zeros((10, 2, 3), dtype=np.float32),
        "SE": np.ones((10, 2, 3), dtype=np.float32),
    }
    stencil = {"anchor": anchor, "direction": direction, "multiplier": multiplier}
    return meanforce, stencil


def test_stencil_pairs_group_by_anchor_metadata_without_fixed_rows_per_anchor():
    meanforce, stencil = _synthetic_stencil()

    pairs = build_stencil_pairs(meanforce, stencil, layer=1.0)

    np.testing.assert_array_equal(pairs.anchor_ids, [101, 203])
    np.testing.assert_array_equal(pairs.directions, [0, 1])
    np.testing.assert_array_equal(pairs.anchor_rows, [0, 5])
    np.testing.assert_array_equal(pairs.positive_rows, [[1, 3], [6, 8]])
    np.testing.assert_array_equal(pairs.negative_rows, [[2, 4], [7, 9]])


def test_measured_hvp_uses_realized_plus_minus_displacement():
    meanforce, stencil = _synthetic_stencil()
    meanforce["R"][1, 0] = [1.0, 0.0, 0.0]
    meanforce["R"][2, 0] = [-1.0, 0.0, 0.0]
    meanforce["R"][3, 1] = [0.0, 2.0, 0.0]
    meanforce["R"][4, 1] = [0.0, -2.0, 0.0]
    meanforce["F"][1, 0] = [-2.0, 0.0, 0.0]
    meanforce["F"][2, 0] = [2.0, 0.0, 0.0]
    meanforce["F"][3, 1] = [0.0, -4.0, 0.0]
    meanforce["F"][4, 1] = [0.0, 4.0, 0.0]
    pairs = build_stencil_pairs(meanforce, stencil)

    measured = compute_measured_hvp(meanforce, pairs, [0])

    np.testing.assert_allclose(measured.v[0, 0, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(measured.hvp[0, 0, 0], [2.0, 0.0, 0.0])
    assert measured.noise is not None


def test_quadratic_model_hvp_matches_analytic_hessian():
    import jax.numpy as jnp

    class QuadraticModel:
        def compute_energy(self, params, coordinates, mask, species):
            del params, mask, species
            stiffness = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            return 0.5 * jnp.sum(stiffness * coordinates**2)

    r0 = np.zeros((1, 2, 3), dtype=np.float32)
    v = np.asarray(
        [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
          [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]]],
        dtype=np.float32,
    )
    expected = v * np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    predicted = evaluate_model_hvp(
        QuadraticModel(), None, r0, v, np.ones((2,), dtype=np.float32), np.zeros((2,), dtype=np.int32)
    )

    np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-6)


def test_identical_hvp_summary_has_zero_error_and_unit_alignment():
    reference = np.asarray([[[[1.0, 0.0, 0.0]]]], dtype=np.float64)
    summary = summarize_hvp(reference, reference)

    assert summary["cosine_median"] == 1.0
    assert summary["norm_ratio_median"] == 1.0
    assert summary["relative_error_median"] == 0.0


def test_model_spec_parser_keeps_label_and_paths_explicit():
    assert parse_model_spec("arm_a=configs/model.yaml:params.pkl") == (
        "arm_a", "configs/model.yaml", "params.pkl"
    )


def test_error_map_provenance_converts_path_values_for_json():
    payload = {"frames": _jsonable_path(Path("reference.npz"))}

    assert json.loads(json.dumps(payload))["frames"] == "reference.npz"



def test_gradient_recovery_provenance_converts_path_values_for_json():
    payload = {"frames": _gradient_jsonable_path(Path("reference.npz"))}

    assert json.loads(json.dumps(payload))["frames"] == "reference.npz"
