import numpy as np

from active_learning.state import (
    aggregate_frame_scores,
    build_gram_cholesky,
    feature_moments,
    leverage_scores,
    normalize_features,
    quantile_thresholds,
)


def test_cholesky_leverage_matches_direct_inverse():
    features = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    ridge = 0.25
    chol = build_gram_cholesky(features, ridge)
    got = leverage_scores(features, chol)
    gram = features.T @ features + ridge * np.eye(features.shape[1])
    expected = np.einsum("ij,jk,ik->i", features, np.linalg.inv(gram), features)
    np.testing.assert_allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)


def test_normalization_and_quantile_thresholds_are_deterministic():
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    mean, scale = feature_moments(features)
    norm1 = normalize_features(features, mean, scale)
    norm2 = normalize_features(features, mean, scale)
    np.testing.assert_allclose(norm1, norm2)

    scores = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
    thresholds = quantile_thresholds(scores, warn_q=0.5, select_q=0.8, abort_q=1.0)
    assert thresholds["warn"] == np.quantile(scores, 0.5)
    assert thresholds["select"] == np.quantile(scores, 0.8)
    assert thresholds["abort"] == np.quantile(scores, 1.0)


def test_frame_aggregation_reports_max_and_quantiles():
    scores = np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float64)
    agg = aggregate_frame_scores(scores, q=0.75)
    assert agg["edge_count"] == 4.0
    assert agg["max"] == 100.0
    assert agg["q95"] == np.quantile(scores, 0.95)
    assert agg["q99"] == np.quantile(scores, 0.99)
    assert agg["agg"] == np.quantile(scores, 0.75)
