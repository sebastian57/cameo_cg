import importlib

import numpy as np
import pytest


metrics = importlib.import_module("04_matched_model_metrics")


def test_force_metrics_report_rmse_and_noise_normalized_error():
    measured = np.zeros((2, 1, 3))
    predicted = np.ones((2, 1, 3))
    se = np.full_like(measured, 0.5)
    out = metrics.evaluate_force_metrics(predicted, measured, se)
    assert out["rmse"] == pytest.approx(1.0)
    assert out["median_abs_z"] == pytest.approx(2.0)


def test_hvp_metrics_perfect_prediction_has_zero_relative_error():
    x = np.arange(12.0).reshape(2, 2, 3)
    assert metrics.evaluate_hvp_metrics(x, x)["relative_error"] == pytest.approx(0.0)
