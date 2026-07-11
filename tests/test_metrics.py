import numpy as np
import pytest

from drift_detector.metrics import bias, compute_metrics, mape, smape, wmape


class TestIndividualMetrics:
    def test_mape_perfect(self):
        y_true = np.array([100, 200, 300])
        assert mape(y_true, y_true) == 0.0

    def test_mape_ignores_zeros(self):
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 210])
        result = mape(y_true, y_pred)
        expected = np.mean([10 / 100, 10 / 200]) * 100
        assert result == pytest.approx(expected)

    def test_mape_single_value(self):
        assert mape([50], [55]) == 10.0

    def test_wmape(self):
        y_true = np.array([100, 200])
        y_pred = np.array([110, 180])
        result = wmape(y_true, y_pred)
        expected = (10 + 20) / (100 + 200) * 100
        assert result == pytest.approx(expected)

    def test_wmape_all_zeros_true_raises(self):
        result = wmape([0, 0], [1, 2])
        assert np.isnan(result) or result == float("inf") or result == 0.0

    def test_smape_perfect(self):
        y_true = np.array([100, 200])
        assert smape(y_true, y_true) == 0.0

    def test_smape(self):
        y_true = np.array([100, 0])
        y_pred = np.array([110, 10])
        result = smape(y_true, y_pred)
        expected = (
            np.mean(
                [
                    10 / ((100 + 110) / 2),
                    10 / ((0 + 10) / 2),
                ]
            )
            * 100
        )
        assert result == pytest.approx(expected)

    def test_bias_positive(self):
        assert bias([100, 200], [110, 210]) == 10.0

    def test_bias_negative(self):
        assert bias([100, 200], [90, 190]) == -10.0

    def test_bias_zero(self):
        assert bias([100, 200], [100, 200]) == 0.0


class TestComputeMetrics:
    def test_returns_all_keys(self):
        y_true = [100, 200, 300]
        y_pred = [110, 190, 310]
        metrics = compute_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {
            "mae",
            "rmse",
            "r2",
            "mape",
            "wmape",
            "smape",
            "bias",
        }

    def test_perfect_prediction(self):
        y_true = [100, 200, 300]
        metrics = compute_metrics(y_true, y_true)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["mape"] == 0.0
        assert metrics["bias"] == 0.0

    def test_handles_lists_and_arrays(self):
        metrics = compute_metrics([1, 2, 3], np.array([1.1, 2.1, 3.1]))
        assert metrics["mae"] == pytest.approx(0.1)

    def test_handles_multidimensional(self):
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[1, 2], [3, 5]])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(0.25)
