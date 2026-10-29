"""Known-input metric checks and shared scoring invariants."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from src.evaluate import (rmse, mae, qlike, smearing_factor, score_forecasts,
                          diebold_mariano, block_bootstrap_ci, evaluate_panel,
                          forecast_bias, mean_directional_accuracy, mdape,
                          pinball_loss, mincer_zarnowitz)


def test_rmse_known() -> None:
    assert rmse(np.array([0, 0]), np.array([3, 4])) == pytest.approx(np.sqrt(12.5))


def test_mae_known() -> None:
    assert mae(np.array([0, 0]), np.array([-3, 4])) == 3.5


def test_forecast_bias_known() -> None:
    actual = np.array([1.0, 2.0])
    forecast = np.array([2.0, 4.0])
    assert forecast_bias(actual, forecast) == pytest.approx(1.5)


def test_mean_directional_accuracy_known() -> None:
    actual_change = np.array([1.0, -2.0, 0.5, -0.5])
    forecast_change = np.array([2.0, -1.0, -0.1, -1.0])
    # signs: act: [+, -, +, -], fcast: [+, -, -, -] -> 3 out of 4 match
    assert mean_directional_accuracy(actual_change, forecast_change) == pytest.approx(0.75)


def test_mdape_known() -> None:
    actual_variance = np.array([1.0, 2.0, 4.0])
    forecast_variance = np.array([1.2, 1.8, 3.0])
    # rel errors: [0.20, 0.10, 0.25], median is 0.20
    assert mdape(actual_variance, forecast_variance) == pytest.approx(0.20)


def test_pinball_loss_known() -> None:
    actual = np.array([2.0, 1.0])
    forecast = np.array([1.0, 2.0])
    # e = [1.0, -1.0]. For tau=0.9: 0.9 * 1.0 = 0.9, -0.1 * -1.0 = 0.1. Mean = 0.5
    assert pinball_loss(actual, forecast, quantile=0.90) == pytest.approx(0.5)


def test_mincer_zarnowitz_known() -> None:
    forecast = np.array([1.0, 2.0, 3.0, 4.0])
    actual = 2.0 * forecast + 1.0
    res = mincer_zarnowitz(actual, forecast)
    assert res["alpha"] == pytest.approx(1.0)
    assert res["beta"] == pytest.approx(2.0)
    assert res["r2_mz"] == pytest.approx(1.0)


def test_qlike_known_orientation() -> None:
    np.testing.assert_allclose(qlike(np.array([1, 2, 4]), np.array([1, 4, 2])),
                               [0, 1 - np.log(2), -0.5 + np.log(2)])


def test_smearing_and_shared_scoring() -> None:
    actual, prediction = np.log([1, 3]), np.zeros(2)
    factor = smearing_factor(actual, prediction)
    assert factor == pytest.approx(2)
    scores = score_forecasts(actual, prediction, factor)
    np.testing.assert_allclose(scores.forecast_variance, [2, 2])
    np.testing.assert_allclose(scores.qlike, qlike(np.array([1, 3]), np.array([2, 2])))
    np.testing.assert_allclose(scores.qlike_raw, qlike(np.array([1, 3]), np.ones(2)))
    assert np.sqrt(scores.squared_error.mean()) == rmse(actual, prediction)
    assert scores.absolute_error.mean() == mae(actual, prediction)
    # Calibration never changes log-scale error and is applied exactly once.
    np.testing.assert_array_equal(scores.squared_error, score_forecasts(actual, prediction, 1).squared_error)


def test_dm_known_hac_and_harvey() -> None:
    result = diebold_mariano(np.array([1, 2, 3, 4]), np.zeros(4), 2)
    # dbar=2.5, gamma0=1.25, gamma1=0.3125, Bartlett lag1 weight=1/2.
    expected = 2.5 / np.sqrt(1.5625 / 4) * np.sqrt(0.375)
    assert result["dm_hac_lags"] == 1
    assert result["dm_stat"] == pytest.approx(expected)
    assert result["dm_pvalue"] == pytest.approx(2 * t.sf(expected, 3))
    reverse = diebold_mariano(np.zeros(4), np.array([1, 2, 3, 4]), 2)
    assert reverse["dm_stat"] == pytest.approx(-expected)


def test_dm_identical_and_constant_difference() -> None:
    assert diebold_mariano(np.ones(50), np.ones(50), 22)["dm_pvalue"] == 1
    assert diebold_mariano(np.ones(50), np.zeros(50), 22)["dm_pvalue"] == 0


def test_block_bootstrap_known_constant() -> None:
    values = np.tile([4, 3, 2, 1], (60, 1))
    np.testing.assert_allclose(block_bootstrap_ci(values, 30, 100, 42), [[2, 3, 2, 1]] * 2)


def test_block_bootstrap_reproducible() -> None:
    values = np.arange(180).reshape(60, 3)
    first = block_bootstrap_ci(values, 10, 100, 42)
    np.testing.assert_array_equal(first, block_bootstrap_ci(values, 10, 100, 42))
    assert (first[0] < first[1]).all()


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_variance_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        qlike(np.array([1]), np.array([bad]))
    with pytest.raises(ValueError):
        mdape(np.array([1]), np.array([bad]))


def test_invalid_quantile_rejected() -> None:
    with pytest.raises(ValueError):
        pinball_loss(np.array([1.0]), np.array([1.0]), quantile=0.0)
    with pytest.raises(ValueError):
        pinball_loss(np.array([1.0]), np.array([1.0]), quantile=1.0)


def test_invalid_metric_shapes_rejected() -> None:
    with pytest.raises(ValueError):
        rmse(np.array([1, 2]), np.array([1]))
    with pytest.raises(ValueError):
        mae(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(3), np.zeros(3), 5)


def test_calibration_excludes_test_and_heldout(monkeypatch: pytest.MonkeyPatch) -> None:
    import arch.bootstrap

    class DummyMCS:
        def __init__(self, losses: pd.DataFrame, **kwargs: object) -> None:
            self.pvalues = pd.DataFrame({"pvalue": 1.0}, index=losses.columns)

        def compute(self) -> None:
            pass

    monkeypatch.setattr(arch.bootstrap, "MCS", DummyMCS)
    rows = []
    for split, start in [("validation", "2024-01-01"), ("test", "2025-01-01")]:
        for symbol in ["BTCUSDT", "SOLUSDT"]:
            for date in pd.date_range(start, periods=35, tz="UTC"):
                rows.append({"split": split, "symbol": symbol, "origin": date - pd.Timedelta(days=1),
                             "target_end": date, "horizon": 1, "model": "HAR-RV",
                             "target": np.log(2) if symbol == "BTCUSDT" and split == "validation" else 0.,
                             "prediction": 0.})
    predictions = pd.DataFrame(rows)
    config = {"held_out": ["SOLUSDT"], "bootstrap_reps": 20, "block_length": 5, "seed": 42}
    _, factors, _ = evaluate_panel(predictions, config)
    changed = predictions.copy()
    changed.loc[changed.symbol.eq("SOLUSDT") | changed["split"].eq("test"), "target"] = np.log(100)
    _, after, _ = evaluate_panel(changed, config)
    pd.testing.assert_frame_equal(factors, after)
    assert factors.smearing_factor.iloc[0] == pytest.approx(2)
    assert factors.n_validation.iloc[0] == 35


def test_incomplete_model_panel_rejected() -> None:
    rows = pd.DataFrame({"split": ["test"] * 3, "symbol": ["BTCUSDT"] * 3,
                         "origin": [1, 1, 2], "horizon": [1] * 3,
                         "model": ["HAR-RV", "Naive", "HAR-RV"]})
    with pytest.raises(ValueError, match="comparison rows"):
        evaluate_panel(rows, {})
