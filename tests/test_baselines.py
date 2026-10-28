"""Causal fit cutoffs, GARCH arithmetic and independent arch recursion checks."""

import numpy as np
import pandas as pd
import pytest
from arch.univariate import GARCH, StudentsT

from src.baselines import fit_har, fit_garch, variance_path, garch_forecasts, FEATURES, RV_FEATURES
from src.data import daily_returns, forecast_examples


def test_returns_require_consecutive_valid_days() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-06"], utc=True)
    daily = pd.DataFrame({"symbol": "X", "date": dates, "close": [100, 110, np.nan, 130, 140],
                          "reason": ["ok", "ok", "missing_day", "ok", "ok"]})
    out = daily_returns(daily)
    assert out["return"].iloc[1] == pytest.approx(np.log(1.1))
    assert out["return"].iloc[[0, 2, 3, 4]].isna().all()
    changed = daily.copy()
    changed.loc[4, "close"] = 1000
    pd.testing.assert_series_equal(out["return"].iloc[:4], daily_returns(changed)["return"].iloc[:4])


def test_garch_missing_day_and_horizon_arithmetic() -> None:
    p = np.array([0.2, 0.1, 0.2, 0.6, 8])
    np.testing.assert_allclose(variance_path(np.array([-2., np.nan, 1]), p), [1, 2, 1.8, 1.38])
    r = pd.Series([-.02, np.nan, .01], index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
    out = garch_forecasts(r, p, [1, 5])
    assert np.exp(out[1].iloc[-1]) == pytest.approx(1.38 / 10000)
    steps = [1.38]
    for _ in range(4):
        steps.append(.2 + .8 * steps[-1])
    assert np.exp(out[5].iloc[-1]) == pytest.approx(np.mean(steps) / 10000)


def test_recursion_matches_arch_on_complete_data() -> None:
    p = np.array([0.2, 0.1, 0.2, 0.6, 8])
    residuals = np.array([-2., 1., -0.5, 2., 0.25])
    expected = np.empty(len(residuals))
    GARCH(p=1, o=1, q=1, power=2).compute_variance(
        p[:4], residuals, expected, 1.0, np.tile([1e-10, 1e10], (len(residuals), 1)))
    np.testing.assert_allclose(variance_path(residuals, p)[:-1], expected)


def test_garch_forecast_future_invariance() -> None:
    p = np.array([0.2, 0.1, 0.2, 0.6, 8])
    r = pd.Series([.01, -.02, .01, .03])
    before = garch_forecasts(r, p, [1, 5, 22])
    r.iloc[2:] = 1
    pd.testing.assert_frame_equal(before.iloc[:2], garch_forecasts(r, p, [1, 5, 22]).iloc[:2])


@pytest.mark.parametrize("features", [FEATURES, RV_FEATURES])
def test_har_excludes_future_and_heldout(features: list[str]) -> None:
    rng = np.random.default_rng(42)
    daily = pd.DataFrame({"symbol": "BTCUSDT", "date": pd.date_range("2023-01-01", periods=400, tz="UTC"),
                          "rv": np.exp(rng.normal(size=400))})
    daily["y"] = np.log(daily.rv)
    held = daily.assign(symbol="SOLUSDT", rv=daily.rv * 100, y=daily.y + np.log(100))
    examples = forecast_examples(pd.concat([daily, held]), 5)
    cutoff = pd.Timestamp("2023-08-01", tz="UTC")
    expected = fit_har(examples, cutoff, 5, ["SOLUSDT"], features)
    changed = examples.copy()
    changed.loc[(changed.target_end > cutoff) | changed.symbol.eq("SOLUSDT"), "target"] = 10000
    np.testing.assert_array_equal(expected, fit_har(changed, cutoff, 5, ["SOLUSDT"], features))


def test_har_feature_variants_differ() -> None:
    dates = pd.date_range("2024-01-01", periods=80, tz="UTC")
    rv = np.tile([1., 9.], 40)
    examples = forecast_examples(pd.DataFrame({"date": dates, "symbol": "X", "rv": rv, "y": np.log(rv)}), 5)
    assert (examples.weekly_rv > examples.weekly).all()
    assert (examples.monthly_rv > examples.monthly).all()


@pytest.mark.parametrize("asymmetric", [False, True])
def test_student_t_fit_converges_and_is_stationary(asymmetric: bool) -> None:
    rng = np.random.default_rng(42)
    r = pd.DataFrame({"X": rng.standard_t(8, 350) / 100, "Y": rng.standard_t(8, 350) / 100})
    r.iloc[50:52, 0] = np.nan
    fitted = fit_garch(r, asymmetric)
    p = fitted.parameters
    assert fitted.observations == 698
    assert p[1] + p[2] / 2 + p[3] < 1
    assert p[1] + p[2] >= -1e-8 and p[4] > 2
    if not asymmetric:
        assert p[2] == 0
    assert np.isfinite(StudentsT().loglikelihood([p[4]], np.array([1.]), np.array([1.])))


def test_walk_forward_schedule_and_fit_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.baselines as module
    dates = pd.date_range("2023-01-01", "2025-02-15", tz="UTC")
    rv = 0.01 + np.arange(len(dates)) / 100000
    source = pd.DataFrame({"symbol": "BTCUSDT", "date": dates, "rv": rv, "y": np.log(rv), "return": 0.01})
    daily = pd.concat([source, source.assign(symbol="SOLUSDT", **{"return": 0.99})])
    examples = pd.concat([forecast_examples(daily, h) for h in [1, 5, 22]], ignore_index=True)
    seen = []

    def fake_fit(returns: pd.DataFrame, asymmetric: bool, maxiter: int,
                 previous: np.ndarray | None = None) -> module.GarchFit:
        assert list(returns.columns) == ["BTCUSDT"]
        assert returns.iloc[-1, 0] == .01
        seen.append(returns.index.max())
        return module.GarchFit(np.array([.2, .1, 0, .7, 8]), 0, len(returns), 1, 1)

    monkeypatch.setattr(module, "fit_garch", fake_fit)
    config = {"held_out": ["SOLUSDT"], "horizons": [1, 5, 22], "refit_days": 30, "garch_maxiter": 10}
    out, audit = module.walk_forward(daily, examples, config)
    assert (out.refit <= out.origin).all()
    assert ((out.origin - out.refit).dt.days < 30).all()
    for _, rows in audit.groupby("split"):
        cutoffs = rows.refit.drop_duplicates().sort_values()
        assert cutoffs.diff().dropna().eq(pd.Timedelta(days=30)).all()
    assert set(seen) == set(audit.refit)
    assert out.model.nunique() == 5
    assert out.groupby(["split", "symbol", "origin", "horizon"]).model.nunique().eq(5).all()
