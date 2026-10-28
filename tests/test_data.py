"""Synthetic arithmetic, quality-control and temporal-causality tests."""

import numpy as np
import pandas as pd
import pytest

from src.data import COLUMNS, daily_variance, forecast_examples


def minutes(days: int = 1, unit: str = "ms") -> pd.DataFrame:
    """Create continuous exponential prices with known five-minute returns."""
    times = pd.date_range("2024-01-01", periods=days * 1440, freq="min", tz="UTC")
    scale = 10**6 if unit == "ms" else 1000
    opens = np.exp(np.arange(len(times)) * 0.00001)
    closes = opens * np.exp(0.00001)
    return pd.DataFrame(dict(zip(COLUMNS, [times.as_unit("ns").asi8 // scale,
        opens, closes, opens, closes, np.ones(len(times)),
        (times.as_unit("ns").asi8 + 60 * 10**9) // scale - 1,
        np.ones(len(times)), np.ones(len(times)), np.ones(len(times)),
        np.ones(len(times)), np.zeros(len(times))])))


@pytest.mark.parametrize("unit", ["ms", "us"])
def test_known_rv(unit: str) -> None:
    result = daily_variance(minutes(unit=unit), "2024-01-01", "2024-01-01")
    assert result.rv.iloc[0] == pytest.approx(288 * (5 * 0.00001) ** 2)
    assert result.y.iloc[0] == pytest.approx(np.log(result.rv.iloc[0]))


@pytest.mark.parametrize("fault", ["gap", "duplicate", "negative", "infinite", "ohlc", "close_time"])
def test_bad_day_dropped(fault: str) -> None:
    raw = minutes(2)
    if fault == "gap":
        raw = raw.drop(10)
    elif fault == "duplicate":
        raw = pd.concat([raw, raw.iloc[[10]]])
    elif fault == "negative":
        raw.loc[10, "close"] = -1
    elif fault == "infinite":
        raw.loc[10, "close"] = np.inf
    elif fault == "ohlc":
        raw.loc[10, "high"] = 0.5
    else:
        raw.loc[10, "close_time"] += 60000
    result = daily_variance(raw, "2024-01-01", "2024-01-02")
    assert np.isnan(result.y.iloc[0])
    assert np.isfinite(result.y.iloc[1])


def test_unlocatable_timestamp_fails() -> None:
    raw = minutes()
    raw.loc[0, "time"] = np.nan
    with pytest.raises(ValueError, match="Unparseable"):
        daily_variance(raw, "2024-01-01", "2024-01-01")


def test_leakage_daily_future_invariance() -> None:
    raw = minutes(2)
    before = daily_variance(raw, "2024-01-01", "2024-01-02")
    raw.loc[1440:, ["open", "high", "low", "close"]] *= np.exp(
        np.arange(1440) * 0.001)[:, None]
    after = daily_variance(raw, "2024-01-01", "2024-01-02")
    pd.testing.assert_series_equal(before.iloc[0], after.iloc[0])
    assert before.rv.iloc[1] != pytest.approx(after.rv.iloc[1])


@pytest.mark.parametrize("horizon", [1, 5, 22])
def test_leakage_features_and_targets(horizon: int) -> None:
    dates = pd.date_range("2023-10-01", "2025-03-01", tz="UTC")
    rv = 1 + np.arange(len(dates), dtype=float) / 100
    daily = pd.DataFrame({"date": dates, "symbol": "BTCUSDT", "rv": rv, "y": np.log(rv)})
    original = forecast_examples(daily, horizon)
    cutoff = pd.Timestamp("2024-07-01", tz="UTC")
    changed = daily.copy()
    changed.loc[changed.date > cutoff, "rv"] = 10000
    changed.loc[changed.date > cutoff, "y"] = np.log(10000)
    altered = forecast_examples(changed, horizon)
    # Labels indexed by endpoint t and features paired with them never use >t.
    pd.testing.assert_frame_equal(original[original.target_end <= cutoff],
                                  altered[altered.target_end <= cutoff])
    # Stronger: features cannot see anything after forecast origin, even labels.
    features = ["daily", "weekly", "monthly", "weekly_rv", "monthly_rv"]
    pd.testing.assert_frame_equal(original.loc[original.origin <= cutoff, features],
                                  altered.loc[altered.origin <= cutoff, features])
    for row in original.itertuples():
        expected = np.log(daily.loc[daily.date.between(row.target_start, row.target_end), "rv"].mean())
        assert row.target == pytest.approx(expected)
        assert row.origin < row.target_start <= row.target_end
    assert (original.target_start.dt.year == original.target_end.dt.year).all()


def test_calendar_gap_not_compressed() -> None:
    dates = pd.date_range("2024-01-01", periods=80, tz="UTC")
    rv = np.arange(1, 81, dtype=float)
    daily = pd.DataFrame({"date": dates, "symbol": "BTCUSDT", "rv": rv, "y": np.log(rv)})
    daily = daily.drop(40)
    result = forecast_examples(daily, 5)
    gap = dates[40]
    assert not ((result.target_start <= gap) & (result.target_end >= gap)).any()
    assert not ((result.origin - pd.Timedelta(days=21) <= gap) & (result.origin >= gap)).any()


@pytest.mark.parametrize("horizon", [1, 5, 22])
def test_target_is_log_arithmetic_mean_rv(horizon: int) -> None:
    dates = pd.date_range("2024-01-01", periods=70, tz="UTC")
    rv = np.tile([1.0, 9.0], 35)
    daily = pd.DataFrame({"date": dates, "symbol": "BTCUSDT", "rv": rv, "y": np.log(rv)})
    result = forecast_examples(daily, horizon)
    last = result.iloc[-1]
    assert last.target == pytest.approx(np.log(rv[-horizon:].mean()))
    if horizon == 1:
        assert last.target == daily.y.iloc[-1]
    else:
        assert last.target != pytest.approx(np.log(rv[-horizon:]).mean())
