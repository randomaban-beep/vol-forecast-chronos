"""Stage 5 calendar features and chronological subset safeguards."""
import numpy as np
import pandas as pd
import pytest
from src.ablations import WEEKDAYS, weekday_features, earliest_fraction, select_stage4_headlines
from src.regimes import conditional_statistics, regime_thresholds
from src.baselines import FEATURES, fit_har
from src.data import forecast_examples
from src.zero_shot import build_contexts


def test_weekdays_use_known_calendar_only():
    rows = pd.DataFrame({'origin': pd.to_datetime(['2024-01-07'] * 3, utc=True),
                         'horizon': [1, 5, 22], 'target': [1., 2., 3.]})
    features = weekday_features(rows)
    assert features.loc[0, WEEKDAYS].sum() == 0  # Monday target
    assert features.loc[1:, 'weekday_6'].eq(1).all()  # Sunday origin
    changed = weekday_features(rows.assign(target=1000., target_end=pd.Timestamp('2099-01-01')))
    pd.testing.assert_frame_equal(features[WEEKDAYS], changed[WEEKDAYS])


def test_fractions_are_nested_earliest_and_preserve_full_order():
    rows = pd.DataFrame({'origin': pd.to_datetime(['2021-01-03', '2021-01-01', '2021-01-02', '2021-01-01']),
                         'symbol': ['A', 'B', 'A', 'A'], 'split': 'train'})
    assert earliest_fraction(rows, .25).tolist() == [3]
    assert earliest_fraction(rows, .5).tolist() == [1, 3]
    np.testing.assert_array_equal(earliest_fraction(rows, 1), np.arange(4))
    with pytest.raises(ValueError):
        earliest_fraction(rows.assign(split='validation'), .5)
    with pytest.raises(ValueError):
        earliest_fraction(rows, 0)


def test_selection_ignores_test_and_heldout():
    rows = pd.DataFrame([dict(model=f'{family}-seed{seed}', horizon=1, split=split,
        symbol=symbol, target=0., prediction=error)
        for family, error in [('A', 1.), ('B', 2.)] for seed in [42, 43, 44]
        for split in ['validation', 'test'] for symbol in ['BTC', 'SOL']])
    expected = select_stage4_headlines(rows, ['SOL'])
    rows.loc[rows['split'].eq('test') | rows.symbol.eq('SOL'), 'prediction'] = -1e8
    pd.testing.assert_frame_equal(expected, select_stage4_headlines(rows, ['SOL']))
    assert expected.loc[expected.selected, 'family'].tolist() == ['A']


def test_regime_threshold_uses_training_only():
    rows = pd.DataFrame({'symbol': ['BTC']*4, 'horizon': [1]*4,
        'split': ['train', 'train', 'validation', 'test'], 'target': np.log([1., 4., 100., 10000.]),
        'target_end': pd.to_datetime(['2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01'], utc=True)})
    assert regime_thresholds(rows).threshold.item() == pytest.approx(1.9)


def test_conditional_bootstrap_preserves_empty_calendar_dates():
    counts = np.array([1., 0., 2., 0.])
    sums = np.tile(counts[None, :, None] * np.array([4., 2., 3., 1.]), (3, 1, 1))
    mean, ci = conditional_statistics(sums, counts, {'seed': 42, 'block_length': 2, 'bootstrap_reps': 100})
    np.testing.assert_allclose(mean, [2., 2., 3., 1.])
    np.testing.assert_allclose(ci, [[2., 2., 3., 1.]]*2)


@pytest.mark.parametrize('horizon', [1, 5, 22])
def test_weekday_fit_excludes_future_targets_and_heldout(horizon):
    dates = pd.date_range('2023-01-01', periods=400, tz='UTC')
    rv = np.exp(np.random.default_rng(42).normal(size=len(dates)))
    daily = pd.DataFrame(dict(date=dates, symbol='BTCUSDT', rv=rv, y=np.log(rv)))
    rows = weekday_features(forecast_examples(pd.concat([daily, daily.assign(symbol='SOLUSDT')]), horizon))
    cutoff = pd.Timestamp('2023-09-01', tz='UTC')
    expected = fit_har(rows, cutoff, horizon, ['SOLUSDT'], FEATURES+WEEKDAYS)
    rows.loc[rows.target_end.gt(cutoff) | rows.symbol.eq('SOLUSDT'), 'target'] = 1e10
    np.testing.assert_array_equal(expected, fit_har(rows, cutoff, horizon, ['SOLUSDT'], FEATURES+WEEKDAYS))


@pytest.mark.parametrize('length', [32, 64, 128, 256])
def test_short_context_is_causal_trailing_slice(length):
    dates = pd.date_range('2023-01-01', periods=400, tz='UTC')
    daily = pd.DataFrame(dict(date=dates, symbol='BTC', y=np.arange(400, dtype=float)))
    daily.loc[280, 'y'] = np.nan
    rows = pd.DataFrame(dict(symbol=['BTC'], origin=[dates[300]], split=['validation']))
    _, full = build_contexts(daily, rows)
    _, short = build_contexts(daily, rows, length)
    np.testing.assert_array_equal(short, full[:, -length:])
    daily.loc[daily.date.gt(dates[300]), 'y'] = 1e10
    np.testing.assert_array_equal(short, build_contexts(daily, rows, length)[1])
