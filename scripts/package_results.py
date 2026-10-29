"""Export small reporting-only CSVs from cached forecasts; never retrain or rewrite old results."""
from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.report import headline_rows, RESULTS


def excluded_windows(rows: pd.DataFrame) -> pd.Series:
    """Exclude every target window intersecting the three UTC shock dates."""
    start = pd.Timestamp('2025-10-10', tz='UTC')
    end = pd.Timestamp('2025-10-12', tz='UTC')
    return (rows.origin + pd.Timedelta(days=1)).le(end) & rows.target_end.ge(start)


def main() -> None:
    """Export scores, illustrative BTC weeks and measured cost components."""
    old = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in RESULTS.iterdir()
           if p.is_file() and not p.name.startswith('stage6_')}
    source = Path('data/stage5/scored.parquet')
    if not source.exists():
        raise SystemExit('Cached scored forecasts required for the new rescore. Figure-only reproduction uses committed CSVs.')
    scored = pd.read_parquet(source)
    scored['family'] = scored.model.str.replace(r'-seed\d+$', '', regex=True)
    head = headline_rows()
    output = []
    for row in head.itertuples():
        subset = scored.loc[scored.family.eq(row.family) & scored.horizon.eq(row.horizon) & scored.symbol_group.eq(row.symbol_group)]
        assert not subset.empty
        for label, data in [('all dates', subset), ('exclude Oct 10-12', subset.loc[~excluded_windows(subset)])]:
            per_seed = data.groupby('model')[['squared_error','absolute_error','qlike_raw','qlike']].mean()
            per_seed['squared_error'] = np.sqrt(per_seed.squared_error)
            means = per_seed.mean()
            output.append(dict(model=row.model, family=row.family, symbol_group=row.symbol_group, horizon=row.horizon,
                sample=label, n_dates=data.origin.nunique(), examples_per_seed=len(data)//len(per_seed),
                rmse=means.squared_error, mae=means.absolute_error, qlike_raw=means.qlike_raw, qlike=means.qlike))
    robust = pd.DataFrame(output)
    for metric in ['qlike_raw','qlike']:
        robust['rank_'+metric] = robust.groupby(['symbol_group','horizon','sample'])[metric].rank(method='min').astype(int)
    robust.to_csv(RESULTS/'stage6_robustness.csv', index=False)
    # Existing scored h=1 predictions provide a small derived fan-chart dataset.
    family = head.loc[head.model.eq('Fine-tuned') & head.horizon.eq(1), 'family'].iloc[0]
    btc = scored.loc[scored.symbol.eq('BTCUSDT') & scored.horizon.eq(1)]
    har = btc.loc[btc.model.eq('HAR-weekday')].set_index('target_end')
    ft = btc.loc[btc.family.eq(family)].copy()
    ft['rv_forecast'] = np.exp(ft.prediction)
    fan = ft.groupby('target_end').rv_forecast.agg(ft_median='median', ft_low='min', ft_high='max')
    fan['realized_rv'] = np.exp(har.target)
    fan['har_weekday'] = np.exp(har.prediction)
    fan['week_start'] = fan.index - pd.to_timedelta(fan.index.dayofweek, unit='D')
    weeks = fan.groupby('week_start').realized_rv.agg(['mean','count'])
    calm = weeks.loc[weeks['count'].eq(7), 'mean'].idxmin()
    volatile = pd.Timestamp('2025-10-06', tz='UTC')
    clips = []
    for label, week in [('Calm week', calm), ('Shock week', volatile)]:
        clips.append(fan.loc[fan.week_start.eq(week)].reset_index().assign(week=label))
    pd.concat(clips).to_csv(RESULTS/'stage6_forecast_weeks.csv', index=False)
    runtime = []
    runtime.append(dict(stage=1, seconds=np.nan, component='Not instrumented; download depends on network/cache', hardware='CPU / network'))
    costs = pd.read_csv(RESULTS/'stage2_costs.csv')
    runtime.append(dict(stage=2, seconds=(costs.fit_seconds+costs.inference_seconds).sum(), component='Recorded fitting + inference only', hardware='CPU'))
    costs = pd.read_csv(RESULTS/'stage3_costs.csv')
    runtime.append(dict(stage=3, seconds=(costs.load_seconds+costs.inference_seconds).sum(), component='Cached model load + inference only', hardware='RTX 3050'))
    costs = pd.read_csv(RESULTS/'stage4_costs.csv')
    runtime.append(dict(stage=4, seconds=costs.fit_seconds.sum(), component='Training, all 18 runs', hardware='RTX 3050'))
    costs = pd.read_csv(RESULTS/'stage5_run_costs.csv')
    runtime.append(dict(stage=5, seconds=costs.loc[costs.reused_from.isna(), 'fit_seconds'].sum(), component='Training, 45 new runs; 9 reused', hardware='RTX 3050'))
    runtime.append(dict(stage=6, seconds=np.nan, component='Figure/report generation; no training; measure locally', hardware='CPU'))
    pd.DataFrame(runtime).to_csv(RESULTS/'stage6_runtime.csv', index=False)
    audit = [dict(path=p, before_sha256=h, after_sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p,h in old.items()]
    assert all(r['before_sha256']==r['after_sha256'] for r in audit)
    pd.DataFrame(audit).to_csv(RESULTS/'stage6_preservation.csv', index=False)


if __name__ == '__main__':
    main()
