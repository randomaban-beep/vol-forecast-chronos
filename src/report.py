"""Frozen CSV adapters for the final report; no model fitting or selection."""
from pathlib import Path
import pandas as pd

RESULTS = Path('results')
MODELS = ['Naive', 'HAR-RV', 'HAR-weekday', 'GARCH-t', 'Zero-shot', 'Fine-tuned']
SCOREBOARD = ['Naive', 'HAR-RV', 'HAR-log-mean-RV', 'HAR-weekday', 'GARCH-t', 'GJR-GARCH-t', 'Zero-shot', 'Fine-tuned']
GROUPS = ['fit-eligible', 'held-out']
COLORS = ['#697386', '#0072B2', '#009E73', '#CC79A7', '#E69F00', '#D55E00']


def headline_rows() -> pd.DataFrame:
    """Read original point estimates/CIs and the frozen per-horizon family choice."""
    baseline = pd.read_csv(RESULTS/'stage2_baselines.csv')
    baseline = baseline.loc[baseline.model.isin(SCOREBOARD)]
    zero = pd.read_csv(RESULTS/'stage3_zero_shot.csv')
    zero = zero.loc[zero.model.eq('Chronos-Bolt-small')].copy()
    zero['family'] = zero.model
    zero['model'] = 'Zero-shot'
    tuned = pd.read_csv(RESULTS/'stage4_finetune.csv')
    selection = pd.read_csv(RESULTS/'stage5_frozen_selection.csv')
    tuned = tuned.merge(selection.loc[selection.selected, ['horizon', 'family']],
        left_on=['horizon','model'], right_on=['horizon','family'], validate='many_to_one')
    tuned['model'] = 'Fine-tuned'
    rows = pd.concat([baseline, pd.read_csv(RESULTS/'stage5_weekday.csv'), zero, tuned], ignore_index=True)
    rows['family'] = rows.family.fillna(rows.model)
    assert not rows.duplicated(['symbol_group','horizon','model']).any()
    return rows


def stage7_headline_table() -> str:
    """Generate the Stage 7 headline table with standard Patton QLIKE, CIs, DM tests, and costs."""
    df = pd.read_csv(RESULTS / 'stage7_headline.csv')
    lines = [
        '| Group | h | Model | log RMSE | MAE | Raw Patton QLIKE [95% CI] | Cal Patton QLIKE [95% CI] | DM p vs Weekday | Params | Train Time |',
        '| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |'
    ]
    for row in df.itertuples():
        p_str = '—' if row.model == 'HAR-weekday' else ('<0.0001' if row.dm_pvalue_vs_weekday < 0.0001 else f'{row.dm_pvalue_vs_weekday:.4f}')
        if row.trainable_parameters >= 1e6:
            p_count = f'{row.trainable_parameters / 1e6:.1f}M'
        elif row.trainable_parameters >= 1e3:
            p_count = f'{row.trainable_parameters / 1e3:.1f}K'
        else:
            p_count = str(int(row.trainable_parameters))
        t_time = f'{row.fit_seconds:.2f}s'
        lines.append(
            f'| {row.symbol_group} | {row.horizon} | {row.model} | {row.rmse:.4f} | {row.mae:.4f} | '
            f'{row.qlike_raw:.4f} [{row.qlike_raw_lo:.4f}, {row.qlike_raw_hi:.4f}] | '
            f'{row.qlike:.4f} [{row.qlike_lo:.4f}, {row.qlike_hi:.4f}] | '
            f'{p_str} | {p_count} | {t_time} |'
        )
    return '\n'.join(lines)


def legacy_headline_table() -> str:
    """Legacy orientation QLIKE table (V_hat/V - log(V_hat/V) - 1). Kept for errata."""
    rows = headline_rows().set_index(['symbol_group','horizon','model'])
    lines = ['| Group | h | ' + ' | '.join(MODELS) + ' |', '| --- | ---: | ' + ' | '.join(['---:']*len(MODELS)) + ' |']
    for group in GROUPS:
        for h in [1,5,22]:
            lines.append(f'| {group} | {h} | ' + ' | '.join(f'{rows.loc[(group,h,m), "qlike_raw"]:.4f}' for m in MODELS) + ' |')
    return '\n'.join(lines)


# Primary headline table is Stage 7 standard Patton QLIKE
headline_table = stage7_headline_table


def markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render data-backed report tables with stable four-decimal formatting."""
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)
