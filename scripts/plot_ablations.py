"""Produce Stage 5 reporting curves, regime tables and retrospective error plots."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.regimes import regime_thresholds, analyze_regimes


def plot_curves(results: Path) -> None:
    """Plot test reporting curves without selecting configurations from them."""
    table = pd.read_csv(results / 'stage5_ablation_summary.csv')
    for axis, label in [('context', 'Context days'), ('fraction', 'Fraction of earliest train windows'), ('rank', 'LoRA rank')]:
        for metric in ['qlike_raw', 'qlike']:
            fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
            for i, group in enumerate(['fit-eligible', 'held-out']):
                for j, horizon in enumerate([1, 5, 22]):
                    ax = axes[i, j]
                    rows = table.loc[table.axis.eq(axis) & table.symbol_group.eq(group) & table.horizon.eq(horizon)].copy()
                    rows['line'] = 'LoRA' if axis == 'rank' else rows['mode']
                    for mode, data in rows.groupby('line'):
                        data = data.sort_values('value')
                        ax.plot(data.value, data[metric], 'o-', label=mode)
                        ax.fill_between(data.value, data[metric+'_lo'], data[metric+'_hi'], alpha=.12)
                    ax.set(title=f'{group}, h={horizon}', xlabel=label, ylabel=metric)
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=8)
            fig.suptitle('Test reporting only; frozen Stage 4 headline unchanged. Bands: 95% seed/date block CI.')
            fig.savefig(results / f'stage5_{axis}_{metric}.png', dpi=150)
            plt.close(fig)
    ranks = table.loc[table.axis.eq('rank')].drop_duplicates('value').sort_values('value')
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(ranks.value, ranks.trainable_parameters, 'o-')
    ax.set(xlabel='LoRA rank', ylabel='Trainable parameters', title='Small-model adapter size')
    ax.grid(alpha=.2)
    fig.savefig(results / 'stage5_rank_parameters.png', dpi=150)
    plt.close(fig)


def error_analysis(scored: pd.DataFrame, selection: pd.DataFrame, daily: pd.DataFrame, results: Path) -> None:
    """Rank mean seed squared log errors, then explain observed series changes."""
    family = selection.loc[selection.selected & selection.horizon.eq(1), 'family'].item()
    rows = scored.loc[scored.model.str.replace(r'-seed\d+$', '', regex=True).eq(family) & scored.horizon.eq(1)]
    worst = rows.groupby(['symbol', 'origin', 'target_end']).agg(target=('target', 'first'),
        prediction=('prediction', 'mean'), prediction_low=('prediction', 'min'), prediction_high=('prediction', 'max'),
        mean_seed_squared_error=('squared_error', 'mean')).nlargest(3, 'mean_seed_squared_error').reset_index()
    descriptions = []
    for rank, row in enumerate(worst.itertuples(), start=1):
        series = daily.loc[daily.symbol.eq(row.symbol)].set_index('date').sort_index()
        actual_rv, predicted_rv = np.exp(row.target), np.exp(row.prediction)
        previous = float(series.loc[row.origin, 'rv'])
        segment = series.loc[row.origin-pd.Timedelta(days=30):row.target_end+pd.Timedelta(days=5)]
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(segment.index, segment.y, '.-', label='Observed daily log RV')
        ax.axvline(row.origin, color='gray', linestyle='--', label='Forecast origin')
        ax.errorbar([row.target_end], [row.prediction],
            yerr=[[row.prediction-row.prediction_low], [row.prediction_high-row.prediction]],
            fmt='D', color='tab:red', capsize=5, label='Mean prediction; range across 3 seeds')
        ax.set(title=f'Worst h=1 error #{rank}: {row.symbol}, {row.target_end:%Y-%m-%d}', ylabel='log RV')
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(results / f'stage5_worst_{rank}.png', dpi=150)
        plt.close(fig)
        change = actual_rv / previous
        direction = 'underpredicted' if actual_rv > predicted_rv else 'overpredicted'
        descriptions.append(f'{rank}. **{row.symbol}, {row.target_end:%Y-%m-%d}**: daily RV was {actual_rv:.6g}, '
            f'{change:.2f} times the previous day. The mean log forecast implied RV {predicted_rv:.6g} '
            f'and {direction} realized variance by a factor of {max(actual_rv/predicted_rv, predicted_rv/actual_rv):.2f}. '
            f'The abrupt change from the observed history explains the large one-day error; the model had no future '
            f'price or news inputs. Daily RV alone cannot identify the cause of the move. '
            f'Mean seed squared log error: {row.mean_seed_squared_error:.3f}.')
    worst.to_csv(results / 'stage5_worst_forecasts.csv', index=False)
    context = ''
    if worst.target_end.dt.strftime('%Y-%m-%d').eq('2025-10-10').all():
        url = 'https://www.binance.com/en/support/announcement/detail/d9cb0d52d7c142a5be4f49732bd8760c'
        context = ('All three cases coincide with the same market event. Binance reported a broad sell-off '
            'and extreme volatility on 2025-10-10 between 20:50 and 22:00 UTC '
            f'([exchange statement, published October 12]({url})). This retrospective context is consistent '
            'with a common shock absent from the log-RV-only inputs; it does not establish each coin\'s causal mechanism.\n\n')
        pd.DataFrame([dict(date='2025-10-10', source_url=url, publication_date='2025-10-12',
            scope='Retrospective exchange account of broad market sell-off; not a model input')]).to_csv(
                results / 'stage5_event_context.csv', index=False)
    Path('data/stage5/error_analysis.md').write_text(context+'\n\n'.join(descriptions)+'\n', encoding='utf-8')


def main() -> None:
    """Regenerate analysis using frozen selection and already scored predictions."""
    results = Path('results')
    config = yaml.safe_load(Path('configs/stage-5-ablations.yaml').read_text())
    scored = pd.read_parquet('data/stage5/scored.parquet')
    selection = pd.read_csv(results / 'stage5_frozen_selection.csv')
    thresholds = regime_thresholds(pd.read_parquet('data/examples.parquet'))
    thresholds.to_csv(results / 'stage5_regime_thresholds.csv', index=False)
    analyze_regimes(scored, thresholds, selection, config).to_csv(results / 'stage5_regimes.csv', index=False)
    error_analysis(scored, selection, pd.read_parquet('data/daily_rv.parquet'), results)
    plot_curves(results)


if __name__ == '__main__':
    main()
