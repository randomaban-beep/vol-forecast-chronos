"""Training-only regime thresholds and calendar-preserving conditional inference."""
import numpy as np
import pandas as pd
from src.finetune_summary import LOSS_COLUMNS, family_mcs
from src.evaluate import diebold_mariano


def regime_thresholds(examples: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol/horizon 90th percentile of train realized volatility only."""
    train = examples.loc[examples['split'].eq('train')].copy()
    if train.target_end.max() >= pd.Timestamp('2024-01-01', tz='UTC'):
        raise ValueError('Regime threshold includes future targets')
    train['volatility'] = np.exp(train.target / 2)
    return train.groupby(['symbol', 'horizon']).volatility.quantile(.9).rename('threshold').reset_index()


def conditional_statistics(sums: np.ndarray, counts: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Seed-mean conditional metrics and CIs resampling full calendar blocks.

    sums axes are seed,date,loss. Zero-observation dates remain present; sampled
    losses are divided by sampled observation counts, not by compressed dates.
    """
    seeds, dates, _ = sums.shape
    if counts.sum() <= 0:
        raise ValueError('Empty regime')
    estimate = sums.sum(axis=1) / counts.sum()
    estimate[:, 0] = np.sqrt(estimate[:, 0])
    rng = np.random.default_rng(config['seed'])
    block, reps = config['block_length'], config['bootstrap_reps']
    starts = rng.integers(dates, size=(reps, int(np.ceil(dates / block))))
    indices = ((starts[..., None] + np.arange(block)) % dates).reshape(reps, -1)[:, :dates]
    chosen = rng.integers(seeds, size=(reps, seeds))
    denominator = counts[indices].sum(axis=1)
    valid = denominator > 0
    if valid.sum() < 2:
        raise ValueError('Too few nonempty bootstrap replications')
    boot = sums[chosen[:, :, None], indices[:, None, :]].sum(axis=2)[valid] / denominator[valid, None, None]
    boot[:, :, 0] = np.sqrt(boot[:, :, 0])
    return estimate.mean(axis=0), np.quantile(boot.mean(axis=1), [.025, .975], axis=0), int(valid.sum())


def analyze_regimes(scored: pd.DataFrame, thresholds: pd.DataFrame, selection: pd.DataFrame,
                    config: dict) -> pd.DataFrame:
    """Compare frozen headline families with HAR and HAR-weekday within regimes."""
    frame = scored.assign(family=scored.model.str.replace(r'-seed\d+$', '', regex=True))
    frame = frame.merge(thresholds, on=['symbol', 'horizon'], validate='many_to_one')
    frame['high'] = np.exp(frame.target / 2) > frame.threshold
    output = []
    for (group, h), panel in frame.groupby(['symbol_group', 'horizon']):
        best = selection.loc[selection.selected & selection.horizon.eq(h), 'family'].item()
        panel = panel.loc[panel.family.isin(['HAR-RV', 'HAR-weekday', best])]
        dates = pd.DatetimeIndex(sorted(panel.origin.unique()))
        for high in [False, True]:
            selected = panel.loc[panel.high.eq(high)]
            reference = selected.loc[selected.model.eq('HAR-RV')]
            counts = reference.groupby('origin').size().reindex(dates, fill_value=0).to_numpy()
            all_sums = {}
            for family, rows in selected.groupby('family'):
                all_sums[family] = np.stack([trial.groupby('origin')[LOSS_COLUMNS].sum().reindex(dates, fill_value=0).to_numpy()
                    for _, trial in rows.groupby('model')])
            # Same calendar and common regime mask for every family; zero-pad gaps.
            # Mean weighted date contribution equals the conditional mean loss.
            panels = {metric: pd.DataFrame({family: sums[:, :, column].mean(axis=0) * len(dates) / counts.sum()
                for family, sums in all_sums.items()}, index=dates)
                for column, metric in [(2, 'qlike'), (3, 'qlike_raw')]}
            memberships = {metric: family_mcs(losses, config) for metric, losses in panels.items()}
            for family, sums in all_sums.items():
                mean, ci, nonempty = conditional_statistics(sums, counts, config)
                row = dict(symbol_group=group, horizon=h, regime='top_decile' if high else 'rest', model=family,
                    n_forecasts_per_seed=int(counts.sum()), n_calendar_dates=len(dates), n_nonempty_dates=int((counts > 0).sum()),
                    bootstrap_reps_nonempty=nonempty, bootstrap_reps_requested=config['bootstrap_reps'])
                for i, metric in enumerate(['rmse', 'mae', 'qlike', 'qlike_raw']):
                    row.update({metric: mean[i], metric+'_lo': ci[0, i], metric+'_hi': ci[1, i]})
                for metric, losses in panels.items():
                    for baseline in ['HAR-RV', 'HAR-weekday']:
                        row.update({f'{k}_{metric}_vs_{baseline}': v for k, v in
                            diebold_mariano(losses[family].to_numpy(), losses[baseline].to_numpy(), int(h)).items()})
                    row[f'mcs_pvalue_{metric}'] = memberships[metric][family]
                    row[f'mcs_95_included_{metric}'] = memberships[metric][family] >= .05
                output.append(row)
    return pd.DataFrame(output)
