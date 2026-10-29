"""Score Stage 5 with the shared evaluator and report both HAR comparisons."""
from pathlib import Path
import sys
import hashlib
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import evaluate_panel, diebold_mariano
from src.finetune_summary import summarize_seeds


def add_weekday_dm(summary: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Add HAC/Harvey comparisons against the new calendar baseline."""
    scored = scored.assign(family=scored.model.str.replace(r'-seed\d+$', '', regex=True))
    losses = scored.groupby(['symbol_group', 'horizon', 'origin', 'family'])[['qlike', 'qlike_raw']].mean()
    out = summary.copy()
    for index, row in out.iterrows():
        subset = losses.xs((row.symbol_group, row.horizon), level=['symbol_group', 'horizon'])
        for metric in ['qlike', 'qlike_raw']:
            panel = subset[metric].unstack('family')
            values = diebold_mariano(panel[row.model].to_numpy(), panel['HAR-weekday'].to_numpy(), int(row.horizon))
            for key, value in values.items():
                out.loc[index, f'{key}_vs_weekday_{metric}'] = value
    return out


def main() -> None:
    """Create only new tables; family MCS includes all previous model families."""
    config = yaml.safe_load(Path('configs/stage-5-ablations.yaml').read_text())
    root, results = Path('data/stage5'), Path('results')
    costs = pd.read_csv(results / 'stage5_run_costs.csv')
    frames = [pd.read_parquet('data/stage2_predictions.parquet'), pd.read_parquet('data/stage3_predictions.parquet')]
    frames += [pd.read_parquet(p) for p in Path('data/stage4').glob('*_predictions.parquet')]
    frames += [pd.read_parquet(p) for p in root.glob('*_predictions.parquet')]
    predictions = pd.concat(frames, ignore_index=True)
    summary, calibration, scored = evaluate_panel(predictions, {**config, 'compute_mcs': False})
    # MCS is computed jointly at family level below, not separately per seed.
    summary = summary.drop(columns=[column for column in summary if column.startswith('mcs_')])
    scored.to_parquet(root / 'scored.parquet', index=False)
    aggregate, mcs = summarize_seeds(scored, summary, costs, config)
    aggregate = add_weekday_dm(aggregate, scored)
    extended_names = costs.loc[costs.extended_budget, 'model'].str.replace(r'-seed\d+$', '', regex=True).unique()
    extended = aggregate.model.isin(extended_names)
    aggregate.loc[extended].to_csv(results / 'stage5_extended_tiny_summary.csv', index=False)
    grid = pd.read_csv(results / 'stage5_grid.csv')
    grid.merge(aggregate.loc[~extended], left_on='family', right_on='model', validate='many_to_many').to_csv(
        results / 'stage5_ablation_summary.csv', index=False)
    weekday = add_weekday_dm(summary.loc[summary.model.eq('HAR-weekday')], scored)
    weekday_predictions = predictions.loc[predictions.model.eq('HAR-weekday')]
    weekday_cost = dict(model='HAR-weekday', trainable_parameters=10,
        fit_seconds=pd.read_csv(results / 'stage5_weekday_fit_audit.csv').fit_seconds.sum(),
        inference_ms_per_1000=weekday_predictions.inference_seconds.mean()*1e6, gpu_memory_mb=0)
    pd.DataFrame([weekday_cost]).to_csv(results / 'stage5_weekday_costs.csv', index=False)
    for key, value in weekday_cost.items():
        if key != 'model':
            weekday[key] = value
    for metric in ['qlike', 'qlike_raw']:
        membership = mcs.loc[mcs.model.eq('HAR-weekday') & mcs.metric.eq(metric),
            ['symbol_group', 'horizon', 'mcs_pvalue', 'mcs_95_included']].rename(columns={
                'mcs_pvalue': f'joint_mcs_pvalue_{metric}', 'mcs_95_included': f'joint_mcs_95_included_{metric}'})
        weekday = weekday.merge(membership, on=['symbol_group', 'horizon'], validate='one_to_one')
    weekday.to_csv(results / 'stage5_weekday.csv', index=False)
    mcs.to_csv(results / 'stage5_mcs.csv', index=False)
    new_names = set(costs.model) | {'HAR-weekday'}
    summary.loc[summary.model.isin(new_names)].to_csv(results / 'stage5_per_seed.csv', index=False)
    calibration.loc[calibration.model.isin(new_names)].to_csv(results / 'stage5_calibration.csv', index=False)
    for flag, label in [(True, 'extended_tiny'), (False, 'ablation')]:
        costs.loc[costs.extended_budget.eq(flag)].to_csv(results / f'stage5_{label}_costs.csv', index=False)
        names = costs.loc[costs.extended_budget.eq(flag), 'model']
        summary.loc[summary.model.isin(names)].to_csv(results / f'stage5_{label}_per_seed.csv', index=False)
        calibration.loc[calibration.model.isin(names)].to_csv(results / f'stage5_{label}_calibration.csv', index=False)
        history = pd.read_csv(results / 'stage5_training_history.csv')
        history.loc[history.model.isin(names)].to_csv(results / f'stage5_{label}_training_history.csv', index=False)
    historical = yaml.safe_load((root / 'historical_hashes.yaml').read_text())
    pd.DataFrame([dict(path=p, before_sha256=v, after_sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                       byte_identical=v == hashlib.sha256(Path(p).read_bytes()).hexdigest())
                  for p, v in historical.items()]).to_csv(results / 'stage5_preservation.csv', index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == v for p, v in historical.items())
    print(weekday[['symbol_group', 'horizon', 'rmse', 'qlike_raw', 'qlike']].to_string(index=False))


if __name__ == '__main__':
    main()
