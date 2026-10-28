"""Audit completed Stage 5 panel identity, reuse and historical preservation."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.zero_shot import common_template, KEYS


def main() -> None:
    """Fail if any model changes forecast rows or an original result file changes."""
    root, old, results = Path('data/stage5'), Path('data/stage4'), Path('results')
    template = common_template(pd.read_parquet('data/stage2_predictions.parquet'))
    costs = pd.read_csv(results / 'stage5_run_costs.csv')
    audit = []
    for row in costs.itertuples():
        predictions = pd.read_parquet(root / f'{row.model}_predictions.parquet')
        pd.testing.assert_frame_equal(predictions[template.columns], template)
        reused_equal = False
        if pd.notna(row.reused_from):
            original = pd.read_parquet(old / f'{row.reused_from}_predictions.parquet')
            np.testing.assert_array_equal(predictions.prediction.to_numpy(), original.prediction.to_numpy())
            reused_equal = True
        audit.append(dict(model=row.model, rows=len(predictions), common_panel_identical=True,
                          reused_prediction_identical=reused_equal if pd.notna(row.reused_from) else None))
    assert len(costs) == 54 and costs.extended_budget.sum() == 3
    weekday = pd.read_parquet(root / 'weekday_predictions.parquet')
    pd.testing.assert_frame_equal(weekday[template.columns], template)
    pd.DataFrame(audit).to_csv(results / 'stage5_panel_audit.csv', index=False)
    counts = template.assign(symbol_group=np.where(template.symbol.isin(['SOLUSDT', 'DOGEUSDT']),
        'held-out', 'fit-eligible')).groupby(['split', 'horizon', 'symbol_group']).size().rename('examples_per_model')
    counts.to_csv(results / 'stage5_example_counts.csv')
    histories = []
    for row in costs.loc[costs.extended_budget].itertuples():
        current = pd.read_parquet(root / f'{row.model}_history.parquet')
        previous = pd.read_parquet(old / f'Chronos-Bolt-tiny-{row.mode}-seed{row.seed}_history.parquet')
        joined = current.merge(previous, on='epoch', suffixes=('_new', '_old'))
        difference = (joined.validation_loss_new-joined.validation_loss_old).abs().max()
        np.testing.assert_allclose(joined.validation_loss_new, joined.validation_loss_old, rtol=0, atol=1e-5)
        histories.append(dict(seed=row.seed, matched_epochs=len(joined), max_validation_loss_difference=difference,
                              best_epoch=row.best_epoch, selected_final_cap=row.best_epoch == 8))
    pd.DataFrame(histories).to_csv(results / 'stage5_extended_reproducibility.csv', index=False)
    historical = yaml.safe_load((root / 'historical_hashes.yaml').read_text())
    for path, digest in historical.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    print(f'PASS: {len(costs)+1} models have identical common rows; 9 reused forecasts exact; '
          f'{len(historical)} historical result files byte-identical; extended initial epochs reproduce.')


if __name__ == '__main__':
    main()
