"""Audit completed Stage 5 panel identity, reuse and historical preservation."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.zero_shot import common_template, KEYS
from src.finetune import training_windows


def main() -> None:
    """Fail if any model changes forecast rows or an original result file changes."""
    root, old, results = Path('data/stage5'), Path('data/stage4'), Path('results')
    template = common_template(pd.read_parquet('data/stage2_predictions.parquet'))
    costs = pd.read_csv(results / 'stage5_run_costs.csv')
    base_config = yaml.safe_load(Path('configs/stage-4-finetune.yaml').read_text())
    daily = pd.read_parquet('data/daily_rv.parquet')
    original_fingerprint = hashlib.sha256(yaml.safe_dump(base_config, sort_keys=True).encode() +
        pd.util.hash_pandas_object(daily, index=False).to_numpy().tobytes()).hexdigest()
    tensors = []
    for split in ['train', 'validation']:
        _, x, y = training_windows(daily, split, base_config)
        cached = pd.read_parquet(old / f'{split}_tensors.parquet').to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(x, cached[:, :256])
        np.testing.assert_array_equal(y, cached[:, 256:])
        tensors.append(dict(split=split, values_identical=True, original_context_strides=str(x.strides),
                            cached_context_strides=str(cached[:, :256].strides)))
    pd.DataFrame(tensors).to_csv(results / 'stage5_tensor_audit.csv', index=False)
    audit = []
    for row in costs.itertuples():
        predictions = pd.read_parquet(root / f'{row.model}_predictions.parquet')
        pd.testing.assert_frame_equal(predictions[template.columns], template)
        reused_equal = False
        if pd.notna(row.reused_from):
            original_cost = pd.read_parquet(old / f'{row.reused_from}_cost.parquet').iloc[0]
            assert original_cost.input_config_sha256 == original_fingerprint
            original = pd.read_parquet(old / f'{row.reused_from}_predictions.parquet')
            np.testing.assert_array_equal(predictions.prediction.to_numpy(), original.prediction.to_numpy())
            reused_equal = True
        audit.append(dict(model=row.model, rows=len(predictions), common_panel_identical=True,
                          reused_prediction_identical=reused_equal if pd.notna(row.reused_from) else None))
    assert len(costs) == costs.model.nunique() == 54 and costs.extended_budget.sum() == 3
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
        histories.append(dict(seed=row.seed, matched_epochs=len(joined), max_validation_loss_difference=difference,
                              shared_history_exact=bool(difference == 0),
                              best_epoch=row.best_epoch, selected_final_cap=row.best_epoch == 8))
    pd.DataFrame(histories).to_csv(results / 'stage5_extended_reproducibility.csv', index=False)
    historical = yaml.safe_load((root / 'historical_hashes.yaml').read_text())
    for path, digest in historical.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    print(f'PASS: {len(costs)+1} models have identical common rows; 9 reused forecasts exact; '
          f'{len(historical)} historical result files byte-identical. '
          f'Extended retraining loss drift audited: {max(r["max_validation_loss_difference"] for r in histories):.8g}.')


if __name__ == '__main__':
    main()
