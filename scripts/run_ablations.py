"""Run cached Stage 5 one-factor ablations without rewriting earlier results."""
import hashlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ablations import ablation_grid, earliest_fraction, extended_family, select_stage4_headlines, weekday_walk_forward
from src.finetune import train_and_predict
from src.zero_shot import common_template, build_contexts, align_forecasts


def main() -> None:
    """Freeze selection, reuse identical runs, and train remaining configurations."""
    base = yaml.safe_load(Path('configs/stage-4-finetune.yaml').read_text())
    config = yaml.safe_load(Path('configs/stage-5-ablations.yaml').read_text())
    if not 3 < config['extended_max_epochs'] <= 8:
        raise ValueError('Extended training cap must be between 4 and 8 epochs')
    root, old, results = Path('data/stage5'), Path('data/stage4'), Path('results')
    root.mkdir(exist_ok=True)
    historical = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in results.glob('stage[234]*') if p.is_file()}
    manifest = root / 'historical_hashes.yaml'
    if manifest.exists():
        assert yaml.safe_load(manifest.read_text()) == historical
    else:
        manifest.write_text(yaml.safe_dump(historical))
    daily = pd.read_parquet('data/daily_rv.parquet')
    template = common_template(pd.read_parquet('data/stage2_predictions.parquet'))
    origins, contexts = build_contexts(daily, template)
    old_predictions = pd.concat([pd.read_parquet(p) for p in old.glob('*_predictions.parquet')])
    selection = select_stage4_headlines(old_predictions, config['held_out'])
    selection.to_csv(results / 'stage5_frozen_selection.csv', index=False)
    old_costs = pd.concat([pd.read_parquet(p) for p in old.glob('*_cost.parquet')])
    mode, audit = extended_family(old_costs)
    audit.to_csv(results / 'stage5_extended_selection.csv', index=False)
    grid = ablation_grid(config)
    grid.to_csv(results / 'stage5_grid.csv', index=False)
    runs = grid.drop_duplicates('family').to_dict('records')
    if mode:
        runs.append(dict(family=f'tiny-{mode}-extended8', mode=mode, context=256, fraction=1., extended=True))
    metadata = pd.read_parquet(old / 'train_windows.parquet')
    train = pd.read_parquet(old / 'train_tensors.parquet').to_numpy(dtype=np.float32)
    val = pd.read_parquet(old / 'validation_tensors.parquet').to_numpy(dtype=np.float32)
    weekday, fit_audit = weekday_walk_forward(pd.read_parquet('data/examples.parquet'), template, config['held_out'])
    weekday.to_parquet(root / 'weekday_predictions.parquet', index=False)
    fit_audit.to_csv(results / 'stage5_weekday_fit_audit.csv', index=False)
    costs, histories, fractions = [], [], []
    fingerprint_data = pd.util.hash_pandas_object(daily, index=False).to_numpy().tobytes()
    original_fingerprint = hashlib.sha256(yaml.safe_dump(base, sort_keys=True).encode() + fingerprint_data).hexdigest()
    for run in runs:
        length, fraction = int(run['context']), float(run['fraction'])
        indices = earliest_fraction(metadata, fraction)
        extended = bool(run.get('extended', False))
        spec = next(s for s in base['models'] if s['name'] == ('Chronos-Bolt-tiny' if extended else 'Chronos-Bolt-small'))
        cfg = {**base, 'context_length': length,
               'max_epochs': config['extended_max_epochs'] if extended else base['max_epochs']}
        fingerprint = hashlib.sha256(yaml.safe_dump({**cfg, 'fraction': fraction}).encode() + fingerprint_data).hexdigest()
        for seed in config['seeds']:
            name = f"{run['family']}-seed{seed}"
            prefix = root / name
            files = {kind: Path(str(prefix) + f'_{kind}.parquet') for kind in ['predictions', 'cost', 'history']}
            reusable = not extended and length == 256 and fraction == 1 and run['mode'] in ['full', 'lora8', 'lora16']
            if all(p.exists() for p in files.values()):
                cost = pd.read_parquet(files['cost']).iloc[0].to_dict()
                assert cost['stage5_fingerprint'] == fingerprint, name
                history = pd.read_parquet(files['history'])
                print(f'Cached {name}', flush=True)
            else:
                print(f"Starting {name}; reused={reusable}; windows={len(indices)}", flush=True)
                if reusable:
                    source = f"{spec['name']}-{run['mode']}-seed{seed}"
                    predictions = pd.read_parquet(old / f'{source}_predictions.parquet').assign(model=name)
                    cost = pd.read_parquet(old / f'{source}_cost.parquet').iloc[0].to_dict()
                    assert cost['input_config_sha256'] == original_fingerprint, source
                    history = pd.read_parquet(old / f'{source}_history.parquet').assign(model=name)
                else:
                    source = ''
                    paths, cost, history = train_and_predict(spec, run['mode'], seed, cfg,
                        (train[indices, 256-length:256], train[indices, 256:]),
                        (val[:, 256-length:256], val[:, 256:]), contexts[:, -length:], Path(str(prefix) + '.pt'))
                    predictions = align_forecasts(template, origins, paths, name)
                    predictions['inference_seconds'] = cost['inference_seconds'] / len(predictions)
                    history['model'] = name
                pd.testing.assert_frame_equal(predictions[template.columns], template)
                cost.update(model=name, stage5_fingerprint=fingerprint, reused_from=source,
                    context=length, fraction=fraction, extended_budget=extended,
                    inference_ms_per_1000=cost['inference_seconds'] / len(predictions) * 1e6)
                predictions.to_parquet(files['predictions'], index=False)
                pd.DataFrame([cost]).to_parquet(files['cost'], index=False)
                history.to_parquet(files['history'], index=False)
            costs.append(cost)
            histories.append(history)
        fractions.append(dict(family=run['family'], windows=len(indices),
            first_origin=metadata.iloc[indices].origin.min(), last_origin=metadata.iloc[indices].origin.max()))
    pd.DataFrame(costs).to_csv(results / 'stage5_run_costs.csv', index=False)
    pd.concat(histories).to_csv(results / 'stage5_training_history.csv', index=False)
    pd.DataFrame(fractions).to_csv(results / 'stage5_data_audit.csv', index=False)
    for path, digest in historical.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    print('Training complete; historical results unchanged.', flush=True)


if __name__ == '__main__':
    main()
