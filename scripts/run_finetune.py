"""Run three-seed full and LoRA adaptation, then use the unchanged shared scorer."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.finetune import training_windows, train_and_predict
from src.zero_shot import common_template, build_contexts, align_forecasts
from src.evaluate import evaluate_panel


def main() -> None:
    """Train each requested configuration once, with resumable fingerprinted caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage-4-finetune.yaml")
    parser.add_argument("--reuse-runs", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    cache, results = Path(config["cache"]), Path(config["results"])
    root = cache / "stage4"
    root.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(cache / "daily_rv.parquet")
    baselines = pd.read_parquet(cache / "stage2_predictions.parquet")
    zero_shot = pd.read_parquet(cache / "stage3_predictions.parquet")
    template = common_template(baselines)
    origins, contexts = build_contexts(daily, template)
    datasets, audits = {}, []
    for split in ["train", "validation"]:
        metadata, x, y = training_windows(daily, split, config)
        datasets[split] = (x, y)
        metadata.to_parquet(root / f"{split}_windows.parquet", index=False)
        pd.DataFrame(np.concatenate([x, y], axis=1)).to_parquet(root / f"{split}_tensors.parquet", index=False)
        audits.append(metadata.groupby(["split", "symbol"]).agg(windows=("origin", "size"),
            first_target=("target_start", "min"), last_target=("target_end", "max")))
    pd.concat(audits).to_csv(results / "stage4_data_audit.csv")
    fingerprint = hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode() +
        pd.util.hash_pandas_object(daily, index=False).to_numpy().tobytes()).hexdigest()
    all_forecasts, all_costs, all_history = [], [], []
    for spec in config["models"]:
        for mode in config["modes"]:
            for seed in config["seeds"]:
                name = f"{spec['name']}-{mode}-seed{seed}"
                path = root / f"{name}_predictions.parquet"
                cost_path = root / f"{name}_cost.parquet"
                history_path = root / f"{name}_history.parquet"
                if args.reuse_runs and path.exists() and cost_path.exists() and history_path.exists():
                    cost = pd.read_parquet(cost_path).iloc[0].to_dict()
                    if cost["input_config_sha256"] != fingerprint:
                        raise ValueError(f"Stale cached run: {name}")
                    predictions = pd.read_parquet(path)
                    history = pd.read_parquet(history_path)
                    pd.testing.assert_frame_equal(predictions[template.columns], template)
                else:
                    print(f"Starting {name}", flush=True)
                    paths, cost, history = train_and_predict(spec, mode, seed, config,
                        datasets["train"], datasets["validation"], contexts, root / f"{name}.pt")
                    predictions = align_forecasts(template, origins, paths, name)
                    predictions["inference_seconds"] = cost["inference_seconds"] / len(predictions)
                    cost.update(model=name, input_config_sha256=fingerprint,
                                inference_ms_per_1000=cost["inference_seconds"] / len(predictions) * 1e6)
                    history["model"] = name
                    predictions.to_parquet(path, index=False)
                    pd.DataFrame([cost]).to_parquet(cost_path, index=False)
                    history.to_parquet(history_path, index=False)
                    pd.concat([origins, pd.DataFrame(paths, columns=[f"day_{i}" for i in range(1, 23)])], axis=1).to_parquet(
                        root / f"{name}_paths.parquet", index=False)
                all_forecasts.append(predictions)
                all_costs.append(cost)
                all_history.append(history)
    tuned = pd.concat(all_forecasts, ignore_index=True)
    costs = pd.DataFrame(all_costs)
    costs.to_csv(results / "stage4_costs.csv", index=False)
    pd.concat(all_history, ignore_index=True).to_csv(results / "stage4_training_history.csv", index=False)
    # Score every seed independently, using the identical evaluator and calibration rule.
    summary, calibration, scored = evaluate_panel(pd.concat([baselines, zero_shot, tuned], ignore_index=True), config)
    scored.to_parquet(root / "scored.parquet", index=False)
    summary.loc[summary.model.isin(tuned.model.unique())].to_csv(results / "stage4_per_seed.csv", index=False)
    calibration.loc[calibration.model.isin(tuned.model.unique())].to_csv(results / "stage4_calibration.csv", index=False)
    from src.finetune_summary import summarize_seeds
    aggregate, mcs = summarize_seeds(scored, summary, costs, config)
    aggregate.to_csv(results / "stage4_finetune.csv", index=False)
    mcs.to_csv(results / "stage4_mcs.csv", index=False)
    print(aggregate[["model", "symbol_group", "horizon", "rmse", "qlike_raw", "qlike"]].to_string(index=False))


if __name__ == "__main__":
    main()
