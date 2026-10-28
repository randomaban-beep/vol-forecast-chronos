"""Append zero-shot Chronos results while preserving frozen Stage 2 rows byte-for-byte."""

import argparse
import hashlib
from io import StringIO
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import evaluate_panel
from src.zero_shot import common_template, build_contexts, align_forecasts, infer_bolt


def append_preserving_baselines(frozen: bytes, new_rows: pd.DataFrame, path: Path) -> None:
    """Append only new models without parsing/reserializing any historical row."""
    old = pd.read_csv(StringIO(frozen.decode()))
    if set(new_rows.model) & set(old.model):
        raise ValueError("Cannot overwrite historical models")
    newline = "\r\n" if b"\r\n" in frozen else "\n"
    extra = new_rows[old.columns].to_csv(index=False, header=False, lineterminator=newline).encode()
    path.write_bytes(frozen + extra)
    if path.read_bytes()[:len(frozen)] != frozen:
        raise AssertionError("Historical bytes changed")


def main() -> None:
    """Run pinned tiny/small inference and the existing evaluator on the shared panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage-3-zero-shot.yaml")
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    cache, results = Path(config["cache"]), Path(config["results"])
    frozen = (results / "stage2_baselines_frozen.csv").read_bytes()
    if not (cache / "stage2_predictions.parquet").exists():
        # Baseline reconstruction must not rewrite committed results/cost timings.
        temporary_results = cache / "stage2_rebuilt_results"
        stage2_config = yaml.safe_load(Path("configs/stage-2-baselines.yaml").read_text())
        stage2_config["results"] = str(temporary_results)
        config_path = cache / "stage2_rebuild.yaml"
        cache.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(stage2_config))
        subprocess.run([sys.executable, "scripts/run_baselines.py", "--config", str(config_path)], check=True)
    baselines = pd.read_parquet(cache / "stage2_predictions.parquet")
    template = common_template(baselines)
    daily = pd.read_parquet(cache / "daily_rv.parquet")
    origins, contexts = build_contexts(daily, template, config["context_length"])
    fingerprint = hashlib.sha256(contexts.tobytes() +
        pd.util.hash_pandas_object(template, index=False).to_numpy().tobytes() +
        yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()
    origins.to_parquet(cache / "stage3_context_audit.parquet", index=False)
    origins.groupby(["split", "symbol"]).agg(paths=("origin", "size"),
        contexts_with_missing=("missing_days", lambda x: int(x.gt(0).sum())),
        max_missing_days=("missing_days", "max")).to_csv(results / "stage3_context_audit.csv")
    prediction_path = cache / "stage3_predictions.parquet"
    if args.reuse_predictions:
        chronos = pd.read_parquet(prediction_path)
        costs = pd.read_parquet(cache / "stage3_costs.parquet")
        if not costs.input_config_sha256.eq(fingerprint).all():
            raise ValueError("Cached predictions do not match current data/configuration")
        for _, group in chronos.groupby("model"):
            pd.testing.assert_frame_equal(group[template.columns].reset_index(drop=True), template)
    else:
        forecasts, measurements = [], []
        for model in config["models"]:
            print(f"Loading {model['id']} at {model['revision']}", flush=True)
            try:
                paths, cost = infer_bolt(contexts, model, config)
            except Exception as exc:
                import torch
                if not model["name"].endswith("small") or not isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                cost = {"model": model["name"], "model_id": model["id"], "revision": model["revision"],
                        "status": "skipped_cuda_oom"}
                paths = None
            if paths is not None:
                path_frame = pd.concat([origins, pd.DataFrame(paths, columns=[f"day_{i}" for i in range(1, 23)])], axis=1)
                path_frame.to_parquet(cache / f"{model['name']}_paths.parquet", index=False)
                out = align_forecasts(template, origins, paths, model["name"])
                cost["forecasts"] = len(out)
                cost["inference_ms_per_1000"] = cost["inference_seconds"] / len(out) * 1e6
                cost["inference_ms_per_1000_paths"] = cost["inference_seconds"] / len(origins) * 1e6
                out["inference_seconds"] = cost["inference_seconds"] / len(out)
                forecasts.append(out)
            cost["input_config_sha256"] = fingerprint
            measurements.append(cost)
        chronos, costs = pd.concat(forecasts, ignore_index=True), pd.DataFrame(measurements)
        chronos.to_parquet(prediction_path, index=False)
        costs.to_parquet(cache / "stage3_costs.parquet", index=False)
    costs.to_csv(results / "stage3_costs.csv", index=False)
    combined = pd.concat([baselines, chronos], ignore_index=True)
    summary, calibration, scored = evaluate_panel(combined, config)
    scored.to_parquet(cache / "stage3_scored.parquet", index=False)
    calibration.loc[calibration.model.isin(chronos.model.unique())].to_csv(results / "stage3_calibration.csv", index=False)
    # MCS depends on its model universe: publish joint membership for ALL models separately.
    mcs_columns = ["symbol_group", "horizon", "model"] + [c for c in summary if c.startswith("mcs_")]
    summary[mcs_columns].assign(model_universe="|".join(sorted(combined.model.unique()))).to_csv(
        results / "stage3_mcs.csv", index=False)
    old = pd.read_csv(StringIO(frozen.decode()))
    identity = ["symbol_group", "horizon", "model"]
    unchanged_metrics = [c for c in summary if c not in identity and not c.startswith("mcs_")]
    reevaluated = summary.merge(old[identity], on=identity).sort_values(identity)
    expected = old.sort_values(identity)
    np.testing.assert_allclose(reevaluated[unchanged_metrics], expected[unchanged_metrics], rtol=1e-12, atol=1e-14)
    new = summary.loc[summary.model.isin(chronos.model.unique())].merge(
        costs[["model", "fit_seconds", "trainable_parameters", "inference_ms_per_1000", "gpu_memory_mb"]], on="model")
    append_preserving_baselines(frozen, new, results / "stage2_baselines.csv")
    new.to_csv(results / "stage3_zero_shot.csv", index=False)
    pd.DataFrame([{"stage2_rows": len(old), "baseline_bytes": len(frozen),
                   "sha256_baseline": hashlib.sha256(frozen).hexdigest(),
                   "prefix_byte_identical": True, "recomputed_non_mcs_metrics_match": True,
                   "rows_per_chronos_model": len(template), "unique_paths_per_model": len(origins)}]).to_csv(
        results / "stage3_preservation_audit.csv", index=False)
    print(new[["symbol_group", "horizon", "model", "rmse", "mae", "qlike_raw", "qlike"]].to_string(index=False))
    print("Stage 2 historical CSV prefix is byte-identical.")


if __name__ == "__main__":
    main()
