"""Reproduce Stage 1 inputs and all Stage 2 baseline result tables."""

import argparse
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import walk_forward
from src.evaluate import evaluate_panel


def main() -> None:
    """Run the complete pipeline, or explicitly reuse existing verified daily inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage-2-baselines.yaml")
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    if not args.reuse_data:
        subprocess.run([sys.executable, "scripts/download_data.py"], check=True)
    config = yaml.safe_load(Path(args.config).read_text())
    cache, results = Path(config["cache"]), Path(config["results"])
    results.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(cache / "daily_rv.parquet")
    examples = pd.read_parquet(cache / "examples.parquet")
    if "return" not in daily:
        raise ValueError("Rebuild Stage 1 caches to derive close-to-close returns")
    returns_audit = daily.groupby(["symbol", "split"]).agg(
        days=("date", "size"), valid_rv=("rv", "count"), valid_returns=("return", "count"))
    returns_audit.to_csv(results / "stage2_return_coverage.csv")
    predictions, fits = walk_forward(daily, examples, config)
    predictions.to_parquet(cache / "stage2_predictions.parquet", index=False)
    fits.to_parquet(cache / "stage2_fits.parquet", index=False)
    fits.to_csv(results / "stage2_fit_audit.csv", index=False)
    summary, calibration, scored = evaluate_panel(predictions, config)
    scored.to_parquet(cache / "stage2_scored.parquet", index=False)
    calibration.to_csv(results / "stage2_calibration.csv", index=False)
    costs = predictions.groupby("model").agg(
        inference_seconds=("inference_seconds", "sum"), forecasts=("prediction", "size"))
    costs["inference_ms_per_1000"] = 1e6 * costs.inference_seconds / costs.forecasts
    costs["fit_seconds"] = fits.groupby("model").fit_seconds.sum()
    costs["trainable_parameters"] = pd.Series({"Naive": 0, "HAR-RV": 4, "HAR-log-mean-RV": 4, "GARCH-t": 4, "GJR-GARCH-t": 5})
    costs["fit_seconds"] = costs.fit_seconds.fillna(0)
    costs["gpu_memory_mb"] = 0
    costs.to_csv(results / "stage2_costs.csv")
    summary = summary.merge(costs[["fit_seconds", "trainable_parameters", "inference_ms_per_1000", "gpu_memory_mb"]],
                            left_on="model", right_index=True)
    summary.to_csv(results / "stage2_baselines.csv", index=False)
    print(summary[["symbol_group", "horizon", "model", "rmse", "mae", "qlike", "dm_pvalue"]].to_string(index=False))
    print(f"Stage 2 elapsed seconds: {perf_counter() - started:.1f}")


if __name__ == "__main__":
    main()
