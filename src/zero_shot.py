"""Calendar-preserving Chronos-Bolt zero-shot inference on the fixed common panel."""

from pathlib import Path
from time import perf_counter
import gc
import os

import numpy as np
import pandas as pd
from scipy.special import logsumexp

KEYS = ["split", "symbol", "origin", "horizon"]


def common_template(baselines: pd.DataFrame) -> pd.DataFrame:
    """Use HAR's exact eligible rows; reject an incomplete baseline model panel."""
    if baselines.duplicated(KEYS + ["model"]).any():
        raise ValueError("Duplicate baseline rows")
    if not baselines.groupby(KEYS).model.nunique().eq(baselines.model.nunique()).all():
        raise ValueError("Baseline rows differ")
    template = baselines.loc[baselines.model.eq("HAR-RV")].copy()
    if template.empty:
        raise ValueError("Missing HAR reference")
    return template.drop(columns=["model", "prediction", "inference_seconds"]).reset_index(drop=True)


def build_contexts(daily: pd.DataFrame, template: pd.DataFrame,
                   length: int = 256) -> tuple[pd.DataFrame, np.ndarray]:
    """Read origin-length+1 through origin inclusive; keep missing days as NaN."""
    if length not in (32, 64, 128, 256):
        raise ValueError("Supported context lengths: 32, 64, 128, 256")
    if daily.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate daily dates")
    origins = template[["split", "symbol", "origin"]].drop_duplicates().sort_values(
        ["split", "symbol", "origin"]).reset_index(drop=True)
    series = {s: g.set_index("date").y.sort_index() for s, g in daily.groupby("symbol")}
    contexts = []
    for row in origins.itertuples():
        dates = pd.date_range(end=row.origin, periods=length, freq="D")
        values = series[row.symbol].reindex(dates).to_numpy(dtype=np.float32)
        if np.isinf(values).any() or not np.isfinite(values).any():
            raise ValueError("Invalid or wholly missing context")
        contexts.append(values)
    origins["context_start"] = origins.origin - pd.Timedelta(days=length - 1)
    origins["context_end"] = origins.origin
    array = np.stack(contexts)
    origins["missing_days"] = np.isnan(array).sum(axis=1)
    return origins, array


def path_to_horizon(paths: np.ndarray, horizon: int) -> np.ndarray:
    """log(mean(exp(daily median log-RV forecasts))), using stable log-sum-exp."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or not 1 <= horizon <= paths.shape[1] or not np.isfinite(paths).all():
        raise ValueError("Expected finite paths covering the requested horizon")
    return logsumexp(paths[:, :horizon], axis=1) - np.log(horizon)


def align_forecasts(template: pd.DataFrame, origins: pd.DataFrame,
                    paths: np.ndarray, name: str) -> pd.DataFrame:
    """Attach path-derived forecasts to exactly the Stage 2 rows, without drops."""
    if len(paths) != len(origins):
        raise ValueError("Path/origin count mismatch")
    lookup = origins[["split", "symbol", "origin"]].copy()
    for h in sorted(template.horizon.unique()):
        lookup[f"mu_{h}"] = path_to_horizon(paths, int(h))
    merged = template.merge(lookup, on=["split", "symbol", "origin"], how="left", validate="many_to_one")
    merged["prediction"] = [row[f"mu_{row['horizon']}"] for row in merged.to_dict("records")]
    merged = merged.drop(columns=[c for c in merged if c.startswith("mu_")])
    if len(merged) != len(template) or not np.isfinite(merged.prediction).all():
        raise ValueError("Missing common-panel forecasts")
    pd.testing.assert_frame_equal(merged[KEYS], template[KEYS])
    merged["model"] = name
    return merged


def infer_bolt(contexts: np.ndarray, model_spec: dict, config: dict
               ) -> tuple[np.ndarray | None, dict]:
    """Freeze a pinned model, warm up, measure device-synchronized batched inference."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from chronos import ChronosBoltPipeline

    if not 1 <= config["batch_size"] <= 32 or contexts.shape[1] != 256:
        raise ValueError("Hardware/context limits exceeded")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    pipeline = ChronosBoltPipeline.from_pretrained(
        model_spec["id"], revision=model_spec["revision"], device_map=device,
        torch_dtype=dtype, use_safetensors=True, cache_dir=str(Path(config["cache"]) / "hf"))
    pipeline.model.eval()
    pipeline.model.requires_grad_(False)
    load_seconds = perf_counter() - started
    median = pipeline.quantiles.index(0.5)
    batch_size = config["batch_size"]

    def predict(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out = pipeline.predict(torch.from_numpy(batch), prediction_length=22,
                                   limit_prediction_length=True)
        return out[:, median, :].numpy()

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    predict(contexts[:batch_size])
    sync()
    started = perf_counter()
    predict(contexts[:batch_size])
    sync()
    batch_seconds = perf_counter() - started
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    cost = {"model": model_spec["name"], "model_id": model_spec["id"], "revision": model_spec["revision"],
            "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
            "dtype": str(dtype), "torch_version": torch.__version__, "batch_size": batch_size,
            "context_length": 256, "prediction_length": 22,
            "load_seconds": load_seconds, "probe_batch_seconds": batch_seconds,
            "total_parameters": sum(p.numel() for p in pipeline.model.parameters()),
            "trainable_parameters": 0, "fit_seconds": 0.0, "paths": len(contexts)}
    if model_spec["name"].endswith("small") and (
            peak_mb > config["small_max_peak_mb"] or batch_seconds > config["small_max_batch_seconds"]):
        cost.update(status="skipped_resource_limit", gpu_memory_mb=peak_mb)
        del pipeline
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return None, cost
    started = perf_counter()
    outputs = []
    for offset in range(0, len(contexts), batch_size):
        outputs.append(predict(contexts[offset:offset + batch_size]))
        if offset % (batch_size * 50) == 0:
            print(f"{model_spec['name']}: {offset}/{len(contexts)} paths", flush=True)
    sync()
    cost["inference_seconds"] = perf_counter() - started
    cost["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    cost["gpu_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else 0.0
    cost["status"] = "complete"
    paths = np.concatenate(outputs)
    if paths.shape != (len(contexts), 22) or not np.isfinite(paths).all():
        raise ValueError("Nonfinite or malformed Chronos paths; cannot alter panel")
    del pipeline
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return paths, cost
