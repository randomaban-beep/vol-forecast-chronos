"""Train-only Chronos-Bolt full/LoRA adaptation with validation-only stopping."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import gc
import os
import random

import numpy as np
import pandas as pd


def training_windows(daily: pd.DataFrame, split: str, config: dict
                     ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build calendar windows, excluding held-out symbols and incomplete target paths.

    Training never reads validation/test rows. Validation contexts may use earlier
    observed validation data. Targets stay wholly within their own partition.
    Missing context entries stay NaN; require a full 256-slot calendar history.
    """
    bounds = {"train": ("2020-01-01", "2023-12-31"), "validation": ("2024-01-01", "2024-12-31")}
    if split not in bounds:
        raise ValueError("Training windows support train/validation only")
    start, end = [pd.Timestamp(x, tz="UTC") for x in bounds[split]]
    length, horizon = config["context_length"], config["prediction_length"]
    if length > 256 or length < 1 or horizon != 22:
        raise ValueError("Invalid training context/horizon")
    metadata, contexts, targets = [], [], []
    eligible = daily.loc[~daily.symbol.isin(config["held_out"]) & (daily.date <= end)]
    for symbol, group in eligible.groupby("symbol"):
        y = group.set_index("date").y.sort_index().asfreq("D")
        for i in range(length - 1, len(y) - horizon):
            target_start, target_end = y.index[i + 1], y.index[i + horizon]
            if target_start < start or target_end > end:
                continue
            target = y.iloc[i + 1:i + horizon + 1].to_numpy(dtype=np.float32)
            context = y.iloc[i - length + 1:i + 1].to_numpy(dtype=np.float32)
            if not np.isfinite(target).all() or not np.isfinite(context).any():
                continue
            if np.isinf(context).any():
                raise ValueError("Infinite context")
            metadata.append({"split": split, "symbol": symbol, "origin": y.index[i],
                             "context_start": y.index[i - length + 1],
                             "target_start": target_start, "target_end": target_end})
            contexts.append(context)
            targets.append(target)
    if not metadata:
        raise ValueError("No eligible training windows")
    return pd.DataFrame(metadata), np.stack(contexts), np.stack(targets)


@dataclass
class EarlyStop:
    """Track validation loss alone, including epoch-zero pretrained performance."""

    patience: int
    min_delta: float
    best_loss: float = float("inf")
    best_epoch: int = -1
    stale: int = 0

    def update(self, epoch: int, validation_loss: float) -> bool:
        """Return whether the checkpoint improved; reject invalid validation loss."""
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        improved = validation_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss, self.best_epoch, self.stale = validation_loss, epoch, 0
        else:
            self.stale += 1
        return improved


def adapt_model(model: object, mode: str, config: dict) -> object:
    """Train all weights, or inject PEFT LoRA on attention query/value projections."""
    if mode == "full":
        model.requires_grad_(True)
        return model
    if mode not in ("lora4", "lora8", "lora16", "lora32"):
        raise ValueError("Supported modes: full, lora4, lora8, lora16, lora32")
    from peft import LoraConfig, get_peft_model
    rank = int(mode.removeprefix("lora"))
    return get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank,
        target_modules=config["lora_targets"], lora_dropout=config["lora_dropout"], bias="none"))


def train_and_predict(model_spec: dict, mode: str, seed: int, config: dict,
                      train: tuple[np.ndarray, np.ndarray], validation: tuple[np.ndarray, np.ndarray],
                      forecast_contexts: np.ndarray, checkpoint: Path
                      ) -> tuple[np.ndarray, dict, pd.DataFrame]:
    """Select one checkpoint using train/validation, then make frozen median forecasts."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from chronos import ChronosBoltPipeline

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if not 1 <= config["batch_size"] <= 32:
        raise ValueError("Batch size exceeds hardware limit")
    pipeline = ChronosBoltPipeline.from_pretrained(model_spec["id"], revision=model_spec["revision"],
        torch_dtype=torch.float32, device_map=device, use_safetensors=True, cache_dir=str(Path(config["cache"]) / "hf"))
    model = adapt_model(pipeline.model, mode, config)
    pipeline.model = model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["full_learning_rate"] if mode == "full" else config["lora_learning_rate"],
        weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dtype == torch.float16)
    generator = torch.Generator().manual_seed(seed)
    train_x, train_y = [torch.from_numpy(v) for v in train]
    val_x, val_y = [torch.from_numpy(v) for v in validation]
    batch_size = config["batch_size"]

    def validation_loss() -> float:
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for i in range(0, len(val_x), batch_size):
                with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                    loss = model(context=val_x[i:i + batch_size].to(device),
                                 target=val_y[i:i + batch_size].to(device)).loss
                total += float(loss) * len(val_x[i:i + batch_size])
        return total / len(val_x)

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stopper = EarlyStop(config["patience"], config["min_delta"])
    history = []
    started = perf_counter()
    initial = validation_loss()
    stopper.update(0, initial)
    torch.save(model.state_dict(), checkpoint)
    history.append({"epoch": 0, "train_loss": np.nan, "validation_loss": initial, "selected": True})
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        indices = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                loss = model(context=train_x[batch].to(device), target=train_y[batch].to(device)).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(batch)
        val_loss = validation_loss()
        improved = stopper.update(epoch, val_loss)
        if improved:
            torch.save(model.state_dict(), checkpoint)
        history.append({"epoch": epoch, "train_loss": total / len(train_x),
                        "validation_loss": val_loss, "selected": improved})
        print(f"{model_spec['name']} {mode} seed={seed} epoch={epoch} val={val_loss:.6f} best={stopper.best_epoch}", flush=True)
        if stopper.stale >= stopper.patience:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = perf_counter() - started
    train_peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    del optimizer, scaler
    model.eval().requires_grad_(False)
    # Match zero-shot's fp16 inference convention; training keeps float32 master weights.
    if device == "cuda":
        model.to(torch.float16)
    median = pipeline.quantiles.index(0.5)
    with torch.inference_mode():
        pipeline.predict(torch.from_numpy(forecast_contexts[:batch_size]), prediction_length=22)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    inference_start = perf_counter()
    paths = []
    with torch.inference_mode():
        for i in range(0, len(forecast_contexts), batch_size):
            output = pipeline.predict(torch.from_numpy(forecast_contexts[i:i + batch_size]), prediction_length=22,
                                      limit_prediction_length=True)
            paths.append(output[:, median, :].numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = perf_counter() - inference_start
    cost = {"base_model": model_spec["name"], "mode": mode, "seed": seed, "revision": model_spec["revision"],
            "trainable_parameters": trainable, "fit_seconds": training_seconds,
            "inference_seconds": inference_seconds, "best_epoch": stopper.best_epoch,
            "epochs_completed": epoch, "initial_validation_loss": initial, "best_validation_loss": stopper.best_loss,
            "train_windows": len(train_x), "validation_windows": len(val_x),
            "gpu_training_peak_mb": train_peak,
            "gpu_inference_peak_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.,
            "device": device, "training_dtype": str(dtype) if device == "cuda" else "torch.float32"}
    paths_array = np.concatenate(paths)
    if not np.isfinite(paths_array).all():
        raise ValueError("Nonfinite forecast path")
    del model, pipeline, train_x, train_y, val_x, val_y
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return paths_array, cost, pd.DataFrame(history)
