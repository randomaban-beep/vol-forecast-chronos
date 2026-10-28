"""Mean-over-seeds metrics, paired seed/date uncertainty and joint model-set MCS."""

import numpy as np
import pandas as pd
from arch.bootstrap import MCS

from src.evaluate import diebold_mariano

LOSS_COLUMNS = ["squared_error", "absolute_error", "qlike", "qlike_raw"]


def seed_statistics(losses: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average seed-specific metrics, not predictions; resample seeds and date blocks.

    Input axes: seed, date, loss. All symbols were averaged within each date.
    The same time blocks are used for all sampled seeds in a replication.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 3 or losses.shape[2] != 4 or not np.isfinite(losses).all():
        raise ValueError("Expected finite seed-by-date-by-four losses")
    seeds, dates, _ = losses.shape
    if seeds < 2 or dates < config["block_length"]:
        raise ValueError("Insufficient seeds/dates")
    per_seed = losses.mean(axis=1)
    per_seed[:, 0] = np.sqrt(per_seed[:, 0])
    rng = np.random.default_rng(config["seed"])
    reps, block = config["bootstrap_reps"], config["block_length"]
    starts = rng.integers(dates, size=(reps, int(np.ceil(dates / block))))
    indices = ((starts[..., None] + np.arange(block)) % dates).reshape(reps, -1)[:, :dates]
    chosen_seeds = rng.integers(seeds, size=(reps, seeds))
    boot = losses[chosen_seeds[:, :, None], indices[:, None, :]].mean(axis=2)
    boot[:, :, 0] = np.sqrt(boot[:, :, 0])
    ci = np.quantile(boot.mean(axis=1), [0.025, 0.975], axis=0)
    return per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1), ci


def summarize_seeds(scored: pd.DataFrame, summaries: pd.DataFrame, costs: pd.DataFrame,
                    config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare averaged seed losses against baselines with one candidate per family."""
    scored = scored.copy()
    scored["family"] = scored.model.str.replace(r"-seed\d+$", "", regex=True)
    costs = costs.copy()
    costs["family"] = costs.model.str.replace(r"-seed\d+$", "", regex=True)
    # Averaging losses, rather than predictions, avoids introducing a seed ensemble.
    panel = scored.groupby(["symbol_group", "horizon", "origin", "family"])[LOSS_COLUMNS].mean()
    result, mcs_rows = [], []
    for (group, h), subset in panel.groupby(level=["symbol_group", "horizon"]):
        date_losses = subset.droplevel(["symbol_group", "horizon"])
        mcs_by_variant = {}
        for metric in ["qlike", "qlike_raw"]:
            losses = date_losses[metric].unstack("family")
            if losses.isna().any().any():
                raise ValueError("Unbalanced joint family panel")
            mcs = MCS(losses, size=.05, reps=config["bootstrap_reps"], block_size=config["block_length"],
                      method="R", bootstrap="circular", seed=config["seed"])
            mcs.compute()
            mcs_by_variant[metric] = mcs.pvalues.iloc[:, 0]
            for family in losses:
                mcs_rows.append({"symbol_group": group, "horizon": h, "model": family,
                    "metric": metric, "mcs_pvalue": mcs_by_variant[metric].loc[family],
                    "mcs_95_included": bool(mcs_by_variant[metric].loc[family] >= .05),
                    "model_universe": "|".join(losses.columns)})
        for family, family_cost in costs.groupby("family"):
            if set(family_cost.seed) != set(config["seeds"]):
                raise ValueError("Incomplete seed set")
            trials = scored.loc[scored.family.eq(family) & scored.symbol_group.eq(group) & scored.horizon.eq(h)]
            means = trials.groupby(["model", "origin"])[LOSS_COLUMNS].mean()
            values = np.stack([means.xs(model, level="model").sort_index().to_numpy()
                               for model in sorted(trials.model.unique())])
            estimate, std, ci = seed_statistics(values, config)
            row = {"model": family, "symbol_group": group, "horizon": h, "seeds": len(family_cost),
                   "n_dates": values.shape[1], "n_forecasts_per_seed": len(trials) // len(family_cost),
                   "trainable_parameters": int(family_cost.trainable_parameters.iloc[0]),
                   "fit_seconds_mean": family_cost.fit_seconds.mean(),
                   "inference_ms_per_1000_mean": family_cost.inference_ms_per_1000.mean(),
                   "gpu_training_peak_mb_max": family_cost.gpu_training_peak_mb.max()}
            for i, metric in enumerate(["rmse", "mae", "qlike", "qlike_raw"]):
                row.update({metric: estimate[i], metric + "_seed_sd": std[i],
                            metric + "_lo": ci[0, i], metric + "_hi": ci[1, i]})
            for metric in ["qlike", "qlike_raw"]:
                losses = date_losses[metric].unstack("family")
                dm = diebold_mariano(losses[family].to_numpy(), losses["HAR-RV"].to_numpy(), int(h))
                suffix = "_raw" if metric.endswith("raw") else ""
                row.update({k + suffix: v for k, v in dm.items()})
                row[f"mcs_pvalue_{metric}"] = mcs_by_variant[metric].loc[family]
                row[f"mcs_95_included_{metric}"] = bool(mcs_by_variant[metric].loc[family] >= .05)
            result.append(row)
    return pd.DataFrame(result), pd.DataFrame(mcs_rows)
