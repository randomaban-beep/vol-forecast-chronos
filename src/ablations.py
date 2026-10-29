"""Calendar-only HAR features, chronological fractions and preregistered ablations."""

from time import perf_counter
import numpy as np
import pandas as pd

from src.baselines import FEATURES, fit_har

WEEKDAYS = [f"weekday_{day}" for day in range(1, 7)]


def weekday_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Six weekday dummies; Monday reference, h=1 target date, otherwise origin.

    The h=1 target calendar date is computed from origin, never read from RV.
    """
    out = rows.copy()
    dates = out.origin + pd.to_timedelta(out.horizon.eq(1).astype(int), unit="D")
    for day, name in enumerate(WEEKDAYS, start=1):
        out[name] = dates.dt.dayofweek.eq(day).astype(float)
    return out


def weekday_walk_forward(examples: pd.DataFrame, template: pd.DataFrame,
                         held_out: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit pooled HAR with weekday dummies at the exact existing refit cutoffs."""
    features = FEATURES + WEEKDAYS
    source, panel = weekday_features(examples), weekday_features(template)
    outputs, audit = [], []
    for (split, cutoff, h), rows in panel.groupby(["split", "refit", "horizon"], sort=True):
        started = perf_counter()
        coef = fit_har(source, cutoff, int(h), held_out, features)
        fitted = perf_counter() - started
        started = perf_counter()
        out = rows.copy()
        out["prediction"] = np.column_stack([np.ones(len(out)), out[features]]) @ coef
        out["model"] = "HAR-weekday"
        out["inference_seconds"] = (perf_counter() - started) / len(out)
        outputs.append(out.drop(columns=WEEKDAYS))
        audit.append({"split": split, "refit": cutoff, "horizon": h, "fit_seconds": fitted,
                      **dict(zip(["intercept"] + features, coef))})
    return pd.concat(outputs).sort_index().reset_index(drop=True), pd.DataFrame(audit)


def earliest_fraction(metadata: pd.DataFrame, fraction: float) -> np.ndarray:
    """Choose floor(fraction*N) earliest origins, breaking date ties by symbol.

    Preserve original sample order after selecting membership so fraction=1
    exactly preserves Stage 4's seed-dependent training order.
    """
    if not 0 < fraction <= 1 or metadata.empty:
        raise ValueError("Fraction must be in (0,1] with nonempty data")
    if not metadata["split"].eq("train").all():
        raise ValueError("Fractions apply only to training windows")
    count = max(1, int(np.floor(len(metadata) * fraction)))
    ordered = metadata.reset_index(drop=True).sort_values(["origin", "symbol"], kind="stable")
    return np.sort(ordered.index.to_numpy()[:count])


def ablation_grid(config: dict) -> pd.DataFrame:
    """One-factor-at-a-time grid; duplicate shared anchor configurations reuse runs."""
    rows = []
    for mode in config["main_modes"]:
        for length in config["contexts"]:
            rows.append({"axis": "context", "value": float(length), "mode": mode, "context": length, "fraction": 1.})
        for fraction in config["fractions"]:
            rows.append({"axis": "fraction", "value": fraction, "mode": mode, "context": 256, "fraction": fraction})
    for rank in config["ranks"]:
        rows.append({"axis": "rank", "value": float(rank), "mode": f"lora{rank}", "context": 256, "fraction": 1.})
    grid = pd.DataFrame(rows)
    grid["family"] = [f"small-{r.mode}-ctx{r.context}-frac{int(round(r.fraction*100))}" for r in grid.itertuples()]
    return grid


def select_stage4_headlines(predictions: pd.DataFrame, held_out: list[str]) -> pd.DataFrame:
    """Freeze per-horizon Stage 4 families using six-symbol validation RMSE only."""
    val = predictions.loc[predictions["split"].eq("validation") & ~predictions.symbol.isin(held_out)].copy()
    val["family"] = val.model.str.replace(r"-seed\d+$", "", regex=True)
    val["error2"] = (val.prediction - val.target) ** 2
    per_seed = val.groupby(["horizon", "family", "model"]).error2.mean().pow(.5)
    means = per_seed.groupby(["horizon", "family"]).mean().rename("validation_rmse").reset_index()
    means["selected"] = False
    for _, group in means.groupby("horizon"):
        selected = group.sort_values(["validation_rmse", "family"]).index[0]
        means.loc[selected, "selected"] = True
    return means


def extended_family(costs: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Among tiny families with a Stage 4 best epoch at cap, use mean validation loss."""
    tiny = costs.loc[costs.base_model.eq("Chronos-Bolt-tiny") & costs["mode"].str.startswith("lora")]
    audit = tiny.groupby("mode").agg(cap_hit=("best_epoch", lambda x: bool(x.eq(3).any())),
        mean_validation_loss=("best_validation_loss", "mean")).reset_index()
    eligible = audit.loc[audit.cap_hit].sort_values(["mean_validation_loss", "mode"])
    if eligible.empty:
        return "", audit.assign(selected=False)
    chosen = eligible.iloc[0]["mode"]
    return chosen, audit.assign(selected=audit["mode"].eq(chosen))
