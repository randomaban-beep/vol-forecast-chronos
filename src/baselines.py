"""Pooled causal baselines, with no parameter fitting on held-out symbols."""

from dataclasses import dataclass
from time import perf_counter

from arch.univariate import StudentsT
from numba import njit
import numpy as np
import pandas as pd
from scipy.optimize import minimize

FEATURES = ["daily", "weekly", "monthly"]
RV_FEATURES = ["daily", "weekly_rv", "monthly_rv"]


def fit_har(examples: pd.DataFrame, cutoff: pd.Timestamp, horizon: int,
            held_out: list[str], features: list[str] = FEATURES) -> np.ndarray:
    """OLS on all six-symbol labels observed by cutoff, with intercept."""
    rows = examples.loc[(examples.target_end <= cutoff) & examples.horizon.eq(horizon)
                        & ~examples.symbol.isin(held_out)]
    if len(rows) < 30:
        raise ValueError("Insufficient HAR training history")
    x = np.column_stack([np.ones(len(rows)), rows[features].to_numpy()])
    return np.linalg.lstsq(x, rows.target.to_numpy(), rcond=None)[0]


@njit(cache=True)
def variance_path(returns: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Conditional percent-return variances, plus next day's forecast.

    Parameters are omega, alpha, gamma, beta, nu. Missing shocks propagate
    expected variance; they contribute no likelihood and never compress time.
    Initial variance is stationary unconditional variance. Zero return mean.
    """
    omega, alpha, gamma, beta = parameters[:4]
    persistence = alpha + gamma / 2 + beta
    variances = np.empty(len(returns) + 1)
    variances[0] = omega / (1 - persistence)
    for i in range(len(returns)):
        if np.isfinite(returns[i]):
            shock = returns[i] ** 2
            variances[i + 1] = omega + (alpha + gamma * (returns[i] < 0)) * shock + beta * variances[i]
        else:
            variances[i + 1] = omega + persistence * variances[i]
    return variances


@dataclass
class GarchFit:
    """Shared GARCH parameters and fit diagnostics."""

    parameters: np.ndarray
    seconds: float
    observations: int
    iterations: int
    objective: float


def fit_garch(returns: pd.DataFrame, asymmetric: bool, maxiter: int = 300,
              previous: np.ndarray | None = None) -> GarchFit:
    """Joint six-series Student-t likelihood using arch's standardized density.

    Each column has an independent variance recursion, sharing parameters.
    Missing-return intervals use expected-shock filtering (quasi likelihood).
    Constrained SLSQP estimates parameters; held-out columns must be excluded.
    """
    started = perf_counter()
    histories = [np.ascontiguousarray(100 * returns[c].to_numpy()) for c in returns]
    masks = [np.isfinite(r) for r in histories]
    observations = sum(int(mask.sum()) for mask in masks)
    if observations < 100:
        raise ValueError("Insufficient GARCH history")
    density = StudentsT()

    def objective(p: np.ndarray) -> float:
        if p[1] + p[2] / 2 + p[3] >= 1 or p[1] + p[2] < 0:
            return 1e10
        total = 0.0
        for r, mask in zip(histories, masks):
            variance = variance_path(r, p)[:-1]
            total -= density.loglikelihood([p[4]], r[mask], variance[mask])
        return float(total / observations)

    variance = float(np.nanmean(np.concatenate(histories) ** 2))
    default = np.array([max(variance * 0.05, 1e-4), 0.08,
                        0.04 if asymmetric else 0.0, 0.85, 8.0])
    starts = [default] if previous is None else [previous, default]
    bounds = [(1e-8, 1000), (0, 0.999), (-0.999, 0.999) if asymmetric else (0, 0),
              (0, 0.999), (2.05, 100)]
    candidates = []
    for start in starts:
        result = minimize(objective, start, method="SLSQP", bounds=bounds,
                          constraints=[{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2] / 2 - p[3]},
                                       {"type": "ineq", "fun": lambda p: p[1] + p[2]}],
                          options={"maxiter": maxiter, "ftol": 1e-9})
        if result.success and np.isfinite(result.fun) and result.x[1] + result.x[2]/2 + result.x[3] <= 0.999001:
            candidates.append(result)
    if not candidates:
        raise RuntimeError(f"GARCH optimization failed: {result.message}")
    best = min(candidates, key=lambda r: r.fun)
    return GarchFit(best.x, perf_counter() - started, observations, best.nit, float(best.fun))


def garch_forecasts(returns: pd.Series, parameters: np.ndarray,
                    horizons: list[int]) -> pd.DataFrame:
    """Filter each observed prefix and forecast mean daily variance in decimal units."""
    next_variance = variance_path(np.ascontiguousarray(100 * returns.to_numpy()), parameters)[1:]
    omega, alpha, gamma, beta = parameters[:4]
    persistence = alpha + gamma / 2 + beta
    forecasts = {}
    for h in horizons:
        step, total = next_variance.copy(), next_variance.copy()
        for _ in range(1, h):
            step = omega + persistence * step
            total += step
        forecasts[h] = np.log(total / h / 10000)
    return pd.DataFrame(forecasts, index=returns.index)


def walk_forward(daily: pd.DataFrame, examples: pd.DataFrame, config: dict
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding pooled fits every 30 calendar days, separately in val/test.

    Only six fit symbols enter estimation at any refit. Held-out series supply
    observed inputs/state, never fitting or calibration targets.
    """
    held_out, horizons = config["held_out"], config["horizons"]
    returns = daily.pivot(index="date", columns="symbol", values="return").asfreq("D")
    fit_symbols = [s for s in returns if s not in held_out]
    predictions, diagnostics = [], []
    previous: dict[str, np.ndarray] = {}
    for split in ["validation", "test"]:
        rows = examples.loc[examples["split"].eq(split)].copy()
        anchor = rows.origin.min()
        rows["refit"] = anchor + pd.to_timedelta(
            ((rows.origin - anchor).dt.days // config["refit_days"]) * config["refit_days"], unit="D")
        for cutoff, batch in rows.groupby("refit", sort=True):
            print(f"{split} refit {cutoff.date()}", flush=True)
            history = returns.loc[:cutoff, fit_symbols]
            fits = {}
            for model, asymmetric in [("GARCH-t", False), ("GJR-GARCH-t", True)]:
                fit = fit_garch(history, asymmetric, config["garch_maxiter"], previous.get(model))
                previous[model] = fit.parameters
                fits[model] = fit
                diagnostics.append({"split": split, "refit": cutoff, "model": model,
                                    "horizon": 0, "fit_seconds": fit.seconds,
                                    "observations": fit.observations, "parameters": 5 if asymmetric else 4,
                                    "iterations": fit.iterations, "objective": fit.objective,
                                    **dict(zip(["omega", "alpha", "gamma", "beta", "nu"], fit.parameters))})
            for h, horizon_rows in batch.groupby("horizon"):
                har_fits = {}
                for model, features in [("HAR-RV", FEATURES), ("HAR-log-mean-RV", RV_FEATURES)]:
                    start = perf_counter()
                    har_fits[model] = fit_har(examples, cutoff, int(h), held_out, features)
                    diagnostics.append({"split": split, "refit": cutoff, "model": model,
                                        "horizon": h, "fit_seconds": perf_counter() - start,
                                        "parameters": 4})
                for model in ["Naive", "HAR-RV", "HAR-log-mean-RV", "GARCH-t", "GJR-GARCH-t"]:
                    start = perf_counter()
                    out = horizon_rows.copy()
                    if model == "Naive":
                        out["prediction"] = out.daily
                    elif model in har_fits:
                        features = FEATURES if model == "HAR-RV" else RV_FEATURES
                        out["prediction"] = np.column_stack([np.ones(len(out)), out[features]]) @ har_fits[model]
                    else:
                        values = []
                        for symbol, symbol_rows in out.groupby("symbol"):
                            path = garch_forecasts(returns.loc[:symbol_rows.origin.max(), symbol],
                                                   fits[model].parameters, [int(h)])
                            values.append(pd.Series(path.loc[symbol_rows.origin, h].to_numpy(), index=symbol_rows.index))
                        out["prediction"] = pd.concat(values).reindex(out.index)
                    elapsed = perf_counter() - start
                    out["model"], out["inference_seconds"] = model, elapsed / len(out)
                    predictions.append(out)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(diagnostics)
