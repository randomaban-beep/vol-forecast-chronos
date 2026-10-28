"""Shared log/variance scoring, HAC DM tests and dependent-data uncertainty."""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t


def _pair(actual: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape or actual.ndim != 1 or not actual.size:
        raise ValueError("Expected nonempty matching one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("Nonfinite actual or forecast")
    return actual, forecast


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root mean squared error on the supplied (log-RV) scale."""
    actual, forecast = _pair(actual, forecast)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error on the supplied (log-RV) scale."""
    actual, forecast = _pair(actual, forecast)
    return float(np.mean(np.abs(actual - forecast)))


def qlike(actual_variance: np.ndarray, forecast_variance: np.ndarray) -> np.ndarray:
    """Point losses: forecast/actual - log(forecast/actual) - 1 (spec orientation)."""
    actual, forecast = _pair(actual_variance, forecast_variance)
    if (actual <= 0).any() or (forecast <= 0).any():
        raise ValueError("QLIKE variances must be positive")
    log_ratio = np.log(forecast) - np.log(actual)
    with np.errstate(over="raise", invalid="raise"):
        return np.expm1(log_ratio) - log_ratio


def smearing_factor(actual_log: np.ndarray, forecast_log: np.ndarray) -> float:
    """Mean exp(validation residual); never estimate from test or held-out data."""
    actual, forecast = _pair(actual_log, forecast_log)
    return float(np.exp(logsumexp(actual - forecast) - np.log(len(actual))))


def score_forecasts(actual_log: np.ndarray, forecast_log: np.ndarray,
                    smearing: float) -> pd.DataFrame:
    """The sole scoring path for every model; apply frozen calibration once."""
    actual, forecast = _pair(actual_log, forecast_log)
    if not np.isfinite(smearing) or smearing <= 0:
        raise ValueError("Smearing must be positive and finite")
    with np.errstate(over="raise", invalid="raise"):
        actual_variance = np.exp(actual)
        predicted_variance = np.exp(forecast) * smearing
    return pd.DataFrame({"squared_error": (forecast - actual) ** 2,
                         "absolute_error": np.abs(forecast - actual),
                         "qlike": qlike(actual_variance, predicted_variance),
                         "qlike_raw": qlike(actual_variance, np.exp(forecast)),
                         "forecast_variance": predicted_variance})


def diebold_mariano(loss: np.ndarray, reference: np.ndarray, horizon: int) -> dict:
    """Two-sided Student-t DM with Bartlett/Newey-West HAC and Harvey correction.

    Date-level loss differences preserve simultaneous cross-symbol dependence.
    Negative statistics favor the candidate. Bandwidth covers overlap and a
    fixed sample-size rule; no test-performance tuning is performed.
    """
    loss, reference = _pair(loss, reference)
    n = len(loss)
    if horizon < 1 or n <= horizon:
        raise ValueError("DM requires n > horizon >= 1")
    d = loss - reference
    centered = d - d.mean()
    bandwidth = min(n - 1, max(horizon - 1, int(np.floor(4 * (n / 100) ** (2 / 9)))))
    long_run = float(centered @ centered / n)
    for lag in range(1, bandwidth + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        long_run += 2 * (1 - lag / (bandwidth + 1)) * covariance
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    if long_run <= 1e-30:
        statistic = 0.0 if np.isclose(d.mean(), 0, atol=1e-15) else float(np.sign(d.mean()) * np.inf)
    else:
        statistic = float(d.mean() / np.sqrt(long_run / n) * correction)
    return {"dm_stat": statistic, "dm_pvalue": float(2 * student_t.sf(abs(statistic), df=n - 1)),
            "dm_hac_lags": bandwidth, "dm_harvey_factor": float(correction)}


def block_bootstrap_ci(daily_losses: np.ndarray, block_length: int,
                       reps: int, seed: int) -> np.ndarray:
    """95% circular block CIs, resampling whole dates jointly.

    Input columns are date-mean squared error, absolute error, calibrated QLIKE,
    and optionally raw QLIKE.
    Square root is applied after averaging each resampled squared-error path.
    """
    values = np.asarray(daily_losses, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (3, 4) or not np.isfinite(values).all():
        raise ValueError("Expected finite date-by-three/four losses")
    n = len(values)
    if not 1 <= block_length <= n or reps < 2:
        raise ValueError("Invalid bootstrap dimensions")
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(reps, int(np.ceil(n / block_length))))
    indices = ((starts[..., None] + np.arange(block_length)) % n).reshape(reps, -1)[:, :n]
    boot = values[indices].mean(axis=1)
    boot[:, 0] = np.sqrt(boot[:, 0])
    return np.quantile(boot, [0.025, 0.975], axis=0)


def evaluate_panel(predictions: pd.DataFrame, config: dict
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate on validation and score a complete common test panel, failing closed."""
    keys = ["split", "symbol", "origin", "horizon"]
    models = sorted(predictions.model.unique())
    if predictions.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate forecasts")
    if not predictions.groupby(keys).model.nunique().eq(len(models)).all():
        raise ValueError("Model comparison rows differ")
    if not np.isfinite(predictions[["target", "prediction"]]).all().all():
        raise ValueError("Failed/nonfinite forecasts; refusing to change sample")
    if not predictions.groupby(keys).target.nunique().eq(1).all():
        raise ValueError("Models have different targets")
    held_out = config["held_out"]
    calibrations, scored = [], []
    for (model, h), rows in predictions.groupby(["model", "horizon"]):
        validation = rows.loc[rows["split"].eq("validation") & ~rows.symbol.isin(held_out)]
        if validation.target_end.max() >= pd.Timestamp("2025-01-01", tz="UTC"):
            raise ValueError("Calibration extends into test")
        factor = smearing_factor(validation.target.to_numpy(), validation.prediction.to_numpy())
        calibrations.append({"model": model, "horizon": h, "smearing_factor": factor,
                             "n_validation": len(validation), "last_target": validation.target_end.max()})
        test = rows.loc[rows["split"].eq("test")].reset_index(drop=True)
        losses = score_forecasts(test.target.to_numpy(), test.prediction.to_numpy(), factor)
        scored.append(pd.concat([test, losses], axis=1))
    scored_frame = pd.concat(scored, ignore_index=True)
    scored_frame["symbol_group"] = np.where(scored_frame.symbol.isin(held_out), "held-out", "fit-eligible")
    summaries = []
    for (group, h), panel in scored_frame.groupby(["symbol_group", "horizon"]):
        counts = panel.groupby(["origin", "model"]).symbol.nunique()
        if not counts.eq(panel.symbol.nunique()).all():
            raise ValueError("Unbalanced date panel")
        daily = panel.groupby(["origin", "model"])[["squared_error", "absolute_error", "qlike", "qlike_raw"]].mean()
        qloss = daily.qlike.unstack("model")
        raw_loss = daily.qlike_raw.unstack("model")
        if qloss.isna().any().any():
            raise ValueError("Missing common date")
        mcs_values = {}
        try:
            from arch.bootstrap import MCS
        except ImportError:
            pass
        else:
            for name, losses in ([("calibrated", qloss), ("raw", raw_loss)]
                                 if config.get("compute_mcs", True) else []):
                mcs = MCS(losses, size=0.05, reps=config["bootstrap_reps"],
                          block_size=config["block_length"], method="R", bootstrap="circular", seed=config["seed"])
                mcs.compute()
                mcs_values[name] = mcs.pvalues.iloc[:, 0]
        for model in models:
            values = daily.xs(model, level="model").to_numpy()
            means = values.mean(axis=0)
            means[0] = np.sqrt(means[0])
            ci = block_bootstrap_ci(values, config["block_length"], config["bootstrap_reps"], config["seed"])
            record = {"symbol_group": group, "horizon": h, "model": model,
                      "n_dates": len(values), "n_forecasts": int(panel.model.eq(model).sum()),
                      **diebold_mariano(qloss[model].to_numpy(), qloss["HAR-RV"].to_numpy(), int(h))}
            record.update({key + "_raw": value for key, value in diebold_mariano(
                raw_loss[model].to_numpy(), raw_loss["HAR-RV"].to_numpy(), int(h)).items()})
            for i, metric in enumerate(["rmse", "mae", "qlike", "qlike_raw"]):
                record.update({metric: means[i], metric + "_lo": ci[0, i], metric + "_hi": ci[1, i]})
            for name in ["calibrated", "raw"]:
                pvalues = mcs_values.get(name)
                record[f"mcs_pvalue_{name}"] = np.nan if pvalues is None else float(pvalues.loc[model])
                record[f"mcs_95_included_{name}"] = None if pvalues is None else bool(pvalues.loc[model] >= 0.05)
            summaries.append(record)
    return pd.DataFrame(summaries), pd.DataFrame(calibrations), scored_frame
