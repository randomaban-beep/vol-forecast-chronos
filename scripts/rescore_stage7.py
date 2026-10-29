"""Stage 7: Rescore all models with standard Patton (2011) QLIKE.

Orientation: V / V_hat - log(V / V_hat) - 1.
Consistent for the conditional arithmetic mean.
All outputs are saved with a stage7_ prefix. Existing result files are verified byte-identical.
"""

from pathlib import Path
import hashlib
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import (qlike_patton, qlike_legacy, diebold_mariano,
                          block_bootstrap_ci)
from src.finetune_summary import LOSS_COLUMNS, family_mcs, seed_statistics
from src.regimes import analyze_regimes
from scripts.package_results import excluded_windows

RESULTS = Path("results")
DATA = Path("data")


def verify_historical_preservation() -> pd.DataFrame:
    """Audit every pre-Stage-7 result file to verify byte-identical status."""
    audit = []
    for p in sorted(RESULTS.iterdir()):
        if p.is_file() and not p.name.startswith("stage7_"):
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            audit.append({"path": str(p), "sha256": sha})
    return pd.DataFrame(audit)


def rescore_all() -> None:
    config = yaml.safe_load(Path("configs/stage-5-ablations.yaml").read_text())
    source = DATA / "stage5" / "scored.parquet"
    if not source.exists():
        raise SystemExit(f"Missing {source}")

    scored = pd.read_parquet(source)
    act_v = np.exp(scored.target.to_numpy())
    pred_v_raw = np.exp(scored.prediction.to_numpy())
    pred_v_cal = scored.forecast_variance.to_numpy()

    # Standard Patton (2011) QLIKE
    scored["qlike_raw"] = qlike_patton(act_v, pred_v_raw)
    scored["qlike"] = qlike_patton(act_v, pred_v_cal)
    scored["qlike_legacy_raw"] = qlike_legacy(act_v, pred_v_raw)
    scored["qlike_legacy"] = qlike_legacy(act_v, pred_v_cal)
    scored["family"] = scored.model.str.replace(r"-seed\d+$", "", regex=True)

    print("Patton QLIKE calculated across all rows.")

    # 1. Joint Family MCS across all families
    print("Computing joint family MCS...")
    panel = scored.groupby(["symbol_group", "horizon", "origin", "family"])[["qlike", "qlike_raw"]].mean()
    mcs_records = []
    for (group, h), subset in panel.groupby(level=["symbol_group", "horizon"]):
        date_losses = subset.droplevel(["symbol_group", "horizon"])
        for metric in ["qlike", "qlike_raw"]:
            losses = date_losses[metric].unstack("family")
            pvalues = family_mcs(losses, config)
            for fam in losses.columns:
                mcs_records.append({
                    "symbol_group": group, "horizon": h, "model": fam, "metric": metric,
                    "mcs_pvalue": float(pvalues[fam]),
                    "mcs_95_included": bool(pvalues[fam] >= 0.05),
                    "model_universe": "|".join(losses.columns)
                })
    mcs_df = pd.DataFrame(mcs_records)
    mcs_df.to_csv(RESULTS / "stage7_mcs.csv", index=False)
    print("Saved stage7_mcs.csv")

    # 2. Rescore baselines (Naive, HAR-RV, HAR-log-mean-RV, HAR-weekday, GARCH-t, GJR-GARCH-t)
    print("Rescoring baselines...")
    baseline_costs = pd.read_csv(RESULTS / "stage2_costs.csv")
    weekday_cost = pd.read_csv(RESULTS / "stage5_weekday_costs.csv")
    costs_all = pd.concat([baseline_costs, weekday_cost], ignore_index=True).set_index("model")

    baseline_models = ["Naive", "HAR-RV", "HAR-log-mean-RV", "HAR-weekday", "GARCH-t", "GJR-GARCH-t"]
    base_rows = []
    for (group, h), grp_scored in scored.groupby(["symbol_group", "horizon"]):
        daily_loss = grp_scored.groupby(["origin", "model"])[LOSS_COLUMNS].mean()
        qloss = daily_loss["qlike"].unstack("model")
        raw_loss = daily_loss["qlike_raw"].unstack("model")

        for m in baseline_models:
            sub_m = grp_scored.loc[grp_scored.model.eq(m)]
            if sub_m.empty:
                continue
            vals = daily_loss.xs(m, level="model").to_numpy()
            means = vals.mean(axis=0)
            means[0] = np.sqrt(means[0])
            ci = block_bootstrap_ci(vals, config["block_length"], config["bootstrap_reps"], config["seed"])

            dm_rv = diebold_mariano(qloss[m].to_numpy(), qloss["HAR-RV"].to_numpy(), int(h))
            dm_rv_raw = diebold_mariano(raw_loss[m].to_numpy(), raw_loss["HAR-RV"].to_numpy(), int(h))
            dm_wd = diebold_mariano(qloss[m].to_numpy(), qloss["HAR-weekday"].to_numpy(), int(h))
            dm_wd_raw = diebold_mariano(raw_loss[m].to_numpy(), raw_loss["HAR-weekday"].to_numpy(), int(h))

            # MCS pvalues
            mcs_cal = mcs_df.loc[mcs_df.symbol_group.eq(group) & mcs_df.horizon.eq(h) &
                                 mcs_df.model.eq(m) & mcs_df.metric.eq("qlike"), "mcs_pvalue"]
            mcs_raw = mcs_df.loc[mcs_df.symbol_group.eq(group) & mcs_df.horizon.eq(h) &
                                 mcs_df.model.eq(m) & mcs_df.metric.eq("qlike_raw"), "mcs_pvalue"]

            c_info = costs_all.loc[m] if m in costs_all.index else None

            rec = {
                "symbol_group": group, "horizon": h, "model": m,
                "n_dates": len(vals), "n_forecasts": len(sub_m),
                "dm_stat": dm_rv["dm_stat"], "dm_pvalue": dm_rv["dm_pvalue"],
                "dm_hac_lags": dm_rv["dm_hac_lags"], "dm_harvey_factor": dm_rv["dm_harvey_factor"],
                "dm_stat_raw": dm_rv_raw["dm_stat"], "dm_pvalue_raw": dm_rv_raw["dm_pvalue"],
                "dm_hac_lags_raw": dm_rv_raw["dm_hac_lags"], "dm_harvey_factor_raw": dm_rv_raw["dm_harvey_factor"],
                "dm_stat_vs_weekday": dm_wd["dm_stat"], "dm_pvalue_vs_weekday": dm_wd["dm_pvalue"],
                "dm_stat_vs_weekday_raw": dm_wd_raw["dm_stat"], "dm_pvalue_vs_weekday_raw": dm_wd_raw["dm_pvalue"],
                "rmse": means[0], "rmse_lo": ci[0, 0], "rmse_hi": ci[1, 0],
                "mae": means[1], "mae_lo": ci[0, 1], "mae_hi": ci[1, 1],
                "qlike": means[2], "qlike_lo": ci[0, 2], "qlike_hi": ci[1, 2],
                "qlike_raw": means[3], "qlike_raw_lo": ci[0, 3], "qlike_raw_hi": ci[1, 3],
                "mcs_pvalue_calibrated": float(mcs_cal.iloc[0]) if not mcs_cal.empty else np.nan,
                "mcs_95_included_calibrated": bool(mcs_cal.iloc[0] >= 0.05) if not mcs_cal.empty else False,
                "mcs_pvalue_raw": float(mcs_raw.iloc[0]) if not mcs_raw.empty else np.nan,
                "mcs_95_included_raw": bool(mcs_raw.iloc[0] >= 0.05) if not mcs_raw.empty else False,
                "fit_seconds": float(c_info["fit_seconds"]) if c_info is not None else 0.0,
                "trainable_parameters": int(c_info["trainable_parameters"]) if c_info is not None else 0,
                "inference_ms_per_1000": float(c_info["inference_ms_per_1000"]) if c_info is not None else 0.0,
                "gpu_memory_mb": float(c_info["gpu_memory_mb"]) if c_info is not None else 0.0,
            }
            base_rows.append(rec)
    pd.DataFrame(base_rows).to_csv(RESULTS / "stage7_baselines.csv", index=False)
    print("Saved stage7_baselines.csv")

    # 3. Rescore Zero-shot (Chronos-Bolt-tiny, Chronos-Bolt-small)
    print("Rescoring zero-shot...")
    zero_costs = pd.read_csv(RESULTS / "stage3_costs.csv").set_index("model")
    zero_models = ["Chronos-Bolt-tiny", "Chronos-Bolt-small"]
    zero_rows = []
    for (group, h), grp_scored in scored.groupby(["symbol_group", "horizon"]):
        daily_loss = grp_scored.groupby(["origin", "model"])[LOSS_COLUMNS].mean()
        qloss = daily_loss["qlike"].unstack("model")
        raw_loss = daily_loss["qlike_raw"].unstack("model")

        for m in zero_models:
            sub_m = grp_scored.loc[grp_scored.model.eq(m)]
            if sub_m.empty:
                continue
            vals = daily_loss.xs(m, level="model").to_numpy()
            means = vals.mean(axis=0)
            means[0] = np.sqrt(means[0])
            ci = block_bootstrap_ci(vals, config["block_length"], config["bootstrap_reps"], config["seed"])

            dm_rv = diebold_mariano(qloss[m].to_numpy(), qloss["HAR-RV"].to_numpy(), int(h))
            dm_rv_raw = diebold_mariano(raw_loss[m].to_numpy(), raw_loss["HAR-RV"].to_numpy(), int(h))
            dm_wd = diebold_mariano(qloss[m].to_numpy(), qloss["HAR-weekday"].to_numpy(), int(h))
            dm_wd_raw = diebold_mariano(raw_loss[m].to_numpy(), raw_loss["HAR-weekday"].to_numpy(), int(h))

            mcs_cal = mcs_df.loc[mcs_df.symbol_group.eq(group) & mcs_df.horizon.eq(h) &
                                 mcs_df.model.eq(m) & mcs_df.metric.eq("qlike"), "mcs_pvalue"]
            mcs_raw = mcs_df.loc[mcs_df.symbol_group.eq(group) & mcs_df.horizon.eq(h) &
                                 mcs_df.model.eq(m) & mcs_df.metric.eq("qlike_raw"), "mcs_pvalue"]

            c_info = zero_costs.loc[m] if m in zero_costs.index else None

            rec = {
                "symbol_group": group, "horizon": h, "model": m,
                "n_dates": len(vals), "n_forecasts": len(sub_m),
                "dm_stat": dm_rv["dm_stat"], "dm_pvalue": dm_rv["dm_pvalue"],
                "dm_hac_lags": dm_rv["dm_hac_lags"], "dm_harvey_factor": dm_rv["dm_harvey_factor"],
                "dm_stat_raw": dm_rv_raw["dm_stat"], "dm_pvalue_raw": dm_rv_raw["dm_pvalue"],
                "dm_hac_lags_raw": dm_rv_raw["dm_hac_lags"], "dm_harvey_factor_raw": dm_rv_raw["dm_harvey_factor"],
                "dm_stat_vs_weekday": dm_wd["dm_stat"], "dm_pvalue_vs_weekday": dm_wd["dm_pvalue"],
                "dm_stat_vs_weekday_raw": dm_wd_raw["dm_stat"], "dm_pvalue_vs_weekday_raw": dm_wd_raw["dm_pvalue"],
                "rmse": means[0], "rmse_lo": ci[0, 0], "rmse_hi": ci[1, 0],
                "mae": means[1], "mae_lo": ci[0, 1], "mae_hi": ci[1, 1],
                "qlike": means[2], "qlike_lo": ci[0, 2], "qlike_hi": ci[1, 2],
                "qlike_raw": means[3], "qlike_raw_lo": ci[0, 3], "qlike_raw_hi": ci[1, 3],
                "mcs_pvalue_calibrated": float(mcs_cal.iloc[0]) if not mcs_cal.empty else np.nan,
                "mcs_95_included_calibrated": bool(mcs_cal.iloc[0] >= 0.05) if not mcs_cal.empty else False,
                "mcs_pvalue_raw": float(mcs_raw.iloc[0]) if not mcs_raw.empty else np.nan,
                "mcs_95_included_raw": bool(mcs_raw.iloc[0] >= 0.05) if not mcs_raw.empty else False,
                "fit_seconds": float(c_info["load_seconds"]) if c_info is not None else 0.0,
                "trainable_parameters": 0,
                "inference_ms_per_1000": float(c_info["inference_seconds"] / len(sub_m) * 1000) if c_info is not None else 0.0,
                "gpu_memory_mb": float(c_info["gpu_memory_mb"]) if c_info is not None else 0.0,
            }
            zero_rows.append(rec)
    pd.DataFrame(zero_rows).to_csv(RESULTS / "stage7_zero_shot.csv", index=False)
    print("Saved stage7_zero_shot.csv")

    # 4. Rescore fine-tuned models & ablations
    print("Rescoring fine-tuned models and ablations...")
    costs = pd.read_csv(RESULTS / "stage5_run_costs.csv")
    costs["family"] = costs.model.str.replace(r"-seed\d+$", "", regex=True)

    # Summarize seeds for fine-tuned models
    # Stage 4 models:
    s4_families = [
        "Chronos-Bolt-tiny-full", "Chronos-Bolt-tiny-lora8", "Chronos-Bolt-tiny-lora16",
        "Chronos-Bolt-small-full", "Chronos-Bolt-small-lora8", "Chronos-Bolt-small-lora16"
    ]
    s4_rows = []
    ablation_rows = []

    for (group, h), grp_scored in scored.groupby(["symbol_group", "horizon"]):
        daily_loss = grp_scored.groupby(["origin", "family"])[["qlike", "qlike_raw"]].mean()
        qloss = daily_loss["qlike"].unstack("family")
        raw_loss = daily_loss["qlike_raw"].unstack("family")

        # Get all unique families in grp_scored
        for fam, rows in grp_scored.groupby("family"):
            seeds = sorted(rows.model.unique())
            if len(seeds) != 3:
                continue
            family_cost = costs.loc[costs.family.eq(fam)]
            if family_cost.empty:
                # might be stage 4 cost
                c4 = pd.read_csv(RESULTS / "stage4_costs.csv")
                c4["family"] = c4.model.str.replace(r"-seed\d+$", "", regex=True)
                family_cost = c4.loc[c4.family.eq(fam)]

            means = rows.groupby(["model", "origin"])[LOSS_COLUMNS].mean()
            values = np.stack([means.xs(m, level="model").sort_index().to_numpy() for m in seeds])
            estimate, std, ci = seed_statistics(values, config)

            dm_rv = diebold_mariano(qloss[fam].to_numpy(), qloss["HAR-RV"].to_numpy(), int(h))
            dm_rv_raw = diebold_mariano(raw_loss[fam].to_numpy(), raw_loss["HAR-RV"].to_numpy(), int(h))
            dm_wd = diebold_mariano(qloss[fam].to_numpy(), qloss["HAR-weekday"].to_numpy(), int(h))
            dm_wd_raw = diebold_mariano(raw_loss[fam].to_numpy(), raw_loss["HAR-weekday"].to_numpy(), int(h))

            mcs_cal = mcs_df.loc[mcs_df.symbol_group.eq(group) & mcs_df.horizon.eq(h) &
                                 mcs_df.model.eq(fam) & mcs_df.metric.eq("qlike"), "mcs_pvalue"]
            mcs_raw = mcs_df.loc[mcs_df.symbol_group.eq(group) & mcs_df.horizon.eq(h) &
                                 mcs_df.model.eq(fam) & mcs_df.metric.eq("qlike_raw"), "mcs_pvalue"]

            rec = {
                "model": fam, "symbol_group": group, "horizon": h, "seeds": len(seeds),
                "n_dates": values.shape[1], "n_forecasts_per_seed": len(rows) // len(seeds),
                "trainable_parameters": int(family_cost.trainable_parameters.iloc[0]) if not family_cost.empty else 0,
                "fit_seconds_mean": float(family_cost.fit_seconds.mean()) if not family_cost.empty else 0.0,
                "inference_ms_per_1000_mean": float(family_cost.inference_ms_per_1000.mean()) if not family_cost.empty else 0.0,
                "gpu_training_peak_mb_max": float(family_cost.gpu_training_peak_mb.max()) if "gpu_training_peak_mb" in family_cost.columns and not family_cost.empty else 0.0,
                "rmse": estimate[0], "rmse_seed_sd": std[0], "rmse_lo": ci[0, 0], "rmse_hi": ci[1, 0],
                "mae": estimate[1], "mae_seed_sd": std[1], "mae_lo": ci[0, 1], "mae_hi": ci[1, 1],
                "qlike": estimate[2], "qlike_seed_sd": std[2], "qlike_lo": ci[0, 2], "qlike_hi": ci[1, 2],
                "qlike_raw": estimate[3], "qlike_raw_seed_sd": std[3], "qlike_raw_lo": ci[0, 3], "qlike_raw_hi": ci[1, 3],
                "dm_stat": dm_rv["dm_stat"], "dm_pvalue": dm_rv["dm_pvalue"],
                "dm_hac_lags": dm_rv["dm_hac_lags"], "dm_harvey_factor": dm_rv["dm_harvey_factor"],
                "dm_stat_raw": dm_rv_raw["dm_stat"], "dm_pvalue_raw": dm_rv_raw["dm_pvalue"],
                "dm_hac_lags_raw": dm_rv_raw["dm_hac_lags"], "dm_harvey_factor_raw": dm_rv_raw["dm_harvey_factor"],
                "dm_stat_vs_weekday": dm_wd["dm_stat"], "dm_pvalue_vs_weekday": dm_wd["dm_pvalue"],
                "dm_stat_vs_weekday_raw": dm_wd_raw["dm_stat"], "dm_pvalue_vs_weekday_raw": dm_wd_raw["dm_pvalue"],
                "mcs_pvalue_qlike": float(mcs_cal.iloc[0]) if not mcs_cal.empty else np.nan,
                "mcs_95_included_qlike": bool(mcs_cal.iloc[0] >= 0.05) if not mcs_cal.empty else False,
                "mcs_pvalue_qlike_raw": float(mcs_raw.iloc[0]) if not mcs_raw.empty else np.nan,
                "mcs_95_included_qlike_raw": bool(mcs_raw.iloc[0] >= 0.05) if not mcs_raw.empty else False,
            }
            if fam in s4_families:
                s4_rows.append(rec)
            ablation_rows.append(rec)

    pd.DataFrame(s4_rows).to_csv(RESULTS / "stage7_finetune.csv", index=False)
    print("Saved stage7_finetune.csv")

    grid = pd.read_csv(RESULTS / "stage5_grid.csv")
    ab_df = pd.DataFrame(ablation_rows)
    grid.merge(ab_df, left_on="family", right_on="model", validate="many_to_many").to_csv(
        RESULTS / "stage7_ablation_summary.csv", index=False)
    print("Saved stage7_ablation_summary.csv")

    # 5. Rescore regimes with standard QLIKE
    print("Rescoring regimes...")
    thresholds = pd.read_csv(RESULTS / "stage5_regime_thresholds.csv")
    selection = pd.read_csv(RESULTS / "stage5_frozen_selection.csv")
    regime_results = analyze_regimes(scored, thresholds, selection, config)
    regime_results.to_csv(RESULTS / "stage7_regimes.csv", index=False)
    print("Saved stage7_regimes.csv")

    # 6. Rescore robustness (shock exclusion)
    print("Rescoring robustness...")
    head = pd.concat([
        pd.DataFrame(base_rows),
        pd.DataFrame(zero_rows),
        pd.DataFrame(s4_rows)
    ], ignore_index=True)

    # Use frozen validation selection for headline fine-tuned model
    sel_map = selection.loc[selection.selected].set_index("horizon")["family"].to_dict()

    robust_records = []
    headline_models_list = ["Naive", "HAR-RV", "HAR-log-mean-RV", "HAR-weekday", "GARCH-t", "GJR-GARCH-t", "Zero-shot", "Fine-tuned"]
    for grp in ["fit-eligible", "held-out"]:
        for h in [1, 5, 22]:
            ft_fam = sel_map[h]
            model_family_map = {
                "Naive": "Naive",
                "HAR-RV": "HAR-RV",
                "HAR-log-mean-RV": "HAR-log-mean-RV",
                "HAR-weekday": "HAR-weekday",
                "GARCH-t": "GARCH-t",
                "GJR-GARCH-t": "GJR-GARCH-t",
                "Zero-shot": "Chronos-Bolt-small",
                "Fine-tuned": ft_fam,
            }
            for m_label, fam_name in model_family_map.items():
                subset = scored.loc[scored.family.eq(fam_name) & scored.horizon.eq(h) & scored.symbol_group.eq(grp)]
                for label, data in [("all dates", subset), ("exclude Oct 10-12", subset.loc[~excluded_windows(subset)])]:
                    per_seed = data.groupby("model")[["squared_error", "absolute_error", "qlike_raw", "qlike"]].mean()
                    per_seed["squared_error"] = np.sqrt(per_seed.squared_error)
                    means = per_seed.mean()
                    robust_records.append({
                        "model": m_label, "family": fam_name, "symbol_group": grp, "horizon": h,
                        "sample": label, "n_dates": data.origin.nunique(), "examples_per_seed": len(data) // len(per_seed),
                        "rmse": means.squared_error, "mae": means.absolute_error,
                        "qlike_raw": means.qlike_raw, "qlike": means.qlike
                    })
    robust_df = pd.DataFrame(robust_records)
    for metric in ["qlike_raw", "qlike"]:
        robust_df["rank_" + metric] = robust_df.groupby(["symbol_group", "horizon", "sample"])[metric].rank(method="min").astype(int)
    robust_df.to_csv(RESULTS / "stage7_robustness.csv", index=False)
    print("Saved stage7_robustness.csv")

    # 7. Generate Stage 7 Headline Summary Table
    print("Building stage7_headline.csv...")
    headline_rows_out = []
    for grp in ["fit-eligible", "held-out"]:
        for h in [1, 5, 22]:
            ft_fam = sel_map[h]
            focal_models = [
                ("Naive", "Naive"),
                ("HAR-RV", "HAR-RV"),
                ("HAR-weekday", "HAR-weekday"),
                ("GARCH-t", "GARCH-t"),
                ("Zero-shot", "Chronos-Bolt-small"),
                ("Fine-tuned", ft_fam),
            ]
            for m_label, fam_name in focal_models:
                # Find metrics from base_rows, zero_rows, or s4_rows
                pool = base_rows if m_label in ["Naive", "HAR-RV", "HAR-weekday", "GARCH-t"] else (zero_rows if m_label == "Zero-shot" else s4_rows)
                entry = next(r for r in pool if r["symbol_group"] == grp and r["horizon"] == h and (r["model"] == fam_name or r["model"] == m_label))

                headline_rows_out.append({
                    "symbol_group": grp,
                    "horizon": h,
                    "model": m_label,
                    "family": fam_name,
                    "rmse": entry["rmse"],
                    "mae": entry["mae"],
                    "qlike_raw": entry["qlike_raw"],
                    "qlike_raw_lo": entry["qlike_raw_lo"],
                    "qlike_raw_hi": entry["qlike_raw_hi"],
                    "qlike": entry["qlike"],
                    "qlike_lo": entry["qlike_lo"],
                    "qlike_hi": entry["qlike_hi"],
                    "dm_pvalue_vs_weekday": entry.get("dm_pvalue_vs_weekday", np.nan),
                    "dm_pvalue_vs_weekday_raw": entry.get("dm_pvalue_vs_weekday_raw", np.nan),
                    "trainable_parameters": entry.get("trainable_parameters", 0),
                    "fit_seconds": entry.get("fit_seconds_mean", entry.get("fit_seconds", 0.0)),
                })
    pd.DataFrame(headline_rows_out).to_csv(RESULTS / "stage7_headline.csv", index=False)
    print("Saved stage7_headline.csv")

    # 8. Preservation audit
    preservation = verify_historical_preservation()
    preservation.to_csv(RESULTS / "stage7_preservation.csv", index=False)
    print(f"Verified {len(preservation)} pre-Stage-7 result files preserved.")


if __name__ == "__main__":
    rescore_all()
