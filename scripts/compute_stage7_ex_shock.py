"""Compute ex-shock headline metrics, paired loss differences, and shock loss shares.

Preserves all historical result files. Saves only new files with stage7_ prefix.
"""

from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import qlike_patton, diebold_mariano, block_bootstrap_ci
from src.finetune_summary import family_mcs
from scripts.package_results import excluded_windows

RESULTS = Path("results")
DATA = Path("data")


def main() -> None:
    config = yaml.safe_load(Path("configs/stage-5-ablations.yaml").read_text())
    source = DATA / "stage5" / "scored.parquet"
    scored = pd.read_parquet(source)

    act_v = np.exp(scored.target.to_numpy())
    pred_v_raw = np.exp(scored.prediction.to_numpy())
    pred_v_cal = scored.forecast_variance.to_numpy()

    scored["qlike_raw"] = qlike_patton(act_v, pred_v_raw)
    scored["qlike"] = qlike_patton(act_v, pred_v_cal)
    scored["squared_error"] = (scored.prediction - scored.target) ** 2
    scored["absolute_error"] = (scored.prediction - scored.target).abs()
    scored["is_shock"] = excluded_windows(scored)
    scored["ex_shock"] = ~scored["is_shock"]

    # 1. Headline models mapping
    focal_specs = [
        ("Naive", "Naive"),
        ("HAR-RV", "HAR-RV"),
        ("HAR-weekday", "HAR-weekday"),
        ("GARCH-t", "GARCH-t"),
        ("Zero-shot", "Chronos-Bolt-small"),
        ("Fine-tuned", None),  # dynamically set per horizon
    ]

    # Pre-aggregate date-level losses for all models
    # Seed models need seed averaging
    scored["family"] = scored.model.str.replace(r"-seed\d+$", "", regex=True)

    # 1. Share of total loss contributed by shock origins
    print("Computing shock loss shares...")
    share_rows = []
    for (grp, h), df_gh in scored.groupby(["symbol_group", "horizon"]):
        ft_family = "Chronos-Bolt-small-full" if h in [1, 5] else "Chronos-Bolt-small-lora16"
        for label, fam in focal_specs:
            target_fam = ft_family if label == "Fine-tuned" else fam
            sub = df_gh.loc[df_gh.family.eq(target_fam)]
            if sub.empty:
                continue
            # Date-level loss averaged across symbols and seeds
            daily = sub.groupby(["origin", "is_shock"])[["qlike", "qlike_raw", "squared_error", "absolute_error"]].mean().reset_index()
            total_qlike = daily["qlike"].sum()
            shock_qlike = daily.loc[daily.is_shock, "qlike"].sum()
            share_qlike = (shock_qlike / total_qlike) * 100.0 if total_qlike > 0 else 0.0

            total_qlike_raw = daily["qlike_raw"].sum()
            shock_qlike_raw = daily.loc[daily.is_shock, "qlike_raw"].sum()
            share_qlike_raw = (shock_qlike_raw / total_qlike_raw) * 100.0 if total_qlike_raw > 0 else 0.0

            n_total_dates = daily.origin.nunique()
            n_shock_dates = daily.loc[daily.is_shock, "origin"].nunique()

            share_rows.append({
                "symbol_group": grp,
                "horizon": h,
                "model": label,
                "family": target_fam,
                "n_total_dates": n_total_dates,
                "n_shock_dates": n_shock_dates,
                "shock_date_pct": (n_shock_dates / n_total_dates) * 100.0,
                "total_loss_qlike": total_qlike,
                "shock_loss_qlike": shock_qlike,
                "shock_share_qlike_pct": share_qlike,
                "shock_share_qlike_raw_pct": share_qlike_raw,
            })
    loss_share_df = pd.DataFrame(share_rows)
    loss_share_df.to_csv(RESULTS / "stage7_shock_loss_share.csv", index=False)
    print("Saved stage7_shock_loss_share.csv")

    # 2. Ex-shock MCS (across all models)
    print("Computing Ex-Shock MCS...")
    ex_mcs_records = []
    for (grp, h), df_gh in scored.loc[scored.ex_shock].groupby(["symbol_group", "horizon"]):
        daily_cand = df_gh.groupby(["origin", "family"])["qlike"].mean().unstack("family")
        pvals = family_mcs(daily_cand, config)
        for fam, pval in pvals.items():
            ex_mcs_records.append({
                "symbol_group": grp, "horizon": h, "family": fam,
                "mcs_pvalue_ex": pval, "mcs_95_included_ex": pval >= 0.05
            })
    ex_mcs_df = pd.DataFrame(ex_mcs_records)

    # 3. Ex-shock headline table
    print("Building Ex-Shock Headline metrics...")
    ex_headline = []
    for (grp, h), df_gh in scored.loc[scored.ex_shock].groupby(["symbol_group", "horizon"]):
        ft_family = "Chronos-Bolt-small-full" if h in [1, 5] else "Chronos-Bolt-small-lora16"
        daily_wd = df_gh.loc[df_gh.family.eq("HAR-weekday")].groupby("origin")["qlike"].mean()

        for label, fam in focal_specs:
            target_fam = ft_family if label == "Fine-tuned" else fam
            sub = df_gh.loc[df_gh.family.eq(target_fam)]
            if sub.empty:
                continue
            daily_m = sub.groupby("origin")[["squared_error", "absolute_error", "qlike_raw", "qlike"]].mean()
            means = daily_m.mean()
            rmse = np.sqrt(means["squared_error"])
            mae = means["absolute_error"]
            qlike_raw = means["qlike_raw"]
            qlike = means["qlike"]

            # DM vs HAR-weekday on ex-shock calibrated qlike
            common = daily_m.index.intersection(daily_wd.index)
            if label == "HAR-weekday":
                dm_p = np.nan
            else:
                dm_res = diebold_mariano(daily_m.loc[common, "qlike"].to_numpy(),
                                         daily_wd.loc[common].to_numpy(), int(h))
                dm_p = dm_res["dm_pvalue"]

            # MCS inclusion
            sub_mcs = ex_mcs_df.loc[ex_mcs_df.symbol_group.eq(grp) &
                                    ex_mcs_df.horizon.eq(h) &
                                    ex_mcs_df.family.eq(target_fam)]
            mcs_inc = sub_mcs["mcs_95_included_ex"].iloc[0] if not sub_mcs.empty else False

            ex_headline.append({
                "symbol_group": grp,
                "horizon": h,
                "model": label,
                "family": target_fam,
                "rmse_ex": rmse,
                "mae_ex": mae,
                "qlike_raw_ex": qlike_raw,
                "qlike_ex": qlike,
                "dm_pvalue_vs_weekday_ex": dm_p,
                "mcs_95_included_ex": mcs_inc,
            })
    ex_headline_df = pd.DataFrame(ex_headline)
    ex_headline_df.to_csv(RESULTS / "stage7_ex_shock.csv", index=False)
    print("Saved stage7_ex_shock.csv")

    # 4. PAIRED loss difference (Fine-tuned minus HAR-weekday) with block-bootstrap 95% CI
    print("Computing Paired Loss Difference (Fine-tuned minus HAR-weekday)...")
    paired_rows = []
    rng = np.random.default_rng(config["seed"])
    reps = config["bootstrap_reps"]
    block = config["block_length"]

    for (grp, h), df_gh in scored.groupby(["symbol_group", "horizon"]):
        ft_family = "Chronos-Bolt-small-full" if h in [1, 5] else "Chronos-Bolt-small-lora16"
        for sample_label, is_ex in [("full-sample", False), ("ex-shock", True)]:
            data = df_gh.loc[df_gh.ex_shock] if is_ex else df_gh
            daily_ft = data.loc[data.family.eq(ft_family)].groupby("origin")["qlike"].mean()
            daily_wd = data.loc[data.family.eq("HAR-weekday")].groupby("origin")["qlike"].mean()
            common = daily_ft.index.intersection(daily_wd.index)
            diffs = daily_ft.loc[common].to_numpy() - daily_wd.loc[common].to_numpy()
            n_dates = len(diffs)

            mean_diff = float(np.mean(diffs))

            # Block bootstrap CI for paired difference
            starts = rng.integers(n_dates, size=(reps, int(np.ceil(n_dates / block))))
            indices = ((starts[..., None] + np.arange(block)) % n_dates).reshape(reps, -1)[:, :n_dates]
            boot_means = np.mean(diffs[indices], axis=1)
            ci_lo, ci_hi = np.quantile(boot_means, [0.025, 0.975])

            # DM test p-value
            dm = diebold_mariano(daily_ft.loc[common].to_numpy(), daily_wd.loc[common].to_numpy(), int(h))

            paired_rows.append({
                "symbol_group": grp,
                "horizon": h,
                "sample": sample_label,
                "n_dates": n_dates,
                "mean_diff": mean_diff,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "dm_stat": dm["dm_stat"],
                "dm_pvalue": dm["dm_pvalue"],
            })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(RESULTS / "stage7_paired_diff.csv", index=False)
    print("Saved stage7_paired_diff.csv")


if __name__ == "__main__":
    main()
