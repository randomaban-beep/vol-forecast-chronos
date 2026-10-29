"""Compute complementary volatility forecasting metrics on the frozen test panel.

Evaluates Mean Directional Accuracy (MDA), Median Absolute Percentage Error (MdAPE),
Forecast Bias, Mincer-Zarnowitz R2, and Quantile Check Loss (Pinball 0.90) from cached forecasts.
Never retrains models or modifies historical results.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.report import headline_rows, RESULTS
from src.evaluate import (mean_directional_accuracy, mdape, forecast_bias,
                          mincer_zarnowitz, pinball_loss)


def compute_extended_metrics(scored: pd.DataFrame, head: pd.DataFrame) -> pd.DataFrame:
    """Compute complementary metrics across all models, horizons, and symbol groups."""
    records = []
    for row in head.itertuples():
        sub = scored.loc[scored.family.eq(row.family) &
                         scored.horizon.eq(row.horizon) &
                         scored.symbol_group.eq(row.symbol_group)]
        if sub.empty:
            continue
        seeds = sub.model.unique()
        seed_records = []
        for s in seeds:
            seed_data = sub.loc[sub.model.eq(s)]
            act_y = seed_data.target.to_numpy()
            pred_y = seed_data.prediction.to_numpy()
            act_change = seed_data.target.to_numpy() - seed_data.daily.to_numpy()
            pred_change = seed_data.prediction.to_numpy() - seed_data.daily.to_numpy()
            act_var = np.exp(act_y)
            pred_var = np.exp(pred_y)

            mda = mean_directional_accuracy(act_change, pred_change) * 100.0
            med_ape = mdape(act_var, pred_var) * 100.0
            bias = forecast_bias(act_y, pred_y)
            mz = mincer_zarnowitz(act_y, pred_y)
            pinball = pinball_loss(act_y, pred_y, quantile=0.90)

            seed_records.append({
                "mda_pct": mda,
                "mdape_pct": med_ape,
                "bias": bias,
                "r2_direct_pct": mz["r2_direct"] * 100.0,
                "r2_mz_pct": mz["r2_mz"] * 100.0,
                "pinball_90": pinball,
            })

        records.append({
            "symbol_group": row.symbol_group,
            "horizon": row.horizon,
            "model": row.model,
            "family": row.family,
            "mda_pct": float(np.mean([r["mda_pct"] for r in seed_records])),
            "mdape_pct": float(np.mean([r["mdape_pct"] for r in seed_records])),
            "bias": float(np.mean([r["bias"] for r in seed_records])),
            "r2_direct_pct": float(np.mean([r["r2_direct_pct"] for r in seed_records])),
            "r2_mz_pct": float(np.mean([r["r2_mz_pct"] for r in seed_records])),
            "pinball_90": float(np.mean([r["pinball_90"] for r in seed_records])),
        })

    return pd.DataFrame(records)


def main() -> None:
    source = Path("data/stage5/scored.parquet")
    if not source.exists():
        raise SystemExit("data/stage5/scored.parquet required to compute extended metrics.")

    scored = pd.read_parquet(source)
    scored["family"] = scored.model.str.replace(r"-seed\d+$", "", regex=True)
    head = headline_rows()

    df = compute_extended_metrics(scored, head)
    out_path = RESULTS / "stage6_extended_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"Exported extended metrics to {out_path}")


if __name__ == "__main__":
    main()
