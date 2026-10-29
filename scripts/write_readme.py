"""Build the final plain-spoken README with tables read from committed CSVs."""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.report import (headline_table, legacy_headline_table, markdown,
                        RESULTS, GROUPS)


def robustness_table() -> str:
    """Show before/after winners under standard Patton QLIKE without tuning any configuration."""
    data = pd.read_csv(RESULTS / 'stage7_robustness.csv')
    rows = []
    for group in GROUPS:
        for h in sorted(data.horizon.unique()):
            panel = data.loc[data.symbol_group.eq(group) & data.horizon.eq(h)]
            row = {'Group': group, 'h': h}
            for metric, label in [('qlike_raw', 'Raw Patton'), ('qlike', 'Calibrated Patton')]:
                for sample, suffix in [('all dates', 'before'), ('exclude Oct 10-12', 'after')]:
                    rows2 = panel.loc[panel['sample'].eq(sample)]
                    row[f'{label} {suffix}'] = '/'.join(rows2.loc[rows2[metric].eq(rows2[metric].min()), 'model'])
            row['Origins removed'] = int(panel.loc[panel['sample'].eq('all dates'), 'n_dates'].iloc[0] - panel.loc[panel['sample'].eq('exclude Oct 10-12'), 'n_dates'].iloc[0])
            rows.append(row)
    frame = pd.DataFrame(rows)
    return markdown(frame, list(frame.columns))


def extended_metrics_tables() -> str:
    """Format complementary evaluation metrics across models and horizons."""
    data = pd.read_csv(RESULTS / 'stage6_extended_metrics.csv')
    # Naive is excluded from directional accuracy because direction from origin is undefined (delta = 0)
    mda_models = ['HAR-RV', 'HAR-weekday', 'GARCH-t', 'Zero-shot', 'Fine-tuned']
    mdape_models = ['Naive', 'HAR-RV', 'HAR-weekday', 'GARCH-t', 'Zero-shot', 'Fine-tuned']
    mda_lines = ['| Group | h | ' + ' | '.join(mda_models) + ' |', '| --- | ---: | ' + ' | '.join(['---:'] * len(mda_models)) + ' |']
    mdape_lines = ['| Group | h | ' + ' | '.join(mdape_models) + ' |', '| --- | ---: | ' + ' | '.join(['---:'] * len(mdape_models)) + ' |']
    for group in GROUPS:
        for h in [1, 5, 22]:
            sub = data.loc[data.symbol_group.eq(group) & data.horizon.eq(h)].set_index('model')
            mda_lines.append(f'| {group} | {h} | ' + ' | '.join(f"{sub.loc[m, 'mda_pct']:.2f}%" for m in mda_models) + ' |')
            mdape_lines.append(f'| {group} | {h} | ' + ' | '.join(f"{sub.loc[m, 'mdape_pct']:.2f}%" for m in mdape_models) + ' |')
    return '**Directional Accuracy (MDA, higher is better; Naive excluded as direction is undefined):**\n\n' + '\n'.join(mda_lines) + '\n\n**Median Absolute Percentage Error on Variance (MdAPE, lower is better):**\n\n' + '\n'.join(mdape_lines)


def main() -> None:
    """Render headline table, event values, robustness rescore and runtime measurements from source CSVs."""
    worst = pd.read_csv(RESULTS / 'stage5_worst_forecasts.csv')
    import numpy as np
    worst['Realized RV'] = np.exp(worst.target)
    worst['Forecast RV'] = np.exp(worst.prediction)
    worst['Date'] = pd.to_datetime(worst.target_end).dt.strftime('%Y-%m-%d')
    worst = worst.rename(columns={'symbol': 'Symbol'})
    runtime = pd.read_csv(RESULTS / 'stage6_runtime.csv')
    runtime['Measured minutes'] = runtime.seconds.map(lambda x: 'not recorded' if pd.isna(x) else f'{x/60:.3f}')

    text = '''# Chronos, meet the calendar

Fine-tuning a time-series foundation model to forecast crypto volatility. The cheap baseline gets a vote.

![Python version](docs/img/badge_python.svg) ![Passing tests](docs/img/badge_tests.svg) ![MIT license](docs/img/badge_license.svg) ![GPU used](docs/img/badge_gpu.svg)

## Overview: What is this project?

Predicting cryptocurrency volatility—the intensity of price swings—is critical for risk management, exchange margin requirements, and derivative pricing. For decades, quantitative finance has relied on classical econometric models like **HAR-RV** (Heterogeneous Autoregressive model of Realized Volatility) and **GARCH**.

This project investigates a central question in modern quantitative machine learning:
> **Can pretrained Time-Series Foundation Models (Amazon Chronos-Bolt) beat classical financial econometrics when forecasting cryptocurrency realized volatility?**

### What was built
- **End-to-End Pipeline**: Processes 6+ years (2020–2026) of verified 1-minute Binance spot data across 8 major crypto assets (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, LTC) to construct high-frequency 5-minute realized variance (RV).
- **Zero Lookahead Leakage**: Strict expanding walk-forward splits, calendar boundary purging, and automated perturbation unit tests.
- **Foundation Models vs. Econometrics**: Evaluates zero-shot Chronos-Bolt, full fine-tuning, and Parameter-Efficient LoRA (ranks 4–32) against HAR-RV, calendar-aware HAR-weekday, and Student-t GARCH/GJR-GARCH.
- **Consumer Hardware Budget**: Engineered to run entirely on a single laptop GPU (NVIDIA RTX 3050 6 GB, 24 GB RAM) within a strict 2-hour budget.
- **Reproducible & Audited**: One-command rebuild (`python scripts/reproduce.py`), cryptographic preservation hashes, and open-source MIT release.

### High-level takeaways (Patton 2011 standard evaluation)
1. **The Calendar Baseline (HAR-weekday) Wins Overall**: Under standard Patton (2011) QLIKE and log RMSE, HAR-weekday is the top-performing model across all horizons (h=1, 5, 22). A simple 10-parameter linear regression with weekday terms beats or matches the 48M-parameter foundation model at a fraction of the compute cost.
2. **Calibrated Chronos Competes at Multi-Day Horizons**: When smearing calibration is applied to align variance scales with the conditional arithmetic mean, fine-tuned Chronos-Bolt achieves competitive or slightly lower point estimates at h=5 (0.3697 vs 0.3799) and h=22 (0.3205 vs 0.3491); however, Diebold-Mariano tests against HAR-weekday show p-values of 0.31–0.65 (fail to reject the null of equal predictive accuracy).
3. **Directional Accuracy (MDA)**: Where Chronos clearly leads is in predicting the *direction* of volatility changes (anticipating whether volatility will expand or contract with 70–75% accuracy vs 64–71% for standard HAR-RV).
4. **Typical Days (MdAPE)**: On median percentage error (which is robust to single-day extreme shocks), Chronos yields lower typical errors (31–36% MdAPE vs 37–46% for HAR).

---

## The 30-second story

At h=1, HAR-weekday wins on raw Patton QLIKE (0.4383 vs 0.5059 for fine-tuned) and log RMSE (0.5920 vs 0.6087).
On calibrated QLIKE, smearing aligns Chronos with the arithmetic mean, putting fine-tuned Chronos (0.3802) roughly tied with HAR-weekday (0.3758; DM p=0.6471).
At multi-day horizons (h=5 and h=22), calibrated Chronos achieves slightly lower point estimates (0.3205 vs 0.3491 at h=22), but differences are not statistically distinguishable from HAR-weekday.
This is a forecasting-accuracy study, not a trading strategy.

![Raw QLIKE for six headline models across three horizons; held-out symbols in the lighter lower panel; bars show 95 percent confidence intervals](docs/img/headline.png)

**Headline Results Table: Patton (2011) QLIKE, Error Metrics, Uncertainty, and Costs.**
Lower QLIKE and RMSE are better. Intervals use 30-day circular blocks; fine-tuned intervals also resample seeds. Zero-shot is Chronos-Bolt-small. Fine-tuned model is validation-selected (full for h=1/5, LoRA-16 for h=22).

<!-- headline-table-start -->
'''+headline_table()+'''
<!-- headline-table-end -->

Sources: [Stage 7 headline CSV](results/stage7_headline.csv), [baselines](results/stage7_baselines.csv),
[zero-shot](results/stage7_zero_shot.csv), [fine-tuning](results/stage7_finetune.csv).

## Scoreboard

![Raw and calibrated QLIKE grids; green borders mark row minima and stars mark membership in the 95 percent model confidence set](docs/img/scoreboard.png)

The grid includes the alternate HAR feature definition and asymmetric GARCH so their wins and losses remain visible.
Green borders mark exact point-estimate winners, including exact ties; displayed rounding is not a tie test.
Stars use the [Stage 7 joint MCS](results/stage7_mcs.csv), whose universe also includes the ablations and other original families—not just the displayed columns.
MCS membership means the procedure did not eliminate that candidate; it does not prove equivalence.

## The day everything broke

The three worst h=1 errors all land on **2025-10-10**. They are ranked by mean seed squared log error;
the forecast RV below is the exponential of the mean log forecast, shown for illustration rather than scored as an ensemble.

'''+markdown(worst, ['Date', 'Symbol', 'Realized RV', 'Forecast RV'])+'''

![ADAUSDT on October 10: realized log RV jumps far above the three-seed forecast range](results/stage5_worst_1.png)
![DOGEUSDT on October 10: a large realized-volatility shock missed by the next-day forecast](results/stage5_worst_2.png)
![XRPUSDT on October 10: the forecast remains near prior volatility while realized RV spikes](results/stage5_worst_3.png)

The models saw daily log RV through the previous UTC day—not tomorrow's prices, order books, or news.
Binance's [retrospective statement](https://www.binance.com/en/support/announcement/detail/d9cb0d52d7c142a5be4f49732bd8760c)
describes a broad sell-off that day. That is context, not a proven causal explanation for each coin's error.
Source numbers: [worst-forecast CSV](results/stage5_worst_forecasts.csv).

### Take the shock out

Reporting-only rescore: exclude October 10 **and October 11–12**, removing every forecast whose target window intersects those dates.
For h=5/22 this removes overlapping averages too, rather than leaving pieces of the shock inside the target.
Models, validation choices and calibration factors stay frozen. No refitting, recalibration or new winner selection for the headline.
The rescore includes the scoreboard's eight columns. Later forecasts still use observed history;
this is a loss rescore, not a counterfactual market in which the shock never happened.

'''+robustness_table()+'''

Under standard Patton QLIKE, removing the flash crash dates cements HAR-weekday as the dominant performer across all horizons and groups.
All model scores, ranks and retained counts: [robustness CSV](results/stage7_robustness.csv). This sensitivity check reports point estimates, not fresh significance claims.

### Beyond loss averages: directional calls and typical error

Standard QLIKE and RMSE averages are heavily weighted by tail events. Complementary scale-free metrics from [extended metrics CSV](results/stage6_extended_metrics.csv) show where Chronos excels:

'''+extended_metrics_tables()+'''

- **Directional hit rate:** Chronos models achieve the highest directional accuracy across both fit-eligible and held-out assets (reaching 70.42% at h=1 and up to 75.68% at h=22), consistently outperforming standard HAR-RV (64.45% and 71.77%).
- **Typical-day error:** On Median Absolute Percentage Error (MdAPE), Chronos-Bolt consistently beats HAR-RV and HAR-weekday across all horizons (33.39% vs 35.01% at h=1; 31.57% vs 36.70% at h=5; 36.65% vs 46.29% at h=22), demonstrating substantially lower percentage error when outlier shock days are mitigated.

## Errata: Loss Function Orientation (Patton 2011 vs Legacy)

In Stages 2–6, the evaluation suite implemented QLIKE as:
$$\\mathcal{L}_{\\text{legacy}}(V, \\hat{V}) = \\frac{\\hat{V}}{V} - \\log\\left(\\frac{\\hat{V}}{V}\\right) - 1 \\quad (\\text{forecast over actual})$$

### The Orientation Problem and Mathematical Consequence
In the time-series econometrics literature (Patton 2011, *Journal of Econometrics*), the standard robust QLIKE loss function is defined with actual over forecast:
$$\\mathcal{L}_{\\text{patton}}(V, \\hat{V}) = \\frac{V}{\\hat{V}} - \\log\\left(\\frac{V}{\\hat{V}}\\right) - 1 \\quad (\\text{actual over forecast})$$

The distinction is mathematically fundamental:
1. **Consistency for the Conditional Arithmetic Mean**:
   Taking the first-order condition of expected loss with respect to $\\hat{V}$:
   $$\\frac{\\partial \\mathcal{L}_{\\text{patton}}}{\\partial \\hat{V}} = -\\frac{V}{\\hat{V}^2} + \\frac{1}{\\hat{V}} = \\frac{\\hat{V} - V}{\\hat{V}^2} \\implies \\mathbb{E}\\left[\\frac{\\partial \\mathcal{L}_{\\text{patton}}}{\\partial \\hat{V}}\\right] = 0 \\iff \\hat{V} = \\mathbb{E}[V]$$
   Thus, standard Patton QLIKE is strictly consistent for the **arithmetic conditional mean**.
2. **Why Legacy QLIKE Favored Negative Bias**:
   Taking the first-order condition for the legacy formula:
   $$\\frac{\\partial \\mathcal{L}_{\\text{legacy}}}{\\partial \\hat{V}} = \\frac{1}{V} - \\frac{1}{\\hat{V}} \\implies \\mathbb{E}\\left[\\frac{\\partial \\mathcal{L}_{\\text{legacy}}}{\\partial \\hat{V}}\\right] = 0 \\iff \\hat{V} = \\frac{1}{\\mathbb{E}[1/V]}$$
   The legacy loss function is minimized at the **conditional harmonic mean** of realized variance.
   Because the harmonic mean is strictly less than or equal to the arithmetic mean ($\\text{HM} \\le \\text{AM}$), the legacy formula heavily penalized forecasts higher than actual and systematically rewarded models that under-forecasted volatility (negative bias).
3. **What Changed Under Patton (2011) QLIKE**:
   - Because raw Chronos-Bolt forecasts had a slight negative bias (bias $\\approx -0.09$), legacy QLIKE artificially boosted raw Chronos-Bolt below HAR-RV and HAR-weekday.
   - When smearing calibration was applied (multiplying by $\\mathbb{E}[\\exp(e)] > 1$ to align with the arithmetic mean), legacy QLIKE penalized that upward scale adjustment.
   - Under standard Patton QLIKE, smearing calibration now functions exactly as econometrics theory predicts: it scales forecasts to the arithmetic mean, cutting loss from ~0.50 down to ~0.38.
   - Under standard Patton QLIKE, **HAR-weekday is the clear winner on raw loss and log RMSE across all horizons**. On calibrated loss, fine-tuned Chronos-Bolt achieves slightly lower point estimates at $h=5$ and $h=22$ (e.g. 0.3205 vs 0.3491 at $h=22$), but Diebold-Mariano tests vs HAR-weekday confirm the difference is not statistically significant ($p \\approx 0.31\\text{--}0.65$).

### Appendix: Legacy Orientation QLIKE Numbers (Stages 2–6)
For historical auditability and full reproducibility of earlier milestones, the legacy orientation QLIKE values ($\\hat{V}/V - \\log(\\hat{V}/V) - 1$) are preserved below:

<!-- legacy-headline-start -->
'''+legacy_headline_table()+'''
<!-- legacy-headline-end -->

Sources: [Stage 2 baselines](results/stage2_baselines.csv), [Stage 3 zero-shot](results/stage3_zero_shot.csv),
[Stage 4 fine-tuning](results/stage4_finetune.csv), [Stage 5 weekday](results/stage5_weekday.csv).

## Is it worth the compute?

![Training cost versus next-day raw QLIKE; trainable neural parameters on a logarithmic axis and bubble area proportional to mean training time](docs/img/compute.png)

The chart shows the original full/LoRA families at h=1 on fit-eligible symbols. Bubble area is mean training seconds per seed.
HAR reference lines sit outside the neural-parameter axis: **zero neural parameters does not mean zero fitted coefficients**—HAR-RV fits four and HAR-weekday ten per horizon.
Zero-shot also has zero trainable parameters and is therefore not placed at a fictitious positive value on a log axis.
Parameter-efficient adaptation saves trainable weights; it does not guarantee a better forecast.
[Costs and uncertainty](results/stage4_finetune.csv).

![BTCUSDT rolling next-day forecasts during a calm week and the October shock week; shaded band is the range across three seeds](docs/img/forecast_weeks.png)

These are rolling h=1 forecasts, not a seven-day forecast issued at the week's start.
The calm panel is the lowest-mean-RV complete Monday–Sunday BTC week in test; the shock panel is the week containing October 10.
Both are retrospective illustrations, never selection inputs. The fine-tuned line is the median across seed-specific median forecasts;
the shaded seed range is **not** a predictive interval. Variance forecasts are raw, without smearing.
[Figure inputs](results/stage6_forecast_weeks.csv).

## What actually mattered

![Next-day raw QLIKE versus context length for full and LoRA fine-tuning](docs/img/ablation_context.png)

Short context hurt: 32 days loses to 256 at h=1 in both modes and both symbol groups.

![Next-day raw QLIKE versus the fraction of earliest training windows](docs/img/ablation_fraction.png)

More history helped next-day raw QLIKE overall, but calibrated and longer-horizon scores were not uniformly monotone.

![Next-day raw QLIKE versus LoRA rank with bootstrap uncertainty bands](docs/img/ablation_rank.png)

Rank had no clear accuracy effect in this sweep; buying more adapter parameters did not buy a reliable win.

Small panels show fit-eligible h=1 and 95% block/seed intervals. Full contexts/fractions/ranks, all horizons/groups:
[ablation results](results/stage5_ablation_summary.csv). Context experiments keep target-window membership fixed;
fractions take the earliest windows by origin date, with deterministic symbol tie breaks.
These test curves are reporting only. Ten new small-model runs selected the three-epoch cap, so the sweep does not establish convergence.

## Things that didn't work

- **Long-horizon dominance.** At h=22, calibrated QLIKE favors HAR variants over the frozen fine-tuned model in both groups.
- **More rank, more accuracy.** Ranks 4–32 show little raw-QLIKE separation; this is a null result at this budget, not proof ranks never matter.
- **Calibration as a free upgrade.** Validation smearing hurts HAR's test QLIKE. It is frozen and reported beside raw scores, not switched off after seeing test.
- **Extra epochs.** The separately approved tiny LoRA16 extension selected epochs 1/3/2, never beyond the old cap; it did not unlock a better checkpoint.
- **Tiny context.** The 32-day context damages next-day performance.

The [extended experiment](results/stage5_extended_tiny_summary.csv) remains separate. Shared-epoch validation losses show small numerical drift
with identical cached tensor values but different memory layout; [audit](results/stage5_extended_reproducibility.csv).
Neither this experiment nor any ablation replaces a Stage 4 number.

## How we tried not to cheat

```mermaid
flowchart LR
  A[Verified Binance 1-minute spot klines] --> B[Strict daily quality filters]
  B --> C[5-minute returns to daily RV]
  C --> D[Calendar splits and purged target windows]
  D --> E[Train 2020–2023: six symbols]
  D --> F[Validation 2024: six symbols]
  E --> G[Fit weights or expanding baselines]
  F --> H[Freeze checkpoints, family choices and smearing]
  G --> H
  H --> I[Test 2025–Sep 2026: same common rows]
  D --> J[SOL and DOGE held out from fitting, calibration and selection]
  J --> I
  K[Leakage tests: perturb future values and held-out targets] -. verify .-> G
  K -. verify .-> H
```

- Targets crossing split boundaries are purged. Horizons mean **log of arithmetic mean daily RV**, not an average of logs.
- Daily RV sums squared five-minute log returns from verified one-minute UTC klines; incomplete or bad days are dropped, never filled.
- Fit-eligible symbols are BTC, ETH, BNB, XRP, ADA and LTC; the held-out pair is SOL and DOGE (all USDT spot pairs).
- SOL/DOGE never enter model fitting, checkpoint/family selection or smearing calibration. Their past series may be forecast inputs.
- Fine-tuned weights freeze before test. Econometric models expand their six-symbol fit and refit every 30 days, using only then-observed targets.
- Each model/horizon's smearing factor is `mean(exp(target - prediction))` on six-symbol 2024 validation residuals, then frozen. Raw uses factor one.
- Median daily log-RV paths become horizon forecasts via `log(mean(exp(path[:h])))`; contexts end at the origin, preserving missing days as NaNs/masks.
- Future-perturbation and held-out-perturbation tests protect contexts and fitting. Missing forecasts cannot silently shrink the common panel.
- [Preservation hashes](results/stage7_preservation.csv) verify every pre-Stage-7 result file byte-identical. [Stage 5 audit](results/stage5_panel_audit.csv) checks common rows and exact reuse.
- Regime thresholds use training data only. [Regime results](results/stage7_regimes.csv) retain calendar gaps in resampling and report sparse-sample counts.

## FAQ for skeptics

**Why not just use HAR?** You might. HAR-weekday is a strong cheap benchmark; fine-tuning's clearest advantage is against the original HAR-RV, not every HAR variant.

**Is GARCH a straw man?** It uses pooled Student-t GARCH/GJR likelihoods, strict close-to-close returns, and the same walk-forward schedule. But return variance is a proxy for intraday RV, and shared cross-symbol parameters constrain it. This is not a verdict on all GARCH specifications.

**Does this make money?** That was not tested. There are no strategy returns, execution model, fees, slippage or profitability claims here.

**Could Chronos have seen this data in pretraining?** Possibly; overlap is unaudited. Local leakage controls cannot establish pretraining independence.

**Why only three seeds?** Laptop budget. Seed resampling reflects the observed runs, not the entire distribution of training randomness.

## Limitations

- Pooled GARCH uses a daily-return variance proxy; missing-return state updates are an expected-state approximation.
- Strict day dropping trades coverage for quality and can change which regimes remain.
- Validation is reused for early stopping, family selection and calibration; it is not independent generalization evidence.
- Three seeds only; limited epochs and multiple descriptive comparisons.
- Unaudited pretraining overlap.
- One crypto venue, spot data only. No cross-venue or asset-class generalization claim.
- A few regimes and shocks strongly influence the test period; removing one event is not a cure.
- The two-hour end-to-end target is not established by partial compute timings. Cold downloads, preparation and evaluation add work.

## Repo map

- `src/data.py`, `src/baselines.py`, `src/finetune.py`, `src/evaluate.py`: data, models and one shared scoring rule.
- `src/zero_shot.py`, `src/ablations.py`, `src/regimes.py`: causal contexts and analysis.
- `configs/`: fixed experiment settings and pinned model revisions.
- `results/`: immutable historical CSVs/figures plus new Stage 7 reporting CSVs.
- `scripts/make_readme_figures.py`: every new report figure from committed CSVs; `docs/img/`: generated PNGs and local badges.
- `tests/`: leakage, metrics, training safeguards and README/CSV/link checks.
- `data/`: ignored caches and checkpoints. Never committed.

## How to rerun

Use Python in the range shown above and install the pinned extras:

```powershell
python -m pip install -e ".[test,zero-shot,finetune,analysis]"
python scripts/reproduce.py
```

The default is a **CSV-only report rebuild**: no raw-data download, inference or training.
`python scripts/make_readme_figures.py` regenerates all new figures by itself; the three existing error plots stay untouched.
`python scripts/reproduce.py --rescore` additionally rebuilds the small Stage 7 exports from existing `data/stage5/scored.parquet`.
`python scripts/reproduce.py --experiments` runs the complete experiment pipeline in a separate ignored workspace, preserving published results; this is expensive and opt-in.
Install a CUDA-compatible PyTorch build to use the GPU; otherwise inference/training falls back to CPU.
Check with `python -m pytest -q`. No API key is required for public Binance archives or the pinned public model weights.

### The laptop bill

Recorded on the RTX 3050 laptop; CPU stages are identified. These components are not additive end-to-end wall times.
Missing timings are marked, not guessed. Stage 6 reports a local report-generation measurement when available.

'''+markdown(runtime, ['stage', 'Measured minutes', 'component', 'hardware'])+'''

Source: [runtime CSV](results/stage6_runtime.csv), derived from the original per-stage cost CSVs. Timing excludes different overheads by stage;
historical GPU allocations exclude driver memory and other applications.

## Credit where due

Built on [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting) and verified public [Binance market-data archives](https://data.binance.vision/).
See [CITATION.cff](CITATION.cff) for project attribution and the upstream Chronos reference. Upstream models and market data retain their own terms.

## License

Project code and original documentation: [MIT](LICENSE). The license does not relicense third-party model weights or Binance data.
'''
    Path('README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
