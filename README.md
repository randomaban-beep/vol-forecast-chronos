# Chronos-Bolt realized-volatility forecasting

A reproducible forecasting-accuracy study comparing Chronos-Bolt with econometric baselines on crypto realized variance. Implementation follows AGENTS.md in six reviewed stages. **Stage 1 only; no model results yet.**

## Reproduce Stage 1

Run from the repository root with Python 3.11 or 3.12:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts/download_data.py
```

The final command regenerates every Stage 1 CSV under `results/`. Configuration is in `configs/stage-1-data.yaml`. Eight workers process one month each, with bounded memory and no GPU. Data transformations are deterministic; seed 42 is reserved for later stochastic stages. Dependencies are pinned in `pyproject.toml`.

## Data conventions

- Source: [Binance public monthly spot 1m klines](https://github.com/binance/binance-public-data), January 2020–September 2026 inclusive, eight symbols. SHA-256 checksums are verified before parsing. Millisecond and microsecond timestamps are normalized to UTC.
- Each day requires 1,440 unique, aligned minute rows with finite positive consistent OHLC prices, valid volume and minute closing timestamps. Invalid, duplicate, missing, and zero-variance days are dropped from usable observations but retained as missing values on the daily calendar. No forward filling, interpolation, or compression across missing days occurs.
- Five-minute prices are midnight open followed by closes at 00:04, 00:09, …, 23:59. RV is the sum of 288 squared log returns; `y = log(RV)`. Using midnight open makes the first return wholly within the day and avoids dependence on the previous archive. This omits any price discontinuity between the previous day's close and midnight open.
- Day t's RV becomes available after its final minute closes. Labels are indexed by **target end t**, with origin t−h; their value is `log(mean(RV))` over origin+1 through t: `log((RV_{t-h+1} + ... + RV_t) / h)`. For h=1 this is daily log RV. The arithmetic mean is taken on the variance scale before logging, for h=5 and h=22 as well. Features end at the origin. A future label must never be used for fitting before its target end has closed. The leakage tests perturb future data and check both feature-origin and target-end cutoffs.
- Training targets end in 2020–2023; validation targets lie wholly within 2024; test targets lie wholly within 2025–September 2026. Targets crossing split boundaries are excluded. Histories can include observations from an earlier partition. SOLUSDT and DOGEUSDT are marked held out; `fit_eligible` excludes them. Later held-out experiments must also exclude them from validation/model selection.

## Outputs and limitations

`data/daily_rv.parquet` contains the full calendar, RV, log RV, quality reason, and split/held-out flags. `data/examples.parquet` contains horizon labels and causal daily/weekly/monthly features. These features prepare the data interface; no baseline has been fitted. Per-month daily parquet and provenance caches allow restart without downloading verified months again. Source ZIPs and minute data are processed in memory and not retained.

Small committed CSVs record coverage, drop reasons by split, example counts, and source checksums. A 404 is reported as `archive_unavailable`, retried on subsequent runs, and never silently assumed to mean pre-listing; inspect the manifest for coverage limitations. Other network errors, checksum mismatches, malformed CSVs, or unassignable timestamps fail the run. Existing verified caches are reused; to pick up upstream archive revisions, remove the relevant local monthly cache pair and rerun.

The full Stage 1 audit verified 641 of 648 monthly archives. SOLUSDT January–July 2020 returned 404 (213 days). Of 2,465 requested days per symbol, drops were 16 each for BTC, ETH, BNB, ADA, DOGE and LTC, 15 for XRP, and 236 for SOL. All validation and test days passed. See [coverage](results/stage1_coverage.csv), [drop reasons](results/stage1_drop_reasons.csv), [example counts](results/stage1_examples.csv), and [source manifest](results/stage1_sources.csv). There are 48,380 complete horizon examples after calendar and split-boundary exclusions. The test suite has 20 passing tests, including four temporal-leakage cases.

The quality rule is deliberately strict and may disproportionately remove unusual market days. Exponentiating a horizon label recovers the arithmetic mean of daily RV. Daily/weekly/monthly input features remain lagged daily log RV and averages of past log RV; this amendment changes the forecast target, not those input features. Stage 1 timing does not establish the two-hour budget for the future full modeling pipeline. This is a forecasting-accuracy study, with no trading-profitability or transaction-cost claims.


## Stage 1 audit details

[Complete example counts by split, horizon and symbol group](results/stage1_example_groups.csv) distinguish the six fit-eligible symbols (BTC, ETH, BNB, XRP, ADA, LTC) from held-out SOL and DOGE. Here fit-eligible names a **symbol group** in every split; the row-level `fit_eligible` flag remains true only for training rows of those six symbols. Validation and test rows are never fitting rows.

[Every dropped training date](results/stage1_dropped_train_dates.csv) is listed per symbol with its reason and a `common_to_all_symbols` flag. The 15 dates common to all eight are: 2020-02-09, 2020-02-19, 2020-03-04, 2020-04-25, 2020-06-28, 2020-11-30, 2020-12-21, 2020-12-25, 2021-02-11, 2021-03-06, 2021-04-20, 2021-04-25, 2021-08-13, 2021-09-29, 2023-03-24. The first five overlap unavailable SOL archives; common exclusion does not establish a common cause.

BTC, ETH, BNB, ADA, DOGE and LTC each drop those 15 dates plus 2021-12-24. XRP drops exactly those 15. SOL drops every date from 2020-01-01 through 2020-08-11 inclusive, plus 2020-11-30, 2020-12-21, 2020-12-25, 2021-02-11, 2021-03-06, 2021-04-20, 2021-04-25, 2021-08-13, 2021-09-29, 2021-12-24, 2023-02-14, and 2023-03-24. The CSV expands the inclusive range into individual dates.

## Shared scoring contract for Stage 2 and later

All models (naive, HAR, GARCH and, when introduced, Chronos) must use one shared evaluation function, the same horizon target, and the same eligible symbol/origin/horizon rows. For horizon h, let `V = mean(daily RV over the next h days)` and `z = log(V)`. Score RMSE and MAE against z using each model's log-scale point forecast `mu`. A native variance forecast (for example GARCH) must first be aggregated to the horizon arithmetic mean and adapted as `mu = log(native forecast)`.

For QLIKE, apply the same residual-smearing calibration procedure to every model: `e = z - mu`, `S_model,h = mean(exp(e))`, and `V_hat = exp(mu) * S_model,h`. Estimate S separately by model and horizon from causal out-of-sample **2024 validation** residuals on the six fit-eligible symbols, on a common calibration row set, then freeze it before test scoring and reuse it for held-out symbols. Do not use in-sample fit residuals, held-out-symbol residuals, or test residuals for calibration. The smearing factor corrects log-to-level retransformation empirically; with zero-mean Gaussian residuals it reduces to `exp(sigma^2 / 2)`. It need not be unbiased if the residual distribution shifts.

The shared evaluator owns this correction and applies it exactly once, including to native-variance models after their log-scale adapter; this is a common calibration policy rather than a claim that a native variance forecast itself requires a lognormal correction. Score `QLIKE = V_hat / V - log(V_hat / V) - 1`, preserving the ratio orientation specified in AGENTS.md. RMSE/MAE use the uncorrected mu. Persist the calibration factors and sample counts, reject nonfinite/nonpositive variance inputs, and report failed forecasts rather than silently changing the comparison sample. Validation-based calibration is part of the final pipeline, so validation QLIKE after calibration is not an independent estimate of generalization. This is a requirement for future stages only; Stage 2 has not been implemented.
