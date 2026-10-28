# Chronos-Bolt realized-volatility forecasting

A reproducible forecasting-accuracy study comparing Chronos-Bolt with econometric baselines on crypto realized variance. Implementation follows AGENTS.md in six reviewed stages. **Stage 2: baseline results available; no Chronos evaluation yet.**

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

`data/daily_rv.parquet` contains the full calendar, RV, log RV, valid final-minute close, decimal daily log return, quality reason, and split/held-out flags. `data/examples.parquet` contains horizon labels and causal daily/weekly/monthly features. Stage 2 fits both HAR feature variants using this data interface. Per-month daily parquet and provenance caches allow restart without downloading verified months again. Source ZIPs and minute data are processed in memory and not retained.

Small committed CSVs record coverage, drop reasons by split, example counts, and source checksums. A 404 is reported as `archive_unavailable`, retried on subsequent runs, and never silently assumed to mean pre-listing; inspect the manifest for coverage limitations. Other network errors, checksum mismatches, malformed CSVs, or unassignable timestamps fail the run. Existing verified caches are reused; to pick up upstream archive revisions, remove the relevant local monthly cache pair and rerun.

The full Stage 1 audit verified 641 of 648 monthly archives. SOLUSDT January–July 2020 returned 404 (213 days). Of 2,465 requested days per symbol, drops were 16 each for BTC, ETH, BNB, ADA, DOGE and LTC, 15 for XRP, and 236 for SOL. All validation and test days passed. See [coverage](results/stage1_coverage.csv), [drop reasons](results/stage1_drop_reasons.csv), [example counts](results/stage1_examples.csv), and [source manifest](results/stage1_sources.csv). There are 48,380 complete horizon examples after calendar and split-boundary exclusions. The original Stage 1 suite includes four temporal-leakage cases; Stage 2 extends these to both HAR feature definitions and return/fit/calibration checks.

The quality rule is deliberately strict and may disproportionately remove unusual market days. Exponentiating a horizon label recovers the arithmetic mean of daily RV. The original HAR uses daily log RV and averages of past log RV. A second HAR uses log arithmetic-mean RV for its weekly/monthly features. Stage 1 timing does not establish the two-hour budget for the future full modeling pipeline. This is a forecasting-accuracy study, with no trading-profitability or transaction-cost claims.


## Stage 1 audit details

[Complete example counts by split, horizon and symbol group](results/stage1_example_groups.csv) distinguish the six fit-eligible symbols (BTC, ETH, BNB, XRP, ADA, LTC) from held-out SOL and DOGE. Here fit-eligible names a **symbol group** in every split; the row-level `fit_eligible` flag remains true only for training rows of those six symbols. Validation and test rows are never fitting rows.

[Every dropped training date](results/stage1_dropped_train_dates.csv) is listed per symbol with its reason and a `common_to_all_symbols` flag. The 15 dates common to all eight are: 2020-02-09, 2020-02-19, 2020-03-04, 2020-04-25, 2020-06-28, 2020-11-30, 2020-12-21, 2020-12-25, 2021-02-11, 2021-03-06, 2021-04-20, 2021-04-25, 2021-08-13, 2021-09-29, 2023-03-24. The first five overlap unavailable SOL archives; common exclusion does not establish a common cause.

BTC, ETH, BNB, ADA, DOGE and LTC each drop those 15 dates plus 2021-12-24. XRP drops exactly those 15. SOL drops every date from 2020-01-01 through 2020-08-11 inclusive, plus 2020-11-30, 2020-12-21, 2020-12-25, 2021-02-11, 2021-03-06, 2021-04-20, 2021-04-25, 2021-08-13, 2021-09-29, 2021-12-24, 2023-02-14, and 2023-03-24. The CSV expands the inclusive range into individual dates.

## Shared scoring contract

All models (naive, HAR, GARCH and, when introduced, Chronos) must use one shared evaluation function, the same horizon target, and the same eligible symbol/origin/horizon rows. For horizon h, let `V = mean(daily RV over the next h days)` and `z = log(V)`. Score RMSE and MAE against z using each model's log-scale point forecast `mu`. A native variance forecast (for example GARCH) must first be aggregated to the horizon arithmetic mean and adapted as `mu = log(native forecast)`.

For QLIKE, apply the same residual-smearing calibration procedure to every model: `e = z - mu`, `S_model,h = mean(exp(e))`, and `V_hat = exp(mu) * S_model,h`. Estimate S separately by model and horizon from causal out-of-sample **2024 validation** residuals on the six fit-eligible symbols, on a common calibration row set, then freeze it before test scoring and reuse it for held-out symbols. Do not use in-sample fit residuals, held-out-symbol residuals, or test residuals for calibration. The smearing factor corrects log-to-level retransformation empirically; with zero-mean Gaussian residuals it reduces to `exp(sigma^2 / 2)`. It need not be unbiased if the residual distribution shifts.

The shared evaluator owns this correction and applies it exactly once, including to native-variance models after their log-scale adapter; this is a common calibration policy rather than a claim that a native variance forecast itself requires a lognormal correction. Score `QLIKE = V_hat / V - log(V_hat / V) - 1`, preserving the ratio orientation specified in AGENTS.md. RMSE/MAE use the uncorrected mu. Persist the calibration factors and sample counts, reject nonfinite/nonpositive variance inputs, and report failed forecasts rather than silently changing the comparison sample. Validation-based calibration is part of the final pipeline, so validation QLIKE after calibration is not an independent estimate of generalization. Stage 2 implements this shared path in `src/evaluate.py`; future models must call it too. Both raw QLIKE (smearing factor 1) and calibrated QLIKE are reported with confidence intervals, DM tests and MCS results; the calibration choice is not selected on test performance.


## Reproduce Stage 2

```powershell
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts/run_baselines.py
```

This single command regenerates Stage 1 inputs and all Stage 2 CSVs. `--reuse-data` skips the download step only when the daily parquet already includes returns and both HAR feature sets. Monthly caches lacking valid daily closes are re-downloaded and checksum-verified automatically. No raw archives, minute data, forecasts or model fits are committed; predictions, scored rows and fitted-parameter intermediates are cached as parquet in ignored `data/`.

### Returns and model definitions

Daily return is `log(C_t / C_{t-1})`, where C is the last 1m close of a UTC day. Both days must pass the identical 1,440-minute RV quality filter, have positive RV, and be consecutive calendar dates. A bad day also invalidates the following return; no price filling or multi-day return is substituted. The first day has no return because the study starts in January 2020. All 366 validation days and 638 test days per symbol have returns. Training return counts are 1,428 for BTC/ETH/BNB/ADA/DOGE/LTC, 1,430 for XRP and 1,212 for SOL; [coverage audit](results/stage2_return_coverage.csv).

- **Naive:** last observed log RV, at every horizon.
- **HAR-RV:** horizon-specific OLS with intercept, daily log RV, mean past 5 daily log RVs, and mean past 22 daily log RVs.
- **HAR-log-mean-RV:** same OLS design, but weekly/monthly inputs are log of arithmetic-mean past RV. The daily feature is unchanged. Both HAR targets are log arithmetic-mean future RV.
- **GARCH-t / GJR-GARCH-t:** zero-mean Student-t GARCH(1,1) and GJR-GARCH(1,1), with returns multiplied by 100 for estimation. Forecasts average the next h daily conditional variances, divide by 10,000, then take logs before shared scoring. They forecast daily return variance as a proxy for daily RV; these are distinct realized quantities.

All fitted models share parameters across the six fit-eligible symbols, with no symbol intercepts. HAR pools eligible examples. GARCH maximizes the sum of six separate-series likelihoods using `arch.univariate.StudentsT` and constrained SciPy SLSQP. A compiled recursion preserves each series' calendar and independent state, with stationary unconditional initialization. It matches `arch.univariate.GARCH.compute_variance` on complete data. GJR permits either sign of gamma subject to alpha >= 0, alpha+gamma >= 0, beta >= 0 and alpha+gamma/2+beta <= 0.999. Degrees of freedom are bounded to [2.05,100]; zero mean, model orders, constraints and optimizer settings are fixed rather than selected using test results.

Missing returns contribute no likelihood; state advances using the expected squared shock and half of it for the negative-shock term under symmetric Student-t innovations. This preserves elapsed days without fabricated returns, but is an expected-state **quasi-likelihood approximation** over missing intervals, not exact integration of missing shocks. This pooled likelihood adapter is intentional: the stock single-series fit would not provide shared held-out parameters with this calendar-gap handling.

### Walk-forward and uncertainty

Separate validation and test schedules start at the first forecast origin (2023-12-31 and 2024-12-31), then refit every 30 calendar days: 13 validation and 22 test refits per fitted model. All available six-symbol histories expand through the refit date. HAR training labels must have ended by that date. Observed earlier test data may enter later scheduled fits, but no future target or test-based hyperparameter selection is allowed. GARCH refits once per date for all horizons; HAR refits separately for each horizon. Between refits, parameters stay fixed while features and GARCH states update daily. SOL/DOGE supply their past features and filter state only, never parameter-fitting or calibration observations.

Every model has the same balanced panel: 638/634/617 test origins for h=1/5/22. Errors are averaged equally across symbols and dates within each group. The 95% percentile CIs use 1,000 circular block-bootstrap samples of 30 consecutive dates, moving all symbols and models together to preserve temporal and cross-symbol dependence. They condition on fitted forecasts and do not refit models or re-estimate calibration in each bootstrap sample.

DM comparisons use daily cross-symbol mean QLIKE differences against **HAR-RV**, separately for raw and calibrated losses. Bartlett/Newey-West HAC bandwidth is max(h-1, floor(4*(T/100)^(2/9))); Harvey's small-sample factor and a t(T-1) reference give two-sided p-values. Negative DM statistics favor the candidate. Individual DM p-values are unadjusted for multiple comparisons. The optional `arch.bootstrap.MCS` is available here: 95% MCS with the R statistic, 1,000 circular-bootstrap samples and block length 30, separately by group, horizon and QLIKE version. Fixed seed is 42. See [arch forecasting documentation](https://arch.readthedocs.io/en/latest/univariate/forecasting.html), [Harvey et al. (1997)](https://doi.org/10.1016/S0169-2070(96)00719-4), and [arch MCS documentation](https://arch.readthedocs.io/en/latest/multiple-comparison/generated/arch.bootstrap.MCS.html).

### Baseline results and costs

[Full baseline table](results/stage2_baselines.csv) contains 30 model/group/horizon rows, RMSE/MAE, raw and calibrated QLIKE, all 95% CIs, both DM tests, both MCS results and costs. [Calibration factors](results/stage2_calibration.csv), [fit audit](results/stage2_fit_audit.csv), and [cost measurements](results/stage2_costs.csv) record the frozen factors, refit parameters and timings.

Original HAR-RV has the lowest raw QLIKE in both groups at every horizon. Smearing worsens HAR and naive QLIKE on this test period, while reducing it for the GARCH variants. The alternative HAR feature definition does not consistently improve performance. These are pooled parameter-transfer baselines, not independently fitted per-symbol benchmarks; the distinction matters for interpreting held-out results.

The daily-close cache rebuild took about 192 seconds; the full five-model fit/evaluation run using those caches took about 17 seconds on this laptop. Runtime is measured, not a future Chronos runtime guarantee. Cost columns aggregate fit time over validation and test refits; inference includes state filtering and dataframe overhead, excludes fitting/calibration/metric computation, and is normalized per 1,000 scalar symbol/horizon forecasts. Parameter counts are per fitted horizon model (HAR has 4 per horizon, GARCH 4 shared across horizons, GJR 5); the additional frozen smearing factor is reported separately. GPU memory is zero because this stage runs on CPU. Timing CSV values may vary across runs; forecast and loss outputs are deterministic for the pinned environment.
