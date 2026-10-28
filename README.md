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
- Day t's RV becomes available after its final minute closes. Labels are indexed by **target end t**, with origin t−h; their value is mean log RV over origin+1 through t. Features end at the origin. A future label must never be used for fitting before its target end has closed. The leakage tests perturb future data and check both feature-origin and target-end cutoffs.
- Training targets end in 2020–2023; validation targets lie wholly within 2024; test targets lie wholly within 2025–September 2026. Targets crossing split boundaries are excluded. Histories can include observations from an earlier partition. SOLUSDT and DOGEUSDT are marked held out; `fit_eligible` excludes them. Later held-out experiments must also exclude them from validation/model selection.

## Outputs and limitations

`data/daily_rv.parquet` contains the full calendar, RV, log RV, quality reason, and split/held-out flags. `data/examples.parquet` contains horizon labels and causal daily/weekly/monthly features. These features prepare the data interface; no baseline has been fitted. Per-month daily parquet and provenance caches allow restart without downloading verified months again. Source ZIPs and minute data are processed in memory and not retained.

Small committed CSVs record coverage, drop reasons by split, example counts, and source checksums. A 404 is reported as `archive_unavailable`, retried on subsequent runs, and never silently assumed to mean pre-listing; inspect the manifest for coverage limitations. Other network errors, checksum mismatches, malformed CSVs, or unassignable timestamps fail the run. Existing verified caches are reused; to pick up upstream archive revisions, remove the relevant local monthly cache pair and rerun.

The quality rule is deliberately strict and may disproportionately remove unusual market days. Labels average **log RV**, so exponentiating a multi-day label gives geometric-mean variance; later variance-scale scoring must use that same target convention. Stage 1 timing does not establish the two-hour budget for the future full modeling pipeline. This is a forecasting-accuracy study, with no trading-profitability or transaction-cost claims.
