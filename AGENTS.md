# Project: Fine-tuning Chronos-Bolt for Realized-Volatility Forecasting

## Goal
Test whether LoRA/fine-tuning a pretrained time-series foundation model (Chronos-Bolt) beats
standard econometric baselines (HAR-RV, GARCH) for forecasting crypto realized volatility.
Deliver a reproducible repo with ONE headline results table and chart at the top of README.md.

## Hardware constraints (must respect)
- Windows laptop, NVIDIA RTX 3050 6 GB, 24 GB RAM.
- Use `amazon/chronos-bolt-tiny` (9M) first, then `amazon/chronos-bolt-small` (48M) if it fits.
- Batch size <= 32, context length <= 256, mixed precision (fp16/bf16), gradient checkpointing if needed.
- Whole pipeline should run end to end in under 2 hours. Cache all intermediate data as parquet.

## Data
- Source: public Binance spot klines from https://data.binance.vision (monthly zips, 1m interval).
- Symbols: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LTCUSDT (8 total).
- Period: 2020-01-01 to 2026-09-30.
- Build daily realized variance RV_t = sum of squared 5-minute log returns, then target y_t = log(RV_t).
- Handle gaps and bad rows explicitly; log how many days were dropped.
- Never commit raw data; provide `scripts/download_data.py`.

## Splits (no lookahead leakage)
- Train: 2020-01 to 2023-12. Validation: 2024-01 to 2024-12. Test: 2025-01 to 2026-09.
- Held-out symbol experiment: train on 6 symbols, test on the 2 held out (SOLUSDT, DOGEUSDT).
- Walk-forward evaluation on test: expanding window for econometric baselines, refit every 30 days.
- Fine-tuned model is trained once on train, early-stopped on validation, never sees test.
- Add a unit test that asserts no feature or target at time t uses data after t.

## Forecast horizons
h = 1, 5, 22 days (next day, next week average, next month average of log RV).

## Models (rows of the results matrix)
1. Naive: y_hat = last observed log RV.
2. HAR-RV (daily, weekly, monthly lags), OLS.
3. GARCH(1,1) with Student-t, and one asymmetric variant (GJR-GARCH or EGARCH), via the `arch` package.
4. Chronos-Bolt zero-shot (no fine-tuning).
5. Chronos-Bolt fine-tuned, full fine-tune.
6. Chronos-Bolt fine-tuned, LoRA (PEFT), rank 8 and 16.

## Metrics (columns)
- RMSE and MAE on log RV, and QLIKE on the variance scale (QLIKE = RV_hat/RV - log(RV_hat/RV) - 1).
- Diebold-Mariano test (HAC variance, Harvey small-sample correction) of each model vs HAR-RV, on QLIKE.
- Model Confidence Set (95%) across all models, if the `arch` MCS implementation is available.
- Cost columns: trainable parameters, training time, inference time per 1000 forecasts, GPU memory.
- Report mean and 95% bootstrap (block bootstrap) confidence intervals.

## Ablations and analysis
- Context length (32, 64, 128, 256) vs QLIKE.
- Training data size (10%, 25%, 50%, 100% of train) vs QLIKE.
- LoRA rank (4, 8, 16, 32) vs QLIKE and trainable parameters.
- Performance in high-volatility regimes (top decile of RV) vs the rest.
- Error analysis: three plots of the worst forecasts and a short written explanation of why they failed.
- Honest negative results are fine. If fine-tuning does not beat HAR-RV, report that clearly.

## Repo structure
```
.
├── AGENTS.md
├── README.md              # headline table + chart first, then method, results, limitations
├── pyproject.toml
├── configs/               # yaml per experiment
├── scripts/download_data.py
├── src/
│   ├── data.py            # klines -> RV
│   ├── baselines.py       # naive, HAR, GARCH
│   ├── finetune.py        # full + LoRA training
│   ├── evaluate.py        # metrics, DM test, MCS, bootstrap
│   └── plots.py
├── tests/                 # leakage test, metric unit tests
├── results/               # csv tables, figures (committed)
└── notebooks/             # optional exploration only
```

## Build in stages (stop and summarize after each stage)
1. Data pipeline, RV construction, leakage test.
2. Baselines (naive, HAR, GARCH) with walk-forward evaluation and metrics code.
3. Chronos-Bolt zero-shot evaluation.
4. Fine-tuning (full, then LoRA) with config-driven runs and fixed seeds (3 seeds each).
5. Ablations and regime analysis.
6. README with headline table, chart, and limitations (no trading claims, no transaction-cost claims).

## Rules
- Fixed random seeds, pinned dependency versions, `make reproduce` or a single script that regenerates all tables.
- Do not tune hyperparameters on the test period.
- Do not claim trading profitability. This is a forecasting-accuracy study.
- Keep code typed and small; add docstrings to public functions.
