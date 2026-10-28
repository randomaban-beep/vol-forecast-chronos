"""Download checksum-verified Binance monthly 1m archives and cache parquet."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import logging
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import COLUMNS, daily_variance, forecast_examples, split_for

BASE = "https://data.binance.vision/data/spot/monthly/klines"
LOG = logging.getLogger(__name__)


def fetch(url: str) -> bytes:
    """Retry transient transport errors; a 404 remains explicit missing coverage."""
    for attempt in range(4):
        try:
            with urlopen(url, timeout=90) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404 or (exc.code < 500 and exc.code != 429):
                raise
            if attempt == 3:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def month_data(job: tuple[str, pd.Period, Path]) -> tuple[pd.DataFrame, dict]:
    """Process one bounded-memory month; cache daily data and source provenance."""
    symbol, month, root = job
    name = f"{symbol}-1m-{month}"
    cache = root / "monthly" / f"{name}.parquet"
    meta = root / "monthly" / f"{name}-source.parquet"
    if cache.exists() and meta.exists():
        return pd.read_parquet(cache), pd.read_parquet(meta).iloc[0].to_dict()
    url = f"{BASE}/{symbol}/1m/{name}.zip"
    start, end = str(month.start_time.date()), str(month.end_time.date())
    try:
        payload = fetch(url)
    except HTTPError as exc:
        if exc.code != 404:
            raise
        # Missing months are deliberately not cached: later reruns retry them.
        daily = daily_variance(pd.DataFrame(columns=COLUMNS), start, end)
        daily["reason"] = "archive_unavailable"
        source = {"symbol": symbol, "month": str(month), "url": url,
                  "status": "404", "sha256": "", "bytes": 0}
    else:
        checksum = fetch(url + ".CHECKSUM").decode().split()[0]
        digest = hashlib.sha256(payload).hexdigest()
        if digest != checksum:
            raise ValueError(f"Checksum mismatch: {url}")
        with ZipFile(BytesIO(payload)) as archive:
            names = archive.namelist()
            if names != [name + ".csv"]:
                raise ValueError(f"Unexpected ZIP members: {names}")
            with archive.open(names[0]) as stream:
                raw = pd.read_csv(stream, header=None)
        if raw.shape[1] != len(COLUMNS):
            raise ValueError(f"Unexpected columns: {url}")
        raw.columns = COLUMNS
        daily = daily_variance(raw, start, end)
        source = {"symbol": symbol, "month": str(month), "url": url,
                  "status": "verified", "sha256": digest, "bytes": len(payload)}
    daily["symbol"] = symbol
    if source["status"] == "verified":
        cache.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(cache.with_suffix(".tmp"), index=False)
        cache.with_suffix(".tmp").replace(cache)
        pd.DataFrame([source]).to_parquet(meta, index=False)
    LOG.info("%s: %d / %d days dropped", name, daily.y.isna().sum(), len(daily))
    return daily, source


def main() -> None:
    """Regenerate all Stage 1 data, labels, coverage tables and source manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage-1-data.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root, results = Path(config["cache"]), Path(config["results"])
    root.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.perf_counter()
    months = pd.period_range(config["start"], config["end"], freq="M")
    jobs = [(symbol, month, root) for symbol in config["symbols"] for month in months]
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        records = list(pool.map(month_data, jobs))
    daily = pd.concat([item[0] for item in records], ignore_index=True)
    daily = daily[daily.date.between(pd.Timestamp(config["start"], tz="UTC"),
                                    pd.Timestamp(config["end"], tz="UTC"))]
    daily["split"] = split_for(daily.date)
    daily["held_out"] = daily.symbol.isin(config["held_out"])
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    daily.to_parquet(root / "daily_rv.parquet", index=False)
    manifest = pd.DataFrame([item[1] for item in records])
    manifest.to_parquet(root / "manifest.parquet", index=False)
    manifest.to_csv(results / "stage1_sources.csv", index=False)
    examples = pd.concat([forecast_examples(daily, h) for h in config["horizons"]])
    examples["held_out"] = examples.symbol.isin(config["held_out"])
    examples["fit_eligible"] = examples["split"].eq("train") & ~examples.held_out
    examples.to_parquet(root / "examples.parquet", index=False)
    counts = daily.groupby("symbol").agg(expected_days=("date", "size"), kept_days=("y", "count"))
    counts["dropped_days"] = counts.expected_days - counts.kept_days
    counts.to_csv(results / "stage1_coverage.csv")
    daily.groupby(["symbol", "split", "reason"]).size().rename("days").to_csv(results / "stage1_drop_reasons.csv")
    examples.groupby(["symbol", "split", "horizon"]).size().rename("examples").to_csv(results / "stage1_examples.csv")
    print(counts.to_string())
    print(f"Stage 1 elapsed seconds: {time.perf_counter() - started:.1f}")


if __name__ == "__main__":
    main()
