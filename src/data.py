"""Causal UTC daily realized variance and explicitly dated forecast labels."""

import numpy as np
import pandas as pd

COLUMNS = ["time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def daily_variance(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Require all 1440 unique valid minutes; never fill gaps or cross UTC days.

    The 288 five-minute returns use midnight open then closes at 00:04,
    00:09, ..., 23:59. Day t's RV is available only after day t closes.
    Unlocatable timestamps reject the archive rather than silently lose rows.
    """
    frame = raw.apply(pd.to_numeric, errors="coerce")
    stamp = frame["time"]
    us = stamp.where(stamp >= 10**14, stamp * 1000)
    frame["time"] = pd.to_datetime(us, unit="us", utc=True, errors="coerce")
    if frame.time.isna().any():
        raise ValueError("Unparseable timestamp: cannot assign bad row to a day")
    first, last = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if ((frame.time < first) | (frame.time >= last + pd.Timedelta(days=1))).any():
        raise ValueError("Archive contains timestamps outside its requested month")
    prices = frame[["open", "high", "low", "close"]]
    close_us = frame.close_time.where(frame.close_time >= 10**14, frame.close_time * 1000)
    close_time = pd.to_datetime(close_us, unit="us", utc=True, errors="coerce")
    valid = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
             & frame.high.ge(prices.max(axis=1)) & frame.low.le(prices.min(axis=1))
             & np.isfinite(frame.volume) & frame.volume.ge(0)
             & frame.time.eq(frame.time.dt.floor("min"))
             & close_time.ge(frame.time + pd.Timedelta(seconds=59))
             & close_time.lt(frame.time + pd.Timedelta(minutes=1)))
    frame["bad"] = ~valid | frame.time.duplicated(keep=False)
    groups = {day: group for day, group in frame.groupby(frame.time.dt.floor("D"))}
    rows = []
    for day in pd.date_range(first, last, freq="D"):
        group = groups.get(day)
        count = 0 if group is None else len(group)
        bad = 0 if group is None else int(group.bad.sum())
        reason = "missing_day" if count == 0 else "bad_or_duplicate_rows" if bad else "missing_minutes" if count != 1440 else "ok"
        rv = np.nan
        if reason == "ok":
            group = group.sort_values("time")
            endpoints = np.r_[group.open.iloc[0], group.close.iloc[4::5].to_numpy()]
            rv = float(np.square(np.diff(np.log(endpoints))).sum())
            if not np.isfinite(rv) or rv <= 0:
                reason, rv = "nonpositive_rv", np.nan
        rows.append((day, rv, np.log(rv) if rv > 0 else np.nan, reason, count, bad))
    return pd.DataFrame(rows, columns=["date", "rv", "y", "reason", "minute_rows", "bad_rows"])


def split_for(date: pd.Series) -> pd.Series:
    """Assign fixed chronological partitions by UTC calendar date."""
    return pd.Series(np.select([date < pd.Timestamp("2024-01-01", tz="UTC"),
                                date < pd.Timestamp("2025-01-01", tz="UTC")],
                               ["train", "validation"], default="test"), index=date.index)


def forecast_examples(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Index labels by target END t; features stop at origin t-h.

    Target is mean log RV over origin+1 through t. It is unavailable at
    origin, and only eligible for fitting at/after t. Calendar gaps stay NaN.
    Samples spanning partition boundaries are excluded.
    """
    if horizon not in (1, 5, 22):
        raise ValueError("Supported horizons: 1, 5, 22")
    parts = []
    for symbol, group in daily.groupby("symbol"):
        y = group.set_index("date").y.sort_index().asfreq("D")
        out = pd.DataFrame({"target": y.rolling(horizon).mean(),
                            "daily": y.shift(horizon),
                            "weekly": y.rolling(5).mean().shift(horizon),
                            "monthly": y.rolling(22).mean().shift(horizon)})
        out.index.name = "target_end"
        out = out.reset_index()
        out["origin"] = out.target_end - pd.Timedelta(days=horizon)
        out["target_start"] = out.origin + pd.Timedelta(days=1)
        out["split"] = split_for(out.target_end)
        out = out[out["split"].eq(split_for(out.target_start))]
        out["symbol"], out["horizon"] = symbol, horizon
        parts.append(out.dropna())
    return pd.concat(parts, ignore_index=True)
