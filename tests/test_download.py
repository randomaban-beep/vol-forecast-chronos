"""Offline tests for download integrity, missing archives and cache reuse."""

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZipFile
import hashlib

import pandas as pd
import pytest

from scripts import download_data as downloader


def test_missing_archive_is_explicit_and_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(url: str) -> bytes:
        raise HTTPError(url, 404, "Not found", {}, None)

    monkeypatch.setattr(downloader, "fetch", missing)
    daily, source = downloader.month_data(("BTCUSDT", pd.Period("2020-01", "M"), tmp_path))
    assert len(daily) == 31
    assert daily.reason.eq("archive_unavailable").all()
    assert source["status"] == "404"
    assert not list(tmp_path.rglob("*.parquet"))


def test_checksum_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, "fetch", lambda url: b"0" * 64 if url.endswith("CHECKSUM") else b"zip")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        downloader.month_data(("BTCUSDT", pd.Period("2020-01", "M"), tmp_path))


def test_verified_cache_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        # One valid minute is intentionally incomplete coverage for the month.
        archive.writestr("BTCUSDT-1m-2020-01.csv", "1577836800000,1,2,1,2,1,1577836859999,1,1,1,1,0\n")
    payload = buffer.getvalue()
    monkeypatch.setattr(downloader, "fetch", lambda url: hashlib.sha256(payload).hexdigest().encode()
                        if url.endswith("CHECKSUM") else payload)
    job = ("BTCUSDT", pd.Period("2020-01", "M"), tmp_path)
    daily, source = downloader.month_data(job)
    assert source["status"] == "verified"
    assert daily.reason.iloc[0] == "missing_minutes"

    def no_network(url: str) -> bytes:
        raise AssertionError("Cache hit must not access network")

    monkeypatch.setattr(downloader, "fetch", no_network)
    cached, cached_source = downloader.month_data(job)
    pd.testing.assert_frame_equal(daily, cached)
    assert source == cached_source
