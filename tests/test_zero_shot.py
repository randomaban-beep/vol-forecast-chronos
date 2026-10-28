"""Common-panel alignment, causal missing-value contexts and horizon conversion."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_zero_shot import append_preserving_baselines
from src.zero_shot import build_contexts, path_to_horizon, align_forecasts, common_template, KEYS


def fixture_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small panel with an interior calendar gap and repeated horizon origins."""
    dates = pd.date_range("2023-01-01", periods=400, tz="UTC")
    daily = pd.DataFrame({"symbol": "BTCUSDT", "date": dates, "y": np.arange(400, dtype=float)})
    daily = daily.drop(200)
    rows = pd.DataFrame([{"split": "validation", "symbol": "BTCUSDT", "origin": dates[300 + i],
                          "horizon": h, "target": 1.0} for i in range(3) for h in [1, 5, 22]])
    return daily, rows


def test_context_dates_end_at_origin_without_filling() -> None:
    daily, rows = fixture_panel()
    origins, contexts = build_contexts(daily, rows)
    assert contexts.shape == (3, 256)
    assert np.isnan(contexts).sum() == 3
    assert contexts[0, 0] == 45 and contexts[0, -1] == 300
    assert np.isnan(contexts[0, 155])  # Day 200 remains exactly in its calendar slot.
    assert (origins.context_end == origins.origin).all()
    assert (origins.context_end - origins.context_start).dt.days.eq(255).all()


def test_no_future_data_in_any_context() -> None:
    daily, rows = fixture_panel()
    origins, contexts = build_contexts(daily, rows)
    for i, row in origins.iterrows():
        changed = daily.copy()
        changed.loc[changed.date > row.origin, "y"] = -9999
        _, new = build_contexts(changed, rows)
        np.testing.assert_array_equal(contexts[i], new[i])


@pytest.mark.parametrize("h", [1, 5, 22])
def test_known_path_conversion(h: int) -> None:
    variance = np.arange(1, 23, dtype=float)
    paths = np.log(np.stack([variance, variance * 3]))
    expected = np.log([variance[:h].mean(), 3 * variance[:h].mean()])
    np.testing.assert_allclose(path_to_horizon(paths, h), expected)
    if h > 1:
        assert not np.allclose(path_to_horizon(paths, h), paths[:, :h].mean(axis=1))


def test_conversion_numerically_stable_and_invalid_paths_fail() -> None:
    np.testing.assert_allclose(path_to_horizon(np.full((2, 22), 1000.), 22), 1000.)
    with pytest.raises(ValueError):
        path_to_horizon(np.full((1, 22), np.nan), 22)


def test_forecasts_exactly_match_common_panel() -> None:
    daily, rows = fixture_panel()
    baselines = pd.concat([rows.assign(model=m, prediction=0., inference_seconds=0.)
                           for m in ["HAR-RV", "Naive"]], ignore_index=True)
    template = common_template(baselines)
    origins, _ = build_contexts(daily, template)
    out = align_forecasts(template, origins, np.zeros((3, 22)), "Chronos-Bolt-tiny")
    pd.testing.assert_frame_equal(out[KEYS], template[KEYS])
    assert len(out) == len(rows) and out.prediction.eq(0).all()
    with pytest.raises(ValueError, match="Baseline rows differ"):
        common_template(baselines.iloc[:-1])
    with pytest.raises(ValueError, match="Missing common-panel"):
        align_forecasts(template, origins.iloc[:-1], np.zeros((2, 22)), "Chronos-Bolt-tiny")


def test_append_preserves_exact_baseline_bytes(tmp_path: Path) -> None:
    original = b"model,qlike\r\nHAR-RV,0.12345678901234567\r\nNaive,1.0000000000000000\r\n"
    path = tmp_path / "results.csv"
    append_preserving_baselines(original, pd.DataFrame({"model": ["Chronos"], "qlike": [.2]}), path)
    assert path.read_bytes().startswith(original)
    with pytest.raises(ValueError, match="overwrite"):
        append_preserving_baselines(original, pd.DataFrame({"model": ["HAR-RV"], "qlike": [.2]}), path)


def test_pinned_bolt_masks_nans_without_imputation() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("chronos")
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    from transformers import T5Config

    config = T5Config(d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
                      num_heads=2, d_kv=8, dropout_rate=0., decoder_start_token_id=0)
    config.chronos_config = {"context_length": 256, "prediction_length": 64,
                            "input_patch_size": 16, "input_patch_stride": 16,
                            "quantiles": [.1, .5, .9], "use_reg_token": False}
    torch.manual_seed(42)
    model = ChronosBoltModelForForecasting(config).eval()
    context = torch.arange(256, dtype=torch.float32).reshape(1, -1)
    context[:, :16] = torch.nan
    context[:, 25] = torch.nan
    with torch.inference_mode():
        _, loc_scale, _, attention_mask = model.encode(context)
        out = model(context=context).quantile_preds
    assert not attention_mask[0, 0] and attention_mask[0, 1]
    assert loc_scale[0].item() == pytest.approx(torch.nanmean(context).item())
    assert torch.isfinite(out).all()
    assert torch.isnan(context).sum() == 17
