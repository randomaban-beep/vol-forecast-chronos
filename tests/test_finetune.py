"""Partition isolation, early stopping and actual PEFT trainable-weight tests."""

import numpy as np
import pandas as pd
import pytest

from src.finetune import training_windows, EarlyStop, adapt_model


def test_training_windows_exclude_future_and_heldout() -> None:
    dates = pd.date_range("2020-01-01", "2025-01-31", tz="UTC")
    daily = pd.DataFrame({"date": dates, "symbol": "BTCUSDT", "y": np.arange(len(dates), dtype=float)})
    daily = pd.concat([daily, daily.assign(symbol="SOLUSDT", y=-10)], ignore_index=True)
    config = {"held_out": ["SOLUSDT"], "context_length": 256, "prediction_length": 22}
    meta, x, y = training_windows(daily, "train", config)
    assert meta.symbol.eq("BTCUSDT").all()
    assert meta.target_end.max() <= pd.Timestamp("2023-12-31", tz="UTC")
    assert (meta.target_start == meta.origin + pd.Timedelta(days=1)).all()
    np.testing.assert_array_equal(x[:, -1] + 1, y[:, 0])
    changed = daily.copy()
    changed.loc[(changed.date >= pd.Timestamp("2024-01-01", tz="UTC")) | changed.symbol.eq("SOLUSDT"), "y"] = 99999
    altered = training_windows(changed, "train", config)
    pd.testing.assert_frame_equal(meta, altered[0])
    np.testing.assert_array_equal(x, altered[1])
    np.testing.assert_array_equal(y, altered[2])
    val_meta, val_x, val_y = training_windows(daily, "validation", config)
    assert val_meta.target_start.min() >= pd.Timestamp("2024-01-01", tz="UTC")
    assert val_meta.target_end.max() <= pd.Timestamp("2024-12-31", tz="UTC")
    changed = daily.copy()
    changed.loc[changed.date >= pd.Timestamp("2025-01-01", tz="UTC"), "y"] = -99999
    after = training_windows(changed, "validation", config)
    np.testing.assert_array_equal(val_x, after[1])
    np.testing.assert_array_equal(val_y, after[2])


def test_training_gaps_preserved_and_incomplete_labels_dropped() -> None:
    dates = pd.date_range("2020-01-01", periods=400, tz="UTC")
    daily = pd.DataFrame({"date": dates, "symbol": "X", "y": np.arange(400, dtype=float)})
    daily.loc[260, "y"] = np.nan
    meta, x, y = training_windows(daily, "train", {"held_out": [], "context_length": 256, "prediction_length": 22})
    assert np.isnan(x).any() and np.isfinite(y).all()
    gap = dates[260]
    assert not ((meta.target_start <= gap) & (meta.target_end >= gap)).any()


def test_validation_only_early_stopping_includes_epoch_zero() -> None:
    stop = EarlyStop(patience=1, min_delta=1e-5)
    assert stop.update(0, 1.)
    assert stop.update(1, .8)
    assert not stop.update(2, .9)
    assert stop.best_epoch == 1 and stop.stale == 1
    assert stop.best_loss == .8
    with pytest.raises(ValueError):
        stop.update(3, np.nan)


@pytest.mark.parametrize("mode", ["full", "lora8", "lora16"])
def test_real_peft_gradient_routing(mode: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import T5Config
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    config = T5Config(d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2,
                      d_kv=8, dropout_rate=0., decoder_start_token_id=0)
    config.chronos_config = {"context_length": 256, "prediction_length": 64,
        "input_patch_size": 16, "input_patch_stride": 16, "quantiles": [.1, .5, .9], "use_reg_token": False}
    model = adapt_model(ChronosBoltModelForForecasting(config), mode, {"lora_targets": ["q", "v"], "lora_dropout": 0.})
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if mode != "full":
        assert all("lora_" in n for n, _ in trainable)
        assert len(trainable) > 0
    model(context=torch.randn(2, 256), target=torch.randn(2, 22)).loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in trainable)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_seed_summary_is_mean_metric_not_ensemble() -> None:
    from src.finetune_summary import seed_statistics
    losses = np.array([np.tile([i*i, i, i, i], (40, 1)) for i in [0., 1., 2.]])
    config = {"seed": 42, "block_length": 10, "bootstrap_reps": 100}
    mean, std, ci = seed_statistics(losses, config)
    np.testing.assert_allclose(mean, [1, 1, 1, 1])
    assert mean[0] != pytest.approx(np.sqrt(5/3))
    np.testing.assert_allclose(std, [1, 1, 1, 1])
    assert ci.shape == (2, 4)
    np.testing.assert_array_equal(ci, seed_statistics(losses, config)[2])


def test_identical_family_losses_share_mcs() -> None:
    from src.finetune_summary import family_mcs
    losses = pd.DataFrame({"zero-shot": np.arange(40.), "epoch-zero-ft": np.arange(40.)})
    result = family_mcs(losses, {"seed": 42, "block_length": 10, "bootstrap_reps": 100})
    assert result.eq(1.).all()
