"""Fixed Supertrend arithmetic and causal forecasts on small fake bars only."""

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e35_supertrend as e35


@pytest.fixture(scope="module")
def market():
    dates = pd.bdate_range("1993-01-01", "1997-12-31")
    rng = np.random.default_rng(35)
    x = np.arange(len(dates))
    return pd.DataFrame({
        "baa": 2 + 0.3 * np.sin(x / 50), "aaa": 1 + 0.1 * np.sin(x / 50),
        "vix": 20 + 3 * np.sin(x / 70),
        "spx": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
        "usdinr": 40 + x / 5000, "ust10y": 4 + 0.3 * np.cos(x / 80),
    }, index=dates)


@pytest.fixture(scope="module")
def bars(market):
    return pd.DataFrame({"high": market["spx"] * 1.01, "low": market["spx"] * 0.99,
                         "close": market["spx"]})


def test_hand_calculated_wilder_atr_final_bands_and_reversals():
    bars = pd.DataFrame({"high": [11, 12, 13, 8, 7, 12, 13, 6],
                         "low": [9, 10, 11, 6, 5, 10, 11, 4],
                         "close": [10, 11, 12, 7, 6, 11, 12, 5]},
                        index=pd.bdate_range("2000-01-03", periods=8))
    signal = e35.supertrend(bars, period=2, multiplier=1)
    np.testing.assert_array_equal(signal["true_range"], [2, 2, 2, 6, 2, 6, 2, 8])
    np.testing.assert_allclose(signal["atr"], [np.nan, 2, 2, 4, 3, 4.5, 3.25, 5.625])
    np.testing.assert_allclose(signal["line"], [np.nan, 13, 13, 11, 9, 6.5, 8.75, 10.625])
    np.testing.assert_array_equal(signal["direction"], [0, -1, -1, -1, -1, 1, 1, -1])


def test_constant_bar_warm_up_and_equality_do_not_flip():
    bars = pd.DataFrame({"high": 10.0, "low": 10.0, "close": 10.0},
                        index=pd.bdate_range("2000-01-03", periods=15))
    signal = e35.supertrend(bars)
    assert signal["line"].iloc[:9].isna().all()
    assert signal["direction"].iloc[:9].eq(0).all()
    assert signal["atr"].iloc[9:].eq(0).all()
    assert signal["direction"].iloc[9:].eq(-1).all()


@pytest.mark.parametrize("bad", ["duplicate", "nan", "reversed", "outside", "zero"])
def test_invalid_ohlc_is_rejected_without_filling(bars, bad):
    changed = bars.iloc[:20].copy()
    if bad == "duplicate":
        changed = pd.concat([changed, changed.iloc[:1]])
    elif bad == "nan":
        changed.iloc[3, 0] = np.nan
    elif bad == "reversed":
        changed.iloc[3, 0] = 1
    elif bad == "outside":
        changed.iloc[3, 2] = changed.iloc[3, 0] + 1
    else:
        changed.iloc[3, 1] = 0
    with pytest.raises(ValueError, match="OHLC|bars"):
        e35.supertrend(changed)


def test_alignment_requires_real_matching_bars_for_every_market_date(market, bars):
    assert e35.validate_alignment(market, bars)["max_close_difference"] == 0
    with pytest.raises(ValueError, match="missing"):
        e35.validate_alignment(market, bars.iloc[1:])
    changed = bars.copy()
    changed["close"] += 0.02
    with pytest.raises(ValueError, match="disagrees"):
        e35.validate_alignment(market, changed)


@pytest.mark.parametrize("horizon", e35.e32.HORIZONS)
def test_future_bar_and_market_removal_or_change_leaves_forecast_and_selection_identical(
    market, bars, horizon,
):
    origin = pd.Timestamp("1996-06-28")
    full = e35.forecast_at(market, bars, horizon, origin,
                           e35.e32.target_history(market, horizon), e35.supertrend(bars))
    prefix = e35.forecast_at(market.loc[:origin], bars.loc[:origin], horizon, origin)
    changed_market, changed_bars = market.copy(), bars.copy()
    changed_market.loc[changed_market.index > origin] *= 10
    changed_bars.loc[changed_bars.index > origin] *= 10
    altered = e35.forecast_at(changed_market, changed_bars, horizon, origin,
                              e35.e32.target_history(changed_market, horizon))
    pd.testing.assert_frame_equal(full.vectors, prefix.vectors, check_exact=True)
    pd.testing.assert_frame_equal(full.vectors, altered.vectors, check_exact=True)
    assert full.audit == prefix.audit == altered.audit
    assert all(full.audit[k] == 0 for k in full.audit if k.endswith("violations"))


def test_bars_after_buffered_cutoff_including_origin_cannot_change_forecast(market, bars):
    origin = pd.Timestamp("1996-06-28")
    _, cutoff = e35.e32.state_at(market, origin)
    original = e35.forecast_at(market, bars, 60, origin)
    changed = bars.copy()
    changed.loc[changed.index > cutoff] *= 100
    actual = e35.forecast_at(market, changed, 60, origin)
    pd.testing.assert_frame_equal(original.vectors, actual.vectors, check_exact=True)
    assert original.audit == actual.audit
    assert original.audit["indicator_bar_date"] == str(cutoff.date())


@pytest.mark.parametrize("same_count", [2, 3])
def test_regime_selection_uses_each_training_origins_buffered_state_and_fixed_fallback(
    market, bars, same_count,
):
    origin = pd.Timestamp("1996-06-28")
    history = e35.e32.target_history(market, 60)
    for i, target in enumerate(e35.e32.TARGETS):
        history[target] = np.arange(len(history), dtype=float) * (i + 1)
    train = e35.e32.completed_training(history, origin)
    signal = e35.supertrend(bars)
    signal["direction"] = -1
    chosen = train.iloc[:same_count]
    signal.loc[chosen["state_observed"], "direction"] = 1
    _, cutoff = e35.e32.state_at(market, origin)
    signal.loc[cutoff, "direction"] = 1
    actual = e35.forecast_at(market, bars, 60, origin, history, signal)
    expected = train if same_count < 3 else chosen
    pd.testing.assert_series_equal(actual.vectors.loc[e35.NAME], expected[e35.e32.TARGETS].mean(),
                                   check_names=False)
    assert actual.audit["same_regime_count"] == same_count
    assert actual.audit["fallback"] == (same_count < 3)
    assert actual.audit["selected_count"] == len(expected)


def test_unavailable_finished_target_cannot_enter_regime_mean(market, bars):
    origin = pd.Timestamp("1996-06-28")
    history = e35.e32.target_history(market, 60)
    waiting = (history["window_end"] <= origin) & (history["available_on"] > origin)
    assert waiting.any()
    original = e35.forecast_at(market, bars, 60, origin, history)
    changed = history.copy()
    changed.loc[waiting, e35.e32.TARGETS] = 1e12
    actual = e35.forecast_at(market, bars, 60, origin, changed)
    pd.testing.assert_frame_equal(original.vectors, actual.vectors, check_exact=True)
    assert original.audit == actual.audit


def test_summary_preserves_pairing_target_units_and_regime_counts():
    records = []
    errors = dict(zip(e35.CANDIDATES, [10.0, 9.0, 7.0], strict=True))
    for origin in pd.date_range("2000-01-31", periods=18, freq="ME"):
        for name in e35.CANDIDATES:
            records.append({"origin": origin, "horizon": 60, "candidate": name,
                            "loss_error_usd_m": errors[name], "cet1_error_pp": 0.25,
                            "future_truncation_identical": True, "fallback": False,
                            "regime_direction": 1, "e32_actual_loss_difference_usd": 0.0,
                            "e32_actual_cet1_difference_pp": 0.0,
                            **{f"forecast_{t}": 0.125 for t in e35.e32.TARGETS},
                            **{f"realised_{t}": 0.0 for t in e35.e32.TARGETS}})
    summary, years = e35.summarise(pd.DataFrame(records))
    own = summary["60"]["candidates"][e35.NAME]
    assert own["ratio_to_e32_mean"] == 0.7
    assert own["cet1_mae_pp"] == 0.25
    assert own["target_mae"] == dict.fromkeys(e35.e32.TARGETS, 0.125)
    assert own["paired"]["e32_mean"]["difference_95ci_usd_m"] == [-3.0, -3.0]
    assert summary["60"]["up_origins"] == 18
    assert summary["60"]["down_origins"] == summary["60"]["fallbacks"] == 0
    assert len(years) == 6
    assert e35.decision(summary["60"])["passed"]
    with pytest.raises(ValueError, match="identical origins"):
        e35.summarise(pd.DataFrame(records).iloc[1:])


@pytest.mark.parametrize("failing", ["ratio", "ci", "no_change"])
def test_gain_requires_each_fixed_condition(failing):
    own = {"ratio_to_e32_mean": 0.8, "loss_mae_usd_m": 8.0,
           "paired": {"e32_mean": {"difference_95ci_usd_m": [-2, -1]}}}
    control = {"loss_mae_usd_m": 9.0}
    if failing == "ratio":
        own["ratio_to_e32_mean"] = 0.851
    elif failing == "ci":
        own["paired"]["e32_mean"]["difference_95ci_usd_m"][1] = 0
    else:
        control["loss_mae_usd_m"] = 7.9
    assert not e35.decision({"candidates": {e35.NAME: own, "e32_no_change": control}})["passed"]


@pytest.mark.parametrize("suffix", [".json", "_predictions.csv", "_per_year.csv"])
def test_existing_artifact_refuses_before_data_loading(tmp_path, monkeypatch, suffix):
    output = tmp_path / "result.json"
    existing = tmp_path / f"result{suffix}"
    existing.write_text("unchanged")

    def forbidden(*args, **kwargs):
        pytest.fail("Refusal must precede input loading")

    monkeypatch.setattr(e35.e33, "load_frozen_controls", forbidden)
    with pytest.raises(FileExistsError, match="Refusing"):
        e35.run(output)
    assert existing.read_text() == "unchanged"
