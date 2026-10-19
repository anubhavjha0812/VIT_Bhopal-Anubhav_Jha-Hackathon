"""E33 contract checks on fake markets/targets; never run the real experiment."""

from functools import cache

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e33_direct_loss as e33


@pytest.fixture(scope="module")
def market():
    dates = pd.bdate_range("1993-01-01", "1997-12-31")
    rng = np.random.default_rng(33)
    x = np.arange(len(dates))
    return pd.DataFrame({
        "baa": 2 + 0.3 * np.sin(x / 50), "aaa": 1 + 0.1 * np.sin(x / 50),
        "vix": 20 + 3 * np.sin(x / 70),
        "spx": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
        "usdinr": 40 + x / 5000, "ust10y": 4 + 0.3 * np.cos(x / 80),
    }, index=dates)


@pytest.fixture
def price():
    @cache
    def fake(vector, sigma):
        # Deliberately nonlinear repricing, with independent CET1 dependence on FX.
        return (vector[0] ** 2 + vector[5] - vector[3] / sigma) * 1e6, 0.13 - vector[4]

    return fake


@pytest.mark.parametrize("horizon", e33.e32.HORIZONS)
def test_rebuild_after_future_removal_or_change_is_identical(market, price, horizon):
    origin = pd.Timestamp("1996-06-28")
    hist = e33.loss_history(market, horizon, price)
    full = e33.forecast_at(market, hist, origin)
    prefix = market.loc[:origin]
    cropped = e33.forecast_at(prefix, e33.loss_history(prefix, horizon, price), origin)
    changed = market.copy()
    changed.loc[changed.index > origin] *= 10
    altered = e33.forecast_at(changed, e33.loss_history(changed, horizon, price), origin)
    pd.testing.assert_frame_equal(full.values, cropped.values, check_exact=True)
    pd.testing.assert_frame_equal(full.values, altered.values, check_exact=True)
    assert full.audit == cropped.audit == altered.audit
    assert full.audit["latest_training_available"] <= str(origin.date())
    assert full.audit["latest_training_end"] <= str(origin.date())
    assert all(full.audit[k] == 0 for k in full.audit if k.endswith("violations"))


def test_training_outcome_waits_for_its_full_window_and_release_buffer(market, price):
    hist = e33.loss_history(market, 60, price)
    origin = pd.Timestamp("1996-06-28")
    train = e33.e32.completed_training(hist, origin)
    finished_but_unavailable = hist[(hist["window_end"] <= origin)
                                   & (hist["available_on"] > origin)]
    assert len(finished_but_unavailable) > 0
    assert not train.index.isin(finished_but_unavailable.index).any()
    row = hist.iloc[0]
    sigma = e33.equity_scale(market, row["state_observed"])
    vector = tuple(float(row[t]) for t in e33.e32.TARGETS)
    expected_loss, expected_cet1 = price(vector, sigma)
    assert row[e33.LOSS] == expected_loss / 1e6
    assert row[e33.CET1] == 100 * expected_cet1


def test_median_heads_are_robust_to_large_outlier_and_constant_states():
    train = pd.DataFrame({"baa": 2.0, "vix": 20.0, "baa_20d": 0.0,
                          e33.LOSS: [0, 0, 0, 0, 100], e33.CET1: [13, 13, 13, 13, 90]})
    state = pd.Series({"baa": 2.0, "vix": 20.0, "baa_20d": 0.0})
    values, audit = e33.fit_heads(train, state)
    assert values[e33.LOSS].tolist() == [0.0, 0.0]
    assert values[e33.CET1].tolist() == [13.0, 13.0]
    assert audit["feature_scale"] == dict.fromkeys(e33.outlook.STATE, 1.0)
    assert audit["target_models"][e33.LOSS]["scaling_denominator"] == 1.0


def test_quantile_uses_conditioning_state_and_allows_predicted_gains():
    x = np.linspace(-2, 2, 81)
    train = pd.DataFrame({"baa": x, "vix": 20.0, "baa_20d": 0.0,
                          e33.LOSS: 10 * x, e33.CET1: 13 - x})
    low = pd.Series({"baa": -1.5, "vix": 20.0, "baa_20d": 0.0})
    high = low.copy()
    high["baa"] = 1.5
    a, _ = e33.fit_heads(train, low)
    b, _ = e33.fit_heads(train, high)
    assert a.loc["direct_quantile", e33.LOSS] < -10
    assert b.loc["direct_quantile", e33.LOSS] > 10
    assert a.loc["direct_quantile", e33.CET1] > b.loc["direct_quantile", e33.CET1]
    assert a.loc["direct_median", e33.LOSS] == b.loc["direct_median", e33.LOSS]


def test_target_units_do_not_change_scaled_penalty_or_forecast():
    x = np.linspace(-2, 2, 81)
    train = pd.DataFrame({"baa": x, "vix": 20.0, "baa_20d": 0.0,
                          e33.LOSS: 10 * x, e33.CET1: 13 - x})
    state = pd.Series({"baa": 1.25, "vix": 20.0, "baa_20d": 0.0})
    expected, _ = e33.fit_heads(train, state)
    scaled = train.copy()
    scaled[e33.LOSS] *= 1000
    actual, _ = e33.fit_heads(scaled, state)
    np.testing.assert_allclose(actual[e33.LOSS], expected[e33.LOSS] * 1000, atol=1e-8)
    np.testing.assert_array_equal(actual[e33.CET1], expected[e33.CET1])


def test_pricing_cache_keys_include_every_move_and_equity_scale(monkeypatch):
    calls = []

    def value(book, base, vector, sigma):
        calls.append((tuple(vector), sigma))
        return float(vector.sum()), sigma

    monkeypatch.setattr(e33.e32, "value_vector", value)
    price = e33.make_pricer(pd.DataFrame(), None)
    one = tuple([0.0] * 6)
    two = (*one[:5], 1.0)
    assert price(one, 0.2) == price(one, 0.2)
    price(one, 0.3)
    price(two, 0.2)
    assert len(calls) == 3


def test_summary_preserves_pairing_units_and_years():
    rows = []
    errors = dict(zip(e33.CANDIDATES, [10.0, 9.0, 8.0, 7.0], strict=True))
    for origin in pd.date_range("2000-01-31", periods=18, freq="ME"):
        for candidate in e33.CANDIDATES:
            rows.append({"origin": origin, "horizon": 60, "candidate": candidate,
                         "loss_error_usd_m": errors[candidate], "cet1_error_pp": 0.25,
                         "future_truncation_identical": True,
                         "e32_actual_loss_difference_usd": 0.0,
                         "e32_actual_cet1_difference_pp": 0.0})
    summary, years = e33.summarise(pd.DataFrame(rows))
    quantile = summary["60"]["candidates"]["direct_quantile"]
    assert quantile["ratio_to_e32_mean"] == 0.7
    assert quantile["cet1_mae_pp"] == 0.25
    assert quantile["paired"]["e32_mean"]["difference_95ci_usd_m"] == [-3.0, -3.0]
    assert quantile["paired"]["direct_median"]["difference_95ci_usd_m"] == [-1.0, -1.0]
    assert len(years) == 8
    assert e33.decision(summary["60"])["passed"]
    with pytest.raises(ValueError, match="identical origins"):
        e33.summarise(pd.DataFrame(rows).iloc[1:])


@pytest.mark.parametrize("failing", ["ratio", "mean_ci", "median_ci", "no_change"])
def test_gain_requires_all_predeclared_conditions(failing):
    quantile = {"ratio_to_e32_mean": 0.8, "loss_mae_usd_m": 8.0,
                "paired": {"e32_mean": {"difference_95ci_usd_m": [-2, -1]},
                           "direct_median": {"difference_95ci_usd_m": [-1, -0.1]}}}
    baseline = {"loss_mae_usd_m": 9.0}
    if failing == "ratio":
        quantile["ratio_to_e32_mean"] = 0.851
    elif failing == "mean_ci":
        quantile["paired"]["e32_mean"]["difference_95ci_usd_m"][1] = 0
    elif failing == "median_ci":
        quantile["paired"]["direct_median"]["difference_95ci_usd_m"][1] = 0
    else:
        baseline["loss_mae_usd_m"] = 7.9
    result = e33.decision({"candidates": {"direct_quantile": quantile,
                                          "e32_no_change": baseline}})
    assert not result["passed"]


@pytest.mark.parametrize("suffix", [".json", "_predictions.csv", "_training.csv", "_per_year.csv"])
def test_existing_result_refuses_before_loading_data(tmp_path, monkeypatch, suffix):
    output = tmp_path / "result.json"
    existing = tmp_path / f"result{suffix}"
    existing.write_text("unchanged")

    def forbidden():
        pytest.fail("Refusal must happen before reading or scoring")

    monkeypatch.setattr(e33, "load_frozen_controls", forbidden)
    with pytest.raises(FileExistsError, match="Refusing"):
        e33.run(output)
    assert existing.read_text() == "unchanged"
