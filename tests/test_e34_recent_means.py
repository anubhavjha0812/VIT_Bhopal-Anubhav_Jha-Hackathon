"""E34 checks on small synthetic markets; no real replay or parameter search."""

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e34_recent_means as e34


@pytest.fixture(scope="module")
def market():
    dates = pd.bdate_range("1993-01-01", "1997-12-31")
    rng = np.random.default_rng(34)
    x = np.arange(len(dates))
    return pd.DataFrame({
        "baa": 2 + 0.3 * np.sin(x / 50), "aaa": 1 + 0.1 * np.sin(x / 50),
        "vix": 20 + 3 * np.sin(x / 70),
        "spx": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
        "usdinr": 40 + x / 5000, "ust10y": 4 + 0.3 * np.cos(x / 80),
    }, index=dates)


def fake_history(n):
    dates = pd.date_range("1990-01-31", periods=n, freq="ME")
    history = pd.DataFrame({t: np.arange(n, dtype=float) for t in e34.e32.TARGETS},
                           index=dates)
    history["window_end"] = dates + pd.Timedelta(days=1)
    history["available_on"] = dates + pd.Timedelta(days=2)
    history["state_observed"] = dates - pd.Timedelta(days=15)
    return history


def test_ewma_half_life_uses_calendar_months_and_normalises():
    dates = pd.DatetimeIndex(["1990-01-31", "1992-01-31", "1992-02-28"])
    weights = e34.ewma_weights(dates, pd.Timestamp("1993-01-29"))
    assert np.all(weights > 0)
    assert weights.sum() == pytest.approx(1)
    assert weights[0] / weights[1] == pytest.approx(0.5)
    assert weights[1] / weights[2] == pytest.approx(2 ** (-1 / 24))
    # Normalised relative weights are unchanged as all observations age together.
    later = e34.ewma_weights(dates, pd.Timestamp("1994-01-31"))
    np.testing.assert_allclose(weights, later, rtol=1e-15)


@pytest.mark.parametrize("dates", [[], ["2000-01-01"]])
def test_ewma_rejects_empty_or_future_observations(dates):
    with pytest.raises(ValueError, match="earlier observations"):
        e34.ewma_weights(pd.DatetimeIndex(dates), pd.Timestamp("1999-01-01"))


@pytest.mark.parametrize("n", [20, 70])
def test_rolling_uses_exactly_last_sixty_or_fixed_warm_up(market, n):
    history = fake_history(n)
    origin = pd.Timestamp("1996-06-28")
    predicted = e34.forecast_at(market, 60, origin, history)
    expected = np.arange(n)[-60:].mean()
    np.testing.assert_array_equal(predicted.vectors.loc["rolling_60m"], [expected] * 6)
    assert predicted.audit["rolling_count"] == min(n, 60)
    assert predicted.audit["rolling_first_origin"] == str(history.index[-min(n, 60)].date())
    expected_ewma = e34.ewma_weights(history.index, origin) @ np.arange(n)
    np.testing.assert_allclose(predicted.vectors.loc["ewma_h24m"], [expected_ewma] * 6)


def test_both_means_respond_to_recent_change_without_looking_at_future(market):
    history = fake_history(70)
    for target in e34.e32.TARGETS:
        history[target] = [0.0] * 60 + [10.0] * 10
    result = e34.forecast_at(market, 60, pd.Timestamp("1996-06-28"), history)
    expanding = history["d_baa"].mean()
    assert expanding < result.vectors.loc["rolling_60m", "d_baa"] < 10
    assert expanding < result.vectors.loc["ewma_h24m", "d_baa"] < 10


@pytest.mark.parametrize("horizon", e34.e32.HORIZONS)
def test_future_removal_and_changes_leave_vectors_weights_and_audits_identical(market, horizon):
    origin = pd.Timestamp("1996-06-28")
    full = e34.forecast_at(market, horizon, origin, e34.e32.target_history(market, horizon))
    prefix = e34.forecast_at(market.loc[:origin], horizon, origin)
    changed = market.copy()
    changed.loc[changed.index > origin] *= 10
    altered = e34.forecast_at(changed, horizon, origin,
                             e34.e32.target_history(changed, horizon))
    pd.testing.assert_frame_equal(full.vectors, prefix.vectors, check_exact=True)
    pd.testing.assert_frame_equal(full.vectors, altered.vectors, check_exact=True)
    assert full.audit == prefix.audit == altered.audit
    assert all(full.audit[k] == 0 for k in full.audit if k.endswith("violations"))


def test_completed_but_not_yet_available_target_cannot_affect_either_mean(market):
    origin = pd.Timestamp("1996-06-28")
    history = e34.e32.target_history(market, 60)
    waiting = (history["window_end"] <= origin) & (history["available_on"] > origin)
    assert waiting.any()
    original = e34.forecast_at(market, 60, origin, history)
    changed = history.copy()
    changed.loc[waiting, e34.e32.TARGETS] = 1e12
    actual = e34.forecast_at(market, 60, origin, changed)
    pd.testing.assert_frame_equal(original.vectors, actual.vectors, check_exact=True)
    assert original.audit == actual.audit


def test_summary_pairs_origins_and_preserves_target_and_cet1_units():
    rows = []
    errors = dict(zip(e34.CANDIDATES, [10.0, 9.0, 8.0, 7.0], strict=True))
    for origin in pd.date_range("2000-01-31", periods=18, freq="ME"):
        for candidate in e34.CANDIDATES:
            rows.append({"origin": origin, "horizon": 60, "candidate": candidate,
                         "loss_error_usd_m": errors[candidate], "cet1_error_pp": 0.25,
                         "future_truncation_identical": True,
                         "e32_actual_loss_difference_usd": 0.0,
                         "e32_actual_cet1_difference_pp": 0.0,
                         **{f"forecast_{t}": 0.125 for t in e34.e32.TARGETS},
                         **{f"realised_{t}": 0.0 for t in e34.e32.TARGETS}})
    summary, years = e34.summarise(pd.DataFrame(rows))
    ewma = summary["60"]["candidates"]["ewma_h24m"]
    assert ewma["ratio_to_e32_mean"] == 0.7
    assert ewma["cet1_mae_pp"] == 0.25
    assert ewma["target_mae"] == dict.fromkeys(e34.e32.TARGETS, 0.125)
    assert ewma["paired"]["e32_mean"]["difference_95ci_usd_m"] == [-3.0, -3.0]
    assert ewma["paired"]["e32_no_change"]["difference_95ci_usd_m"] == [-2.0, -2.0]
    assert len(years) == 8
    assert e34.decision(summary["60"])["passed"]
    with pytest.raises(ValueError, match="identical origins"):
        e34.summarise(pd.DataFrame(rows).iloc[1:])


@pytest.mark.parametrize("failing", ["ratio", "mean_ci", "no_change"])
def test_gain_requires_each_predeclared_condition(failing):
    ewma = {"ratio_to_e32_mean": 0.8, "loss_mae_usd_m": 8.0,
            "paired": {"e32_mean": {"difference_95ci_usd_m": [-2, -1]}}}
    baseline = {"loss_mae_usd_m": 9.0}
    if failing == "ratio":
        ewma["ratio_to_e32_mean"] = 0.851
    elif failing == "mean_ci":
        ewma["paired"]["e32_mean"]["difference_95ci_usd_m"][1] = 0
    else:
        baseline["loss_mae_usd_m"] = 7.9
    assert not e34.decision({"candidates": {"ewma_h24m": ewma,
                                           "e32_no_change": baseline}})["passed"]


@pytest.mark.parametrize("suffix", [".json", "_predictions.csv", "_per_year.csv"])
def test_existing_artifact_refuses_before_loading_data(tmp_path, monkeypatch, suffix):
    output = tmp_path / "result.json"
    existing = tmp_path / f"result{suffix}"
    existing.write_text("unchanged")

    def forbidden():
        pytest.fail("Refusal must precede data loading or scoring")

    monkeypatch.setattr(e34.e33, "load_frozen_controls", forbidden)
    with pytest.raises(FileExistsError, match="Refusing"):
        e34.run(output)
    assert existing.read_text() == "unchanged"
