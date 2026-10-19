"""E32 contract checks on fake data only; never run the real one-shot replay."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e32_forecast_replay as e32
from riskengine.module_b import outlook
from riskengine.module_b.shock import Market


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(32)
    dates = pd.bdate_range("1993-01-01", "1997-12-31")
    n = len(dates)
    baa = 2 + np.sin(np.arange(n) / 43) * 0.2
    return pd.DataFrame({
        "baa": baa, "aaa": 0.5 * baa, "vix": 20 + 5 * baa + rng.normal(size=n),
        "spx": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "usdinr": 30 * np.exp(np.cumsum(rng.normal(0, 0.001, n))),
        "ust10y": 4 + np.sin(np.arange(n) / 80),
    }, index=dates)


@pytest.mark.parametrize("horizon", e32.HORIZONS)
def test_future_removal_and_future_perturbation_leave_forecast_identical(market, horizon):
    origin = pd.Timestamp("1996-06-28")
    full = e32.forecast_at(market, origin, horizon, e32.target_history(market, horizon))
    cropped = e32.forecast_at(market.loc[:origin], origin, horizon)
    changed = market.copy()
    changed.loc[changed.index > origin] *= 100
    perturbed = e32.forecast_at(changed, origin, horizon, e32.target_history(changed, horizon))
    pd.testing.assert_frame_equal(full.vectors, cropped.vectors, check_exact=True)
    pd.testing.assert_frame_equal(full.vectors, perturbed.vectors, check_exact=True)
    assert full.audit == cropped.audit == perturbed.audit


@pytest.mark.parametrize("horizon", e32.HORIZONS)
def test_training_targets_finished_and_available_by_origin(market, horizon):
    hist = e32.target_history(market, horizon)
    origin = pd.Timestamp("1996-06-28")
    train = e32.completed_training(hist, origin)
    expected = hist[(hist.index < origin) & (hist["window_end"] <= origin)
                    & (hist["available_on"] <= origin)]
    pd.testing.assert_frame_equal(train, expected)
    assert len(train) >= 3
    assert (train["window_end"] <= origin).all()
    assert (train["available_on"] <= origin).all()
    # A finished target still inside the availability buffer must remain excluded.
    recent = hist[(hist["window_end"] <= origin) & (hist["available_on"] > origin)]
    assert len(recent) > 0
    assert not train.index.isin(recent.index).any()
    for row in hist.itertuples():
        assert (market.index.get_loc(row.available_on) - market.index.get_loc(row.window_end)
                == e32.AVAILABILITY_SESSIONS)
        assert market.index.get_loc(row.window_end) - market.index.get_loc(row.Index) == horizon


def test_state_uses_buffered_observation_and_trailing_twenty_sessions(market):
    origin = pd.Timestamp("1996-06-28")
    state, cutoff = e32.state_at(market, origin)
    j = market.index.get_loc(origin) - 10
    assert cutoff == market.index[j]
    assert state["baa"] == market["baa"].iloc[j]
    assert state["vix"] == market["vix"].iloc[j]
    assert state["baa_20d"] == market["baa"].iloc[j] - market["baa"].iloc[j - 20]
    altered = market.copy()
    altered.loc[altered.index > cutoff] *= 10
    pd.testing.assert_series_equal(state, e32.state_at(altered, origin)[0])


def test_fixed_weights_and_mean_and_ols_reuse(market):
    mean, model = pd.Series([2.0, 10.0]), pd.Series([6.0, -2.0])
    assert e32.combine(mean, model).tolist() == [3.0, 7.0]
    origin = pd.Timestamp("1996-06-28")
    forecast = e32.forecast_at(market, origin, 60)
    train = e32.completed_training(e32.target_history(market.loc[:origin], 60), origin)
    state, _ = e32.state_at(market, origin)
    expected_model, expected_mean = outlook.forecast(train, state, e32.TARGETS)
    np.testing.assert_array_equal(forecast.vectors.loc["historical_mean"], expected_mean)
    np.testing.assert_array_equal(forecast.vectors.loc["state_model"], expected_model)
    np.testing.assert_array_equal(forecast.vectors.loc["combination"],
                                  0.75 * expected_mean + 0.25 * expected_model)


def test_analog_training_scales_and_earlier_tie_break(market):
    origin = pd.Timestamp("1996-06-28")
    hist = e32.target_history(market, 60)
    # Every state has the same distance. The earlier three origins win the tie.
    hist.loc[:, outlook.STATE] = 1.0
    train = e32.completed_training(hist, origin)
    forecast = e32.forecast_at(market, origin, 60, hist)
    assert forecast.audit["neighbours"] == [str(d.date()) for d in train.index[:3]]
    np.testing.assert_array_equal(forecast.vectors.loc["clean_analog"],
                                  train.iloc[:3][e32.TARGETS].mean())


def test_too_little_training_is_an_error(market):
    with pytest.raises(ValueError, match="fewer than 3"):
        e32.forecast_at(market, pd.Timestamp("1993-04-30"), 60)


def test_origins_are_monthly_and_share_the_sixty_day_cutoff(market):
    origins = e32.eligible_origins(market)
    assert origins[0] == pd.Timestamp("1995-01-31")
    assert (market.index.get_indexer(origins) + 60 < len(market)).all()
    all_months = outlook.month_ends(market.loc["1995":])
    assert origins.equals(all_months[all_months <= origins[-1]])
    # Origins are selected by calendar/target availability, not future outcome size.
    changed = market.copy()
    changed["baa"] = np.arange(len(changed)) * 100.0
    assert origins.equals(e32.eligible_origins(changed))


def test_fixed_shock_is_equity_down_ten_percent_and_rates_up_two_points():
    vector = e32.fixed_shock()
    assert np.expm1(vector["spx_ret"]) == pytest.approx(-0.1)
    assert vector["d_ust"] == 2.0
    assert (vector[["d_baa", "d_aaa", "d_vix", "fx_ret"]] == 0).all()


def test_bootstrap_is_paired_deterministic_and_uses_six_month_blocks():
    indices = e32.bootstrap_indices(18)
    np.testing.assert_array_equal(indices, e32.bootstrap_indices(18))
    assert indices.shape == (1000, 18)
    assert indices.min() >= 0 and indices.max() < 18
    assert (np.diff(indices.reshape(1000, 3, 6), axis=2) == 1).all()
    base = np.arange(18, dtype=float)
    difference = base + 2 - base
    assert e32.paired_interval(difference, indices) == [2.0, 2.0]
    assert e32.paired_interval(-difference, indices) == [-2.0, -2.0]
    with pytest.raises(ValueError, match="six-month"):
        e32.bootstrap_indices(5)


def test_dm_horizon_lags_and_zero_difference():
    assert e32.dm(np.zeros(24), 60) == {
        "statistic": 0.0, "p_two_sided": 1.0, "newey_west_lag": 3}
    assert e32.dm(np.arange(24, dtype=float) % 7, 20)["newey_west_lag"] == 1


@pytest.mark.parametrize(("ratio", "interval", "wins"), [
    (0.85, [-5.0, -0.01], True), (0.851, [-5.0, -0.01], False),
    (0.8, [-5.0, 0.0], False), (0.8, [-5.0, 1.0], False),
])
def test_combination_requires_both_fixed_conditions(ratio, interval, wins):
    primary = {"candidates": {"combination": {
        "ratio_to_mean": ratio, "paired_difference_95ci_usd_m": interval}}}
    decision = e32.combination_decision(primary)
    assert decision == ("gain over the historical mean" if wins
                        else "no gain over the historical mean")


@pytest.mark.parametrize("suffix", [".json", "_predictions.csv", "_per_year.csv"])
def test_existing_output_refuses_before_any_computation(tmp_path, monkeypatch, suffix):
    output = tmp_path / "result.json"
    existing = tmp_path / f"result{suffix}"
    existing.write_text("unchanged")

    def forbidden(*args, **kwargs):
        pytest.fail("A second run must refuse before reading data or computing")

    monkeypatch.setattr(e32.pd, "read_csv", forbidden)
    monkeypatch.setattr(e32, "replay", forbidden)
    with pytest.raises(FileExistsError, match="Refusing to rerun"):
        e32.run(output)
    assert existing.read_text() == "unchanged"


def test_valuation_reuses_e22_and_maps_units_and_z(monkeypatch):
    base = Market("base", 0.5, 1.5, 0, 0, 0, 70.0, 0, 1.92, 0)
    vector = e32.fixed_shock()
    calls = []

    class Result:
        loss_usd, cet1_after = 123.0, 0.12

    def fake_revalue(book, obligor, notches, mk, *, rates):
        calls.append((obligor, notches, mk, rates))
        return Result()

    monkeypatch.setattr(e32, "revalue", fake_revalue)
    assert e32.value_vector(pd.DataFrame(), base, vector, 0.2) == (123.0, 0.12)
    obligor, notches, mk, rates = calls[0]
    assert (obligor, notches, rates) == (None, 0, "moved")
    assert mk.d_rate == 2.0 and mk.rate0 == 1.92
    assert mk.z == pytest.approx(-np.log(0.9) / 0.2)
    assert replace(mk, episode="base", z=0, d_rate=0) == base


def test_prepare_market_never_backfills_and_rejects_duplicate_dates(market):
    incomplete = market.copy()
    incomplete.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        e32.prepare_market(incomplete)
    missing = market.copy()
    missing.iloc[1, 0] = np.nan
    assert e32.prepare_market(missing).iloc[1, 0] == missing.iloc[0, 0]
    with pytest.raises(ValueError, match="unique"):
        e32.prepare_market(pd.concat([market, market.iloc[:1]]))


def test_summary_uses_paired_origins_units_and_all_six_targets():
    rows = []
    errors = dict(zip(e32.CANDIDATES, [10.0, 12.0, 30.0, 9.0, 11.0, 8.0], strict=True))
    for origin in pd.date_range("2000-01-31", periods=18, freq="ME"):
        for candidate in e32.CANDIDATES:
            rows.append({
                "origin": origin, "horizon": 60, "candidate": candidate,
                "window_end": origin + pd.Timedelta(days=90),
                "training_count": 50, "future_truncation_identical": True,
                "target_end_violations": 0, "loss_error_usd_m": errors[candidate],
                "cet1_error_pp": 0.5,
                **{f"forecast_{key}": 3.0 for key in e32.TARGETS},
                **{f"realised_{key}": 1.0 for key in e32.TARGETS},
            })
    summary, years = e32.summarise(pd.DataFrame(rows))
    assert summary["60"]["origins"] == 18
    combo = summary["60"]["candidates"]["combination"]
    assert combo["loss_mae_usd_m"] == 8.0 and combo["ratio_to_mean"] == 0.8
    assert combo["paired_difference_95ci_usd_m"] == [-2.0, -2.0]
    assert combo["cet1_mae_pp"] == 0.5
    assert combo["target_mae"] == dict.fromkeys(e32.TARGETS, 2.0)
    assert len(years) == 12
    assert set(years["year"]) == {2000, 2001}
    assert e32.combination_decision(summary["60"]) == "gain over the historical mean"
    # A missing candidate-origin pair must not silently change the benchmark sample.
    with pytest.raises(ValueError, match="identical scored origins"):
        e32.summarise(pd.DataFrame(rows).iloc[1:])
