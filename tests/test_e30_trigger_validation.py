import json

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e30_trigger_validation as e30


def frame():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "event_day": pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 2),
            "vol20": [0.1, 0.3, 0.3, 0.2, 0.4, 0.05],
            "n_prev_24h": [0, 1, 1, 0, 1, 0],
            "abs_car01": [0.05, 0.06, 0.01, 0.02, 0.07, 0.01],
            "z": [3.0, 4.0, 1.0, 2.0, 5.0, 0.0],
        }
    )


def test_metrics_precision_recall_lift_and_undefined_ratios():
    row = e30.metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 0])
    assert row["n"] == 6 and row["fired"] == 3 and row["true_positives"] == 2
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["recall"] == pytest.approx(2 / 3)
    assert row["base_rate"] == 0.5
    assert row["lift"] == pytest.approx(4 / 3)
    assert e30.metrics([0, 0], [1, 0])["precision"] is None
    assert e30.metrics([1, 0], [0, 0])["recall"] is None
    assert e30.metrics([1, 0], [0, 0])["lift"] is None


def test_thresholds_use_fit_predictions_from_one_frozen_model(monkeypatch):
    fit = pd.DataFrame({"move": [1.0, 2.0, 3.0, 4.0], "year": [2009, 2012, 2015, 2018]})
    sets = {"test": pd.DataFrame({"move": [100.0, 200.0], "year": [2019, 2020]})}
    calls = []

    def fake(name, tr, te, cols):
        calls.append(name)
        pd.testing.assert_frame_equal(tr, fit)
        assert cols == ["move"]
        return te["move"].to_numpy()

    monkeypatch.setattr(e30, "fit_predict", fake)
    thresholds, predictions = e30.predict_sets(fit, sets, ["move"])
    assert calls == ["d_clipped_no_vol"]
    assert thresholds == {
        "p95": np.percentile(fit["move"], 95),
        "p99": np.percentile(fit["move"], 99),
    }
    np.testing.assert_array_equal(predictions["test"], [100, 200])


def test_matched_fire_counts_corroboration_formula_and_seeded_random(monkeypatch):
    rows = frame()
    monkeypatch.setattr(e30, "formula", lambda t: np.array([6, 7, 8, 9, 1, 1]))
    move = np.array([0.1, 0.9, 0.95, 0.2, 0.99, 0.3])
    thresholds = {"p95": 0.3, "p99": 0.95}
    triggers = e30.make_triggers(rows, move, thresholds)
    for level, k in [(95, 4), (99, 2)]:
        assert triggers[f"v3_p{level}"].sum() == k
        assert triggers[f"volatility_p{level}"].sum() == k
        assert triggers[f"random_p{level}"].sum() == k
    np.testing.assert_array_equal(triggers["volatility_p99"], [0, 1, 0, 0, 1, 0])
    np.testing.assert_array_equal(triggers["v3_p99_corroborated"], [0, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(triggers["formula_ge_7"], [0, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(triggers["formula_ge_8"], [0, 0, 1, 1, 0, 0])
    again = e30.make_triggers(rows, move, thresholds)
    for key in triggers:
        np.testing.assert_array_equal(triggers[key], again[key])
    assert e30.top_k(rows["vol20"].to_numpy(), 0).sum() == 0
    assert e30.top_k(rows["vol20"].to_numpy(), len(rows)).all()


def test_cluster_bootstrap_is_deterministic_and_resamples_whole_stock_days():
    rows = frame()
    triggers = {
        "v3_p99": np.array([1, 0, 0, 0, 1, 0], dtype=bool),
        "volatility_p99": np.array([0, 0, 1, 0, 1, 0], dtype=bool),
    }
    labels = {e30.PRIMARY_LABEL: rows["abs_car01"].to_numpy() >= 0.05}
    result = e30.cluster_bootstrap(rows, triggers, labels)
    assert result == e30.cluster_bootstrap(rows, triggers, labels)
    assert result["n_clusters"] == 3  # AAA and BBB on the same day are separate clusters

    # Independent, explicit resampling of all rows in the sampled clusters.
    rng = np.random.default_rng(0)
    p, r, difference = [], [], []
    for _ in range(1000):
        draw = rng.integers(3, size=3)
        indices = np.concatenate([np.array([2 * c, 2 * c + 1]) for c in draw])
        a = e30.metrics(triggers["v3_p99"][indices], labels[e30.PRIMARY_LABEL][indices])
        b = e30.metrics(triggers["volatility_p99"][indices], labels[e30.PRIMARY_LABEL][indices])
        if a["precision"] is not None:
            p.append(a["precision"])
        if a["recall"] is not None:
            r.append(a["recall"])
        if a["precision"] is not None and b["precision"] is not None:
            difference.append(a["precision"] - b["precision"])
    intervals = result["intervals"]["v3_p99"][e30.PRIMARY_LABEL]
    np.testing.assert_allclose(intervals["precision_ci95"], np.quantile(p, [0.025, 0.975]))
    np.testing.assert_allclose(intervals["recall_ci95"], np.quantile(r, [0.025, 0.975]))
    delta = result["differences"][e30.PRIMARY_LABEL]
    np.testing.assert_allclose(delta["ci95"], np.quantile(difference, [0.025, 0.975]))
    assert delta["valid_resamples"] == len(difference)


def test_evaluate_reports_both_labels_intervals_and_random_expectation(monkeypatch):
    monkeypatch.setattr(e30, "formula", lambda t: np.full(len(t), 7))
    out = e30.evaluate(frame(), np.array([0.1, 0.2, 0.9, 0.8, 1.0, 0.3]), {"p95": 0.8, "p99": 0.9})
    for trigger, labels in out["triggers"].items():
        assert set(labels) == {e30.PRIMARY_LABEL, "z_ge_3"}
        for row in labels.values():
            assert row["n"] == 6
            assert len(row["precision_ci95"]) == len(row["recall_ci95"]) == 2
            if trigger.startswith("random_"):
                assert row["expected_precision"] == row["base_rate"] == 0.5
    assert out["triggers"]["formula_ge_8"][e30.PRIMARY_LABEL]["precision_ci95"] == [None, None]
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize(
    "delta,lower,adopt",
    [
        (0.1, 0.01, True),
        (0.1, 0.0, False),
        (0.1, -0.1, False),
        (-0.1, -0.2, False),
        (None, None, False),
    ],
)
def test_adoption_requires_positive_paired_interval_in_both_fresh_years(delta, lower, adopt):
    def year(d, lo):
        return {
            "v3_p99_minus_volatility_precision": {
                e30.PRIMARY_LABEL: {"precision_difference": d, "ci95": [lo, 0.2]}
            }
        }

    sets = {"F_fresh_2019": year(0.1, 0.01), "F_fresh_2020": year(delta, lower)}
    result = e30.decision(sets)
    assert result["adopt"] is adopt
    assert result["line"] == (e30.ADOPT if adopt else e30.KEEP)


def test_run_refuses_existing_result_before_loading_data(tmp_path, monkeypatch):
    result = tmp_path / "e30.json"
    result.write_text('{"already": "run"}')
    monkeypatch.setattr(e30, "RESULTS", result)

    def fail():
        pytest.fail("must refuse before reading any data")

    monkeypatch.setattr(e30, "build_dataset", fail)
    with pytest.raises(FileExistsError, match="run once"):
        e30.run()
    assert result.read_text() == '{"already": "run"}'
