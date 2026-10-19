"""Invented probability fixtures only; never score the real benchmark in tests."""

import numpy as np
import pytest

from riskengine.eval import e39_calibration as cal


def test_equal_mass_ece_hand_computed():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 0, 0, 1])
    # Sorted bins: [.6 correct, .7 correct], [.8 wrong, .9 correct].
    assert cal.ece(p, y, bins=2) == pytest.approx(0.35)
    table = cal.reliability(p, y, bins=2)
    assert [b["count"] for b in table] == [2, 2]
    assert table[0]["confidence"] == pytest.approx(0.65)


def test_metrics_hand_computed_and_abstain_empty():
    p, y = np.array([[0.8, 0.2], [0.4, 0.6]]), np.array([0, 0])
    metrics = cal.metrics(p, y)
    assert metrics["Brier"] == pytest.approx((0.08 + 0.72) / 2)
    assert metrics["NLL"] == pytest.approx(-np.log([0.8, 0.4]).mean())
    assert metrics["accuracy"] == 0.5
    table = cal.abstain_table(p, y)
    assert table[2]["coverage"] == 0.5
    assert table[2]["accuracy"] == 1
    assert table[-1]["accuracy"] is None


@pytest.mark.parametrize("temperature", [0.05, 0.5, 1, 2, 100])
def test_temperature_keeps_argmax(temperature):
    p = np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
    scaled = cal.temperature_scale(p, temperature)
    assert np.array_equal(scaled.argmax(axis=1), p.argmax(axis=1))
    assert np.allclose(scaled.sum(axis=1), 1)


def test_crossfit_uses_only_other_folds(monkeypatch):
    p = np.column_stack((np.linspace(0.51, 0.9, 20), np.linspace(0.49, 0.1, 20)))
    y = np.array([0, 1] * 10)
    folds, _ = cal.make_folds(y)
    seen = []

    def fake_fit(train_p, train_y):
        seen.append((train_p.copy(), train_y.copy()))
        return 2.0

    monkeypatch.setattr(cal, "fit_temperature", fake_fit)
    after, audit = cal.crossfit(p, y, folds)
    for fold, (fit_p, fit_y) in enumerate(seen):
        assert np.array_equal(fit_p, p[folds != fold])
        assert np.array_equal(fit_y, y[folds != fold])
        assert not set(audit[fold]["train_indices"]) & set(audit[fold]["test_indices"])
    assert sorted(i for f in audit for i in f["test_indices"]) == list(range(20))
    assert np.array_equal(after.argmax(axis=1), p.argmax(axis=1))
    assert np.array_equal(cal.make_folds(y)[0], folds)


def test_folds_allow_rare_classes():
    folds, method = cal.make_folds(np.array([0] * 15 + [1, 2]))
    assert set(folds) == set(range(5))
    assert "stratified" in method


def test_repeat_lock_and_existing_result_refused(tmp_path):
    result, lock = tmp_path / "result.json", tmp_path / "run.lock"
    cal.claim_run(result, lock)
    with pytest.raises(FileExistsError):
        cal.claim_run(result, lock)
    result.write_text("{}")
    lock.unlink()
    with pytest.raises(FileExistsError):
        cal.claim_run(result, lock)


def test_invalid_probabilities_refused():
    with pytest.raises(ValueError, match="sum to one"):
        cal.validate_probabilities(np.array([[0.9, 0.9]]))


def test_platt_crossfit_is_binary_oof():
    y = np.tile([0, 1, 0, 1], 5)
    p = np.tile([0.2, 0.4, 0.6, 0.8], 5)
    folds, _ = cal.make_folds(y)
    after, audit = cal.platt_crossfit(p, y, folds)
    assert np.isfinite(after).all() and ((after >= 0) & (after <= 1)).all()
    for fold in audit:
        assert not set(fold["train_indices"]) & set(fold["test_indices"])


def test_reference_hash_failure_stops_before_fit(tmp_path, monkeypatch):
    reference = tmp_path / "reference.csv"
    reference.write_text("item_id,label\nx,Other\n")
    cache = tmp_path / "cache.npz"
    cache.with_suffix(".json").write_text('{"labels_sha256":"wrong"}')
    monkeypatch.setattr(cal, "REFERENCE", reference)
    monkeypatch.setattr(cal, "CACHES", {"event_model": cache})
    with pytest.raises(ValueError, match="label hash"):
        cal.load_d1()


def test_bootstrap_paired_seed_is_reproducible(monkeypatch):
    monkeypatch.setattr(cal, "N_BOOTSTRAP", 20)
    p, y = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]]), np.array([0, 1, 1])
    result = cal.bootstrap_ece(p, p, y)
    assert result == cal.bootstrap_ece(p, p, y)
    assert result["after_minus_before_95"] == [0, 0]
