"""E23 export and evaluation logic on synthetic data (no real labels exist yet)."""

import pandas as pd
import pytest

from riskengine.eval.credit_gold import (
    HARD_NEGATIVE,
    LABEL_COLS,
    _weighted,
    blind,
    draw,
    evaluate,
    headline_id,
    stratify,
    validate,
    wilson,
)


def pool(n_fired=5, n_rule=80, n_hard=90, n_other=120):
    rows = []
    for i in range(n_fired):
        rows.append((f"Moody's cuts Boeing to junk {i}", "Credit downgrade", True))
    for i in range(n_rule):
        rows.append((f"Fitch downgrades J.P. Morgan 2006-FL2 class {i}", "Credit downgrade", False))
    for i in range(n_hard):
        rows.append((f"Fitch affirms Disney at A, outlook stable {i}", None, False))
    for i in range(n_other):
        rows.append((f"Apple unveils a new phone {i}", None, False))
    df = pd.DataFrame(rows, columns=["headline", "rule_class", "fired"])
    df["headline_id"] = [headline_id("news", "2015-01-01", h, "BA") for h in df["headline"]]
    df["event_date"], df["target_company"] = "2015-01-01", "BA"
    df["source_set"], df["period"] = "history", "2010-2016"
    return df


def test_headline_id_is_stable_and_distinct():
    a = headline_id("news", "2015-01-01", "x", "BA")
    assert a == headline_id("news", "2015-01-01", "x", "BA")
    assert a != headline_id("news", "2015-01-01", "x", "DIS")


def test_hard_negative_pattern():
    for text in ("Fitch Affirms BofA", "Fitch Downgrades JPMorgan 2006-FL2", "Netflix sells bonds"):
        assert HARD_NEGATIVE.search(text)
    assert not HARD_NEGATIVE.search("Apple unveils a new phone")


def test_strata_and_sample_sizes_are_reproducible():
    p = pool()
    assert stratify(p).value_counts().to_dict() == {"S4": 120, "S3": 90, "S2": 80, "S1": 5}
    a, sizes = draw(p)
    b, _ = draw(p)
    assert a["headline_id"].tolist() == b["headline_id"].tolist()  # same seed, same sample
    assert sizes == {"S1": 5, "S2": 80, "S3": 90, "S4": 120}
    assert a["stratum"].value_counts().to_dict() == {"S2": 60, "S3": 60, "S4": 40, "S1": 5}


def test_blind_file_hides_every_system_output():
    sample, _ = draw(pool())
    b = blind(sample)
    assert list(b.columns) == ["headline_id", "headline", "event_date", "target_company",
                               *LABEL_COLS]  # fmt: skip
    for hidden in ("rule_class", "fired", "stratum", "impact", "gate_reason"):
        assert hidden not in b.columns
    assert (b["adjudication_status"] == "unreviewed").all()


def labelled(sample, fire_s1=4):
    lab = blind(sample)
    lab["adjudication_status"] = "reviewed"
    lab["target_company_match_correct"] = "Y"
    lab["credit_event_relevant"] = "N"
    lab["event_subtype"] = "other"
    lab["existing_gate_should_fire"] = "N"
    s1 = sample.index[sample["stratum"] == "S1"]
    lab.loc[s1[:fire_s1], ["credit_event_relevant", "existing_gate_should_fire"]] = "Y"
    lab.loc[s1[:fire_s1], ["event_subtype", "materiality"]] = ["downgrade", "medium"]
    s4 = sample.index[sample["stratum"] == "S4"]
    lab.loc[s4[:2], ["credit_event_relevant", "existing_gate_should_fire"]] = "Y"  # missed
    lab.loc[s4[:2], ["event_subtype", "materiality"]] = ["default", "high"]
    return lab


def test_validate_flags_incomplete_scored_rows():
    sample, _ = draw(pool())
    lab = labelled(sample)
    assert validate(lab) == []
    lab.loc[0, "event_subtype"] = "nonsense"
    assert any("event_subtype" in e for e in validate(lab))
    lab.loc[0, "adjudication_status"] = "unreviewed"  # unreviewed rows are not checked
    assert validate(lab) == []


def test_weighted_recall_counts_missed_events_in_other_strata():
    sample, sizes = draw(pool())
    df = labelled(sample).merge(sample[["headline_id", "stratum"]], on="headline_id")
    w = _weighted(df, sizes, {"S1"}, "existing_gate_should_fire")
    assert w["precision"] == pytest.approx(4 / 5)
    # S4: 2 of 40 sampled are positives, out of 120 in the pool → 6 estimated misses
    assert w["recall"] == pytest.approx(4 / (4 + 6))


def test_evaluate_scores_only_reviewed_rows():
    sample, sizes = draw(pool())
    lab = labelled(sample)
    lab.loc[lab.index[-1], "adjudication_status"] = "unreviewed"
    meta = sample.drop(columns=["headline"])
    res = evaluate(lab, meta, {"stratum_population": sizes})
    assert res["rows_by_status"] == {"reviewed": len(lab) - 1, "unreviewed": 1}
    assert res["gate"]["precision_on_fired"]["n"] == 5
    assert res["target_match_precision"]["recall"].startswith("not measurable")


def test_wilson():
    w = wilson(79, 93)
    assert w["share"] == pytest.approx(0.8495, abs=1e-4)
    assert w["ci95"] == pytest.approx([0.763, 0.9082], abs=1e-3)


def test_ai_labelled_rows_are_scored():
    sample, sizes = draw(pool())
    lab = labelled(sample)
    lab["adjudication_status"] = "ai_labelled"
    res = evaluate(lab, sample.drop(columns=["headline"]), {"stratum_population": sizes})
    assert res["rows_by_status"] == {"ai_labelled": len(lab)}
    assert res["gate"]["precision_on_fired"]["n"] == 5
