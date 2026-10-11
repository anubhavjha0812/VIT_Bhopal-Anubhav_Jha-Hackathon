"""E24 label merging, audit selection, consistency rules and audit metrics (K2), and the
ignore rules for raw AI outputs and final labels (K4)."""

import math
import subprocess

import pandas as pd
import pytest

from riskengine.eval.e24_gate import GateError
from riskengine.eval.e24_labels import (
    EXCLUDE,
    adjudication_sheet,
    audit_metrics,
    audit_rows,
    merge_labels,
    validate_merged,
    wilson,
)
from riskengine.paths import ROOT

IDS = [f"d{i:02d}" for i in range(20)]


def ai(labels, ambiguous=None):
    ambiguous = ambiguous or ["N"] * len(labels)
    return pd.DataFrame({"item_id": IDS[: len(labels)], "label": labels,
                         "ambiguous": ambiguous, "rationale": "the words"})  # fmt: skip


def scenario():
    """d00-d14 agree (Other); d15 disagree; d16 ambiguous; d17 invalid; d18-d19 agree (M&A)."""
    a = ["Other"] * 15 + ["Earnings", "Other", "Weather", "M&A", "M&A"]
    b = ["Other"] * 15 + ["Other", "Other", "Other", "M&A", "M&A"]
    amb_a = ["N"] * 16 + ["Y"] + ["N"] * 3
    return ai(a, amb_a), ai(b)


def human_for(sheet, label=None, exclude=()):
    h = sheet[["item_id"]].copy()
    h["human_label"] = [EXCLUDE if i in exclude else (label or "Other") for i in h["item_id"]]
    h["human_rationale"] = "checked"
    return h


# --- deterministic audit selection ---


@pytest.mark.parametrize("n", [0, 1, 4, 5, 7, 10, 17])
def test_audit_size_is_ceil_of_20_percent(n):
    picked = audit_rows([f"x{i}" for i in range(n)])
    assert len(picked) == math.ceil(0.2 * n)


def test_audit_selection_is_deterministic_and_order_free():
    ids = [f"x{i}" for i in range(30)]
    assert audit_rows(ids) == audit_rows(list(reversed(ids)))
    assert set(audit_rows(ids)) <= set(ids)
    assert audit_rows(ids, seed=25) != audit_rows(ids)  # the seed matters


# --- merging and consistency ---


def test_sheet_lists_every_row_a_person_must_label():
    a, b = scenario()
    sheet = adjudication_sheet(a, b, IDS)
    reasons = dict(zip(sheet["item_id"], sheet["reason"], strict=True))
    assert reasons["d15"] == "disagreement"
    assert reasons["d16"] == "ambiguous"
    assert reasons["d17"] == "invalid AI label"
    audited = [i for i, r in reasons.items() if r == "audit"]
    assert len(audited) == math.ceil(0.2 * 17)  # 17 agreement rows (d00-d14, d18, d19)


def test_merge_sources_and_audited_rows_use_the_human_label():
    a, b = scenario()
    sheet = adjudication_sheet(a, b, IDS)
    audited = sheet.loc[sheet["reason"] == "audit", "item_id"].tolist()
    m = merge_labels(a, b, human_for(sheet, label="Earnings", exclude={"d17"}), IDS)
    by = m.set_index("item_id")
    for i in audited:
        assert by.loc[i, "label_source"] == "ai_agreed_human_audited"
        assert by.loc[i, "label"] == "Earnings" == by.loc[i, "human_label"]
    assert by.loc["d15", "label_source"] == "human_adjudicated"
    assert by.loc["d16", "label_source"] == "human_adjudicated"
    assert by.loc["d16", "ambiguous"] == "Y"
    assert by.loc["d17", "label_source"] == "excluded" and by.loc["d17", "label"] == ""
    plain = by[by["label_source"] == "ai_agreed"]
    assert (plain["label"] == plain["chatgpt_label"]).all() and (plain["human_label"] == "").all()


@pytest.mark.parametrize(
    "fix, msg",
    [
        (lambda a, b, h: (pd.concat([a, a.iloc[:1]]), b, h), "duplicate"),
        (lambda a, b, h: (a.iloc[:-1], b, h), "do not match"),
        (lambda a, b, h: (a.drop(columns=["ambiguous"]), b, h), "lacks columns"),
        (lambda a, b, h: (a, b, h.iloc[:-1]), "exactly the rows"),
        (lambda a, b, h: (a, b, pd.concat([h, h.iloc[:1].assign(item_id="d00")])), "exactly"),
        (lambda a, b, h: (a, b, h.assign(human_label="Weather")), "12 classes or EXCLUDE"),
        (lambda a, b, h: (a, b, h.assign(human_rationale="")), "rationale"),
        (lambda a, b, h: (a, b, pd.concat([h, h.iloc[:1]])), "duplicate"),
    ],
)
def test_malformed_inputs_are_rejected(fix, msg):
    a, b = scenario()
    h = human_for(adjudication_sheet(a, b, IDS))
    with pytest.raises(GateError, match=msg):
        merge_labels(*fix(a, b, h), IDS)


def merged():
    a, b = scenario()
    return merge_labels(a, b, human_for(adjudication_sheet(a, b, IDS), exclude={"d17"}), IDS)


@pytest.mark.parametrize(
    "tamper, msg",
    [
        (lambda m: m.assign(label=m["label"].where(m["label_source"] != "ai_agreed", "M&A")),
         "ai_agreed"),
        (lambda m: m.assign(human_label=m["human_label"].where(
            m["label_source"] != "ai_agreed_human_audited", "Earnings")), "audited rows"),
        (lambda m: m.assign(audit_selected=m["item_id"].eq("d15") | m["audit_selected"]),
         "audit rows must be agreement rows"),
        (lambda m: m.assign(label_source=m["label_source"].replace("excluded", "ai_agreed")),
         "ai_agreed"),
        (lambda m: m.assign(audit_selected="maybe"), "True or False"),
    ],
)  # fmt: skip
def test_tampered_merges_are_rejected(tamper, msg):
    with pytest.raises(GateError, match=msg):
        validate_merged(tamper(merged()))


def test_csv_round_trip_keeps_false_flags(tmp_path):
    m = merged()
    path = tmp_path / "m.csv"
    m.to_csv(path, index=False)
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_merged(back)  # "False" must not read as True
    assert audit_metrics(back)["audit"]["audited_rows"] == int(m["audit_selected"].sum())


# --- audit metrics ---


def test_audit_metrics_denominators():
    a, b = scenario()
    sheet = adjudication_sheet(a, b, IDS)
    audited = sheet.loc[sheet["reason"] == "audit", "item_id"].tolist()
    # the person corrects one audited row and excludes another
    h = human_for(sheet, exclude={"d17", audited[1]})
    h.loc[h["item_id"] == audited[0], "human_label"] = "Earnings"
    r = audit_metrics(merge_labels(a, b, h, IDS))
    assert r["audit"]["eligible_agreement_rows"] == 17
    assert r["audit"]["audited_rows"] == len(audited) == 4
    assert r["audit"]["human_correction_rate"]["k"] == 1
    assert r["audit"]["human_correction_rate"]["n"] == 4
    assert r["audit"]["human_exclusion_rate"]["k"] == 1
    assert r["audit"]["human_exclusion_rate"]["n"] == 4
    assert r["excluded_rows_all"] == 2 and r["invalid_ai_rows"] == 1
    assert r["kappa"]["rows_used"] == 18  # 20 - ambiguous d16 - invalid d17
    assert r["kappa"]["rows_left_out"] == {"invalid AI label": 1, "ambiguous (valid labels)": 1}


def test_audit_metrics_with_no_agreement_rows():
    a = ai(["Other", "M&A", "Earnings"])
    b = ai(["M&A", "Other", "Other"])
    ids = IDS[:3]
    sheet = adjudication_sheet(a, b, ids)
    r = audit_metrics(merge_labels(a, b, human_for(sheet), ids))
    assert r["audit"]["audited_rows"] == 0
    assert r["audit"]["human_correction_rate"]["rate"] is None
    assert r["audit"]["eligible_agreement_rows"] == 0


def test_kappa_is_none_when_only_one_class_is_used():
    a, b = ai(["Other"] * 5), ai(["Other"] * 5)
    sheet = adjudication_sheet(a, b, IDS[:5])
    r = audit_metrics(merge_labels(a, b, human_for(sheet), IDS[:5]))
    assert r["kappa"]["value"] is None and r["kappa"]["rows_used"] == 5


def test_wilson_matches_known_interval():
    w = wilson(79, 93)
    assert w["ci95"] == pytest.approx([0.763, 0.9082], abs=1e-3)
    assert wilson(0, 0)["rate"] is None


# --- K4: ignore rules ---


@pytest.mark.parametrize(
    "path, ignored",
    [
        ("data/gold/e24/raw/chatgpt_final_news.csv", True),
        ("data/gold/e24/raw/gemini_D1.csv", True),
        ("data/gold/e24/final_news_labels.csv", True),
        ("data/gold/e24/final_tweets_labels.csv", True),
        ("data/gold/e24/merged_final_news_labels.csv", True),
        ("data/gold/e24/final_tweets_to_label.csv", True),
        ("data/gold/e24/GUIDE.md", False),
    ],
)
def test_git_ignores_raw_outputs_and_final_labels(path, ignored):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert (r.returncode == 0) == ignored
