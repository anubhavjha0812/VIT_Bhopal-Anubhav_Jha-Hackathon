"""E38b invented fixtures and old-sample reproduction; never open the new sample."""

import json

import pandas as pd
import pytest

from riskengine.eval.linker_extended import metrics, subject_links
from riskengine.ingest.linking import Linker
from riskengine.ingest.linking_extended import ExtendedLinker
from riskengine.paths import DATA


@pytest.fixture(scope="module")
def candidate():
    return ExtendedLinker.from_csv(candidate=2)


@pytest.mark.parametrize(
    "firm,ticker",
    [
        ("J.P. Morgan", "JPM"),
        ("J. P. Morgan", "JPM"),
        ("JP Morgan", "JPM"),
        ("Bank of America Merrill Lynch", "BAC"),
        ("BofA/Merrill", "BAC"),
        ("Goldman Sachs", "GS"),
        ("Morgan Stanley", "MS"),
        ("Citigroup", "C"),
        ("Wells Fargo", "WFC"),
        ("U.S. Bancorp", "USB"),
    ],
)
def test_research_desk_with_out_of_catalog_target(candidate, firm, ticker):
    text = f"UPDATE: {firm} Comments on Prospects on Fictitious Robotics"
    assert ticker in candidate.research_attribution(text)
    assert ticker in candidate.analyst_only(text)
    assert subject_links(candidate, text) == set()


@pytest.mark.parametrize(
    "action",
    [
        "Maintains Neutral on",
        "Reiterates Buy on",
        "Reinstates Coverage on",
        "Initiates on",
        "Upgrades",
        "Downgrades",
        "Raises Price Target on",
        "Lowers PO on",
        "Cuts PT on",
        "Comments on",
        "Reviews",
        "Sees Value in",
        "Says Buy",
        "Reports on",
        "Adjusts Estimates on",
    ],
)
def test_research_forms_do_not_require_a_catalog_target(candidate, action):
    text = f"Bank of America {action} Fictional Industrials"
    assert subject_links(candidate, text) == set()


@pytest.mark.parametrize(
    "firm,key",
    [
        ("Barclays", "Barclays"),
        ("BMO Capital", "BMO"),
        ("Mizuho", "Mizuho"),
        ("Oppenheimer", "Oppenheimer"),
        ("RBC Capital", "RBC"),
        ("UBS", "UBS"),
        ("Credit Suisse", "Credit Suisse"),
        ("Deutsche Bank", "Deutsche Bank"),
    ],
)
def test_outside_brokers_are_recognised_as_attribution_without_catalog_expansion(
    candidate, firm, key
):
    assert candidate.research_attribution(f"{firm} Raises Price Target on Fictitious Robotics") == {
        key
    }
    assert set(candidate.tickers) == set(ExtendedLinker.from_csv().tickers)


@pytest.mark.parametrize(
    "text",
    [
        "Fictitious Robotics upgraded to Buy by Bank of America",
        "Fictitious Robotics downgraded at Goldman Sachs",
        "Fictitious Robotics cut to Neutral by Morgan Stanley",
    ],
)
def test_passive_rating_attribution(candidate, text):
    assert subject_links(candidate, text) == set()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Microsoft Price Target Raised to $800", {"MSFT"}),
        ("Microsoft Target Price Raised to $800", {"MSFT"}),
        ("Microsoft PT, PO and price objective raised", {"MSFT"}),
        ("Microsoft earnings estimate has a Target of $800", {"MSFT"}),
        ("Target earnings rise; Microsoft Price Target raised", {"TGT", "MSFT"}),
        ("Target Corporation sets its revenue target price", {"TGT"}),
        ("$TGT price target raised by Goldman Sachs", {"TGT"}),
    ],
)
def test_price_language_does_not_create_a_target_company(candidate, text, expected):
    assert subject_links(candidate, text) == expected


@pytest.mark.parametrize(
    "text,ticker",
    [
        ("Bank of America earnings beat expectations", "BAC"),
        ("JPMorgan probe intensifies", "JPM"),
        ("Bank of America loses customers", "BAC"),
        ("Bank of America signs a lease", "BAC"),
        ("Bank of America CEO account hacked", "BAC"),
        ("Goldman Sachs faces a rate cut", "GS"),
        ("Bank of America Says Its Earnings Rise", "BAC"),
        ("Goldman Sachs Raises Its Dividend", "GS"),
        ("$GS Comments on Fictitious Robotics", "GS"),
    ],
)
def test_real_bank_subject_and_own_cashtag_stay(candidate, text, ticker):
    assert subject_links(candidate, text) == {ticker}


def test_original_links_are_kept_except_the_existing_risky_name_guards(candidate):
    old = Linker.from_csv()
    text = "Microsoft, Google, Facebook, Nvidia, AMD, Netflix, Disney, Tesla and Boeing earnings"
    assert set(candidate.link(text)) & set(old.tickers) == set(old.link(text))
    for text in ["Apple harvest lifts profits", "Amazon river threatens stores"]:
        assert old.link(text) and not candidate.link(text)
    # Missing company context remains the E38 guard; E38b cannot silently loosen it.
    assert old.link("Apple unveils a concept") == ("AAPL",)
    assert candidate.link("Apple unveils a concept") == ()
    for ticker in old.tickers:
        text = f"{old.company_name(ticker)} reports quarterly earnings"
        assert set(candidate.link(text)) & set(old.tickers) == set(old.link(text))


def test_candidate_one_reproduces_every_saved_e38_prediction_and_metric():
    stored = json.loads((DATA / "outputs/linker_extended.json").read_text())
    one = ExtendedLinker.from_csv()
    assert one.candidate == 1
    gold, predicted = [], []
    for row in stored["set_a_rows"]:
        assert list(one.link(row["text"])) == row["extended_mentions"]
        pred = subject_links(one, row["text"])
        assert sorted(pred) == row["extended_subjects"]
        gold.append(set(row["gold"]))
        predicted.append(pred)
    assert metrics(gold, predicted) == stored["set_a"]["extended"]


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError, match="candidate"):
        ExtendedLinker.from_csv(candidate=4)


def test_set_b_judgements_require_every_actual_removed_link(candidate):
    from riskengine.eval.linker_extended_v2 import (
        judgement_template,
        set_b_comparisons,
        validate_judgements,
    )

    frame = pd.DataFrame(
        [
            {
                "headline_id": "invented-harvest",
                "headline": "Apple harvest boosts farm profits",
                "target_company": "AAPL",
            },
            {
                "headline_id": "invented-concept",
                "headline": "Apple unveils a concept",
                "target_company": "AAPL",
            },
        ]
    )
    expected = judgement_template(set_b_comparisons(frame, Linker.from_csv(), candidate))
    assert len(expected) == 4  # both mention/subject views for each invented example
    actual = expected.copy()
    with pytest.raises(ValueError, match="reason"):
        validate_judgements(expected, actual)
    actual["reason"] = "Invented fixture judgement."
    actual["removed_link_was_subject"] = actual.id.map(
        {"invented-harvest": "no", "invented-concept": "yes"}
    )
    assert validate_judgements(expected, actual) == 1  # no double count across views
    with pytest.raises(ValueError, match="exact frozen"):
        validate_judgements(expected, actual.iloc[:-1])


def test_e38b_fixed_decision_checks_each_condition():
    from riskengine.eval.linker_extended_v2 import adoption

    old, new = {"recall": 0.25}, {"precision": 0.95, "recall": 0.4}
    assert adoption(old, new, 0, 58)["adopt_live"]
    assert not adoption(old, {**new, "precision": 0.9499}, 0, 60)["adopt_live"]
    assert not adoption(old, {**new, "recall": 0.25}, 0, 60)["adopt_live"]
    assert not adoption(old, new, 1, 60)["adopt_live"]
    assert not adoption(old, new, 0, 57)["adopt_live"]
    assert not adoption(old, {**new, "precision": None}, 0, 60)["adopt_live"]


def test_e38b_repeat_run_refused_before_loading_or_scoring(tmp_path, monkeypatch):
    from riskengine.eval import linker_extended_v2 as evaluation

    result = tmp_path / "result.json"
    result.write_text("{}")
    monkeypatch.setattr(evaluation, "verify_seal", lambda work: pytest.fail("must not load"))
    with pytest.raises(FileExistsError, match="second scored run"):
        evaluation.score_once(result=result, work=tmp_path)


def test_e38b_verifies_candidate_hash_before_scoring(tmp_path):
    from riskengine.eval.linker_extended_v2 import rules_text, verify_seal

    candidate = tmp_path / "fake_candidate.txt"
    candidate.write_text("changed")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "rules_text": rules_text(),
                "files_sha256": {str(candidate): "wrong digest"},
            }
        )
    )
    with pytest.raises(ValueError, match="sealed file changed"):
        verify_seal(tmp_path)
