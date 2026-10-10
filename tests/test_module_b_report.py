import json

import pytest

from riskengine.module_b.book import load_book
from riskengine.module_b.report import TOP, alert_multiple, matched_words, memo, run


@pytest.fixture(scope="module")
def boeing():
    return run(load_book(), "BA", 1, "covid2020", 1.0)


def test_matched_words_keep_the_longest_match():
    text = "Moody's downgrades Boeing to Baa2 from Baa1"
    assert matched_words(text) == ["Moody's downgrades"]
    assert matched_words("Fitch says JPMorgan defaulted on its bonds") == ["Fitch", "defaulted on"]
    assert matched_words("Apple beats on earnings") == []


def test_alert_multiple_follows_the_shock_switch():
    assert alert_multiple(9, "flat") == 1.0
    assert alert_multiple(9, "scaled") == pytest.approx(0.9)


def test_report_is_json_and_adds_up(boeing):
    json.dumps(boeing)  # the API returns it as is
    s, c = boeing["scenario"], boeing["capital"]
    assert (s["rating_before"], s["rating_after"]) == ("A-", "BBB+")
    assert c["cet1_after"] < c["cet1_before"]
    assert sum(boeing["by_instrument"].values()) == pytest.approx(c["loss_usd"], rel=1e-6)
    assert sum(o["loss"] for o in boeing["obligors"]) == pytest.approx(c["loss_usd"], rel=1e-6)
    assert sum(boeing["stages"].values()) == 15  # one revolver per company
    losses = [abs(p["loss"]) for p in boeing["top_losses"]]
    assert len(losses) == TOP and losses == sorted(losses, reverse=True)
    mc = boeing["monte_carlo"]
    assert sum(mc["histogram"]["after"]) == mc["draws"]
    assert mc["after"]["es"] >= mc["after"]["var"]
    assert 0 < boeing["reverse_stress"] < 1  # the COVID episode alone already breaches


def test_memo_quotes_the_alert_and_the_numbers(boeing):
    alert = {"published_at": "2020-04-10T09:00:00", "source": "news", "event": "Credit downgrade",
             "text": "Moody's downgrades Boeing", "matched": ["Moody's downgrades"],
             "gate": "fired: rating agency", "impact": 8}  # fmt: skip
    text = memo(boeing, alert)
    assert '"Moody\'s downgrades Boeing"' in text
    assert "fired: rating agency" in text
    assert "BA A- → BBB+" in text
    assert f"{100 * boeing['capital']['cet1_after']:.2f}%" in text


def test_market_only_run_has_no_obligor():
    rep = run(load_book(), None, 0, "euro2011", 1.0)
    assert rep["scenario"]["rating_before"] is None
    assert "Scenario:" not in memo(rep)
    assert rep["capital"]["cet1_after"] < rep["capital"]["cet1_before"]
