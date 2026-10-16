"""General rule boundaries and amended B–D routing; no actual D2/final rows are used."""

import numpy as np
import pytest

from riskengine.eval import e24_rules
from riskengine.eval.e24_dev import Config, build_predictor

NAMES = ["Geopolitical", "Macroeconomic", "M&A", "Product news", "Earnings", "Analyst rating",
         "Legal/Regulatory", "Management change", "Other"]  # fmt: skip


@pytest.mark.parametrize(
    "text",
    [
        "Acme unveils a new processor",
        "Acme releases a phone",
        "Acme recalls vehicles after a battery fault",
        "Acme signs supply agreement for devices",
        "Retailer to sell 200 products from its new partner",
        "Acme's new tablet expands its device portfolio",
        "Acme testing online platform to let customers compare loans",
    ],
)
def test_actual_product_events(text):
    assert e24_rules.product_rule(text)


@pytest.mark.parametrize(
    "text",
    [
        "What to know about a crew spacecraft launch",
        "Acme remains on track to launch local production in a new factory",
        "Acme launches an investigation into a factory incident",
        "Acme releases quarterly financial results",
        "Acme launches an IPO",
        "Acme shares fall after it unveils a new phone",
        "Acme cuts the price of its existing car",
        "Acme prototype spotted on the motorway",
        "Acme CEO says the company's position is strong",
    ],
)
def test_company_and_mission_news_are_not_product_events(text):
    assert not e24_rules.product_rule(text)


@pytest.mark.parametrize(
    "text",
    [
        "Cedar Wealth Management LLC Buys Acme Corp, Beta Inc, Sells Delta Corp",
        "Cedar Office Inc Buys Treasury Bond ETF, Equity Index Fund, ...",
        "Cedar Investment Advisors Ltd Sells Acme Corp, Beta Corp, ...",
        "Cedar Advisory Group LLC Buys Core S&P Total U.S. ...",
    ],
)
def test_investment_holdings_are_other(text):
    assert e24_rules.holdings_rule(text)


@pytest.mark.parametrize(
    "text",
    [
        "Cedar to buy Acme for $2 billion",
        "Cedar Inc Buys Acme, Beta in takeover agreement",
        "Cedar Bank Buys a stake in Acme, expands its business",
        "Cedar acquires Acme in a merger",
        "Cedar Capital Buys Acme, Beta for $2 billion",
    ],
)
def test_real_deal_language_is_not_a_holdings_list(text):
    assert not e24_rules.holdings_rule(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Cedar upgrades Acme to Buy", "Analyst rating"),
        ("Cedar Downgrades Acme: Weak Consumer Demand", "Analyst rating"),
        ("UPDATE: Cedar Downgrades Acme As Earnings Weaken", "Analyst rating"),
        ("Top Upgrades, Downgrades For Monday", "Analyst rating"),
        ("Five Biggest Price Target Changes", "Analyst rating"),
        ("Cedar maintains Buy on Acme, raises price target to $50", "Analyst rating"),
        ("Cedar reiterates Buy on Acme, raises tgt to $50", "Analyst rating"),
        ("Cedar maintains Buy on Acme, $50 price target", "Other"),
        ("Acme shares are trading higher after Cedar upgrades it to Buy", None),
        ("Stock Market Recovers From Losses; Acme Downgraded", None),
        ("Fitch downgrades Acme's debt rating", None),
        ("An analyst remains optimistic about Acme", None),
    ],
)
def test_analyst_changes_reiterations_and_primary_event(text, expected):
    assert e24_rules.analyst_rule(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "What's in store for Acme in Q3 earnings?",
        "Notable earnings before Friday's open",
        "Earnings season kicks off",
        "Five earnings charts to watch",
    ],
)
def test_unquantified_previews_are_other(text):
    assert e24_rules.earnings_preview_rule(text)
    assert not e24_rules.earnings_rule(text)


@pytest.mark.parametrize(
    "text",
    [
        "Acme reports Q2 EPS $2 ahead of earnings call",
        "Acme posts a quarterly profit",
        "Acme reports Q4 operating margin 20%",
        "Acme CFO expects the operating trough in Q2",
    ],
)
def test_actual_results_and_guidance_survive_preview_filter(text):
    assert not e24_rules.earnings_preview_rule(text)
    assert e24_rules.earnings_rule(text)


def fixed_model(label):
    def probs(texts):
        p = np.full((len(texts), len(NAMES)), 0.001)
        p[:, NAMES.index(label)] = 0.992
        return NAMES, p

    return probs


def test_amended_routing_leaves_A_unchanged_and_credit_first():
    texts = [
        "Cedar Wealth Management LLC Buys Acme Corp, Beta Inc, ...",
        "Cedar upgrades Acme to Buy",
        "Notable earnings before Monday's open",
        "Fitch downgrades Acme debt to junk",
    ]
    a, a_sources = build_predictor(Config("A"), fixed_model("M&A"))(texts)
    assert a == ["M&A", "M&A", "M&A", "Credit downgrade"]
    assert a_sources == ["model", "model", "model", "credit_rule"]
    for candidate in "BCD":
        cfg = Config(candidate, 0.5 if candidate == "D" else None)
        labels, sources = build_predictor(cfg, fixed_model("M&A"))(texts)
        assert labels[:2] == ["Other", "Analyst rating"]
        assert sources[:2] == ["holdings_rule", "analyst_rule"]
        assert labels[-1] == "Credit downgrade" and sources[-1] == "credit_rule"
        assert labels[2] == ("Other" if candidate in "CD" else "M&A")


def test_preview_rule_overrides_model_earnings_for_C_and_D():
    text = ["What's in store for Acme Q3 earnings?"]
    assert build_predictor(Config("A"), fixed_model("Earnings"))(text)[0] == ["Earnings"]
    assert build_predictor(Config("B"), fixed_model("Earnings"))(text)[0] == ["Earnings"]
    for cfg in (Config("C"), Config("D", 0.5)):
        labels, sources = build_predictor(cfg, fixed_model("Earnings"))(text)
        assert labels == ["Other"] and sources == ["earnings_preview_rule"]
