"""E16: nine learned event classes (docs/DECISIONS.md)."""

from pathlib import Path

import pytest

from riskengine.eval.datasets import TOPIC_NAMES
from riskengine.nlp.events import (
    CREDIT_CLASSES,
    EVENT_CLASSES,
    LEARNED_CLASSES,
    OWN_TOPIC,
    TOPIC_TO_EVENT,
    TOPIC_TO_EVENT_E2,
    classify_keywords,
    get_event_classifier,
    topic_to_event,
)
from riskengine.nlp.impact import CLASS_SEVERITY

E16_MAPPING = {
    "Politics": "Geopolitical",
    "Fed | Central Banks": "Macroeconomic",
    "Macro": "Macroeconomic",
    "Treasuries | Corporate Debt": "Macroeconomic",
    "Energy | Oil": "Macroeconomic",
    "Gold | Metals | Materials": "Macroeconomic",
    "Currencies": "Macroeconomic",
    "M&A | Investments": "M&A",
    "IPO": "M&A",
    "Company | Product News": "Product news",
    "Earnings": "Earnings",
    "Dividend": "Earnings",
    "Analyst Update": "Analyst rating",
    "Legal | Regulation": "Legal/Regulatory",
    "Personnel Change": "Management change",
    "General News | Opinion": "Other",
    "Markets": "Other",
    "Stock Commentary": "Other",
    "Stock Movement": "Other",
    "Financials": "Other",
}


def test_twelve_classes_nine_learned():
    assert len(EVENT_CLASSES) == 12
    assert set(LEARNED_CLASSES) == set(EVENT_CLASSES) - CREDIT_CLASSES
    assert len(LEARNED_CLASSES) == 9


@pytest.mark.parametrize("topic", TOPIC_NAMES)
def test_every_tfn_topic_maps_as_registered(topic):
    assert topic_to_event(topic) == E16_MAPPING[topic]


def test_mapping_covers_all_topics_and_only_learned_classes():
    assert set(TOPIC_TO_EVENT) == set(TOPIC_NAMES)
    assert set(TOPIC_TO_EVENT.values()) == set(LEARNED_CLASSES)


def test_e2_table_still_reproduces():
    assert topic_to_event("Earnings", TOPIC_TO_EVENT_E2) == "Other"
    assert topic_to_event("Energy | Oil", TOPIC_TO_EVENT_E2) == "Other"
    assert topic_to_event("Politics", TOPIC_TO_EVENT_E2) == "Geopolitical"


@pytest.mark.parametrize("label", LEARNED_CLASSES)
def test_a_nine_class_model_passes_through(label):
    assert topic_to_event(label) == label


def test_every_class_has_a_registered_severity():
    new = [c for c in LEARNED_CLASSES if c not in TOPIC_TO_EVENT_E2.values()]
    assert {c: CLASS_SEVERITY[c] for c in new} == {
        "Product news": 4,
        "Earnings": 5,
        "Analyst rating": 4,
        "Legal/Regulatory": 6,
        "Management change": 5,
        "Other": 3,
    }


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Apple unveils new iPhone", "Product news"),
        ("Microsoft beats earnings estimates", "Earnings"),
        ("Tesla CFO steps down", "Management change"),
        ("SEC sues Meta over privacy", "Legal/Regulatory"),
        ("Morgan Stanley raises Apple price target to $300", "Analyst rating"),
        ("Moody's downgrades Boeing to Baa2", "Credit downgrade"),
    ],
)
def test_keyword_rules_for_the_new_classes(text, label):
    assert classify_keywords(text) == label


SMOKE = [
    ("Apple unveils new iPhone", "Product news"),
    ("Goldman downgrades Apple to Sell", "Analyst rating"),
    pytest.param(
        "Microsoft beats earnings estimates",
        "Earnings",
        # known miss, reported in the README: the model says Other (0.84); TFN earnings tweets
        # usually carry the numbers ("EPS $1.40 beats by $0.05"). strict: a fix shows up here.
        marks=pytest.mark.xfail(strict=True, reason="E16 model: Other"),
    ),
    ("Tesla CFO steps down", "Management change"),
    ("SEC sues Meta over privacy", "Legal/Regulatory"),
    ("Moody's cuts Boeing to junk", "Credit downgrade"),  # credit rule, not Analyst rating
]


@pytest.fixture(scope="module")
def engine_events():
    if not (Path(OWN_TOPIC) / "model.onnx").exists():
        pytest.skip("no engine model")
    return get_event_classifier("model")


@pytest.mark.parametrize(("text", "label"), SMOKE)
def test_engine_model_smoke(engine_events, text, label):
    assert engine_events([text]) == [label]
