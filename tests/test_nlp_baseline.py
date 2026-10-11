import pytest

from riskengine.nlp.events import EVENT_CLASSES, classify_keywords
from riskengine.nlp.impact import CLASS_SEVERITY, impact_formula


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Hertz files for Chapter 11 bankruptcy", "Default"),
        ("Moody's downgrades Boeing to Baa2", "Credit downgrade"),
        ("Company seeks covenant waiver from lenders", "Covenant breach"),
        ("Microsoft in talks to acquire TikTok US", "M&A"),
        ("US-China tensions escalate over Hong Kong", "Geopolitical"),
        ("Fed holds interest rates near zero", "Macroeconomic"),
        ("Apple unveils new iPhone SE", "Product news"),
        ("Shares trade flat on Tuesday", "Other"),
    ],
)
def test_keyword_rules(text, label):
    assert classify_keywords(text) == label


def test_most_severe_rule_wins():
    assert classify_keywords("Downgrade after missed coupon, default feared") == "Default"


def test_every_class_has_a_severity():
    assert set(CLASS_SEVERITY) == set(EVENT_CLASSES)


@pytest.mark.parametrize("event", EVENT_CLASSES)
@pytest.mark.parametrize("sentiment", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("source", ["news", "tweets"])
def test_impact_always_1_to_10(event, sentiment, source):
    assert 1 <= impact_formula(event, sentiment, source) <= 10


def test_strong_default_news_crosses_module_b_trigger():
    assert impact_formula("Default", -0.9, "news") > 7
    assert impact_formula("Other", -0.9, "tweets") <= 7


def test_model_events_keep_credit_rules(monkeypatch, tmp_path):
    from riskengine.nlp import classifier, events

    class FakeTopics:
        def __init__(self, path):
            pass

        def predict(self, texts):
            return ["Politics", "Earnings"]

    monkeypatch.setattr(classifier, "TextClassifier", FakeTopics)
    monkeypatch.setattr(events, "OWN_TOPIC", tmp_path)  # exists
    classify = events.get_event_classifier("model")
    out = classify(["Senate passes China bill", "Hertz files for Chapter 11"])
    # model decides normal classes; credit rules still win until credit classes are trained
    assert out == ["Geopolitical", "Default"]
