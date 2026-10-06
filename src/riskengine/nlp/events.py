"""Event class per text.

- keywords: rules, first match wins (most severe first). The baseline.
- model: our topic classifier fine-tuned on Twitter Financial News Topic; its 20 topics are
  mapped to our classes with TOPIC_TO_EVENT (docs/DECISIONS.md, E2/E5).
"""

import re
from collections.abc import Callable, Sequence

from riskengine.paths import MODELS

OWN_TOPIC = MODELS / "topic"

EVENT_CLASSES = [
    "Default",
    "Credit downgrade",
    "Covenant breach",
    "M&A",
    "Geopolitical",
    "Macroeconomic",
    "Product launch",
    "Other",
]

_RULES: list[tuple[str, str]] = [
    ("Default", r"\bdefault(s|ed)?\b|bankrupt|chapter 11|insolven|missed (a )?(coupon|payment)"),
    ("Credit downgrade", r"downgrad|rating (cut|lowered)|outlook (to|cut to) negative|junk status"),
    (
        "Covenant breach",
        r"covenant|waiver from (its )?lenders|breach(ed)? (of )?(its )?(loan|debt)",
    ),
    ("M&A", r"acqui(re|res|red|sition)|merger|takeover|buyout|\bto buy\b|\bstake in\b"),
    (
        "Geopolitical",
        r"china|tariff|sanction|\bwar\b|geopolit|election|military|russia|iran|hong kong",
    ),
    (
        "Macroeconomic",
        r"\bfed\b|federal reserve|inflation|interest rate|\bgdp\b|jobless|unemployment|"
        r"recession|stimulus|\bcpi\b|treasury yield",
    ),
    (
        "Product launch",
        r"\blaunch|unveil|rolls? out|new (iphone|product|model|service)|introduces",
    ),
]
_COMPILED = [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in _RULES]


CREDIT_CLASSES = {"Default", "Credit downgrade", "Covenant breach"}

TOPIC_TO_EVENT = {
    "Fed | Central Banks": "Macroeconomic",
    "Macro": "Macroeconomic",
    "M&A | Investments": "M&A",
    "Politics": "Geopolitical",
}  # every other topic -> "Other"


def classify_keywords(text: str) -> str:
    for label, pattern in _COMPILED:
        if pattern.search(text):
            return label
    return "Other"


def topic_to_event(topic: str) -> str:
    return TOPIC_TO_EVENT.get(topic, "Other")


def get_event_classifier(kind: str) -> Callable[[Sequence[str]], list[str]]:
    """A function: texts -> event class per text."""
    if kind == "keywords":
        return lambda texts: [classify_keywords(t) for t in texts]
    if kind == "model":
        from riskengine.nlp.classifier import TextClassifier

        if not OWN_TOPIC.exists():
            raise FileNotFoundError(f"own topic model not found at {OWN_TOPIC}")
        topics = TextClassifier(str(OWN_TOPIC))

        def classify(texts: Sequence[str]) -> list[str]:
            # The topic data has no credit events, so the credit rules still apply on top
            # until the credit classes are trained (step 2b).
            rules = [classify_keywords(t) for t in texts]
            model = [topic_to_event(t) for t in topics.predict(texts)]
            return [r if r in CREDIT_CLASSES else m for r, m in zip(rules, model, strict=True)]

        return classify
    raise NotImplementedError(f"event model {kind!r} is not built yet")
