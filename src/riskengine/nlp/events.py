"""Event class per text.

- keywords: rules, first match wins (most severe first). The baseline.
- model: our topic classifier fine-tuned on Twitter Financial News Topic; its 20 topics are
  mapped to our classes with TOPIC_TO_EVENT (docs/DECISIONS.md, E2/E5). bge-small, int8 ONNX in
  models/topic (E11). The topic data has no credit events, so the credit classes come from the
  stricter E13 rules (`credit_event`) on top.
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

# --- E13 credit rules (engine only; the keyword baseline above is unchanged) ---
_AGENCY = r"(?:Moody['’]?s|Fitch|DBRS|Standard (?:&|and) Poor['’]?s|S&P Global Ratings)"
_CUT = r"(?:downgrad\w*|cuts?|cutting|lower(?:s|ed|ing)|slash(?:es|ed|ing)?)"
_CREDIT_WORD = (
    r"(?:junk|credit rating|IDR|issuer default rating|investment[- ]grade|high[- ]yield|"
    r"bond rating|debt rating|CreditWatch|rating watch negative)"
)
# Plain "S&P" is usually the index, so it counts as the agency only when it acts or is cited.
AGENCY = re.compile(rf"\b{_AGENCY}|\bS&P\s+{_CUT}\b|\b(?:by|at)\s+S&P\b(?!\s*\d)", re.IGNORECASE)
CREDIT_WORD = re.compile(_CREDIT_WORD, re.IGNORECASE)
DOWNGRADE = re.compile(
    rf"\b{_AGENCY}.{{0,60}}?\b{_CUT}\b"  # Fitch downgrades Disney
    rf"|\bS&P\s+{_CUT}\b"  # S&P cuts Ford
    rf"|\b{_CUT}\b.{{0,60}}?\b(?:by|at)\s+(?:{_AGENCY}|S&P\b)"  # rating lowered at S&P
    rf"|\b{_CUT}\b.{{0,60}}?\b{_CREDIT_WORD}"  # cuts Boeing to junk
    rf"|\b{_CREDIT_WORD}.{{0,40}}?\b{_CUT}\b"  # credit rating lowered
    rf"|\bjunk (?:status|territory)\b"
    rf"|\b{_AGENCY}.{{0,60}}?outlook.{{0,30}}?negative",
    re.IGNORECASE | re.DOTALL,
)
DEFAULT = re.compile(
    r"bankrupt|chapter 11|insolven|"
    r"missed (?:a |an |its )?(?:coupon|interest|debt|bond|loan) payment|"
    r"\bdefault(?:s|ed)? on\b|\bdefaulted\b",
    re.IGNORECASE,
)
_DEBT = r"(?:debt|loans?|bonds?|coupons?|payments?|credit|creditors?|lenders?|obligations?)"
# plain "default" only next to a debt word: "credit defaults", "default on its loans"
_DEFAULT_NEAR_DEBT = re.compile(
    rf"\b{_DEBT}\W+(?:\w+\W+){{0,2}}default(?:s|ed)?\b|\bdefault(?:s|ed)?\W+(?:\w+\W+){{0,2}}{_DEBT}\b",
    re.IGNORECASE,
)
# an opinion or rumour, not a filing: "won't go bankrupt", "may file for bankruptcy"
_HEDGED_BANKRUPTCY = re.compile(
    r"\b(?:go|going|goes|could|may|might|will|would|won['’]?t|wont|brink of|risk of|verge of)"
    r"(?:\W+\w+){0,3}?\W+bankrupt\w*",
    re.IGNORECASE,
)
COVENANT = re.compile(dict(_RULES)["Covenant breach"], re.IGNORECASE)
_RATING_ACTION = (
    r"(?:downgrad\w*|upgrad\w*|initiat\w*|reiterat\w*|maintain\w*|reaffirm\w*|cuts?|"
    r"raise[sd]?|lower(?:s|ed)|lift(?:s|ed)|boost\w*|set)"
)
_STOCK_RATING = (
    r"(?:buy|sell|hold|neutral|overweight|underweight|outperform|underperform|equal[- ]weight|"
    r"(?:market|sector|peer) perform|in[- ]line|price target|PT|target|rating)"
)
ANALYST_RATING = re.compile(
    rf"\b{_RATING_ACTION}\b.{{0,80}}?\b{_STOCK_RATING}\b"
    rf"|\b{_STOCK_RATING}\b.{{0,60}}?\b{_RATING_ACTION}\b",
    re.IGNORECASE | re.DOTALL,
)


def names_agency(text: str) -> bool:
    """A credit rating agency is named as acting or cited (not the S&P index)."""
    return bool(AGENCY.search(text))


def is_analyst_rating(text: str) -> bool:
    """A stock analyst's rating or price target change, with no agency or credit word."""
    return (
        bool(ANALYST_RATING.search(text))
        and not names_agency(text)
        and not CREDIT_WORD.search(text)
        and not DEFAULT.search(text)
    )


def credit_event(text: str) -> str | None:
    """E13: a credit class if the text reports one, else None (most severe first)."""
    facts = _HEDGED_BANKRUPTCY.sub(" ", text)
    if DEFAULT.search(facts) or _DEFAULT_NEAR_DEBT.search(facts):
        return "Default"
    if is_analyst_rating(text):
        return None
    if DOWNGRADE.search(text):
        return "Credit downgrade"
    if COVENANT.search(text):
        return "Covenant breach"
    return None


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
            model = [topic_to_event(t) for t in topics.predict(texts)]
            return [credit_event(t) or m for t, m in zip(texts, model, strict=True)]

        return classify
    raise NotImplementedError(f"event model {kind!r} is not built yet")
