"""Event class per text. Baseline: keyword rules, first match wins (most severe first)."""

import re

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


def classify_keywords(text: str) -> str:
    for label, pattern in _COMPILED:
        if pattern.search(text):
            return label
    return "Other"
