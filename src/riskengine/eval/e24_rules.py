"""E24 candidate rules (docs/DECISIONS.md, E24). Evaluation only: the engine does not use them.

Written from the E24 class definitions before any D1 label existed; changed only after
looking at D1 (never D2 or the final sets). Its sha256 is frozen with the chosen candidate.

- Earnings rule (candidates C, D): the text reports results, revenue, profit or loss figures,
  beats or misses, guidance or a dividend. A preview or opinion about coming results is not
  results, so "ahead of earnings" style texts are skipped.
- Product rule (candidates B, C, D): the text reports a product event of the company: a launch,
  unveiling, release, recall, or a deal to supply or buy a product. "Press release" is not one.
"""

import re

EARNINGS = re.compile(
    r"\b(?:earnings|quarterly (?:results?|profit|loss|revenue|sales)|"
    r"(?:q[1-4]|first|second|third|fourth)[- ]quarter (?:results?|profit|loss|revenue|sales|"
    r"earnings)|net (?:income|profit|loss)|eps|revenue (?:rose|fell|grew|jumped|dropped|of)|"
    r"(?:beats?|misse[sd]|miss|tops?|topped) (?:estimates|expectations|forecasts)|guidance|"
    r"dividends?|full[- ]year (?:results?|profit|outlook))\b",
    re.IGNORECASE,
)
EARNINGS_PREVIEW = re.compile(
    r"\b(?:ahead of|before|preview|previews|what to (?:watch|expect)|expect(?:s|ed)?)\b"
    r".{0,30}\b(?:earnings|results)\b",
    re.IGNORECASE,
)
PRODUCT = re.compile(
    r"\b(?:launch(?:es|ed|ing)?|unveil(?:s|ed|ing)?|(?<!press )releas(?:e|es|ed|ing)|"
    r"recall(?:s|ed|ing)?|rolls? out|rolled out|debut(?:s|ed)?|"
    r"supply (?:deal|agreement|contract))\b",
    re.IGNORECASE,
)


def earnings_rule(text: str) -> bool:
    return bool(EARNINGS.search(text)) and not EARNINGS_PREVIEW.search(text)


def product_rule(text: str) -> bool:
    return bool(PRODUCT.search(text))
