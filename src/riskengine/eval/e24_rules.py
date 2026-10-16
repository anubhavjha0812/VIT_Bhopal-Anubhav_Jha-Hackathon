"""D1-designed E24 evaluation rules; production candidate A is unchanged.

The fund-holding and analyst routing amendment is recorded in docs/E24_AMENDMENT.md.
Only D1 examples informed these patterns. No company names or row identifiers are matched.
"""

import re

from riskengine.nlp.events import is_analyst_rating, names_agency

EARNINGS = re.compile(
    r"\b(?:earnings|quarterly (?:results?|profit|loss|revenue|sales)|"
    r"(?:q[1-4]|first|second|third|fourth)[- ]quarter (?:results?|profit|loss|revenue|sales|"
    r"earnings)|net (?:income|profit|loss)|eps|revenue (?:rose|fell|grew|jumped|dropped|of)|"
    r"(?:beats?|misse[sd]|miss|tops?|topped) (?:estimates|expectations|forecasts)|guidance|"
    r"dividends?|operating margin|full[- ]year (?:results?|profit|outlook))\b",
    re.IGNORECASE,
)
EARNINGS_PREVIEW = re.compile(
    r"\b(?:ahead of|before|preview\w*|what['’]?s in store|what is in store|"
    r"what to (?:watch|expect))\b.{0,100}\b(?:earnings|results)\b"
    r"|\b(?:earnings|results)\b.{0,35}\b(?:before|ahead of)\b"
    r"|\bearnings (?:season|calendar|charts)\b",
    re.IGNORECASE,
)
ACTUAL_RESULTS = re.compile(
    r"\b(?:eps|margin|revenue|sales|earnings|profit|loss)\b.{0,24}"
    r"(?:\$\s*\d|\d[\d,.]*\s*(?:%|billion|million))"
    r"|\b(?:posts?|reports?|reported)\b.{0,35}\b(?:earnings|profit|loss|results)\b"
    r"|\bearnings (?:beat|miss)\b",
    re.IGNORECASE,
)
OPERATING_GUIDANCE = re.compile(
    r"\b(?:ceo|cfo|chief executive|chief financial officer)\b.{0,80}"
    r"\b(?:expects?|forecast\w*|guidance|trough|bottom)\b.{0,50}"
    r"\b(?:q[1-4]|(?:first|second|third|fourth) quarter)\b",
    re.IGNORECASE,
)
PRODUCT = re.compile(
    r"\b(?:launch(?:es|ed|ing)?|unveil(?:s|ed|ing)?|(?<!press )releas(?:e|es|ed|ing)|"
    r"recall(?:s|ed|ing)?|rolls? out|rolled out|debut(?:s|ed)?|"
    r"supply (?:deal|agreement|contract))\b",
    re.IGNORECASE,
)
NONPRODUCT_OBJECT = re.compile(
    r"^\W+(?:\w+\W+){0,3}(?:production|manufacturing|factory|plant|investigation|"
    r"probe|lawsuit|campaign|ipo|bond|debt|earnings|results|report|statement|"
    r"prison|custody|conference|hiring)\b",
    re.IGNORECASE,
)
SPACE_FLIGHT = re.compile(
    r"\b(?:rocket|spacecraft|spaceship|mission|orbital|astronaut|crew|satellite)\b",
    re.IGNORECASE,
)
PRODUCT_WORD = r"(?:products?|devices?|phones?|tablets?|laptops?|chips?|gpus?|vehicles?|services?)"
NEW_PRODUCT = re.compile(rf"\bnew\b.{{0,60}}\b{PRODUCT_WORD}\b", re.IGNORECASE)
ANNOUNCEMENT = re.compile(r"\b(?:announc\w*|introduc\w*|expands?|available|shipping)\b", re.I)
PRODUCT_DEAL = re.compile(rf"\bto (?:sell|supply|buy)\b.{{0,70}}\b{PRODUCT_WORD}\b", re.IGNORECASE)
USER_TRIAL = re.compile(
    r"\b(?:beta|testing|tests?|piloting)\b.{0,45}\b(?:platform|app|service|software)\b"
    r".{0,35}\b(?:to let|for (?:users|customers|merchants)|available)\b",
    re.IGNORECASE,
)
HOLDING_ACTION = re.compile(r"\b(?:buys|sells)\b", re.IGNORECASE)
INVESTOR = re.compile(
    r"\b(?:capital|wealth|management|investment|advisors?|advisory|portfolio|fund|"
    r"partners|office|llc|ltd|inc|bank)\b",
    re.IGNORECASE,
)
DEAL_TERMS = re.compile(
    r"\b(?:acquir\w*|acquisition|merger|takeover|buyout|agreement|stake in|to buy)\b"
    r"|\bfor\s+\$",
    re.IGNORECASE,
)
ROUNDUP = re.compile(
    r"\bupgrades?\s*,?\s*(?:and\s+)?downgrades?\b|\bprice[- ]target changes?\b",
    re.IGNORECASE,
)
LEADING_RECOMMENDATION = re.compile(
    r"^(?:update:\s*)?[^:;.!?]{2,90}\b(?:upgrades?|downgrades?|initiates? coverage)\b",
    re.IGNORECASE,
)
REITERATION = re.compile(r"\b(?:maintains?|reiterat\w*|reaffirm\w*)\b", re.IGNORECASE)
RATING_CHANGE = re.compile(
    r"\b(?:upgrad\w*|downgrad\w*|initiat\w*)\b"
    r"|\b(?:raise[sd]?|cuts?|lower[sed]*|lift[sed]*|boost\w*)\b.{0,30}"
    r"\b(?:price targets?|targets?|tgts?|ratings?)\b",
    re.IGNORECASE,
)


def _stock_move(text: str) -> bool:
    """A stock move leads the headline; later events supply the explanation."""
    lead = re.split(
        r"\b(?:after|amid|following|despite|because|due to)\b|[;:]",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    return bool(
        re.search(
            r"\b(?:shares?|stocks?|sellers?|equities)\b.{0,65}"
            r"\b(?:trading|higher|lower|rise|fall|jump\w*|surge\w*|dip|rebound|slump|"
            r"gain\w*|rally|recover\w*|up|down|unaffected)\b",
            lead,
            re.I,
        )
    )


def holdings_rule(text: str) -> bool:
    """An investor's comma-separated holding changes are not corporate acquisitions."""
    action = HOLDING_ACTION.search(text)
    if not action or DEAL_TERMS.search(text):
        return False
    holder, assets = text[: action.start()], text[action.end() :]
    truncated_instrument = bool(
        re.search(r"\.{3}|…", assets)
        and re.search(r"\b(?:etfs?|funds?|index|bonds?|treasury|s&p)\b", assets, re.I)
    )
    return bool(INVESTOR.search(holder) and ("," in assets or truncated_instrument))


def analyst_rule(text: str) -> str | None:
    """Route rating changes; an unchanged-only reiteration stays Other under the guide."""
    if _stock_move(text) or names_agency(text):
        return None
    detected = is_analyst_rating(text)
    if not (detected or ROUNDUP.search(text) or LEADING_RECOMMENDATION.search(text)):
        return None
    if detected and REITERATION.search(text) and not RATING_CHANGE.search(text):
        return "Other"
    return "Analyst rating"


def earnings_preview_rule(text: str) -> bool:
    """Unquantified previews and earnings-season commentary are not actual results."""
    return bool(EARNINGS_PREVIEW.search(text)) and not bool(ACTUAL_RESULTS.search(text))


def earnings_rule(text: str) -> bool:
    return (
        not _stock_move(text)
        and not earnings_preview_rule(text)
        and bool(EARNINGS.search(text) or OPERATING_GUIDANCE.search(text))
    )


def product_rule(text: str) -> bool:
    if _stock_move(text):
        return False
    if PRODUCT_DEAL.search(text) or USER_TRIAL.search(text):
        return True
    if NEW_PRODUCT.search(text) and ANNOUNCEMENT.search(text):
        return True
    for action in PRODUCT.finditer(text):
        if NONPRODUCT_OBJECT.search(text[action.end() :]):
            continue
        if action.group().lower().startswith("launch") and SPACE_FLIGHT.search(text):
            continue
        return True
    return False
