"""Precision rules for the credit gate (docs/experiments/E36_GATE_PRECISION.md).

Two category rules that only remove alerts the gate has already fired:
1. securitization: a rating of a deal or a servicer, not of the company's own debt
2. affirmation: an affirmation, upgrade or stable/positive outlook is not a downgrade
"""

import re

_DEAL_ID = r"\b(?:19|20)\d{2}-(?=[A-Za-z0-9]{0,7}[A-Za-z])[A-Za-z0-9]{1,8}\b"
_SERIES = r"\bSeries\s+(?:19|20)\d{2}\b"
_PHRASES = (
    r"\b(?:Commercial Mortgage|Mortgage Trust|Mortgage Finance|Mortgage Securities|Pass-Through"
    r"|Asset-Backed|Collateralized)\b|\b(?:Tranche|Servicer|Servicing)s?\b"
)
SECURITIZATION = re.compile(f"{_DEAL_ID}|{_SERIES}|{_PHRASES}", re.IGNORECASE)
ACRONYMS = re.compile(r"\b(?:CMBS|RMBS|CLO|ABS)\b")  # capital letters only
CORPORATE_RATING = re.compile(
    r"\bIDR\b|issuer default rating|senior unsecured|long-term rating", re.IGNORECASE
)

NOT_ADVERSE = re.compile(
    r"\baffirm(?:s|ed|ing)?\b|\bupgrad(?:e|es|ed|ing)\b"
    r"|\boutlook\b[^.;]*\b(?:stable|positive)\b|\bremoved from (?:watch|review)\b",
    re.IGNORECASE,
)
ADVERSE = re.compile(  # incl. an outlook revised to negative (E36b)
    r"\boutlook\s+(?:is\s+)?(?:revised|lowered|changed|cut)\s+to\s+negative\b"
    r"|\b(?:revises|lowers|changes)\s+outlook\s+to\s+negative\b"
    r"|\bdowngrad(?:e|es|ed|ing)\b|\bcuts?\b|\blower(?:s|ed|ing)?\b|\bslash(?:es|ed)?\b"
    r"|\bwithdraw(?:s|n)?\b",
    re.IGNORECASE,
)


def is_securitization(text: str) -> bool:
    hit = SECURITIZATION.search(text) or ACRONYMS.search(text)
    return bool(hit) and not CORPORATE_RATING.search(text)


def is_not_adverse(text: str, event: str) -> bool:
    return (
        event == "Credit downgrade" and bool(NOT_ADVERSE.search(text)) and not ADVERSE.search(text)
    )


def blocked_reason(text: str, event: str) -> str | None:
    """Name of the first precision rule that blocks this alert, or None."""
    if is_securitization(text):
        return "securitization"
    if is_not_adverse(text, event):
        return "affirmation or upgrade"
    return None
