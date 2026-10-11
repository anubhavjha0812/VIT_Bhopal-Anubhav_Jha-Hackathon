"""Is a headline about the company it names, or does it only mention it? (DECISIONS E17)

Headlines filed in FNSPID under another symbol are linked to our companies by name. About half
only mention the company (a fund's buy list, a conference a bank hosts, a bank rating another
stock, a fund notice). `reason()` returns the first rule a (headline, company) pair fails, or
None if it is kept. The rules were written from one sample of 100 such headlines (seed 7) and
frozen before the sample that measures them was drawn.
"""

import re

from riskengine.ingest.linking import Linker

BANKS = {"JPM", "BAC"}
LEAD_WORDS = 6  # the company must be named within the first 6 words (after a wire prefix)

# Another company or thing that carries our company's name.
VETO = {
    "AAPL": r"Apple (?:Hospitality|REIT|Leisure|Bank|Daily|pie)|Big Apple",
    "AMD": r"macular degeneration",
    "AMZN": r"Amazon (?:rainforest|River|basin)|Amazonas",
    "DIS": r"Euro Disney|Disneyland Paris",
    "GOOGL": r"Google Ventures",
    "KO": (
        r"Coca-?Cola (?:Bottling|Amatil|Hellenic|FEMSA|Icecek|Enterprises|Consolidated|West|"
        r"European Partners|Bottlers|Central Japan|East Japan)|"
        r"(?:Mikuni|Swire|Hokkaido|Shikoku|Kita Kyushu|Okinawa|Arca|Andina|Embonor) Coca-?Cola"
    ),
    "TSLA": r"Nikola Tesla|Tesla coil|Tesla Exploration",
}
_VETO = {tk: re.compile(p, re.IGNORECASE) for tk, p in VETO.items()}

NOTICE = re.compile(r"^\s*REG\s*-|net asset value|\bform 8\.5\b|\bPDMR\b", re.IGNORECASE)
# a fund's buy/sell list: "Marsico Capital Buys Walt Disney Co, Celgene, ... Sells ..."
HOLDINGS = re.compile(
    r"\b(?:Buys|Sells)\b[^,]+,[^,]+,|\bBuys\b.*\bSells\b|\bSells\b.*\bBuys\b|\b13F\b|"
    r"Acquires Stake|Still Bullish On|"
    r"\b(?:Associates|Capital|Management|Advisors|Advisers|LLC|Llc|Partners|Fund|Investments|"
    r"Wealth)\b[^,]*\b(?:Buys|Sells)\b"
)
CONFERENCE = re.compile(r"conference|to present at|to participate in|fireside chat", re.I)
# a bank's research desk on another company: "Bank of America Maintains Salix at Buy",
# "J.P. Morgan Adjusts PRE 1Q Estimates", "... shares attractive ... -BofA"
_BANK = r"(?:JPMorgan|JP Morgan|J\.P\. Morgan|Bank of America|BofA)"
BANK_DESK = re.compile(
    rf"^(?:UPDATE:\s*)?{_BANK}(?:\s+Merrill Lynch|\s+Securities|\s+Chase(?:\s*&\s*Co\.?)?)?\s+"
    r"(?:adjusts|maintains|reiterates|raises|lowers|cuts|initiates|upgrades|downgrades|"
    r"resumes|assumes|starts|sets|boosts|trims)\b.*"
    r"(?:estimate|\bPO\b|\bPT\b|price (?:target|objective)|rating|overweight|underweight|"
    r"\bbuy\b|\bsell\b|\bhold\b|neutral|outperform|underperform)"
    rf"|[-–—]\s*{_BANK}\s*$"
    rf"|^{_BANK}['’]s\s+(?:stock|top)\s+picks",
    re.IGNORECASE,
)
# a list of several stocks: "Auto Stock Roundup: GM, Honda, Tesla And More"
MARKET_LIST = re.compile(
    r"roundup|highlights|highlighted as|bull and bear|stocks to watch|winners (?:&|and) losers|"
    r"after-hours trading|pre-?market (?:movers|prep)|mid-day movers",
    re.IGNORECASE,
)
MOJIBAKE = re.compile(r"[ÃÂ]")  # UTF-8 read as Latin-1, nearly always a non-English headline
_PREFIX = re.compile(
    r"^(?:\W*(?:UPDATE\s*\d*|BRIEF|EXCLUSIVE|RPT|WRAPUP\s*\d*|CORRECTED|REFILE|"
    r"STXNEWS\s+\w+|BUZZ|PRESS DIGEST[^-:]*)\s*[-:])+\s*",
    re.IGNORECASE,
)


def non_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    share = sum(not c.isascii() for c in letters) / len(letters) if letters else 0.0
    return share > 0.1 or bool(MOJIBAKE.search(text))


def former_staff(linker: Linker, text: str, ticker: str) -> bool:
    """Named only as someone's past employer: "INRIX Appoints Former Microsoft CFO ..."."""
    return any(
        re.search(r"\b(?:former|ex-)\s*$", text[: m.start()], re.IGNORECASE)
        for m in linker.alias_patterns[ticker].finditer(text)
    )


def named_early(linker: Linker, text: str, ticker: str) -> bool:
    """The company's name or cashtag appears within the first LEAD_WORDS words."""
    head = " ".join(_PREFIX.sub("", text).split()[:LEAD_WORDS])
    return ticker in linker.link(head)


def reason(linker: Linker, text: str, ticker: str, by_symbol: bool = False) -> str | None:
    """The first rule the pair fails, or None. `by_symbol`: FNSPID filed it under this company
    (then only the language rule applies; those links come from the source, E17)."""
    if non_english(text):
        return "not English"
    if by_symbol:
        return None
    if ticker in _VETO and _VETO[ticker].search(text):
        return "another company's name"
    if NOTICE.search(text):
        return "fund notice"
    if HOLDINGS.search(text):
        return "holdings list"
    if MARKET_LIST.search(text):
        return "market list"
    if former_staff(linker, text, ticker):
        return "former employer"
    if ticker in BANKS:
        if CONFERENCE.search(text):
            return "bank conference"
        if ticker in linker.analyst_only(text) or BANK_DESK.search(text):
            return "bank as analyst"
    if not named_early(linker, text, ticker):
        return "named late"
    return None
