"""Opt-in live catalog. Original Linker and batch paths are unchanged (E38)."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from riskengine.ingest.linking import Linker, _analyst_pattern, _desk_pattern
from riskengine.paths import REFERENCE

EXTENDED_PATH = REFERENCE / "universe_extended.csv"
CONTEXT = re.compile(
    r"\b(?:shares?|stocks?|earnings|revenues?|profits?|losses?|dividends?|quarter\w*|fiscal|"
    r"buybacks?|securities|bonds?|debt|credit|analysts?|ratings?|upgrad\w*|downgrad\w*|"
    r"acquir\w*|acquisition\w*|merger\w*|ceo|cfo|company|corporation|incorporated|payments?|"
    r"cards?|cloud|software|vehicles?|trucks?|automaker\w*|retail\w*|stores?|banks?|banking|"
    r"deposits?|machinery|equipment|iphones?|ipads?|macbooks?)\b",
    re.IGNORECASE,
)
SUFFIX = re.compile(
    r"^\s+(?:Inc(?:orporated)?|Corp(?:oration)?|Co(?:mpany)?|Ltd|Limited|PLC)\b\.?",
    re.IGNORECASE,
)
BLOCKS = {
    "AAPL": r"apple\s+(?:harvest|orchard|fruit|tree|pie|cider|juice)",
    "AMZN": r"amazon\s+(?:river|rainforest|jungle|basin)",
    "TGT": r"target\s+(?:price|practice|audience|goal|market)",
    "V": r"visa[ -]+(?:application|applicant|immigration|tourist|travel|work permit|free)",
    "F": r"ford\s+(?:(?:the|a)\s+)?(?:river|stream|creek)|river crossing",
    "ORCL": r"oracle\s+of\s+omaha|ancient oracle|mythical oracle",
    "COF": r"(?!)",
    "CAT": r"caterpillar\s+(?:larvae|larva|insect|butterfly|cocoon)",
    "DOW": r"dow\s+(?:jones|industrial|rises|falls|gains|drops|climbs|slides|jumps)",
}
LOOKALIKES = {tk: re.compile(rf"\b(?:{pat})s?\b", re.I) for tk, pat in BLOCKS.items()}
SHORT = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "TGT": "Target",
    "V": "Visa",
    "F": "Ford",
    "ORCL": "Oracle",
    "COF": "Capital One",
    "CAT": "Caterpillar",
    "DOW": "Dow",
}

# Candidate 2/3: research-firm names, not new company aliases/catalog entries.
RESEARCH_FIRMS = {
    "JPM": r"JPMorgan|JP Morgan|J\.P\. Morgan|J\. P\. Morgan",
    "BAC": r"Bank of America(?: Merrill Lynch)?|BofA(?:/Merrill)?",
    "GS": r"Goldman Sachs(?: Group)?",
    "MS": r"Morgan Stanley",
    "C": r"Citigroup|Citi",
    "WFC": r"Wells Fargo",
    "USB": r"U\.S\. Bancorp|US Bancorp|U\.S\. Bank|US Bank",
    "Barclays": r"Barclays",
    "BMO": r"BMO(?: Capital(?: Markets)?)?|Bank of Montreal",
    "Mizuho": r"Mizuho",
    "Oppenheimer": r"Oppenheimer",
    "RBC": r"RBC(?: Capital(?: Markets)?)?|Royal Bank of Canada",
    "UBS": r"UBS",
    "Credit Suisse": r"Credit Suisse",
    "Deutsche Bank": r"Deutsche Bank",
}
FIRM_END = (
    r"(?:\s+(?:Chase\s*&\s*Co\.?|Group|Inc\.?|Corp\.?|Corporation|Securities|"
    r"Global Research|Capital Markets))*"
)
RESEARCH_ACTION = re.compile(
    r"^\s*(?:[:,-]\s*)?(?:(?:Earlier|Previously)\s+)?"
    r"(?P<verb>maintain(?:s|ed)?|reiterate[sd]?|reinstate[sd]?|initiate[sd]?|"
    r"upgrade[sd]?|downgrade[sd]?|raise[sd]?|lower(?:s|ed)?|cuts?|cut|"
    r"comment(?:s|ed)?(?:\s+on)?|review(?:s|ed)?|sees?|sa(?:ys|id)|"
    r"reports?(?:\s+on)?|adjust(?:s|ed)?|gives?|gave)\b",
    re.I,
)
PASSIVE_RATING = re.compile(
    r"\b(?:upgrade[sd]?|downgrade[sd]?|cut|maintain(?:s|ed)?|rate[sd]?|"
    r"reiterate[sd]?|initiate[sd]?)\b[^;.!?]{0,120}\b(?:at|by)\s*$",
    re.I,
)
OWN_TOPIC = re.compile(
    r"^\s*[\"'‘’“”]*(?:(?:its|our|the bank's|record)\s+)?"
    r"(?:earnings|profits?|revenues?|customers?|leases?|hack\w*|probe|"
    r"dividends?|jobs|workforce|staff|costs|rate cuts?)\b",
    re.I,
)
RATING_DETAIL = re.compile(
    r"\b(?:price\s+target|target\s+price|PT|PO|price\s+objective|ratings?|"
    r"upgrad\w*|downgrad\w*|overweight|underweight|outperform|underperform|"
    r"neutral|buy|sell|hold)\b",
    re.I,
)
PRICE_LANGUAGE = re.compile(
    r"\b(?:price\s+target|target\s+price|PT|PO|price\s+objective|"
    r"target\s+of\s*\$\s*[\d,.]+)\b",
    re.I,
)
RATING_VERB = re.compile(
    r"\b(?:upgrade[sd]?|downgrade[sd]?|initiate[sd]?|maintain(?:s|ed)?)\b", re.I
)
RATING_FORM_V3 = re.compile(
    r"\b(?:ratings?|overweight|underweight|outperform|underperform|neutral|"
    r"price\s+target|target\s+price|PT|PO|price\s+objective|"
    r"(?:to|at)\s+(?:buy|sell|hold)|(?:buy|sell|hold)\s+ratings?)\b",
    re.I,
)
QUOTE_VERB = re.compile(
    r"^(?:says?|said|sees?|comments?(?:\s+on)?|reviews?|reports?(?:\s+on)?|gives?)$", re.I
)
RESEARCH_TOPIC = re.compile(r"\b(?:forecasts?|outlooks?|estimates?|default activity)\b", re.I)
LEGAL_DEFENCE = re.compile(
    r"\b(?:reject\w*|den(?:y|ies|ied)|defend\w*)\b.*\b(?:allegations?|assertions?|claims?|accusations?)\b",
    re.I,
)
REPORTED_OWN_ACTION = re.compile(r"^\s+to\s+(?:keep|close|sell|buy|hire|end|seek|open)\b", re.I)
EXTERNAL_LEGAL_NAME = re.compile(
    r"\b[A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*){0,4}\s+(?:Inc|Corp|Corporation|Co|Company|Ltd|PLC)\b"
)
OTHER_LEGAL_NAME = re.compile(
    r"^\s+(?:[A-Z][\w&'-]*\s+){1,4}(?:Inc|Corp|Corporation|Co|Company|Ltd|PLC)\b"
)
RELATION = re.compile(r"^\s+(?:suppliers?|customers?|partners?)(?:['’]s)?\b", re.I)
INSOLVENCY = re.compile(r"\b(?:bankrupt\w*|defaults?)\b", re.I)
LIST_SEPARATOR = re.compile(r"\s*(?:,|;|and|or|&|/)\s*(?:and\s+)?", re.I)


@dataclass(frozen=True)
class ExtendedLinker(Linker):
    index_members: frozenset[str] = field(default_factory=frozenset)
    candidate: int = 1
    research_patterns: dict[str, re.Pattern] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: Path = EXTENDED_PATH, candidate: int = 1) -> "ExtendedLinker":
        if candidate not in (1, 2, 3):
            raise ValueError("extended linker candidate must be 1, 2 or 3")
        base = Linker.from_csv(path)
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        if table.ticker.duplicated().any() or not set(table.in_index) <= {"true", "false"}:
            raise ValueError("invalid extended catalog")
        analysts, desks = dict(base.analyst_patterns), dict(base.desk_patterns)
        for row in table[table.ticker.isin(["GS", "MS", "C", "WFC", "USB"])].itertuples():
            names = "|".join(map(re.escape, row.aliases.split("|")))
            analysts[row.ticker], desks[row.ticker] = _analyst_pattern(names), _desk_pattern(names)
        return cls(
            base.cashtag_to_ticker,
            base.alias_patterns,
            analysts,
            desks,
            base.company_names,
            frozenset(table.loc[table.in_index == "true", "ticker"]),
            candidate,
            {
                key: re.compile(rf"\b(?:{names})\b{FIRM_END}")
                for key, names in RESEARCH_FIRMS.items()
            }
            if candidate in (2, 3)
            else {},
        )

    def link(self, text: str, use_names: bool = True) -> tuple[str, ...]:
        found = set(super().link(text, use_names=False))
        listed = self._company_list(text) if self.candidate == 3 and use_names else set()
        if use_names:
            for ticker, pattern in self.alias_patterns.items():
                for match in pattern.finditer(text):
                    if (
                        self.candidate in (2, 3)
                        and ticker == "TGT"
                        and any(
                            phrase.start() <= match.start() and match.end() <= phrase.end()
                            for phrase in PRICE_LANGUAGE.finditer(text)
                        )
                    ):
                        continue
                    if self.candidate == 3:
                        after = text[match.end() :]
                        relation = RELATION.search(after)
                        if relation and INSOLVENCY.search(after[relation.end() :].split(";", 1)[0]):
                            continue
                        if (
                            ticker in self.index_members
                            and ticker in SHORT
                            and OTHER_LEGAL_NAME.search(after)
                            and not (ticker == "AMZN" and re.match(r"\s+Web Services\b", after))
                        ):
                            continue
                    short = SHORT.get(ticker)
                    if short is None or match.group() != short:
                        found.add(ticker)
                        break
                    if SUFFIX.search(text[match.end() :]):
                        found.add(ticker)
                        break
                    window = text[max(0, match.start() - 80) : match.end() + 80]
                    # Check phrases touching this occurrence, not unrelated distant mentions.
                    blocked = any(
                        b.start() < match.end() and b.end() > match.start()
                        for b in LOOKALIKES[ticker].finditer(text)
                    )
                    if not blocked and (
                        CONTEXT.search(window)
                        or self.candidate == 3
                        and (ticker in self.index_members or ticker in listed)
                    ):
                        found.add(ticker)
                        break
        return tuple(sorted(found))

    def _company_list(self, text: str) -> set[str]:
        spans = sorted(
            (m.start(), m.end(), tk)
            for tk, pat in self.alias_patterns.items()
            for m in pat.finditer(text)
        )
        found = set()
        for left, right in zip(spans, spans[1:], strict=False):
            if (
                left[2] != right[2]
                and left[1] <= right[0]
                and LIST_SEPARATOR.fullmatch(text[left[1] : right[0]])
            ):
                found.update((left[2], right[2]))
        return found

    def research_attribution(self, text: str) -> set[str]:
        """Research desks even when their target is outside this company's catalog."""
        if self.candidate == 3:
            return self._research_attribution_v3(text)
        found = set()
        for firm, pattern in self.research_patterns.items():
            for match in pattern.finditer(text):
                tail = text[match.end() :]
                action = RESEARCH_ACTION.search(tail)
                if action:
                    rest = tail[action.end() :]
                    # An actual change to the bank's own business is not a stock rating.
                    if OWN_TOPIC.search(rest) and not RATING_DETAIL.search(rest):
                        continue
                    found.add(firm)
                elif PASSIVE_RATING.search(text[max(0, match.start() - 160) : match.start()]):
                    found.add(firm)
        return found

    def _research_attribution_v3(self, text: str) -> set[str]:
        found = set()
        for firm, pattern in self.research_patterns.items():
            for match in pattern.finditer(text):
                tail = text[match.end() :]
                action = RESEARCH_ACTION.search(tail)
                before = text[max(0, match.start() - 160) : match.start()]
                passive = PASSIVE_RATING.search(before) or (
                    re.search(r"\b(?:at|by)\s*$", before, re.I) and RATING_FORM_V3.search(before)
                )
                if passive:
                    found.add(firm)
                    continue
                if not action:
                    # Explicit desk roles quoting another named company remain attribution.
                    role = re.match(r"\s+(?:analysts?|strategists?|economists?)\b", tail, re.I)
                    if role and (set(Linker.link(self, tail)) - {firm}):
                        found.add(firm)
                    continue
                rest = tail[action.end() :]
                # A rating/target form can have a target outside the 99-company catalog.
                rating = RATING_FORM_V3.search(rest) or RATING_VERB.fullmatch(action.group("verb"))
                if rating and rest.strip():
                    found.add(firm)
                    continue
                if (
                    LEGAL_DEFENCE.search(rest)
                    or REPORTED_OWN_ACTION.search(rest)
                    or OWN_TOPIC.search(rest)
                ):
                    continue
                external = set(Linker.link(self, rest)) - {firm}
                if RESEARCH_TOPIC.search(rest) or (
                    QUOTE_VERB.fullmatch(action.group("verb"))
                    and (external or EXTERNAL_LEGAL_NAME.search(rest))
                ):
                    found.add(firm)
        return found

    def analyst_only(self, text: str) -> set[str]:
        if self.candidate == 3:
            own = {self.cashtag_to_ticker.get(tag) for tag in self.cashtags_in(text)}
            return (self.research_attribution(text) & set(self.tickers)) - own
        found = super().analyst_only(text)
        if self.candidate == 2:
            own = {self.cashtag_to_ticker.get(tag) for tag in self.cashtags_in(text)}
            found |= (self.research_attribution(text) & set(self.tickers)) - own
        return found
