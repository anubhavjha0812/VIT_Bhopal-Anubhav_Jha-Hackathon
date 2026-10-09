"""Real inputs for the Module B book and shocks (E14), fetched once and stored small in the repo.

uv run python -m riskengine.module_b.sources            # sec_debt.csv + market_history.csv
uv run python -m riskengine.module_b.sources --tenk     # credit_line_candidates.csv (read by hand)

- SEC EDGAR XBRL: each company's debt at its last filing on or before the book date, with the tag
  and accession of every number. Free, no key; SEC asks for a User-Agent (env SEC_USER_AGENT).
- 10-K text: sentences naming a revolving credit facility and an amount, for companies whose
  credit line has no XBRL tag. Candidates only; the chosen line goes into sec_credit_lines.csv by
  hand, with its sentence.
- ratings.csv is by hand: S&P rating at the book date, with the evidence and a source link
  (bond pricing term sheets and 10-Ks on EDGAR first, else news of the next rating action).
- FRED (no key): Moody's Baa and Aaa spreads over the 10-year Treasury, VIX, USD/INR; plus the
  S&P 500 from yfinance. Daily, 1990 on, for the shocks and the crisis backtest.
"""

import argparse
import html
import io
import json
import os
import re
import time

import httpx
import pandas as pd

from riskengine.ingest.prices import MARKET, fetch_prices
from riskengine.paths import RAW, REFERENCE

BOOK_DATE = "2019-12-31"
OLDEST = "2018-12-31"  # a filing older than one year before the book date is not used
SEC_DEBT_PATH = REFERENCE / "sec_debt.csv"
CANDIDATES_PATH = REFERENCE / "credit_line_candidates.csv"
MARKET_PATH = REFERENCE / "market_history.csv"
SEC_RAW = RAW / "sec"
USER_AGENT = os.environ.get("SEC_USER_AGENT", "riskengine-hackathon research admin@example.com")

UNIVERSE_PATH = REFERENCE / "universe.csv"
# Exxon moved to a new CIK in 2025; its 2019 filings are under the old one.
CIK_OVERRIDE = {"XOM": 34088}

# Tags are tried in order; the first with a value wins. Companies tag the same thing differently.
SHORT = ["DebtCurrent"]  # else current long-term debt + commercial paper / short-term borrowings
CURRENT_LTD = ["LongTermDebtCurrent"]
PAPER = ["CommercialPaper", "ShortTermBorrowings", "OtherShortTermBorrowings"]
LONG = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"]
LONG_INCL_CURRENT = [
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
]
TOTAL = ["DebtAndCapitalLeaseObligations", "DebtLongtermAndShorttermCombinedAmount"]
CREDIT_LINE_TAGS = [
    "LineOfCreditFacilityMaximumBorrowingCapacity",
    "LineOfCreditFacilityRemainingBorrowingCapacity",
    "DebtInstrumentUnusedBorrowingCapacityAmount",
]
_REPAY = "LongTermDebtMaturitiesRepaymentsOfPrincipal"
MATURITIES = {
    "due_y1": f"{_REPAY}InNextTwelveMonths", "due_y2": f"{_REPAY}InYearTwo",
    "due_y3": f"{_REPAY}InYearThree", "due_y4": f"{_REPAY}InYearFour",
    "due_y5": f"{_REPAY}InYearFive", "due_after_y5": f"{_REPAY}AfterYearFive",
}  # fmt: skip
FRED_SERIES = {
    "BAA10Y": "baa", "AAA10Y": "aaa", "DGS10": "ust10y", "VIXCLS": "vix", "DEXINUS": "usdinr",
}  # fmt: skip
MARKET_START, MARKET_END = "1990-01-01", "2024-01-01"


def _get(url: str) -> bytes:
    time.sleep(0.2)  # SEC allows 10 requests a second
    r = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.content


def universe() -> pd.DataFrame:
    """Our companies (ticker, name, sector); banks (Financials) are lenders, not borrowers."""
    return pd.read_csv(UNIVERSE_PATH)


def cik(ticker: str) -> int:
    """SEC company ID, from SEC's ticker list (cached), so a new ticker needs no code change."""
    if ticker in CIK_OVERRIDE:
        return CIK_OVERRIDE[ticker]
    path = SEC_RAW / "company_tickers.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get("https://www.sec.gov/files/company_tickers.json"))
    table = {r["ticker"]: r["cik_str"] for r in json.loads(path.read_text()).values()}
    return table[ticker]


def company_facts(ticker: str) -> dict:
    """All XBRL facts of one company (cached raw, not committed)."""
    path = SEC_RAW / f"{ticker}_facts.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik(ticker):010d}.json"
        path.write_bytes(_get(url))
    return json.loads(path.read_text())


def latest_fact(facts: dict, tags: list[str]) -> dict | None:
    """Newest USD value dated in (OLDEST, BOOK_DATE], as first reported (no later restatement)."""
    gaap = facts["facts"].get("us-gaap", {})
    for tag in tags:
        rows = [
            x
            for x in gaap.get(tag, {}).get("units", {}).get("USD", [])
            if OLDEST < x["end"] <= BOOK_DATE and x["form"] in ("10-K", "10-Q")
        ]
        if rows:
            end = max(r["end"] for r in rows)
            x = min((r for r in rows if r["end"] == end), key=lambda r: r["filed"])
            return {"value": x["val"], "tag": tag, "end": x["end"], "form": x["form"],
                    "accn": x["accn"]}  # fmt: skip
    return None


def _src(hit: dict) -> str:
    return f"{hit['tag']} {hit['end']} {hit['form']} {hit['accn']}"


def debt_figures(facts: dict) -> dict:
    """Short-term, long-term and total debt, credit line and repayment schedule, each with a source.

    short = DebtCurrent, else current long-term debt + commercial paper (or short-term borrowings);
    long = non-current long-term debt, else long-term debt incl. current minus its current part;
    total = the combined tag, else short + long.
    """
    row: dict = {}
    if hit := latest_fact(facts, SHORT):
        row |= {"short": hit["value"], "short_src": _src(hit)}
    else:
        parts = [h for tags in (CURRENT_LTD, PAPER) if (h := latest_fact(facts, tags))]
        if parts:
            row |= {"short": sum(h["value"] for h in parts),
                    "short_src": " + ".join(_src(h) for h in parts)}  # fmt: skip
    if hit := latest_fact(facts, LONG):
        row |= {"long": hit["value"], "long_src": _src(hit)}
    elif hit := latest_fact(facts, LONG_INCL_CURRENT):
        current = latest_fact(facts, CURRENT_LTD) or latest_fact(facts, SHORT)
        row |= {"long": hit["value"] - (current["value"] if current else 0),
                "long_src": _src(hit) + (f" - {_src(current)}" if current else "")}  # fmt: skip
    if hit := latest_fact(facts, TOTAL):
        row |= {"total": hit["value"], "total_src": _src(hit)}
    elif "short" in row or "long" in row:
        row |= {"total": row.get("short", 0) + row.get("long", 0), "total_src": "short + long"}
    if hit := latest_fact(facts, CREDIT_LINE_TAGS):
        row |= {"credit_line": hit["value"], "credit_line_src": _src(hit)}
    due = {k: latest_fact(facts, [tag]) for k, tag in MATURITIES.items()}
    row |= {k: h["value"] for k, h in due.items() if h}
    if any(due.values()):
        row["due_src"] = _src(next(h for h in due.values() if h))
    return row


def sec_debt() -> pd.DataFrame:
    """One row per company: debt figures in USD, each with its tag, date, form and accession."""
    out = [
        {"ticker": t, "entity": (f := company_facts(t))["entityName"], **debt_figures(f)}
        for t in universe()["ticker"]
    ]
    cols = ["ticker", "entity", "short", "long", "total", "credit_line", *MATURITIES]
    df = pd.DataFrame(out).reindex(columns=[*cols, "short_src", "long_src", "total_src",
                                            "credit_line_src", "due_src"])  # fmt: skip
    return df


# --- 10-K text: credit-line sentences ------------------------------------------------------------

_MONEY = r"\$\s?\d[\d,.]*\s*(?:billion|million)"
_FACILITY = r"\b(?:revolving|credit facilit\w*|credit agreements?|lines? of credit)\b"
CREDIT_LINE = re.compile(
    rf"[^.]*{_FACILITY}[^.]*{_MONEY}[^.]*\.|[^.]*{_MONEY}[^.]*{_FACILITY}[^.]*\.", re.I
)


def ten_k_text(ticker: str) -> tuple[str, str, str]:
    """(accession, report date, plain text) of the 10-K for the fiscal year ending in 2019."""
    base = "https://data.sec.gov/submissions/"
    sub = json.loads(_get(f"{base}CIK{cik(ticker):010d}.json"))["filings"]
    pages = [sub["recent"], *(json.loads(_get(base + f["name"])) for f in sub["files"])]
    rows = [
        (page["reportDate"][i], page["accessionNumber"][i], page["primaryDocument"][i])
        for page in pages
        for i, form in enumerate(page["form"])
        if form == "10-K" and "2019-01-01" <= page["reportDate"][i] <= BOOK_DATE
    ]
    if not rows:
        raise LookupError(f"{ticker}: no 10-K for fiscal 2019")
    report, accn, doc = max(rows)
    path = SEC_RAW / f"{ticker}_10k.htm"
    if not path.exists():
        url = f"https://www.sec.gov/Archives/edgar/data/{cik(ticker)}/{accn.replace('-', '')}/{doc}"
        path.write_bytes(_get(url))
    text = re.sub(r"<[^>]+>", " ", path.read_text(errors="ignore"))
    return accn, report, " ".join(html.unescape(text).split())


def credit_line_candidates(tickers: list[str]) -> pd.DataFrame:
    out = []
    for ticker in tickers:
        accn, report, text = ten_k_text(ticker)
        for m in CREDIT_LINE.finditer(text):
            sentence = m.group(0).strip()
            if len(sentence) < 600:  # longer matches are tables run together
                amounts = re.findall(_MONEY, sentence, re.I)
                out.append({"ticker": ticker, "accn": accn, "report_date": report,
                            "amounts": "; ".join(amounts), "sentence": sentence})  # fmt: skip
    return pd.DataFrame(out).drop_duplicates(["ticker", "sentence"])


# --- Market history ----------------------------------------------------------------------------


def fred(series: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    df = pd.read_csv(io.BytesIO(_get(url)), na_values=".")
    df.columns = ["date", series]
    return df.set_index(pd.to_datetime(df["date"]))[series]


def market_history() -> pd.DataFrame:
    """Daily: Baa and Aaa spread, 10-year Treasury (%), VIX, USD/INR, S&P 500 close, 1990-2023."""
    cols = {name: fred(series) for series, name in FRED_SERIES.items()}
    spx = fetch_prices([MARKET], MARKET_START, MARKET_END).set_index("date")["close"]
    cols["spx"] = spx.set_axis(pd.to_datetime(spx.index))
    df = pd.DataFrame(cols).loc[MARKET_START:MARKET_END]
    return df.dropna(subset=["baa", "spx"]).rename_axis("date").reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenk", action="store_true", help="credit-line sentences from 10-Ks")
    args = parser.parse_args()
    REFERENCE.mkdir(parents=True, exist_ok=True)
    if args.tenk:
        debt = pd.read_csv(SEC_DEBT_PATH)
        banks = universe().query("sector == 'Financials'")["ticker"]
        no_line = debt["credit_line"].isna() & ~debt["ticker"].isin(banks)
        missing = debt.loc[no_line, "ticker"].tolist()
        df = credit_line_candidates(missing)
        df.to_csv(CANDIDATES_PATH, index=False)
        print(f"{len(df)} candidate sentences for {missing} -> {CANDIDATES_PATH}")
        return
    debt = sec_debt()
    debt.to_csv(SEC_DEBT_PATH, index=False)
    shown = debt[["ticker", "short", "long", "total", "credit_line", "due_y1"]].set_index("ticker")
    print((shown / 1e9).round(2).to_string())
    mkt = market_history()
    mkt.to_csv(MARKET_PATH, index=False, float_format="%.4f")
    first, last = mkt["date"].min(), mkt["date"].max()
    print(f"{len(mkt)} days {first:%Y-%m-%d} – {last:%Y-%m-%d} -> {MARKET_PATH}")


if __name__ == "__main__":
    main()
