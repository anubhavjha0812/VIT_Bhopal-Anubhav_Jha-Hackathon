"""The bank's book at 31 Dec 2019, built from real filings (docs/DECISIONS.md, E14).

uv run python -m riskengine.module_b.book   # data/reference/book.csv

Per company the bank holds SHARE of its real debt:
- revolver: SHARE × credit line (XBRL tag, else the 10-K sentence, else short-term debt);
  undrawn, so EAD = 75% of the commitment (Basel F-IRB credit conversion factor)
- bonds: SHARE × long-term debt, split by the 10-K repayment schedule into years 2-5 and
  "after year 5" (taken as 10 years); a missing "after year 5" is the rest of the long-term debt
- banks (Financials) are capped at the largest non-bank exposure
Hedges: CDS protection bought on the 3 lowest-rated obligors (25% of exposure), sold on the 3
highest-rated (10%); pay-fixed interest rate swaps with the 5 largest (20%), for CVA.
"""

import pandas as pd

from riskengine.module_b.params import notch
from riskengine.module_b.sources import REFERENCE, SEC_DEBT_PATH, universe

BOOK_PATH = REFERENCE / "book.csv"
CREDIT_LINES_PATH = REFERENCE / "sec_credit_lines.csv"
RATINGS_PATH = REFERENCE / "ratings.csv"

SHARE = 0.005  # 0.5% of each company's debt (E14; checked by sensitivity)
CCF = 0.75  # undrawn revolver → EAD
LGD = 0.45  # F-IRB senior unsecured
REVOLVER_YEARS = 5.0  # syndicated revolvers usually run 5 years
AFTER_Y5_YEARS = 10.0  # "after year 5" bucket
NO_SCHEDULE_YEARS = AFTER_Y5_YEARS  # no schedule tag: all long-term debt goes in the last bucket
CDS_BOUGHT, CDS_SOLD, IRS = (3, 0.25), (3, 0.10), (5, 0.20)  # (how many names, share of exposure)
HEDGE_YEARS = 5.0
BUCKETS = {"due_y2": 2.0, "due_y3": 3.0, "due_y4": 4.0, "due_y5": 5.0}
COLUMNS = [
    "position", "ticker", "rating", "instrument", "notional", "ead", "lgd", "years", "source",
]  # fmt: skip


def credit_line(debt: pd.Series, lines: pd.DataFrame) -> tuple[float, str]:
    """Commitment size and where it came from."""
    if pd.notna(debt["credit_line"]):
        return debt["credit_line"], f"XBRL {debt['credit_line_src']}"
    hit = lines[(lines["ticker"] == debt["ticker"]) & lines["credit_line"].notna()]
    if len(hit):
        return hit["credit_line"].iloc[0], f"10-K sentence {hit['accn'].iloc[0]}"
    return (debt["short"] if pd.notna(debt["short"]) else 0.0), "short-term debt (no credit line)"


def bond_buckets(debt: pd.Series) -> dict[float, float]:
    """Long-term debt split by year of repayment: {years: amount}, summing to the long-term debt."""
    long = debt["long"] if pd.notna(debt["long"]) else 0.0
    if long <= 0:
        return {}
    due = {years: debt[col] for col, years in BUCKETS.items() if pd.notna(debt[col])}
    rest = debt["due_after_y5"] if pd.notna(debt["due_after_y5"]) else long - sum(due.values())
    due[AFTER_Y5_YEARS] = max(rest, 0.0)
    total = sum(due.values())
    if total <= 0:
        return {NO_SCHEDULE_YEARS: long}
    return {years: long * amount / total for years, amount in due.items() if amount > 0}


def build_book(share: float = SHARE) -> pd.DataFrame:
    """One row per position: ticker, rating, instrument, notional, EAD (USD), years, source."""
    debt = pd.read_csv(SEC_DEBT_PATH)
    lines = pd.read_csv(CREDIT_LINES_PATH)
    ratings = pd.read_csv(RATINGS_PATH).set_index("ticker")["rating"]
    banks = set(universe().query("sector == 'Financials'")["ticker"])

    rows = []
    for _, d in debt.iterrows():
        line, src = credit_line(d, lines)
        rows.append({"ticker": d["ticker"], "instrument": "revolver", "notional": share * line,
                     "years": REVOLVER_YEARS, "source": src})  # fmt: skip
        for years, amount in bond_buckets(d).items():
            rows.append({"ticker": d["ticker"], "instrument": "bond", "notional": share * amount,
                         "years": years, "source": f"long-term debt {d['long_src']}"})  # fmt: skip
    book = pd.DataFrame(rows)
    book = book[book["notional"] > 0]

    exposure = book.groupby("ticker")["notional"].sum()
    cap = exposure.drop(list(banks), errors="ignore").max()
    for bank in banks & set(exposure.index):
        if exposure[bank] > cap:
            book.loc[book["ticker"] == bank, "notional"] *= cap / exposure[bank]
            book.loc[book["ticker"] == bank, "source"] += " (capped at largest non-bank)"
    exposure = book.groupby("ticker")["notional"].sum()

    worst_first = sorted(exposure.index, key=lambda t: (-notch(ratings[t]), -exposure[t]))
    best_first = sorted(exposure.index, key=lambda t: (notch(ratings[t]), -exposure[t]))
    largest = exposure.sort_values(ascending=False).index
    for instrument, names, (n, frac) in [
        ("cds_bought", worst_first, CDS_BOUGHT),
        ("cds_sold", best_first, CDS_SOLD),
        ("irs_pay_fixed", largest, IRS),
    ]:
        for t in names[:n]:
            rows_ = {"ticker": t, "instrument": instrument, "notional": frac * exposure[t],
                     "years": HEDGE_YEARS, "source": f"{frac:.0%} of exposure (E14)"}  # fmt: skip
            book = pd.concat([book, pd.DataFrame([rows_])], ignore_index=True)

    book["rating"] = book["ticker"].map(ratings)
    book["ead"] = book["notional"].where(book["instrument"] == "bond", 0.0)
    book.loc[book["instrument"] == "revolver", "ead"] = CCF * book["notional"]
    book["lgd"] = LGD
    book.insert(0, "position", [f"P{i:03d}" for i in range(1, len(book) + 1)])
    return book[COLUMNS]


def load_book() -> pd.DataFrame:
    return pd.read_csv(BOOK_PATH)


def main() -> None:
    book = build_book()
    book.to_csv(BOOK_PATH, index=False, float_format="%.2f")
    by = book.pivot_table(index="ticker", columns="instrument", values="notional", aggfunc="sum")
    print((by / 1e6).round(1).fillna(0).to_string())
    print(f"{len(book)} positions, loan+bond EAD ${book['ead'].sum() / 1e9:.2f}bn -> {BOOK_PATH}")


if __name__ == "__main__":
    main()
