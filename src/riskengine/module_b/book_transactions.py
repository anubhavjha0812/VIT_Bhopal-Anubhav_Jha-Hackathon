"""A separate, synthetic merchant-flow sleeve (docs/MODULE_B_TRANSACTIONS.md).

uv run python -m riskengine.module_b.book_transactions --from-raw  # stream raw -> sample + book
uv run python -m riskengine.module_b.book_transactions             # sample -> same book
uv run python -m riskengine.module_b.book_transactions --stress    # separate stress JSON

Flows seed hypothetical exposures and rating buckets, not real merchant credit estimates.
The SEC book, reports, alerts and dashboard defaults are not changed.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from riskengine.module_b.book import (
    CCF,
    CDS_BOUGHT,
    CDS_SOLD,
    COLUMNS,
    HEDGE_YEARS,
    IRS,
    LGD,
    REVOLVER_YEARS,
)
from riskengine.module_b.params import notch
from riskengine.module_b.sources import BOOK_DATE, REFERENCE
from riskengine.paths import DATA

RAW = DATA / "raw" / "transactions"
SAMPLE_PATH = REFERENCE / "transactions_monthly_sample.csv"
BOOK_PATH = REFERENCE / "book_transactions.csv"
STRESS_PATH = DATA / "outputs" / "module_b_transactions_stress.json"
SOURCE_URL = "https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets"
SEED = 0
CHUNKSIZE = 250_000
MONTHS = 12
MIN_ACTIVE_MONTHS = 10
MAX_MERCHANTS = 60
TERM_COUNT, TERM_MULTIPLE, TERM_YEARS = 30, 6, 3.0
BOND_COUNT, BOND_MULTIPLE, BOND_YEARS = 10, 3, 5.0
REVOLVER_MULTIPLE = 2
RATINGS = ("A", "BBB", "BB", "B", "CCC")
BOOK_COLUMNS = [*COLUMNS, "book", "synthetic"]
SAMPLE_COLUMNS = ["merchant_id", "ticker", "month", "flow_usd", "mcc", "sector", "synthetic"]
RAW_COLUMNS = ["date", "amount", "merchant_id", "mcc"]


def _tie(identifier: str) -> tuple[str, str]:
    return hashlib.sha256(f"{SEED}|{identifier}".encode()).hexdigest(), identifier


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="mixed", errors="coerce")


def reference_months(date_chunks: Iterable[pd.Series]) -> pd.PeriodIndex:
    """Last 12 observed full calendar months; exclude an incomplete trailing month."""
    first, last = None, None
    observed = set()
    for values in date_chunks:
        dates = _dates(values).dropna()
        if dates.empty:
            continue
        lo, hi = dates.min(), dates.max()
        first = lo if first is None else min(first, lo)
        last = hi if last is None else max(last, hi)
        observed.update(dates.dt.to_period("M").unique())
    if last is None:
        raise ValueError("No valid transaction dates")
    end = last.to_period("M")
    if last.day != end.days_in_month:
        end -= 1
    months = pd.period_range(end=end, periods=MONTHS, freq="M")
    if first.normalize() > months[0].start_time or not set(months).issubset(observed):
        raise ValueError("Need 12 full observed calendar months")
    return months


def _cents(values: pd.Series) -> pd.Series:
    """Dollar strings -> nullable integer cents; reject malformed or sub-cent amounts."""
    clean = (values.astype("string").str.strip().str.replace("$", "", regex=False)
             .str.replace(",", "", regex=False))
    valid = clean.str.fullmatch(r"[+-]?\d+(?:\.\d{1,2})?").fillna(False)
    cents = pd.Series(pd.NA, index=values.index, dtype="Int64")
    cents.loc[valid] = (pd.to_numeric(clean.loc[valid]) * 100).round().astype("int64")
    return cents


def aggregate_chunks(
    chunks: Iterable[pd.DataFrame], months: pd.PeriodIndex, mcc_codes: dict[str, str]
) -> pd.DataFrame:
    """Aggregate purchases in bounded chunks, then export only the selected merchants."""
    totals = defaultdict(int)
    for chunk in chunks:
        dates = _dates(chunk["date"])
        within = dates.ge(months[0].start_time) & dates.lt((months[-1] + 1).start_time)
        if "errors" in chunk:
            within &= chunk["errors"].fillna("").str.strip().eq("")
        frame = chunk.loc[within, ["amount", "merchant_id", "mcc"]].copy()
        frame["month"] = dates.loc[within].dt.to_period("M").astype(str)
        frame["merchant_id"] = frame["merchant_id"].astype("string").str.strip()
        frame["mcc"] = frame["mcc"].astype("string").str.strip()
        frame["cents"] = _cents(frame.pop("amount"))
        good = (frame["cents"].gt(0) & frame["merchant_id"].notna()
                & frame["merchant_id"].ne("") & frame["mcc"].notna() & frame["mcc"].ne(""))
        grouped = frame.loc[good].groupby(["merchant_id", "month", "mcc"])["cents"].sum()
        for key, value in grouped.items():
            totals[key] += int(value)
    if not totals:
        raise ValueError("No eligible purchases in the reference window")
    grouped = pd.DataFrame([(*key, value) for key, value in totals.items()],
                           columns=["merchant_id", "month", "mcc", "cents"])
    monthly = grouped.groupby(["merchant_id", "month"])["cents"].sum()
    active = monthly.groupby(level="merchant_id").size()
    flows = monthly.groupby(level="merchant_id").sum()
    eligible = active[active >= MIN_ACTIVE_MONTHS].index
    chosen = sorted(eligible, key=lambda m: (-int(flows[m]), _tie(m)))[:MAX_MERCHANTS]
    if not chosen:
        raise ValueError("No merchants have 10 positive months")
    mcc_flows = grouped.groupby(["merchant_id", "mcc"])["cents"].sum()
    rows = []
    for merchant in chosen:
        codes = mcc_flows.loc[merchant]
        code = min(codes.index, key=lambda c: (-int(codes[c]), _tie(c)))
        sector = mcc_codes.get(code, f"Unknown MCC {code}")
        for month in months.astype(str):
            rows.append([merchant, f"M{merchant}", month,
                         int(monthly.get((merchant, month), 0)) / 100,
                         code, sector, True])
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def sample_from_raw(raw: Path = RAW, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    """Two streaming passes; never read customer/card fields or the entire raw file."""
    path = raw / "transactions_data.csv"
    headers = pd.read_csv(path, nrows=0).columns
    missing = set(RAW_COLUMNS) - set(headers)
    if missing:
        raise ValueError(f"Missing transaction columns: {sorted(missing)}")
    date_chunks = pd.read_csv(path, usecols=["date"], dtype="string", chunksize=chunksize)
    months = reference_months(c["date"] for c in date_chunks)
    print(f"Reference window: {months[0]} to {months[-1]}; aggregating purchases", flush=True)
    cols = [*RAW_COLUMNS, *(["errors"] if "errors" in headers else [])]
    chunks = pd.read_csv(path, usecols=cols, dtype="string", chunksize=chunksize)
    mcc_codes = json.loads((raw / "mcc_codes.json").read_text())
    return aggregate_chunks(chunks, months, mcc_codes)


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"merchant_id": "string", "mcc": "string"})


def _merchants(sample: pd.DataFrame) -> pd.DataFrame:
    """Validate the portable sample and assign synthetic ratings without stress information."""
    if list(sample.columns) != SAMPLE_COLUMNS or sample.empty:
        raise ValueError("Invalid monthly sample schema")
    if (not sample["synthetic"].eq(True).all() or sample.isna().any().any()
            or not sample["ticker"].eq("M" + sample["merchant_id"].astype(str)).all()):
        raise ValueError("Sample must contain synthetic merchant IDs only")
    if sample.duplicated(["merchant_id", "month"]).any():
        raise ValueError("Duplicate merchant/month")
    months = pd.PeriodIndex(sorted(sample["month"].unique()), freq="M")
    expected = pd.period_range(months[0], periods=MONTHS, freq="M")
    if len(months) != MONTHS or not months.equals(expected):
        raise ValueError("Sample must span 12 consecutive months")
    values = _cents(sample["flow_usd"])
    if values.isna().any() or values.lt(0).any():
        raise ValueError("Sample flows must be nonnegative dollar amounts")
    frame = sample.assign(cents=values.astype("int64"))
    rows = []
    for merchant, group in frame.groupby("merchant_id", sort=True):
        group = group.sort_values("month")
        if (len(group) != MONTHS or group["cents"].gt(0).sum() < MIN_ACTIVE_MONTHS
                or group["mcc"].nunique() != 1 or group["sector"].nunique() != 1):
            raise ValueError("Invalid merchant coverage or sector")
        mean = group["cents"].mean()
        rows.append({"merchant_id": str(merchant), "ticker": f"M{merchant}",
                     "flow_cents": int(group["cents"].sum()), "mean_usd": mean / 100,
                     "cv": group["cents"].std(ddof=0) / mean,
                     "mcc": group["mcc"].iloc[0], "sector": group["sector"].iloc[0]})
    if len(rows) > MAX_MERCHANTS:
        raise ValueError("Sample exceeds the 60-merchant cap")
    rows.sort(key=lambda r: (-r["flow_cents"], _tie(r["merchant_id"])))
    volatility = sorted(rows, key=lambda r: (r["cv"], _tie(r["merchant_id"])))
    for rank, row in enumerate(volatility):
        row["rating"] = RATINGS[len(RATINGS) * rank // len(rows)]
    return pd.DataFrame(rows)


def build_book(sample: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed flow mapping and the existing hedge sizes to a monthly sample."""
    merchants = _merchants(sample)
    rows = []
    for rank, m in enumerate(merchants.to_dict("records")):
        source = f"Synthetic flow proxy; MCC {m['mcc']}: {m['sector']}"
        instruments = [("revolver", REVOLVER_MULTIPLE, REVOLVER_YEARS)]
        if rank < TERM_COUNT:
            instruments.append(("term_loan", TERM_MULTIPLE, TERM_YEARS))
        if rank < BOND_COUNT:
            instruments.append(("bond", BOND_MULTIPLE, BOND_YEARS))
        for inst, multiple, years in instruments:
            n = round(multiple * m["mean_usd"], 2)
            ead = round(n * CCF, 2) if inst == "revolver" else n
            rows.append({"ticker": m["ticker"], "rating": m["rating"], "instrument": inst,
                         "notional": n, "ead": ead, "years": years, "source": source})
    cash = pd.DataFrame(rows)
    exposure = cash.groupby("ticker")["notional"].sum().to_dict()
    ratings = merchants.set_index("ticker")["rating"].to_dict()
    worst = sorted(exposure, key=lambda t: (-notch(ratings[t]), -exposure[t], _tie(t[1:])))
    best = sorted(exposure, key=lambda t: (notch(ratings[t]), -exposure[t], _tie(t[1:])))
    largest = sorted(exposure, key=lambda t: (-exposure[t], _tie(t[1:])))
    for inst, names, (count, share) in [
        ("cds_bought", worst, CDS_BOUGHT), ("cds_sold", best, CDS_SOLD),
        ("irs_pay_fixed", largest, IRS),
    ]:
        for ticker in names[:count]:
            rows.append({"ticker": ticker, "rating": ratings[ticker], "instrument": inst,
                         "notional": round(share * exposure[ticker], 2), "ead": 0.0,
                         "years": HEDGE_YEARS,
                         "source": f"Synthetic proxy; {share:.0%} of cash notional (E14 rule)"})
    book = pd.DataFrame(rows)
    book.insert(0, "position", [f"TP{i:03d}" for i in range(1, len(book) + 1)])
    book["lgd"], book["book"], book["synthetic"] = LGD, "transactions", True
    return book[BOOK_COLUMNS]


def write_book(book: pd.DataFrame, path: Path = BOOK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    book.to_csv(path, index=False, float_format="%.2f", lineterminator="\n")


def stress_results(book: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Run only existing market scenarios; do not write any existing Module B output."""
    from riskengine.module_b.revalue import revalue
    from riskengine.module_b.shock import EPISODES

    episodes = {}
    for episode, dates in EPISODES.items():
        result = revalue(book, episode=episode, rates="fixed")
        episodes[episode] = {
            "dates": list(dates), "loss_usd": result.loss_usd,
            "value_before_usd": float(result.positions["value_before"].sum()),
            "value_after_usd": float(result.positions["value_after"].sum()),
            "rwa_before_inr": float(result.rwa_before_inr),
            "rwa_after_inr": float(result.rwa_after_inr),
            "capital_before_inr": float(result.capital_before_inr),
            "capital_after_inr": float(result.capital_after_inr),
            "cet1_before": float(result.cet1_before), "cet1_after": float(result.cet1_after),
        }
    cash = book[book["instrument"].isin(["revolver", "term_loan", "bond"])]
    return {
        "book": "transactions", "synthetic": True, "currency": "USD",
        "disclosure": "Transaction-flow proxy; exposures and ratings are synthetic, not real.",
        "source_url": SOURCE_URL, "license": "see the dataset page", "seed": SEED,
        "reference_window": [sample["month"].min(), sample["month"].max()],
        "market_base_date": BOOK_DATE,
        "stress_method": "Market only; multiple 1; no obligor downgrade; fixed Treasury rates. "
                         "Existing stock-basket market factor, not merchant observations.",
        "merchants": int(book["ticker"].nunique()), "positions": len(book),
        "positions_per_instrument": book["instrument"].value_counts().to_dict(),
        "rating_counts": book.drop_duplicates("ticker")["rating"].value_counts().to_dict(),
        "cash_ead_usd": float(cash["ead"].sum()),
        "cash_notional_usd": float(cash["notional"].sum()),
        "derivative_notional_usd": float(book.loc[~book.index.isin(cash.index), "notional"].sum()),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-raw", action="store_true", help="Stream raw CSV and rebuild sample")
    parser.add_argument("--stress", action="store_true", help="Write the separate stress JSON")
    args = parser.parse_args()
    sample = sample_from_raw() if args.from_raw else load_sample(SAMPLE_PATH)
    book = build_book(sample)
    if args.from_raw:
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(SAMPLE_PATH, index=False, float_format="%.2f", lineterminator="\n")
    write_book(book, BOOK_PATH)
    print(f"{book['ticker'].nunique()} synthetic merchants, {len(book)} positions, "
          f"cash EAD ${book['ead'].sum():,.2f} -> {BOOK_PATH}")
    if args.stress:
        output = stress_results(book, sample)
        output["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (SAMPLE_PATH, BOOK_PATH)}
        STRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STRESS_PATH.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"Separate stress results -> {STRESS_PATH}")


if __name__ == "__main__":
    main()
