import pandas as pd

from riskengine.module_b.sources import (
    CREDIT_LINE,
    REFERENCE,
    debt_figures,
    latest_fact,
)


def fact(val, end, filed, form="10-K", accn="a"):
    return {"val": val, "end": end, "filed": filed, "form": form, "accn": accn}


def facts(**tags):
    return {"facts": {"us-gaap": {k: {"units": {"USD": v}} for k, v in tags.items()}}}


def test_latest_fact_takes_newest_date_as_first_reported():
    f = facts(DebtCurrent=[
        fact(1, "2019-09-30", "2019-10-30", "10-Q"),
        fact(7, "2019-12-31", "2020-01-31", accn="first"),
        fact(8, "2019-12-31", "2021-01-31", accn="restated"),  # later 10-K, comparative column
        fact(9, "2020-03-31", "2020-04-30", "10-Q"),  # after the book date
    ])  # fmt: skip
    hit = latest_fact(f, ["DebtCurrent"])
    assert (hit["value"], hit["accn"]) == (7, "first")


def test_latest_fact_ignores_old_filings_and_other_forms():
    f = facts(DebtCurrent=[fact(1, "2018-06-30", "2018-07-30"), fact(2, "2019-12-31", "x", "8-K")])
    assert latest_fact(f, ["DebtCurrent"]) is None


def test_debt_figures_combined_tag():
    f = facts(
        DebtAndCapitalLeaseObligations=[fact(27_302, "2019-12-31", "2020-01-31")],
        DebtCurrent=[fact(7_340, "2019-12-31", "2020-01-31")],
        LongTermDebtAndCapitalLeaseObligations=[fact(19_962, "2019-12-31", "2020-01-31")],
    )
    row = debt_figures(f)
    assert (row["short"], row["long"], row["total"]) == (7_340, 19_962, 27_302)


def test_debt_figures_adds_commercial_paper_when_no_debt_current():
    f = facts(
        LongTermDebtCurrent=[fact(10, "2019-12-28", "2020-01-29")],
        CommercialPaper=[fact(5, "2019-12-28", "2020-01-29")],
        LongTermDebtNoncurrent=[fact(93, "2019-12-28", "2020-01-29")],
    )
    row = debt_figures(f)
    assert (row["short"], row["long"], row["total"]) == (15, 93, 108)
    assert row["total_src"] == "short + long"


def test_debt_figures_long_from_total_long_term_minus_current():
    f = facts(
        DebtCurrent=[fact(1.4, "2019-12-31", "2020-02-13")],
        LongTermDebt=[fact(10.4, "2019-12-31", "2020-02-13")],
    )
    assert round(debt_figures(f)["long"], 6) == 9.0


def test_credit_line_regex_finds_facility_sentence():
    text = "We have $4.0 billion of revolving credit facilities expiring in July 2023. Other text."
    assert CREDIT_LINE.search(text).group(0).startswith("We have $4.0 billion")


def test_reference_files_are_complete():
    ratings = pd.read_csv(REFERENCE / "ratings.csv")
    debt = pd.read_csv(REFERENCE / "sec_debt.csv")
    universe = pd.read_csv(REFERENCE / "universe.csv")
    assert set(ratings["ticker"]) == set(universe["ticker"]) == set(debt["ticker"])
    assert ratings["source"].str.startswith("https://").all()
