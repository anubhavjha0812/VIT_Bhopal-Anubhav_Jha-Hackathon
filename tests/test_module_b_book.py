import pandas as pd
import pytest

from riskengine.module_b.book import AFTER_Y5_YEARS, CCF, SHARE, bond_buckets, build_book
from riskengine.module_b.params import notch


@pytest.fixture(scope="module")
def book():
    return build_book()


def debt_row(**kw):
    base = {"long": 100.0, "due_y2": None, "due_y3": None, "due_y4": None, "due_y5": None,
            "due_after_y5": None}  # fmt: skip
    return pd.Series({**base, **kw})


def test_buckets_sum_to_long_term_debt_and_rest_goes_after_year_5():
    b = bond_buckets(debt_row(due_y2=10, due_y5=20))
    assert sum(b.values()) == pytest.approx(100)
    assert b[AFTER_Y5_YEARS] == pytest.approx(70)


def test_buckets_scale_face_value_schedule_to_carrying_amount():
    b = bond_buckets(debt_row(due_after_y5=150))  # face above carrying (Netflix-like)
    assert b == {AFTER_Y5_YEARS: pytest.approx(100)}


def test_no_long_term_debt_no_bonds():
    assert bond_buckets(debt_row(long=None)) == {}


def test_boeing_is_half_a_percent_of_real_debt(book):
    ba = book[(book["ticker"] == "BA") & book["instrument"].isin(["bond", "revolver"])]
    bonds = ba.loc[ba["instrument"] == "bond", "notional"].sum()
    revolver = ba.loc[ba["instrument"] == "revolver", "notional"].sum()
    assert bonds == pytest.approx(SHARE * 19.962e9)
    assert revolver == pytest.approx(SHARE * 6.62e9)  # unused credit line, XBRL


def test_revolver_ead_uses_ccf_and_hedges_have_no_ead(book):
    rev = book[book["instrument"] == "revolver"]
    assert rev["ead"].tolist() == pytest.approx((CCF * rev["notional"]).tolist())
    hedges = book[book["instrument"].str.startswith(("cds", "irs"))]
    assert (hedges["ead"] == 0).all()


def test_banks_capped_at_largest_non_bank(book):
    funded = book[book["instrument"].isin(["bond", "revolver"])]
    exposure = funded.groupby("ticker")["notional"].sum()
    largest = exposure.drop(["JPM", "BAC"]).max()
    assert exposure["JPM"] <= largest * (1 + 1e-9)
    assert exposure["BAC"] <= largest * (1 + 1e-9)


def test_protection_bought_on_worst_sold_on_best(book):
    bought = book.loc[book["instrument"] == "cds_bought", "rating"].map(notch)
    sold = book.loc[book["instrument"] == "cds_sold", "rating"].map(notch)
    assert len(bought) == len(sold) == 3
    assert bought.min() > sold.max()


def test_every_position_has_rating_and_source(book):
    assert book["rating"].notna().all()
    assert book["source"].str.len().gt(0).all()
    assert book["position"].is_unique
