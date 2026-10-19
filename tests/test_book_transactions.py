import json

import pandas as pd
import pytest

from riskengine.module_b import book_transactions as bt
from riskengine.module_b.book import CCF, CDS_BOUGHT, CDS_SOLD, COLUMNS, IRS, LGD
from riskengine.module_b.params import pd_1y
from riskengine.module_b.revalue import _position, ecl, irb_rwa
from riskengine.module_b.shock import Market


@pytest.fixture
def transactions():
    rows = []
    for merchant in range(1, 66):
        for month in range(1, 13):
            rows.append({"date": f"2023-{month:02d}-28", "merchant_id": str(merchant),
                         "amount": f"${merchant * 100 + month * (merchant % 7):.2f}",
                         "mcc": "5411", "errors": None})
    # High-volume merchant with only nine active months must not displace eligible merchants.
    rows.extend({"date": f"2023-{month:02d}-28", "merchant_id": "999",
                 "amount": "$999999.00", "mcc": "5411", "errors": ""}
                for month in range(1, 10))
    return pd.DataFrame(rows)


def aggregate(frame, size=37):
    chunks = (frame.iloc[i:i + size] for i in range(0, len(frame), size))
    return bt.aggregate_chunks(chunks, pd.period_range("2023-01", "2023-12", freq="M"),
                               {"5411": "Grocery Stores, Supermarkets"})


def test_reference_window_excludes_partial_last_month():
    dates = pd.Series([*(f"2023-{m:02d}-01" for m in range(1, 13)),
                       "2023-12-31", "2024-01-15", "bad"])
    months = bt.reference_months([dates.iloc[:5], dates.iloc[5:]])
    assert list(months.astype(str)) == [f"2023-{m:02d}" for m in range(1, 13)]
    months = bt.reference_months([pd.concat([dates, pd.Series(["2024-01-31"])])])
    assert str(months[0]) == "2023-02" and str(months[-1]) == "2024-01"


def test_reference_window_rejects_missing_coverage():
    with pytest.raises(ValueError, match="12 full"):
        bt.reference_months([pd.Series(["2023-01-01", "2023-12-31"])])


def test_purchase_errors_dates_and_dominant_mcc(transactions):
    extra = pd.DataFrame([
        {"date": "2023-01-02", "amount": "$1,234.56", "errors": " ", "mcc": "5812"},
        {"date": "2023-01-02", "amount": "$-77.00", "errors": "", "mcc": "5411"},
        {"date": "2023-01-02", "amount": "$999999.00", "errors": "Bad PIN", "mcc": "5411"},
        {"date": "2023-01-02", "amount": "broken", "errors": "", "mcc": "5411"},
        {"date": "2023-01-02", "amount": "$1.001", "errors": "", "mcc": "5411"},
        {"date": "2024-01-01", "amount": "$999999.00", "errors": "", "mcc": "5411"},
        {"date": "invalid", "amount": "$999999.00", "errors": "", "mcc": "5411"},
    ]).assign(merchant_id="65")
    original = aggregate(transactions).query("merchant_id == '65'")["flow_usd"].sum()
    sample = aggregate(pd.concat([transactions, extra], ignore_index=True))
    selected = sample.query("merchant_id == '65'")
    assert selected["flow_usd"].sum() == pytest.approx(original + 1234.56)
    assert selected["mcc"].eq("5411").all()  # Most purchase flow, not last observed MCC.
    no_errors_column = aggregate(transactions.drop(columns="errors"))
    pd.testing.assert_frame_equal(no_errors_column, aggregate(transactions))


def test_caps_coverage_and_rating_quantiles(transactions):
    sample = aggregate(transactions)
    assert len(sample) == 720
    assert set(sample["merchant_id"]) == {str(m) for m in range(6, 66)}
    book = bt.build_book(sample)
    assert len(book) == 111
    assert book["instrument"].value_counts().to_dict() == {
        "revolver": 60, "term_loan": 30, "bond": 10,
        "irs_pay_fixed": 5, "cds_bought": 3, "cds_sold": 3,
    }
    merchants = bt._merchants(sample)
    assert merchants["rating"].value_counts().to_dict() == dict.fromkeys(bt.RATINGS, 12)
    for better, worse in zip(bt.RATINGS, bt.RATINGS[1:], strict=False):
        assert (merchants.loc[merchants["rating"] == better, "cv"].max()
                <= merchants.loc[merchants["rating"] == worse, "cv"].min())


def test_amounts_ead_maturities_and_hedges(transactions):
    sample = aggregate(transactions)
    book = bt.build_book(sample)
    largest = sample.iloc[0]["ticker"]
    mean = sample[sample["ticker"] == largest]["flow_usd"].mean()
    positions = book[book["ticker"] == largest].set_index("instrument")
    for inst, multiple, years in [("revolver", 2, 5), ("term_loan", 6, 3), ("bond", 3, 5)]:
        assert positions.loc[inst, "notional"] == round(multiple * mean, 2)
        factor = CCF if inst == "revolver" else 1
        assert positions.loc[inst, "ead"] == round(factor * round(multiple * mean, 2), 2)
        assert positions.loc[inst, "years"] == years
    assert book["lgd"].eq(LGD).all()
    cash = book[book["instrument"].isin(["revolver", "term_loan", "bond"])]
    exposure = cash.groupby("ticker")["notional"].sum()
    for inst, (_, share) in [("cds_bought", CDS_BOUGHT), ("cds_sold", CDS_SOLD),
                              ("irs_pay_fixed", IRS)]:
        hedges = book[book["instrument"] == inst]
        assert hedges["ead"].eq(0).all() and hedges["years"].eq(5).all()
        assert hedges["notional"].tolist() == [round(exposure[t] * share, 2)
                                               for t in hedges["ticker"]]
    assert set(book.loc[book["instrument"] == "cds_bought", "rating"]) == {"CCC"}
    assert set(book.loc[book["instrument"] == "cds_sold", "rating"]) == {"A"}


def test_deterministic_chunks_input_order_and_ties(transactions):
    # Force all eligible merchants to tie on flow and CV; the seed key decides membership.
    frame = transactions[transactions["merchant_id"] != "999"].assign(amount="$100.00")
    first = aggregate(frame, size=1)
    second = aggregate(frame.sample(frac=1, random_state=42), size=200)
    pd.testing.assert_frame_equal(first, second)
    expected = sorted(map(str, range(1, 66)), key=bt._tie)[:60]
    assert first["merchant_id"].drop_duplicates().tolist() == expected
    pd.testing.assert_frame_equal(bt.build_book(first),
                                  bt.build_book(second.sample(frac=1, random_state=5)))


def test_flags_columns_no_real_names_and_sample_roundtrip(transactions, tmp_path):
    sample = aggregate(transactions)
    book = bt.build_book(sample)
    assert list(book.columns) == [*COLUMNS, "book", "synthetic"]
    assert book["book"].eq("transactions").all() and book["synthetic"].eq(True).all()
    assert book["ticker"].str.fullmatch(r"M\d+").all()
    assert book["source"].str.startswith("Synthetic").all()
    assert list(sample.columns) == bt.SAMPLE_COLUMNS
    assert not {"client_id", "card_id", "address", "card_number"} & set(sample.columns)
    sample_path, first, second = [tmp_path / n for n in ("sample.csv", "a.csv", "b.csv")]
    sample.to_csv(sample_path, index=False, float_format="%.2f")
    bt.write_book(book, first)
    bt.write_book(bt.build_book(bt.load_sample(sample_path)), second)
    assert first.read_bytes() == second.read_bytes()


def test_from_raw_streams_only_needed_columns(transactions, tmp_path, monkeypatch):
    frame = pd.concat([transactions, pd.DataFrame([{
        "date": "2023-01-01", "merchant_id": "1", "amount": "$-77.00", "mcc": "5411",
    }, {"date": "2023-12-31", "merchant_id": "1", "amount": "$0.00", "mcc": "5411"}])])
    frame.assign(client_id="private", card_id="private").to_csv(
        tmp_path / "transactions_data.csv", index=False)
    (tmp_path / "mcc_codes.json").write_text(json.dumps({"5411": "Grocery Stores, Supermarkets"}))
    real_read = pd.read_csv
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(bt.pd, "read_csv", read)
    actual = bt.sample_from_raw(tmp_path, chunksize=41)
    pd.testing.assert_frame_equal(actual, aggregate(transactions))
    assert len(calls) == 3 and calls[0]["nrows"] == 0
    assert calls[1]["usecols"] == ["date"]
    assert set(calls[2]["usecols"]) == {*bt.RAW_COLUMNS, "errors"}
    assert all(c["chunksize"] == 41 for c in calls[1:])


def test_default_cli_needs_only_portable_sample(transactions, tmp_path, monkeypatch):
    sample_path, book_path = tmp_path / "sample.csv", tmp_path / "book.csv"
    aggregate(transactions).to_csv(sample_path, index=False, float_format="%.2f")
    monkeypatch.setattr(bt, "SAMPLE_PATH", sample_path)
    monkeypatch.setattr(bt, "BOOK_PATH", book_path)
    monkeypatch.setattr("sys.argv", ["book_transactions"])
    monkeypatch.setattr(bt, "sample_from_raw", lambda: pytest.fail("Default must not read raw"))
    bt.main()
    assert len(pd.read_csv(book_path)) == 111


@pytest.mark.parametrize("rating,stg", [("BB", 1), ("B-", 2), ("D", 3)])
def test_funded_term_loan_uses_principal_less_ecl_and_credit_rwa(rating, stg):
    p = pd.Series({"instrument": "term_loan", "notional": 1000.0, "ead": 1000.0,
                   "lgd": LGD, "years": 3.0})
    s = pd.Series({"rating_before": "BB", "rating_after": rating,
                   "spread_before": 3.0, "spread_after": 5.0,
                   "pd_before": pd_1y("BB"), "pd_after": pd_1y(rating)})
    value, rwa, actual_stage = _position(p, s, .02, .02, "after")
    assert actual_stage == stg
    assert value == pytest.approx(1000 - ecl(pd_1y(rating), LGD, 1000, 3, stg))
    assert rwa == pytest.approx(irb_rwa(pd_1y(rating), LGD, 3, 1000))


def test_rejects_modified_sample(transactions):
    sample = aggregate(transactions)
    with pytest.raises(ValueError, match="synthetic merchant"):
        bt.build_book(sample.assign(ticker="Real Company"))
    with pytest.raises(ValueError, match="Duplicate"):
        bt.build_book(pd.concat([sample, sample.iloc[:1]], ignore_index=True))


def test_separate_stress_prices_synthetic_ids_without_company_correlations(
    transactions, monkeypatch,
):
    monkeypatch.setattr("riskengine.module_b.shock.market", lambda name: Market(
        name, 1.0, 2.0, 0.5, 1.0, 2.0, 70.0, 0.05, 2.0))
    monkeypatch.setattr("riskengine.module_b.shock.correlation",
                        lambda: pytest.fail("Merchant IDs have no company correlations"))
    sample = aggregate(transactions)
    output = bt.stress_results(bt.build_book(sample), sample)
    assert output["synthetic"] is True and output["book"] == "transactions"
    assert set(output["episodes"]) == {"gfc2008", "euro2011", "covid2020"}
    for episode in output["episodes"].values():
        assert episode["loss_usd"] > 0
        assert episode["loss_usd"] == pytest.approx(
            episode["value_before_usd"] - episode["value_after_usd"])
        assert episode["cet1_before"] == pytest.approx(.13)
    json.dumps(output, allow_nan=False)
