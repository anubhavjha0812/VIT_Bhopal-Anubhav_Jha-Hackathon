import pandas as pd

from riskengine import ledger

RECORDS = [
    {"kind": "credit_alert", "at": "2020-05-01T12:00:00+00:00", "payload": {"ticker": "DIS"}},
    {"kind": "module_a_weights", "at": "2020-05-02", "payload": {"weights": {"AAPL": 0.5}}},
]


def entries():
    return ledger.chain(RECORDS, {"signals_sha256": "abc"})


def test_chain_is_deterministic_and_verifies():
    assert entries() == entries()
    assert ledger.verify(entries()) == (True, None)


def test_an_edited_payload_is_found():
    e = entries()
    e[1]["payload"]["ticker"] = "XOM"
    assert ledger.verify(e) == (False, 1)


def test_a_removed_or_reordered_line_is_found():
    e = entries()
    assert ledger.verify(e[:1] + e[2:]) == (False, 1)
    assert ledger.verify([e[0], e[2], e[1]]) == (False, 1)


def test_the_first_line_holds_the_versions():
    first = entries()[0]
    assert first["kind"] == "versions" and first["payload"]["signals_sha256"] == "abc"


def test_write_read_round_trip(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger.write(entries(), path)
    assert ledger.read(path) == entries()
    ledger.export(entries(), tmp_path / "l.csv")
    assert len(pd.read_csv(tmp_path / "l.csv")) == 3


def test_alert_and_weight_records():
    fired = pd.DataFrame(
        [
            {
                "doc_id": "d1",
                "published_at": pd.Timestamp("2020-05-01 12:00", tz="UTC"),
                "ticker": "DIS",
                "event": "Credit downgrade",
                "impact": 9,
                "reason": "fired: rating agency",
                "text": "Fitch downgrades Disney",
            }
        ]
    )
    rec = ledger.alert_records(fired)[0]
    assert rec["kind"] == "credit_alert" and rec["payload"]["impact"] == 9
    w = pd.DataFrame({"AAPL": [0.6], "MSFT": [0.4]}, index=pd.to_datetime(["2020-05-04"]))
    wr = ledger.weight_records(w, "entity")[0]
    assert wr["at"] == "2020-05-04" and wr["payload"]["weights"]["AAPL"] == 0.6
