import pandas as pd

from riskengine.replay import events, sse, texts


def signals() -> pd.DataFrame:
    # NY dates: a = Fri 1 May, b = Mon 4 May (09:00 NY), c = Mon 4 May
    return pd.DataFrame(
        {
            "doc_id": ["b", "a", "a", "c"],
            "published_at": pd.to_datetime(
                [
                    "2020-05-04 13:00+00:00",
                    "2020-05-01 15:00+00:00",
                    "2020-05-01 15:00+00:00",
                    "2020-05-04 20:00+00:00",
                ]
            ),
            "source": ["tweets", "news", "news", "news"],
            "ticker": ["JPM", "JPM", "BAC", "BAC"],
            "text": ["y", "x", "x", "z"],
            "sentiment": [0.1, -0.8, -0.8, 0.3],
            "event": ["Other", "Default", "Default", "M&A"],
            "impact": [3, 9, 9, 6],
            # as gate.py would mark them: a fires for JPM only
            "fired": [False, True, False, False],
            "reason": ["not credit", "fired: rating agency", "not the obligor", "not credit"],
        }
    )


def weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-05-01", "2020-05-04", "2020-05-05"],
            "JPM": [0.5, 0.4, 0.6],
            "BAC": [0.5, 0.6, 0.4],
        }
    )


def test_texts_are_one_row_each_in_time_order():
    out = texts(signals())
    assert out["doc_id"].tolist() == ["a", "b", "c"]
    assert out.loc[0, "tickers"] == ["JPM", "BAC"]


def test_weights_never_come_before_the_texts_they_use():
    kinds = [(k, p.get("doc_id") or p.get("date")) for k, p in events(signals(), weights())]
    assert kinds == [
        ("weights", "2020-05-01"),  # uses texts before 1 May: none in this sample
        ("signal", "a"),
        ("alert", "a"),  # the gate fired for one of its companies
        ("weights", "2020-05-04"),  # sent only after every text dated before 4 May
        ("signal", "b"),
        ("signal", "c"),
        ("weights", "2020-05-05"),  # after the last text
    ]


def test_sse_message_format():
    assert sse("signal", '{"a": 1}') == 'event: signal\ndata: {"a": 1}\n\n'


def test_analyst_bank_is_listed_apart_and_alerts_name_the_obligor():
    # "iRobot downgraded by JPMorgan": JPM is only the analyst, no subject in our universe
    s = signals().assign(role=["subject", "subject", "analyst", "analyst"])
    out = texts(s)
    assert out.loc[0, "tickers"] == ["JPM"] and out.loc[0, "analysts"] == ["BAC"]
    assert out.loc[2, "tickers"] == [] and out.loc[2, "analysts"] == ["BAC"]
    alerts = [p for k, p in events(s, weights()) if k == "alert"]
    assert [(p["doc_id"], p["obligors"], p["gate"]) for p in alerts] == [
        ("a", ["JPM"], "fired: rating agency")
    ]
