import pandas as pd
import pytest
from fastapi.testclient import TestClient

from riskengine import api
from riskengine.config import Config
from riskengine.nlp.events import get_event_classifier


class FakeScorer:
    name = "fake"

    def score(self, texts):
        return [-0.9 for _ in texts]


@pytest.fixture
def client(monkeypatch):
    rows = pd.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "published_at": pd.to_datetime(
                ["2020-05-01 12:00+00:00", "2020-05-01 13:00+00:00", "2020-05-02 09:00+00:00"]
            ),
            "source": ["news", "tweets", "news"],
            "ticker": ["JPM", "JPM", "BAC"],
            "text": ["x", "y", "z"],
            "sentiment": [-0.5, 0.1, 0.3],
            "event": ["Default", "Other", "M&A"],
            "impact": [9, 3, 6],
            "model": ["finbert+keywords+formula"] * 3,
        }
    )
    monkeypatch.setattr(api, "stored_signals", lambda: rows)
    monkeypatch.setattr(api, "scorer", lambda: FakeScorer())
    # keyword rules: no model download in tests
    monkeypatch.setattr(api, "load_config", lambda: Config())
    monkeypatch.setattr(api, "event_classifier", lambda: get_event_classifier("keywords"))
    return TestClient(api.app)


def test_signals_filter_by_ticker_and_impact(client):
    out = client.get("/signals", params={"ticker": "jpm", "min_impact": 8}).json()
    assert [r["doc_id"] for r in out] == ["a"]


def test_daily_aggregates_per_stock_day(client):
    out = client.get("/daily", params={"ticker": "JPM"}).json()
    assert out == [
        {"date": "2020-05-01", "ticker": "JPM", "sentiment": -0.2, "texts": 2, "max_impact": 9}
    ]


def test_analyze_links_scores_and_classifies(client):
    out = client.post("/analyze", json={"text": "JPMorgan warns of loan defaults"}).json()
    assert out["tickers"] == ["JPM"]
    assert out["event"] == "Default"
    assert out["impact"] > 7
    assert out["model"] == "fake+keywords+formula"


def test_tickers_are_sorted_and_unique(client):
    assert client.get("/tickers").json() == ["BAC", "JPM"]


def test_module_a_endpoints(client, monkeypatch):
    weights = pd.DataFrame({"date": ["2020-04-09"], "AAPL": [0.5], "JPM": [0.5]})
    metrics = pd.DataFrame({"total_return": [0.25]}, index=["equal"])
    monkeypatch.setattr(api, "module_a_outputs", lambda: (weights, metrics))
    assert client.get("/module-a/weights").json() == [
        {"date": "2020-04-09", "AAPL": 0.5, "JPM": 0.5}
    ]
    assert client.get("/module-a/metrics").json() == {"equal": {"total_return": 0.25}}


def test_replay_streams_signals_alerts_and_weights(client, monkeypatch):
    weights = pd.DataFrame({"date": ["2020-05-04"], "JPM": [0.5], "BAC": [0.5]})
    monkeypatch.setattr(api, "module_a_outputs", lambda: (weights, None))
    body = client.get("/replay", params={"rate": 1000}).text
    kinds = [line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event")]
    # a (1 May, impact 9), b (1 May), weights for 4 May, c (2 May), end
    assert kinds == ["signal", "alert", "signal", "signal", "weights", "end"]


def test_replay_limit_and_ticker_filter(client, monkeypatch):
    weights = pd.DataFrame({"date": ["2020-05-04"], "JPM": [0.5], "BAC": [0.5]})
    monkeypatch.setattr(api, "module_a_outputs", lambda: (weights, None))
    body = client.get("/replay", params={"rate": 1000, "ticker": "bac", "limit": 1}).text
    assert body.count("event: signal") == 1
    assert '"doc_id": "c"' in body
    assert body.rstrip().endswith('data: {"texts": 1}')


def test_overview_counts_texts_events_and_alerts(client, monkeypatch):
    weights = pd.DataFrame({"date": ["2020-05-01", "2020-05-04"], "JPM": [1.0, 1.0]})
    monkeypatch.setattr(api, "module_a_outputs", lambda: (weights, None))
    out = client.get("/overview").json()
    assert out["texts"] == 3
    assert out["texts_by_source"] == {"news": 2, "tweets": 1}
    assert out["events"] == {"Default": 1, "Other": 1, "M&A": 1}
    assert out["alerts"] == 1  # only "a" has impact above 7
    assert out["trading_days"] == 2
    assert out["engine_models"] == {
        "sentiment": "finbert (baseline)",
        "events": "keywords (baseline)",
    }


def test_scoreboard_keeps_summary_fields_only(client, monkeypatch):
    row = {
        "model": "m", "n": 10, "accuracy": 0.9, "accuracy_ci95": [0.8, 0.95], "macro_f1": 0.85,
        "macro_f1_ci95": [0.7, 0.9], "texts_per_s": 50.0, "model_size_mb": 34.0,
        "per_class_f1": {"positive": 0.9}, "peak_rss_mb": 900.0,
    }  # fmt: skip
    monkeypatch.setattr(api, "eval_results", lambda: {"Sentiment": [row]})
    out = client.get("/scoreboard").json()
    assert list(out) == ["Sentiment"]
    assert "per_class_f1" not in out["Sentiment"][0]
    assert out["Sentiment"][0]["accuracy"] == 0.9


def test_prices_for_one_ticker_and_404_for_unknown(client, monkeypatch):
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-05-01", "2020-05-04", "2020-05-01"]),
            "ticker": ["JPM", "JPM", "BAC"],
            "close": [90.0, 91.5, 22.0],
        }
    )
    monkeypatch.setattr(api, "window_prices", lambda: rows)
    out = client.get("/prices", params={"ticker": "jpm"}).json()
    assert out == [{"date": "2020-05-01", "close": 90.0}, {"date": "2020-05-04", "close": 91.5}]
    assert client.get("/prices", params={"ticker": "XYZ"}).status_code == 404
