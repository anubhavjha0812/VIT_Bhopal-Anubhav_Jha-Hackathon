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
