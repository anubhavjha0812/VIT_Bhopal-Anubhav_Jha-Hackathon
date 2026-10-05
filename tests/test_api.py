import pandas as pd
import pytest
from fastapi.testclient import TestClient

from riskengine import api


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
