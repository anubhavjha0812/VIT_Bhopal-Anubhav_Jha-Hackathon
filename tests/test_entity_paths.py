"""Invented headlines and fake models check sentiment parity across entry points."""

import json
from dataclasses import replace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from riskengine import api
from riskengine.config import Config
from riskengine.ingest.linking import Linker
from riskengine.live import LiveFeed
from riskengine.nlp.impact import impact_formula
from riskengine.pipeline import _entity_scores, score_docs

TEXT = "Apple beats estimates, but Boeing faces a probe"
PAIR_SCORES = {"AAPL": 0.876543, "BA": -0.765432, "BAC": -0.456789, "TSLA": 0.654321}
HEADLINE_SCORE = -0.234567


class FakeScorer:
    def __init__(self, linker, kind="entity"):
        self.name = kind
        self.by_name = {linker.company_name(tk): score for tk, score in PAIR_SCORES.items()}
        self.pair_calls, self.headline_calls = [], []

    def score(self, texts):
        self.headline_calls.append(list(texts))
        return [HEADLINE_SCORE] * len(texts)

    def score_entities(self, texts, entities):
        assert self.name == "entity", "headline modes must not call the pair model"
        self.pair_calls.append(list(zip(entities, texts, strict=True)))
        return [self.by_name[name] for name in entities]


class FakeNewsClient:
    def fetch(self):
        return (
            [
                {
                    "text": TEXT,
                    "url": "https://example.com/invented-two-company-story",
                    "published_at": "2026-10-19T10:00:00+00:00",
                    "source": "gdelt",
                }
            ],
            "fixture",
            None,
        )


@pytest.fixture
def paths(monkeypatch, tmp_path):
    linker = Linker.from_csv()
    scorer = FakeScorer(linker)
    config = Config(sentiment="entity")
    monkeypatch.setattr(api, "linker", lambda: linker)
    monkeypatch.setattr(api, "scorer", lambda: scorer)
    monkeypatch.setattr(api, "load_config", lambda: config)
    monkeypatch.setattr(api, "event_classifier", lambda: lambda texts: ["Earnings"] * len(texts))
    monkeypatch.setattr(api, "GdeltClient", lambda linked: FakeNewsClient())
    monkeypatch.setattr(
        api, "LiveFeed", lambda client, analyze: LiveFeed(client, analyze, tmp_path / "live.jsonl")
    )
    api.live_feed.cache_clear()
    yield TestClient(api.app), scorer, linker, config
    api.live_feed.cache_clear()


def test_analyze_scores_each_linked_company_and_preserves_existing_fields(paths):
    client, scorer, linker, _ = paths
    response = client.post("/analyze", json={"text": TEXT})
    assert response.status_code == 200
    out = response.json()
    assert [row["ticker"] for row in out["entities"]] == list(linker.link(TEXT))
    assert {row["ticker"]: row["sentiment"] for row in out["entities"]} == {
        "AAPL": 0.8765,
        "BA": -0.7654,
    }
    assert out["tickers"] == ["AAPL", "BA"] and out["analysts"] == []
    assert out["sentiment"] == 0.8765  # first subject; not a pooled multi-company score
    assert out["sentiment_path"] == "entity"
    assert {row["sentiment_path"] for row in out["entities"]} == {"entity"}
    assert out["impact"] == impact_formula("Earnings", 0.8765, "news")
    assert out["event"] == "Earnings" and out["model"] == "entity+keywords+formula"
    assert out["analyst_rating"] is False
    assert scorer.pair_calls == [
        [(linker.company_name("AAPL"), TEXT), (linker.company_name("BA"), TEXT)]
    ]
    assert scorer.headline_calls == []  # known names do not load the fallback model


def test_analyze_live_cache_and_batch_use_identical_pair_scores(paths):
    client, scorer, linker, config = paths
    analyzed = client.post("/analyze", json={"text": TEXT}).json()
    feed = api.live_feed()  # exercise the real API -> LiveFeed -> analyze callback wiring
    feed.poll_once()
    assert feed.snapshot()["error"] is None
    live = feed.snapshot()["signals"][0]
    restored = LiveFeed(FakeNewsClient(), feed.analyze, feed.path)
    assert restored.snapshot()["signals"][0]["entities"] == live["entities"]
    assert json.loads(feed.path.read_text())["entities"] == analyzed["entities"]
    rows = pd.DataFrame({"text": [TEXT, TEXT], "ticker": ["AAPL", "BA"]})
    expected = _entity_scores(rows, scorer, linker)
    assert expected == [0.8765, -0.7654]
    for out in (analyzed, live):
        assert [row["sentiment"] for row in out["entities"]] == expected
        assert out["sentiment_path"] == "entity"
    docs = pd.DataFrame(
        {
            "doc_id": ["invented"],
            "published_at": pd.to_datetime(["2026-10-19T10:00:00Z"]),
            "source": ["news"],
            "text": [TEXT],
            "tickers": [list(linker.link(TEXT))],
        }
    )
    batch = score_docs(docs, config, scorer, lambda texts: ["Earnings"], linker)
    assert batch["sentiment"].tolist() == expected


def test_no_company_keeps_the_headline_fallback(paths):
    client, scorer, _, _ = paths
    out = client.post("/analyze", json={"text": "Central bank reviews monetary conditions"}).json()
    assert out["tickers"] == out["analysts"] == out["entities"] == []
    assert out["sentiment"] == round(HEADLINE_SCORE, 4)
    assert out["sentiment_path"] == "headline"
    assert out["impact"] == impact_formula("Earnings", HEADLINE_SCORE, "news")
    assert scorer.pair_calls == [] and len(scorer.headline_calls) == 1


@pytest.mark.parametrize("kind", ["own", "finbert"])
def test_headline_modes_preserve_existing_scores_and_impact(paths, kind, monkeypatch):
    client, _, linker, _ = paths
    scorer = FakeScorer(linker, kind)
    monkeypatch.setattr(api, "scorer", lambda: scorer)
    monkeypatch.setattr(api, "load_config", lambda: Config(sentiment=kind))
    out = client.post("/analyze", json={"text": TEXT, "source": "tweets"}).json()
    assert out["sentiment"] == round(HEADLINE_SCORE, 4)
    assert out["sentiment_path"] == "headline"
    assert [row["sentiment"] for row in out["entities"]] == [round(HEADLINE_SCORE, 4)] * 2
    assert {row["sentiment_path"] for row in out["entities"]} == {"headline"}
    assert out["impact"] == impact_formula("Earnings", HEADLINE_SCORE, "tweets")
    assert out["model"] == f"{kind}+keywords+formula"
    assert scorer.pair_calls == [] and scorer.headline_calls == [[TEXT]]


@pytest.mark.parametrize("known_tickers", [("AAPL",), ("BA",), ()])
def test_missing_company_name_reuses_the_batch_fallback(paths, monkeypatch, known_tickers):
    client, scorer, linker, _ = paths
    linker = replace(linker, company_names={tk: linker.company_name(tk) for tk in known_tickers})
    monkeypatch.setattr(api, "linker", lambda: linker)
    out = client.post("/analyze", json={"text": TEXT}).json()
    assert scorer.headline_calls == [[TEXT]]  # one fallback shared by all unnamed companies
    expected = _entity_scores(
        pd.DataFrame({"text": [TEXT, TEXT], "ticker": ["AAPL", "BA"]}), scorer, linker
    )
    assert [row["sentiment"] for row in out["entities"]] == expected
    assert [row["sentiment_path"] for row in out["entities"]] == [
        "entity" if tk in known_tickers else "headline" for tk in ("AAPL", "BA")
    ]
    assert out["sentiment"] == expected[0]
    assert out["sentiment_path"] == ("entity" if "AAPL" in known_tickers else "headline")


def test_analyst_bank_stays_separate_and_top_level_uses_the_subject(paths):
    client, _, linker, _ = paths
    text = "BofA Downgrades Tesla to Underperform"
    out = client.post("/analyze", json={"text": text}).json()
    assert out["tickers"] == ["TSLA"] and out["analysts"] == ["BAC"]
    assert [row["ticker"] for row in out["entities"]] == list(linker.link(text))
    assert [row["role"] for row in out["entities"]] == ["analyst", "subject"]
    assert out["sentiment"] == round(PAIR_SCORES["TSLA"], 4)
