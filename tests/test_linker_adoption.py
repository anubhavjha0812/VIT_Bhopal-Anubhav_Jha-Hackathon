"""Production selection checks use invented news and fake scorers only."""

from dataclasses import replace

import pytest

from riskengine import api
from riskengine.config import load_config
from riskengine.live import LiveFeed


def test_default_live_linker_selects_candidate_three():
    api.extended_linker.cache_clear()
    try:
        assert load_config().extended_linker is True
        selected = api.live_linker()
        assert selected.candidate == 3
        assert len(selected.tickers) == 99
        assert selected.link("Apple seeks permission for an engineering campus") == ("AAPL",)
        assert len(api.linker().tickers) == 15
    finally:
        api.extended_linker.cache_clear()


@pytest.mark.parametrize(
    "text,ticker,in_index",
    [
        ("Apple seeks permission for an engineering campus", "AAPL", True),
        ("Costco reports earnings", "COST", False),
    ],
)
def test_api_and_live_feed_use_the_selected_candidate(
    monkeypatch, tmp_path, text, ticker, in_index
):
    config = replace(load_config(), sentiment="own", events="keywords")
    monkeypatch.setattr(api, "load_config", lambda: config)

    class Scorer:
        name = "own"

        def score(self, texts):
            return [0.25] * len(texts)

    captured = []

    class Client:
        def __init__(self, linker):
            captured.append(linker)

        def fetch(self):
            return (
                [
                    {
                        "text": text,
                        "url": "https://example.com/invented-news",
                        "source": "gdelt",
                        "published_at": "2026-10-19T12:00:00Z",
                    }
                ],
                "fixture",
                None,
            )

    monkeypatch.setattr(api, "scorer", Scorer)
    monkeypatch.setattr(api, "event_classifier", lambda: lambda texts: ["Other"] * len(texts))
    monkeypatch.setattr(api, "GdeltClient", Client)
    monkeypatch.setattr(
        api, "LiveFeed", lambda client, analyze: LiveFeed(client, analyze, tmp_path / "feed.jsonl")
    )
    api.extended_linker.cache_clear()
    api.live_feed.cache_clear()
    try:
        selected = api.live_linker()
        assert selected.candidate == 3
        analyzed = api.analyze(api.AnalyzeIn(text=text))
        assert analyzed["tickers"] == [ticker]
        assert analyzed["entities"][0]["in_index"] is in_index
        feed = api.live_feed()
        feed.poll_once()
        assert captured == [selected]
        assert feed.snapshot()["error"] is None
        assert feed.snapshot()["signals"][0]["entities"] == analyzed["entities"]
    finally:
        api.live_feed.cache_clear()
        api.extended_linker.cache_clear()
