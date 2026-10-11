import io
import json
import zipfile

import httpx

from riskengine.ingest.gdelt import DATA_URL, DOC_URL, GdeltClient
from riskengine.ingest.linking import Linker
from riskengine.live import LiveFeed


def headline(text="Apple reports strong earnings", url="https://example.com/apple"):
    return {
        "text": text, "url": url, "published_at": "2026-10-11T12:00:00+00:00",
        "domain": "example.com", "source": "gdelt", "live_source": "gdelt_doc",
    }


def analyze(text):
    return {
        "tickers": ["AAPL"], "analysts": [], "sentiment": 0.8,
        "event": "Earnings", "impact": 8, "model": "test",
    }


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.fail = False

    def fetch(self):
        if self.fail:
            raise RuntimeError("offline")
        return self.rows, "gdelt_doc", None


def test_poll_deduplicates_by_url_and_headline_and_restores_cache(tmp_path):
    raw = headline()
    source = FakeClient([
        raw, {**raw, "url": "https://syndication.com/apple"},
        {**raw, "text": "Updated title"},
    ])
    path = tmp_path / "signals.jsonl"
    feed = LiveFeed(source, analyze, path)
    feed.poll_once()
    feed.poll_once()
    assert feed.snapshot()["total"] == 1
    assert len(path.read_text().splitlines()) == 1
    restored = LiveFeed(source, analyze, path)
    restored.poll_once()
    assert restored.snapshot()["total"] == 1
    assert restored.snapshot("aapl", 8)["high_impact"] == 1
    assert restored.snapshot("JPM")["signals"] == []
    assert restored.snapshot(min_impact=9)["signals"] == []


def test_network_failure_keeps_previous_signals(tmp_path):
    source = FakeClient([headline()])
    feed = LiveFeed(source, analyze, tmp_path / "live.jsonl")
    feed.poll_once()
    successful = feed.snapshot()["last_success"]
    source.fail = True
    feed.poll_once()
    assert feed.snapshot()["error"] == "offline"
    assert feed.snapshot()["last_success"] == successful
    assert feed.snapshot()["total"] == 1
    assert not feed.snapshot()["polling"]
    source.fail = False
    feed.poll_once()
    assert feed.snapshot()["error"] is None


def test_failed_analysis_can_be_retried(tmp_path):
    def broken(text):
        raise RuntimeError("missing model")

    feed = LiveFeed(FakeClient([headline()]), broken, tmp_path / "live.jsonl")
    feed.poll_once()
    assert feed.snapshot()["total"] == 0
    assert feed.snapshot()["error"] == "missing model"
    feed.analyze = analyze
    feed.poll_once()
    assert feed.snapshot()["total"] == 1


def test_doc_request_normalizes_and_filters_articles():
    def respond(request):
        assert str(request.url).startswith(DOC_URL)
        assert request.url.params["timespan"] == "24h"
        assert request.url.params["mode"] == "ArtList"
        rows = [
            {"title": "Apple earnings &amp; revenue rise", "url": "https://example.com/a",
             "seendate": "20261011T120000Z"},
            {"title": "Local football match", "url": "https://example.com/b",
             "seendate": "20261011T120000Z"},
            {"title": "Apple earnings", "url": "javascript:alert(1)",
             "seendate": "20261011T120000Z"},
            {"title": "Apple earnings", "url": "https://example.com/c", "seendate": "bad"},
        ]
        return httpx.Response(200, json={"articles": rows})

    client = GdeltClient(Linker.from_csv(), httpx.Client(transport=httpx.MockTransport(respond)))
    rows, source, warning = client.fetch()
    assert len(rows) == 1
    assert rows[0]["text"] == "Apple earnings & revenue rise"
    assert rows[0]["published_at"] == "2026-10-11T12:00:00+00:00"
    assert source == "gdelt_doc" and warning is None


def gkg_zip():
    fields = [""] * 27
    fields[1], fields[3], fields[4] = (
        "20261011120000", "example.com", "https://example.com/apple"
    )
    fields[26] = "<PAGE_TITLE>Apple earnings rise</PAGE_TITLE>"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("data.csv", "\t".join(fields) + "\n")
    return stream.getvalue()


def test_doc_throttling_falls_back_to_gkg_and_skips_missing_latest_file():
    requests = []

    def respond(request):
        url = str(request.url)
        requests.append(url)
        if url.startswith(DOC_URL):
            return httpx.Response(429)
        if url.endswith("lastupdate.txt"):
            return httpx.Response(200, text=f"123 456 {DATA_URL}20261011121500.gkg.csv.zip\n")
        if "121500" in url:
            return httpx.Response(404)
        return httpx.Response(200, content=gkg_zip())

    client = GdeltClient(Linker.from_csv(), httpx.Client(transport=httpx.MockTransport(respond)))
    rows, source, warning = client.fetch()
    assert rows[0]["text"] == "Apple earnings rise"
    assert source == "gdelt_gkg" and "fallback" in warning
    assert client.fetch()[0] == []
    assert sum("120000.gkg.csv.zip" in url for url in requests) == 1


def test_live_api_uses_feed_and_validates_filters(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from riskengine import api

    feed = LiveFeed(FakeClient([headline()]), analyze, tmp_path / "live.jsonl")
    feed.poll_once()
    monkeypatch.setattr(feed, "start", lambda: None)
    monkeypatch.setattr(feed, "refresh", lambda: None)
    monkeypatch.setattr(api, "live_feed", lambda: feed)
    client = TestClient(api.app)
    assert client.get("/live").json()["total"] == 1
    assert client.get("/live?ticker=JPM").json()["signals"] == []
    assert client.get("/live?min_impact=11").status_code == 422
    assert client.post("/live/refresh").json() == {"status": "scheduled"}
    assert json.loads(feed.path.read_text())["source"] == "gdelt"
