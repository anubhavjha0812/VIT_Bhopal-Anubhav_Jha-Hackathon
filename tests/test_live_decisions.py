"""Invented scored rows, fake weights and isolated journals; no network or inference."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from riskengine import ledger
from riskengine.config import Config, load_config
from riskengine.ingest.linking import Linker
from riskengine.live import LiveFeed
from riskengine.live_decisions import LiveDecisions, company_rows, decision_key, stress_preview
from riskengine.paths import DATA, ROOT


def invented(**changes):
    return {
        "text": "Fitch cuts Apple's issuer credit rating to BBB from A",
        "url": "https://example.test/invented-credit",
        "doc_id": "invented-credit",
        "published_at": "2026-10-19T09:00:00+00:00",
        "source": "gdelt",
        "live_source": "fixture",
        "tickers": ["AAPL"],
        "analysts": [],
        "sentiment": -0.8,
        "sentiment_path": "headline",
        "event": "Credit downgrade",
        "impact": 9,
        "model": "fake-weights+model_rules+formula",
        "ingested_at": (datetime.now(UTC) - timedelta(seconds=2)).isoformat(),
        **changes,
    }


@pytest.fixture
def setup(tmp_path):
    weights, book = tmp_path / "fake.onnx", tmp_path / "book.csv"
    weights.write_bytes(b"fixture weights, never loaded")
    book.write_text("ticker,ead\nAAPL,100\n")
    calls = []

    def preview(row, config, path):
        calls.append(row.copy())
        return {
            "dry_run": True,
            "module_a": "no change",
            "capital": {
                "loss_usd": 10.0,
                "cet1_before": 0.13,
                "cet1_after": 0.12,
            },
        }

    options = {
        "path": tmp_path / "live" / "decisions.jsonl",
        "ledger_path": tmp_path / "live" / "ledger.jsonl",
        "book_path": book,
        "files": {"fake/model.onnx": weights},
        "preview": preview,
    }
    config = Config(sentiment="own", events="model_rules", live_decisions=True)
    return config, options, calls


def flow(setup, **changes):
    config, options, _ = setup
    return LiveDecisions(config, Linker.from_csv(), **{**options, **changes})


def test_fired_row_preview_gate_provenance_and_ledger(setup):
    engine = flow(setup)
    row = invented()
    (rec,) = engine.process([row])
    assert rec["status"] == "decided" and rec["gate"]["fired"] is True
    assert rec["gate"]["reason"] == "fired: rating agency"
    assert rec["preview"]["capital"]["loss_usd"] == 10
    assert rec["module_a"] == "no change" and rec["execution"] == "preview only"
    assert rec["input"] == row
    assert len(rec["input_hash"]) == len(rec["model_hash"]) == len(rec["book_hash"]) == 64
    assert rec["code_version"] and len(rec["code_hash"]) == 64
    assert len(rec["config_hash"]) == 64 and rec["model_files_sha256"]
    entries = ledger.read(engine.ledger_path)
    assert ledger.verify(entries) == (True, None)
    assert len(entries) == 2 and entries[-1]["hash"] == rec["ledger_hash"]
    assert json.loads(engine.path.read_text()) == rec


def test_non_alert_is_skipped_with_no_preview(setup):
    engine = flow(setup)
    (rec,) = engine.process([invented(event="Earnings")])
    assert rec["status"] == "skipped" and rec["gate"]["reason"] == "not credit"
    assert rec["preview"] is None and setup[2] == []


def test_same_key_twice_and_after_restart_never_repeats(setup):
    row, engine = invented(), flow(setup)
    (first,) = engine.process([row])
    assert engine.process([row]) == []
    assert flow(setup).process([row]) == []
    assert len(engine.path.read_text().splitlines()) == 1
    assert len(setup[2]) == 1
    variant = {
        **row,
        "url": "https://other.test/syndicated",
        "text": "  " + row["text"].swapcase() + "  ",
    }
    assert decision_key(company_rows(row)[0], engine.provenance()) == first["key"]
    assert engine.process([variant]) == []


def test_error_retry_once_then_terminal(setup):
    attempts = []

    def temporary_failure(row, config, book):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("invented temporary preview failure")
        return {"dry_run": True, "loss_usd": 10}

    engine, row = flow(setup, preview=temporary_failure), invented()
    (error,) = engine.process([row])
    assert error["status"] == "error" and error["gate"]["fired"]
    assert error["error"] == "invented temporary preview failure"
    assert engine.health([row])["backlog"] == 1
    restarted = flow(setup, preview=temporary_failure)
    (decided,) = restarted.process([])  # error input survives outside the feed's cache
    assert decided["status"] == "decided" and decided["attempt"] == 2
    assert decided["key"] == error["key"]
    assert restarted.process([row]) == [] and len(attempts) == 2
    assert ledger.verify(ledger.read(engine.ledger_path)) == (True, None)


def test_recover_ledger_commit_after_interrupted_index_write(setup, monkeypatch):
    engine, row = flow(setup), invented()

    def disk_failure(rec):
        raise OSError("index write interrupted")

    monkeypatch.setattr(engine, "_append_index", disk_failure)
    assert engine.process([row]) == []
    assert engine.last_error == "index write interrupted"
    assert len(setup[2]) == 1 and not engine.path.exists()
    restarted = flow(setup)
    assert restarted.process([row]) == []
    assert len(setup[2]) == 1  # recover committed preview, do not recompute
    assert len(engine.path.read_text().splitlines()) == 1
    assert len(ledger.read(engine.ledger_path)) == 2


def test_concurrent_instances_write_one_terminal_decision(setup):
    engines, row = [flow(setup), flow(setup)], invented()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda engine: engine.process([row]), engines))
    assert sorted(map(len, results)) == [0, 1]
    assert len(setup[2]) == 1
    assert len(engines[0].path.read_text().splitlines()) == 1
    assert ledger.verify(ledger.read(engines[0].ledger_path)) == (True, None)


def test_off_flag_writes_nothing(setup):
    config, options, _ = setup
    engine = LiveDecisions(replace(config, live_decisions=False), Linker.from_csv(), **options)
    assert engine.process([invented()]) == []
    assert engine.health([])["enabled"] is False
    assert not engine.path.parent.exists() and not engine.ledger_path.exists()
    assert load_config().live_decisions is False
    with pytest.raises(ValueError, match="boolean"):
        Config(live_decisions="true")


def test_journal_paths_cannot_alias_or_target_historical_ledger(setup):
    with pytest.raises(ValueError, match="separate paths"):
        flow(setup, ledger_path=setup[1]["path"])
    with pytest.raises(ValueError, match="historical ledger"):
        flow(setup, path=ledger.LEDGER)


def test_supplied_entity_scores_and_roles_not_rescored(setup):
    engine = flow(setup)
    row = invented(
        text="Fitch cuts Apple's credit rating; JPMorgan comments",
        entities=[
            {"ticker": "JPM", "role": "analyst", "sentiment": -0.2, "sentiment_path": "entity"},
            {"ticker": "AAPL", "role": "subject", "sentiment": -0.6, "sentiment_path": "entity"},
        ],
    )
    records = engine.process([row])
    assert [r["status"] for r in records] == ["skipped", "decided"]
    assert records[0]["gate"]["reason"] == "analyst bank"
    assert records[1]["company"]["sentiment"] == -0.6
    assert setup[2][0]["impact"] == row["impact"]


def test_severity_and_e36b_rules_are_reused(setup):
    engine = flow(setup)
    (low,) = engine.process([invented(impact=10, impact_formula=6)])
    assert low["status"] == "skipped" and low["gate"]["reason"] == "impact ≤ 7"
    (blocked,) = engine.process(
        [
            invented(
                text="Fitch affirms Apple's credit rating; outlook stable",
                doc_id="affirmed",
            )
        ]
    )
    assert blocked["status"] == "skipped"
    assert blocked["gate"]["reason"].startswith("blocked:")


def test_corroboration_uses_earlier_distinct_context(setup):
    engine = flow(setup)
    early = invented(text="Apple defaults on debt payments", event="Default")
    later = invented(
        text="Apple misses a bond payment to lenders",
        event="Default",
        published_at="2026-10-19T09:30:00+00:00",
        doc_id="later",
    )
    results = engine.process([later, early])  # ingestion may be newest first
    by_text = {r["company"]["text"]: r for r in results}
    assert by_text[early["text"]]["status"] == "skipped"
    assert by_text[later["text"]]["status"] == "decided"
    assert by_text[later["text"]]["gate"]["reason"] == "fired: 2 sources in 24 h"


def test_retry_restores_original_corroboration_after_cache_eviction(setup):
    attempts = []

    def preview(row, config, book):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("invented preview failure")
        return {"dry_run": True}

    early = invented(text="Apple defaults on debt payments", event="Default")
    later = invented(
        text="Apple misses a bond payment",
        event="Default",
        published_at="2026-10-19T09:30:00+00:00",
    )
    original = flow(setup, preview=preview).process([early, later])
    assert [r["status"] for r in original] == ["skipped", "error"]
    (retried,) = flow(setup, preview=preview).process([])
    assert retried["status"] == "decided" and len(attempts) == 2
    assert retried["gate"]["reason"] == "fired: 2 sources in 24 h"


def test_changed_model_or_book_creates_new_audited_context(setup):
    engine, row = flow(setup), invented()
    (one,) = engine.process([row])
    setup[1]["files"]["fake/model.onnx"].write_bytes(b"different fixture weights")
    (two,) = engine.process([row])
    assert one["key"] != two["key"] and one["model_hash"] != two["model_hash"]
    setup[1]["book_path"].write_text("ticker,ead\nAAPL,101\n")
    (three,) = engine.process([row])
    assert three["key"] != two["key"] and three["book_hash"] != two["book_hash"]


def test_missing_provenance_and_invalid_row_are_explicit_errors(setup):
    missing = flow(setup, files={"missing": Path("/does-not-exist/fake.onnx")})
    (rec,) = missing.process([invented()])
    assert rec["status"] == "error" and rec["model_hash"] is None
    assert "model-file hash" in rec["error"] and rec["gate"]["fired"] is True
    engine = flow(setup)
    result = engine.process([invented(impact=float("nan")), invented(event="Earnings")])
    assert sorted(r["status"] for r in result) == ["error", "skipped"]


def test_edited_or_partial_journal_fails_visibly(setup):
    engine, row = flow(setup), invented()
    engine.process([row])
    entries = ledger.read(engine.ledger_path)
    entries[-1]["payload"]["preview"]["capital"]["loss_usd"] = 999
    ledger.write(entries, engine.ledger_path)
    assert ledger.verify(ledger.read(engine.ledger_path))[0] is False
    assert engine.process([row]) == [] and "verification failed" in engine.last_error
    assert engine.health([row])["backlog"] is None


def test_health_lag_errors_and_missing_values(setup):
    engine, row = flow(setup), invented()
    assert engine.health([row])["pending"] == 1
    engine.process([row])
    health = engine.health([row])
    assert health["backlog"] == health["error_count"] == 0
    assert health["lag"]["count"] == 1 and health["lag"]["median_seconds"] >= 2
    engine.process([invented(text="Apple reports profits", event="Earnings", ingested_at=None)])
    assert engine.health([])["lag"]["count"] == 1


def test_append_deduplicates_preserves_chain_and_protects_historical_file(tmp_path):
    path = tmp_path / "live.jsonl"
    rec = {"kind": "live_decision", "at": "2026-10-19", "payload": {"attempt_id": "key:1"}}
    first = ledger.append(rec, path=path, versions={"fixture": True})
    assert ledger.append(rec, path=path) == first
    ledger.append({**rec, "payload": {"attempt_id": "key:2"}}, path=path)
    assert len(ledger.read(path)) == 3 and ledger.verify(ledger.read(path)) == (True, None)
    with pytest.raises(ValueError, match="historical ledger"):
        ledger.append(rec, path=ledger.LEDGER)
    with path.open("a") as stream:
        stream.write('{"partial":')
    with pytest.raises(ValueError, match="Incomplete"):
        ledger.append(rec, path=path)


def test_default_preview_calls_existing_revaluation(setup, monkeypatch):
    from riskengine.module_b import report

    calls = []
    capital = {"loss_usd": 12, "cet1_before": 0.13, "cet1_after": 0.12}
    from types import SimpleNamespace

    def revalue(book, ticker, notches, episode, multiple, rates):
        calls.append((ticker, notches, episode, multiple, rates))
        return SimpleNamespace(positions=pd.DataFrame({"ticker": ["AAPL"], "loss": [12]}))

    monkeypatch.setattr(report, "revalue", revalue)
    monkeypatch.setattr(report, "_capital", lambda result: capital)
    cfg = replace(setup[0], shock="scaled", rates="moved")
    result = stress_preview({**invented(), "ticker": "AAPL"}, cfg, setup[1]["book_path"])
    assert calls[0][0] == "AAPL" and calls[0][-2:] == (0.9, "moved")
    assert result["dry_run"] and result["obligor_loss_usd"] == 12
    assert result["capital"] == capital


def test_flow_keeps_real_protected_files_byte_identical(setup):
    protected = [
        *sorted((DATA / "outputs").glob("*")),
        *sorted((ROOT / "src/riskengine/module_b").glob("*.py")),
        *[
            ROOT / p
            for p in (
                "src/riskengine/gate.py",
                "src/riskengine/gate_rules.py",
                "src/riskengine/module_a.py",
                "src/riskengine/pipeline.py",
                "src/riskengine/replay.py",
                "docs/others/DECISIONS.md",
                "docs/others/DECISION_MODEL_BENCH.md",
            )
        ],
        *sorted((ROOT / "docs/others").glob("LINKER_EXTENDED*.md")),
    ]

    def hashes():
        return {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()
        }

    before = hashes()
    flow(setup).process([invented(), invented(text="Apple reports profits", event="Earnings")])
    assert before == hashes()


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.fail = False

    def fetch(self):
        if self.fail:
            raise RuntimeError("invented source offline")
        return self.rows, "fixture", None


def fake_analyze(text):
    return {
        k: v
        for k, v in invented().items()
        if k
        in {
            "tickers",
            "analysts",
            "sentiment",
            "sentiment_path",
            "event",
            "impact",
            "model",
        }
    }


def test_feed_restarts_without_duplicate_and_retries_without_new_news(setup, tmp_path):
    attempts = []

    def preview(row, config, book):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("invented preview unavailable")
        return {"dry_run": True, "loss_usd": 10}

    row = invented()
    client = FakeClient([row])
    engine = flow(setup, preview=preview)
    feed = LiveFeed(client, fake_analyze, tmp_path / "source.jsonl", decisions=engine)
    feed.poll_once()
    assert feed.snapshot()["total"] == 1 and engine.health([])["error_count"] == 1
    client.fail = True
    feed.poll_once()  # source failure must not prevent journal retry
    assert engine.health([])["error_count"] == 0 and len(attempts) == 2
    restarted = LiveFeed(
        FakeClient([row]), fake_analyze, feed.path, decisions=flow(setup, preview=preview)
    )
    restarted.poll_once()
    assert len(attempts) == 2 and len(engine.path.read_text().splitlines()) == 2
    assert ledger.verify(ledger.read(engine.ledger_path)) == (True, None)


def test_feed_survives_decision_failure_and_off_flow_writes_no_decisions(setup, tmp_path):
    class BrokenDecisions:
        def process(self, rows):
            raise OSError("invented disk full")

    feed = LiveFeed(
        FakeClient([invented()]), fake_analyze, tmp_path / "feed.jsonl", decisions=BrokenDecisions()
    )
    feed.poll_once()
    assert feed.snapshot()["total"] == 1 and feed.snapshot()["error"] is None
    assert feed.decision_error == "invented disk full"
    config, options, _ = setup
    disabled = LiveDecisions(replace(config, live_decisions=False), Linker.from_csv(), **options)
    feed.decisions = disabled
    feed.poll_once()
    assert not disabled.path.exists() and not disabled.ledger_path.exists()


def test_restart_processes_persisted_unhandled_scores_even_when_source_is_offline(setup, tmp_path):
    row = invented()
    path = tmp_path / "feed.jsonl"
    LiveFeed(FakeClient([row]), fake_analyze, path).poll_once()
    engine = flow(setup)
    source = FakeClient([])
    source.fail = True
    restarted = LiveFeed(source, fake_analyze, path, decisions=engine)
    assert restarted.health()["decisions"]["pending"] == 1
    restarted.poll_once()
    assert engine.health([])["status_counts"] == {"decided": 1}
    assert restarted.snapshot()["error"] == "invented source offline"


def test_edited_index_is_not_silently_repaired(setup):
    engine, row = flow(setup), invented()
    engine.process([row])
    record = json.loads(engine.path.read_text())
    record["status"] = "skipped"
    engine.path.write_text(json.dumps(record) + "\n")
    assert engine.process([row]) == []
    assert "index differs" in engine.last_error
    assert len(setup[2]) == 1


def test_health_missing_source_and_metrics_do_not_start_poll(tmp_path):
    client = FakeClient([])
    feed = LiveFeed(client, fake_analyze, tmp_path / "source.jsonl")
    health = feed.health()
    assert health["last_poll"] is None and health["newest_item_age_seconds"] is None
    assert health["source_freshness"]["status"] == "missing"
    assert health["reference_metrics"]["precision"] == "not available"
    assert health["reference_metrics"]["ECE"] == "not available"
    assert not feed.path.exists()


def test_api_opt_in_wires_same_analyze_linker_and_read_only_health(setup, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from riskengine import api, live_decisions

    config, _, _ = setup
    engine = flow(setup)
    scorer_calls = []

    class FakeScorer:
        name = "own"

        def score(self, texts):
            scorer_calls.extend(texts)
            return [-0.8] * len(texts)

    def client_factory(linker):
        assert linker is engine.linker
        return FakeClient([invented()])

    monkeypatch.setattr(api, "load_config", lambda: config)
    monkeypatch.setattr(api, "linker", lambda: engine.linker)
    monkeypatch.setattr(api, "GdeltClient", client_factory)
    monkeypatch.setattr(api, "scorer", lambda: FakeScorer())
    monkeypatch.setattr(
        api, "event_classifier", lambda: lambda texts: ["Credit downgrade"] * len(texts)
    )
    monkeypatch.setattr(
        api, "LiveFeed", lambda client, analyze: LiveFeed(client, analyze, tmp_path / "feed.jsonl")
    )
    monkeypatch.setattr(live_decisions, "LiveDecisions", lambda cfg, linker: engine)
    api.live_feed.cache_clear()
    try:
        client = TestClient(api.app)
        health = client.get("/live/health")
        assert health.status_code == 200 and health.json()["decisions"]["enabled"] is True
        assert scorer_calls == [] and not engine.path.exists()
        feed = api.live_feed()
        feed.poll_once()
        result = client.get("/live/health").json()
        assert result["decisions"]["status_counts"] == {"decided": 1}
        assert result["decisions"]["backlog"] == 0
        assert result["recent_scores"][0]["model"] == "own+model_rules+formula"
        assert result["recent_scores"][0]["sentiment_path"] == "headline"
        assert len(scorer_calls) == 1  # decision processing never calls a scorer again
    finally:
        api.live_feed.cache_clear()
