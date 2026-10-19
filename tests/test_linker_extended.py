"""Invented fixtures only; the batch digest was captured before the extension."""

import hashlib
import json

import pandas as pd
import pytest

from riskengine import api
from riskengine.config import Config
from riskengine.ingest.linking import Linker
from riskengine.ingest.linking_extended import EXTENDED_PATH, ExtendedLinker
from riskengine.paths import DATA
from riskengine.pipeline import score_docs


class BatchScorer:
    name = "entity"

    def score_entities(self, texts, entities):
        return [0.12567 if name == "Apple Inc." else -0.314159 for name in entities]

    def score(self, texts):
        return [0.76543] * len(texts)


def batch_digest(extended_flag=False):
    old = Linker.from_csv()
    texts = ["Apple earnings rise as Costco shares fall", "BofA Downgrades Tesla to Sell"]
    docs = pd.DataFrame(
        {
            "doc_id": ["invented-a", "invented-b"],
            "published_at": pd.to_datetime(["2020-05-01T10:00:00Z"] * 2),
            "source": ["news"] * 2,
            "text": texts,
            "tickers": [old.link(text) for text in texts],
        }
    )
    out = score_docs(
        docs,
        Config(sentiment="entity", extended_linker=extended_flag),
        BatchScorer(),
        lambda texts: ["Earnings"] * len(texts),
    )
    raw = out.to_json(orient="records", date_format="iso").encode()
    return hashlib.sha256(raw).hexdigest(), out


def test_default_batch_keeps_pre_extension_digest():
    digest, out = batch_digest()
    assert digest == "fe7badc6afaabe08a36adf785c37e1df4f32db8775cf17fc714bd8f26e8bea3c"
    assert batch_digest(extended_flag=True)[0] == digest
    assert out.ticker.tolist() == ["AAPL", "BAC", "TSLA"]
    assert len(Linker.from_csv().tickers) == 15
    assert "COST" not in Linker.from_csv().link("Costco shares rise")
    assert Linker.from_csv().link("Apple harvest") == ("AAPL",)


def test_catalog_matches_local_prices_and_preserves_original_fields():
    original = pd.read_csv(DATA / "reference/universe.csv", dtype=str)
    extended = pd.read_csv(EXTENDED_PATH, dtype=str)
    pd.testing.assert_frame_equal(extended.iloc[:15][original.columns], original)
    priced = set(pd.read_csv(DATA / "prices/prices_fresh.csv", usecols=["ticker"]).ticker)
    assert set(extended.ticker) == set(original.ticker) | (priced - {"^GSPC", "^VIX"})
    assert len(extended) == 99
    linked = ExtendedLinker.from_csv()
    assert linked.index_members == frozenset(original.ticker)
    assert extended.iloc[:15].in_index.eq("true").all()
    assert extended.iloc[15:].in_index.eq("false").all()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Apple harvest boosts farm earnings", ()),
        ("Apple Inc. harvest festival", ("AAPL",)),
        ("Amazon river threatens stores", ()),
        ("Amazon retail earnings improve", ("AMZN",)),
        ("Target price raised for oil stocks", ()),
        ("Target Corporation opens a shop", ("TGT",)),
        ("Visa applications strain banks", ()),
        ("Visa Inc. travel announcement", ("V",)),
        ("Ford the river near the bank", ()),
        ("Ford Motor Company opens a plant", ("F",)),
        ("Oracle of Omaha discusses earnings", ()),
        ("Oracle Corporation introduces a product", ("ORCL",)),
        ("Capital One is the title", ()),
        ("Capital One bank deposits rise", ("COF",)),
        ("Caterpillar larvae damage equipment", ()),
        ("Caterpillar machinery orders rise", ("CAT",)),
        ("Dow Jones stock index falls", ()),
        ("Dow Inc. reports sales", ("DOW",)),
        ("Gap in coverage and $GPS", ()),
        ("Chevron announces a project", ("CVX",)),
        ("$aapl and $f and $v", ("AAPL", "F", "V")),
        ("A pineapple beside a target", ()),
    ],
)
def test_guarded_company_names(text, expected):
    assert ExtendedLinker.from_csv().link(text) == expected


def test_names_can_be_disabled_and_original_loader_accepts_an_explicit_path():
    assert ExtendedLinker.from_csv().link("Costco $F", use_names=False) == ("F",)
    assert "COST" in Linker.from_csv(EXTENDED_PATH).link("Costco")


def test_extra_broker_desks_reuse_original_exclusion_and_cashtag_override():
    from riskengine.eval.linker_extended import subject_links

    linked = ExtendedLinker.from_csv()
    assert subject_links(linked, "Goldman Sachs upgrades Costco to Buy") == {"COST"}
    assert subject_links(linked, "$GS upgrades Costco to Buy") == {"GS", "COST"}
    assert subject_links(linked, "Wells Fargo analysts discuss Microsoft") == {"MSFT"}
    assert subject_links(linked, "Morgan Stanley earnings improve") == {"MS"}


def test_live_api_flag_index_metadata_and_stored_gate_isolation(monkeypatch, tmp_path):
    from riskengine.live import LiveFeed

    old, extended = Linker.from_csv(), ExtendedLinker.from_csv()
    monkeypatch.setattr(api, "linker", lambda: old)
    monkeypatch.setattr(api, "extended_linker", lambda: extended)
    monkeypatch.setattr(api, "load_config", lambda: Config(extended_linker=True))
    monkeypatch.setattr(api, "scorer", BatchScorer)
    monkeypatch.setattr(api, "event_classifier", lambda: lambda texts: ["Other"] * len(texts))
    text = "Costco shares rise alongside Apple earnings"
    out = api.analyze(api.AnalyzeIn(text=text))
    assert {e["ticker"]: e["in_index"] for e in out["entities"]} == {"AAPL": True, "COST": False}
    assert all(not {"exposure", "book", "portfolio"} & e.keys() for e in out["entities"])
    captured = []

    class Client:
        def __init__(self, linker):
            captured.append(linker)

        def fetch(self):
            return (
                [
                    {
                        "text": text,
                        "url": "https://example.com/fiction",
                        "source": "gdelt",
                        "published_at": "2026-10-19T10:00:00Z",
                    }
                ],
                "fixture",
                None,
            )

    monkeypatch.setattr(api, "GdeltClient", Client)
    monkeypatch.setattr(
        api, "LiveFeed", lambda client, analyze: LiveFeed(client, analyze, tmp_path / "feed.jsonl")
    )
    api.live_feed.cache_clear()
    try:
        feed = api.live_feed()
        feed.poll_once()
        assert captured == [extended]
        assert feed.snapshot()["signals"][0]["entities"] == out["entities"]
    finally:
        api.live_feed.cache_clear()

    monkeypatch.setattr(api, "stored_signals", lambda: pd.DataFrame())
    monkeypatch.setattr(api, "gate", lambda rows, threshold, linker: linker)
    assert api.gated_signals() is old
    monkeypatch.setattr(api, "load_config", lambda: Config(extended_linker=False))
    assert api.live_linker() is old
    assert api.analyze(api.AnalyzeIn(text="Costco shares rise"))["entities"] == []


def test_boolean_flag_is_validated():
    with pytest.raises(ValueError, match="boolean"):
        Config(extended_linker="false")


def test_blind_sampling_is_balanced_unique_and_order_independent():
    from riskengine.eval.linker_extended import draw

    frame = pd.DataFrame(
        [
            {"text": "One company", "date": "2020", "symbols": ["FB"], "cache": "history"},
            {"text": "One company!", "date": "2021", "symbols": ["OTHER"], "cache": "fresh"},
            {"text": "Second company", "date": "2020", "symbols": ["AAPL"], "cache": "fresh"},
            {"text": "Third company", "date": "2020", "symbols": ["OUT"], "cache": "history"},
            {"text": "Fourth company", "date": "2020", "symbols": ["OUT"], "cache": "history"},
        ]
    )
    a, counts = draw(frame, {"META", "AAPL"}, n=2)
    b, _ = draw(frame.iloc[::-1], {"META", "AAPL"}, n=2)
    pd.testing.assert_frame_equal(a, b)
    assert counts == {"inside": 2, "outside": 2}
    assert a.stratum.value_counts().to_dict() == {"inside": 2, "outside": 2}
    assert a.key.is_unique
    assert "gold_entities" not in a
    with pytest.raises(ValueError, match="insufficient"):
        draw(frame, {"AAPL", "META"}, n=3)


def test_metrics_do_not_reward_abstention_and_decision_is_fixed():
    from riskengine.eval.linker_extended import decision, metrics

    scores = metrics([{"A", "B"}, {"C"}, set()], [{"A", "X"}, set(), set()])
    assert scores["precision"] == 0.5 and scores["recall"] == 0.25
    assert scores["wrong_links"] == 1 and scores["empty_both"] == 1
    assert scores["f1"] == 0.25 and scores["micro_recall"] == 1 / 3
    assert metrics([{"A"}], [set()])["precision"] is None
    old = {"precision": 1.0, "recall": 0.3}
    assert decision(old, {"precision": 0.95, "recall": 0.4}, 0)
    assert not decision(old, {"precision": 0.949, "recall": 0.4}, 0)
    assert not decision(old, {"precision": 0.99, "recall": 0.4}, 1)
    assert not decision(old, {"precision": 1.0, "recall": 0.3}, 0)


def test_completed_labels_and_repeat_run_are_checked_before_scoring(tmp_path):
    from riskengine.eval.linker_extended import load_labels, score_once

    sample = tmp_path / "labels.csv"
    pd.DataFrame(
        [{"id": "a", "text": "fiction", "gold_entities": "[]", "rationale": "No firm"}]
    ).to_csv(sample, index=False)
    frame, gold = load_labels(sample, expected=1)
    assert len(frame) == 1 and gold == [set()]
    with pytest.raises(ValueError, match="count"):
        load_labels(sample)
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"already": "scored"}))
    with pytest.raises(FileExistsError, match="second run"):
        score_once(result=result, work=tmp_path)
