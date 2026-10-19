"""Invented frames exercise E44; no real sample, labels or models are read."""

import hashlib
import json
import re

import pandas as pd
import pytest

from riskengine.eval.e44_event_test import (
    RULES,
    Reservoir,
    blind_files,
    check,
    draw_clean,
    exclusion_inputs,
    key,
    readable,
    sample_once,
    specification,
)


def spec():
    return {
        "seed": 44,
        "sources": ["news.csv"],
        "columns": ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"],
        "date_start": "1999-01-01",
        "date_end": "2023-12-31",
        "chunk_rows": 3,
        "reservoir_per_hint": 3,
        "near_reference_block": 2,
        "exclusion_rounds": 10,
        "patterns": {
            "Covenant breach": r"\bcovenant\b",
            "Default": r"\bbankruptcy\b",
            "Product news": r"\bunveils\b",
        },
        "quotas": {"Covenant breach": 2, "Default": 2, "Product news": 2},
        "other_quota": 2,
        "baseline_cache": "baseline.json",
        "baseline_manifest": "baseline_manifest.json",
        "exclusion_specs": [["excluded.csv", "headline"]],
        "guide": "guide.md",
    }


def news(texts):
    return pd.DataFrame(
        {
            "Date": ["2012-06-01"] * len(texts),
            "Article_title": texts,
            "Stock_symbol": ["FAKE"] * len(texts),
            "Url": ["https://example.com/news"] * len(texts),
            "Publisher": ["Fixture"] * len(texts),
        }
    )


@pytest.fixture
def inputs(tmp_path):
    cfg = spec()
    prior = ["Unrelated preserved news about planetary astronomy observations"]
    (tmp_path / "baseline.json").write_text(json.dumps({"corpus": {"prior": prior}}))
    (tmp_path / "baseline_manifest.json").write_text(
        json.dumps(
            {
                "exclusions": {
                    "prior": {
                        "rows": len(prior),
                        "sha256": hashlib.sha256(
                            json.dumps(prior, ensure_ascii=False).encode()
                        ).hexdigest(),
                    }
                }
            }
        )
    )
    for name in ["final_news_to_label.csv", "final_tweets_to_label.csv"]:
        p = tmp_path / "data/gold/e24" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": []}).to_csv(p, index=False)
    pd.DataFrame(
        {"headline": ["Already inspected quarterly statement from an invented corporation"]}
    ).to_csv(tmp_path / "excluded.csv", index=False)
    (tmp_path / "guide.md").write_text("A fixed invented guide for fixture hashing.\n")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Fixed fixture protocol.\n```json\n" + json.dumps(cfg) + "\n```\n")
    texts = [
        "Fictional lender announces waiver of loan covenant today",
        "Another fictional lender confirms amended loan covenant terms",
        "Acme corporation files bankruptcy petition on Monday",
        "Example corporation reports bankruptcy filing in Delaware",
        "Imaginary company unveils novel imaging device for hospitals",
        "Fictional business unveils miniature camera for scientists",
        "Investors discuss possible changes at imaginary companies",
        "Stock traders discuss prospects for fictional businesses",
    ]
    news(texts).to_csv(tmp_path / "news.csv", index=False)
    return cfg, protocol, tmp_path / "sample"


def test_priority_uses_fixed_hints_not_labels():
    reservoir = Reservoir(spec(), set())
    assert reservoir.hint("Bankruptcy filing follows a covenant violation") == "Covenant breach"
    assert reservoir.hint("Bankruptcy court unveils a new procedure") == "Default"
    assert reservoir.hint("Company shares trade sideways after sales") == "Other"


def test_credit_hint_line_anchoring_is_equivalent_to_original_suffix_search():
    cfg = specification(RULES)
    original = re.compile(cfg["patterns"]["Credit downgrade"], re.I)
    fast = dict(Reservoir(cfg, set()).patterns)["Credit downgrade"]
    for company in ["Fitch", "Moody's", "S&P", "DBRS", "Standard Poor's", "Fictional"]:
        for action in [
            "cuts credit rating",
            "negative outlook",
            "reports results",
            "downgrades debt",
        ]:
            for separator in [" ", "\n", "\r\n"]:
                for text in [
                    f"{company}{separator}{action}",
                    f"{action}{separator}{company}",
                    f"Unrelated fictional company news{separator}{company} {action}",
                ]:
                    assert bool(fast.search(text)) == bool(original.search(text)), text


@pytest.mark.parametrize(
    "text",
    [
        "Too short",
        "<p>Company reports its quarterly earnings today</p>",
        "https://example.com a company reports quarterly results today",
        "Компания объявляет новый финансовый результат сегодня",
    ],
)
def test_readability_is_mechanical(text):
    assert not readable(text)


def test_chunk_boundaries_deduplication_and_caps_are_deterministic():
    cfg = spec()
    texts = [f"Fictional business unveils new imaging device variant {i}" for i in range(12)]
    frame = news(texts + [texts[0], texts[0] + "!"])
    a, b = Reservoir(cfg, set()), Reservoir(cfg, set())
    a.add(frame, "news.csv", "source-hash", 0)
    for start in range(0, len(frame), 4):
        b.add(frame.iloc[start : start + 4], "news.csv", "source-hash", start)
    pd.testing.assert_frame_equal(a.frame(), b.frame())
    assert len(a.frame()) == 3
    assert a.frame().key.is_unique
    assert a.occurrences["Product news"] == 14


def test_date_window_and_exact_prior_text_exclusions():
    text = "Fictional corporation unveils novel imaging devices today"
    frame = news(
        [
            text,
            "Known prior headline about fictional trading activity",
            "Outside window headline about fictional trading activity",
            "Malformed date headline about fictional trading activity",
        ]
    )
    frame.loc[2, "Date"], frame.loc[3, "Date"] = "2024-01-01", "invalid"
    reservoir = Reservoir(spec(), {key(frame.iloc[1].Article_title)})
    reservoir.add(frame, "news.csv", "source-hash", 0)
    assert reservoir.frame().text.tolist() == [text]
    assert reservoir.stats["exact_prior_occurrence_drops"] == 1
    assert reservoir.stats["invalid_or_outside_date"] == 2
    assert key("ABC, NEWS! https://example.com/z") == key("abc news")


def test_near_leak_is_removed_and_same_hint_refills():
    prior = (
        "Fictional corporation unveils a new advanced imaging device designed for medical "
        "research laboratories around the entire world"
    )
    near = prior + " today"
    safe = "Another business unveils an innovative toy for young children"
    frame = pd.DataFrame(
        {
            "text": [near, safe],
            "key": [key(near), key(safe)],
            "hint": ["Product news"] * 2,
            "draw_rank": ["a", "b"],
        }
    )
    cfg = spec()
    cfg["quotas"], cfg["other_quota"] = {"Product news": 1}, 0
    kept, audit = draw_clean(frame, cfg, {"earlier": [prior]})
    assert kept.text.tolist() == [safe]
    assert audit["near_dropped_unique"] == 1


def test_short_buckets_are_reported_without_padding():
    cfg = spec()
    reservoir = Reservoir(cfg, set())
    reservoir.add(
        news(["Fictional company files bankruptcy petition in court"]), "news.csv", "hash", 0
    )
    kept, audit = draw_clean(
        reservoir.frame(),
        cfg,
        {"prior": ["Astronomers record unusual observations on another planet"]},
    )
    assert len(kept) == 1
    assert audit["short_hint_buckets"]["Covenant breach"] == {"target": 2, "drawn": 0}


def test_sample_is_blind_checked_once_and_ignores_future_labels(tmp_path, inputs):
    _, protocol, work = inputs
    report = sample_once(tmp_path, protocol, work)
    blind = pd.read_csv(work / "sample.csv")
    assert list(blind.columns) == ["id", "text"]
    assert len(blind) == report["sample_rows"] == 8
    assert (work / "provenance.csv").stat().st_mode & 0o777 == 0o600
    assert (work / "sampling.json").stat().st_mode & 0o777 == 0o600
    # These invented future files must never be read by --check.
    (work / "labels_final.csv").write_text("invalid label file -- do not read")
    assert check(tmp_path, protocol, work)["check"] == "passed"
    with pytest.raises(FileExistsError, match="repeat draw refused"):
        sample_once(tmp_path, protocol, work)
    blind.loc[0, "text"] = "Tampered headline about a completely fictional business"
    blind.to_csv(work / "sample.csv", index=False)
    with pytest.raises(ValueError, match="sample hash changed"):
        check(tmp_path, protocol, work)


def test_check_rejects_changed_exclusion_and_protocol(tmp_path, inputs):
    _, protocol, work = inputs
    sample_once(tmp_path, protocol, work)
    pd.DataFrame({"headline": ["Changed prior headline about a fictional business"]}).to_csv(
        tmp_path / "excluded.csv", index=False
    )
    with pytest.raises(ValueError, match="exclusions changed"):
        check(tmp_path, protocol, work)
    protocol.write_text(protocol.read_text() + "Changed rules.\n")
    with pytest.raises(ValueError, match="protocol, guide or sampler hash changed"):
        check(tmp_path, protocol, work)


def test_required_exclusions_fail_closed_and_cached_snapshot_is_validated(tmp_path, inputs):
    cfg, _, _ = inputs
    (tmp_path / "excluded.csv").unlink()
    with pytest.raises(FileNotFoundError):
        exclusion_inputs(tmp_path, cfg)
    (tmp_path / "baseline.json").write_text('{"corpus":{"partial":[]}}')
    with pytest.raises(ValueError, match="every original exclusion group"):
        exclusion_inputs(tmp_path, cfg)


def test_blind_shuffle_does_not_depend_on_candidate_order():
    reservoir = Reservoir(spec(), set())
    reservoir.add(
        news(
            [
                "Acme company files bankruptcy petition in court",
                "Different corporation unveils miniature camera for scientists",
            ]
        ),
        "news.csv",
        "hash",
        0,
    )
    frame = reservoir.frame()
    a, ap = blind_files(frame, 44)
    b, bp = blind_files(frame.iloc[::-1], 44)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ap, bp)
