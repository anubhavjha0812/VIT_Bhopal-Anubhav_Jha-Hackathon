"""Keyword-only held-out sampling: small synthetic frames, no gate or model calls."""

import hashlib

import pandas as pd

from riskengine.eval import credit_heldout
from riskengine.eval.credit_heldout import assignments, clean_remainder, draw, without_used


def sample_frame(texts):
    return pd.DataFrame(
        {
            "text": texts,
            "tickers": "AAA",
            "published_at": pd.date_range("2012-01-01", periods=len(texts), tz="UTC"),
        }
    )


def test_assignments_are_disjoint_and_negative_patterns_take_priority():
    texts = pd.Series(
        [
            "Fitch downgrades issuer debt",
            "Acme files Chapter 11 bankruptcy",
            "Acme requests covenant waiver",
            "Acme undertakes distressed restructuring",
            "Fitch cuts ratings on classes of CMBS notes",
            "Bank downgrades Acme to Sell",
            "Lender has default exposure",
            "Bank finances another borrower",
            "Acme opens a shop",
        ]
    )
    got = assignments(texts)
    assert got.sampling_stratum.tolist() == [
        "rating_action",
        "default_bankruptcy",
        "covenant_waiver",
        "restructuring_distress",
        "hard_negative_pattern",
        "hard_negative_pattern",
        "hard_negative_pattern",
        "hard_negative_pattern",
        "",
    ]
    assert got.negative_pattern.iloc[4:8].tolist() == [
        "securitization",
        "stock_rating",
        "counterparty",
        "wrong_company_pattern",
    ]


def test_draw_is_deterministic_has_target_counts_and_no_duplicate_texts():
    f = sample_frame(
        [
            "Fitch downgrades Cedar debt",
            "Moody's cuts Elm outlook",
            "Acme files Chapter 11",
            "Baker defaults on loan payments",
            "Cobalt seeks covenant waiver",
            "Delta needs distressed restructuring",
            "Fitch cuts CMBS classes of notes",
            "Bank downgrades Orbit to Sell",
        ]
    )
    targets = {
        "rating_action": 2,
        "default_bankruptcy": 2,
        "covenant_waiver": 1,
        "restructuring_distress": 1,
        "hard_negative_pattern": 2,
    }
    first = draw(f, targets, seed=3)
    assert first.equals(draw(f, targets, seed=3))
    assert first.equals(draw(f.iloc[::-1], targets, seed=3))
    assert first.sampling_stratum.value_counts().to_dict() == targets
    assert first.text.is_unique


def test_draw_fills_scarce_credit_strata_without_increasing_negative_quota():
    f = sample_frame(
        [
            "Fitch cuts Cedar",
            "Moody's reviews Elm",
            "Fitch downgrades Birch",
            "Bank downgrades Orbit to Sell",
            "Bank upgrades Lake to Buy",
        ]
    )
    picked = draw(
        f, {"rating_action": 1, "default_bankruptcy": 2, "hard_negative_pattern": 1}, seed=4
    )
    assert len(picked) == 4
    assert (picked.sampling_stratum == "hard_negative_pattern").sum() == 1


def test_without_used_removes_normalized_pool_and_extra_matches():
    f = sample_frame(["Acme DEFAULTS!", "Baker seeks waiver", "Cedar files bankruptcy"])
    out = without_used(f, ["acme defaults"], ["Baker seeks waiver"])
    assert out.text.tolist() == ["Cedar files bankruptcy"]


def test_clean_remainder_excludes_pool_extra_and_prior_corpora():
    texts = [
        "Acme defaults on debt",
        "Baker covenant waiver",
        "Cedar bankrupt today",
        "Fitch cuts Falcon credit rating",
    ]
    h = sample_frame(texts).rename(columns={"tickers": "ticker"}).assign(why=None, role="subject")
    clean, _, audit = clean_remainder(h, {"reserved": [texts[2]]}, [texts[0]], [texts[1]])
    assert clean.text.tolist() == [texts[3]]
    assert audit["frame_rows"] == 4 and audit["clean_remainder_rows"] == 1


def test_build_reads_identity_columns_only_and_writes_an_unlabelled_sheet(tmp_path, monkeypatch):
    from riskengine import history
    from riskengine.eval import e24

    source = tmp_path / "history.csv"
    sample_frame(
        ["Fitch downgrades Cedar debt", "Acme files bankruptcy", "Bank downgrades Orbit to Sell"]
    ).rename(columns={"tickers": "ticker"}).assign(
        why=None,
        role="subject",
        event="must not read",
        fired="must not read",
        label="must not read",
    ).to_csv(source, index=False)
    pool, extra = tmp_path / "pool.csv", tmp_path / "extra.csv"
    pd.DataFrame({"text": ["Previously published phone specifications"]}).to_csv(pool, index=False)
    pd.DataFrame({"text": ["Unrelated wheat harvest weather"]}).to_csv(extra, index=False)
    monkeypatch.setattr(history, "HISTORY_PATH", source)
    monkeypatch.setattr(credit_heldout, "POOL_PATH", pool)
    monkeypatch.setattr(credit_heldout, "EXTRA_PATH", extra)
    monkeypatch.setattr(credit_heldout, "ROOT", tmp_path)
    monkeypatch.setattr(credit_heldout, "SAMPLE", tmp_path / "to_label.csv")
    monkeypatch.setattr(credit_heldout, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(
        credit_heldout,
        "TARGETS",
        {
            "rating_action": 1,
            "default_bankruptcy": 1,
            "hard_negative_pattern": 1,
        },
    )
    monkeypatch.setattr(
        e24, "exclusion_corpus", lambda: {"reserved": ["Completely separate event"]}
    )
    monkeypatch.setattr(credit_heldout, "exclusions", lambda corpus: corpus)
    original = pd.read_csv

    def checked_read(path, *args, **kwargs):
        if path == source:
            assert kwargs["usecols"] == credit_heldout.HISTORY_COLUMNS
        elif path in (pool, extra):
            assert kwargs["usecols"] == ["text"]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", checked_read)
    dry = credit_heldout.build(write=False)
    assert dry["sample_rows"] == 3 and not credit_heldout.SAMPLE.exists()
    result = credit_heldout.build(write=True)
    sheet = original(credit_heldout.SAMPLE)
    assert sheet.columns.tolist() == ["id", "text", "date", "tickers"]
    assert sheet.id.is_unique and len(sheet) == 3
    assert result["sample_sha256"] == hashlib.sha256(credit_heldout.SAMPLE.read_bytes()).hexdigest()
    assert not result["model_or_gate_run"] and not result["sample_has_labels_or_model_outputs"]
