"""E24 gate: fails closed on every broken precondition (synthetic files in a temp folder)."""

import hashlib
import json

import pandas as pd
import pytest

from riskengine.eval.e24_gate import (
    CLASSES,
    PINNED,
    GateError,
    Paths,
    _run_final,
    check_candidate,
    check_frozen,
    check_record,
    freeze_candidate,
    load_dev_labels,
    paired_bootstrap,
    run_final,
    section_sha,
    validate_labels,
    verdict,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def p(tmp_path):
    gold, outputs, backup = tmp_path / "gold", tmp_path / "outputs", tmp_path / "backup"
    for d in (gold, outputs, backup):
        d.mkdir()
    news = ["Apple unveils a phone", "Fed raises rates", "Exxon shares fall"]
    pd.DataFrame(
        {
            "item_id": ["N000", "N001", "N002"],
            "text": news,
            "date": "2017-01-02",
            "companies": "AAPL",
        }
    ).to_csv(gold / "final_news_to_label.csv", index=False)
    pd.DataFrame({"item_id": ["T000", "T001"], "text": ["$TSLA earnings beat", "lol"],
                  "date": "2022-01-02", "companies": "TSLA"}).to_csv(
        gold / "final_tweets_to_label.csv", index=False)  # fmt: skip
    pd.DataFrame({"doc_id": ["d1", "d2", "d3"], "split": ["D1", "D2", "D1"]}).to_csv(
        gold / "e3_split.csv", index=False
    )
    (gold / "GUIDE.md").write_text("guide v1\n")
    decisions = tmp_path / "DECISIONS.md"
    decisions.write_text("# x\n\n## E24. Test entry\n- rule\n\n## E4. Next\n")
    files = {n: sha(gold / n) for n in ("final_news_to_label.csv", "final_tweets_to_label.csv",
                                        "e3_split.csv")}  # fmt: skip
    (outputs / "e24_final_sample.json").write_text(json.dumps({"files_sha256": files}))
    lines = []
    for n in files:
        (backup / n).write_bytes((gold / n).read_bytes())
        lines.append(f"{files[n]}  {n}")
    (backup / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    paths = Paths(gold=gold, outputs=outputs, backup=backup, decisions=decisions, root=tmp_path)
    paths.section_sha = section_sha(decisions)
    return paths


def labels(ids, label="Other", source="ai_agreed"):
    return pd.DataFrame({"item_id": ids, "label": label, "ambiguous": "N",
                         "rationale": "the words say so", "label_source": source})  # fmt: skip


# --- frozen sample ---


def test_frozen_sample_passes_when_untouched(p):
    assert check_frozen(p)["files_sha256"]


def test_hash_mismatch_fails_closed(p):
    (p.gold / "final_news_to_label.csv").write_text("item_id,text\nN000,changed\n")
    with pytest.raises(GateError, match="hash mismatch"):
        check_frozen(p)


def test_changed_decisions_fails_closed(p):
    p.decisions.write_text(p.decisions.read_text().replace("- rule", "- rule, edited"))
    with pytest.raises(GateError, match="DECISIONS"):
        check_frozen(p)


def test_missing_backup_fails_closed(p):
    (p.backup / "e3_split.csv").unlink()
    with pytest.raises(GateError, match="backup"):
        check_frozen(p)


# --- malformed labels ---


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda d: d.drop(columns=["rationale"]), "lacks columns"),
        (lambda d: d.assign(label="Weather"), "outside the 12"),
        (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate"),
        (lambda d: d.iloc[:2], "do not match"),
        (lambda d: d.assign(ambiguous="maybe"), "Y or N"),
        (lambda d: d.assign(rationale=""), "empty rationale"),
        (lambda d: d.assign(label_source="gpt"), "label_source"),
    ],
)
def test_malformed_labels_are_rejected(mutate, msg):
    good = labels(["N000", "N001", "N002"])
    validate_labels(good, ["N000", "N001", "N002"])
    with pytest.raises(GateError, match=msg):
        validate_labels(mutate(good), ["N000", "N001", "N002"])


def test_excluded_rows_need_no_label():
    d = labels(["N000", "N001"])
    d.loc[1, ["label", "rationale", "label_source"]] = ["", "", "excluded"]
    validate_labels(d, ["N000", "N001"])


# --- leakage between final labels and development ---


def test_dev_loader_refuses_final_items(p, tmp_path):
    path = tmp_path / "d1_labels.csv"
    labels(["d1", "d3", "N000"]).to_csv(path, index=False)
    with pytest.raises(GateError, match="final-set items"):
        load_dev_labels(path, "D1", p)


def test_dev_loader_refuses_final_files_and_other_splits(p, tmp_path):
    path = tmp_path / "final_news_labels.csv"
    labels(["d1", "d3"]).to_csv(path, index=False)
    with pytest.raises(GateError, match="final labels cannot"):
        load_dev_labels(path, "D1", p)
    ok = tmp_path / "d1_labels.csv"
    labels(["d1", "d3"]).to_csv(ok, index=False)
    assert len(load_dev_labels(ok, "D1", p)) == 2
    with pytest.raises(GateError, match="D1 or D2"):
        load_dev_labels(ok, "final", p)


# --- annotation records (missing metadata) ---


def write_record(p, labeller, set_name, **over):
    prompt = p.root / f"{labeller}_prompt.md"
    prompt.write_text("prompt\n")
    raw = p.root / f"{labeller}_{set_name}_raw.csv"
    raw.write_text("item_id,label\n")
    rec = {"labeller": labeller, "set": set_name, "model_as_shown_by_interface": "not shown",
           "interface": "web app", "annotation_date_utc": "2099-01-01T00:00:00Z",
           "settings": "web default, not controllable", "prompt_file": prompt.name,
           "prompt_sha256": sha(prompt), "guide_sha256": sha(p.gold / "GUIDE.md"),
           "input_file_sha256": sha(p.gold / f"{set_name}_to_label.csv"),
           "raw_output_file": raw.name, "raw_output_sha256": sha(raw)} | over  # fmt: skip
    (p.gold / "records").mkdir(exist_ok=True)
    path = p.gold / "records" / f"{labeller}_{set_name}.json"
    path.write_text(json.dumps(rec))
    return path


def test_record_with_missing_metadata_fails(p):
    path = write_record(p, "chatgpt", "final_news", model_as_shown_by_interface=None)
    with pytest.raises(GateError, match="lacks"):
        check_record(path, p, "chatgpt", "final_news")


def test_record_with_wrong_prompt_hash_fails(p):
    path = write_record(p, "gemini", "final_news", prompt_sha256="0" * 64)
    with pytest.raises(GateError, match="prompt_sha256"):
        check_record(path, p, "gemini", "final_news")


# --- frozen candidate ---


def test_candidate_rules(p, tmp_path):
    rules = tmp_path / "rules.py"
    rules.write_text("RULES = []\n")
    with pytest.raises(GateError, match="threshold from the fixed grid"):
        freeze_candidate(p, "D", 0.42, None, rules)
    with pytest.raises(GateError, match="only candidate D"):
        freeze_candidate(p, "C", 0.5, None, rules)
    freeze_candidate(p, "D", 0.5, 1.3, rules)
    assert check_candidate(p)["threshold"] == 0.5
    with pytest.raises(GateError, match="already frozen"):
        freeze_candidate(p, "A", None, None, None)
    rules.write_text("RULES = ['changed']\n")
    with pytest.raises(GateError, match="rules file changed"):
        check_candidate(p)


def test_freeze_refused_once_final_labels_exist(p):
    labels(["N000", "N001", "N002"]).to_csv(p.gold / "final_news_labels.csv", index=False)
    with pytest.raises(GateError, match="freeze must come first"):
        freeze_candidate(p, "A", None, None, None)


# --- K5: freeze before final annotation, UTC timestamps ---


@pytest.mark.parametrize(
    "where", ["records/chatgpt_final_news.json", "raw/gemini_final_tweets.csv"]
)
def test_freeze_refused_once_final_records_or_raw_outputs_exist(p, where):
    f = p.gold / where
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("{}")
    with pytest.raises(GateError, match="freeze must come first"):
        freeze_candidate(p, "A", None, None, None)


def test_freeze_ignores_dev_records(p):
    f = p.gold / "records" / "chatgpt_D1.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("{}")
    frozen = freeze_candidate(p, "A", None, None, None)
    assert frozen["frozen_at_utc"].endswith("Z")
    assert check_candidate(p)["frozen_at_utc"] == frozen["frozen_at_utc"]


def test_final_record_dated_before_the_freeze_is_rejected(p):
    freeze_candidate(p, "A", None, None, None)
    path = write_record(p, "chatgpt", "final_news", annotation_date_utc="2020-01-01T00:00:00Z")
    with pytest.raises(GateError, match="before the candidate was frozen"):
        check_record(path, p, "chatgpt", "final_news")


def test_final_record_needs_a_frozen_candidate(p):
    path = write_record(p, "chatgpt", "final_news")
    with pytest.raises(GateError, match="frozen candidate"):
        check_record(path, p, "chatgpt", "final_news")


@pytest.mark.parametrize(
    "stamp", ["2099-01-01", "2099-01-01T00:00:00", "2099-01-01T05:30:00+05:30", "soon", 20991231]
)
def test_malformed_or_non_utc_timestamps_fail_closed(p, stamp):
    freeze_candidate(p, "A", None, None, None)
    path = write_record(p, "gemini", "final_tweets", annotation_date_utc=stamp)
    with pytest.raises(GateError, match="timestamp"):
        check_record(path, p, "gemini", "final_tweets")


def test_candidate_without_freeze_time_fails_closed(p):
    freeze_candidate(p, "A", None, None, None)
    data = json.loads(p.frozen.read_text())
    del data["frozen_at_utc"]
    for f in (p.frozen, p.backup / p.frozen.name):
        f.chmod(0o644)
        f.write_text(json.dumps(data, indent=2))
    with pytest.raises(GateError, match="timestamp"):
        check_candidate(p)


# --- one-time final evaluation ---


def ready_for_final(p, candidate="A"):
    rules = None
    if candidate != "A":
        rules = p.root / "rules.py"
        rules.write_text("RULES = []\n")
    freeze_candidate(p, candidate, 0.5 if candidate == "D" else None, None, rules)
    for s in ("final_news", "final_tweets"):
        for who in ("chatgpt", "gemini"):
            write_record(p, who, s)
    labels(["N000", "N001", "N002"]).to_csv(p.gold / "final_news_labels.csv", index=False)
    labels(["T000", "T001"]).to_csv(p.gold / "final_tweets_labels.csv", index=False)


def fake_factory(seen=None):
    """A stand-in for the engine: candidate A says Other, any other candidate says M&A."""

    def factory(cfg):
        if seen is not None:
            seen.append(cfg)
        lab, src = ("Other", "model") if cfg["candidate"] == "A" else ("M&A", "product_rule")
        return lambda texts: ([lab] * len(texts), [src] * len(texts))

    return factory


def test_run_final_takes_no_predictor():
    import inspect

    assert list(inspect.signature(run_final).parameters) == ["p"]


def test_final_needs_a_frozen_candidate(p):
    with pytest.raises(GateError, match="frozen candidate"):
        _run_final(p, fake_factory())


def test_final_builds_predictors_from_the_frozen_config(p):
    ready_for_final(p)
    seen = []
    _run_final(p, fake_factory(seen))
    assert [c["candidate"] for c in seen] == ["A", "A", "A", "A"]  # A twice per set
    assert all("frozen_at_utc" in c for c in seen if c != {"candidate": "A"})


def test_final_runs_once_and_refuses_a_second_attempt(p):
    ready_for_final(p)
    res = _run_final(p, fake_factory())
    assert res["final_news"]["n"] == 3 and res["switch_engine"] is False
    with pytest.raises(GateError, match="already ran"):
        _run_final(p, fake_factory())
    p.marker.unlink()  # deleting the outputs marker is not enough: the backup copy remains
    with pytest.raises(GateError, match="already ran"):
        _run_final(p, fake_factory())


def test_final_fails_closed_on_missing_record(p):
    ready_for_final(p)
    (p.gold / "records" / "gemini_final_tweets.json").unlink()
    with pytest.raises(GateError, match="annotation record"):
        _run_final(p, fake_factory())


def test_changed_pinned_code_blocks_the_final(p):
    for f in PINNED:
        (p.root / f).parent.mkdir(parents=True, exist_ok=True)
        (p.root / f).write_text(f"pinned {f}\n")
    ready_for_final(p)
    (p.root / PINNED[1]).write_text("rules edited after the freeze\n")
    with pytest.raises(GateError, match="changed since the candidate was frozen"):
        _run_final(p, fake_factory())


def test_final_report_has_every_e24_field(p):
    ready_for_final(p, candidate="B")
    labels(["N000", "N001", "N002"], label="M&A", source="human_adjudicated").to_csv(
        p.gold / "final_news_labels.csv", index=False
    )
    r = _run_final(p, fake_factory())["final_news"]
    for k in ("n", "excluded", "ambiguous", "support", "absent_classes", "low_support_classes",
              "A", "chosen", "paired", "verdict", "secondary_human_touched"):  # fmt: skip
        assert k in r
    for k in ("macro_f1", "per_class", "confusion", "rule_coverage", "model_coverage",
              "abstention_rate"):  # fmt: skip
        assert k in r["chosen"]
    assert r["chosen"]["rule_coverage"] == {"product_rule": 1.0}
    assert r["A"]["model_coverage"] == 1.0
    assert r["low_support_classes"] == ["M&A"] and "Other" in r["absent_classes"]
    assert r["secondary_human_touched"]["n"] == 3


# --- metrics ---


def test_bootstrap_keeps_all_12_classes_when_classes_are_absent():
    y = ["Other"] * 8 + ["Earnings"] * 2
    base = ["Other"] * 10
    chosen = ["Other"] * 8 + ["Earnings"] * 2
    r = paired_bootstrap(y, base, chosen, n=200)
    # macro over 12 fixed classes: perfect on 2 classes → 2/12, base: Other F1 ≈ 0.889 → /12
    assert r["delta"] == pytest.approx(2 / 12 - (16 / 18) / 12, abs=1e-9)
    assert len(CLASSES) == 12


@pytest.mark.parametrize(
    "ci, bound, want",
    [([0.01, 0.10], 0.0, "pass"), ([-0.10, -0.01], 0.0, "fail"), ([-0.02, 0.05], 0.0,
     "inconclusive"), ([0.01, 0.30], 0.0, "inconclusive"), ([-0.04, 0.05], -0.05, "pass"),
     ([-0.12, -0.06], -0.05, "fail")],
)  # fmt: skip
def test_verdict_rules(ci, bound, want):
    assert verdict(ci, bound) == want


# --- the real prompts and guide (files in the repo) ---


def test_prompts_share_identical_instructions_and_leak_nothing():
    from riskengine.paths import DATA

    folder = DATA / "gold" / "e24" / "prompts"

    def block(name):
        text = (folder / name).read_text()
        start, end = text.index("=== E24 ANNOTATION"), text.index("=== END OF INSTRUCTIONS ===")
        return text, text[start:end]

    gpt, gpt_block = block("chatgpt_prompt.md")
    gem, gem_block = block("gemini_prompt.md")
    assert gpt_block == gem_block
    for label in CLASSES:
        assert label in gpt_block
    for text in (gpt, gem):  # no item ids, engine outputs or results in a prompt
        for banned in ("N000", "T000", "engine predicted", "macro-F1", "confidence"):
            assert banned not in text


def test_final_accepts_claude_as_labeller_b(p):
    ready_for_final(p)
    for s in ("final_news", "final_tweets"):
        (p.gold / "records" / f"gemini_{s}.json").rename(p.gold / "records" / f"claude_{s}.json")
        path = p.gold / "records" / f"claude_{s}.json"
        data = json.loads(path.read_text())
        path.write_text(json.dumps({**data, "labeller": "claude"}))
    res = _run_final(p, fake_factory())
    assert res["final_news"]["n"] == 3
