"""E38c uses invented fixtures and prior results, never the v3 test folder."""

import json

import pytest

from riskengine.eval.linker_extended import metrics, subject_links
from riskengine.ingest.linking_extended import ExtendedLinker
from riskengine.paths import DATA


@pytest.fixture(scope="module")
def candidate():
    return ExtendedLinker.from_csv(candidate=3)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Apple seeks permission for an engineering campus", {"AAPL"}),
        ("Apple sees an opportunity in digital music", {"AAPL"}),
        ("Bank of America says it rejects allegations in a civil suit", {"BAC"}),
        ("BofA cuts the mortgage branch manager position", {"BAC"}),
        ("BofA said to keep a business unit during restructuring", {"BAC"}),
        ("Apple asks a judge for access to patent documents", {"AAPL"}),
        ("Apple buys a radio drama studio", {"AAPL"}),
        ("Goldman Sachs buys a technology business", {"GS"}),
        ("Apple, Amazon, Oracle and Ford discuss a partnership", {"AAPL", "AMZN", "ORCL", "F"}),
        ("Target, Oracle and Capital One at the opening ceremony", {"TGT", "ORCL", "COF"}),
        ("Amazon Web Services introduces a regional hub", {"AMZN"}),
        ("Amazon Web Services Hosts Fictional Corp", {"AMZN"}),
        ("Apple Watch receives a design award", {"AAPL"}),
    ],
)
def test_true_subject_groups_are_kept(candidate, text, expected):
    assert subject_links(candidate, text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Apple supplier Fictional Glass files for bankruptcy", set()),
        ("Apple supplier Intel files for bankruptcy", {"INTC"}),
        ("Amazon customer Fictional Widgets defaults on debt", set()),
        ("Microsoft partner Fictional Glass enters bankruptcy", set()),
        ("Apple Harbor Corporation enters bankruptcy", set()),
        ("Apple supplier Intel files bankruptcy; Apple seeks court relief", {"AAPL", "INTC"}),
        ("JPMorgan Says Nvidia faces a difficult year", {"NVDA"}),
        ("JPMorgan cuts the industry outlook after weak demand", set()),
        ("Bank of America initiates coverage on Fictional Robotics to Buy", set()),
        ("Microsoft Price Target raised by Goldman Sachs", {"MSFT"}),
        ("Target earnings lift shares; Microsoft Price Target raised", {"TGT", "MSFT"}),
        ("Apple harvest and Amazon rainforest threaten bank earnings", set()),
    ],
)
def test_correct_removal_groups_and_rating_forms(candidate, text, expected):
    assert subject_links(candidate, text) == expected


def test_cashtag_and_nonoriginal_guards_are_preserved(candidate):
    assert candidate.link("$aapl and $f", use_names=False) == ("AAPL", "F")
    assert candidate.link("Target awaits a surprise") == ()
    assert candidate.link("Visa applications increase bank costs") == ()
    assert subject_links(candidate, "$GS upgrades Fictional Widgets to Buy") == {"GS"}


def test_candidate_two_reproduces_all_saved_rows_and_metrics():
    stored = json.loads((DATA / "outputs/linker_extended_v2.json").read_text())
    two = ExtendedLinker.from_csv(candidate=2)
    gold, predictions = [], []
    for row in stored["set_a_rows"]:
        assert list(two.link(row["text"])) == row["candidate_2_mentions"]
        prediction = subject_links(two, row["text"])
        assert sorted(prediction) == row["candidate_2_subjects"]
        gold.append(set(row["gold"]))
        predictions.append(prediction)
    assert metrics(gold, predictions) == stored["set_a"]["candidate_2"]


def test_missing_final_reference_stops_before_linker_or_result(tmp_path, monkeypatch):
    from riskengine.eval import linker_extended_v3 as evaluation

    monkeypatch.setattr(evaluation, "verify_design", lambda design: {})
    monkeypatch.setattr(
        evaluation.ExtendedLinker, "from_csv", lambda **kwargs: pytest.fail("no linker")
    )
    result = tmp_path / "result.json"
    with pytest.raises(FileNotFoundError, match="await labels_final"):
        evaluation.score_once(result=result, work=tmp_path, design=tmp_path)
    assert not result.exists() and not result.with_suffix(".run_lock").exists()


def test_repeat_run_refused_without_opening_any_reference(tmp_path, monkeypatch):
    from riskengine.eval import linker_extended_v3 as evaluation

    result = tmp_path / "result.json"
    result.write_text("{}")
    monkeypatch.setattr(evaluation, "verify_design", lambda design: pytest.fail("must not load"))
    with pytest.raises(FileExistsError, match="second scored run"):
        evaluation.score_once(result=result, work=tmp_path, design=tmp_path)


def test_frozen_label_hash_is_checked_before_sample_read(tmp_path):
    from riskengine.eval.linker_extended_v3 import read_frozen_reference

    (tmp_path / "labels_final.csv").write_text("deliberately changed")
    (tmp_path / "labels_final_record.json").write_text(
        json.dumps({"labels_final_sha256": "0" * 64})
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_frozen_reference(tmp_path)


def test_candidate_seal_checked_before_reference_read(tmp_path):
    from riskengine.eval.linker_extended_v3 import rules_text, verify_design

    candidate = tmp_path / "candidate.txt"
    candidate.write_text("changed")
    (tmp_path / "candidate_manifest.json").write_text(
        json.dumps(
            {
                "rules_text": rules_text(),
                "files_sha256": {str(candidate): "wrong digest"},
            }
        )
    )
    with pytest.raises(ValueError, match="sealed candidate/design file changed"):
        verify_design(tmp_path)
