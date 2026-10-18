"""E27 checks use invented rows and temporary model fixtures, never the real test split."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e27_entity_eval as e27
from riskengine.eval.e20 import paired
from riskengine.eval.e24_inference import EvaluationProbs


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "sno": ["1", "1", "2", "3"],
            "headline": [
                "Alpha gains as Beta falls",
                "Alpha gains as Beta falls",
                "Gamma holds steady",
                "Delta rises",
            ],
            "entity": ["Alpha", "Beta", "Gamma", "Delta"],
            "label": ["positive", "negative", "neutral", "positive"],
            "conflict": [True, True, False, False],
            "n_entities": [2, 2, 1, 1],
        }
    )


def fake_probs(labels, calls):
    def predict(inputs):
        calls.append(inputs)
        # The engine export's order need not match LABELS or the entity model's order.
        names = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        matrix = np.eye(3)[[[name.lower() for name in names].index(label) for label in labels]]
        return names, matrix

    return predict


def quick_paired(y, a, b, n, seed):
    assert n == 1000 and seed == 0
    return {
        "b_minus_a": e27.macro_f1(y, b) - e27.macro_f1(y, a),
        "ci95": [-0.1, 0.8],
        "verdict": "tie",
    }


def test_pair_order_config_mapping_metrics_and_registered_bootstrap(frame, monkeypatch):
    a_calls, b_calls, bootstrap_calls = [], [], []

    def bootstrap(y, a, b, n, seed):
        bootstrap_calls.append((y.copy(), a.copy(), b.copy(), n, seed))
        return paired(y, a, b, n=n, seed=seed)

    monkeypatch.setattr(e27, "paired", bootstrap)
    result = e27.score_frame(
        frame,
        fake_probs(["positive", "neutral", "positive"], a_calls),
        fake_probs(frame["label"].tolist(), b_calls),
    )
    assert a_calls == [["Alpha gains as Beta falls", "Gamma holds steady", "Delta rises"]]
    assert b_calls == [list(zip(frame["entity"], frame["headline"], strict=True))]
    primary, all_rows = result["metrics"]["conflicting"], result["metrics"]["all"]
    assert primary["rows"] == 2 and all_rows["rows"] == 4
    assert primary["A"]["accuracy"] == 0.5 and primary["B"]["accuracy"] == 1.0
    assert primary["B"]["macro_f1"] == pytest.approx(2 / 3)  # fixed three classes
    assert all_rows["A"]["accuracy"] == 0.75 and all_rows["B"]["macro_f1"] == 1.0
    assert all(n == 1000 and seed == 0 for *_, n, seed in bootstrap_calls)
    assert len(bootstrap_calls) == 2
    assert primary["paired_b_minus_a"]["ci95"][0] <= primary["paired_b_minus_a"]["ci95"][1]
    assert result["paired_bootstrap"]["sampling_unit"] == "entity row"
    assert result["rule"]["switch_to_b"] and not result["rule"]["applied_to_engine"]


@pytest.mark.parametrize(
    "primary_a,primary_b,all_a,all_b,switch",
    [
        (0.40, 0.45, 0.80, 0.78, True),  # exact registered boundaries
        (0.40, 0.449, 0.80, 0.90, False),
        (0.40, 0.60, 0.80, 0.779, False),
        (0.40, 0.60, 0.80, 0.80, True),
        (None, None, 0.80, 0.90, False),
    ],
)
def test_fixed_rule(primary_a, primary_b, all_a, all_b, switch):
    result = e27.decide(primary_a, primary_b, all_a, all_b)
    assert result["switch_to_b"] is switch
    assert result["recommended_engine"] == ("B" if switch else "A")


def test_interval_crossing_zero_does_not_add_a_new_switch_condition(frame, monkeypatch):
    monkeypatch.setattr(e27, "paired", quick_paired)
    result = e27.score_frame(
        frame,
        fake_probs(["positive", "neutral", "positive"], []),
        fake_probs(frame["label"].tolist(), []),
    )
    assert result["metrics"]["conflicting"]["paired_b_minus_a"]["ci95"][0] < 0
    assert result["rule"]["switch_to_b"]


def test_no_conflicting_rows_cannot_recommend_switch(frame, monkeypatch):
    monkeypatch.setattr(e27, "paired", quick_paired)
    frame = frame.iloc[2:].reset_index(drop=True)
    result = e27.score_frame(
        frame,
        fake_probs(frame["label"].tolist(), []),
        fake_probs(frame["label"].tolist(), []),
    )
    assert result["metrics"]["conflicting"]["A"] is None
    assert not result["rule"]["switch_to_b"]


def training_run(seed, score):
    history = [{"epoch": 1, "dev_macro_f1": score - 0.1}, {"epoch": 2, "dev_macro_f1": score}]
    return {"seed": seed, "best": history[1].copy(), "history": history}


def test_seed_choice_uses_dev_only_and_preserves_first_on_tie():
    runs = [
        training_run(seed, score) for seed, score in zip(e27.SEEDS, [0.7, 0.8, 0.8], strict=True)
    ]
    for run, test_score in zip(runs, [1.0, 0.0, 0.99], strict=True):
        run["test_macro_f1"] = test_score
        run["best"]["test_macro_f1"] = test_score
    assert e27.select_seed(runs)["seed"] == 43
    runs[0]["best"]["dev_macro_f1"] = 0.9
    with pytest.raises(ValueError, match="dev macro-F1 winner"):
        e27.select_seed(runs)


def test_selection_records_missing_all_seed_summary_and_checks_optional_summary(tmp_path):
    runs = [
        training_run(seed, score) for seed, score in zip(e27.SEEDS, [0.7, 0.8, 0.9], strict=True)
    ]
    source = tmp_path / "source_metrics.json"
    source.write_text(json.dumps(runs[2]))
    result = e27.load_selection(tmp_path)
    assert result["seed"] == 44 and not result["all_seed_comparison_verified"]
    assert result["source_metrics_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"runs": runs, "chosen": runs[2]}))
    assert e27.load_selection(tmp_path, summary)["all_seed_comparison_verified"]
    summary.write_text(json.dumps({"runs": runs, "chosen": runs[0]}))
    with pytest.raises(ValueError, match="dev-only seed winner"):
        e27.load_selection(tmp_path, summary)


@pytest.mark.parametrize("problem", ["test_only", "later_epoch_tie", "seed_order", "nonfinite"])
def test_invalid_dev_selection_records_are_rejected(problem):
    runs = [training_run(seed, 0.8) for seed in e27.SEEDS]
    if problem == "test_only":
        del runs[0]["history"][0]["dev_macro_f1"]
        runs[0]["history"][0]["test_macro_f1"] = 0.99
    elif problem == "later_epoch_tie":
        runs[0]["history"].append({"epoch": 3, "dev_macro_f1": 0.8})
        runs[0]["best"] = runs[0]["history"][-1].copy()
    elif problem == "seed_order":
        runs.reverse()
    else:
        runs[0]["history"][0]["dev_macro_f1"] = float("nan")
    with pytest.raises(ValueError):
        e27.select_seed(runs)


def test_csv_read_once_with_hash_from_the_same_bytes(frame, tmp_path, monkeypatch):
    path = tmp_path / "test.csv"
    frame.to_csv(path, index=False)
    raw = path.read_bytes()
    original, reads = type(path).read_bytes, []

    def read(self):
        reads.append(self)
        return original(self)

    monkeypatch.setattr(type(path), "read_bytes", read)
    loaded, digest = e27.read_split(path)
    assert reads == [path]
    assert digest == hashlib.sha256(raw).hexdigest()
    assert loaded["conflict"].tolist() == [True, True, False, False]
    assert loaded["entity"].tolist() == frame["entity"].tolist()


@pytest.mark.parametrize(
    "column,value",
    [
        ("label", "bullish"),
        ("conflict", "maybe"),
        ("n_entities", "1.5"),
        ("n_entities", "1"),
        ("headline", " "),
        ("entity", None),
    ],
)
def test_bad_split_values_fail_without_inference(frame, tmp_path, column, value):
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    path = tmp_path / "test.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        e27.read_split(path)


def test_dry_run_only_prints_counts_without_loading_models_or_writing(
    frame, tmp_path, monkeypatch, capsys
):
    for split in ("train", "dev", "test"):
        frame.to_csv(tmp_path / f"{split}.csv", index=False)

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not load models, selection metadata or score")

    for name in ("EvaluationProbs", "EntityEvaluationProbs", "load_selection", "score"):
        monkeypatch.setattr(e27, name, forbidden)
    out = tmp_path / "uncreated/output.json"
    e27.main(["--dry-run", "--data-dir", str(tmp_path), "--out", str(out)])
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"train", "dev", "test"}
    assert all(row["rows"] == 4 and row["conflicting_rows"] == 2 for row in result.values())
    assert not out.parent.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_score_reads_only_test_once_and_preserves_completed_or_failed_attempt(
    frame, tmp_path, monkeypatch, fail
):
    frame.to_csv(tmp_path / "test.csv", index=False)  # no train/dev files exist
    out = tmp_path / "outputs/report.json"
    model = tmp_path / "model"
    monkeypatch.setattr(e27, "load_selection", lambda *args: {"seed": 44})
    monkeypatch.setattr(e27, "paired", quick_paired)
    a = fake_probs(["positive", "neutral", "positive"], [])
    b = fake_probs(frame["label"].tolist(), [])
    a.inference_metadata = b.inference_metadata = {"batch_size": 1}
    monkeypatch.setattr(e27, "EvaluationProbs", lambda *args: a)
    monkeypatch.setattr(e27, "EntityEvaluationProbs", lambda *args: b)
    read, reads = e27.read_split, []

    def read_once(path):
        assert json.loads(out.read_text())["status"] == "started"
        reads.append(path)
        if fail:
            raise ValueError("fixture read failure")
        return read(path)

    monkeypatch.setattr(e27, "read_split", read_once)
    if fail:
        with pytest.raises(ValueError, match="fixture read failure"):
            e27.score(tmp_path, model, out)
    else:
        result = e27.score(tmp_path, model, out)
        assert result["selection"]["seed"] == 44 and result["rule"]["switch_to_b"]
    saved = out.read_bytes()
    assert json.loads(saved)["status"] == ("failed" if fail else "complete")
    assert reads == [tmp_path / "test.csv"]
    with pytest.raises(FileExistsError, match="already attempted"):
        e27.score(tmp_path, model, out)
    assert out.read_bytes() == saved


def test_existing_report_blocks_even_loading_selection(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    out.write_text("preserve\n")

    def forbidden(*args):
        pytest.fail("existing result must block before model metadata or test reads")

    monkeypatch.setattr(e27, "load_selection", forbidden)
    with pytest.raises(FileExistsError):
        e27.score(tmp_path, tmp_path, out)
    assert out.read_text() == "preserve\n"


@pytest.mark.parametrize(
    "names,matrix",
    [
        (["LABEL_0", "LABEL_1", "LABEL_2"], [[0, 0, 1]]),
        (["negative", "neutral", "positive"], [[float("nan"), 0, 1]]),
        (["negative", "neutral", "positive"], [[0, 1]]),
    ],
)
def test_invalid_model_output_is_rejected(names, matrix):
    with pytest.raises(ValueError):
        e27._predict(lambda texts: (names, matrix), ["fixture"])


@pytest.fixture
def tiny_pair_model(tmp_path):
    """A local dynamic-int8 graph and explicit pair tokenizer; no training or downloads."""
    import onnx
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, PreTrainedTokenizerFast

    words = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "Alpha", "Beta", "gains", "falls"]
    tokenizer = Tokenizer(WordLevel(dict(zip(words, range(len(words)), strict=True)), "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_input_names=["input_ids", "attention_mask", "token_type_ids"],
    ).save_pretrained(tmp_path)
    BertConfig(id2label=dict(enumerate(e27.LABELS))).save_pretrained(tmp_path)
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["floats"], to=TensorProto.FLOAT),
        helper.make_node("DynamicQuantizeLinear", ["floats"], ["q", "scale", "zero"]),
        helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["restored"]),
        helper.make_node("ReduceMean", ["restored"], ["mean"], axes=[1], keepdims=1),
        helper.make_node("Neg", ["mean"], ["negative"]),
        helper.make_node("Concat", ["negative", "mean", "negative"], ["logits"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_pair_int8",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "l"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, tmp_path / "model.onnx")
    return tmp_path


def test_entity_pair_encoding_reuses_per_row_inference_and_is_batch_independent(tiny_pair_model):
    model = e27.EntityEvaluationProbs(tiny_pair_model)
    pairs = [("Alpha", "Alpha gains Beta falls"), ("Beta", "Beta falls"), ("Alpha", "gains")]
    names, expected = model(pairs)
    assert names == e27.LABELS
    for size in (1, 2, 64):
        got = np.concatenate([model(pairs[i : i + size])[1] for i in range(0, len(pairs), size)])
        assert np.array_equal(got, expected)
    assert np.array_equal(model(pairs[::-1])[1][::-1], expected)
    assert model([])[1].shape == (0, 3)
    assert model.inference_metadata["input_order"] == ["entity", "headline"]
    assert model.inference_metadata["batch_size"] == 1
    tokenizer = model.probs.tokenizer.tokenizer
    actual = model.probs.tokenizer(
        [pairs[0]], padding=True, truncation=True, max_length=64, return_tensors="np"
    )
    direct = tokenizer(
        [pairs[0][0]],
        [pairs[0][1]],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="np",
    )
    assert all(np.array_equal(actual[key], direct[key]) for key in direct)
    assert 1 in actual["token_type_ids"][0]  # genuine pair, not concatenated text
    assert len(actual["input_ids"][0]) <= 64
    assert not np.array_equal(EvaluationProbs(tiny_pair_model)([pairs[0][1]])[1], expected[:1])


def test_pair_adapter_passes_entity_first_without_changing_inference_options():
    calls = []

    def tokenizer(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace()

    kwargs = {"padding": True, "truncation": True, "max_length": 64, "return_tensors": "np"}
    e27._PairTokenizer(tokenizer)([("Alpha", "Alpha gains")], **kwargs)
    assert calls == [((["Alpha"], ["Alpha gains"]), kwargs)]


def test_optional_summary_must_match_exported_winner(tmp_path):
    runs = [
        training_run(seed, score) for seed, score in zip(e27.SEEDS, [0.9, 0.8, 0.7], strict=True)
    ]
    summary = {"runs": runs, "chosen": copy.deepcopy(runs[0])}
    (tmp_path / "source_metrics.json").write_text(json.dumps(runs[2]))
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="dev-only seed winner"):
        e27.load_selection(tmp_path, path)
