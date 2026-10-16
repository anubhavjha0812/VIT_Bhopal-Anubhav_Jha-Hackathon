"""Real ONNX inference invariance, cache identity and freeze-method enforcement."""

import json

import numpy as np
import pytest

from riskengine.eval.e24_inference import (
    EvaluationProbs,
    inference_metadata,
    load_probability_cache,
    save_probability_cache,
)


@pytest.fixture
def tiny_model(tmp_path):
    """A local dynamic-int8 graph and tokenizer, with no downloaded assets."""
    import onnx
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, PreTrainedTokenizerFast

    vocab = {word: i for i, word in enumerate(["[UNK]", "[PAD]", "a", "long", "headline"])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    ).save_pretrained(tmp_path)
    BertConfig(id2label={0: "Other", 1: "Earnings"}).save_pretrained(tmp_path)
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["float_ids"], to=TensorProto.FLOAT),
        helper.make_node("DynamicQuantizeLinear", ["float_ids"], ["q", "scale", "zero"]),
        helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["restored"]),
        helper.make_node("ReduceMean", ["restored"], ["mean"], axes=[1], keepdims=1),
        helper.make_node("Neg", ["mean"], ["negative"]),
        helper.make_node("Concat", ["mean", "negative"], ["logits"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_int8",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "l"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, tmp_path / "model.onnx")
    return tmp_path


def test_same_rows_bit_identical_across_request_groups_and_orders(tiny_model):
    probs = EvaluationProbs(tiny_model)
    texts = ["a", "a long headline", "headline long", "a a a a a", "unknown"]
    names, expected = probs(texts)
    for batch in (1, 2, 8, 64):
        pieces = [probs(texts[i : i + batch])[1] for i in range(0, len(texts), batch)]
        assert np.array_equal(np.concatenate(pieces), expected)
    _, reversed_p = probs(texts[::-1])
    assert np.array_equal(reversed_p[::-1], expected)
    assert np.array_equal(probs([texts[0]])[1][0], expected[0])
    assert np.array_equal(EvaluationProbs(tiny_model)(texts)[1], expected)
    assert probs([])[0] == names and probs([])[1].shape == (0, len(names))
    options = probs.session.get_session_options()
    assert options.intra_op_num_threads == options.inter_op_num_threads == 1
    assert probs.inference_metadata["batch_size"] == 1


def test_cache_roundtrip_and_row_order_check(tiny_model, tmp_path):
    probs = EvaluationProbs(tiny_model)
    names, p = probs(["a", "long headline"])
    path = tmp_path / "e24_probs.npz"
    save_probability_cache(path, ["one", "two"], names, p, probs.inference_metadata)
    got_names, got, _ = load_probability_cache(path, ["one", "two"], probs.inference_metadata)
    assert got_names == names and np.array_equal(got, p)
    with pytest.raises(ValueError, match="ids/order"):
        load_probability_cache(path, ["two", "one"], probs.inference_metadata)
    with pytest.raises(FileExistsError):
        save_probability_cache(path, ["one", "two"], names, p, probs.inference_metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "legacy_batch64"),
        ("batch_size", 64),
        ("onnxruntime_version", "different"),
        ("intra_op_num_threads", 2),
        ("model_sha256", "wrong"),
        ("padding", "max_length"),
    ],
)
def test_cache_method_or_settings_mismatch_fails(tiny_model, tmp_path, field, value):
    metadata = inference_metadata(tiny_model)
    path = tmp_path / "e24_probs.npz"
    save_probability_cache(path, ["one"], ["Other"], np.ones((1, 1)), metadata)
    with pytest.raises(ValueError, match="method/settings mismatch"):
        load_probability_cache(path, ["one"], metadata | {field: value})


def test_legacy_cache_without_method_is_rejected_before_array_load(tmp_path):
    path = tmp_path / "e24_legacy.npz"
    path.with_suffix(".json").write_text(json.dumps({"batch_size": 64}))
    with pytest.raises(ValueError, match="method/settings mismatch"):
        load_probability_cache(path, [], {"method": "new"})


def test_cache_modified_bytes_fail(tiny_model, tmp_path):
    metadata = inference_metadata(tiny_model)
    path = tmp_path / "e24_probs.npz"
    save_probability_cache(path, ["one"], ["Other"], np.ones((1, 1)), metadata)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        load_probability_cache(path, ["one"], metadata)


def test_production_event_probabilities_and_labels_match_evaluation(tiny_model, monkeypatch):
    from riskengine.eval.e24_dev import Config, build_predictor, engine_probs
    from riskengine.nlp import events

    monkeypatch.setattr(events, "OWN_TOPIC", tiny_model)
    production = events.get_event_probs()
    evaluation = engine_probs()
    assert isinstance(production, EvaluationProbs)
    texts = ["a", "a long headline", "headline", "Hertz files for Chapter 11"]
    names, expected = evaluation(texts)
    got_names, actual = production(texts)
    assert got_names == names
    assert production.inference_metadata == evaluation.inference_metadata
    assert np.array_equal(actual, expected)
    assert np.array_equal(production(texts[::-1])[1][::-1], expected)
    for size in (1, 2, 8, 64):
        grouped = [production(texts[i : i + size])[1] for i in range(0, len(texts), size)]
        assert np.array_equal(np.concatenate(grouped), expected)
    classify = events.get_event_classifier("model")
    eval_labels, _ = build_predictor(Config("A"), evaluation)(texts)
    assert classify(texts) == eval_labels
    assert classify(texts)[-1] == "Default"  # credit precedence remains unchanged
    assert classify([]) == []


def test_generic_and_sentiment_classifiers_keep_batch64(tiny_model):
    from riskengine.nlp.classifier import TextClassifier
    from riskengine.nlp.sentiment import TransformerSentiment

    # Use valid sentiment class names in this isolated temporary model fixture.
    config_path = tiny_model / "config.json"
    config = json.loads(config_path.read_text())
    config["id2label"] = {"0": "negative", "1": "positive"}
    config["label2id"] = {"negative": 0, "positive": 1}
    config_path.write_text(json.dumps(config))
    generic = TextClassifier(str(tiny_model))
    sentiment = TransformerSentiment(str(tiny_model), name="fixture")
    for classifier in (generic, sentiment):
        assert classifier.batch_size == classifier.max_length == 64
        options = classifier.session.get_session_options()
        assert options.intra_op_num_threads == options.inter_op_num_threads == 0


def test_keyword_events_do_not_construct_a_model(monkeypatch):
    from riskengine.nlp import events

    def unexpected():
        raise AssertionError("keyword inference must not load an event model")

    monkeypatch.setattr(events, "get_event_probs", unexpected)
    assert events.get_event_classifier("keywords")(["Acme unveils new product"]) == ["Product news"]


def test_production_event_probs_missing_model_fails_clearly(tmp_path, monkeypatch):
    from riskengine.nlp import events

    monkeypatch.setattr(events, "OWN_TOPIC", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="own topic model not found"):
        events.get_event_probs()
