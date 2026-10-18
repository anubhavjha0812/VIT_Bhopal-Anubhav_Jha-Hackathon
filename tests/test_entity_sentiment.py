"""Production entity sentiment uses fake probabilities, not real headline/model scoring."""

import numpy as np
import pytest

from riskengine.eval import e27_entity_eval
from riskengine.nlp import sentiment


@pytest.fixture
def scorer(tmp_path, monkeypatch):
    (tmp_path / "model.onnx").touch()
    calls = []

    def probs(pairs):
        calls.append(pairs)
        # Deliberately use a different label order, including mixed case.
        return ["Positive", "negative", "neutral"], np.array([[0.8, 0.1, 0.1], [0.1, 0.6, 0.3]])

    monkeypatch.setattr(e27_entity_eval, "EntityEvaluationProbs", lambda path: probs)
    return sentiment.EntitySentiment(tmp_path), calls


def test_pair_input_and_probability_difference(scorer):
    model, calls = scorer
    assert model.score_entities(["A rises, B falls"] * 2, ["Company A", "Company B"]) == (
        pytest.approx([0.7, -0.5])
    )
    assert calls == [[("Company A", "A rises, B falls"), ("Company B", "A rises, B falls")]]
    assert model._fallback is None


def test_entity_fallback_loads_own_only_when_needed(scorer, monkeypatch):
    model, calls = scorer
    loads = []

    class Fallback:
        def score(self, texts):
            return [-0.4 for _ in texts]

    def get_scorer(kind):
        loads.append(kind)
        return Fallback()

    monkeypatch.setattr(sentiment, "get_scorer", get_scorer)
    assert model.score([]) == [] and loads == []
    assert model.score(["headline"]) == [-0.4]
    assert model.score(["another"]) == [-0.4]
    assert loads == ["own"] and calls == []


def test_empty_pairs_and_unequal_lengths(scorer):
    model, calls = scorer
    assert model.score_entities([], []) == [] and calls == []
    with pytest.raises(ValueError):
        model.score_entities(["headline"], [])


def test_entity_factory_and_existing_factories_keep_their_modes(tmp_path, monkeypatch):
    own = object()
    entity = object()
    calls = []

    def transformer(*args, **kwargs):
        calls.append((args, kwargs))
        return own

    monkeypatch.setattr(sentiment, "TransformerSentiment", transformer)
    monkeypatch.setattr(sentiment, "EntitySentiment", lambda: entity)
    monkeypatch.setattr(sentiment, "OWN_SENTIMENT", tmp_path)
    assert sentiment.get_scorer("entity") is entity
    assert sentiment.get_scorer("own") is own
    assert sentiment.get_scorer("finbert") is own
    assert calls == [((sentiment.OWN_SENTIMENT,), {"name": "own"}), ((), {})]


def test_missing_entity_model_has_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="entity sentiment model not found"):
        sentiment.EntitySentiment(tmp_path)


@pytest.mark.parametrize(
    "names,p",
    [
        (["LABEL_0", "LABEL_1", "LABEL_2"], [[1, 0, 0]]),
        (["negative", "positive", "neutral"], [[float("nan"), 0, 1]]),
        (["negative", "positive", "neutral"], [[0, 1]]),
    ],
)
def test_invalid_entity_probabilities_are_rejected(scorer, names, p):
    model, _ = scorer
    model.probs = lambda pairs: (names, p)
    with pytest.raises(ValueError):
        model.score_entities(["headline"], ["Company"])
