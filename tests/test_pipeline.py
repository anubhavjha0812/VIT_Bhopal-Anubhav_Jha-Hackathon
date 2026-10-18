from dataclasses import replace

import pandas as pd
import pytest

from riskengine.config import Config
from riskengine.ingest.linking import Linker
from riskengine.nlp.events import get_event_classifier
from riskengine.nlp.impact import impact_formula
from riskengine.pipeline import score_docs, subjects


class FakeScorer:
    name = "fake"

    def score(self, texts):
        return [-0.5 for _ in texts]


def test_score_docs_tags_a_bank_that_only_gives_a_rating_as_analyst():
    docs = pd.DataFrame(
        {
            "doc_id": ["a", "b"],
            "published_at": pd.to_datetime(["2020-05-01 12:00+00:00"] * 2),
            "source": ["news", "news"],
            "text": [
                "BofA Downgrades Tesla, Says Company 'Faces Several Hurdles'",
                "Bank of America posts record loan losses",
            ],
            "tickers": [["BAC", "TSLA"], ["BAC"]],
        }
    )
    out = score_docs(
        docs, Config(), FakeScorer(), get_event_classifier("keywords"), Linker.from_csv()
    )
    roles = out.set_index(["doc_id", "ticker"])["role"].to_dict()
    assert roles == {("a", "BAC"): "analyst", ("a", "TSLA"): "subject", ("b", "BAC"): "subject"}
    assert subjects(out)["ticker"].tolist() == ["TSLA", "BAC"]


def test_subjects_keeps_everything_in_files_without_roles():
    old = pd.DataFrame({"ticker": ["JPM", "BAC"]})
    assert subjects(old).equals(old)


class FakeEntityScorer:
    name = "entity"

    def __init__(self):
        self.entity_calls, self.headline_calls = [], []

    def score(self, texts):
        self.headline_calls.append(list(texts))
        return [-0.345678 for _ in texts]

    def score_entities(self, texts, entities):
        self.entity_calls.append(list(zip(entities, texts, strict=True)))
        return [
            {"Apple Inc.": 0.987654, "Microsoft Corp.": -0.123456}.get(name, -0.5)
            for name in entities
        ]


@pytest.fixture
def two_company_docs():
    return pd.DataFrame(
        {
            "doc_id": ["pair"],
            "published_at": pd.to_datetime(["2020-05-01T12:00:00Z"]),
            "source": ["news"],
            "text": ["Apple gains as Microsoft falls"],
            "tickers": [["AAPL", "MSFT"]],
        }
    )


def test_entity_sentiment_and_impact_are_per_company_after_explode(two_company_docs):
    scorer = FakeEntityScorer()
    original = two_company_docs.copy(deep=True)
    out = score_docs(
        two_company_docs,
        Config(sentiment="entity"),
        scorer,
        lambda texts: ["Earnings"] * len(texts),
        Linker.from_csv(),
    )
    assert out["ticker"].tolist() == ["AAPL", "MSFT"]
    assert out["sentiment"].tolist() == [0.9877, -0.1235]
    assert out["impact"].tolist() == [6, 4]
    assert out["model"].tolist() == ["entity+keywords+formula"] * 2
    assert scorer.entity_calls == [
        [("Apple Inc.", original["text"][0]), ("Microsoft Corp.", original["text"][0])]
    ]
    assert scorer.headline_calls == []  # known names do not load the fallback model
    pd.testing.assert_frame_equal(two_company_docs, original)


@pytest.mark.parametrize("kind", ["own", "finbert"])
def test_headline_modes_keep_doc_sentiment_and_doc_impact(two_company_docs, kind, monkeypatch):
    from riskengine import pipeline

    scorer, impacts = FakeEntityScorer(), []

    def impact(*args):
        impacts.append(args)
        return impact_formula(*args)

    monkeypatch.setattr(pipeline, "impact_formula", impact)
    out = score_docs(
        two_company_docs,
        Config(sentiment=kind),
        scorer,
        lambda texts: ["Earnings"] * len(texts),
        Linker.from_csv(),
    )
    assert out["sentiment"].tolist() == [-0.3457, -0.3457]
    assert out["impact"].nunique() == 1 and len(impacts) == 1
    assert scorer.headline_calls == [two_company_docs["text"].tolist()]
    assert scorer.entity_calls == []


@pytest.mark.parametrize("missing_name", [None, "", "   "])
def test_missing_name_uses_headline_fallback_only_for_that_row(two_company_docs, missing_name):
    scorer, linker = FakeEntityScorer(), Linker.from_csv()
    names = dict(linker.company_names)
    if missing_name is None:
        names.pop("MSFT")
    else:
        names["MSFT"] = missing_name
    linker = replace(linker, company_names=names)
    out = score_docs(
        two_company_docs,
        Config(sentiment="entity"),
        scorer,
        lambda texts: ["Earnings"] * len(texts),
        linker,
    )
    assert out["sentiment"].tolist() == [0.9877, -0.3457]
    assert out["impact"].tolist() == [6, 5]
    assert scorer.entity_calls == [[("Apple Inc.", two_company_docs["text"][0])]]
    assert scorer.headline_calls == [two_company_docs["text"].tolist()]


def test_multiple_missing_names_share_one_headline_fallback(two_company_docs):
    scorer = FakeEntityScorer()
    linker = replace(Linker.from_csv(), company_names={})
    out = score_docs(
        two_company_docs,
        Config(sentiment="entity"),
        scorer,
        lambda texts: ["Earnings"] * len(texts),
        linker,
    )
    assert out["sentiment"].tolist() == [-0.3457, -0.3457]
    assert scorer.headline_calls == [two_company_docs["text"].tolist()]
    assert scorer.entity_calls == []


@pytest.mark.parametrize(
    "text,event,agency,analyst_rating",
    [
        ("BofA downgrades Apple and Microsoft after Fitch action", "Credit downgrade", True, False),
        ("BofA downgrades Apple and Microsoft to Underperform", "Analyst rating", False, True),
    ],
)
def test_entity_impact_keeps_agency_bonus_and_analyst_roles(
    two_company_docs, text, event, agency, analyst_rating
):
    scorer = FakeEntityScorer()
    docs = two_company_docs.assign(text=text)
    docs["tickers"] = [["AAPL", "MSFT", "BAC"]]
    out = score_docs(
        docs,
        Config(sentiment="entity"),
        scorer,
        lambda texts: [event] * len(texts),
        Linker.from_csv(),
    )
    assert out["impact"].tolist() == [
        impact_formula(event, score, "news", agency) for score in out["sentiment"]
    ]
    assert out["analyst_rating"].tolist() == [analyst_rating] * len(out)
    assert out.loc[out["ticker"] == "BAC", "role"].tolist() == ["analyst"]


def test_entity_output_order_survives_duplicate_input_indices(two_company_docs):
    scorer = FakeEntityScorer()
    docs = pd.concat([two_company_docs, two_company_docs.assign(doc_id="second")])
    assert docs.index.tolist() == [0, 0]
    out = score_docs(
        docs,
        Config(sentiment="entity"),
        scorer,
        lambda texts: ["Earnings"] * len(texts),
        Linker.from_csv(),
    )
    assert out["doc_id"].tolist() == ["pair", "pair", "second", "second"]
    assert out["sentiment"].tolist() == [0.9877, -0.1235, 0.9877, -0.1235]


def test_empty_entity_input_does_not_score(two_company_docs):
    scorer = FakeEntityScorer()
    out = score_docs(
        two_company_docs.iloc[:0],
        Config(sentiment="entity"),
        scorer,
        lambda texts: [],
        Linker.from_csv(),
    )
    assert out.empty and scorer.entity_calls == scorer.headline_calls == []
