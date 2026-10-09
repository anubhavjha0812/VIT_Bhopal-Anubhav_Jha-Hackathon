import pandas as pd

from riskengine.config import Config
from riskengine.ingest.linking import Linker
from riskengine.nlp.events import get_event_classifier
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
