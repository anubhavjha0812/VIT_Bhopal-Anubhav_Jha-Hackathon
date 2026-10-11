from riskengine.eval.datasets import (
    EVENT9_TEST_CLASSES,
    EVENT_TEST_CLASSES,
    TOPIC_TO_EVENT,
    TOPIC_TO_EVENT_E2,
    tfns_test,
    topic_test,
)


def test_tfns_test_set_is_complete_and_mapped():
    df = tfns_test()
    assert len(df) == 2388
    assert set(df["label"]) == {"negative", "neutral", "positive"}


def test_topic_test_set_maps_only_to_scored_classes():
    df = topic_test()
    assert len(df) == 4117
    assert set(df["label"]) == set(EVENT_TEST_CLASSES)
    # mapping fixed in docs/DECISIONS.md E2
    assert df.loc[df["topic"] == "Politics", "label"].eq("Geopolitical").all()
    assert set(TOPIC_TO_EVENT_E2.values()) == {"Macroeconomic", "M&A", "Geopolitical"}


def test_topic_test_set_has_the_e16_labels():
    df = topic_test()
    assert set(df["label9"]) == set(EVENT9_TEST_CLASSES)
    assert set(TOPIC_TO_EVENT.values()) == set(EVENT9_TEST_CLASSES)
    # E16 widened Macroeconomic; E2 rows keep their own labels
    oil = df["topic"] == "Energy | Oil"
    assert df.loc[oil, "label9"].eq("Macroeconomic").all()
    assert df.loc[oil, "label"].eq("Other").all()
