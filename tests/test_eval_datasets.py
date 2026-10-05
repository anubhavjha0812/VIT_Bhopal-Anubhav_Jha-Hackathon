from riskengine.eval.datasets import EVENT_TEST_CLASSES, TOPIC_TO_EVENT, tfns_test, topic_test


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
    assert set(TOPIC_TO_EVENT.values()) == {"Macroeconomic", "M&A", "Geopolitical"}
