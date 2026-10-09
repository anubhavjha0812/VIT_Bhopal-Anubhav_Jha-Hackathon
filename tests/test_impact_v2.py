import numpy as np
import pandas as pd

from riskengine.impact.features import novelty
from riskengine.impact.train_v2 import expected_move


def rows(items):
    return pd.DataFrame(
        {
            "ticker": [t for t, _, _ in items],
            "published_at": pd.to_datetime([ts for _, ts, _ in items], utc=True),
            "text": [x for _, _, x in items],
        }
    )


def test_novelty_repeat_story_is_not_new():
    df = rows(
        [
            ("AAPL", "2019-01-02 10:00", "Apple cuts revenue guidance on China"),
            ("AAPL", "2019-01-02 12:00", "Apple cuts revenue guidance on China"),
        ]
    )
    assert novelty(df).tolist() == [1.0, 0.0]


def test_novelty_ignores_other_stocks_and_old_news():
    df = rows(
        [
            ("AAPL", "2019-01-02 10:00", "Apple cuts revenue guidance on China"),
            ("MSFT", "2019-01-02 11:00", "Apple cuts revenue guidance on China"),
            ("AAPL", "2019-01-06 11:00", "Apple cuts revenue guidance on China"),  # > 72 h later
        ]
    )
    assert novelty(df).tolist() == [1.0, 1.0, 1.0]


def test_novelty_partial_overlap_is_between_0_and_1():
    df = rows(
        [
            ("AAPL", "2019-01-02 10:00", "Apple cuts revenue guidance on China"),
            ("AAPL", "2019-01-02 11:00", "Apple shares fall after guidance cut"),
        ]
    )
    assert 0 < novelty(df)[1] < 1


def test_novelty_does_not_look_ahead():
    df = rows(
        [
            ("AAPL", "2019-01-02 12:00", "Apple cuts revenue guidance on China"),
            ("AAPL", "2019-01-02 10:00", "Apple cuts revenue guidance on China"),  # earlier row
        ]
    )
    assert novelty(df).tolist() == [0.0, 1.0]


def test_expected_move_scales_by_volatility():
    df = pd.DataFrame({"vol20": [0.01, 0.02]})
    move = expected_move(np.log1p(np.array([1.0, 1.0])), df)
    assert np.allclose(move, [np.sqrt(2) * 0.01, np.sqrt(2) * 0.02])
    assert expected_move(np.array([-5.0]), df.iloc[:1]).tolist() == [0.0]  # never negative


def test_headlines_last_24h_counts_only_the_previous_day():
    from riskengine.impact.features import headlines_last_24h

    df = rows(
        [
            ("AAPL", "2019-01-01 09:00", "a"),  # 25 h before the last row
            ("AAPL", "2019-01-02 09:00", "b"),
            ("AAPL", "2019-01-02 10:00", "c"),
            ("MSFT", "2019-01-02 09:30", "d"),
        ]
    )
    assert headlines_last_24h(df).tolist() == [0, 1, 1, 0]
