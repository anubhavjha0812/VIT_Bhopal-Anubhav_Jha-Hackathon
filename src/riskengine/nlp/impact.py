"""Impact score 1-10. Baseline: fixed formula from event severity and sentiment strength."""

CLASS_SEVERITY = {
    "Default": 9,
    "Credit downgrade": 8,
    "Covenant breach": 7,
    "Geopolitical": 6,
    "Macroeconomic": 6,
    "M&A": 6,
    "Product launch": 4,
    "Other": 3,
}
NEWS_BONUS = 1  # an edited news headline counts for more than a single social post


def impact_formula(event: str, sentiment: float, source: str) -> int:
    """severity x (0.6 + 0.4 |sentiment|), +1 for news, clipped to 1-10."""
    raw = CLASS_SEVERITY[event] * (0.6 + 0.4 * abs(sentiment))
    raw += NEWS_BONUS if source == "news" else 0
    return int(min(10, max(1, round(raw))))
