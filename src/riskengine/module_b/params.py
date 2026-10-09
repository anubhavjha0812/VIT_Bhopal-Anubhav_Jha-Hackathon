"""Rating scale and credit parameters for Module B (docs/DECISIONS.md, E14)."""

# S&P long-term scale, best to worst; the index is the notch (AAA = 0).
SCALE = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
]  # fmt: skip


def notch(rating: str) -> int:
    return SCALE.index(rating)
