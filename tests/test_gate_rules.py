import pandas as pd
import pytest

from riskengine.gate import gate
from riskengine.gate_rules import blocked_reason
from riskengine.ingest.linking import Linker


@pytest.fixture(scope="module")
def linker():
    return Linker.from_csv()


@pytest.mark.parametrize(
    "text",
    [
        "Fitch Downgrades Acme Commercial Mortgage Trust 2004-ABC7",
        "Moody's cuts three classes of Foo Bank Series 2011 pass-through notes",
        "S&P lowers ratings on Foo Bank CLO tranches",
        "Moody's cuts Foo Bank mortgage servicing ratings",
        "Fitch downgrades Foo Bank 2007-RR3; removes from watch",
    ],
)
def test_securitization_is_blocked(text):
    assert blocked_reason(text, "Credit downgrade") == "securitization"


@pytest.mark.parametrize(
    "text",
    [
        "Fitch affirms Foo Corp at 'BBB'; outlook stable",
        "S&P upgrades Foo Corp to BBB+",
        "Moody's: outlook on Foo Corp positive",
    ],
)
def test_affirmation_and_upgrade_are_blocked(text):
    assert blocked_reason(text, "Credit downgrade") == "affirmation or upgrade"


@pytest.mark.parametrize(
    "text",
    [
        "Moody's downgrades Foo Corp to Baa3",
        "Fitch cuts Foo Corp's IDR to 'BB'; outlook stable",
        "S&P lowers Foo Corp to junk, outlook negative",
        "Fitch affirms Foo Corp but downgrades its subordinated notes",
        "Fitch downgrades Foo Corp's senior unsecured notes tranche to BB",
        "Foo Corp files for Chapter 11 bankruptcy",
        "Foo Corp cut to 'B+' as 2020-21 earnings sink",
        "Fitch affirms Foo Corp's IDR at 'B-'; outlook revised to negative",
        "Moody's affirms Foo Corp at Baa2, lowers outlook to negative",
    ],
)
def test_real_downgrades_and_other_classes_stay(text):
    event = "Default" if "Chapter 11" in text else "Credit downgrade"
    assert blocked_reason(text, event) is None


def test_an_affirmation_with_a_stable_outlook_or_remaining_negative_is_still_blocked():
    assert blocked_reason(
        "Fitch affirms Foo Corp at 'A'; outlook remains negative", "Credit downgrade"
    )
    assert blocked_reason("Fitch affirms Foo Corp at 'A'; outlook stable", "Credit downgrade")


def test_an_affirmation_rule_does_not_apply_to_default():
    assert blocked_reason("Foo Corp upgrades its plant after bankruptcy filing", "Default") is None


def test_the_gate_flag_off_keeps_the_old_result(linker):
    now = pd.Timestamp("2020-05-01 12:00", tz="UTC")
    text = "Fitch downgrades JPMorgan Commercial Mortgage Trust 2005-CIBC12"
    signals = pd.DataFrame(
        [
            {
                "doc_id": "d0",
                "published_at": now,
                "ticker": "JPM",
                "text": text,
                "event": "Credit downgrade",
                "impact": 9,
                "source": "news",
            }
        ]
    )
    off = gate(signals, 7, linker, precision_rules=False)
    on = gate(signals, 7, linker, precision_rules=True)
    assert bool(off["fired"].iloc[0])
    assert not bool(on["fired"].iloc[0])
    assert on["reason"].iloc[0] == "blocked: securitization"


def test_the_gate_flag_keeps_a_real_downgrade(linker):
    now = pd.Timestamp("2020-05-01 12:00", tz="UTC")
    signals = pd.DataFrame(
        [
            {
                "doc_id": "d0",
                "published_at": now,
                "ticker": "DIS",
                "text": "Fitch downgrades Disney's IDR to 'A-' from 'A'",
                "event": "Credit downgrade",
                "impact": 9,
                "source": "news",
            }
        ]
    )
    assert bool(gate(signals, 7, linker, precision_rules=True)["fired"].iloc[0])
