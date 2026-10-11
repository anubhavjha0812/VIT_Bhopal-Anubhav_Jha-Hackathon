from riskengine.eval.leaks import flags, norm


def test_norm_drops_urls_case_and_punctuation():
    assert norm("Apple UP 5%! https://t.co/x") == "apple up 5"


def test_flags_exact_near_and_clean():
    long = "Apple shares rise after strong iPhone sales in China lift quarterly revenue guidance"
    train = [long, "Oil falls as supply grows"]
    test = [
        long.lower() + "!",  # exact after normalising
        long + " again",  # near copy
        "Fed holds rates steady",  # unseen
    ]
    f = flags(test, train)
    assert f["exact"].tolist() == [True, False, False]
    assert f["seen"].tolist() == [True, True, False]
