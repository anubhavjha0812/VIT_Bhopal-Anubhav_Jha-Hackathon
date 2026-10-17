"""Small fake titles and HTTP responses only; never open evaluation rows or models."""

import json

import httpx
import pandas as pd
import pytest

from riskengine.eval import e25_official_titles as official


def frame(texts, sources=None):
    return pd.DataFrame(
        {
            "text": texts,
            "source": sources or ["ecb"] * len(texts),
            "date": ["2014-04-03"] * len(texts),
            "url": [f"https://example.test/title/{i}" for i in range(len(texts))],
        }
    )


def test_eligibility_uses_original_dates_length_english_and_fixed_cues():
    f = frame(
        [
            "ECB announces monetary policy operations for banks",
            "Foreign Secretary welcomes ceasefire in Syria",
            "Foreign Secretary statement on Japan",
            "Foreign Secretary visits Canada",  # country/minister alone is insufficient
            "ECB appoints a new board member",  # official source alone is insufficient
            "Monetary policy",  # too short
            "Décisions de politique monétaire sur les taux",
            "Новые sanctions against the military forces",
            "ECB publishes a financial stability review",
            "ECB announces a new monetary policy decision",
            "ECB announces oil prices for the coming year",
        ],
        ["ecb", "govuk_fco", "govuk_fco", "govuk_fco"] + ["ecb"] * 7,
    )
    f.loc[8, "date"] = "2009-12-31"
    f.loc[9, "date"] = "2017-01-01"
    f.loc[10, "date"] = None
    out, _ = official.eligible(f)
    assert set(out["text"]) == set(f.loc[:2, "text"])
    assert out["cue_class"].tolist() == ["Macro", "Geo", "Geo"]


def test_window_includes_2010_start_and_all_of_2016():
    f = frame(["ECB monetary policy decision on banks", "ECB financial stability review for banks"])
    f["date"] = ["2010-01-01T00:00:00Z", "2016-12-31T23:59:59Z"]
    out, _ = official.eligible(f)
    assert out["date"].tolist() == ["2010-01-01", "2016-12-31"]


def test_excludes_normalized_exact_and_near_matches_and_rechecks_final_sample():
    held = "ECB publishes financial stability review assessing risks to banks and markets this year"
    near = held + " today"
    f = frame(
        [
            held.upper() + "!",
            near,
            "ECB reports balance of payments for the euro area",
        ]
    )
    corpus = {"fake reserved set": [held]}
    out, audit = official.clean_sample(f, corpus)
    assert out["text"].tolist() == [f.loc[2, "text"]]
    assert audit["exclusion_passes"][0]["excluded_total"] >= 1
    official.assert_clean(out, corpus)
    assert list(out) == official.COLUMNS
    assert not {"label", "event", "label_source", "prediction"} & set(out)


def test_dedup_drops_exact_near_and_same_title_on_different_dates():
    text = "ECB publishes financial stability review assessing risks to banks and markets this year"
    f = frame(
        [text, text.upper() + "!", text + " today", "ECB announces monetary policy operations"]
    )
    f["date"] = ["2010-01-01", "2012-01-01", "2013-01-01", "2014-01-01"]
    out, _ = official.clean_sample(f, {})
    assert out["text"].tolist() == [text, f.loc[3, "text"]]


def test_caps_and_seed_are_deterministic_under_input_reordering():
    f = frame([f"unique title number {i}" for i in range(30)])
    f["cue_class"] = ["Macro"] * 20 + ["Geo"] * 10
    a = official.draw(f, cap=4, seed=25)
    b = official.draw(f.sample(frac=1, random_state=99), cap=4, seed=25)
    pd.testing.assert_frame_equal(a, b)
    assert a["cue_class"].value_counts().to_dict() == {"Macro": 4, "Geo": 4}
    assert not a.equals(official.draw(f, cap=4, seed=26))


def test_empty_and_all_rejected_frames_keep_the_output_schema():
    for f in (frame([]), frame(["short"]), frame(["ECB monetary policy decision for banks"])):
        corpus = {"fake exclusion": f["text"].tolist()}
        out, _ = official.clean_sample(f, corpus)
        assert out.empty
        assert out.columns.tolist() == official.COLUMNS


def test_ecb_parser_reads_archive_links_and_url_dates_without_article_fetch():
    html = b"""<a href="/press/pr/date/2014/html/pr140403.en.html">
      ECB monetary <span>policy</span> decisions &amp; operations</a>
      <a href="/press/pr/date/2014/html/pr140403.fr.html">French</a>
      <a href="/pub/pdf/scpwps/paper.en.pdf">Named author paper</a>"""
    rows = official.ecb_rows(html, "https://www.ecb.europa.eu/archive/")
    assert len(rows) == 1
    assert rows[0]["date"] == "2014-04-03"
    assert rows[0]["text"] == "ECB monetary policy decisions & operations"


def test_govuk_parser_uses_update_time_as_upper_bound_and_drops_translation_urls():
    payload = {
        "results": [
            {
                "title": "Foreign Secretary statement on Syria",
                "link": "/government/news/syria",
                "public_timestamp": "2014-03-01T12:00:00Z",
            },
            {
                "title": "Foreign Secretary statement on Iran",
                "link": "/government/news/iran",
                "public_timestamp": "2020-01-01",
            },
            {
                "title": "Foreign Secretary statement on Egypt",
                "link": "/government/news/egypt.fr",
                "public_timestamp": "2014-01-01",
            },
        ]
    }
    rows = official.govuk_rows(payload)
    assert rows[0]["date"] == "2014-03-01T12:00:00Z"
    assert len(rows) == 2
    out, _ = official.eligible(pd.DataFrame(rows))
    assert out["text"].tolist() == [payload["results"][0]["title"]]


def test_govuk_requests_only_metadata_and_paginates():
    urls = []

    class FakeFetch:
        def get(self, url):
            urls.append(url)
            params = httpx.URL(url).params
            assert httpx.URL(url).params.get_list("fields") == [
                "title", "link", "public_timestamp"
            ]
            assert "order" not in params
            assert params["filter_organisations[]"] == "foreign-commonwealth-office"
            i = int(params["start"])
            return json.dumps(
                {
                    "total": 2,
                    "results": [
                        {
                            "title": f"Foreign Secretary statement on Syria {i}",
                            "link": f"/government/news/syria-{i}",
                            "public_timestamp": "2014-04-03",
                        }
                    ],
                }
            ).encode()

    assert len(official.fetch_govuk(FakeFetch())) == 2
    assert len(urls) == 2


def test_raw_cache_replays_and_dry_run_writes_nothing(tmp_path):
    calls = []

    def respond(request):
        calls.append(str(request.url))
        body = "User-agent: *\nAllow: /\n" if request.url.path == "/robots.txt" else "raw archive"
        return httpx.Response(200, text=body)

    url = "https://www.ecb.europa.eu/archive/"
    for write in (False, True, False):
        f = official.Fetcher(
            write, tmp_path, httpx.Client(transport=httpx.MockTransport(respond)), 0
        )
        assert f.get(url) == b"raw archive"
        f.close()
        if not write and len(calls) == 2:
            assert list(tmp_path.iterdir()) == []
    assert len(calls) == 4  # third fetch used the two cached raw responses
    assert len(list(tmp_path.glob("*.body"))) == 2
    for path in tmp_path.glob("*.json"):
        meta = json.loads(path.read_text())
        assert meta["sha256"] == official.sha(path.with_suffix(".body").read_bytes())


@pytest.mark.parametrize("robots,status", [("User-agent: *\nDisallow: /\n", 200), ("", 403)])
def test_robot_denial_or_http_block_stops_source_without_bypass(tmp_path, robots, status):
    calls = []

    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(status, text=robots)

    f = official.Fetcher(False, tmp_path, httpx.Client(transport=httpx.MockTransport(respond)), 0)
    with pytest.raises(official.SourceUnavailable):
        f.get("https://www.ecb.europa.eu/archive/")
    assert calls == ["https://www.ecb.europa.eu/robots.txt"]
    f.close()


def test_blocked_build_is_honest_and_does_not_read_exclusion_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "SAMPLE", tmp_path / "sample.csv")
    monkeypatch.setattr(official, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(official, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(official, "ROOT", tmp_path)

    def no_transport(fetcher):
        return frame([]), {
            k: {"status": "unavailable", "rows": 0, "reason": "fake DNS failure"}
            for k in official.SOURCES
        }

    monkeypatch.setattr(official, "collect", no_transport)
    monkeypatch.setattr(
        official, "load_exclusions", lambda: pytest.fail("no source titles to filter")
    )
    dry = official.build(False)
    assert not (tmp_path / "sample.csv").exists()
    assert not (tmp_path / "manifest.json").exists()
    built = official.build(True)
    assert built["sample_sha256"] == dry["sample_sha256"]
    assert built["status"] == "blocked" and built["leak_check"] == "not_run"
    assert built["sample_rows"] == 0 and pd.read_csv(tmp_path / "sample.csv").empty


def test_final_guard_is_used_on_a_nonempty_sample(monkeypatch):
    checked = []
    original = official.assert_clean

    def guard(sample, corpus):
        checked.append(len(sample))
        original(sample, corpus)

    monkeypatch.setattr(official, "assert_clean", guard)
    official.clean_sample(
        frame(["ECB publishes a financial stability review"]),
        {"fake exclusion": ["a completely unrelated reserved item"]},
    )
    assert checked == [1]


def test_successful_build_hash_schema_terms_and_dry_run_match(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "ROOT", tmp_path)
    monkeypatch.setattr(official, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(official, "SAMPLE", tmp_path / "sample.csv")
    monkeypatch.setattr(official, "MANIFEST", tmp_path / "manifest.json")
    raw = frame(
        [
            "ECB publishes a financial stability review",
            "Foreign Secretary welcomes ceasefire in Syria",
        ],
        ["ecb", "govuk_fco"],
    )
    monkeypatch.setattr(
        official,
        "collect",
        lambda _: (raw, {k: {"status": "fake test input", "rows": 0} for k in official.SOURCES}),
    )
    monkeypatch.setattr(
        official, "load_exclusions", lambda: {"fake reserved set": ["unrelated reserved item"]}
    )
    dry = official.build(False)
    assert not (tmp_path / "sample.csv").exists()
    built = official.build(True)
    content = (tmp_path / "sample.csv").read_bytes()
    assert built["status"] == "built" and built["leak_check"] == "passed"
    assert built["sample_sha256"] == dry["sample_sha256"] == official.sha(content)
    assert pd.read_csv(tmp_path / "sample.csv").columns.tolist() == official.COLUMNS
    assert built["by_cue_class"] == {"Macro": 1, "Geo": 1}
    assert built["cue_class_is_label"] is False
    assert built["sources"]["ecb"]["attribution"] == official.SOURCES["ecb"]["attribution"]
    assert "licence" in built["sources"]["govuk_fco"]
    assert "formal" in built["style_note"]
