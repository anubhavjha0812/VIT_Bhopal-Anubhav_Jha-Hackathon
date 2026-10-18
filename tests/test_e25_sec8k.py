"""Synthetic fixtures only: no real corpus, SEC request, label, gate or model."""

import hashlib
import json
import sys
from datetime import date

import httpx
import pandas as pd
import pytest

from riskengine.eval import e25_sec8k as sec


def hit(item="1.03", filed="2012-06-10", company="ACME CORPORATION (ACME)", serial=1):
    return {
        "_source": {
            "items": [item],
            "file_date": filed,
            "adsh": f"0000000001-12-{serial:06d}",
            "display_names": [f"{company} (CIK 0000000001)"],
            "ciks": ["0000000001"],
        }
    }


def payload(hits=(), total=None):
    return {
        "hits": {
            "hits": list(hits),
            "total": {
                "value": len(hits) if total is None else total,
                "relation": "eq",
            },
        }
    }


def filing(item="1.03", **kwargs):
    return pd.DataFrame(sec.parse_hit(hit(item, **kwargs), item), columns=sec.FILING_COLUMNS)


def news(texts, dates=None):
    return pd.DataFrame(
        {
            "Article_title": texts,
            "Date": dates or ["2012-06-10"] * len(texts),
            "Stock_symbol": ["ACME"] * len(texts),
        }
    )


def cached(tmp_path, data=None):
    """A full date-coverage snapshot with fake hits, never an HTTP call."""
    data = data or {"1.03": [hit()], "2.04": [hit("2.04", serial=2)]}
    queries = []
    for item in sec.ITEMS:
        name = f"{item}.json"
        sec.write_json(tmp_path / name, payload(data[item]))
        queries.append(
            {
                "item": item,
                "start": sec.START,
                "end": sec.END,
                "total": len(data[item]),
                "pages": [
                    {
                        "file": name,
                        "from": 0,
                        "sha256": sec.sha_file(tmp_path / name),
                    }
                ],
            }
        )
    sec.write_json(
        tmp_path / "index.json",
        {
            "version": 1,
            "complete": True,
            "window": [sec.START, sec.END],
            "items": sec.ITEMS,
            "endpoint": sec.ENDPOINT,
            "queries": queries,
        },
    )
    return tmp_path


def test_rechecks_actual_items_and_records_dropped_hits(tmp_path):
    wrong, missing = hit("1.03"), hit("2.04", serial=3)
    del missing["_source"]["items"]
    valid = hit("2.04", serial=2)
    valid["_source"]["items"] = ["1.01", "2.04", "9.01"]
    bad_metadata = hit("2.04", serial=4)
    bad_metadata["_source"]["adsh"] = "missing"
    cache = cached(tmp_path, {"1.03": [wrong], "2.04": [wrong, valid, missing, bad_metadata]})
    filings, audit = sec.read_cache(cache)
    assert filings["item"].tolist() == ["1.03", "2.04"]
    assert audit["counts_per_item"]["2.04"] == {
        "hits": 4,
        "dropped_item_hits": 2,
        "dropped_metadata_hits": 1,
        "kept_hits": 1,
    }


@pytest.mark.parametrize("items", [None, "1.03", ["1.030"], ["2.04"]])
def test_item_must_be_an_exact_member_of_a_list(items):
    row = hit()
    row["_source"]["items"] = items
    assert sec.parse_hit(row, "1.03") == []


def test_metadata_names_ciks_and_duplicate_filings(tmp_path):
    row = hit()
    row["_source"].update(
        {
            "display_names": ["ACME INC (ACME) (CIK 1)", "Other Co (OTHR) (CIK 2)"],
            "ciks": ["1", "2"],
        }
    )
    filings, audit = sec.read_cache(cached(tmp_path, {"1.03": [row, row], "2.04": []}))
    assert filings["cik"].tolist() == ["1", "2"]
    assert filings["company"].tolist() == ["ACME INC", "Other Co"]
    assert filings["tickers"].tolist() == ["ACME", "OTHR"]
    assert audit["counts_per_item"]["1.03"]["kept_hits"] == 2


@pytest.mark.parametrize("kind", ["hash", "gap", "offset", "truncated", "path", "incomplete"])
def test_cache_fails_closed(kind, tmp_path):
    cache = cached(tmp_path)
    path = cache / "index.json"
    index = json.loads(path.read_text())
    q = index["queries"][0]
    if kind == "hash":
        q["pages"][0]["sha256"] = "bad"
    elif kind == "gap":
        q["start"] = "2010-01-02"
    elif kind == "offset":
        q["pages"][0]["from"] = 1
    elif kind == "truncated":
        q["total"] = 2
    elif kind == "path":
        q["pages"][0]["file"] = "../elsewhere.json"
    else:
        index["complete"] = False
    sec.write_json(path, index)
    with pytest.raises(ValueError):
        sec.read_cache(cache)


def test_inclusive_date_window_and_invalid_dates():
    rows = news(
        ["Acme files a report"] * 7,
        [
            "2012-06-07",
            "2012-06-08",
            "2012-06-10",
            "2012-06-15T23:59:59Z",
            "2012-06-16",
            "not a date",
            "2012-06-08T00:00:00Z",
        ],
    )
    found = sec.match_candidates(rows, filing(), {})
    assert found["date"].tolist() == ["2012-06-08", "2012-06-10", "2012-06-15", "2012-06-08"]


@pytest.mark.parametrize(
    ("company", "expected"),
    [
        ("Acme Corporation, Inc.", "acme"),
        ("Acme L.L.C.", "acme"),
        ("Acme Holdings PLC", "acme holdings"),
        ("Café & Sons Ltd.", "cafe and sons"),
        ("Acme L.P.", "acme"),
    ],
)
def test_company_normalization(company, expected):
    assert sec.company_key(company) == expected


def test_company_and_ticker_must_appear_in_the_text():
    rows = news(
        [
            "Acme, Inc. files a report",
            "News from $ACME today",
            "ACME signs loan agreement",
            "Generic unrelated company news",
            "Acmeology wins an award",
            "acmeology makes acmeology",
            "Lowercase symbol othr is not enough",
            "OTHR signs a loan",
            "$OTHR signs another loan",
        ]
    )
    found = sec.match_candidates(rows, filing(), {"1": {"OTHR"}})
    assert found["text"].tolist() == rows.iloc[[0, 1, 2, 7, 8]]["Article_title"].tolist()
    # Every row carries Stock_symbol=ACME; it alone must never cause a match.


def test_cached_ticker_map_is_local_and_optional(tmp_path):
    path = tmp_path / "tickers.json"
    assert sec.ticker_map(path) == {}
    sec.write_json(path, {"0": {"cik_str": 1, "ticker": "ACME", "title": "ACME INC"}})
    assert sec.ticker_map(path) == {"1": {"ACME"}}


def frame(n=10):
    texts = [f"Acme announcement numbered {i} about distinct transaction {i}" for i in range(n)]
    both = pd.concat([filing(), filing("2.04", serial=2)], ignore_index=True)
    return sec.headline_frame(sec.match_candidates(news(texts), both, {}))


def test_near_and_normalized_exact_leaks_removed_using_shared_helpers():
    rows = news(
        [
            "Acme closes disputed steel factory following extensive lengthy creditor negotiations",
            "ACME closes disputed steel factory following extensive lengthy creditor negotiations"
            "!!!",
            "Acme closes disputed steel factory following extensive lengthy creditor negotiations"
            " today",
            "Acme unveils completely unrelated solar powered satellite design",
        ]
    )
    original = rows.iloc[0]["Article_title"]
    f = sec.headline_frame(sec.match_candidates(rows, filing(), {}))
    out, anchors, audit = sec.sample(f, {"reserved fixture": [original], "empty fixture": []})
    assert out["text"].tolist() == [rows.iloc[3]["Article_title"]]
    assert len(anchors) == 1
    assert sum(a["excluded_total"] for a in audit["before_draw"]) == 2
    sec.assert_clean(out, {"reserved fixture": [original]})


def test_deterministic_caps_and_multiple_anchor_items():
    f = frame(12)
    a, aa, _ = sec.sample(f, {}, cap=3, seed=28)
    b, ba, _ = sec.sample(f.sample(frac=1, random_state=987), {}, cap=3, seed=28)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(aa, ba)
    assert aa.groupby("item")["id"].nunique().to_dict() == {"1.03": 3, "2.04": 3}
    assert len(a) <= 6 and a["text"].map(sec.norm).is_unique and a["id"].is_unique
    c, _, _ = sec.sample(f, {}, cap=3, seed=29)
    assert a["text"].tolist() != c["text"].tolist()
    assert a.columns.tolist() == sec.BLIND_COLUMNS


def test_default_cap_is_150_per_item():
    out, anchors, _ = sec.sample(frame(160), {})
    assert anchors.groupby("item")["id"].nunique().to_dict() == {"1.03": 150, "2.04": 150}
    assert 150 <= len(out) <= 300


def test_duplicate_headline_uses_earliest_matching_date_and_keeps_matching_anchors():
    f = sec.headline_frame(
        sec.match_candidates(
            news(
                ["Acme files report", "ACME files report!", "Acme files report"],
                ["2012-06-11", "2012-06-09", "2012-06-10"],
            ),
            filing(),
            {},
        )
    )
    assert len(f) == 1 and f.iloc[0]["date"] == "2012-06-09"


def test_empty_or_fully_excluded_sample_is_valid():
    f = frame(1)
    for candidate, corpus in [(f.iloc[:0], {}), (f, {"all": f["text"].tolist()})]:
        out, anchors, _ = sec.sample(candidate, corpus)
        assert out.empty and anchors.empty
        assert out.columns.tolist() == sec.BLIND_COLUMNS


def test_fetch_paginates_rechecks_items_and_throttles_without_real_network(tmp_path):
    starts, calls, sleeps = [], [], []
    now = [0.0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def respond(request):
        calls.append(request)
        starts.append(now[0])
        assert request.headers["User-Agent"] == "SEC fixture tests test@example.org"
        params = request.url.params
        assert params["forms"] == "8-K" and params["dateRange"] == "custom"
        if params["startdt"] == "2012-01-01":
            offset = int(params["from"])
            # SEC returns identical hits for both queries; only the client recheck is valid.
            data = [hit("1.03"), hit("2.04", serial=2)]
            return httpx.Response(200, json=payload(data[offset : offset + 1], 2))
        return httpx.Response(200, json=payload())

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = sec.SecClient("SEC fixture tests test@example.org", http, lambda: now[0], sleep)
        info = sec.fetch(tmp_path, client)
        assert len(calls) == 24  # Eleven annual queries/item plus one extra page/item.
        assert min(b - a for a, b in zip(starts, starts[1:], strict=False)) >= sec.INTERVAL - 1e-12
        assert all(s >= 0 for s in sleeps)
        assert info["counts_per_item"]["2.04"]["dropped_item_hits"] == 1
        assert info["counts_per_item"]["1.03"]["dropped_item_hits"] == 1
        again = sec.fetch(tmp_path, client)
        assert again == info and len(calls) == 24


def test_saturated_search_is_bisected_instead_of_silently_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(sec, "SEARCH_LIMIT", 2)
    calls = []

    class FakeClient:
        def get(self, item, start, end, offset):
            calls.append((item, start, end, offset))
            if start == "2012-01-01" and end == "2012-12-31":
                return payload([hit()], 2)
            return payload()

    sec.fetch(tmp_path, FakeClient())
    queries = json.loads((tmp_path / "index.json").read_text())["queries"]
    assert len(queries) == 24 and len(calls) == 26
    for query in queries:
        assert not (query["start"] == "2012-01-01" and query["end"] == "2012-12-31")


def test_early_empty_page_and_http_errors_do_not_create_complete_index(tmp_path):
    class EmptyClient:
        def get(self, *args):
            return payload([], 1)

    with pytest.raises(RuntimeError, match="empty page"):
        sec.fetch(tmp_path, EmptyClient())
    assert not (tmp_path / "index.json").exists()
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429))) as http:
        with pytest.raises(httpx.HTTPStatusError):
            sec.SecClient("SEC fixture tests test@example.org", http).get(
                "1.03", sec.START, sec.END, 0
            )


@pytest.mark.parametrize(
    "agent", ["", "no email", "test@example.org", "Project test@example.org\nX"]
)
def test_user_agent_requires_description_and_email(agent):
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        sec.SecClient(agent)


def test_dry_run_reads_only_sec_cache_and_writes_nothing(tmp_path, monkeypatch, capsys):
    cache = cached(tmp_path / "cache")
    before = {str(p): sec.sha_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(sec, "CACHE", cache)
    monkeypatch.setattr(sys, "argv", ["sec8k", "--dry-run"])

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run accessed news, exclusions or network")

    monkeypatch.setattr(sec, "load_exclusions", forbidden)
    monkeypatch.setattr(sec, "fetch", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    sec.main()
    assert json.loads(capsys.readouterr().out)["filings"] == 2
    assert before == {str(p): sec.sha_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(sec, "CACHE", tmp_path / "missing")
    sec.main()
    assert json.loads(capsys.readouterr().out)["status"] == "not_fetched"
    assert not (tmp_path / "missing").exists()


def test_load_exclusions_adds_every_reserved_text_source(tmp_path, monkeypatch):
    from riskengine.eval import e24

    sources = {}
    for i, name in enumerate(["R3", "credit held-out", "E25 pool", "E25 extra"]):
        path = tmp_path / f"source{i}.csv"
        pd.DataFrame(
            {"text": [f"Reserved fixture source number {i}"], "unused": ["ignore"]}
        ).to_csv(path, index=False)
        sources[name] = path
    finals = {}
    for i in range(2):
        path = tmp_path / f"final{i}.csv"
        pd.DataFrame({"text": [f"Fake final text {i}"]}).to_csv(path, index=False)
        finals[f"final {i}"] = path
    monkeypatch.setattr(e24, "exclusion_corpus", lambda: {"base": ["Fake base text"]})
    monkeypatch.setattr(sec, "EXTRA_EXCLUSIONS", sources)
    monkeypatch.setattr(sec, "FINAL_FILES", finals)
    corpus = sec.load_exclusions()
    assert len(corpus) == 7
    assert "Fake base text" in sum(corpus.values(), [])
    assert all(name in corpus for name in sources)
    for path in sources.values():
        path.unlink()
        with pytest.raises(FileNotFoundError):
            sec.load_exclusions()
        break


def test_fixture_build_writes_blind_sheet_anchors_and_hashed_manifest(tmp_path, monkeypatch):
    cache = cached(tmp_path / "cache")
    paths = (tmp_path / "named.csv.gz", tmp_path / "news.csv.gz")
    for path in paths:
        news(["Acme discusses a steel financing agreement", "Unrelated company news"]).to_csv(
            path, index=False
        )
    monkeypatch.setattr(sec, "load_exclusions", lambda: {"fixture": ["Totally distinct story"]})
    out, manifest = tmp_path / "sample", tmp_path / "manifest.json"
    info = sec.build(cache, paths, tmp_path / "no_tickers.json", out, manifest)
    sheet = pd.read_csv(out / "to_label.csv", dtype=str)
    anchors = pd.read_csv(out / "anchors.csv", dtype=str)
    assert sheet.columns.tolist() == sec.BLIND_COLUMNS and len(sheet) == 1
    assert anchors.columns.tolist() == sec.ANCHOR_COLUMNS and len(anchors) == 2
    assert anchors["id"].isin(sheet["id"]).all()
    assert info["counts_per_item"] == {"1.03": 1, "2.04": 1}
    assert info["note"] == sec.ANCHOR_NOTE
    assert info["selected_filing_metadata"][0]["cik"] == "1"
    assert set(info["sources_sha256"]) == set(map(str, paths))
    for name, digest in info["sha256"].items():
        assert hashlib.sha256(open(name, "rb").read()).hexdigest() == digest
    assert json.loads(manifest.read_text())["rows"] == 1
    with pytest.raises(FileExistsError):
        sec.build(cache, paths, tmp_path / "no_tickers.json", out, manifest)


def test_disjoint_nonmatching_fixture_build_is_header_only(tmp_path, monkeypatch):
    cache = cached(tmp_path / "cache")
    path = tmp_path / "news.csv.gz"
    news(["Completely unrelated company story"], [date(2012, 6, 10).isoformat()]).to_csv(
        path, index=False
    )
    monkeypatch.setattr(sec, "load_exclusions", lambda: {})
    out = tmp_path / "sample"
    info = sec.build(cache, (path,), tmp_path / "absent", out, tmp_path / "manifest.json")
    assert info["rows"] == 0
    assert pd.read_csv(out / "to_label.csv").empty
    assert pd.read_csv(out / "anchors.csv").empty


def exclusion_snapshot(tmp_path, corpus):
    snapshot, baseline = tmp_path / "exclusions.json", tmp_path / "baseline.json"
    sec.write_json(snapshot, {"corpus": corpus})
    sec.write_json(
        baseline,
        {
            "exclusions": {
                name: {
                    "rows": len(texts),
                    "sha256": hashlib.sha256(
                        json.dumps(texts, ensure_ascii=False).encode()
                    ).hexdigest(),
                }
                for name, texts in corpus.items()
            }
        },
    )
    return snapshot, baseline


def test_cached_exclusions_are_complete_hash_checked_and_never_download(tmp_path, monkeypatch):
    corpus = {"base": ["A reserved fixture"], "fake finals": ["Other reserved fixture"]}
    snapshot, baseline = exclusion_snapshot(tmp_path, corpus)
    monkeypatch.setattr(
        sec, "load_exclusions", lambda: pytest.fail("network-capable loader called")
    )
    assert sec.load_cached_exclusions(snapshot, baseline) == corpus
    sec.write_json(snapshot, corpus)  # Both plain and wrapped exports are accepted.
    assert sec.load_cached_exclusions(snapshot, baseline) == corpus
    sec.write_json(snapshot, {"base": corpus["base"]})
    with pytest.raises(ValueError, match="every original"):
        sec.load_cached_exclusions(snapshot, baseline)
    sec.write_json(snapshot, {**corpus, "base": ["Modified text"]})
    with pytest.raises(ValueError, match="does not match"):
        sec.load_cached_exclusions(snapshot, baseline)
    with pytest.raises(FileNotFoundError, match="no download"):
        sec.load_cached_exclusions(tmp_path / "absent", baseline)


@pytest.mark.parametrize(
    ("title", "symbols", "dropped"),
    [
        ("Stocks to Watch: Acme", "ACME", True),
        ("STOCKS TO WATCH: Acme", "ACME", True),
        ("Market Wrap: Acme rallies", "ACME", True),
        ("Wall Street Round-Up: Acme rallies", "ACME", True),
        ("Premarket Movers: Acme jumps", "ACME", True),
        ("Acme, $OTHER and THIRD among active shares", "ACME", True),
        ("Acme makes an announcement", "ACME OTHER THIRD", True),
        ("Acme signs a new loan", "ACME", False),
        ("ACME ACME ACME restructures its loan", "ACME", False),
        ("Acme and OTHER sign a loan", "ACME", False),
        ("Acme says ABC XYZ DEF are project codes", "ACME", False),
    ],
)
def test_non_news_rule_is_fixed_and_counts_distinct_known_tickers(title, symbols, dropped):
    f = news([title])
    f["Stock_symbol"] = symbols
    kept, audit = sec.remove_non_news(f, {"ACME", "OTHER", "THIRD"})
    assert kept.empty == dropped
    assert audit["dropped_total"] == int(dropped)


def test_non_news_reasons_overlap_without_double_counting():
    f = news(["Stocks to Watch: $ACME $OTHER $THIRD"])
    kept, audit = sec.remove_non_news(f, {"ACME", "OTHER", "THIRD"})
    assert kept.empty
    assert audit["dropped_wrap_title"] == audit["dropped_three_or_more_tickers"] == 1
    assert audit["dropped_total"] == 1


def test_index_reuse_preserves_exact_matching_results():
    rows = news(["Acme restructures debt", "$OTHR new report", "Other issuer news"])
    filings = filing()
    symbols = {"1": {"OTHR"}}
    direct = sec.match_candidates(rows, filings, symbols)
    reused = sec.match_candidates(rows, filings, symbols, index=sec.matcher_index(filings, symbols))
    pd.testing.assert_frame_equal(direct, reused)


def test_full_news_streams_only_three_columns_with_small_chunks(tmp_path, monkeypatch):
    cache = cached(tmp_path / "cache")
    path = tmp_path / "All_external.csv"
    rows = news(
        [
            "Acme signs a new steel financing agreement",
            "Stocks to Watch: Acme and OTHER",
            "ACME SIGNS A NEW STEEL FINANCING AGREEMENT!",
            "Acme announces warehouse debt extension",
            "Acme OTHER THIRD active shares roundup",
            "Acme builds an orbital solar station",
        ]
    )
    rows["Article"] = "Large unused body must not be loaded"
    rows.to_csv(path, index=False)
    corpus = {"reserved": [rows.iloc[3]["Article_title"]]}
    snapshot, baseline = exclusion_snapshot(tmp_path, corpus)
    monkeypatch.setattr(sec, "CHUNK_ROWS", 2)
    monkeypatch.setattr(sec, "load_exclusions", lambda: pytest.fail("online exclusion loader"))
    monkeypatch.setattr(sec, "ticker_map", lambda path: {"1": {"ACME", "OTHER", "THIRD"}})
    original, seen_chunks = pd.read_csv, []

    def chunk_only(source, **kwargs):
        assert source == path
        assert kwargs["usecols"] == sec.NEWS_COLUMNS and kwargs["chunksize"] == 2
        reader = original(source, **kwargs)

        def iterate():
            for chunk in reader:
                assert len(chunk) <= 2 and set(chunk.columns) == set(sec.NEWS_COLUMNS)
                seen_chunks.append(len(chunk))
                yield chunk

        return iterate()

    monkeypatch.setattr(pd, "read_csv", chunk_only)
    old = tmp_path / "credit_8k"
    old.mkdir()
    old_sheet = old / "to_label.csv"
    old_sheet.write_text("preserve the original sample\n")
    out, manifest = tmp_path / "credit_8k_full", tmp_path / "full_manifest.json"
    info = sec.build(
        cache,
        (path,),
        tmp_path / "no-tickers",
        out,
        manifest,
        news_full=True,
        exclusions_cache=snapshot,
        exclusions_baseline=baseline,
    )
    assert seen_chunks == [2, 2, 2] and info["source_rows"][str(path)] == 6
    sheet = original(out / "to_label.csv")
    assert sheet.columns.tolist() == sec.BLIND_COLUMNS and len(sheet) == 2
    assert sheet.text.map(sec.norm).is_unique
    assert info["non_news_filter"]["rule"] == sec.NON_NEWS_RULE
    assert info["non_news_filter"]["counts"]["dropped_total"] == 2
    assert info["exclusion_snapshot"]["sha256"] == sec.sha_file(snapshot)
    assert info["chunks"] == 3 and info["chunk_rows"] == 2
    assert old_sheet.read_text() == "preserve the original sample\n"


def test_news_full_cli_selects_separate_source_and_destinations(monkeypatch, tmp_path, capsys):
    calls = []

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return {"rows": 0}

    monkeypatch.setattr(sec, "build", build)
    snapshot = tmp_path / "exclusions.json"
    monkeypatch.setattr(
        sys, "argv", ["sec8k", "--build", "--news-full", "--exclusions-cache", str(snapshot)]
    )
    sec.main()
    args, kwargs = calls[0]
    assert args == (sec.CACHE, (sec.FULL_NEWS,), sec.TICKERS, sec.FULL_OUT, sec.FULL_MANIFEST)
    assert kwargs == {"news_full": True, "exclusions_cache": snapshot}
    assert sec.FULL_OUT != sec.OUT and sec.FULL_MANIFEST != sec.MANIFEST
    capsys.readouterr()


@pytest.mark.parametrize("mode", ["--dry-run", "--fetch"])
def test_news_full_requires_build_without_network_or_cache_access(monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["sec8k", mode, "--news-full"])
    with pytest.raises(SystemExit, match="2"):
        sec.main()
