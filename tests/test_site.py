import pytest
from fastapi.testclient import TestClient

from riskengine import site


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html>dashboard</html>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path.parent / "secret.txt").write_text("not served")
    monkeypatch.setattr(site, "DIST", tmp_path)
    return TestClient(site.app)


def test_built_files_are_served(client):
    assert client.get("/assets/app.js").text == "console.log(1)"


def test_router_paths_get_the_index(client):
    for path in ("/", "/engine", "/replay?autoplay=1"):
        assert client.get(path).text == "<html>dashboard</html>"


def test_files_outside_the_build_are_not_served(client):
    assert client.get("/../secret.txt").text == "<html>dashboard</html>"
    assert client.get("/%2e%2e/secret.txt").text == "<html>dashboard</html>"


def test_api_is_mounted_under_api(client, monkeypatch):
    monkeypatch.setattr(site.api, "eval_results", lambda: {"Sentiment": []})
    assert client.get("/api/scoreboard").json() == {"Sentiment": []}


def test_missing_build_says_how_to_build(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "DIST", tmp_path / "nothing")
    out = TestClient(site.app).get("/")
    assert out.status_code == 503
    assert "npm run build" in out.json()["detail"]
