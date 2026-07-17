"""
tests/test_hermes_auth.py
=========================
Hermes bearer-token middleware: off by default, enforced when HERMES_API_KEY
is set, /health always open.
"""
from fastapi.testclient import TestClient

from hermes.api import build_app


class _Gov:
    async def get_ges_report(self):
        return {"forge": 42.0}

    class context:  # noqa: N801 - attribute namespace only
        chroma = None
        obs = None


def _client(fake_redis):
    app = build_app(llm=None, redis=fake_redis, fuse=None,
                    governor=_Gov(), model="claude-sonnet-4-6")
    return TestClient(app)


def test_no_key_means_open(fake_redis, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    c = _client(fake_redis)
    assert c.get("/ges").status_code == 200


def test_key_set_blocks_without_header(fake_redis, monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "s3cret")
    c = _client(fake_redis)
    r = c.get("/ges")
    assert r.status_code == 401


def test_key_set_allows_with_correct_bearer(fake_redis, monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "s3cret")
    c = _client(fake_redis)
    r = c.get("/ges", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200
    assert r.json() == {"forge": 42.0}


def test_wrong_token_rejected(fake_redis, monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "s3cret")
    c = _client(fake_redis)
    r = c.get("/ges", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_health_always_open(fake_redis, monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "s3cret")
    c = _client(fake_redis)
    assert c.get("/health").status_code == 200
