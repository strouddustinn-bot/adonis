"""
tests/test_hermes_stream.py
===========================
/ask/stream emits SSE frames (meta → deltas → done), records GES, caches the
answer, and streams a cache hit as a single delta on the next identical call.
"""
import json

import pytest
from fastapi.testclient import TestClient

from glasswing.governor import PreparedCall
from hermes.api import build_app
from tests.conftest import FakeLLM


class _Ctx:
    chroma = None
    obs = None

    def __init__(self):
        self.turns = []

    async def append_turn(self, sid, role, text):
        self.turns.append((sid, role, text))


class _Gov:
    """Stub governor with a simple per-message cache so the second identical
    call reports a cache hit, mirroring GlasswingGovernor semantics."""
    def __init__(self):
        self.context = _Ctx()
        self._cache = {}
        self.ges_calls = []

    async def prepare(self, session_id, message, *a, **k):
        if message in self._cache:
            return PreparedCall(system="", messages=[], active_agents=[],
                                think_depth="cache", cache_hit=True,
                                cached_result=self._cache[message])
        return PreparedCall(
            system="sys",
            messages=[{"role": "user", "content": message}],
            active_agents=["hermes", "forge"], think_depth="shallow",
            efficiency={"prep_ms": 5},
        )

    async def cache_result(self, message, answer):
        self._cache[message] = answer

    def record_ges(self, *a, **k):
        self.ges_calls.append((a, k))


def _frames(resp):
    out = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


def _client(monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    gov = _Gov()
    llm = FakeLLM(lambda kw: "hello streamed world")
    app = build_app(llm=llm, redis=None, fuse=None, governor=gov,
                    model="claude-sonnet-4-6")
    return TestClient(app), gov


def test_stream_emits_meta_deltas_done(monkeypatch):
    client, gov = _client(monkeypatch)
    r = client.post("/ask/stream", json={"message": "hi there", "session_id": "s1"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = _frames(r)
    types = [f["type"] for f in frames]
    assert types[0] == "meta"
    assert "delta" in types
    assert types[-1] == "done"
    # Reassembled answer matches.
    answer = "".join(f["text"] for f in frames if f["type"] == "delta")
    assert answer == "hello streamed world"
    # GES recorded once for the live (non-cache) call.
    assert len(gov.ges_calls) == 1


def test_stream_cache_hit_single_delta(monkeypatch):
    client, gov = _client(monkeypatch)
    client.post("/ask/stream", json={"message": "same q", "session_id": "s1"})
    r2 = client.post("/ask/stream", json={"message": "same q", "session_id": "s1"})
    frames = _frames(r2)
    assert frames[0]["type"] == "meta" and frames[0]["cache_hit"] is True
    deltas = [f for f in frames if f["type"] == "delta"]
    assert len(deltas) == 1
    assert deltas[0]["text"] == "hello streamed world"
    assert frames[-1] == {"type": "done", "cache_hit": True}
