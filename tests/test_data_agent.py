"""
tests/test_data_agent.py
========================
DATA agent: deterministic CSV/JSON profiling + aggregate inference. The LLM
interpretation layer is exercised separately via a fake.
"""
import pytest

from openclaw.agents.data import DataAgent
from persona.soul_layer import PersonaLayer


class _Gov:
    persona = PersonaLayer("claude")


class _ApproveFuse:
    async def evaluate(self, action):
        from prometheus.fuse import FuseDecision, FuseLevel, IntentScore
        return FuseDecision(action=action, score=IntentScore(),
                            level=FuseLevel.GREEN, approved=True, reason="ok")


CSV = "name,age,city\nAda,36,London\nAlan,41,London\nGrace,79,NYC\n"
JSON = '[{"name":"Ada","age":36},{"name":"Alan","age":41}]'


def _agent(fake_redis, fake_llm, fuse=None):
    return DataAgent(fake_llm, fake_redis, fuse=fuse, governor=_Gov())


async def test_profile_csv(fake_redis, fake_llm):
    a = _agent(fake_redis, fake_llm)
    res = await a.handle({"type": "profile", "data": CSV}, "s1")
    assert res["status"] == "ok"
    assert res["rows"] == 3
    assert set(res["columns"]) == {"name", "age", "city"}
    assert res["stats"]["age"]["type"] == "numeric"
    assert res["stats"]["age"]["max"] == 79.0
    assert res["stats"]["age"]["min"] == 36.0
    assert res["stats"]["city"]["type"] == "categorical"
    assert res["stats"]["city"]["top"] == "London"
    assert res["stats"]["city"]["top_count"] == 2


async def test_profile_json_autodetect(fake_redis, fake_llm):
    a = _agent(fake_redis, fake_llm)
    res = await a.handle({"type": "profile", "data": JSON}, "s1")
    assert res["rows"] == 2
    assert res["stats"]["age"]["mean"] == 38.5


async def test_profile_empty(fake_redis, fake_llm):
    a = _agent(fake_redis, fake_llm)
    res = await a.handle({"type": "profile", "data": ""}, "s1")
    assert res["status"] == "ok"
    assert res["rows"] == 0


async def test_query_computes_mean(fake_redis):
    from tests.conftest import FakeLLM
    seen = {}

    def responder(kw):
        seen["user"] = kw["messages"][0]["content"]
        return "The average age is 52."

    a = _agent(fake_redis, FakeLLM(responder), fuse=_ApproveFuse())
    res = await a.handle({"type": "data_query", "data": CSV,
                          "question": "what is the average age?"}, "s1")
    assert res["status"] == "ok"
    assert res["computed"] == {"column": "age", "op": "mean", "value": 52.0}
    # The computed aggregate is passed into the LLM prompt.
    assert "mean" in seen["user"]


async def test_query_max(fake_redis, fake_llm):
    a = _agent(fake_redis, fake_llm, fuse=_ApproveFuse())
    res = await a.handle({"type": "data_query", "data": CSV,
                          "question": "highest age in the dataset?"}, "s1")
    assert res["computed"] == {"column": "age", "op": "max", "value": 79.0}


async def test_query_blocked_by_fuse(fake_redis, fake_llm):
    class _Deny:
        async def evaluate(self, action):
            from prometheus.fuse import FuseDecision, FuseLevel, IntentScore
            return FuseDecision(action=action, score=IntentScore(),
                                level=FuseLevel.RED, approved=False, reason="no")
    a = _agent(fake_redis, fake_llm, fuse=_Deny())
    res = await a.handle({"type": "data_query", "data": CSV, "question": "avg age"}, "s1")
    assert res["status"] == "blocked"
