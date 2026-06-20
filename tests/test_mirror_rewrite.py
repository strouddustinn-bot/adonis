"""
tests/test_mirror_rewrite.py
============================
Mirror's self-rewrite loop: adoption is opt-in (MIRROR_AUTOWRITE), always
fuse-gated, persisted to a reversible runtime override store, and actually
consumed by base_agent.llm_call. Rollback clears it.
"""
import pytest

from openclaw.agents.mirror import MirrorAgent
from openclaw.agents.forge import ForgeAgent
from persona.soul_layer import PersonaLayer
from prometheus.fuse import PrometheusFuse


class _Ctx:
    obs = None


class _Gov:
    def __init__(self):
        self.context = _Ctx()
        self.persona = PersonaLayer("claude")


class _FakeFuse:
    def __init__(self, approved):
        self._approved = approved

    async def evaluate(self, action):
        from prometheus.fuse import FuseDecision, FuseLevel, IntentScore
        return FuseDecision(action=action, score=IntentScore(), level=FuseLevel.GREEN,
                            approved=self._approved, reason="test")


def _mirror(fake_redis, fake_llm, fuse):
    return MirrorAgent(fake_llm, fake_redis, fuse=fuse, governor=_Gov())


PROPOSAL = {"new_prompt": "Be maximally terse. Lead with the answer.",
            "rationale": "reduce tokens", "expected_improvement": "+15%"}


async def test_autowrite_off_only_queues(fake_redis, fake_llm, monkeypatch):
    monkeypatch.delenv("MIRROR_AUTOWRITE", raising=False)
    m = _mirror(fake_redis, fake_llm, _FakeFuse(True))
    res = await m._adopt_proposal("forge", PROPOSAL, "s1")
    assert res == {"applied": False, "mode": "queued"}
    # No override persisted.
    assert await fake_redis.get("adonis:prompt_override:forge") is None


async def test_autowrite_on_and_approved_persists_override(fake_redis, fake_llm, monkeypatch):
    monkeypatch.setenv("MIRROR_AUTOWRITE", "1")
    m = _mirror(fake_redis, fake_llm, _FakeFuse(True))
    res = await m._adopt_proposal("forge", PROPOSAL, "s1")
    assert res["applied"] is True
    assert res["mode"] == "autowrite"
    stored = await fake_redis.get("adonis:prompt_override:forge")
    assert stored is not None
    assert b"maximally terse" in stored


async def test_autowrite_blocked_by_fuse(fake_redis, fake_llm, monkeypatch):
    monkeypatch.setenv("MIRROR_AUTOWRITE", "1")
    m = _mirror(fake_redis, fake_llm, _FakeFuse(False))
    res = await m._adopt_proposal("forge", PROPOSAL, "s1")
    assert res == {"applied": False, "mode": "blocked"}
    assert await fake_redis.get("adonis:prompt_override:forge") is None


async def test_real_fuse_approves_benign_rewrite(fake_redis, fake_llm, monkeypatch):
    monkeypatch.setenv("MIRROR_AUTOWRITE", "1")
    m = _mirror(fake_redis, fake_llm, PrometheusFuse(fake_llm, fake_redis))
    res = await m._adopt_proposal("scout", PROPOSAL, "s1")
    assert res["applied"] is True


async def test_rollback_clears_override(fake_redis, fake_llm):
    m = _mirror(fake_redis, fake_llm, _FakeFuse(True))
    await fake_redis.set("adonis:prompt_override:forge", "x")
    res = await m._rollback_prompt({"agent": "forge"}, "s1")
    assert res["status"] == "ok"
    assert res["rolled_back"] is True
    assert await fake_redis.get("adonis:prompt_override:forge") is None


async def test_agent_consumes_override_in_llm_call(fake_redis):
    """base_agent.llm_call must layer the adopted directive into the system
    prompt it sends to the model."""
    seen = {}

    def responder(kwargs):
        seen["system"] = kwargs["system"]
        return "ok"

    from tests.conftest import FakeLLM
    forge = ForgeAgent(FakeLLM(responder), fake_redis, fuse=None, governor=_Gov())
    # No override yet → directive absent.
    await forge.llm_call("base system", "hi")
    assert "MIRROR-OPTIMISED DIRECTIVE" not in seen["system"]
    # Adopt an override, then the next call must include it.
    await fake_redis.set("adonis:prompt_override:forge", "Answer in one word.")
    await forge.llm_call("base system", "hi")
    assert "MIRROR-OPTIMISED DIRECTIVE" in seen["system"]
    assert "Answer in one word." in seen["system"]
