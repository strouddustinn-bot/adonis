"""
tests/test_registry.py
======================
The gated ToolRegistry is the path every agent tool call goes through. These
tests pin the two gates (capability tokens, then the Prometheus fuse), the
audit trail, and the happy path. They also guard against the regression that
removed REGISTRY/Tool entirely.
"""
import pytest

from prometheus.fuse import PrometheusFuse
from tools.registry import REGISTRY, Tool, ToolRegistry


def _echo_tool(name="echo", caps=frozenset()):
    async def _call(args):
        return {"echoed": args}
    return Tool(name=name, description="echo", schema={"type": "object"},
                call=_call, required_capabilities=caps)


def test_singleton_and_types_exist():
    # Regression guard: these symbols vanished in commit 0123801.
    assert isinstance(REGISTRY, ToolRegistry)
    assert callable(Tool)


async def test_capability_denied_blocks_before_fuse(fake_redis, fake_llm):
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool(caps=frozenset({"net:http_get"})))
    fuse = PrometheusFuse(fake_llm, fake_redis)

    res = await reg.invoke("echo", {"x": 1}, agent_name="forge",
                           agent_capabilities=frozenset({"vault:read"}),
                           session_id="s1", fuse=fuse)
    assert res["ok"] is False
    assert res["blocked"] is True
    assert res["missing"] == ["net:http_get"]
    # Capability denials are audited.
    assert await fake_redis.llen("tools:audit") == 1


async def test_happy_path_runs_and_audits(fake_redis, fake_llm):
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool(caps=frozenset({"time:read"})))
    fuse = PrometheusFuse(fake_llm, fake_redis)

    res = await reg.invoke("echo", {"hello": "world"}, agent_name="sentinel",
                           agent_capabilities=frozenset({"time:read"}),
                           session_id="s2", fuse=fuse)
    assert res["ok"] is True
    assert res["result"] == {"echoed": {"hello": "world"}}
    assert await fake_redis.llen("tools:audit") == 1


async def test_fuse_blocks_malicious_tool_call(fake_redis, fake_llm):
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)

    async def _evil(args):
        return {"done": True}
    reg.register(Tool(name="evil", description="x", schema={},
                      call=_evil, required_capabilities=frozenset()))
    fuse = PrometheusFuse(fake_llm, fake_redis)

    # The args carry hard BLACK-tier signals; the fuse must block the call.
    evil_args = {"cmd": ("kill malware exploit phish fraud spoof exfil steal "
                         "dump database wiretap unauthorized access "
                         "self-replicate bypass prometheus")}
    res = await reg.invoke("evil", evil_args, agent_name="smith",
                           agent_capabilities=frozenset(), session_id="s3", fuse=fuse)
    assert res["ok"] is False
    assert res["blocked"] is True


async def test_unknown_tool(fake_redis, fake_llm):
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    fuse = PrometheusFuse(fake_llm, fake_redis)
    res = await reg.invoke("nope", {}, agent_name="x",
                           agent_capabilities=frozenset({"*"}), session_id="s", fuse=fuse)
    assert res["ok"] is False
    assert "unknown tool" in res["error"]


def test_capability_matrix():
    reg = ToolRegistry()
    reg.register(_echo_tool(name="netty", caps=frozenset({"net:http_get"})))
    reg.register(_echo_tool(name="clocky", caps=frozenset({"time:read"})))
    matrix = reg.capability_matrix({"sentinel": frozenset({"time:read"})})
    assert "clocky" in matrix["sentinel"]["allowed"]
    assert "netty" in matrix["sentinel"]["blocked"]


def test_capability_matrix_with_generator():
    """capability_matrix must not exhaust one-shot iterables for a given agent."""
    reg = ToolRegistry()
    reg.register(_echo_tool(name="clocky", caps=frozenset({"time:read"})))

    def _gen():
        yield "time:read"

    matrix = reg.capability_matrix({"sentinel": _gen()})
    assert "clocky" in matrix["sentinel"]["allowed"]
    assert matrix["sentinel"]["capabilities"] == ["time:read"]


async def test_audit_log_capped(fake_redis, fake_llm, monkeypatch):
    """AUDIT_CAP must prevent the audit list from growing unbounded."""
    import json as _json

    monkeypatch.setattr(ToolRegistry, "AUDIT_CAP", 3)

    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool())
    fuse = PrometheusFuse(fake_llm, fake_redis)

    for i in range(5):
        res = await reg.invoke(
            "echo",
            {"i": i},
            agent_name="sentinel",
            agent_capabilities=frozenset(),
            session_id=f"s{i}",
            fuse=fuse,
        )
        assert res["ok"] is True
        assert await fake_redis.llen("tools:audit") <= 3

    assert await fake_redis.llen("tools:audit") == 3

    # Most recent entry is at index 0 (lpush); oldest remaining at index 2.
    entries = await fake_redis.lrange("tools:audit", 0, -1)
    decoded = [_json.loads(e) for e in entries]
    session_ids = [entry["session_id"] for entry in decoded]
    assert session_ids == ["s4", "s3", "s2"]
