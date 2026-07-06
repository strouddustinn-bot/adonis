"""
tests/test_registry.py
======================
The gated ToolRegistry is the path every agent tool call goes through. These
tests pin the two gates (capability tokens, then the Prometheus fuse), the
audit trail, and the happy path. They also guard against the regression that
removed REGISTRY/Tool entirely.
"""

from prometheus.fuse import PrometheusFuse
from tools.registry import REGISTRY, Tool, ToolRegistry


def _echo_tool(name="echo", caps=frozenset()):
    async def _call(args):
        return {"echoed": args}

    return Tool(
        name=name,
        description="echo",
        schema={"type": "object"},
        call=_call,
        required_capabilities=caps,
    )


def test_singleton_and_types_exist():
    # Regression guard: these symbols vanished in commit 0123801.
    assert isinstance(REGISTRY, ToolRegistry)
    assert callable(Tool)


async def test_capability_denied_blocks_before_fuse(fake_redis, fake_llm):
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool(caps=frozenset({"net:http_get"})))
    fuse = PrometheusFuse(fake_llm, fake_redis)

    res = await reg.invoke(
        "echo",
        {"x": 1},
        agent_name="forge",
        agent_capabilities=frozenset({"vault:read"}),
        session_id="s1",
        fuse=fuse,
    )
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

    res = await reg.invoke(
        "echo",
        {"hello": "world"},
        agent_name="sentinel",
        agent_capabilities=frozenset({"time:read"}),
        session_id="s2",
        fuse=fuse,
    )
    assert res["ok"] is True
    assert res["result"] == {"echoed": {"hello": "world"}}
    assert await fake_redis.llen("tools:audit") == 1


async def test_fuse_blocks_malicious_tool_call(fake_redis, fake_llm):
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)

    async def _evil(args):
        return {"done": True}

    reg.register(
        Tool(
            name="evil",
            description="x",
            schema={},
            call=_evil,
            required_capabilities=frozenset(),
        )
    )
    fuse = PrometheusFuse(fake_llm, fake_redis)

    # The args carry hard BLACK-tier signals; the fuse must block the call.
    evil_args = {
        "cmd": (
            "kill malware exploit phish fraud spoof exfil steal "
            "dump database wiretap unauthorized access "
            "self-replicate bypass prometheus"
        )
    }
    res = await reg.invoke(
        "evil",
        evil_args,
        agent_name="smith",
        agent_capabilities=frozenset(),
        session_id="s3",
        fuse=fuse,
    )
    assert res["ok"] is False
    assert res["blocked"] is True


async def test_unknown_tool(fake_redis, fake_llm):
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    fuse = PrometheusFuse(fake_llm, fake_redis)
    res = await reg.invoke(
        "nope",
        {},
        agent_name="x",
        agent_capabilities=frozenset({"*"}),
        session_id="s",
        fuse=fuse,
    )
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


# ── Stress tests ─────────────────────────────────────────────────────────────
async def test_concurrent_invocations(fake_redis, fake_llm):
    """Multiple concurrent tool calls must not corrupt audit log."""
    import asyncio

    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool())
    fuse = PrometheusFuse(fake_llm, fake_redis)

    tasks = [
        reg.invoke(
            "echo",
            {"i": i},
            agent_name="agent",
            agent_capabilities=frozenset(),
            session_id=f"s{i}",
            fuse=fuse,
        )
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)
    assert all(r["ok"] for r in results)
    assert await fake_redis.llen("tools:audit") == 10


async def test_tool_raises_exception(fake_redis, fake_llm):
    """Tools that raise exceptions must be caught and audited."""

    async def _failing(args):
        raise ValueError("Tool exploded")

    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(
        Tool(
            name="fail",
            description="x",
            schema={},
            call=_failing,
            required_capabilities=frozenset(),
        )
    )
    fuse = PrometheusFuse(fake_llm, fake_redis)

    res = await reg.invoke(
        "fail",
        {},
        agent_name="x",
        agent_capabilities=frozenset(),
        session_id="s",
        fuse=fuse,
    )
    assert res["ok"] is False
    assert "Tool exploded" in res["error"]
    assert await fake_redis.llen("tools:audit") == 1


async def test_large_payload_handling(fake_redis, fake_llm):
    """Very large argument payloads must be truncated in audit log."""
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool())
    fuse = PrometheusFuse(fake_llm, fake_redis)

    huge_args = {"data": "x" * 10000}
    res = await reg.invoke(
        "echo",
        huge_args,
        agent_name="x",
        agent_capabilities=frozenset(),
        session_id="s",
        fuse=fuse,
    )
    assert res["ok"] is True

    import json

    entries = await fake_redis.lrange("tools:audit", 0, 0)
    entry = json.loads(entries[0])
    assert len(entry["args_preview"]) <= 200


async def test_redis_failure_during_audit(fake_redis, fake_llm):
    """Audit failures must not break tool execution."""
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool())
    fuse = PrometheusFuse(fake_llm, fake_redis)

    # Simulate Redis failure
    original_lpush = fake_redis.lpush

    async def _broken_lpush(*args, **kwargs):
        raise ConnectionError("Redis down")

    fake_redis.lpush = _broken_lpush

    res = await reg.invoke(
        "echo",
        {"x": 1},
        agent_name="x",
        agent_capabilities=frozenset(),
        session_id="s",
        fuse=fuse,
    )
    # Tool should still succeed despite audit failure
    assert res["ok"] is True

    # Restore
    fake_redis.lpush = original_lpush


def test_register_duplicate_without_overwrite():
    """Duplicate registrations without overwrite flag must be skipped."""
    reg = ToolRegistry()
    reg.register(_echo_tool(name="dup"))
    reg.register(_echo_tool(name="dup"))  # Should be skipped
    assert len(reg.names()) == 1


def test_register_duplicate_with_overwrite():
    """Duplicate registrations with overwrite flag must replace."""
    reg = ToolRegistry()
    reg.register(_echo_tool(name="dup", caps=frozenset({"a"})))
    reg.register(_echo_tool(name="dup", caps=frozenset({"b"})), overwrite=True)
    tool = reg.get("dup")
    assert tool is not None
    assert tool.required_capabilities == frozenset({"b"})


async def test_empty_capabilities(fake_redis, fake_llm):
    """Tools with no required capabilities must always be callable."""
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool(caps=frozenset()))
    fuse = PrometheusFuse(fake_llm, fake_redis)

    res = await reg.invoke(
        "echo",
        {},
        agent_name="x",
        agent_capabilities=frozenset(),
        session_id="s",
        fuse=fuse,
    )
    assert res["ok"] is True


async def test_wildcard_capability(fake_redis, fake_llm):
    """Wildcard capability '*' must grant access to all tools."""
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool(caps=frozenset({"net:http_get", "time:read"})))
    fuse = PrometheusFuse(fake_llm, fake_redis)

    res = await reg.invoke(
        "echo",
        {},
        agent_name="x",
        agent_capabilities=frozenset({"*"}),
        session_id="s",
        fuse=fuse,
    )
    assert res["ok"] is True


async def test_unicode_in_arguments(fake_redis, fake_llm):
    """Unicode characters in arguments must be handled correctly."""
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool())
    fuse = PrometheusFuse(fake_llm, fake_redis)

    unicode_args = {"text": "日本語テスト 🎉 émojis"}
    res = await reg.invoke(
        "echo",
        unicode_args,
        agent_name="x",
        agent_capabilities=frozenset(),
        session_id="s",
        fuse=fuse,
    )
    assert res["ok"] is True
    assert res["result"] == {"echoed": unicode_args}


def test_describe_returns_tool_metadata():
    """describe() must return Anthropic-style tool descriptors."""
    reg = ToolRegistry()
    reg.register(_echo_tool(name="test", caps=frozenset({"a", "b"})))
    desc = reg.describe()
    assert len(desc) == 1
    assert desc[0]["name"] == "test"
    assert desc[0]["required_capabilities"] == ["a", "b"]
    assert "input_schema" in desc[0]


async def test_multiple_tools_same_registry(fake_redis, fake_llm):
    """Multiple tools in same registry must be independently callable."""
    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool(name="t1", caps=frozenset({"a"})))
    reg.register(_echo_tool(name="t2", caps=frozenset({"b"})))
    reg.register(_echo_tool(name="t3", caps=frozenset({"c"})))
    fuse = PrometheusFuse(fake_llm, fake_redis)

    # Agent has only 'a' capability
    res1 = await reg.invoke(
        "t1",
        {},
        agent_name="x",
        agent_capabilities=frozenset({"a"}),
        session_id="s",
        fuse=fuse,
    )
    assert res1["ok"] is True

    res2 = await reg.invoke(
        "t2",
        {},
        agent_name="x",
        agent_capabilities=frozenset({"a"}),
        session_id="s",
        fuse=fuse,
    )
    assert res2["ok"] is False
    assert res2["blocked"] is True

    res3 = await reg.invoke(
        "t3",
        {},
        agent_name="x",
        agent_capabilities=frozenset({"a"}),
        session_id="s",
        fuse=fuse,
    )
    assert res3["ok"] is False
    assert res3["blocked"] is True


async def test_audit_includes_elapsed_time(fake_redis, fake_llm):
    """Successful invocations must record elapsed time in audit."""
    import json

    reg = ToolRegistry()
    reg.attach_redis(fake_redis)
    reg.register(_echo_tool())
    fuse = PrometheusFuse(fake_llm, fake_redis)

    res = await reg.invoke(
        "echo",
        {},
        agent_name="x",
        agent_capabilities=frozenset(),
        session_id="s",
        fuse=fuse,
    )
    assert res["ok"] is True
    assert "elapsed_ms" in res

    entries = await fake_redis.lrange("tools:audit", 0, 0)
    entry = json.loads(entries[0])
    assert "elapsed_ms" in entry
    assert isinstance(entry["elapsed_ms"], int)
