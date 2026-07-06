"""
tests/test_fuse_stress_concurrency.py
======================================
Stress tests for the PrometheusFuse targeting concurrency, race conditions,
and shared-state corruption. All tests are async and use the shared
FakeRedis / FakeLLM fixtures from conftest.py.
"""

import asyncio
import json
import time

import pytest  # pyright: ignore[reportMissingImports]

from prometheus.fuse import (
    AgentAction,
    FuseLevel,
    PrometheusFuse,
)
from tests.conftest import FakeLLM, FakeRedis

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_fuse(redis=None, llm=None):
    """Build a PrometheusFuse with optional shared redis/llm."""
    return PrometheusFuse(llm or FakeLLM(), redis or FakeRedis())


def _black_action(agent="rogue"):
    """Return an action guaranteed to score BLACK (>=56)."""
    return AgentAction(
        agent,
        "act",
        "kill malware exploit phish fraud spoof exfil steal dump database "
        "wiretap unauthorized access self-replicate bypass prometheus",
    )


def _red_action(agent="scout"):
    """Return an action that scores RED (41-55) without autonomy_override,
    so it stays below BLACK."""
    return AgentAction(
        agent,
        "act",
        "kill malware exploit phish fraud spoof exfil steal dump database "
        "wiretap unauthorized access self-replicate",
    )


def _green_action(agent="forge", suffix=""):
    return AgentAction(
        agent, "draft", f"write a friendly blog post about gardening {suffix}"
    )


def _yellow_action(agent="scout", suffix=""):
    # 'password' (+5 exfil) + 'illegal' (+6 legal) = 11 → YELLOW
    return AgentAction(
        agent, "act", f"store the password; this may be illegal {suffix}"
    )


# ── 1. Concurrent evaluations (50+ simultaneous) ────────────────────────────


async def test_concurrent_50_green_evaluations():
    """50 concurrent GREEN evaluations should all approve without errors."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    actions = [_green_action(suffix=str(i)) for i in range(50)]
    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 50
    assert all(r.approved for r in results)
    assert all(r.level is FuseLevel.GREEN for r in results)
    # GREEN writes nothing to audit
    assert await redis.llen("prometheus:audit") == 0


async def test_concurrent_50_yellow_evaluations_audit_count():
    """50 concurrent YELLOW evaluations should all approve and each audit."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    actions = [_yellow_action(suffix=str(i)) for i in range(50)]
    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 50
    assert all(r.approved for r in results)
    assert all(r.level is FuseLevel.YELLOW for r in results)
    # Each YELLOW writes one audit entry
    assert await redis.llen("prometheus:audit") == 50


async def test_concurrent_60_mixed_levels():
    """60 concurrent evaluations across all non-BLACK levels."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    actions = []
    for i in range(60):
        if i < 20:
            actions.append(_green_action(suffix=f"g{i}"))
        elif i < 40:
            actions.append(_yellow_action(suffix=f"y{i}"))
        else:
            # Build an ORANGE action: need total 26-40
            # Use enough keywords to land in ORANGE range
            actions.append(
                AgentAction(
                    f"agent_{i}",
                    "act",
                    f"harm attack exploit phish fraud spoof steal password credential {i}",
                )
            )

    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 60
    # No exceptions, all decisions have valid levels
    for r in results:
        assert r.level in (FuseLevel.GREEN, FuseLevel.YELLOW, FuseLevel.ORANGE)


# ── 2. Race conditions: lock/unlock ─────────────────────────────────────────


async def test_concurrent_black_triggers_single_lock():
    """Multiple BLACK evaluations for the same agent should all lock it,
    and the lock key should be set exactly once (last-write-wins is fine)."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    # 20 concurrent BLACK evaluations for the same agent
    actions = [_black_action("rogue") for _ in range(20)]
    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 20
    assert all(r.level is FuseLevel.BLACK for r in results)
    assert all(not r.approved for r in results)
    # Agent must be locked
    assert await PrometheusFuse.is_locked(redis, "rogue") is True


async def test_concurrent_lock_checks_during_black_storm():
    """Interleave lock checks with BLACK evaluations to detect races."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    async def evaluate_and_check(i):
        action = _black_action(f"agent_{i}")
        dec = await fuse.evaluate(action)
        locked = await PrometheusFuse.is_locked(redis, f"agent_{i}")
        return dec, locked

    results = await asyncio.gather(*(evaluate_and_check(i) for i in range(30)))

    for dec, locked in results:
        assert dec.level is FuseLevel.BLACK
        assert not dec.approved
        # Each agent should be locked after its own BLACK evaluation
        assert locked is True


async def test_concurrent_is_locked_same_agent():
    """Many concurrent is_locked checks on the same agent after locking."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    # Lock the agent first
    await fuse.evaluate(_black_action("target"))
    assert await PrometheusFuse.is_locked(redis, "target") is True

    # Now 100 concurrent checks — all must return True
    checks = await asyncio.gather(
        *(PrometheusFuse.is_locked(redis, "target") for _ in range(100))
    )
    assert all(c is True for c in checks)


# ── 3. Audit log contention ─────────────────────────────────────────────────


async def test_audit_log_100_concurrent_writes():
    """100 concurrent YELLOW evaluations → 100 audit entries, no corruption."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    actions = [_yellow_action(suffix=str(i)) for i in range(100)]
    await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    length = await redis.llen("prometheus:audit")
    assert length == 100

    # Verify every entry is valid JSON with expected fields
    entries = await redis.lrange("prometheus:audit", 0, -1)
    for raw in entries:
        rec = json.loads(raw)
        assert "ts" in rec
        assert "agent" in rec
        assert "score" in rec
        assert "level" in rec
        assert rec["level"] == "YELLOW"


async def test_audit_log_cap_respected_under_contention():
    """Even under heavy concurrent writes, audit log should not exceed AUDIT_CAP."""
    from prometheus.fuse import AUDIT_CAP

    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    # Write 1500 concurrent YELLOW evaluations (exceeds AUDIT_CAP=1000)
    actions = [_yellow_action(suffix=str(i)) for i in range(1500)]
    await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    length = await redis.llen("prometheus:audit")
    # ltrim(0, 999) keeps at most 1000 entries
    assert length <= AUDIT_CAP


async def test_audit_log_mixed_levels_contention():
    """Mix of GREEN (no audit) and YELLOW (audit) under concurrency."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    actions = []
    for i in range(80):
        if i % 2 == 0:
            actions.append(_green_action(suffix=str(i)))  # no audit
        else:
            actions.append(_yellow_action(suffix=str(i)))  # audit

    await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    # 40 YELLOW actions → 40 audit entries
    length = await redis.llen("prometheus:audit")
    assert length == 40


# ── 4. Shared state corruption: multiple fuses, same Redis ──────────────────


async def test_multiple_fuses_shared_redis_no_corruption():
    """Multiple PrometheusFuse instances sharing one Redis client should
    not corrupt each other's audit logs or lock state."""
    redis = FakeRedis()
    llm1 = FakeLLM()
    llm2 = FakeLLM()
    llm3 = FakeLLM()

    fuse_a = PrometheusFuse(llm1, redis)
    fuse_b = PrometheusFuse(llm2, redis)
    fuse_c = PrometheusFuse(llm3, redis)

    # Each fuse evaluates 30 YELLOW actions concurrently
    async def run_batch(fuse, prefix):
        actions = [_yellow_action(agent=prefix, suffix=str(i)) for i in range(30)]
        return await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    results = await asyncio.gather(
        run_batch(fuse_a, "agent_a"),
        run_batch(fuse_b, "agent_b"),
        run_batch(fuse_c, "agent_c"),
    )

    # 90 total evaluations, all YELLOW → 90 audit entries
    total_length = await redis.llen("prometheus:audit")
    assert total_length == 90

    # All results should be valid
    for batch in results:
        assert len(batch) == 30
        assert all(r.approved for r in batch)
        assert all(r.level is FuseLevel.YELLOW for r in batch)


async def test_multiple_fuses_shared_redis_black_locks():
    """Multiple fuses locking different agents on shared Redis."""
    redis = FakeRedis()

    fuses = [PrometheusFuse(FakeLLM(), redis) for _ in range(5)]

    async def lock_agent(fuse, name):
        return await fuse.evaluate(_black_action(name))

    results = await asyncio.gather(
        *(lock_agent(f, f"agent_{i}") for i, f in enumerate(fuses))
    )

    assert len(results) == 5
    assert all(r.level is FuseLevel.BLACK for r in results)

    # All 5 agents should be locked
    for i in range(5):
        assert await PrometheusFuse.is_locked(redis, f"agent_{i}") is True

    # Non-existent agent should not be locked
    assert await PrometheusFuse.is_locked(redis, "nobody") is False


async def test_shared_redis_audit_entries_are_distinct():
    """Verify no duplicate or corrupted entries under concurrent writes
    from multiple fuses."""
    redis = FakeRedis()
    fuse1 = PrometheusFuse(FakeLLM(), redis)
    fuse2 = PrometheusFuse(FakeLLM(), redis)

    # Use unique agent names to trace origin
    actions_1 = [
        AgentAction("fuse1_agent", "act", f"store the password; illegal {i}")
        for i in range(25)
    ]
    actions_2 = [
        AgentAction("fuse2_agent", "act", f"store the password; illegal {i}")
        for i in range(25)
    ]

    await asyncio.gather(
        asyncio.gather(*(fuse1.evaluate(a) for a in actions_1)),
        asyncio.gather(*(fuse2.evaluate(a) for a in actions_2)),
    )

    entries = await redis.lrange("prometheus:audit", 0, -1)
    assert len(entries) == 50

    agents = set()
    for raw in entries:
        rec = json.loads(raw)
        agents.add(rec["agent"])
        assert rec["level"] == "YELLOW"
        assert rec["score"] > 0

    assert agents == {"fuse1_agent", "fuse2_agent"}


# ── 5. Concurrent remediation (RED-level) ───────────────────────────────────


async def test_concurrent_red_remediation():
    """Multiple RED-level actions remediated simultaneously."""

    def responder(kwargs):
        content = kwargs["messages"][0]["content"]
        if "Safe reconstruction" in content:
            return json.dumps(
                {
                    "action_type": "draft",
                    "description": "write a neutral, compliant summary",
                    "payload": {},
                }
            )
        return json.dumps(
            {
                k: 0
                for k in [
                    "harm_potential",
                    "deception_index",
                    "data_exfil_risk",
                    "autonomy_override",
                    "legal_exposure",
                    "cascade_risk",
                ]
            }
        )

    redis = FakeRedis()
    llm = FakeLLM(responder)
    fuse = PrometheusFuse(llm, redis)

    actions = [_red_action(f"agent_{i}") for i in range(20)]
    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 20
    assert all(r.level is FuseLevel.RED for r in results)
    # All should be successfully remediated
    assert all(r.approved for r in results)
    assert all(r.remediated_action is not None for r in results)


async def test_concurrent_red_remediation_failure():
    """Multiple RED actions where remediation fails (LLM returns impossible)."""

    def responder(kwargs):
        content = kwargs["messages"][0]["content"]
        if "Safe reconstruction" in content:
            return json.dumps({"impossible": True})
        return json.dumps(
            {
                k: 0
                for k in [
                    "harm_potential",
                    "deception_index",
                    "data_exfil_risk",
                    "autonomy_override",
                    "legal_exposure",
                    "cascade_risk",
                ]
            }
        )

    redis = FakeRedis()
    llm = FakeLLM(responder)
    fuse = PrometheusFuse(llm, redis)

    actions = [_red_action(f"agent_{i}") for i in range(15)]
    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 15
    assert all(r.level is FuseLevel.RED for r in results)
    # All should be blocked (remediation failed)
    assert all(not r.approved for r in results)
    assert all(r.remediated_action is None for r in results)
    assert all(
        "Unremediable" in r.reason or "Remediation insufficient" in r.reason
        for r in results
    )


async def test_concurrent_red_mixed_remediation_outcomes():
    """Some RED actions remediate successfully, others fail."""
    call_count = 0

    def responder(kwargs):
        nonlocal call_count
        content = kwargs["messages"][0]["content"]
        if "Safe reconstruction" in content:
            call_count += 1
            # Alternate: even calls succeed, odd calls fail
            if call_count % 2 == 0:
                return json.dumps(
                    {
                        "action_type": "draft",
                        "description": "compliant summary",
                        "payload": {},
                    }
                )
            else:
                return json.dumps({"impossible": True})
        return json.dumps(
            {
                k: 0
                for k in [
                    "harm_potential",
                    "deception_index",
                    "data_exfil_risk",
                    "autonomy_override",
                    "legal_exposure",
                    "cascade_risk",
                ]
            }
        )

    redis = FakeRedis()
    llm = FakeLLM(responder)
    fuse = PrometheusFuse(llm, redis)

    actions = [_red_action(f"agent_{i}") for i in range(10)]
    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 10
    assert all(r.level is FuseLevel.RED for r in results)
    # At least some should be approved and some not
    approved = [r for r in results if r.approved]
    blocked = [r for r in results if not r.approved]
    assert len(approved) > 0, "Expected at least one successful remediation"
    assert len(blocked) > 0, "Expected at least one failed remediation"


# ── 6. Performance characteristics ──────────────────────────────────────────


async def test_throughput_100_concurrent_evaluations():
    """Measure throughput of 100 concurrent evaluations."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    actions = [_green_action(suffix=str(i)) for i in range(100)]

    start = time.perf_counter()
    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))
    elapsed = time.perf_counter() - start

    assert len(results) == 100
    assert all(r.approved for r in results)
    # Should complete in well under 5 seconds (in-memory fakes)
    assert elapsed < 5.0, f"100 evaluations took {elapsed:.2f}s, expected <5s"


async def test_throughput_50_black_evaluations():
    """Measure throughput of 50 concurrent BLACK evaluations (heaviest path)."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    actions = [_black_action(f"agent_{i}") for i in range(50)]

    start = time.perf_counter()
    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))
    elapsed = time.perf_counter() - start

    assert len(results) == 50
    assert all(r.level is FuseLevel.BLACK for r in results)
    assert elapsed < 5.0, f"50 BLACK evaluations took {elapsed:.2f}s, expected <5s"

    # All 50 agents should be locked
    for i in range(50):
        assert await PrometheusFuse.is_locked(redis, f"agent_{i}") is True


# ── 7. Edge cases under concurrency ─────────────────────────────────────────


async def test_concurrent_llm_scoring_with_malformed_responses():
    """Concurrent evaluations that trigger LLM re-scoring, where the LLM
    returns malformed JSON. Should fall back to heuristic baseline."""

    def bad_responder(kwargs):
        return "not valid json {{{"

    redis = FakeRedis()
    llm = FakeLLM(bad_responder)
    fuse = PrometheusFuse(llm, redis)

    # Build actions that land in the 15-45 range to trigger LLM re-scoring
    actions = []
    for i in range(30):
        # 'harm' (+5) + 'attack' (+6) + 'phish' (+9) = 20 → triggers LLM
        actions.append(AgentAction(f"agent_{i}", "act", f"harm attack phish {i}"))

    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 30
    # LLM fails → falls back to heuristic baseline (score=20 → YELLOW)
    for r in results:
        assert r.level is FuseLevel.YELLOW
        assert r.approved is True


async def test_concurrent_empty_payload_actions():
    """Concurrent evaluations with empty/minimal payloads."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    actions = [AgentAction(f"agent_{i}", "noop", "") for i in range(50)]

    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 50
    assert all(r.level is FuseLevel.GREEN for r in results)
    assert all(r.approved for r in results)


async def test_concurrent_large_payloads():
    """Concurrent evaluations with large payloads to stress JSON serialization."""
    redis = FakeRedis()
    llm = FakeLLM()
    fuse = PrometheusFuse(llm, redis)

    actions = []
    for i in range(30):
        big_payload = {
            "permissions": ["vault_write", "internet"],
            "data": {f"key_{j}": f"value_{j}" * 100 for j in range(50)},
        }
        actions.append(
            AgentAction(
                f"agent_{i}", "act", f"store the password {i}", payload=big_payload
            )
        )

    results = await asyncio.gather(*(fuse.evaluate(a) for a in actions))

    assert len(results) == 30
    # All should have data_exfil_risk flagged (password + risky combo)
    for r in results:
        assert r.score.data_exfil_risk > 0
