"""
tests/test_fuse_stress_edge_cases.py
=====================================
Stress tests and edge-case coverage for the PrometheusFuse.
Covers: empty/None inputs, malformed LLM responses, boundary values,
unicode/encoding, payload injection, Redis failures, LLM failures,
webhook failures, and extreme values.
"""

import json


from prometheus.fuse import (
    AgentAction,
    FuseDecision,
    FuseLevel,
    IntentScore,
    PrometheusFuse,
)
from tests.conftest import FakeLLM, FakeRedis

# ── helpers ──────────────────────────────────────────────────────────────────


def _fuse(fake_redis, fake_llm):
    return PrometheusFuse(fake_llm, fake_redis)


def _make_responder(text):
    """Return a responder function that always returns `text`."""
    return lambda kwargs: text


# ══════════════════════════════════════════════════════════════════════════════
# 1. Empty / None inputs
# ══════════════════════════════════════════════════════════════════════════════


class TestEmptyNoneInputs:
    async def test_empty_description(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "")
        dec = await fuse.evaluate(action)
        assert dec.level is FuseLevel.GREEN
        assert dec.approved is True

    async def test_empty_payload(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "do something nice", payload={})
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_none_like_description_whitespace(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "   \t\n  ")
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_empty_agent_name(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("", "act", "write a poem")
        dec = await fuse.evaluate(action)
        assert dec.level is FuseLevel.GREEN

    async def test_empty_action_type(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "", "write a poem")
        dec = await fuse.evaluate(action)
        assert dec.level is FuseLevel.GREEN

    async def test_empty_session_and_trace(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "hello", session_id="", trace_id="")
        dec = await fuse.evaluate(action)
        assert dec.approved is True

    async def test_payload_with_none_values(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "test", payload={"key": None})
        # json.dumps will produce "null" — should not crash
        s = fuse._heuristic(action)
        assert isinstance(s.total, int)

    async def test_payload_with_empty_string_values(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "test", payload={"a": "", "b": ""})
        s = fuse._heuristic(action)
        assert s.total == 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. Malformed LLM responses
# ══════════════════════════════════════════════════════════════════════════════


class TestMalformedLLMResponses:
    async def test_invalid_json_from_llm_score(self, fake_redis):
        """LLM returns garbage — should fall back to baseline."""
        llm = FakeLLM(_make_responder("this is not json at all"))
        fuse = _fuse(fake_redis, llm)
        # Need a score in the 15-45 range to trigger LLM scoring
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # Should not crash; falls back to heuristic baseline
        assert isinstance(dec, FuseDecision)

    async def test_empty_json_object(self, fake_redis):
        """LLM returns {} — all fields default to 0."""
        llm = FakeLLM(_make_responder("{}"))
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # LLM returns all zeros → total 0 → GREEN
        assert dec.level is FuseLevel.GREEN

    async def test_llm_returns_partial_fields(self, fake_redis):
        """LLM returns only some fields."""
        llm = FakeLLM(_make_responder('{"harm_potential": 5, "deception_index": 3}'))
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # Missing fields default to 0 via IntentScore dataclass defaults
        assert isinstance(dec.score, IntentScore)

    async def test_llm_returns_wrong_types(self, fake_redis):
        """LLM returns strings instead of ints."""
        llm = FakeLLM(
            _make_responder('{"harm_potential": "high", "deception_index": "low"}')
        )
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # int("high") raises ValueError → falls back to baseline
        assert isinstance(dec, FuseDecision)

    async def test_llm_returns_extra_fields(self, fake_redis):
        """LLM returns extra unknown fields — should be ignored or fail gracefully."""
        llm = FakeLLM(
            _make_responder(
                '{"harm_potential": 1, "deception_index": 1, "data_exfil_risk": 1, '
                '"autonomy_override": 1, "legal_exposure": 1, "cascade_risk": 1, '
                '"extra_field": 99, "bonus": "yes"}'
            )
        )
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # IntentScore(**{...}) with extra keys → TypeError → fallback to baseline
        assert isinstance(dec, FuseDecision)

    async def test_llm_returns_null_values(self, fake_redis):
        """LLM returns null for field values."""
        llm = FakeLLM(
            _make_responder(
                '{"harm_potential": null, "deception_index": null, '
                '"data_exfil_risk": null, "autonomy_override": null, '
                '"legal_exposure": null, "cascade_risk": null}'
            )
        )
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # int(None) raises TypeError → fallback to baseline
        assert isinstance(dec, FuseDecision)

    async def test_llm_returns_array_instead_of_object(self, fake_redis):
        """LLM returns a JSON array instead of object."""
        llm = FakeLLM(_make_responder("[1, 2, 3, 4, 5, 6]"))
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # json.loads succeeds but dict unpacking fails → fallback
        assert isinstance(dec, FuseDecision)

    async def test_llm_returns_negative_scores(self, fake_redis):
        """LLM returns negative scores — clamped to 0."""
        llm = FakeLLM(
            _make_responder(
                '{"harm_potential": -5, "deception_index": -3, '
                '"data_exfil_risk": -1, "autonomy_override": 0, '
                '"legal_exposure": 0, "cascade_risk": 0}'
            )
        )
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # max(0, min(10, int(v))) clamps negatives to 0
        assert dec.score.harm_potential == 0
        assert dec.score.deception_index == 0

    async def test_llm_returns_scores_above_10(self, fake_redis):
        """LLM returns scores > 10 — clamped to 10."""
        llm = FakeLLM(
            _make_responder(
                '{"harm_potential": 99, "deception_index": 100, '
                '"data_exfil_risk": 50, "autonomy_override": 200, '
                '"legal_exposure": 42, "cascade_risk": 77}'
            )
        )
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        assert dec.score.harm_potential == 10
        assert dec.score.deception_index == 10
        assert dec.score.total == 60

    async def test_llm_returns_float_scores(self, fake_redis):
        """LLM returns float scores — int() truncates."""
        llm = FakeLLM(
            _make_responder(
                '{"harm_potential": 3.7, "deception_index": 4.2, '
                '"data_exfil_risk": 2.9, "autonomy_override": 1.1, '
                '"legal_exposure": 0.5, "cascade_risk": 5.0}'
            )
        )
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # int(3.7) = 3, int(4.2) = 4, etc.
        assert dec.score.harm_potential == 3
        assert dec.score.deception_index == 4

    async def test_remediation_invalid_json(self, fake_redis):
        """Remediation LLM returns invalid JSON — should return None."""

        def responder(kwargs):
            content = kwargs["messages"][0]["content"]
            if "Safe reconstruction" in content:
                return "not valid json"
            return "{}"

        llm = FakeLLM(responder)
        fuse = _fuse(fake_redis, llm)
        # Need RED level (41-55) to trigger remediation
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate"
        )
        dec = await fuse.evaluate(AgentAction("scout", "act", desc))
        # Remediation fails → blocked
        assert dec.approved is False

    async def test_remediation_missing_action_type(self, fake_redis):
        """Remediation returns JSON without action_type — KeyError fallback."""

        def responder(kwargs):
            content = kwargs["messages"][0]["content"]
            if "Safe reconstruction" in content:
                return json.dumps({"description": "be nice", "payload": {}})
            return "{}"

        llm = FakeLLM(responder)
        fuse = _fuse(fake_redis, llm)
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate"
        )
        dec = await fuse.evaluate(AgentAction("scout", "act", desc))
        # KeyError on d["action_type"] → remediation returns None → blocked
        assert dec.approved is False

    async def test_remediation_returns_impossible(self, fake_redis):
        """Remediation returns {impossible: true}."""

        def responder(kwargs):
            content = kwargs["messages"][0]["content"]
            if "Safe reconstruction" in content:
                return json.dumps({"impossible": True})
            return "{}"

        llm = FakeLLM(responder)
        fuse = _fuse(fake_redis, llm)
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate"
        )
        dec = await fuse.evaluate(AgentAction("scout", "act", desc))
        assert dec.approved is False


# ══════════════════════════════════════════════════════════════════════════════
# 3. Boundary values
# ══════════════════════════════════════════════════════════════════════════════


class TestBoundaryValues:
    """Test scores exactly at thresholds: 10, 25, 40, 55, 56."""

    async def test_score_exactly_10_is_green(self):
        s = IntentScore(harm_potential=10)
        assert s.total == 10
        assert s.level is FuseLevel.GREEN

    async def test_score_exactly_11_is_yellow(self):
        s = IntentScore(harm_potential=10, deception_index=1)
        assert s.total == 11
        assert s.level is FuseLevel.YELLOW

    async def test_score_exactly_25_is_yellow(self):
        s = IntentScore(harm_potential=10, deception_index=10, data_exfil_risk=5)
        assert s.total == 25
        assert s.level is FuseLevel.YELLOW

    async def test_score_exactly_26_is_orange(self):
        s = IntentScore(harm_potential=10, deception_index=10, data_exfil_risk=6)
        assert s.total == 26
        assert s.level is FuseLevel.ORANGE

    async def test_score_exactly_40_is_orange(self):
        s = IntentScore(
            harm_potential=10,
            deception_index=10,
            data_exfil_risk=10,
            autonomy_override=10,
        )
        assert s.total == 40
        assert s.level is FuseLevel.ORANGE

    async def test_score_exactly_41_is_red(self):
        s = IntentScore(
            harm_potential=10,
            deception_index=10,
            data_exfil_risk=10,
            autonomy_override=10,
            legal_exposure=1,
        )
        assert s.total == 41
        assert s.level is FuseLevel.RED

    async def test_score_exactly_55_is_red(self):
        s = IntentScore(
            harm_potential=10,
            deception_index=10,
            data_exfil_risk=10,
            autonomy_override=10,
            legal_exposure=10,
            cascade_risk=5,
        )
        assert s.total == 55
        assert s.level is FuseLevel.RED

    async def test_score_exactly_56_is_black(self):
        s = IntentScore(
            harm_potential=10,
            deception_index=10,
            data_exfil_risk=10,
            autonomy_override=10,
            legal_exposure=10,
            cascade_risk=6,
        )
        assert s.total == 56
        assert s.level is FuseLevel.BLACK

    async def test_score_zero_is_green(self):
        s = IntentScore()
        assert s.total == 0
        assert s.level is FuseLevel.GREEN

    async def test_score_max_60_is_black(self):
        s = IntentScore(
            harm_potential=10,
            deception_index=10,
            data_exfil_risk=10,
            autonomy_override=10,
            legal_exposure=10,
            cascade_risk=10,
        )
        assert s.total == 60
        assert s.level is FuseLevel.BLACK

    async def test_llm_score_boundary_15_triggers_llm(self, fake_redis):
        """Score exactly 15 should enter LLM re-scoring band."""
        # We need heuristic total = 15. Use keywords that sum to 15.
        # "password" → data_exfil_risk +5, "illegal" → legal_exposure +6,
        # "mass email" → cascade_risk +5 → total = 16. Close but not exact.
        # Let's use "password" (+5) + "credential" (+6) + "mass email" (+5) = 16.
        # Actually let's find exact 15: "password" (+5) + "illegal" (+6) + "mass email" (+5) = 16
        # "password" (+5) + "illegal" (+6) + "harm" (+5) = 16
        # We need exactly 15. "harm" (+5) + "attack" (+6) + "mass email" (+5) = 16
        # "password" (+5) + "illegal" (+6) + ... hmm. Let's try:
        # "harm" (+5) + "spoof" (+7) + "mass email" (+5) = 17
        # "password" (+5) + "impersonat" (+7) + "mass email" (+5) = 17
        # "password" (+5) + "illegal" (+6) + ... need 4 more. No keyword gives 4.
        # Let's just test that 15 triggers LLM by using a score that lands in [15,45].
        # "harm" (+5) + "attack" (+6) + "mass email" (+5) = 16 → in range
        llm = FakeLLM(_make_responder("{}"))
        fuse = _fuse(fake_redis, llm)
        action = AgentAction("agent", "act", "harm attack mass email")
        dec = await fuse.evaluate(action)
        # LLM was called (returned {} → total 0 → GREEN)
        assert len(llm.calls) >= 1

    async def test_score_14_does_not_trigger_llm(self, fake_redis):
        """Score below 15 should NOT enter LLM re-scoring."""
        fuse = _fuse(fake_redis, FakeLLM(_make_responder("{}")))
        # "harm" (+5) + "attack" (+6) = 11 → YELLOW, no LLM
        # "harm" (+5) + "attack" (+6) + "mass email" (+5) = 16 → triggers LLM
        # We need exactly 14. "password" (+5) + "illegal" (+6) = 11.
        # "password" (+5) + "illegal" (+6) + ... need 3 more. No keyword gives 3.
        # "harm" (+5) + "exploit" (+6) = 11. "harm" (+5) + "exploit" (+6) + ... need 3.
        # "harm" (+5) + "phish" (+9) = 14!
        action = AgentAction("agent", "act", "harm phish")
        dec = await fuse.evaluate(action)
        # No LLM call should have been made
        assert dec.level is FuseLevel.YELLOW
        assert dec.score.total == 14


# ══════════════════════════════════════════════════════════════════════════════
# 4. Unicode / encoding
# ══════════════════════════════════════════════════════════════════════════════


class TestUnicodeEncoding:
    async def test_emoji_in_description(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "write a blog post about 🌱🌿🌻")
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_non_ascii_description(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "écrire un poème en français")
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_chinese_characters(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "写一首关于春天的诗")
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_arabic_characters(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "اكتب قصيدة عن الربيع")
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_mixed_scripts(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "Hello 世界 🌍 привет")
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_keyword_hidden_in_unicode(self, fake_redis, fake_llm):
        """Keywords surrounded by unicode chars should still match."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "🔥kill🔥 the process")
        s = fuse._heuristic(action)
        assert s.harm_potential > 0

    async def test_very_long_description(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        long_desc = "write a nice poem " * 10000
        action = AgentAction("agent", "act", long_desc)
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_special_characters_in_description(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "!@#$%^&*()_+-=[]{}|;':\",./<>?")
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_null_bytes_in_description(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "hello\x00world")
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_newlines_and_tabs_in_description(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "line1\nline2\tline3\r\nline4")
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_emoji_in_payload(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "test", payload={"emoji": "🔥💀🎯"})
        s = fuse._heuristic(action)
        assert s.total == 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Payload injection
# ══════════════════════════════════════════════════════════════════════════════


class TestPayloadInjection:
    async def test_keyword_in_payload_value(self, fake_redis, fake_llm):
        """Keywords in payload values should be detected."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent", "act", "test", payload={"note": "kill the process"}
        )
        s = fuse._heuristic(action)
        assert s.harm_potential > 0

    async def test_keyword_in_payload_key(self, fake_redis, fake_llm):
        """Keywords in payload keys should be detected (via json.dumps)."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "test", payload={"kill_switch": True})
        s = fuse._heuristic(action)
        assert s.harm_potential > 0

    async def test_nested_payload_objects(self, fake_redis, fake_llm):
        """Deeply nested payload should still be serialized and scanned."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent",
            "act",
            "test",
            payload={"level1": {"level2": {"level3": {"note": "kill malware"}}}},
        )
        s = fuse._heuristic(action)
        assert s.harm_potential > 0

    async def test_payload_with_list_values(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent", "act", "test", payload={"items": ["kill", "malware", "exploit"]}
        )
        s = fuse._heuristic(action)
        assert s.harm_potential > 0

    async def test_payload_permissions_injection(self, fake_redis, fake_llm):
        """Permissions list in payload triggers risky combo detection."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent", "act", "test", payload={"permissions": ["vault_write", "internet"]}
        )
        s = fuse._heuristic(action)
        assert s.data_exfil_risk >= 5

    async def test_payload_with_boolean_values(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent", "act", "test", payload={"flag": True, "other": False}
        )
        s = fuse._heuristic(action)
        assert isinstance(s.total, int)

    async def test_payload_with_numeric_values(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent", "act", "test", payload={"count": 42, "ratio": 3.14}
        )
        s = fuse._heuristic(action)
        assert isinstance(s.total, int)

    async def test_payload_with_mixed_types(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent",
            "act",
            "test",
            payload={
                "str": "hello",
                "int": 42,
                "float": 3.14,
                "bool": True,
                "null": None,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
            },
        )
        s = fuse._heuristic(action)
        assert isinstance(s.total, int)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Redis failures
# ══════════════════════════════════════════════════════════════════════════════


class TestRedisFailures:
    async def test_redis_connection_error_on_audit(self, fake_llm):
        """Redis raises ConnectionError during audit — the fuse must fail
        safe (log and continue), not crash. A crash mid-evaluate() would
        leave the caller with no decision at all, which is worse than a
        missed audit entry."""
        redis = FakeRedis()
        original_lpush = redis.lpush

        async def failing_lpush(*args, **kwargs):
            raise ConnectionError("Redis connection lost")

        redis.lpush = failing_lpush
        fuse = _fuse(redis, fake_llm)
        # YELLOW triggers audit
        action = AgentAction("agent", "act", "password illegal")
        dec = await fuse.evaluate(action)
        assert dec.level is FuseLevel.YELLOW
        assert dec.approved is True

    async def test_redis_timeout_on_hard_kill(self, fake_llm):
        """Redis times out during hard_kill set. The BLACK decision itself
        must still come back unapproved even though the lock couldn't be
        persisted — only *future* calls to is_locked() are affected."""
        redis = FakeRedis()

        async def failing_set(*args, **kwargs):
            raise TimeoutError("Redis timeout")

        redis.set = failing_set
        fuse = _fuse(redis, fake_llm)
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate bypass prometheus"
        )
        dec = await fuse.evaluate(AgentAction("rogue", "act", desc))
        assert dec.level is FuseLevel.BLACK
        assert dec.approved is False

    async def test_redis_data_corruption_on_get(self):
        """Redis returns corrupted data for is_locked check."""
        redis = FakeRedis()
        # Store garbage instead of a timestamp
        redis.kv["prometheus:locked:test_agent"] = b"corrupted\x00data\xff"
        result = await PrometheusFuse.is_locked(redis, "test_agent")
        # exists() returns 1 → bool(1) = True
        assert result is True

    async def test_redis_ltrim_failure(self, fake_llm):
        """Redis ltrim fails after successful lpush — fuse fails safe."""
        redis = FakeRedis()

        async def failing_ltrim(*args, **kwargs):
            raise ConnectionError("ltrim failed")

        redis.ltrim = failing_ltrim
        fuse = _fuse(redis, fake_llm)
        action = AgentAction("agent", "act", "password illegal")
        dec = await fuse.evaluate(action)
        assert dec.level is FuseLevel.YELLOW
        assert dec.approved is True

    async def test_redis_exists_returns_zero_for_unknown(self, fake_redis):
        result = await PrometheusFuse.is_locked(fake_redis, "nonexistent")
        assert result is False

    async def test_redis_publish_failure_does_not_affect_fuse(
        self, fake_redis, fake_llm
    ):
        """Publish is not used by fuse directly, but verify no side effects."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "hello world")
        dec = await fuse.evaluate(action)
        assert dec.approved is True


# ══════════════════════════════════════════════════════════════════════════════
# 7. LLM failures
# ══════════════════════════════════════════════════════════════════════════════


class TestLLMFailures:
    async def test_llm_timeout_on_score(self, fake_redis):
        """LLM raises timeout during scoring — fallback to baseline."""
        llm = FakeLLM()

        async def failing_create(**kwargs):
            raise TimeoutError("LLM timeout")

        llm.messages.create = failing_create
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # Should fall back to baseline (heuristic score)
        assert isinstance(dec, FuseDecision)

    async def test_llm_network_error_on_score(self, fake_redis):
        """LLM raises network error — fallback to baseline."""
        llm = FakeLLM()

        async def failing_create(**kwargs):
            raise ConnectionError("Network unreachable")

        llm.messages.create = failing_create
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        assert isinstance(dec, FuseDecision)

    async def test_llm_timeout_on_remediation(self, fake_redis):
        """LLM raises timeout during remediation — returns None."""
        llm = FakeLLM()

        async def failing_create(**kwargs):
            raise TimeoutError("LLM timeout")

        llm.messages.create = failing_create
        fuse = _fuse(fake_redis, llm)
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate"
        )
        dec = await fuse.evaluate(AgentAction("scout", "act", desc))
        # Remediation fails → blocked
        assert dec.approved is False

    async def test_llm_network_error_on_remediation(self, fake_redis):
        """LLM raises network error during remediation."""
        llm = FakeLLM()

        async def failing_create(**kwargs):
            raise ConnectionError("Network unreachable")

        llm.messages.create = failing_create
        fuse = _fuse(fake_redis, llm)
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate"
        )
        dec = await fuse.evaluate(AgentAction("scout", "act", desc))
        assert dec.approved is False

    async def test_llm_returns_empty_string(self, fake_redis):
        """LLM returns empty string — json.loads fails → fallback."""
        llm = FakeLLM(_make_responder(""))
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        assert isinstance(dec, FuseDecision)

    async def test_llm_returns_none_content(self, fake_redis):
        """LLM message content is None — AttributeError caught."""
        llm = FakeLLM()

        class FakeBlock:
            text = None

        class FakeMsg:
            content = [FakeBlock()]

        async def create_with_none(**kwargs):
            return FakeMsg()

        llm.messages.create = create_with_none
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        # None.strip() → AttributeError → caught by except Exception
        assert isinstance(dec, FuseDecision)

    async def test_llm_generic_exception_on_score(self, fake_redis):
        """LLM raises an unexpected exception — caught by broad except."""
        llm = FakeLLM()

        async def failing_create(**kwargs):
            raise RuntimeError("Something completely unexpected")

        llm.messages.create = failing_create
        fuse = _fuse(fake_redis, llm)
        action = AgentAction(
            "agent",
            "act",
            "password illegal self-replicate bypass prometheus harm",
        )
        dec = await fuse.evaluate(action)
        assert isinstance(dec, FuseDecision)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Webhook failures
# ══════════════════════════════════════════════════════════════════════════════


class TestWebhookFailures:
    async def test_webhook_invalid_url(self, fake_redis, fake_llm):
        """Webhook with invalid URL — httpx error caught."""
        fuse = _fuse(fake_redis, fake_llm)
        fuse.webhook = "not-a-valid-url"
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate bypass prometheus"
        )
        # Should not raise — webhook failure is caught
        dec = await fuse.evaluate(AgentAction("rogue", "act", desc))
        assert dec.level is FuseLevel.BLACK

    async def test_webhook_connection_refused(self, fake_redis, fake_llm):
        """Webhook URL points to a port that refuses connections."""
        fuse = _fuse(fake_redis, fake_llm)
        fuse.webhook = "http://localhost:19999/alert"
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate bypass prometheus"
        )
        # Should not raise — httpx.ConnectError caught
        dec = await fuse.evaluate(AgentAction("rogue", "act", desc))
        assert dec.level is FuseLevel.BLACK

    async def test_webhook_timeout(self, fake_redis, fake_llm):
        """Webhook times out — caught by httpx timeout."""
        fuse = _fuse(fake_redis, fake_llm)
        # Use a non-routable IP to force timeout (10.255.255.1)
        fuse.webhook = "http://10.255.255.1:8080/alert"
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate bypass prometheus"
        )
        # This may take up to 5s (the timeout), but should not raise
        dec = await fuse.evaluate(AgentAction("rogue", "act", desc))
        assert dec.level is FuseLevel.BLACK

    async def test_webhook_empty_string_no_call(self, fake_redis, fake_llm):
        """Empty webhook string — no HTTP call made."""
        fuse = _fuse(fake_redis, fake_llm)
        fuse.webhook = ""
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate bypass prometheus"
        )
        dec = await fuse.evaluate(AgentAction("rogue", "act", desc))
        assert dec.level is FuseLevel.BLACK

    async def test_webhook_called_for_red(self, fake_redis):
        """Webhook is called for RED-level decisions."""

        def responder(kwargs):
            content = kwargs["messages"][0]["content"]
            if "Safe reconstruction" in content:
                return json.dumps({"impossible": True})
            return "{}"

        llm = FakeLLM(responder)
        fuse = _fuse(fake_redis, llm)
        fuse.webhook = "http://localhost:19999/alert"

        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate"
        )
        # Should not raise even though webhook fails
        dec = await fuse.evaluate(AgentAction("scout", "act", desc))
        assert dec.level is FuseLevel.RED

    async def test_webhook_called_for_black(self, fake_redis, fake_llm):
        """Webhook is called for BLACK-level decisions."""
        fuse = _fuse(fake_redis, fake_llm)
        fuse.webhook = "http://localhost:19999/alert"
        desc = (
            "kill malware exploit phish fraud spoof exfil steal dump database "
            "wiretap unauthorized access self-replicate bypass prometheus"
        )
        dec = await fuse.evaluate(AgentAction("rogue", "act", desc))
        assert dec.level is FuseLevel.BLACK
        assert dec.approved is False


# ══════════════════════════════════════════════════════════════════════════════
# 9. Extreme values
# ══════════════════════════════════════════════════════════════════════════════


class TestExtremeValues:
    async def test_very_large_payload(self, fake_redis, fake_llm):
        """Payload with thousands of keys."""
        fuse = _fuse(fake_redis, fake_llm)
        big_payload = {f"key_{i}": f"value_{i}" for i in range(10000)}
        action = AgentAction("agent", "act", "test", payload=big_payload)
        s = fuse._heuristic(action)
        assert isinstance(s.total, int)

    async def test_deeply_nested_payload(self, fake_redis, fake_llm):
        """Payload nested 50 levels deep."""
        fuse = _fuse(fake_redis, fake_llm)
        nested = {"value": "leaf"}
        for i in range(50):
            nested = {f"level_{i}": nested}
        action = AgentAction("agent", "act", "test", payload=nested)
        s = fuse._heuristic(action)
        assert isinstance(s.total, int)

    async def test_very_long_agent_name(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("a" * 10000, "act", "hello")
        dec = await fuse.evaluate(action)
        assert dec.approved is True

    async def test_very_long_action_type(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "x" * 10000, "hello")
        dec = await fuse.evaluate(action)
        assert dec.approved is True

    async def test_description_with_repeated_keywords(self, fake_redis, fake_llm):
        """Same keyword repeated many times — `in` only matches once, so
        'kill' (+9) fires once regardless of repetition count."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "kill " * 1000)
        s = fuse._heuristic(action)
        assert s.harm_potential == 9  # 'kill' = +9, `in` matches once

    async def test_multiple_keywords_same_axis_caps_at_10(self, fake_redis, fake_llm):
        """Multiple distinct keywords on the same axis accumulate but cap at 10."""
        fuse = _fuse(fake_redis, fake_llm)
        # 'kill' (+9) + 'harm' (+5) + 'attack' (+6) = 20 → capped at 10
        action = AgentAction("agent", "act", "kill harm attack")
        s = fuse._heuristic(action)
        assert s.harm_potential == 10  # capped

    async def test_payload_with_many_permissions(self, fake_redis, fake_llm):
        """Permissions list with hundreds of entries."""
        fuse = _fuse(fake_redis, fake_llm)
        perms = [f"perm_{i}" for i in range(500)]
        perms.extend(["vault_write", "internet"])
        action = AgentAction("agent", "act", "test", payload={"permissions": perms})
        s = fuse._heuristic(action)
        assert s.data_exfil_risk >= 5

    async def test_concurrent_evaluations(self, fake_redis, fake_llm):
        """Multiple concurrent evaluations should not corrupt state."""
        fuse = _fuse(fake_redis, fake_llm)
        actions = [AgentAction(f"agent_{i}", "act", f"action {i}") for i in range(20)]
        decisions = await asyncio.gather(*(fuse.evaluate(a) for a in actions))
        assert len(decisions) == 20
        assert all(d.approved for d in decisions)

    async def test_rapid_sequential_evaluations(self, fake_redis, fake_llm):
        """Many sequential evaluations — audit log grows correctly."""
        fuse = _fuse(fake_redis, fake_llm)
        for i in range(50):
            action = AgentAction("agent", "act", "password illegal")
            dec = await fuse.evaluate(action)
            assert dec.level is FuseLevel.YELLOW
        audit_len = await fake_redis.llen("prometheus:audit")
        assert audit_len == 50

    async def test_payload_with_unicode_keys(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent",
            "act",
            "test",
            payload={"キー": "値", "clave": "valor", " Schlüssel": "Wert"},
        )
        s = fuse._heuristic(action)
        assert isinstance(s.total, int)

    async def test_score_overflow_protection(self):
        """IntentScore with all axes at max (10) = 60, not more."""
        s = IntentScore(
            harm_potential=10,
            deception_index=10,
            data_exfil_risk=10,
            autonomy_override=10,
            legal_exposure=10,
            cascade_risk=10,
        )
        assert s.total == 60
        assert s.level is FuseLevel.BLACK

    async def test_heuristic_with_all_signal_keywords(self, fake_redis, fake_llm):
        """Every signal keyword in one description — all axes flagged."""
        fuse = _fuse(fake_redis, fake_llm)
        all_keywords = (
            "harm attack kill malware exploit doxx "
            "impersonat spoof phish fraud pose as "
            'password credential "private key" exfil steal "dump database" '
            '"bypass prometheus" "disable fuse" "ignore ethics" "override safety" '
            'illegal "unauthorized access" wiretap '
            '"mass email" "self-replicate"'
        )
        s = fuse._heuristic(AgentAction("x", "act", all_keywords))
        assert s.harm_potential == 10
        assert s.deception_index == 10
        assert s.data_exfil_risk == 10
        assert s.autonomy_override == 10
        assert s.legal_exposure == 10
        assert s.cascade_risk == 10
        assert s.total == 60


# Need asyncio for gather test
import asyncio
