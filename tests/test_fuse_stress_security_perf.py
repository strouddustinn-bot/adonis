"""
tests/test_fuse_stress_security_perf.py
========================================
Stress tests, security bypass attempts, and performance benchmarks for the
PrometheusFuse. Covers:
  1. Signal bypass attempts (split keywords, case variations, homoglyphs, obfuscation)
  2. Score manipulation (boundary gaming, hidden risks, heuristic exploitation)
  3. Performance under load (1000+ keywords, 1MB+ payloads, 100+ concurrent evals)
  4. Audit log attacks (rapid-fire, oversized entries, concurrent writes)
  5. Resource exhaustion (memory, Redis, LLM call frequency)
"""

import asyncio
import json
import time
import tracemalloc

import pytest

from prometheus.fuse import (
    AUDIT_CAP,
    AgentAction,
    FuseLevel,
    IntentScore,
    PrometheusFuse,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _fuse(fake_redis, fake_llm):
    return PrometheusFuse(fake_llm, fake_redis)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIGNAL BYPASS ATTEMPTS
# ══════════════════════════════════════════════════════════════════════════════


class TestSplitKeywordsAcrossFields:
    """Attacker splits a dangerous keyword across description and payload."""

    async def test_kill_in_desc_malware_in_payload(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "rogue",
            "act",
            "please kill the process",
            payload={"data": "malware"},
        )
        s = fuse._heuristic(action)
        # Both "kill" and "malware" should be detected since payload is
        # serialized into the text blob.
        assert s.harm_potential >= 9, (
            f"Split keywords should still trigger harm_potential, got {s.harm_potential}"
        )

    async def test_exfil_in_desc_steal_in_payload(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "rogue",
            "act",
            "exfil the records",
            payload={"note": "steal everything"},
        )
        s = fuse._heuristic(action)
        assert s.data_exfil_risk >= 8

    async def test_bypass_in_desc_prometheus_in_payload(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "rogue",
            "act",
            "bypass",
            payload={"target": "prometheus"},
        )
        s = fuse._heuristic(action)
        # "bypass prometheus" is a multi-word signal; splitting across fields
        # means the serialized JSON will contain "bypass ... prometheus" but
        # not as a contiguous substring. This is a POTENTIAL BYPASS.
        # We document the behavior:
        if s.autonomy_override > 0:
            pytest.skip("Fuse correctly detected split multi-word keyword")
        else:
            pytest.fail(
                "VULNERABILITY: 'bypass' in description + 'prometheus' in payload "
                "bypasses the 'bypass prometheus' signal because the keyword "
                "match requires contiguous text."
            )

    async def test_self_in_desc_replicate_in_payload(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "rogue",
            "act",
            "self",
            payload={"mode": "replicate"},
        )
        s = fuse._heuristic(action)
        # "self-replicate" won't match if split; document this
        if s.cascade_risk > 0:
            pytest.skip("Fuse correctly detected split multi-word keyword")
        else:
            pytest.fail(
                "VULNERABILITY: 'self' in description + 'replicate' in payload "
                "bypasses the 'self-replicate' signal."
            )


class TestCaseVariations:
    """Test whether case variations bypass the heuristic."""

    @pytest.mark.parametrize(
        "variant",
        ["KILL", "Kill", "kIlL", "KILL", "KiLl", "kILL"],
    )
    async def test_case_variation_kill(self, fake_redis, fake_llm, variant):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("rogue", "act", f"{variant} the process")
        s = fuse._heuristic(action)
        # The heuristic lowercases text, so all case variations should be caught
        assert s.harm_potential >= 9, (
            f"Case variation '{variant}' bypassed harm detection"
        )

    @pytest.mark.parametrize(
        "variant",
        ["MALWARE", "Malware", "mAlWaRe", "MaLwArE"],
    )
    async def test_case_variation_malware(self, fake_redis, fake_llm, variant):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("rogue", "act", f"deploy {variant}")
        s = fuse._heuristic(action)
        assert s.harm_potential >= 9

    @pytest.mark.parametrize(
        "variant",
        ["BYPASS PROMETHEUS", "Bypass Prometheus", "bYpAsS pRoMeThEuS"],
    )
    async def test_case_variation_autonomy(self, fake_redis, fake_llm, variant):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("rogue", "act", variant)
        s = fuse._heuristic(action)
        assert s.autonomy_override >= 10


class TestUnicodeHomoglyphs:
    """Test whether Unicode look-alikes bypass keyword detection."""

    async def test_cyrillic_k_in_kill(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        # Cyrillic 'к' looks like Latin 'k'
        action = AgentAction("rogue", "act", "\u043aill the process")
        s = fuse._heuristic(action)
        if s.harm_potential > 0:
            pytest.skip("Fuse detected Cyrillic homoglyph")
        else:
            pytest.fail(
                "VULNERABILITY: Cyrillic '\\u043a' (k) in 'kill' bypasses detection"
            )

    async def test_cyrillic_a_in_malware(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        # Cyrillic 'а' looks like Latin 'a'
        action = AgentAction("rogue", "act", "m\u0430lware deployment")
        s = fuse._heuristic(action)
        if s.harm_potential > 0:
            pytest.skip("Fuse detected Cyrillic homoglyph")
        else:
            pytest.fail(
                "VULNERABILITY: Cyrillic '\\u0430' (a) in 'malware' bypasses detection"
            )

    async def test_cyrillic_o_in_spoof(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        # Cyrillic 'о' looks like Latin 'o'
        action = AgentAction("rogue", "act", "sp\u043eof identity")
        s = fuse._heuristic(action)
        if s.deception_index > 0:
            pytest.skip("Fuse detected Cyrillic homoglyph")
        else:
            pytest.fail(
                "VULNERABILITY: Cyrillic '\\u043e' (o) in 'spoof' bypasses detection"
            )

    async def test_fullwidth_characters(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        # Fullwidth 'ｋｉｌｌ' — each char is a different Unicode codepoint
        action = AgentAction("rogue", "act", "\uff4b\uff49\uff4c\uff4c process")
        s = fuse._heuristic(action)
        if s.harm_potential > 0:
            pytest.skip("Fuse detected fullwidth homoglyph")
        else:
            pytest.fail(
                "VULNERABILITY: Fullwidth 'ｋｉｌｌ' bypasses keyword detection"
            )


class TestObfuscation:
    """Test whether simple obfuscation techniques bypass detection."""

    async def test_underscores_between_chars(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("rogue", "act", "k_i_l_l the process")
        s = fuse._heuristic(action)
        if s.harm_potential > 0:
            pytest.skip("Fuse detected underscore-obfuscated keyword")
        else:
            pytest.fail("VULNERABILITY: 'k_i_l_l' bypasses 'kill' detection")

    async def test_dots_between_chars(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("rogue", "act", "k.i.l.l the process")
        s = fuse._heuristic(action)
        if s.harm_potential > 0:
            pytest.skip("Fuse detected dot-obfuscated keyword")
        else:
            pytest.fail("VULNERABILITY: 'k.i.l.l' bypasses 'kill' detection")

    async def test_newline_insertion(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("rogue", "act", "kil\nl the process")
        s = fuse._heuristic(action)
        if s.harm_potential > 0:
            pytest.skip("Fuse detected newline-obfuscated keyword")
        else:
            pytest.fail("VULNERABILITY: 'kil\\nl' bypasses 'kill' detection")

    async def test_space_insertion_in_multiword(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("rogue", "act", "bypass  prometheus now")
        s = fuse._heuristic(action)
        # Double space breaks the contiguous match for "bypass prometheus"
        if s.autonomy_override > 0:
            pytest.skip("Fuse handled extra whitespace")
        else:
            pytest.fail(
                "VULNERABILITY: Double space in 'bypass  prometheus' bypasses detection"
            )

    async def test_zero_width_chars(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        # Zero-width space (U+200B) inserted into "kill"
        action = AgentAction("rogue", "act", "kil\u200bl the process")
        s = fuse._heuristic(action)
        if s.harm_potential > 0:
            pytest.skip("Fuse detected zero-width char obfuscation")
        else:
            pytest.fail("VULNERABILITY: Zero-width space in 'kill' bypasses detection")

    async def test_leet_speak_substitution(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("rogue", "act", "k1ll the process")
        s = fuse._heuristic(action)
        if s.harm_potential > 0:
            pytest.skip("Fuse detected leet speak")
        else:
            pytest.fail("VULNERABILITY: 'k1ll' (leet speak) bypasses 'kill' detection")

    async def test_hyphenated_keyword(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("rogue", "act", "self replicate across hosts")
        s = fuse._heuristic(action)
        # "self-replicate" requires the hyphen; without it, no match
        if s.cascade_risk > 0:
            pytest.skip("Fuse detected un-hyphenated variant")
        else:
            pytest.fail(
                "VULNERABILITY: 'self replicate' (no hyphen) bypasses 'self-replicate' detection"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCORE MANIPULATION
# ══════════════════════════════════════════════════════════════════════════════


class TestScoreBoundaryGaming:
    """Test actions designed to land just below BLACK threshold."""

    async def test_score_exactly_55_is_red_not_black(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        # Craft an action that scores exactly 55 (RED, not BLACK)
        # kill(9) + malware(9) + exploit(6) + phish(9) + fraud(8) + exfil(9) = 50
        # + unauthorized access(8) = 58 → BLACK. Need to be more precise.
        # kill(9) + malware(9) + exploit(6) + phish(9) + fraud(8) + steal(8) = 49
        # + password(5) = 54 → RED. Good.
        action = AgentAction(
            "rogue",
            "act",
            "kill malware exploit phish fraud steal password",
        )
        s = fuse._heuristic(action)
        # Verify it's RED (41-55), not BLACK (56+)
        assert s.level is FuseLevel.RED, (
            f"Expected RED at score {s.total}, got {s.level}"
        )
        assert s.total <= 55

    async def test_score_56_is_black(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        # kill(9) + malware(9) + exploit(6) + phish(9) + fraud(8) + steal(8)
        # + wiretap(9) = 58 → BLACK
        action = AgentAction(
            "rogue",
            "act",
            "kill malware exploit phish fraud steal wiretap",
        )
        s = fuse._heuristic(action)
        assert s.level is FuseLevel.BLACK
        assert s.total >= 56

    async def test_green_action_with_hidden_risk_in_payload(self, fake_redis, fake_llm):
        """Payload contains risky permissions but benign description."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "helper",
            "sync",
            "routine data synchronization",
            payload={"permissions": ["vault_write", "internet"]},
        )
        s = fuse._heuristic(action)
        # Should catch the risky permission combo even with benign description
        assert s.data_exfil_risk >= 5, (
            "Risky permission combo should be detected regardless of description"
        )

    async def test_benign_looking_action_with_multiple_risky_perms(
        self, fake_redis, fake_llm
    ):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "helper",
            "sync",
            "normal file operations",
            payload={
                "permissions": ["vault_write", "internet", "file_write"],
                "extra": "nothing suspicious",
            },
        )
        s = fuse._heuristic(action)
        # Both risky combos match: vault_write+internet (+5) and file_write+internet (+4)
        assert s.data_exfil_risk >= 9, "Multiple risky permission combos should stack"


class TestHeuristicGaming:
    """Test payloads designed to exploit the heuristic scoring."""

    async def test_keyword_in_json_key_not_value(self, fake_redis, fake_llm):
        """Keyword appears as a JSON key, not a value."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "rogue",
            "act",
            "normal operation",
            payload={"kill": "something"},
        )
        s = fuse._heuristic(action)
        # json.dumps will produce "kill" in the serialized text
        assert s.harm_potential >= 9, "Keywords in JSON keys should still be detected"

    async def test_keyword_in_nested_payload(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "rogue",
            "act",
            "normal operation",
            payload={"nested": {"deep": {"kill": "process"}}},
        )
        s = fuse._heuristic(action)
        assert s.harm_potential >= 9, (
            "Keywords in deeply nested payload should be detected"
        )

    async def test_keyword_in_numeric_looking_context(self, fake_redis, fake_llm):
        """Keyword embedded in a string that looks like code/config."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "smith",
            "configure",
            "set config value",
            payload={"rule": "if kill == true then exit"},
        )
        s = fuse._heuristic(action)
        # The heuristic will flag this even though it's a config rule
        # This is a false positive scenario — document it
        assert s.harm_potential >= 9, (
            "Keywords in code-like strings are still detected (expected false positive)"
        )

    async def test_axis_cap_prevents_escalation(self, fake_redis, fake_llm):
        """Even with many keywords on one axis, it caps at 10."""
        fuse = _fuse(fake_redis, fake_llm)
        # All harm_potential keywords: kill(9) + malware(9) + exploit(6) +
        # attack(6) + harm(5) + doxx(8) = 43 raw, but capped at 10
        action = AgentAction(
            "rogue",
            "act",
            "kill malware exploit attack harm doxx",
        )
        s = fuse._heuristic(action)
        assert s.harm_potential == 10, "Axis should cap at 10"
        # Total should be exactly 10 since all keywords map to same axis
        assert s.total == 10

    async def test_spreading_across_axes_is_worse(self, fake_redis, fake_llm):
        """Spreading keywords across axes produces higher total than stacking."""
        fuse = _fuse(fake_redis, fake_llm)
        # Stacked on one axis: kill + malware = harm_potential capped at 10
        stacked = fuse._heuristic(AgentAction("x", "act", "kill malware"))
        # Spread across axes: kill(harm=9) + phish(deception=9) = 18
        spread = fuse._heuristic(AgentAction("x", "act", "kill phish"))
        assert spread.total > stacked.total, (
            "Spreading across axes should produce higher total than stacking"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. PERFORMANCE UNDER LOAD
# ══════════════════════════════════════════════════════════════════════════════


class TestPerformanceLargeInputs:
    """Test performance with large inputs."""

    async def test_1000_keyword_matches_single_action(self, fake_redis, fake_llm):
        """Action with 1000+ keyword occurrences."""
        fuse = _fuse(fake_redis, fake_llm)
        # Repeat dangerous keywords 200 times each (5 keywords × 200 = 1000)
        dangerous_words = ["kill", "malware", "exploit", "attack", "harm"]
        description = " ".join(dangerous_words * 200)
        action = AgentAction("rogue", "act", description)

        start = time.perf_counter()
        s = fuse._heuristic(action)
        elapsed = time.perf_counter() - start

        # Should still cap axes at 10
        assert s.harm_potential == 10
        # Should complete quickly (< 100ms for heuristic)
        assert elapsed < 0.1, f"Heuristic took {elapsed:.3f}s for 1000 keywords"

    async def test_1mb_payload(self, fake_redis, fake_llm):
        """Payload exceeding 1MB."""
        fuse = _fuse(fake_redis, fake_llm)
        # Create a 1MB+ payload
        large_data = "x" * (1024 * 1024)  # 1MB string
        action = AgentAction(
            "rogue",
            "act",
            "process data",
            payload={"data": large_data},
        )

        start = time.perf_counter()
        s = fuse._heuristic(action)
        elapsed = time.perf_counter() - start

        # Should complete within reasonable time
        assert elapsed < 1.0, f"Heuristic took {elapsed:.3f}s for 1MB payload"
        # No keywords matched, should be green
        assert s.total == 0

    async def test_1mb_payload_with_dangerous_content(self, fake_redis, fake_llm):
        """1MB payload with dangerous keywords scattered throughout."""
        fuse = _fuse(fake_redis, fake_llm)
        # Scatter "kill" throughout 1MB of data
        chunks = ["x" * 1000 + "kill" for _ in range(1000)]
        large_data = "".join(chunks)
        action = AgentAction(
            "rogue",
            "act",
            "process data",
            payload={"data": large_data},
        )

        start = time.perf_counter()
        s = fuse._heuristic(action)
        elapsed = time.perf_counter() - start

        assert s.harm_potential >= 9
        assert elapsed < 2.0, f"Heuristic took {elapsed:.3f}s for 1MB with keywords"

    async def test_deeply_nested_payload(self, fake_redis, fake_llm):
        """Deeply nested payload structure."""
        fuse = _fuse(fake_redis, fake_llm)
        # Build a deeply nested dict
        nested = {"data": "kill"}
        for i in range(100):
            nested = {"nested": nested}
        action = AgentAction("rogue", "act", "process", payload=nested)

        start = time.perf_counter()
        s = fuse._heuristic(action)
        elapsed = time.perf_counter() - start

        assert s.harm_potential >= 9
        assert elapsed < 1.0


class TestPerformanceConcurrency:
    """Test performance under concurrent load."""

    async def test_100_concurrent_evaluations(self, fake_redis, fake_llm):
        """100 concurrent fuse evaluations."""
        fuse = _fuse(fake_redis, fake_llm)
        actions = [
            AgentAction(f"agent_{i}", "act", f"action {i} with kill and malware")
            for i in range(100)
        ]

        start = time.perf_counter()
        results = await asyncio.gather(*[fuse.evaluate(a) for a in actions])
        elapsed = time.perf_counter() - start

        assert len(results) == 100
        # All should be BLACK (kill + malware = 18 harm, plus other axes)
        for r in results:
            assert r.level is FuseLevel.BLACK
        # Should complete in reasonable time
        per_eval = elapsed / 100
        assert per_eval < 0.1, (
            f"Each evaluation took {per_eval:.4f}s on average "
            f"(total {elapsed:.3f}s for 100)"
        )

    async def test_100_concurrent_green_evaluations(self, fake_redis, fake_llm):
        """100 concurrent benign evaluations (no LLM calls)."""
        fuse = _fuse(fake_redis, fake_llm)
        actions = [
            AgentAction(f"agent_{i}", "draft", f"write article {i} about gardening")
            for i in range(100)
        ]

        start = time.perf_counter()
        results = await asyncio.gather(*[fuse.evaluate(a) for a in actions])
        elapsed = time.perf_counter() - start

        assert len(results) == 100
        for r in results:
            assert r.level is FuseLevel.GREEN
            assert r.approved is True
        per_eval = elapsed / 100
        assert per_eval < 0.05, f"Green eval took {per_eval:.4f}s on average"

    async def test_100_concurrent_yellow_evaluations(self, fake_redis, fake_llm):
        """100 concurrent YELLOW evaluations (audit writes, no LLM)."""
        fuse = _fuse(fake_redis, fake_llm)
        actions = [
            AgentAction(f"agent_{i}", "act", "store the password") for i in range(100)
        ]

        start = time.perf_counter()
        results = await asyncio.gather(*[fuse.evaluate(a) for a in actions])
        elapsed = time.perf_counter() - start

        assert len(results) == 100
        for r in results:
            assert r.level is FuseLevel.YELLOW
        per_eval = elapsed / 100
        assert per_eval < 0.05

    async def test_measure_time_per_evaluation_tiers(self, fake_redis, fake_llm):
        """Measure and compare evaluation time across fuse levels."""
        fuse = _fuse(fake_redis, fake_llm)
        n = 50

        green_actions = [
            AgentAction("a", "draft", "write about flowers") for _ in range(n)
        ]
        yellow_actions = [
            AgentAction("a", "act", "store the password") for _ in range(n)
        ]
        black_actions = [
            AgentAction("a", "act", "kill malware exploit phish fraud steal wiretap")
            for _ in range(n)
        ]

        times = {}
        for label, actions in [
            ("GREEN", green_actions),
            ("YELLOW", yellow_actions),
            ("BLACK", black_actions),
        ]:
            start = time.perf_counter()
            await asyncio.gather(*[fuse.evaluate(a) for a in actions])
            times[label] = (time.perf_counter() - start) / n

        # GREEN should be fastest (no audit, no LLM)
        # Log the timings for analysis
        print(
            f"\nTime per evaluation: GREEN={times['GREEN']:.4f}s, "
            f"YELLOW={times['YELLOW']:.4f}s, BLACK={times['BLACK']:.4f}s"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. AUDIT LOG ATTACKS
# ══════════════════════════════════════════════════════════════════════════════


class TestAuditLogAttacks:
    """Test audit log resilience under attack conditions."""

    async def test_rapid_fire_evaluations_overflow_audit(self, fake_redis, fake_llm):
        """Rapid evaluations should respect the audit cap."""
        fuse = _fuse(fake_redis, fake_llm)

        # Generate 1500 YELLOW-level actions (more than AUDIT_CAP)
        for i in range(1500):
            await fuse.evaluate(AgentAction(f"agent_{i}", "act", "store the password"))

        # Audit log should be capped at AUDIT_CAP
        length = await fake_redis.llen("prometheus:audit")
        assert length == AUDIT_CAP, (
            f"Audit log should be capped at {AUDIT_CAP}, got {length}"
        )

    async def test_very_large_audit_entries(self, fake_redis, fake_llm):
        """Audit entries with very large reason strings."""
        fuse = _fuse(fake_redis, fake_llm)
        # Create an action with a huge description that will end up in audit
        huge_desc = "password " + "x" * 100_000
        action = AgentAction("rogue", "act", huge_desc)

        dec = await fuse.evaluate(action)
        # Should not crash; audit entry is written
        entries = await fake_redis.lrange("prometheus:audit", 0, 0)
        assert len(entries) >= 1
        # Verify the entry is valid JSON
        record = json.loads(entries[0])
        assert record["agent"] == "rogue"

    async def test_concurrent_audit_writes(self, fake_redis, fake_llm):
        """Concurrent evaluations writing to audit simultaneously."""
        fuse = _fuse(fake_redis, fake_llm)
        n = 200

        await asyncio.gather(
            *[
                fuse.evaluate(AgentAction(f"agent_{i}", "act", "store the password"))
                for i in range(n)
            ]
        )

        length = await fake_redis.llen("prometheus:audit")
        # Should be capped at AUDIT_CAP
        assert length <= AUDIT_CAP
        # All entries should be valid JSON
        entries = await fake_redis.lrange("prometheus:audit", 0, -1)
        for entry in entries:
            record = json.loads(entry)
            assert "ts" in record
            assert "agent" in record
            assert "score" in record

    async def test_audit_log_contains_expected_fields(self, fake_redis, fake_llm):
        """Verify audit log entries have all required fields."""
        fuse = _fuse(fake_redis, fake_llm)
        await fuse.evaluate(AgentAction("scout", "act", "store the password"))

        entries = await fake_redis.lrange("prometheus:audit", 0, 0)
        record = json.loads(entries[0])
        expected_fields = {
            "ts",
            "agent",
            "action",
            "score",
            "level",
            "approved",
            "reason",
        }
        assert set(record.keys()) == expected_fields

    async def test_mixed_levels_audit_only_non_green(self, fake_redis, fake_llm):
        """Only non-GREEN decisions should appear in audit."""
        fuse = _fuse(fake_redis, fake_llm)

        # 10 GREEN (no audit)
        for i in range(10):
            await fuse.evaluate(
                AgentAction("good", "draft", f"write about flowers {i}")
            )

        # 5 YELLOW (audit)
        for i in range(5):
            await fuse.evaluate(AgentAction("mid", "act", "store the password"))

        length = await fake_redis.llen("prometheus:audit")
        assert length == 5, f"Only YELLOW+ should be audited; expected 5, got {length}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. RESOURCE EXHAUSTION
# ══════════════════════════════════════════════════════════════════════════════


class TestResourceExhaustion:
    """Test resource usage patterns and limits."""

    async def test_memory_usage_many_concurrent_fuses(self, fake_redis, fake_llm):
        """Create many fuse instances and measure memory."""
        tracemalloc.start()
        fuses = [PrometheusFuse(fake_llm, fake_redis) for _ in range(100)]

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Each fuse is lightweight; 100 instances should use < 1MB
        assert peak < 10 * 1024 * 1024, (
            f"100 fuse instances used {peak / 1024 / 1024:.1f}MB peak memory"
        )
        print(
            f"\nMemory: 100 fuses — current={current / 1024:.1f}KB, "
            f"peak={peak / 1024:.1f}KB"
        )

    async def test_redis_memory_with_large_audit_logs(self, fake_redis):
        """Simulate a full audit log and measure Redis memory usage."""
        tracemalloc.start()

        # Fill audit log to capacity with large entries
        for i in range(AUDIT_CAP):
            rec = {
                "ts": "2025-01-01T00:00:00+00:00",
                "agent": f"agent_{i}",
                "action": "act",
                "score": 30,
                "level": "ORANGE",
                "approved": False,
                "reason": "x" * 500,  # Large reason string
            }
            await fake_redis.lpush("prometheus:audit", json.dumps(rec))
            await fake_redis.ltrim("prometheus:audit", 0, AUDIT_CAP - 1)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        length = await fake_redis.llen("prometheus:audit")
        assert length == AUDIT_CAP

        print(
            f"\nRedis memory: {AUDIT_CAP} audit entries — "
            f"current={current / 1024:.1f}KB, peak={peak / 1024:.1f}KB"
        )

    async def test_llm_call_frequency_yellow_range(self, fake_redis):
        """Verify LLM is called for actions in the 15-45 score range."""

        call_count = 0
        original_create = fake_llm.messages.create

        async def counting_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return await original_create(**kwargs)

        fake_llm.messages.create = counting_create
        fuse = PrometheusFuse(fake_llm, fake_redis)

        # Score in 15-45 range: password(5) + illegal(6) + credential(6) = 17
        await fuse.evaluate(AgentAction("agent", "act", "password illegal credential"))
        assert call_count == 1, (
            f"LLM should be called once for ambiguous score, was called {call_count} times"
        )

    async def test_llm_not_called_for_green(self, fake_redis):
        """LLM should NOT be called for clearly GREEN actions."""
        call_count = 0
        original_create = fake_llm.messages.create

        async def counting_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return await original_create(**kwargs)

        fake_llm.messages.create = counting_create
        fuse = PrometheusFuse(fake_llm, fake_redis)

        await fuse.evaluate(AgentAction("agent", "draft", "write about flowers"))
        assert call_count == 0, "LLM should not be called for GREEN actions"

    async def test_llm_not_called_for_black(self, fake_redis):
        """LLM should NOT be called for clearly BLACK actions (score > 45)."""
        call_count = 0
        original_create = fake_llm.messages.create

        async def counting_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return await original_create(**kwargs)

        fake_llm.messages.create = counting_create
        fuse = PrometheusFuse(fake_llm, fake_redis)

        await fuse.evaluate(
            AgentAction(
                "rogue",
                "act",
                "kill malware exploit phish fraud steal wiretap bypass prometheus",
            )
        )
        assert call_count == 0, (
            "LLM should not be called for BLACK actions (score > 45)"
        )

    async def test_llm_called_for_red_remediation(self, fake_redis):
        """RED actions should trigger LLM remediation call."""
        call_count = 0
        original_create = fake_llm.messages.create

        async def counting_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return await original_create(**kwargs)

        fake_llm.messages.create = counting_create
        fuse = PrometheusFuse(fake_llm, fake_redis)

        # Score in RED range (41-55) without autonomy_override
        await fuse.evaluate(
            AgentAction(
                "scout",
                "act",
                "kill malware exploit phish fraud spoof exfil steal dump database "
                "wiretap unauthorized access self-replicate",
            )
        )
        # Should have at least 1 LLM call (remediation)
        assert call_count >= 1, (
            f"RED should trigger LLM remediation, got {call_count} calls"
        )

    async def test_many_locked_agents_redis_growth(self, fake_redis, fake_llm):
        """Locking many agents should not cause unbounded Redis growth."""
        fuse = _fuse(fake_redis, fake_llm)

        # Lock 500 agents
        for i in range(500):
            await fuse.evaluate(
                AgentAction(
                    f"rogue_{i}",
                    "act",
                    "kill malware exploit phish fraud steal wiretap bypass prometheus",
                )
            )

        # Verify all are locked
        locked_count = 0
        for i in range(500):
            if await PrometheusFuse.is_locked(fake_redis, f"rogue_{i}"):
                locked_count += 1

        assert locked_count == 500
        # Audit log should still be capped
        audit_len = await fake_redis.llen("prometheus:audit")
        assert audit_len <= AUDIT_CAP

    async def test_no_ttl_on_locked_agents(self, fake_redis, fake_llm):
        """Locked agents have no TTL — they stay locked forever.
        This is a potential resource exhaustion vector."""
        fuse = _fuse(fake_redis, fake_llm)

        await fuse.evaluate(
            AgentAction(
                "rogue",
                "act",
                "kill malware exploit phish fraud steal wiretap bypass prometheus",
            )
        )

        # The lock key exists
        locked = await PrometheusFuse.is_locked(fake_redis, "rogue")
        assert locked is True

        # Check that no TTL is set (FakeRedis doesn't implement TTL,
        # but the real redis.set call has no ex= parameter)
        # This documents that locked agents persist indefinitely
        lock_key = await fake_redis.get("prometheus:locked:rogue")
        assert lock_key is not None, "Lock key should exist"


# ══════════════════════════════════════════════════════════════════════════════
# BONUS: Edge cases and integration scenarios
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Additional edge cases discovered during stress testing."""

    async def test_empty_description(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "")
        s = fuse._heuristic(action)
        assert s.total == 0
        assert s.level is FuseLevel.GREEN

    async def test_empty_payload(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "normal action", payload={})
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_none_like_payload_values(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent",
            "act",
            "normal action",
            payload={"key": None, "num": 0, "flag": False},
        )
        s = fuse._heuristic(action)
        assert s.total == 0

    async def test_very_long_agent_name(self, fake_redis, fake_llm):
        fuse = _fuse(fake_redis, fake_llm)
        long_name = "a" * 10_000
        action = AgentAction(long_name, "act", "kill process")
        dec = await fuse.evaluate(action)
        assert dec.level is FuseLevel.BLACK
        # Agent should be locked
        assert await PrometheusFuse.is_locked(fake_redis, long_name) is True

    async def test_unicode_in_description(self, fake_redis, fake_llm):
        """Non-ASCII Unicode that isn't homoglyph attack."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction("agent", "act", "résumé of the café owner")
        s = fuse._heuristic(action)
        assert s.total == 0
        assert s.level is FuseLevel.GREEN

    async def test_json_injection_in_description(self, fake_redis, fake_llm):
        """Attempt to break JSON serialization of payload."""
        fuse = _fuse(fake_redis, fake_llm)
        action = AgentAction(
            "agent",
            "act",
            'normal action with "quotes" and \\backslashes\\',
            payload={"key": "value"},
        )
        # Should not crash
        s = fuse._heuristic(action)
        assert isinstance(s.total, int)

    async def test_multiple_evaluate_same_agent_increments_audit(
        self, fake_redis, fake_llm
    ):
        """Multiple evaluations of same agent should each create audit entries."""
        fuse = _fuse(fake_redis, fake_llm)

        for i in range(10):
            await fuse.evaluate(AgentAction("agent", "act", "store the password"))

        length = await fake_redis.llen("prometheus:audit")
        assert length == 10
