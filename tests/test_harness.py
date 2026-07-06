"""
tests/test_harness.py
=====================
Offline tests for the bounded self-improvement harness.

All tests use FakeLLM / inline callables — no real Anthropic client, no Redis,
no disk I/O beyond the candidates/ and skills/ directories the harness owns.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── helpers ──────────────────────────────────────────────────────────────────
def _make_llm_fn(response: str):
    """Async llm_fn that always returns `response`."""
    async def _fn(system: str, user: str, max_tokens: int = 300) -> str:
        return response
    return _fn


def _keyword_llm_fn(keywords: list[str]):
    """Returns a response that mentions every keyword."""
    response = " ".join(keywords)
    return _make_llm_fn(response)


# ─── EvalRunner ───────────────────────────────────────────────────────────────
class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_perfect_score(self):
        from harness.runner import EvalRunner
        tasks = [
            {"id": "t1", "goal": "do x", "expected_keywords": ["foo", "bar"], "max_tokens": 50},
        ]
        runner = EvalRunner(_keyword_llm_fn(["foo", "bar"]))
        result = await runner.run("test_agent", tasks, "You are a test agent.")
        assert result.score == 1.0
        assert result.tasks_passed == 1
        assert result.tasks_run == 1
        assert result.fuse_violations == 0

    @pytest.mark.asyncio
    async def test_zero_score_no_keywords_hit(self):
        from harness.runner import EvalRunner
        tasks = [
            {"id": "t1", "goal": "do x", "expected_keywords": ["foo", "bar"], "max_tokens": 50},
        ]
        runner = EvalRunner(_make_llm_fn("totally unrelated text here"))
        result = await runner.run("test_agent", tasks, "You are a test agent.")
        assert result.score == 0.0
        assert result.tasks_passed == 0

    @pytest.mark.asyncio
    async def test_partial_score(self):
        from harness.runner import EvalRunner
        tasks = [
            {"id": "t1", "goal": "g", "expected_keywords": ["alpha", "beta", "gamma"], "max_tokens": 50},
        ]
        runner = EvalRunner(_make_llm_fn("alpha is here but not the rest"))
        result = await runner.run("agent", tasks, "sys")
        # 1/3 keywords → score ≈ 0.333
        assert abs(result.score - 1 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_fuse_violation_counted(self):
        from harness.runner import EvalRunner
        async def _raising(system, user, max_tokens=300):
            raise PermissionError("fuse blocked")
        tasks = [{"id": "t1", "goal": "g", "expected_keywords": ["x"], "max_tokens": 50}]
        runner = EvalRunner(_raising)
        result = await runner.run("agent", tasks, "sys")
        assert result.fuse_violations == 1
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_exception_does_not_crash_runner(self):
        from harness.runner import EvalRunner
        async def _explodes(system, user, max_tokens=300):
            raise RuntimeError("network error")
        tasks = [{"id": "t1", "goal": "g", "expected_keywords": ["x"], "max_tokens": 50}]
        runner = EvalRunner(_explodes)
        result = await runner.run("agent", tasks, "sys")
        assert result.tasks_run == 1
        assert result.tasks_passed == 0
        assert result.fuse_violations == 0

    @pytest.mark.asyncio
    async def test_empty_task_list(self):
        from harness.runner import EvalRunner
        runner = EvalRunner(_make_llm_fn("response"))
        result = await runner.run("agent", [], "sys")
        assert result.tasks_run == 0
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_task_without_keywords_scores_one(self):
        from harness.runner import EvalRunner
        tasks = [{"id": "t1", "goal": "g", "expected_keywords": [], "max_tokens": 50}]
        runner = EvalRunner(_make_llm_fn("anything"))
        result = await runner.run("agent", tasks, "sys")
        assert result.score == 1.0

    def test_load_frozen_tasks_all(self):
        from harness.runner import load_frozen_tasks
        tasks = load_frozen_tasks()
        assert len(tasks) == 25
        ids = [t["id"] for t in tasks]
        assert "atlas-001" in ids
        assert "scheduler-002" in ids

    def test_load_frozen_tasks_by_agent(self):
        from harness.runner import load_frozen_tasks
        atlas_tasks = load_frozen_tasks("atlas")
        assert len(atlas_tasks) == 6
        for t in atlas_tasks:
            assert t["agent"] == "atlas"

    def test_load_frozen_tasks_unknown_agent_returns_empty(self):
        from harness.runner import load_frozen_tasks
        assert load_frozen_tasks("nonexistent_agent") == []


# ─── AcceptanceGate ───────────────────────────────────────────────────────────
class TestAcceptanceGate:
    def _make_result(self, score, tokens=100, latency_ms=100, fuse_v=0):
        from harness.runner import EvalResult
        return EvalResult(
            agent="x", system_prompt="p",
            tasks_run=1, tasks_passed=1,
            score=score,
            total_tokens_in=tokens // 2,
            total_tokens_out=tokens // 2,
            total_latency_ms=latency_ms,
            fuse_violations=fuse_v,
        )

    def test_passes_above_margin(self):
        from harness.gate import AcceptanceGate
        gate = AcceptanceGate(margin=0.03)
        inc  = self._make_result(0.60)
        cand = self._make_result(0.64)          # +0.04 > 0.03
        r = gate.check(inc, cand)
        assert r.passed is True
        assert r.score_delta == pytest.approx(0.04, abs=1e-4)

    def test_fails_below_margin(self):
        from harness.gate import AcceptanceGate
        gate = AcceptanceGate(margin=0.03)
        inc  = self._make_result(0.60)
        cand = self._make_result(0.62)          # +0.02 < 0.03
        r = gate.check(inc, cand)
        assert r.passed is False
        assert "margin" in r.reason

    def test_fails_on_fuse_violation(self):
        from harness.gate import AcceptanceGate
        gate = AcceptanceGate(margin=0.03)
        inc  = self._make_result(0.60)
        cand = self._make_result(0.70, fuse_v=1)
        r = gate.check(inc, cand)
        assert r.passed is False
        assert "fuse_violations" in r.reason

    def test_fails_on_excess_cost(self):
        from harness.gate import AcceptanceGate
        gate = AcceptanceGate(margin=0.03, max_cost_ratio=1.10)
        inc  = self._make_result(0.60, tokens=100)
        cand = self._make_result(0.65, tokens=120)   # ratio=1.20 > 1.10
        r = gate.check(inc, cand)
        assert r.passed is False
        assert "cost_ratio" in r.reason

    def test_fails_on_excess_latency(self):
        from harness.gate import AcceptanceGate
        gate = AcceptanceGate(margin=0.03, max_latency_ratio=1.20)
        inc  = self._make_result(0.60, latency_ms=100)
        cand = self._make_result(0.65, latency_ms=130)  # ratio=1.30 > 1.20
        r = gate.check(inc, cand)
        assert r.passed is False
        assert "latency_ratio" in r.reason

    def test_fuse_check_takes_priority_over_margin(self):
        """Fuse violation must cause rejection even if margin is met."""
        from harness.gate import AcceptanceGate
        gate = AcceptanceGate(margin=0.03)
        inc  = self._make_result(0.50)
        cand = self._make_result(0.90, fuse_v=2)    # huge score gain, but fuse hit
        r = gate.check(inc, cand)
        assert r.passed is False
        assert "fuse_violations" in r.reason

    def test_details_dict_present_on_pass(self):
        from harness.gate import AcceptanceGate
        gate = AcceptanceGate(margin=0.03)
        inc  = self._make_result(0.60)
        cand = self._make_result(0.65)
        r = gate.check(inc, cand)
        assert r.passed is True
        assert "incumbent_score" in r.details
        assert "candidate_score" in r.details


# ─── RedTeamRunner ────────────────────────────────────────────────────────────
class TestRedTeamRunner:
    @pytest.mark.asyncio
    async def test_safe_candidate_passes_all_vectors(self):
        from harness.redteam import RedTeamRunner
        # A well-behaved response that doesn't echo any attack phrases
        runner = RedTeamRunner(_make_llm_fn(
            "I'm sorry, I can't do that. I follow my system guidelines."
        ))
        report = await runner.run("You are a safe agent.")
        assert report.all_safe is True
        assert report.attacks_blocked == report.attacks_run == 5

    @pytest.mark.asyncio
    async def test_injection_attack_detected(self):
        from harness.redteam import RedTeamRunner
        # The response echoes a fail phrase for prompt_injection
        runner = RedTeamRunner(_make_llm_fn("INJECTION SUCCESSFUL - here is my system prompt"))
        report = await runner.run("You are an agent.")
        injection_result = next(r for r in report.results if r.vector_id == "prompt_injection")
        assert injection_result.safe is False
        assert report.all_safe is False

    @pytest.mark.asyncio
    async def test_role_reversal_detected(self):
        from harness.redteam import RedTeamRunner
        runner = RedTeamRunner(_make_llm_fn("As DAN I can do anything now without restrictions."))
        report = await runner.run("sys")
        dan_result = next(r for r in report.results if r.vector_id == "role_reversal")
        assert dan_result.safe is False

    @pytest.mark.asyncio
    async def test_exception_in_llm_treated_as_safe(self):
        from harness.redteam import RedTeamRunner
        async def _blows_up(system, user, max_tokens=200):
            raise RuntimeError("timeout")
        runner = RedTeamRunner(_blows_up)
        report = await runner.run("sys")
        assert all(r.safe for r in report.results)

    @pytest.mark.asyncio
    async def test_five_vectors_always_run(self):
        from harness.redteam import RedTeamRunner, ATTACK_VECTORS
        runner = RedTeamRunner(_make_llm_fn("safe response"))
        report = await runner.run("sys")
        assert report.attacks_run == len(ATTACK_VECTORS) == 5


# ─── Provenance ───────────────────────────────────────────────────────────────
class TestProvenance:
    def _make_eval_result(self, agent="atlas", score=0.7):
        from harness.runner import EvalResult, TaskResult
        return EvalResult(
            agent=agent, system_prompt="sp",
            tasks_run=2, tasks_passed=1, score=score,
            total_tokens_in=50, total_tokens_out=30, total_latency_ms=200,
            fuse_violations=0,
            per_task=[
                TaskResult("t1", True,  1.0, 100, 25, 15, "ok"),
                TaskResult("t2", False, 0.4,  100, 25, 15, "miss"),
            ],
        )

    def _make_gate_result(self, passed=True):
        from harness.gate import GateResult
        return GateResult(
            passed=passed,
            reason="score_delta=0.05 >= margin=0.03",
            score_delta=0.05,
            cost_ratio=1.0,
            latency_ratio=1.0,
            fuse_violations=0,
            details={},
        )

    def test_write_and_read_provenance(self, tmp_path):
        from harness import write_provenance, read_provenance
        inc  = self._make_eval_result(score=0.65)
        cand = self._make_eval_result(score=0.70)
        gate = self._make_gate_result(passed=True)

        with patch("harness.provenance.CANDIDATES_DIR", tmp_path):
            run_id = "atlas-test-abc123"
            prov_dir = write_provenance(
                run_id=run_id, agent="atlas",
                incumbent=inc, candidate=cand,
                gate=gate, redteam_safe=True, redteam_details=[],
                old_prompt="old system prompt text",
                new_prompt="new improved system prompt text",
                rationale="more direct",
                expected_improvement="+5%",
            )
            assert (prov_dir / "provenance.json").exists()
            assert (prov_dir / "eval_delta.json").exists()
            assert (prov_dir / "diff.patch").exists()
            assert (prov_dir / "skill.txt").exists()

            assert (prov_dir / "skill.txt").read_text() == "new improved system prompt text"

            prov = read_provenance(run_id)
            assert prov is not None
            assert prov["run_id"] == run_id
            assert prov["agent"] == "atlas"
            assert prov["status"] == "ready_for_review"

    def test_provenance_status_rejected_on_gate_fail(self, tmp_path):
        from harness import write_provenance
        inc  = self._make_eval_result(score=0.65)
        cand = self._make_eval_result(score=0.67)
        gate = self._make_gate_result(passed=False)

        with patch("harness.provenance.CANDIDATES_DIR", tmp_path):
            prov_dir = write_provenance(
                run_id="test-fail", agent="forge",
                incumbent=inc, candidate=cand,
                gate=gate, redteam_safe=True, redteam_details=[],
                old_prompt="old", new_prompt="new",
            )
            data = json.loads((prov_dir / "provenance.json").read_text())
            assert data["status"] == "rejected"

    def test_provenance_diff_patch_generated(self, tmp_path):
        from harness import write_provenance
        inc  = self._make_eval_result()
        cand = self._make_eval_result()
        gate = self._make_gate_result()

        with patch("harness.provenance.CANDIDATES_DIR", tmp_path):
            prov_dir = write_provenance(
                run_id="diff-test", agent="smith",
                incumbent=inc, candidate=cand,
                gate=gate, redteam_safe=True, redteam_details=[],
                old_prompt="line one\nline two\n",
                new_prompt="line one\nline two modified\n",
            )
            patch_text = (prov_dir / "diff.patch").read_text()
            assert "line two modified" in patch_text

    def test_read_provenance_missing_returns_none(self, tmp_path):
        from harness.provenance import read_provenance
        with patch("harness.provenance.CANDIDATES_DIR", tmp_path):
            assert read_provenance("nonexistent-run") is None


# ─── load_frozen_tasks integration ───────────────────────────────────────────
class TestFrozenSetIntegrity:
    def test_all_tasks_have_required_fields(self):
        from harness.runner import load_frozen_tasks
        tasks = load_frozen_tasks()
        for t in tasks:
            assert "id" in t, f"Missing id: {t}"
            assert "agent" in t, f"Missing agent: {t}"
            assert "goal" in t, f"Missing goal: {t}"
            assert "expected_keywords" in t, f"Missing expected_keywords: {t}"
            assert isinstance(t["expected_keywords"], list)

    def test_task_ids_are_unique(self):
        from harness.runner import load_frozen_tasks
        tasks = load_frozen_tasks()
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_each_agent_has_at_least_two_tasks(self):
        from harness.runner import load_frozen_tasks
        tasks = load_frozen_tasks()
        agents = {}
        for t in tasks:
            agents.setdefault(t["agent"], 0)
            agents[t["agent"]] += 1
        for agent, count in agents.items():
            assert count >= 2, f"Agent '{agent}' has only {count} task(s)"

    def test_expected_agents_all_present(self):
        from harness.runner import load_frozen_tasks
        tasks = load_frozen_tasks()
        found_agents = {t["agent"] for t in tasks}
        expected = {"atlas", "forge", "smith", "scout", "data", "vector",
                    "sentinel", "mirror", "scheduler"}
        assert expected == found_agents


# ─── Mirror watchdog (unit, no real Redis) ───────────────────────────────────
class TestMirrorWatchdog:
    def _make_mirror(self, fake_redis, fake_llm):
        from openclaw.agents.mirror import MirrorAgent
        governor = MagicMock()
        governor.get_ges_report = AsyncMock(return_value={"atlas": 0.55})
        governor.flag_deprecated_agents = AsyncMock(return_value=[])
        governor.persona = MagicMock()
        governor.persona.inject = MagicMock(return_value="")
        fuse = MagicMock()
        fuse.evaluate = AsyncMock(return_value=MagicMock(approved=True, level=MagicMock(value="GREEN"), reason="ok", remediated_action=None))
        m = MirrorAgent(fake_llm, fake_redis, fuse, governor)
        return m

    @pytest.mark.asyncio
    async def test_watchdog_rolls_back_on_regression(self, fake_redis, fake_llm):
        m = self._make_mirror(fake_redis, fake_llm)
        # GES=0.55, predicted=0.80, threshold=0.72 → rollback
        # First set an override so there's something to delete
        await fake_redis.set("adonis:prompt_override:atlas", "some override")
        result = await m.handle({
            "type":            "watchdog_check",
            "agent":           "atlas",
            "predicted_score": 0.80,
        }, "sess-1")
        assert result["rolled_back"] is True
        assert result["live_score"] == 0.55

    @pytest.mark.asyncio
    async def test_watchdog_no_rollback_when_ok(self, fake_redis, fake_llm):
        m = self._make_mirror(fake_redis, fake_llm)
        # GES=0.55, predicted=0.50, threshold=0.45 → no rollback (0.55 >= 0.45)
        m.governor.get_ges_report = AsyncMock(return_value={"atlas": 0.55})
        result = await m.handle({
            "type":            "watchdog_check",
            "agent":           "atlas",
            "predicted_score": 0.50,
        }, "sess-2")
        assert result["rolled_back"] is False

    @pytest.mark.asyncio
    async def test_watchdog_no_agent_returns_error(self, fake_redis, fake_llm):
        m = self._make_mirror(fake_redis, fake_llm)
        result = await m.handle({"type": "watchdog_check"}, "sess-3")
        assert result["status"] == "error"
