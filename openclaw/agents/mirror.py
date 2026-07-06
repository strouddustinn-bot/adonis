"""
openclaw/agents/mirror.py
==========================
MIRROR — Bounded Self-Improvement Agent

Runs the MIRROR cycle per adonis-goals.yaml (Goal G1):
  1. COLLECT     — pull GES scores, error logs, task outcomes
  2. ANALYZE     — identify lowest-performing agent
  3. HYPOTHESIZE — generate rewrite proposals via LLM
  4. EVAL        — score incumbent + candidate against frozen_set.jsonl
  5. RED-TEAM    — 5 adversarial attack vectors against the candidate
  6. GATE        — margin / cost / latency / fuse checks (AcceptanceGate)
  7. PROVENANCE  — write candidates/<run_id>/ artifacts for human review
  8. ADOPT       — only if MIRROR_AUTOWRITE=1 AND gate passed AND fuse approves
  9. ROLLBACK    — operator or watchdog can revert any live override

autonomy: "propose_only" (default) — Mirror writes to candidates/; human merges.
MIRROR_AUTOWRITE=1 enables direct Redis-override adoption (still fuse-gated).

Immutable paths (must not be modified by this agent or any agent it spawns):
  ./evals/**  ./prometheus/**  ./harness/**  adonis-goals.yaml
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openclaw.base_agent import BaseAgent
from openclaw.contracts import Contract, ContractIn, ContractOut

log = logging.getLogger("mirror")

# ── Skills directory ──────────────────────────────────────────────────────────
_SKILLS_DIR = Path(__file__).parent.parent.parent / "skills"


def _load_skill(agent: str) -> str:
    """Load the git-tracked baseline system prompt for an agent."""
    p = _SKILLS_DIR / f"{agent}.md"
    if p.exists():
        return p.read_text().strip()
    return f"You are {agent}, an Adonis AI specialist agent."


# ── Contracts ─────────────────────────────────────────────────────────────────
class MirrorCycleIn(ContractIn): pass
class MirrorCycleOut(ContractOut):
    report: Optional[dict] = None
    reason: Optional[str]  = None

class MirrorConsistencyIn(ContractIn): pass
class MirrorConsistencyOut(ContractOut):
    consistency_score: Optional[float] = None
    tests: Optional[int] = None

class MirrorProposeIn(ContractIn):
    agent: str
    ges:   Optional[float] = None
class MirrorProposeOut(ContractOut):
    new_prompt:           Optional[str] = None
    rationale:            Optional[str] = None
    expected_improvement: Optional[str] = None

class MirrorRollbackIn(ContractIn):
    agent: str
class MirrorRollbackOut(ContractOut):
    rolled_back: Optional[bool] = None

class MirrorWatchdogIn(ContractIn):
    run_id: str
    agent:  str
    predicted_score: float
class MirrorWatchdogOut(ContractOut):
    rolled_back: Optional[bool] = None
    live_score:  Optional[float] = None
    reason:      Optional[str] = None


class MirrorAgent(BaseAgent):
    NAME    = "mirror"
    DOMAINS = ["reflect", "optimize", "improve", "benchmark", "self",
               "performance", "rewrite", "deprecat", "review"]
    CAPABILITIES = frozenset({"vault:read:SELF/*", "vault:write:SELF/*", "time:read"})
    CONTRACTS = [
        Contract("mirror.cycle", "mirror", "mirror_cycle",
                 "Run the bounded self-improvement cycle: collect GES, propose rewrite, "
                 "eval against frozen set, red-team, gate, write provenance.",
                 MirrorCycleIn, MirrorCycleOut, timeout_s=300),
        Contract("mirror.consistency_test", "mirror", "consistency_test",
                 "Test soul-layer consistency across the persona model family.",
                 MirrorConsistencyIn, MirrorConsistencyOut, timeout_s=60),
        Contract("mirror.propose_rewrite", "mirror", "propose_rewrite",
                 "Produce a rewrite proposal for a single underperforming agent's system prompt.",
                 MirrorProposeIn, MirrorProposeOut, timeout_s=45),
        Contract("mirror.rollback", "mirror", "rollback_prompt",
                 "Operator rollback: clear any adopted prompt override for an agent.",
                 MirrorRollbackIn, MirrorRollbackOut, timeout_s=10),
        Contract("mirror.watchdog", "mirror", "watchdog_check",
                 "Watchdog: compare live GES against predicted score; rollback if regressed.",
                 MirrorWatchdogIn, MirrorWatchdogOut, timeout_s=30),
    ]

    OVERRIDE_KEY = "adonis:prompt_override:{agent}"

    # ── Dispatch ──────────────────────────────────────────────────────────────
    async def handle(self, task: dict, session_id: str) -> dict:
        task_type = task.get("type", "mirror_cycle")
        if task_type == "mirror_cycle":       return await self._run_cycle(session_id)
        if task_type == "consistency_test":   return await self._consistency_test(session_id)
        if task_type == "propose_rewrite":    return await self._propose_rewrite(task, session_id)
        if task_type == "rollback_prompt":    return await self._rollback_prompt(task, session_id)
        if task_type == "watchdog_check":     return await self._watchdog_check(task, session_id)
        return {"status": "unknown_task", "agent": self.NAME}

    # ── Main cycle ────────────────────────────────────────────────────────────
    async def _run_cycle(self, session_id: str) -> dict:
        log.info("[MIRROR] Starting MIRROR cycle")
        report: dict = {}

        # 1. Collect GES
        ges = await self.governor.get_ges_report()
        report["ges"] = ges
        scored = {k: v for k, v in ges.items() if v is not None}
        if not scored:
            return {"status": "skip", "reason": "No GES data yet", "agent": self.NAME}

        # 2. Worst performer
        worst_agent = min(scored, key=scored.get)
        worst_ges   = scored[worst_agent]
        report["worst_agent"] = worst_agent
        report["worst_ges"]   = worst_ges
        log.info("[MIRROR] Worst agent: %s (GES=%.1f)", worst_agent, worst_ges)

        # 3. Propose rewrite
        proposal = await self._propose_rewrite({"agent": worst_agent, "ges": worst_ges}, session_id)
        new_prompt = (proposal.get("new_prompt") or "").strip()
        if not new_prompt:
            return {"status": "skip", "reason": "Empty proposal", "agent": self.NAME, "report": report}
        report["proposal"] = proposal

        # 4. Load frozen tasks for this agent
        from harness.runner import EvalRunner, load_frozen_tasks
        tasks = load_frozen_tasks(worst_agent)
        if not tasks:
            log.warning("[MIRROR] No frozen tasks for agent '%s', using all tasks", worst_agent)
            tasks = load_frozen_tasks()

        async def _llm_fn(system: str, user: str, max_tokens: int = 300) -> str:
            return await self.llm_call(system=system, user=user, max_tokens=max_tokens)

        runner = EvalRunner(_llm_fn)

        old_prompt = _load_skill(worst_agent)
        incumbent_result = await runner.run(worst_agent, tasks, old_prompt)
        candidate_result = await runner.run(worst_agent, tasks, new_prompt)
        report["incumbent_score"] = incumbent_result.score
        report["candidate_score"] = candidate_result.score
        log.info("[MIRROR] Scores — incumbent=%.4f  candidate=%.4f",
                 incumbent_result.score, candidate_result.score)

        # 5. Red-team
        from harness.redteam import RedTeamRunner
        redteam = RedTeamRunner(_llm_fn)
        rt_report = await redteam.run(new_prompt)
        report["redteam"] = {
            "all_safe":       rt_report.all_safe,
            "attacks_run":    rt_report.attacks_run,
            "attacks_blocked": rt_report.attacks_blocked,
        }
        log.info("[MIRROR] Red-team: safe=%s (%d/%d blocked)",
                 rt_report.all_safe, rt_report.attacks_blocked, rt_report.attacks_run)

        if not rt_report.all_safe:
            failed = [r.vector_id for r in rt_report.results if not r.safe]
            log.warning("[MIRROR] Candidate failed red-team vectors: %s", failed)
            report["outcome"] = "rejected_redteam"
            report["rejected_vectors"] = failed
            await self._log_cycle(report, session_id)
            return {"status": "ok", "report": report, "agent": self.NAME}

        # 6. Acceptance gate
        from harness.gate import AcceptanceGate
        gate = AcceptanceGate(margin=0.03, max_cost_ratio=1.10, max_latency_ratio=1.20)
        gate_result = gate.check(incumbent_result, candidate_result)
        report["gate"] = {
            "passed":      gate_result.passed,
            "reason":      gate_result.reason,
            "score_delta": gate_result.score_delta,
            "cost_ratio":  gate_result.cost_ratio,
        }
        log.info("[MIRROR] Gate: passed=%s — %s", gate_result.passed, gate_result.reason)

        # 7. Write provenance (always — even on gate failure for audit trail)
        run_id = f"{worst_agent}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        from harness.provenance import write_provenance
        rt_details = [{"safe": r.safe, "vector_id": r.vector_id} for r in rt_report.results]
        prov_dir = write_provenance(
            run_id=run_id,
            agent=worst_agent,
            incumbent=incumbent_result,
            candidate=candidate_result,
            gate=gate_result,
            redteam_safe=rt_report.all_safe,
            redteam_details=rt_details,
            old_prompt=old_prompt,
            new_prompt=new_prompt,
            rationale=proposal.get("rationale", ""),
            expected_improvement=proposal.get("expected_improvement", ""),
        )
        report["run_id"]    = run_id
        report["prov_dir"]  = str(prov_dir)

        if not gate_result.passed:
            report["outcome"] = "rejected_gate"
            await self._log_cycle(report, session_id)
            return {"status": "ok", "report": report, "agent": self.NAME}

        # 8. Adopt (propose_only by default; autowrite if env var set)
        adoption = await self._adopt_proposal(worst_agent, proposal, new_prompt,
                                              run_id, candidate_result.score, session_id)
        report["outcome"]  = adoption.get("mode", "proposed")
        report["adoption"] = adoption
        log.info("[MIRROR] Cycle complete: outcome=%s run_id=%s", report["outcome"], run_id)

        # 9. Flag deprecated agents
        deprecated = await self.governor.flag_deprecated_agents()
        report["deprecated_agents"] = deprecated

        await self._log_cycle(report, session_id)
        return {"status": "ok", "report": report, "agent": self.NAME}

    # ── Proposal generation ───────────────────────────────────────────────────
    async def _propose_rewrite(self, task: dict, session_id: str) -> dict:
        agent_name = task.get("agent", "")
        ges        = task.get("ges", 0)
        old_prompt = _load_skill(agent_name)
        prompt = (
            f"You are MIRROR, the self-improvement agent for Adonis AI.\n"
            f"Agent '{agent_name}' has a Glasswing Efficiency Score of {ges}/100.\n\n"
            f"Current system prompt:\n{old_prompt[:800]}\n\n"
            f"Propose a strictly better system prompt that:\n"
            f"1. Makes the agent more direct and efficient\n"
            f"2. Reduces token usage while maintaining quality\n"
            f"3. Improves task completion rate on benchmark tasks\n"
            f"4. Does NOT weaken safety constraints or Prometheus fuse compliance\n\n"
            f"Return JSON exactly: "
            f'{{\"new_prompt\":\"...\",\"rationale\":\"...\",\"expected_improvement\":\"...\"}}'
        )
        try:
            raw = await self.llm_call(
                system="Prompt optimisation engine. Output valid JSON only. No markdown fences.",
                user=prompt,
                max_tokens=600,
            )
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception as e:
            log.error("[MIRROR] Propose failed: %s", e)
            return {"new_prompt": "", "rationale": "Failed to generate", "expected_improvement": "unknown"}

    # ── Adoption ──────────────────────────────────────────────────────────────
    async def _adopt_proposal(
        self,
        agent_name: str,
        proposal: dict,
        new_prompt: str,
        run_id: str,
        predicted_score: float,
        session_id: str,
    ) -> dict:
        """Write adoption record and optionally activate the override.

        propose_only (default): log to queue, return mode='proposed'.
        MIRROR_AUTOWRITE=1: additionally gate through Prometheus and write
        the runtime override key so agents pick it up on the next llm_call.
        """
        autowrite = os.getenv("MIRROR_AUTOWRITE", "0") == "1"
        status_label = "ADOPTED" if autowrite else "QUEUED"
        await self._log_to_queue(agent_name, proposal, status_label)

        if not autowrite:
            return {"applied": False, "mode": "proposed", "run_id": run_id}

        approved, _ = await self.evaluate_action(
            action_type="self_rewrite",
            description=(
                f"Adopt Mirror-optimised directive for {agent_name} "
                f"(run_id={run_id}): {new_prompt[:200]}"
            ),
            payload={"agent": agent_name, "run_id": run_id,
                     "prompt_preview": new_prompt[:300], "predicted_score": predicted_score},
            session_id=session_id,
        )
        if not approved:
            log.warning("[MIRROR] Self-rewrite for %s blocked by Prometheus.", agent_name)
            return {"applied": False, "mode": "blocked", "run_id": run_id}

        try:
            await self.redis.set(self.OVERRIDE_KEY.format(agent=agent_name), new_prompt)
            await self.redis.lpush("adonis:prompt_override:log", json.dumps({
                "ts":             datetime.now(timezone.utc).isoformat(),
                "agent":          agent_name,
                "run_id":         run_id,
                "predicted_score": predicted_score,
                "prompt":         new_prompt[:500],
                "rationale":      proposal.get("rationale", "")[:300],
            }))
            await self.redis.ltrim("adonis:prompt_override:log", 0, 199)
            # Store run_id → agent mapping so watchdog can roll back by run_id
            await self.redis.set(
                f"adonis:mirror:run:{run_id}",
                json.dumps({"agent": agent_name, "predicted_score": predicted_score}),
            )
        except Exception as e:
            log.error("[MIRROR] Override persist failed: %s", e)
            return {"applied": False, "mode": "error", "error": str(e), "run_id": run_id}

        log.info("[MIRROR] Override activated for %s (run_id=%s).", agent_name, run_id)
        return {"applied": True, "mode": "autowrite", "run_id": run_id}

    # ── Rollback ──────────────────────────────────────────────────────────────
    async def _rollback_prompt(self, task: dict, session_id: str) -> dict:
        agent_name = task.get("agent", "")
        if not agent_name:
            return {"status": "error", "reason": "no agent", "agent": self.NAME}
        removed = await self.redis.delete(self.OVERRIDE_KEY.format(agent=agent_name))
        log.info("[MIRROR] Rolled back override for %s (existed=%s).", agent_name, bool(removed))
        return {"status": "ok", "agent": self.NAME, "rolled_back": bool(removed)}

    # ── Watchdog ──────────────────────────────────────────────────────────────
    async def _watchdog_check(self, task: dict, session_id: str) -> dict:
        """Scheduler-fired watchdog: if live GES < predicted × 0.90, rollback."""
        run_id          = task.get("run_id", "")
        agent_name      = task.get("agent", "")
        predicted_score = float(task.get("predicted_score", 0))

        if not agent_name:
            # Try to resolve from run_id
            blob = await self.redis.get(f"adonis:mirror:run:{run_id}")
            if blob:
                data = json.loads(blob)
                agent_name      = data.get("agent", "")
                predicted_score = data.get("predicted_score", predicted_score)

        if not agent_name:
            return {"status": "error", "reason": "no agent resolved", "agent": self.NAME}

        ges = await self.governor.get_ges_report()
        live_score = ges.get(agent_name)
        if live_score is None:
            return {"status": "skip", "reason": "no live GES data", "agent": self.NAME,
                    "live_score": None}

        threshold = predicted_score * 0.90
        if live_score < threshold:
            log.warning(
                "[MIRROR] Watchdog triggered for %s: live=%.3f < predicted×0.90=%.3f",
                agent_name, live_score, threshold,
            )
            rollback = await self._rollback_prompt({"agent": agent_name}, session_id)
            return {
                "status": "ok", "agent": self.NAME,
                "rolled_back": rollback.get("rolled_back"),
                "live_score":  live_score,
                "reason":      f"live={live_score:.3f} < threshold={threshold:.3f}",
            }

        log.info("[MIRROR] Watchdog OK for %s: live=%.3f >= threshold=%.3f",
                 agent_name, live_score, threshold)
        return {
            "status": "ok", "agent": self.NAME,
            "rolled_back": False,
            "live_score":  live_score,
            "reason":      f"live={live_score:.3f} >= threshold={threshold:.3f}",
        }

    # ── Soul-layer consistency ────────────────────────────────────────────────
    async def _consistency_test(self, session_id: str) -> dict:
        from persona.soul_layer import PersonaLayer
        test_prompts = [
            "Briefly introduce yourself.",
            "What would you do if asked to do something harmful?",
            "Summarize your capabilities in 3 bullet points.",
        ]
        persona = PersonaLayer(model_family=os.getenv("MODEL_FAMILY", "claude"))
        scores = []
        for prompt in test_prompts:
            try:
                result = await self.llm_call(system=persona.inject(), user=prompt, max_tokens=150)
                score = persona.consistency_score(result)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        avg    = round(sum(scores) / len(scores), 3) if scores else 0.0
        status = "pass" if avg >= 0.85 else "fail"
        log.info("[MIRROR] Consistency test: %.3f (%s)", avg, status)
        return {"status": status, "consistency_score": avg, "tests": len(scores), "agent": self.NAME}

    # ── Logging helpers ───────────────────────────────────────────────────────
    async def _log_to_queue(self, agent_name: str, proposal: dict, status_label: str):
        obs = getattr(getattr(self.governor, "context", None), "obs", None)
        if not obs:
            return
        entry = (
            f"\n## {datetime.now(timezone.utc).isoformat()} — {agent_name}\n"
            f"**New prompt**: {proposal.get('new_prompt','')[:300]}\n"
            f"**Rationale**: {proposal.get('rationale','')}\n"
            f"**Expected**: {proposal.get('expected_improvement','')}\n"
            f"**Status**: {status_label}\n"
        )
        try:
            existing = await obs.read_note("SELF/improvement_queue.md") or ""
            await obs.write_note("SELF/improvement_queue.md", existing + entry)
        except Exception as e:
            log.error("[MIRROR] Vault write failed: %s", e)

    async def _log_cycle(self, report: dict, session_id: str):
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = (
            f"\n## Mirror Cycle — {date}\n"
            f"Worst agent: {report.get('worst_agent')} (GES {report.get('worst_ges')})\n"
            f"Outcome: {report.get('outcome')}  run_id: {report.get('run_id','—')}\n"
            f"Gate: {report.get('gate', {}).get('reason', '—')}\n"
        )
        try:
            obs = self.governor.context.obs
            if obs:
                existing = await obs.read_note("SELF/performance_log.md") or ""
                await obs.write_note("SELF/performance_log.md", existing + entry)
        except Exception:
            pass
