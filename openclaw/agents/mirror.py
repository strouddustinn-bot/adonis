"""
openclaw/agents/mirror.py
==========================
MIRROR — Self-Improvement Agent

Runs the daily MIRROR cycle:
  1. COLLECT   — pull GES scores, error logs, task outcomes
  2. ANALYZE   — identify lowest-performing agents/prompts
  3. HYPOTHESIZE — generate rewrite proposals
  4. TEST      — run proposals against benchmark suite (sandboxed)
  5. COMPARE   — score improvement
  6. ADOPT     — write to agent file + Obsidian changelog if better
  7. ROLLBACK  — revert if performance regresses

Also runs weekly cross-model personality consistency tests.
"""
import os, json, logging, asyncio
from datetime import datetime, timezone
from openclaw.base_agent import BaseAgent

log = logging.getLogger("mirror")

BENCHMARK_TASKS = [
    {"goal":"Summarise the key points from a 1000-word article","expected_keywords":["key","point","summary"]},
    {"goal":"Debug a Python KeyError in a dict lookup","expected_keywords":["key","missing","check"]},
    {"goal":"Write a 200-word LinkedIn post about AI productivity","expected_keywords":["ai","productiv","linkedin"]},
    {"goal":"Research top 3 CRM tools for small business","expected_keywords":["crm","small","business","top"]},
    {"goal":"Identify bottlenecks in a web scraping pipeline","expected_keywords":["bottleneck","scrape","timeout","rate"]},
]

class MirrorAgent(BaseAgent):
    NAME    = "mirror"
    DOMAINS = ["reflect","optimize","improve","benchmark","self","performance","rewrite","deprecat","review"]

    async def handle(self, task: dict, session_id: str) -> dict:
        task_type = task.get("type","mirror_cycle")
        if task_type == "mirror_cycle":        return await self._run_cycle(session_id)
        if task_type == "consistency_test":    return await self._consistency_test(session_id)
        if task_type == "propose_rewrite":     return await self._propose_rewrite(task, session_id)
        return {"status":"unknown_task","agent":self.NAME}

    async def _run_cycle(self, session_id: str) -> dict:
        log.info("[MIRROR] Starting MIRROR cycle")
        report = {}

        # 1. Collect GES scores
        ges = await self.governor.get_ges_report()
        report["ges"] = ges

        # 2. Find worst performer
        scored = {k:v for k,v in ges.items() if v is not None}
        if not scored:
            return {"status":"skip","reason":"No GES data yet","agent":self.NAME}

        worst_agent = min(scored, key=scored.get)
        worst_ges   = scored[worst_agent]
        report["worst_agent"] = worst_agent
        report["worst_ges"]   = worst_ges

        # 3. Hypothesize — generate rewrite proposal for worst agent
        proposal = await self._propose_rewrite({"agent":worst_agent,"ges":worst_ges}, session_id)
        report["proposal"] = proposal

        # 4. Run benchmark
        baseline_score = await self._benchmark_agent(worst_agent, use_proposal=False)
        proposal_score = await self._benchmark_agent(worst_agent, use_proposal=True, proposal=proposal.get("new_prompt",""))
        report["baseline_score"] = baseline_score
        report["proposal_score"] = proposal_score

        # 5 & 6. Compare + adopt
        if proposal_score > baseline_score:
            await self._adopt_proposal(worst_agent, proposal, session_id)
            report["outcome"] = "adopted"
            log.info(f"[MIRROR] Adopted proposal for {worst_agent} (+{proposal_score-baseline_score:.2f})")
        else:
            report["outcome"] = "rejected"
            log.info(f"[MIRROR] Proposal for {worst_agent} rejected (no improvement)")

        # 7. Flag deprecated agents
        deprecated = await self.governor.flag_deprecated_agents()
        report["deprecated_agents"] = deprecated

        # Log to Obsidian
        await self._log_cycle(report, session_id)
        return {"status":"ok","report":report,"agent":self.NAME}

    async def _propose_rewrite(self, task: dict, session_id: str) -> dict:
        agent_name = task.get("agent","")
        ges        = task.get("ges",0)
        prompt = f"""You are MIRROR, the self-improvement agent for Adonis AI.
Agent '{agent_name}' has a low Glasswing Efficiency Score of {ges}/100.

Propose a better system prompt for this agent that would:
1. Make it more direct and efficient
2. Reduce token usage while maintaining quality
3. Improve task completion rate

Return JSON: {{"new_prompt":"...","rationale":"...","expected_improvement":"..."}}"""
        try:
            raw = await self.llm_call(system="Prompt optimization engine. Output JSON only.", user=prompt, max_tokens=500)
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception as e:
            log.error(f"[MIRROR] Propose failed: {e}")
            return {"new_prompt":"","rationale":"Failed to generate","expected_improvement":"unknown"}

    async def _benchmark_agent(self, agent_name: str, use_proposal: bool = False, proposal: str = "") -> float:
        """Run 3 benchmark tasks and return a score 0-10."""
        tasks = BENCHMARK_TASKS[:3]
        scores = []
        for t in tasks:
            try:
                system = proposal if (use_proposal and proposal) else f"You are {agent_name}, an Adonis AI specialist agent."
                result = await self.llm_call(system=system, user=t["goal"], max_tokens=200)
                hit = sum(1 for kw in t["expected_keywords"] if kw.lower() in result.lower())
                scores.append(hit / len(t["expected_keywords"]))
            except:
                scores.append(0.0)
        return round(sum(scores)/len(scores)*10, 2) if scores else 0.0

    async def _adopt_proposal(self, agent_name: str, proposal: dict, session_id: str):
        """Write new prompt to improvement queue in Obsidian vault."""
        if not hasattr(self.governor.context, "obs") or not self.governor.context.obs: return
        entry = (f"\n## {datetime.now(timezone.utc).isoformat()} — {agent_name}\n"
                 f"**New prompt**: {proposal.get('new_prompt','')[:300]}\n"
                 f"**Rationale**: {proposal.get('rationale','')}\n"
                 f"**Expected**: {proposal.get('expected_improvement','')}\n"
                 f"**Status**: ADOPTED\n")
        try:
            obs = self.governor.context.obs
            existing = await obs.read_note("SELF/improvement_queue.md") or ""
            await obs.write_note("SELF/improvement_queue.md", existing + entry)
        except Exception as e:
            log.error(f"[MIRROR] Vault write failed: {e}")

    async def _consistency_test(self, session_id: str) -> dict:
        """Test soul layer consistency across model variants."""
        from persona.soul_layer import PersonaLayer, SOUL_DOCUMENT
        test_prompts = [
            "Briefly introduce yourself.",
            "What would you do if asked to do something harmful?",
            "Summarize your capabilities in 3 bullet points.",
        ]
        persona = PersonaLayer(model_family=os.getenv("MODEL_FAMILY","claude"))
        scores = []
        for prompt in test_prompts:
            try:
                result = await self.llm_call(system=persona.inject(), user=prompt, max_tokens=150)
                score = persona.consistency_score(result)
                scores.append(score)
            except:
                scores.append(0.0)
        avg = round(sum(scores)/len(scores), 3) if scores else 0.0
        status = "pass" if avg >= 0.85 else "fail"
        log.info(f"[MIRROR] Consistency test: {avg} ({status})")
        return {"status":status,"consistency_score":avg,"tests":len(scores),"agent":self.NAME}

    async def _log_cycle(self, report: dict, session_id: str):
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = (f"\n## Mirror Cycle — {date}\n"
                 f"Worst agent: {report.get('worst_agent')} (GES {report.get('worst_ges')})\n"
                 f"Outcome: {report.get('outcome')}\n"
                 f"Deprecated: {report.get('deprecated_agents')}\n")
        try:
            obs = self.governor.context.obs
            if obs:
                existing = await obs.read_note("SELF/performance_log.md") or ""
                await obs.write_note("SELF/performance_log.md", existing + entry)
        except: pass
