"""
openclaw/agents/atlas.py
=========================
ATLAS — Orchestrator Agent (shared, always active)

Decomposes complex goals into sub-tasks, assigns them to specialist agents
via Redis pub/sub, collects results, and assembles the final response.

Uses asyncio.gather() for parallel independent sub-tasks (Glasswing rule #3).
"""
import os, json, logging, asyncio, uuid
from openclaw.base_agent import BaseAgent

log = logging.getLogger("atlas")

class AtlasAgent(BaseAgent):
    NAME    = "atlas"
    DOMAINS = ["orchestration","planning","task","manage","decompose","goal","coordinate","multi"]

    async def handle(self, task: dict, session_id: str) -> dict:
        goal = task.get("goal") or task.get("content","")
        if not goal:
            return {"status":"error","reason":"No goal provided","agent":self.NAME}

        # Decompose goal into sub-tasks
        subtasks = await self._decompose(goal, session_id)
        if not subtasks:
            return {"status":"error","reason":"Could not decompose goal","agent":self.NAME}

        log.info(f"[ATLAS] Decomposed into {len(subtasks)} subtasks for session {session_id}")

        # Dispatch independent tasks in parallel
        independent = [s for s in subtasks if not s.get("depends_on")]
        dependent   = [s for s in subtasks if s.get("depends_on")]

        results = {}
        if independent:
            dispatched = await asyncio.gather(*[self._dispatch(s, session_id) for s in independent])
            for s, r in zip(independent, dispatched):
                results[s["id"]] = r

        # Sequential for dependent tasks
        for s in dependent:
            dep_result = results.get(s["depends_on"], {})
            s["context"] = dep_result
            results[s["id"]] = await self._dispatch(s, session_id)

        # Synthesise final response
        synthesis = await self._synthesise(goal, results)
        return {"status":"ok","goal":goal,"subtasks":len(subtasks),"synthesis":synthesis,"agent":self.NAME}

    async def _decompose(self, goal: str, session_id: str) -> list[dict]:
        prompt = f"""Decompose this goal into 2-5 concrete subtasks for a multi-agent system.
Available agents: vector (web/SEO/leads), forge (content/writing), scout (research), smith (code/debug), sentinel (monitoring), mirror (optimization).
Return ONLY JSON array: [{{"id":"t1","agent":"agentname","task":"description","depends_on":null}}, ...]
Goal: {goal}"""
        try:
            raw = await self.llm_call(
                system="Task decomposition engine. Output valid JSON only.",
                user=prompt, max_tokens=600)
            # Strip markdown fences if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception as e:
            log.error(f"[ATLAS] Decompose failed: {e}")
            return [{"id":"t1","agent":"forge","task":goal,"depends_on":None}]

    async def _dispatch(self, subtask: dict, session_id: str) -> dict:
        """Publish sub-task to agent channel and wait for result."""
        agent   = subtask.get("agent","forge")
        channel = f"adonis:agent:{agent}"
        trace   = str(uuid.uuid4())[:8]
        result_key = f"atlas:result:{trace}"

        payload = {
            "type":       "task",
            "content":    subtask.get("task",""),
            "context":    subtask.get("context",{}),
            "session_id": session_id,
            "trace_id":   trace,
            "result_key": result_key,
        }
        await self.redis.publish(channel, json.dumps(payload))
        log.debug(f"[ATLAS] Dispatched to {agent} (trace {trace})")

        # Wait for result with timeout
        for _ in range(30):  # 30s timeout
            raw = await self.redis.get(result_key)
            if raw:
                await self.redis.delete(result_key)
                return json.loads(raw)
            await asyncio.sleep(1)
        return {"status":"timeout","agent":agent,"trace":trace}

    async def _synthesise(self, goal: str, results: dict) -> str:
        results_text = json.dumps({k: v.get("result", v) for k, v in results.items()}, indent=2)[:2000]
        return await self.llm_call(
            system="Synthesis engine. Combine agent results into a single coherent response. Be direct and complete.",
            user=f"Goal: {goal}\n\nAgent results:\n{results_text}", max_tokens=1000)
