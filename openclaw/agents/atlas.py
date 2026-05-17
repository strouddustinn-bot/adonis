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
from openclaw.contracts import Contract, ContractIn, ContractOut

log = logging.getLogger("atlas")


class AtlasOrchestrateIn(ContractIn):
    goal:    str = ""
    content: str = ""


class AtlasOrchestrateOut(ContractOut):
    goal:      str
    subtasks:  int
    synthesis: str


class AtlasAgent(BaseAgent):
    NAME    = "atlas"
    DOMAINS = ["orchestration","planning","task","manage","decompose","goal","coordinate","multi"]
    # Atlas dispatches; it doesn't directly hit external resources.
    CAPABILITIES = frozenset({"time:read"})
    CONTRACTS = [
        Contract(
            name="atlas.orchestrate", agent="atlas", task_type="task",
            description="Decompose a free-form goal into subtasks routed to specialist contracts; return synthesised result.",
            input_model=AtlasOrchestrateIn, output_model=AtlasOrchestrateOut,
            timeout_s=180,
        ),
    ]

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
        # Show Atlas the contract catalog so it picks a specific contract
        # per subtask. Schemas are omitted to keep the prompt short — the
        # contract name + description is enough for routing.
        catalog = []
        registry = getattr(self.governor, "contract_registry", None)
        if registry:
            catalog = registry.catalog_brief()
        cat_lines = "\n".join(f"  - {c['name']} (agent={c['agent']}, type={c['task_type']}): {c['description']}"
                              for c in catalog) or "  (no contracts registered)"

        prompt = f"""Decompose this goal into 2-5 concrete subtasks. For each subtask, pick
a CONTRACT from the catalog below — that determines which agent runs it and
what task_type to use.

Catalog:
{cat_lines}

Return ONLY a JSON array of objects:
  [{{"id":"t1","contract":"<contract.name>","task":"<short instruction>",
     "args":{{...input fields for that contract...}},
     "depends_on": null}}, ...]

Goal: {goal}"""
        try:
            raw = await self.llm_call(
                system="Task decomposition engine. Output valid JSON only.",
                user=prompt, max_tokens=900)
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception as e:
            log.error(f"[ATLAS] Decompose failed: {e}")
            return [{"id":"t1","contract":"forge.draft_post","task":goal,"args":{"content":goal},"depends_on":None}]

    async def _dispatch(self, subtask: dict, session_id: str) -> dict:
        """Dispatch a subtask under its contract, with retries + fallback.

        - Resolves the contract (subtask.contract or subtask.agent + type).
        - Runs up to contract.max_retries with exponential backoff.
        - On final exhaustion, fires the fallback_contract if defined.
        """
        registry = getattr(self.governor, "contract_registry", None)
        contract = None
        if registry:
            cname = subtask.get("contract")
            if cname and cname in registry.by_name:
                contract = registry.by_name[cname]
            elif subtask.get("agent") and subtask.get("type"):
                contract = registry.find_for_task(
                    {"type": subtask["type"]}, agent_name=subtask["agent"])

        agent     = (contract.agent     if contract else subtask.get("agent",  "forge"))
        task_type = (contract.task_type if contract else subtask.get("type",   "task"))
        max_tries = (contract.max_retries if contract else 1)
        timeout   = (contract.timeout_s   if contract else 30)
        backoff   = (contract.backoff_base_s if contract else 1.0)

        # Inputs: explicit `args` from decomposition, else fall back to the
        # legacy {content, context} shape.
        args = subtask.get("args")
        if not isinstance(args, dict):
            args = {"content": subtask.get("task",""), "context": subtask.get("context",{})}

        last = None
        for attempt in range(1, max_tries + 1):
            res = await self._send_once(agent, task_type, args, session_id, timeout, contract)
            last = res
            if self._is_success(res):
                if attempt > 1: res["_attempts"] = attempt
                return res
            if attempt < max_tries:
                wait = backoff * (2 ** (attempt - 1))
                log.info("[ATLAS] %s attempt %d/%d failed (%s) — retry in %.1fs",
                         (contract.name if contract else agent), attempt, max_tries,
                         res.get("status","?"), wait)
                await asyncio.sleep(wait)

        # Retries exhausted; try the fallback contract once if defined.
        if contract and contract.fallback_contract and registry:
            fb = registry.by_name.get(contract.fallback_contract)
            if fb:
                log.info("[ATLAS] %s exhausted; falling back to %s", contract.name, fb.name)
                fb_args = {**args, "_fell_back_from": contract.name}
                res = await self._send_once(fb.agent, fb.task_type, fb_args, session_id, fb.timeout_s, fb)
                res["_fallback_of"] = contract.name
                return res

        return last or {"status":"error","agent":agent,"reason":"dispatch failed (no attempts)"}

    async def _send_once(self, agent: str, task_type: str, args: dict,
                         session_id: str, timeout_s: int, contract) -> dict:
        trace      = str(uuid.uuid4())[:8]
        result_key = f"atlas:result:{trace}"
        payload = {
            "type":       task_type,
            "session_id": session_id,
            "trace_id":   trace,
            "result_key": result_key,
            **args,
        }
        if contract: payload["contract"] = contract.name
        await self.redis.publish(f"adonis:agent:{agent}", json.dumps(payload))
        log.debug(f"[ATLAS] Dispatched to {agent} ({task_type} / trace {trace})")

        # Add a small buffer so we don't time out before the agent's own SLA fires.
        deadline = asyncio.get_event_loop().time() + timeout_s + 5
        while asyncio.get_event_loop().time() < deadline:
            raw = await self.redis.get(result_key)
            if raw:
                await self.redis.delete(result_key)
                try:
                    return json.loads(raw)
                except Exception:
                    return {"status":"error","agent":agent,"reason":"malformed result"}
            await asyncio.sleep(0.3)
        return {"status":"timeout","agent":agent,"trace":trace}

    @staticmethod
    def _is_success(res: dict) -> bool:
        if not isinstance(res, dict): return False
        if "_contract_error" in res:  return False
        return res.get("status") == "ok"

    async def _synthesise(self, goal: str, results: dict) -> str:
        results_text = json.dumps({k: v.get("result", v) for k, v in results.items()}, indent=2)[:2000]
        return await self.llm_call(
            system="Synthesis engine. Combine agent results into a single coherent response. Be direct and complete.",
            user=f"Goal: {goal}\n\nAgent results:\n{results_text}", max_tokens=1000)
