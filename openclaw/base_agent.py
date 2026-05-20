"""
openclaw/base_agent.py
=======================
Abstract base class for all Adonis agents.
Every agent (built-in and community plugin) must extend this.

Lifecycle:
  1. __init__  — register with Redis, check Prometheus lock
  2. run()     — main async loop, subscribes to Redis channel
  3. handle()  — override in subclass to define agent behaviour
  4. emit()    — publish result to adonis:results channel

All actions must pass through self.evaluate_action() before execution.
"""
import asyncio
import json
import logging
from typing import Optional
from tools.registry import ToolRegistry, ToolProxy
from observability.tracer import get_tracer
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

log = logging.getLogger("base_agent")

class BaseAgent:
    def __init__(self, llm, tool_proxy=None):
        self.tracer = get_tracer()
        self.llm = llm
        self.tool_proxy = tool_proxy
    """
    Subclass example:
        class MyAgent(BaseAgent):
            NAME = "myagent"
            DOMAINS = ["mytask", "something"]
            CAPABILITIES = frozenset({"net:http_get", "vault:read"})

            async def handle(self, task: dict, session_id: str) -> dict:
                # do work, call self.evaluate_action() before any risky op
                return {"result": "done", "data": {...}}

    CAPABILITIES is the structural permission set for this agent's tool
    calls. The tool registry rejects any call to a tool whose required
    capabilities aren't covered. See tools/capabilities.py for the grammar.
    """
    NAME    = "base"
    DOMAINS = []
    CAPABILITIES: frozenset[str] = frozenset()

    def __init__(self, anthropic_client, redis_client, fuse, governor):
        self.llm      = anthropic_client
        self.redis    = redis_client
        self.fuse     = fuse
        self.governor = governor
        self.channel  = f"adonis:agent:{self.NAME}"
        self._alive   = True

    async def _check_lock(self):
        from prometheus.fuse import PrometheusFuse
        if await PrometheusFuse.is_locked(self.redis, self.NAME):
            raise RuntimeError(f"[{self.NAME.upper()}] Agent is locked by Prometheus. Operator release required.")

    async def run(self):
        """Main loop — subscribe to Redis channel and dispatch tasks.

        For every incoming task we:
          1. Find the matching contract (if any) on this agent.
          2. Validate the input against the contract's pydantic model.
             Non-strict contracts attach _contract_warning and proceed.
          3. Run handle() under asyncio.wait_for(contract.timeout_s).
          4. Validate the result against the output model and attach
             _contract_warning if it doesn't fit.
          5. Emit, record outcome, and write to result_key for the caller.
        """
        await self._check_lock()
        log.info(f"[{self.NAME.upper()}] Starting. Listening on {self.channel}")
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.channel)
        async for message in pubsub.listen():
            if not self._alive: break
            if message["type"] != "message": continue
            task = {}
            try:
                task = json.loads(message["data"])
                session_id = task.get("session_id", "unknown")
                trace_id   = task.get("trace_id", str(uuid.uuid4())[:8])
                result_key = task.get("result_key")
                log.debug(f"[{self.NAME.upper()}] Task received: {task.get('type','?')} session={session_id}")

                result = await self._run_under_contract(task, session_id)

                await self.emit(result, session_id, trace_id)
                await self._record_outcome(task, result, session_id, trace_id)
                if result_key:
                    # Atlas (and the hermes API) poll redis.get(result_key) for the response.
                    await self.redis.setex(result_key, 60, json.dumps(result))
            except Exception as e:
                log.error(f"[{self.NAME.upper()}] Error: {e}")
                err = {"status": "error", "error": str(e), "agent": self.NAME}
                await self.emit(err, task.get("session_id",""), "")
                rk = task.get("result_key")
                if rk:
                    await self.redis.setex(rk, 60, json.dumps(err))

    async def _run_under_contract(self, task: dict, session_id: str) -> dict:
        """Apply contract validation + SLA to a single handle() invocation."""
        registry = getattr(self.governor, "contract_registry", None)
        contract = registry.find_for_task(task, agent_name=self.NAME) if registry else None

        if contract is None:
            # No contract → legacy path, no enforcement.
            return await self.handle(task, session_id)

        # Input validation
        ok, err, normalised = contract.validate_input(task)
        warnings: list[str] = []
        if not ok:
            if contract.strict:
                return {"status": "error", "agent": self.NAME,
                        "reason": f"contract input rejected: {err}",
                        "contract": contract.name}
            warnings.append(f"input: {err}")

        # Run under SLA
        try:
            result = await asyncio.wait_for(self.handle(task, session_id),
                                            timeout=contract.timeout_s)
        except asyncio.TimeoutError:
            log.warning("[%s] contract %s timed out after %ds",
                        self.NAME.upper(), contract.name, contract.timeout_s)
            return {"status": "timeout", "agent": self.NAME,
                    "contract": contract.name, "timeout_s": contract.timeout_s}

        if not isinstance(result, dict):
            result = {"status": "error", "agent": self.NAME, "result": result}

        # Output validation (lenient by default)
        ok, err = contract.validate_output(result)
        if not ok:
            if contract.strict:
                result["status"] = "error"
                result["_contract_error"] = err
            else:
                warnings.append(f"output: {err}")

        result["_contract"] = contract.name
        if warnings:
            result["_contract_warning"] = "; ".join(warnings)
        return result

    @abstractmethod
    async def handle(self, task: dict, session_id: str) -> dict:
        """Override in each agent. Return result dict."""
        ...

    async def emit(self, result: dict, session_id: str, trace_id: str):
        """Publish result to the shared results channel."""
        payload = {
            "agent":      self.NAME,
            "session_id": session_id,
            "trace_id":   trace_id,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "result":     result,
        }
        await self.redis.publish("adonis:results", json.dumps(payload))

    async def evaluate_action(self, action_type: str, description: str, payload: dict = None, session_id: str = "") -> tuple[bool, dict]:
        """
        Gate any risky operation through Prometheus Fuse.
        Returns (approved, safe_payload_to_use)
        """
        from prometheus.fuse import AgentAction
        action = AgentAction(
            agent_name=self.NAME, action_type=action_type,
            description=description, payload=payload or {},
            session_id=session_id
        )
        decision = await self.fuse.evaluate(action)
        if decision.approved:
            safe_action = decision.remediated_action or action
            return True, safe_action.payload
        return False, {}

    async def llm_call(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """Convenience wrapper for direct LLM calls with soul layer injected."""
        system_with_soul = self.governor.persona.inject(system)
        r = await self.llm.messages.create(
            model=os.getenv("ADONIS_MODEL", "claude-sonnet-4-6"),
            max_tokens=max_tokens,
            system=system_with_soul,
            messages=[{"role":"user","content":user}]
        )
        return r.content[0].text

    @property
    def tools(self):
        from tools.registry import REGISTRY
        return REGISTRY

    async def use_tool(self, name: str, args: dict, session_id: str = "") -> dict:
        """Invoke a registered tool, gated by capability tokens + Prometheus."""
        return await self.tools.invoke(
            name, args,
            agent_name=self.NAME,
            agent_capabilities=self.CAPABILITIES,
            session_id=session_id,
            fuse=self.fuse,
        )

    async def _extract_and_upsert(self, fg, task: dict, result: dict, session_id: str, trace_id: str):
        """Background job: extract structured facts from a winning result
        and feed them through the FactGraph. Failures are swallowed."""
        try:
            from memory.extractor import extract_facts
            hint = (task.get("goal") or task.get("content") or task.get("task") or "")
            digest_parts = [f"Task: {hint}" if hint else ""]
            for k in ("synthesis", "draft", "summary", "answer"):
                v = (result or {}).get(k)
                if isinstance(v, str) and v:
                    digest_parts.append(f"{k}: {v[:1500]}")
            text = "\n".join(p for p in digest_parts if p)
            candidates = await extract_facts(text, self.llm)
            for c in candidates:
                await fg.add(
                    c.entity, c.attribute, c.value,
                    confidence=c.confidence,
                    source_agent=self.NAME,
                    session_id=session_id, trace_id=trace_id,
                )
        except Exception as e:
            log.debug(f"[{self.NAME.upper()}] fact extraction skipped: {e}")

    async def _record_outcome(self, task: dict, result: dict, session_id: str, trace_id: str):
        """Persist a win/loss record so the system can learn what works.

        Wins go to `adonis:wins:{agent}` (capped 500); losses to
        `adonis:losses:{agent}` (capped 200). Every 10th win is also
        distilled into the L3 semantic vault so Atlas can retrieve prior
        winning approaches when decomposing similar goals later.
        """
        try:
            status = (result or {}).get("status", "")
            is_win = status == "ok"
            is_loss = status in ("error", "blocked", "timeout", "failed")
            if not (is_win or is_loss):
                return

            hint = (task.get("goal") or task.get("content") or task.get("task") or "")[:240]
            record = {
                "ts":         datetime.now(timezone.utc).isoformat(),
                "agent":      self.NAME,
                "session_id": session_id,
                "trace_id":   trace_id,
                "task_type":  task.get("type", ""),
                "hint":       hint,
                "result_keys": sorted([k for k in (result or {}).keys() if k not in {"status", "agent"}])[:10],
            }
            key = f"adonis:{'wins' if is_win else 'losses'}:{self.NAME}"
            await self.redis.lpush(key, json.dumps(record))
            await self.redis.ltrim(key, 0, 499 if is_win else 199)

            if is_win:
                count = await self.redis.incr(f"adonis:wins:counter:{self.NAME}")
                if count % 10 == 0:
                    try:
                        ctx = self.governor.context
                        digest = f"AGENT {self.NAME} succeeded at: {hint}\nResult keys: {record['result_keys']}"
                        await ctx.distil_to_l3(session_id, digest,
                                               metadata={"agent": self.NAME, "kind": "win"})
                    except Exception as e:
                        log.debug(f"[{self.NAME.upper()}] win L3 distil skipped: {e}")

                # Fact extraction — fire-and-forget so it never slows responses.
                fg = getattr(self.governor, "fact_graph", None)
                if fg and hint:
                    asyncio.create_task(self._extract_and_upsert(fg, task, result, session_id, trace_id))
        except Exception as e:
            log.warning(f"[{self.NAME.upper()}] outcome record failed: {e}")

    def stop(self):
        self._alive = False
