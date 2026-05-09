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
import os, json, logging, asyncio, uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

log = logging.getLogger("base_agent")

class BaseAgent(ABC):
    """
    Subclass example:
        class MyAgent(BaseAgent):
            NAME = "myagent"
            DOMAINS = ["mytask", "something"]

            async def handle(self, task: dict, session_id: str) -> dict:
                # do work, call self.evaluate_action() before any risky op
                return {"result": "done", "data": {...}}
    """
    NAME    = "base"
    DOMAINS = []

    def __init__(self, anthropic_client, redis_client, fuse, governor):
        self.llm      = anthropic_client
        self.redis    = redis_client
        self.fuse     = fuse
        self.governor = governor
        self.channel  = f"adonis:agent:{self.NAME}"
        self._alive   = True
        self._check_lock()

    def _check_lock(self):
        from prometheus.fuse import PrometheusFuse
        if PrometheusFuse.is_locked(self.redis, self.NAME):
            raise RuntimeError(f"[{self.NAME.upper()}] Agent is locked by Prometheus. Operator release required.")

    async def run(self):
        """Main loop — subscribe to Redis channel and dispatch tasks."""
        log.info(f"[{self.NAME.upper()}] Starting. Listening on {self.channel}")
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.channel)
        async for message in pubsub.listen():
            if not self._alive: break
            if message["type"] != "message": continue
            try:
                task = json.loads(message["data"])
                session_id = task.get("session_id", "unknown")
                trace_id   = task.get("trace_id", str(uuid.uuid4())[:8])
                log.debug(f"[{self.NAME.upper()}] Task received: {task.get('type','?')} session={session_id}")
                result = await self.handle(task, session_id)
                await self.emit(result, session_id, trace_id)
            except Exception as e:
                log.error(f"[{self.NAME.upper()}] Error: {e}")
                await self.emit({"error": str(e), "agent": self.NAME}, task.get("session_id",""), "")

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
            model=os.getenv("ADONIS_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=max_tokens,
            system=system_with_soul,
            messages=[{"role":"user","content":user}]
        )
        return r.content[0].text

    def stop(self):
        self._alive = False
