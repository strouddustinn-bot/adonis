"""
glasswing/governor.py
======================
Glasswing Efficiency Governor — orchestrates all 5 efficiency layers per LLM call.

Per-call pipeline (in order):
  1. MoE Router       — select minimum active agents + think depth
  2. Infinite Context — build compressed context within token budget
  3. Soul Layer       — inject personality spec
  4. Think Mode       — wrap prompt with CoT depth marker
  5. Cache Check      — hit Redis before any LLM call
  6. GES Scoring      — track and report efficiency per agent/session

Glasswing rule: if an answer exists in Redis/Obsidian in <200ms, the LLM is never called.
"""
import os, json, logging, time, asyncio
from dataclasses import dataclass, field

log = logging.getLogger("glasswing")

@dataclass
class PreparedCall:
    system:        str
    messages:      list[dict]
    active_agents: list[str]
    think_depth:   str
    cache_hit:     bool = False
    cached_result: str  = ""
    efficiency:    dict = field(default_factory=dict)

class GlasswingGovernor:
    """
    Usage:
        gov = GlasswingGovernor(llm, redis, chroma, obsidian, model_family="claude")
        prep = await gov.prepare(session_id, user_message, task_system_prompt)
        if prep.cache_hit:
            return prep.cached_result
        result = await llm_call(prep.system, prep.messages)
        gov.record_ges(session_id, prep.active_agents, tokens_used, tasks_done, cache_hit=False)
    """

    CACHE_TTL      = 3600       # 1h response cache
    GES_WINDOW     = 100        # sessions to average for GES
    DEPRECATE_DAYS = 30         # flag agents unused this long

    def __init__(self, anthropic_client, redis_client, chroma_client=None, obsidian_bridge=None, model_family: str = "claude"):
        from compression.quantum_compress import QuantumCompressor
        from persona.soul_layer import PersonaLayer
        from context.infinite_engine import InfiniteContextEngine
        from routing.moe_router import MoERouter

        self.llm      = anthropic_client
        self.redis    = redis_client
        self.persona  = PersonaLayer(model_family=model_family)
        self.qc       = QuantumCompressor(anthropic_client, redis_client)
        self.context  = InfiniteContextEngine(self.qc, redis_client, chroma_client, obsidian_bridge)
        self.router   = MoERouter()

    async def prepare(
        self,
        session_id:          str,
        user_message:        str,
        task_system_prompt:  str = "",
        token_budget:        int = 12000,
    ) -> PreparedCall:
        start = time.monotonic()

        # 1. Cache check — never call LLM for identical recent queries
        cache_key = f"glasswing:cache:{hash(user_message) % 10**10}"
        cached = await self.redis.get(cache_key)
        if cached:
            log.debug(f"[GW] Cache hit for session {session_id} ({time.monotonic()-start:.3f}s)")
            return PreparedCall(system="", messages=[], active_agents=[], think_depth="cache",
                                cache_hit=True, cached_result=cached.decode())

        # 2. MoE routing
        route = self.router.route(user_message)

        # 3. Build compressed context
        messages = await self.context.build_context(session_id, user_message, token_budget)

        # 4. Soul layer injection
        system = self.persona.inject(task_system_prompt)

        # 5. Apply think mode to the last user message
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = self.router.apply_think_mode(
                messages[-1]["content"], route.think_depth
            )

        elapsed = time.monotonic() - start
        log.info(f"[GW] Prepared call in {elapsed:.3f}s | agents={route.active_agents} | think={route.think_depth.value}")

        return PreparedCall(
            system=system, messages=messages,
            active_agents=route.active_agents,
            think_depth=route.think_depth.value,
            efficiency={**route.efficiency, "prep_ms": round(elapsed*1000)},
        )

    async def cache_result(self, user_message: str, result: str):
        """Store LLM result for future cache hits."""
        key = f"glasswing:cache:{hash(user_message) % 10**10}"
        await self.redis.setex(key, self.CACHE_TTL, result)

    def record_ges(self, session_id: str, agents: list[str], tokens_used: int, tasks_done: int, cache_hit: bool = False, speed_ms: int = 0):
        """
        Glasswing Efficiency Score per agent.
        GES = (tasks_done / tokens_used) * cache_hit_rate * speed_bonus
        Written synchronously — call after each response.
        """
        if not agents or tokens_used == 0: return
        base_ges = (tasks_done / tokens_used) * 1000  # normalise
        cache_bonus  = 1.5 if cache_hit else 1.0
        speed_bonus  = max(0.5, 1.0 - speed_ms/10000)
        ges = round(min(100, base_ges * cache_bonus * speed_bonus), 2)

        for agent in agents:
            key = f"glasswing:ges:{agent}"
            self.redis.lpush(key, ges)
            self.redis.ltrim(key, 0, self.GES_WINDOW - 1)
        log.debug(f"[GW] GES recorded: {ges} for {agents}")

    async def get_ges_report(self) -> dict[str, float]:
        """Return average GES for all agents. Used by MIRROR cycle."""
        from routing.moe_router import AGENT_REGISTRY
        report = {}
        for agent in AGENT_REGISTRY:
            raw = await self.redis.lrange(f"glasswing:ges:{agent.name}", 0, -1)
            if raw:
                scores = [float(v) for v in raw]
                report[agent.name] = round(sum(scores)/len(scores), 2)
            else:
                report[agent.name] = None  # Never used
        return report

    async def flag_deprecated_agents(self) -> list[str]:
        """Return agents with no GES data (unused for DEPRECATE_DAYS). MIRROR uses this."""
        report = await self.get_ges_report()
        return [name for name, score in report.items() if score is None]
