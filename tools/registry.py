"""
tools/registry.py
==================
Tool registry — uniform interface for every external capability.

Every tool call passes two policy gates before execution:
  1) Capability check (deterministic) — does the calling agent hold the
     capabilities this tool declares as required?
  2) Prometheus fuse (intent-scored) — is this specific action ethically
     permissible given the action's description and payload?

Capability strings are documented in `tools/capabilities.py`.

Built-in tools register themselves at import time. External MCP servers
plug in at startup via `tools.mcp_client.attach_mcp_servers()`.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Any, Iterable

from tools.capabilities import covers

log = logging.getLogger("tools")


@dataclass
class Tool:
    name:                  str
    description:           str
    schema:                dict
    call:                  Callable[[dict], Awaitable[Any]]
    origin:                str = "builtin"   # "builtin" | "mcp:<server>"
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    metadata:              dict = field(default_factory=dict)


class ToolRegistry:
    """Global, in-process tool catalog. Singleton per supervisord process.

    Maintains an audit log in Redis (`tools:audit`, capped at 1000 entries)
    of every capability check — allow or deny — with context (agent, tool,
    args preview, missing caps if any, prometheus level)."""

    AUDIT_LIST = "tools:audit"
    AUDIT_CAP  = 1000

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._redis = None  # set by attach_redis() at boot

    def attach_redis(self, redis_client) -> None:
        """Wire a Redis client so audit entries persist beyond this process."""
        self._redis = redis_client

    def register(self, tool: Tool, *, overwrite: bool = False) -> None:
        if tool.name in self._tools and not overwrite:
            log.warning("[TOOLS] Skipping duplicate registration: %s", tool.name)
            return
        self._tools[tool.name] = tool
        log.info("[TOOLS] Registered %s (%s)", tool.name, tool.origin)

    async def _audit(self, **fields) -> None:
        if not self._redis: return
        rec = {"ts": datetime.now(timezone.utc).isoformat(), **fields}
        try:
            await self._redis.lpush(self.AUDIT_LIST, json.dumps(rec))
            await self._redis.ltrim(self.AUDIT_LIST, 0, self.AUDIT_CAP - 1)
        except Exception as e:
            log.debug("[TOOLS] audit write failed: %s", e)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def describe(self) -> list[dict]:
        """Anthropic-style tool descriptors for prompting/tool-use."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.schema,
             "required_capabilities": sorted(t.required_capabilities)}
            for t in self._tools.values()
        ]

    def capability_matrix(self, agent_capabilities: dict[str, Iterable[str]]) -> dict:
        """For UI / introspection: which tools each agent is allowed to call."""
        out: dict = {}
        for agent, caps in agent_capabilities.items():
            allowed, blocked = [], []
            for t in self._tools.values():
                ok, _ = covers(caps, t.required_capabilities)
                (allowed if ok else blocked).append(t.name)
            out[agent] = {"capabilities": sorted(caps), "allowed": sorted(allowed), "blocked": sorted(blocked)}
        return out

    async def invoke(
        self,
        name:                str,
        args:                dict,
        *,
        agent_name:          str,
        agent_capabilities:  Iterable[str] = (),
        session_id:          str,
        fuse,
    ) -> dict:
        """
        Run a tool call, gated by capability tokens AND the Prometheus fuse.
        Returns {ok:bool, result:..., error?:str, blocked?:bool, ...}.
        """
        tool = self._tools.get(name)
        if not tool:
            await self._audit(decision="unknown_tool", agent=agent_name, tool=name)
            return {"ok": False, "error": f"unknown tool: {name}"}

        # 1) Structural gate: capability tokens.
        ok, missing = covers(agent_capabilities, tool.required_capabilities)
        if not ok:
            log.warning("[TOOLS] capability denial: %s -> %s (missing %s)",
                        agent_name, name, missing)
            await self._audit(
                decision="capability_denied", agent=agent_name, tool=name,
                required=sorted(tool.required_capabilities), missing=missing,
                args_preview=str(args)[:200], session_id=session_id,
            )
            return {
                "ok":      False,
                "blocked": True,
                "reason":  f"capability denied; missing {missing}",
                "required": sorted(tool.required_capabilities),
                "missing":  missing,
            }

        # 2) Intent gate: Prometheus.
        from prometheus.fuse import AgentAction
        action = AgentAction(
            agent_name=agent_name,
            action_type=f"tool:{name}",
            description=f"{agent_name} calls tool {name} with args {str(args)[:200]}",
            payload={"tool": name, "args": args, "origin": tool.origin,
                     "required_capabilities": sorted(tool.required_capabilities)},
            session_id=session_id,
        )
        decision = await fuse.evaluate(action)
        if not decision.approved:
            await self._audit(
                decision="fuse_blocked", agent=agent_name, tool=name,
                level=decision.level.value, reason=decision.reason,
                args_preview=str(args)[:200], session_id=session_id,
            )
            return {
                "ok":      False,
                "blocked": True,
                "level":   decision.level.value,
                "reason":  decision.reason,
            }

        safe_args = decision.remediated_action.payload.get("args", args) if decision.remediated_action else args

        t0 = time.monotonic()
        try:
            result = await tool.call(safe_args)
            elapsed = int((time.monotonic()-t0)*1000)
            await self._audit(
                decision="ok", agent=agent_name, tool=name,
                level=decision.level.value, elapsed_ms=elapsed,
                args_preview=str(args)[:200], session_id=session_id,
            )
            return {"ok": True, "result": result, "elapsed_ms": elapsed}
        except Exception as e:
            log.warning("[TOOLS] %s failed: %s", name, e)
            elapsed = int((time.monotonic()-t0)*1000)
            await self._audit(
                decision="exception", agent=agent_name, tool=name,
                error=str(e)[:200], elapsed_ms=elapsed, session_id=session_id,
            )
            return {"ok": False, "error": str(e), "elapsed_ms": elapsed}


# Module-level singleton.
REGISTRY = ToolRegistry()
