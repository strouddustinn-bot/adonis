"""
tools/registry.py
==================
Tool registry — uniform interface for every external capability.

This module exposes TWO complementary registries:

  1. `ToolRegistry` (+ the module-level `REGISTRY` singleton) — the
     capability- and fuse-gated catalog that every agent actually calls
     through. Built-in tools register themselves at import time; external
     MCP servers plug in at startup via `tools.mcp_client`. This is the
     load-bearing path used by `base_agent`, `hermes/api`, `supervisord`.

  2. `DynamicToolRegistry` / `ToolSchema` / `ToolProxy` — a schema-driven
     registry with JSON-Schema validation and fallback chains, populated by
     `tools.discovery` from OpenAPI specs. Optional, additive layer.

Every call through `ToolRegistry.invoke` passes two policy gates before
execution:
  1) Capability check (deterministic) — does the calling agent hold the
     capabilities this tool declares as required?
  2) Prometheus fuse (intent-scored) — is this specific action ethically
     permissible given the action's description and payload?

Capability strings are documented in `tools/capabilities.py`.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from tools.capabilities import covers

log = logging.getLogger("tools")


# ─────────────────────────────────────────────────────────────────────────────
# Primary registry — capability + fuse gated. Everything calls through this.
# ─────────────────────────────────────────────────────────────────────────────
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
            caps = list(caps)  # materialize once; safe for generators / one-shot iterables
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


# ─────────────────────────────────────────────────────────────────────────────
# Secondary registry — schema-validated, fallback chains, OpenAPI discovery.
# Additive layer used by tools/discovery.py. Kept separate from the gated
# REGISTRY above so the two concerns don't collide.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ToolSchema:
    name: str
    endpoint: Optional[str] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)
    response_schema: Dict[str, Any] = field(default_factory=dict)
    fallback_chain: List[Dict[str, Any]] = field(default_factory=list)
    timeout: int = 30
    retry_limit: int = 3


class DynamicToolRegistry:
    """Schema-driven registry for tools discovered from OpenAPI specs.

    Distinct from `ToolRegistry`: this one does JSON-Schema validation and
    drives `ToolProxy`'s fallback chains. It does NOT perform capability or
    fuse gating — callers that need gating must go through `REGISTRY`.
    """
    def __init__(self, redis_client=None):
        self.tools: Dict[str, ToolSchema] = {}
        self.redis = redis_client

    def register_tool(self, schema: ToolSchema):
        """Register or update a tool definition."""
        self.tools[schema.name] = schema
        log.info(f"[REGISTRY] Tool '{schema.name}' registered/updated.")

    def get_tool(self, name: str) -> ToolSchema:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered in the Adonis Registry.")
        return self.tools[name]

    def validate_input(self, tool_name: str, data: Dict[str, Any]) -> bool:
        from jsonschema import ValidationError, validate
        tool = self.get_tool(tool_name)
        if not tool.input_schema:
            return True
        try:
            validate(instance=data, schema=tool.input_schema)
            return True
        except ValidationError as e:
            log.error(f"[REGISTRY] Input validation failed for '{tool_name}': {e.message}")
            return False

    def validate_output(self, tool_name: str, data: Dict[str, Any]) -> bool:
        from jsonschema import ValidationError, validate
        tool = self.get_tool(tool_name)
        if not tool.response_schema:
            return True
        try:
            validate(instance=data, schema=tool.response_schema)
            return True
        except ValidationError as e:
            log.error(f"[REGISTRY] Output validation failed for '{tool_name}': {e.message}")
            return False


class ToolProxy:
    """
    Execution wrapper that handles the Fallback Chain over a
    `DynamicToolRegistry`.
    Sequence: Primary -> Retry -> Fallback 1 -> Fallback 2 -> LLM Estimate.
    """
    def __init__(self, registry: DynamicToolRegistry, execution_map: Dict[str, Callable]):
        self.registry = registry
        self.execution_map = execution_map

    async def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.registry.get_tool(tool_name)

        # 1. Input Validation
        if not self.registry.validate_input(tool_name, args):
            return {"status": "error", "reason": "Invalid input schema", "agent": "registry"}

        # 2. Primary Attempt with Retries
        for attempt in range(tool.retry_limit + 1):
            try:
                result = await self._execute(tool_name, args)
                if self.registry.validate_output(tool_name, result):
                    return result
                log.warning(f"[PROXY] {tool_name} passed but failed response schema. Attempt {attempt+1}")
            except Exception as e:
                log.warning(f"[PROXY] {tool_name} failed: {e}. Attempt {attempt+1}/{tool.retry_limit+1}")
                if attempt == tool.retry_limit:
                    break
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # 3. Fallback Chain Execution
        for fallback in tool.fallback_chain:
            try:
                fallback_type = fallback.get("type")
                if fallback_type == "backup_api":
                    backup_name = fallback.get("name")
                    log.info(f"[PROXY] Switching to backup API: {backup_name}")
                    return await self.call(backup_name, args)

                elif fallback_type == "cached_data":
                    max_age = fallback.get("max_age", "1h")
                    log.info(f"[PROXY] Attempting cached retrieval (max_age={max_age})")
                    return {"status": "cached", "data": "...", "stale": True}

                elif fallback_type == "llm_estimate":
                    log.info(f"[PROXY] Triggering LLM heuristic estimate")
                    return {"status": "estimated", "data": "approximate value", "source": "llm_heuristic"}
            except Exception as e:
                log.error(f"[PROXY] Fallback {fallback} failed: {e}")

        return {"status": "critical_failure", "reason": "All fallback chains exhausted", "agent": "registry"}

    async def _execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Internal executor that maps tool name to the actual function."""
        fn = self.execution_map.get(tool_name)
        if not fn:
            raise NotImplementedError(f"No execution function mapped for tool '{tool_name}'")
        return await fn(args)
