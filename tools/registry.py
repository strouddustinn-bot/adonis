"""
tools/registry.py
==================
Tool registry — uniform interface for every external capability.

Every tool call passes through Prometheus before execution. A tool is just
a name, a JSON-schema describing its arguments, and an async callable.

Built-in tools register themselves at import time. External MCP servers
plug in at startup via tools.mcp_client.attach_mcp_servers().
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Any

log = logging.getLogger("tools")


@dataclass
class Tool:
    name:        str
    description: str
    schema:      dict
    call:        Callable[[dict], Awaitable[Any]]
    origin:      str = "builtin"   # "builtin" | "mcp:<server>"
    metadata:    dict = field(default_factory=dict)


class ToolRegistry:
    """Global, in-process tool catalog. Singleton per supervisord process."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, overwrite: bool = False) -> None:
        if tool.name in self._tools and not overwrite:
            log.warning("[TOOLS] Skipping duplicate registration: %s", tool.name)
            return
        self._tools[tool.name] = tool
        log.info("[TOOLS] Registered %s (%s)", tool.name, tool.origin)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def describe(self) -> list[dict]:
        """Anthropic-style tool descriptors for prompting/tool-use."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.schema}
            for t in self._tools.values()
        ]

    async def invoke(
        self,
        name:        str,
        args:        dict,
        *,
        agent_name:  str,
        session_id:  str,
        fuse,
    ) -> dict:
        """
        Run a tool call, gated through Prometheus.
        Returns {ok:bool, result:..., error?:str, blocked?:bool, level?:str}.
        """
        tool = self._tools.get(name)
        if not tool:
            return {"ok": False, "error": f"unknown tool: {name}"}

        from prometheus.fuse import AgentAction
        action = AgentAction(
            agent_name=agent_name,
            action_type=f"tool:{name}",
            description=f"{agent_name} calls tool {name} with args {str(args)[:200]}",
            payload={"tool": name, "args": args, "origin": tool.origin},
            session_id=session_id,
        )
        decision = await fuse.evaluate(action)
        if not decision.approved:
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
            return {"ok": True, "result": result, "elapsed_ms": int((time.monotonic()-t0)*1000)}
        except Exception as e:
            log.warning("[TOOLS] %s failed: %s", name, e)
            return {"ok": False, "error": str(e), "elapsed_ms": int((time.monotonic()-t0)*1000)}


# Module-level singleton.
REGISTRY = ToolRegistry()
