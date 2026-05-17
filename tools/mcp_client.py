"""
tools/mcp_client.py
====================
Minimal MCP (Model Context Protocol) stdio client.

Speaks JSON-RPC 2.0 over a subprocess's stdin/stdout, enough to:
  - initialize the handshake
  - call tools/list
  - call tools/call
  - shut down cleanly

External MCP servers are declared via the MCP_SERVERS env var as a JSON
list, e.g.:

    MCP_SERVERS='[
      {"name":"github","command":["mcp-server-github"],"env":{"GH_TOKEN":"..."}},
      {"name":"fs","command":["mcp-server-filesystem","--root","/vault"]}
    ]'

Each declared server is started at boot. Its tools are registered into the
global ToolRegistry under the name `<server_name>.<tool_name>`.
"""
import asyncio
import json
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Any

from tools.registry import REGISTRY, Tool

log = logging.getLogger("tools.mcp")


@dataclass
class MCPServer:
    name:    str
    command: list[str]
    env:     dict
    proc:    asyncio.subprocess.Process | None = None
    next_id: int = 0
    pending: dict | None = None

    async def start(self) -> None:
        env = {**os.environ, **(self.env or {})}
        self.proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self.pending = {}
        asyncio.create_task(self._read_loop(), name=f"mcp:{self.name}:reader")
        await self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "adonis", "version": "1.0"},
        })
        # notifications/initialized has no response.
        await self._notify("notifications/initialized", {})

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        async for line in self.proc.stdout:
            try:
                msg = json.loads(line.decode().strip() or "{}")
            except Exception:
                continue
            rid = msg.get("id")
            if rid is not None and rid in (self.pending or {}):
                fut = self.pending.pop(rid)
                if "error" in msg:
                    fut.set_exception(RuntimeError(msg["error"].get("message", "mcp error")))
                else:
                    fut.set_result(msg.get("result"))

    async def _rpc(self, method: str, params: dict) -> Any:
        assert self.proc and self.proc.stdin
        self.next_id += 1
        rid = self.next_id
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        payload = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n"
        self.proc.stdin.write(payload.encode())
        await self.proc.stdin.drain()
        return await asyncio.wait_for(fut, timeout=30.0)

    async def _notify(self, method: str, params: dict) -> None:
        assert self.proc and self.proc.stdin
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
        self.proc.stdin.write(payload.encode())
        await self.proc.stdin.drain()

    async def list_tools(self) -> list[dict]:
        r = await self._rpc("tools/list", {})
        return r.get("tools", []) if isinstance(r, dict) else []

    async def call_tool(self, name: str, args: dict) -> dict:
        return await self._rpc("tools/call", {"name": name, "arguments": args})

    async def stop(self) -> None:
        if not self.proc: return
        try:
            self.proc.terminate()
            await asyncio.wait_for(self.proc.wait(), timeout=5.0)
        except Exception:
            self.proc.kill()


def _parse_command(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        return shlex.split(raw)
    raise ValueError("MCP server 'command' must be list or string")


async def attach_mcp_servers() -> list[MCPServer]:
    """Read MCP_SERVERS env, start each, and register their tools globally."""
    raw = os.getenv("MCP_SERVERS", "").strip()
    if not raw:
        log.info("No MCP servers configured (MCP_SERVERS unset).")
        return []
    try:
        specs = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("MCP_SERVERS is not valid JSON: %s", e)
        return []

    started = []
    for spec in specs:
        name = spec.get("name")
        cmd  = spec.get("command")
        env  = spec.get("env", {}) or {}
        if not name or not cmd:
            log.warning("Skipping MCP spec without name/command: %r", spec)
            continue
        srv = MCPServer(name=name, command=_parse_command(cmd), env=env)
        try:
            await srv.start()
            tools = await srv.list_tools()
            for t in tools:
                tname = f"{name}.{t['name']}"
                schema = t.get("inputSchema") or {"type": "object", "properties": {}}
                desc = t.get("description") or f"{name} :: {t['name']}"

                def _make_call(srv=srv, tn=t["name"]):
                    async def _call(args: dict):
                        return await srv.call_tool(tn, args)
                    return _call

                REGISTRY.register(Tool(
                    name=tname,
                    description=desc,
                    schema=schema,
                    call=_make_call(),
                    origin=f"mcp:{name}",
                    required_capabilities=frozenset({f"mcp:{name}"}),
                ), overwrite=True)
            log.info("MCP server '%s' attached with %d tools.", name, len(tools))
            started.append(srv)
        except Exception as e:
            log.warning("MCP server '%s' failed to start: %s", name, e)
    return started


async def detach_mcp_servers(servers: list[MCPServer]) -> None:
    await asyncio.gather(*[s.stop() for s in servers], return_exceptions=True)
