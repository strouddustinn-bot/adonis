"""
openclaw/agents/sentinel.py
============================
SENTINEL — System Health specialist.

Probes every dependency Adonis relies on and reports a structured health
snapshot. Used by ops dashboards, by Mirror to detect runtime drift, and
on demand via `POST /task` with a "health" query.

Task shape:
  {type:"health"}   — full probe
  {type:"audit_summary", window?:int}  — fuse audit distribution over last N entries
"""
import asyncio
import json
import logging
import time

from openclaw.base_agent import BaseAgent

log = logging.getLogger("sentinel")


class SentinelAgent(BaseAgent):
    NAME    = "sentinel"
    DOMAINS = ["monitor", "health", "alert", "status", "uptime", "check"]
    # Probes internal services only — no outbound traffic, no vault writes.
    CAPABILITIES = frozenset({"time:read"})

    async def handle(self, task: dict, session_id: str) -> dict:
        task_type = task.get("type", "health")
        if task_type == "audit_summary":
            return await self._audit_summary(task)
        return await self._health()

    async def _health(self) -> dict:
        redis_probe, ges, audit_len, locked = await asyncio.gather(
            self._probe_redis(),
            self.governor.get_ges_report(),
            self._safe_llen("prometheus:audit"),
            self._locked_agents(),
        )

        chroma = self._probe_chroma()
        obsidian = await self._probe_obsidian()

        ges_fresh = {k: ("fresh" if v is not None else "stale") for k, v in ges.items()}

        return {
            "status":   "ok",
            "agent":    self.NAME,
            "redis":    redis_probe,
            "chroma":   chroma,
            "obsidian": obsidian,
            "prometheus": {
                "audit_entries": int(audit_len),
                "locked_agents": locked,
            },
            "ges":      ges,
            "ges_state": ges_fresh,
            "tools":    self._tool_inventory(),
        }

    async def _probe_redis(self) -> dict:
        t0 = time.monotonic()
        try:
            ok = await self.redis.ping()
            return {"ok": bool(ok), "latency_ms": int((time.monotonic()-t0)*1000)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _probe_chroma(self) -> dict:
        ctx = self.governor.context
        if not getattr(ctx, "chroma", None):
            return {"ok": False, "reason": "disabled"}
        try:
            # collection.count() is cheap and confirms reachability.
            return {"ok": True, "count": int(ctx.chroma.count())}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _probe_obsidian(self) -> dict:
        ctx = self.governor.context
        if not getattr(ctx, "obs", None):
            return {"ok": False, "reason": "disabled"}
        t0 = time.monotonic()
        try:
            await ctx.obs.search("adonis")
            return {"ok": True, "latency_ms": int((time.monotonic()-t0)*1000)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _safe_llen(self, key: str) -> int:
        try:
            return int(await self.redis.llen(key))
        except Exception:
            return -1

    async def _locked_agents(self) -> list[str]:
        try:
            keys = await self.redis.keys("prometheus:locked:*")
            return [k.decode().split(":")[-1] if isinstance(k, (bytes, bytearray)) else k.split(":")[-1] for k in keys]
        except Exception:
            return []

    def _tool_inventory(self) -> dict:
        names = self.tools.names()
        return {"count": len(names), "names": names[:50]}

    async def _audit_summary(self, task: dict) -> dict:
        window = max(10, min(int(task.get("window", 200)), 1000))
        raw = await self.redis.lrange("prometheus:audit", 0, window - 1)
        levels = {}
        agents = {}
        for r in raw:
            try:
                rec = json.loads(r)
            except Exception:
                continue
            levels[rec.get("level", "?")] = levels.get(rec.get("level", "?"), 0) + 1
            agents[rec.get("agent", "?")] = agents.get(rec.get("agent", "?"), 0) + 1
        return {
            "status":   "ok",
            "agent":    self.NAME,
            "window":   len(raw),
            "by_level": levels,
            "by_agent": agents,
        }
