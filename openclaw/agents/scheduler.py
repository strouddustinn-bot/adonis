"""
openclaw/agents/scheduler.py
=============================
SCHEDULER — recurring / deferred task dispatch.

Nothing in Adonis previously *drove* the "recurring" cycles the README talks
about (Mirror's daily self-improvement, Sentinel health sweeps). Scheduler
does: it persists interval jobs in Redis and ticks on a timer, publishing each
due job's payload to its target agent's pub/sub channel — exactly the shape
Atlas and the specialists already consume.

Task shapes (control plane, via pub/sub like every other agent):
  {type:"schedule_add",    name:"daily-mirror", every_s:86400,
                           target:"mirror", payload:{"type":"mirror_cycle"}}
  {type:"schedule_list"}
  {type:"schedule_remove", name:"daily-mirror"}

Each stored job: {every_s, target, payload, next_run, created}. On every tick,
jobs whose next_run has passed are published and rescheduled.
"""
import asyncio
import json
import logging
import os
import time
from typing import Optional

from openclaw.base_agent import BaseAgent
from openclaw.contracts import Contract, ContractIn, ContractOut

log = logging.getLogger("scheduler")


class SchedAddIn(ContractIn):
    name:    str
    every_s: int
    target:  Optional[str]  = "atlas"
    payload: Optional[dict] = None
class SchedAddOut(ContractOut):
    name:     Optional[str]   = None
    next_run: Optional[float] = None

class SchedListIn(ContractIn): pass
class SchedListOut(ContractOut):
    jobs: list = []

class SchedRemoveIn(ContractIn):
    name: str
class SchedRemoveOut(ContractOut):
    removed: Optional[bool] = None


class SchedulerAgent(BaseAgent):
    NAME    = "scheduler"
    DOMAINS = ["schedule", "cron", "recurring", "interval", "timer",
               "periodic", "every", "daily", "reminder", "defer"]
    CAPABILITIES = frozenset({"time:read"})
    HASH_KEY = "adonis:schedule"
    TICK_S   = max(1, int(os.getenv("SCHEDULER_TICK_S", "5")))
    CONTRACTS = [
        Contract("scheduler.add", "scheduler", "schedule_add",
                 "Register a recurring job that publishes a payload to a target agent every N seconds.",
                 SchedAddIn, SchedAddOut, timeout_s=10),
        Contract("scheduler.list", "scheduler", "schedule_list",
                 "List all registered scheduled jobs.",
                 SchedListIn, SchedListOut, timeout_s=10),
        Contract("scheduler.remove", "scheduler", "schedule_remove",
                 "Remove a scheduled job by name.",
                 SchedRemoveIn, SchedRemoveOut, timeout_s=10),
    ]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._ticker = None

    # ── control plane ────────────────────────────────────────────────────────
    async def handle(self, task: dict, session_id: str) -> dict:
        task_type = task.get("type", "schedule_list")
        if task_type == "schedule_add":
            return await self._add(task)
        if task_type == "schedule_remove":
            return await self._remove(task)
        return await self._list()

    async def _add(self, task: dict) -> dict:
        name    = (task.get("name") or "").strip()
        every_s = int(task.get("every_s", 0))
        if not name or every_s <= 0:
            return {"status": "error", "reason": "name and positive every_s required", "agent": self.NAME}
        job = {
            "every_s":  every_s,
            "target":   task.get("target", "atlas"),
            "payload":  task.get("payload") or {},
            "next_run": time.time() + every_s,
            "created":  time.time(),
        }
        await self.redis.hset(self.HASH_KEY, name, json.dumps(job))
        log.info("[SCHED] Registered job '%s' every %ss -> %s", name, every_s, job["target"])
        return {"status": "ok", "agent": self.NAME, "name": name, "next_run": job["next_run"]}

    async def _remove(self, task: dict) -> dict:
        name = (task.get("name") or "").strip()
        removed = await self.redis.hdel(self.HASH_KEY, name)
        return {"status": "ok", "agent": self.NAME, "removed": bool(removed)}

    async def _list(self) -> dict:
        raw = await self.redis.hgetall(self.HASH_KEY)
        jobs = []
        for name, blob in (raw or {}).items():
            name = name.decode() if isinstance(name, (bytes, bytearray)) else name
            blob = blob.decode() if isinstance(blob, (bytes, bytearray)) else blob
            try:
                jobs.append({"name": name, **json.loads(blob)})
            except Exception:
                continue
        return {"status": "ok", "agent": self.NAME, "jobs": jobs}

    # ── tick loop ────────────────────────────────────────────────────────────
    async def _dispatch_due(self, now: Optional[float] = None) -> list[str]:
        """Publish every job whose next_run has passed and reschedule it.
        Returns the names dispatched (handy for tests)."""
        now = now if now is not None else time.time()
        raw = await self.redis.hgetall(self.HASH_KEY)
        fired: list[str] = []
        for name, blob in (raw or {}).items():
            name = name.decode() if isinstance(name, (bytes, bytearray)) else name
            blob = blob.decode() if isinstance(blob, (bytes, bytearray)) else blob
            try:
                job = json.loads(blob)
            except Exception:
                continue
            if job.get("next_run", 0) > now:
                continue
            payload = dict(job.get("payload") or {})
            payload.setdefault("session_id", f"sched:{name}")
            payload.setdefault("type", payload.get("type", "task"))
            try:
                await self.redis.publish(f"adonis:agent:{job['target']}", json.dumps(payload))
                fired.append(name)
                log.info("[SCHED] Fired '%s' -> %s", name, job["target"])
            except Exception as e:
                log.warning("[SCHED] dispatch of '%s' failed: %s", name, e)
                continue
            job["next_run"] = now + job["every_s"]
            await self.redis.hset(self.HASH_KEY, name, json.dumps(job))
        return fired

    async def _tick_loop(self):
        log.info("[SCHED] Ticker started (every %ss).", self.TICK_S)
        while self._alive:
            try:
                await self._dispatch_due()
            except Exception as e:
                log.warning("[SCHED] tick error: %s", e)
            await asyncio.sleep(self.TICK_S)

    async def run(self):
        """Run the pub/sub control loop and the interval ticker together."""
        self._ticker = asyncio.create_task(self._tick_loop(), name="scheduler:ticker")
        try:
            await super().run()
        finally:
            if self._ticker:
                self._ticker.cancel()
