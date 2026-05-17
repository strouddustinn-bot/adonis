"""
hermes/api.py
==============
Hermes — Adonis HTTP interface layer.

Receives user messages, runs them through the Glasswing pipeline, returns
answers. Multi-agent goals route to Atlas via the Redis pub/sub bus.

Endpoints:
  POST /ask     — single-turn Q&A through Glasswing cache+context+soul
  POST /task    — multi-agent goal orchestrated by Atlas
  GET  /health  — runtime liveness + dependency status
  GET  /audit   — recent Prometheus audit records
  GET  /ges     — current Glasswing Efficiency Score report
"""
import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

log = logging.getLogger("hermes")
STATIC_DIR = Path(__file__).parent / "static"


class AskIn(BaseModel):
    message:    str
    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:10]}")
    max_tokens: int = 1024


class TaskIn(BaseModel):
    goal:       str
    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:10]}")
    timeout_s:  int = 60


def build_app(*, llm, redis, fuse, governor, model: str) -> FastAPI:
    app = FastAPI(title="Adonis Hermes", version="1.0")

    app.state.llm      = llm
    app.state.redis    = redis
    app.state.fuse     = fuse
    app.state.governor = governor
    app.state.model    = model

    @app.post("/ask")
    async def ask(body: AskIn):
        ctx = governor.context
        await ctx.append_turn(body.session_id, "user", body.message)

        prep = await governor.prepare(body.session_id, body.message)
        if prep.cache_hit:
            await ctx.append_turn(body.session_id, "assistant", prep.cached_result)
            return {
                "answer":     prep.cached_result,
                "cache_hit":  True,
                "agents":     [],
                "think":      prep.think_depth,
                "session_id": body.session_id,
            }

        try:
            response = await llm.messages.create(
                model=model,
                max_tokens=body.max_tokens,
                system=prep.system,
                messages=prep.messages,
            )
        except Exception as e:
            log.error("LLM call failed: %s", e)
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

        answer = response.content[0].text if response.content else ""
        tokens_used = (response.usage.input_tokens + response.usage.output_tokens) if getattr(response, "usage", None) else 0

        await governor.cache_result(body.message, answer)
        await ctx.append_turn(body.session_id, "assistant", answer)
        governor.record_ges(
            body.session_id, prep.active_agents,
            tokens_used=max(1, tokens_used), tasks_done=1,
            cache_hit=False, speed_ms=prep.efficiency.get("prep_ms", 0),
        )

        return {
            "answer":     answer,
            "cache_hit":  False,
            "agents":     prep.active_agents,
            "think":      prep.think_depth,
            "tokens":     tokens_used,
            "efficiency": prep.efficiency,
            "session_id": body.session_id,
        }

    @app.post("/task")
    async def task(body: TaskIn):
        trace_id   = uuid.uuid4().hex[:10]
        result_key = f"hermes:result:{trace_id}"
        payload = {
            "type":       "task",
            "goal":       body.goal,
            "session_id": body.session_id,
            "trace_id":   trace_id,
            "result_key": result_key,
        }
        await redis.publish("adonis:agent:atlas", json.dumps(payload))

        deadline = asyncio.get_event_loop().time() + body.timeout_s
        while asyncio.get_event_loop().time() < deadline:
            raw = await redis.get(result_key)
            if raw:
                await redis.delete(result_key)
                return {"trace_id": trace_id, "result": json.loads(raw), "session_id": body.session_id}
            await asyncio.sleep(0.25)
        raise HTTPException(status_code=504, detail=f"Atlas did not respond within {body.timeout_s}s")

    @app.get("/health")
    async def health():
        report = {"adonis": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}
        try:
            await redis.ping()
            report["redis"] = "ok"
        except Exception as e:
            report["redis"] = f"down: {e}"

        chroma = getattr(governor.context, "chroma", None)
        report["chroma"] = "ok" if chroma else "disabled"

        obs = getattr(governor.context, "obs", None)
        report["obsidian"] = "ok" if obs else "disabled"

        try:
            audit_len = await redis.llen("prometheus:audit")
            report["fuse_audit_entries"] = int(audit_len)
        except Exception:
            report["fuse_audit_entries"] = -1
        return report

    @app.get("/audit")
    async def audit(n: int = 20):
        raw = await redis.lrange("prometheus:audit", 0, max(0, n - 1))
        return [json.loads(r) for r in raw]

    @app.get("/ges")
    async def ges():
        return await governor.get_ges_report()

    @app.get("/config")
    async def config_snapshot():
        """Read-only runtime snapshot. All secrets are masked."""
        import hashlib
        from pathlib import Path
        from tools.registry import REGISTRY

        def _mask(v: str, head: int = 8, tail: int = 4) -> str:
            if not v: return ""
            if len(v) <= head + tail: return "•" * len(v)
            return f"{v[:head]}…{v[-tail:]}"

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        obs_token = os.getenv("OBSIDIAN_TOKEN", "")
        webhook = os.getenv("PROMETHEUS_ALERT_WEBHOOK", "")

        fuse_path = Path(__file__).resolve().parent.parent / "prometheus" / "fuse.py"
        pinned = os.getenv("PROMETHEUS_HASH", "UNSET")
        current = hashlib.sha256(fuse_path.read_bytes()).hexdigest() if fuse_path.exists() else ""

        mcp_raw = os.getenv("MCP_SERVERS", "").strip()
        mcp_names: list[str] = []
        mcp_status = "disabled"
        if mcp_raw:
            try:
                specs = json.loads(mcp_raw)
                mcp_names = [s.get("name", "?") for s in specs]
                mcp_status = "configured"
            except Exception:
                mcp_status = "invalid JSON"

        ctx = getattr(governor, "context", None)
        chroma_ok = bool(getattr(ctx, "chroma", None))
        obs_obj   = getattr(ctx, "obs", None)

        try:
            from routing.moe_router import AGENT_REGISTRY
            agent_names = [a.name for a in AGENT_REGISTRY]
        except Exception:
            agent_names = []

        return {
            "system": {
                "model":      model,
                "hermes_port": int(os.getenv("HERMES_PORT", "8088")),
                "log_level":  os.getenv("LOG_LEVEL", "INFO"),
            },
            "auth": {
                "anthropic_key_set":    bool(api_key.startswith("sk-ant-")),
                "anthropic_key_preview": _mask(api_key, 10, 4) if api_key else "<unset>",
            },
            "safety": {
                "fuse_hash_pinned":  pinned[:16] + ("…" if len(pinned) > 16 else "") if pinned != "UNSET" else "UNSET",
                "fuse_hash_current": (current[:16] + "…") if current else "?",
                "match":             bool(current) and pinned == current,
                "alert_webhook":     "configured" if webhook else "disabled",
            },
            "memory": {
                "redis_url":  os.getenv("REDIS_URL", "redis://localhost:6379"),
                "chroma_url": os.getenv("CHROMA_URL", ""),
                "chroma_wired":   chroma_ok,
                "obsidian_url":   os.getenv("OBSIDIAN_API", "") or "<disabled>",
                "obsidian_wired": bool(obs_obj),
                "obsidian_token": _mask(obs_token, 4, 4) if obs_token else "<unset>",
            },
            "mcp": {
                "status": mcp_status,
                "servers": mcp_names,
            },
            "tools": REGISTRY.names(),
            "agents": agent_names,
        }

    # ── Web UI ───────────────────────────────────────────────────────────
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(str(STATIC_DIR / "index.html"))

        @app.get("/ui", include_in_schema=False)
        async def ui_alias():
            return RedirectResponse("/")
    else:
        log.warning("Web UI assets missing at %s — UI disabled.", STATIC_DIR)

    return app
