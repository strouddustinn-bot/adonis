"""
supervisord.py — Adonis runtime supervisor.

Single-process async runner: builds shared infra clients, the Prometheus
fuse, and the Glasswing governor, then starts every implemented agent's
pubsub loop under one event loop. SIGTERM/SIGINT trigger clean shutdown.

Name shadows the unix `supervisor` pip package by design — the
docker-compose entrypoint imports this module directly via
`import supervisord; supervisord.main()`.
"""
import asyncio
import logging
import os
import signal
import sys

import redis.asyncio as aioredis
from anthropic import AsyncAnthropic

from glasswing.governor import GlasswingGovernor
from prometheus.fuse import PrometheusFuse
from memory.obsidian_bridge import ObsidianBridge
from tools.builtin import register_builtins
from tools.mcp_client import attach_mcp_servers, detach_mcp_servers

from openclaw.agents.atlas import AtlasAgent
from openclaw.agents.forge import ForgeAgent
from openclaw.agents.mirror import MirrorAgent
from openclaw.agents.scout import ScoutAgent
from openclaw.agents.sentinel import SentinelAgent
from openclaw.agents.smith import SmithAgent
from openclaw.agents.vector import VectorAgent

log = logging.getLogger("supervisord")


def _build_obsidian() -> ObsidianBridge | None:
    if not os.getenv("OBSIDIAN_API"):
        log.info("Obsidian disabled (OBSIDIAN_API unset).")
        return None
    bridge = ObsidianBridge()
    log.info("Obsidian bridge constructed: %s", bridge.base)
    return bridge


def _build_chroma_collection():
    """Return a Chroma collection or None if unreachable. The engine expects
    a collection-like object with .add() and .query()."""
    url = os.getenv("CHROMA_URL", "")
    if not url:
        log.info("Chroma disabled (CHROMA_URL unset).")
        return None
    try:
        import chromadb
        from urllib.parse import urlparse
        u = urlparse(url)
        client = chromadb.HttpClient(host=u.hostname or "chromadb", port=u.port or 8000)
        client.heartbeat()
        collection = client.get_or_create_collection("adonis_l3")
        log.info("Chroma collection ready: adonis_l3 @ %s", url)
        return collection
    except Exception as e:
        log.warning("Chroma unreachable (%s) — L3 semantic retrieval disabled.", e)
        return None

# routing/moe_router.py AGENT_REGISTRY also lists "hermes" (the interface
# / routing layer, represented by MoERouter dispatching to specialists)
# and "prometheus" (the safety circuit, instantiated as PrometheusFuse).
# Neither is a worker agent, so they don't appear here.
AGENT_CLASSES = [
    AtlasAgent, ForgeAgent, MirrorAgent, ScoutAgent,
    SentinelAgent, SmithAgent, VectorAgent,
]


async def _amain() -> int:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    redis = aioredis.from_url(redis_url, decode_responses=False)
    await redis.ping()
    log.info("Redis OK: %s", redis_url)

    llm = AsyncAnthropic()

    obsidian = _build_obsidian()
    chroma = _build_chroma_collection()

    fuse = PrometheusFuse(llm, redis, obsidian_bridge=obsidian)
    governor = GlasswingGovernor(llm, redis, chroma_client=chroma, obsidian_bridge=obsidian)

    register_builtins(obsidian=obsidian)
    mcp_servers = await attach_mcp_servers()

    agents = []
    for cls in AGENT_CLASSES:
        try:
            agents.append(cls(llm, redis, fuse, governor))
            log.info("Constructed agent: %s", cls.__name__)
        except Exception as e:
            log.error("Failed to construct %s: %s", cls.__name__, e)

    if not agents:
        log.error("No agents constructed; exiting.")
        return 1

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    tasks = [asyncio.create_task(a.run(), name=f"agent:{a.NAME}") for a in agents]
    log.info("%d agents idling on pubsub.", len(tasks))

    # Hermes HTTP entrypoint — the only way humans actually talk to Adonis.
    import uvicorn
    from hermes.api import build_app
    api_port = int(os.getenv("HERMES_PORT", "8088"))
    app = build_app(
        llm=llm, redis=redis, fuse=fuse, governor=governor,
        model=os.getenv("ADONIS_MODEL", "claude-sonnet-4-6"),
    )
    server = uvicorn.Server(uvicorn.Config(
        app, host="0.0.0.0", port=api_port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=False,
    ))
    tasks.append(asyncio.create_task(server.serve(), name="hermes:api"))
    log.info("Hermes API serving on :%d", api_port)
    log.info("Awaiting shutdown signal.")

    await stop.wait()
    log.info("Shutdown signal received.")
    server.should_exit = True
    for a in agents:
        a.stop()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await detach_mcp_servers(mcp_servers)
    await redis.aclose()
    log.info("Clean shutdown.")
    return 0


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
