"""
tests/test_scheduler_agent.py
=============================
SCHEDULER agent: add/list/remove jobs and fire due ones to their target
agent's channel, rescheduling as it goes.
"""
import json
import time


from openclaw.agents.scheduler import SchedulerAgent


def _agent(fake_redis, fake_llm):
    return SchedulerAgent(fake_llm, fake_redis, fuse=None, governor=None)


async def test_add_and_list(fake_redis, fake_llm):
    a = _agent(fake_redis, fake_llm)
    res = await a.handle({"type": "schedule_add", "name": "daily-mirror",
                          "every_s": 86400, "target": "mirror",
                          "payload": {"type": "mirror_cycle"}}, "s1")
    assert res["status"] == "ok"
    listing = await a.handle({"type": "schedule_list"}, "s1")
    names = [j["name"] for j in listing["jobs"]]
    assert names == ["daily-mirror"]
    assert listing["jobs"][0]["target"] == "mirror"


async def test_add_validation(fake_redis, fake_llm):
    a = _agent(fake_redis, fake_llm)
    res = await a.handle({"type": "schedule_add", "name": "", "every_s": 0}, "s1")
    assert res["status"] == "error"


async def test_remove(fake_redis, fake_llm):
    a = _agent(fake_redis, fake_llm)
    await a.handle({"type": "schedule_add", "name": "j", "every_s": 10}, "s1")
    res = await a.handle({"type": "schedule_remove", "name": "j"}, "s1")
    assert res["removed"] is True
    listing = await a.handle({"type": "schedule_list"}, "s1")
    assert listing["jobs"] == []


async def test_dispatch_due_fires_and_reschedules(fake_redis, fake_llm):
    a = _agent(fake_redis, fake_llm)
    await a.handle({"type": "schedule_add", "name": "hb", "every_s": 60,
                    "target": "sentinel", "payload": {"type": "health"}}, "s1")
    # Not due yet (next_run ~ now+60).
    assert await a._dispatch_due(now=time.time()) == []
    # Force due by advancing the clock past next_run.
    fired = await a._dispatch_due(now=time.time() + 120)
    assert fired == ["hb"]
    # Published to the target channel with the payload.
    channels = [c for c, _ in fake_redis.published]
    assert "adonis:agent:sentinel" in channels
    msg = json.loads(fake_redis.published[-1][1])
    assert msg["type"] == "health"
    assert msg["session_id"] == "sched:hb"
    # Rescheduled forward, so an immediate re-tick does nothing.
    assert await a._dispatch_due(now=time.time() + 120) == []


async def test_malformed_job_entry_ignored_in_list_and_dispatch(fake_redis, fake_llm):
    a = _agent(fake_redis, fake_llm)

    # Inject a malformed entry directly into the hash to exercise the defensive
    # json.loads() handling in both _list and _dispatch_due.
    await fake_redis.hset(SchedulerAgent.HASH_KEY, "bad", b"{not-json")

    # Also register a valid job so we can verify normal behavior is unaffected.
    await a.handle(
        {"type": "schedule_add", "name": "hb", "every_s": 60,
         "target": "sentinel", "payload": {"type": "health"}},
        "s1",
    )

    # Listing must skip the malformed entry and still return the valid one.
    listing = await a.handle({"type": "schedule_list"}, "s1")
    job_names = {job["name"] for job in listing["jobs"]}
    assert "hb" in job_names
    assert "bad" not in job_names

    # Dispatching must also skip the malformed entry without raising.
    fired = await a._dispatch_due(now=time.time() + 120)
    assert fired == ["hb"]


async def test_run_launches_and_cancels_ticker(fake_redis, fake_llm):
    import asyncio
    import contextlib
    a = _agent(fake_redis, fake_llm)

    async def _fake_super_run():
        await asyncio.sleep(0.01)

    # Patch the pub/sub loop so run() returns quickly; assert the ticker spun up
    # and was torn down when run() exited.
    import openclaw.base_agent as ba
    orig = ba.BaseAgent.run
    ba.BaseAgent.run = lambda self: _fake_super_run()
    try:
        await a.run()
    finally:
        ba.BaseAgent.run = orig
    assert a._ticker is not None
    with contextlib.suppress(asyncio.CancelledError):
        await a._ticker
    assert a._ticker.done()
