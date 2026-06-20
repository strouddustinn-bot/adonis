"""
tests/conftest.py
=================
Shared fakes for the Adonis test suite. No network, no real Redis, no real
Anthropic. Everything the units-under-test touch is stubbed here so the
suite runs fully offline (and fast) in CI.
"""
import json
from collections import defaultdict

import pytest


class FakeRedis:
    """In-memory async stand-in for the subset of redis.asyncio used by
    Adonis: string get/set/setex/delete/exists/incr, list lpush/ltrim/
    lrange/llen, keys, ping. Values are stored as bytes to mirror the real
    client (decode_responses=False)."""

    def __init__(self):
        self.kv: dict = {}
        self.lists: dict = defaultdict(list)
        self.counters: dict = defaultdict(int)
        self.hashes: dict = defaultdict(dict)
        self.published: list = []   # (channel, message) tuples

    @staticmethod
    def _b(v):
        if isinstance(v, bytes):
            return v
        return str(v).encode()

    async def ping(self):
        return True

    async def set(self, k, v):
        self.kv[k] = self._b(v)
        return True

    async def setex(self, k, ttl, v):
        self.kv[k] = self._b(v)
        return True

    async def get(self, k):
        return self.kv.get(k)

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self.kv:
                del self.kv[k]; n += 1
            if k in self.lists:
                del self.lists[k]; n += 1
        return n

    async def exists(self, k):
        return 1 if (k in self.kv or k in self.lists or k in self.counters) else 0

    async def incr(self, k):
        self.counters[k] += 1
        return self.counters[k]

    async def lpush(self, k, *vals):
        for v in vals:
            self.lists[k].insert(0, self._b(v))
        return len(self.lists[k])

    async def ltrim(self, k, start, end):
        if end == -1:
            self.lists[k] = self.lists[k][start:]
        else:
            self.lists[k] = self.lists[k][start:end + 1]
        return True

    async def lrange(self, k, start, end):
        if end == -1:
            return list(self.lists[k][start:])
        return list(self.lists[k][start:end + 1])

    async def llen(self, k):
        return len(self.lists[k])

    async def keys(self, pattern="*"):
        import fnmatch
        all_keys = list(self.kv) + list(self.lists) + list(self.counters)
        return [self._b(k) for k in all_keys if fnmatch.fnmatch(k, pattern)]

    # ── hashes ──────────────────────────────────────────────────────────────
    async def hset(self, key, field, value):
        new = field not in self.hashes[key]
        self.hashes[key][field] = self._b(value)
        return 1 if new else 0

    async def hget(self, key, field):
        return self.hashes[key].get(field)

    async def hgetall(self, key):
        return dict(self.hashes[key])

    async def hdel(self, key, *fields):
        n = 0
        for f in fields:
            if f in self.hashes[key]:
                del self.hashes[key][f]; n += 1
        return n

    async def hlen(self, key):
        return len(self.hashes[key])

    async def publish(self, channel, message):
        self.published.append((channel, message))
        return 0

    async def aclose(self):
        return True


class FakeMessage:
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]
        self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()


class _FakeStream:
    """Async context manager mimicking anthropic's streaming response."""
    def __init__(self, text):
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        # Chunk into a few pieces to exercise the delta path.
        async def _it():
            words = self._text.split(" ")
            for i, w in enumerate(words):
                yield (w if i == 0 else " " + w)
        return _it()

    async def get_final_message(self):
        return FakeMessage(self._text)


class FakeMessages:
    def __init__(self, responder):
        self._responder = responder

    async def create(self, **kwargs):
        text = self._responder(kwargs)
        return FakeMessage(text)

    def stream(self, **kwargs):
        return _FakeStream(self._responder(kwargs))


class FakeLLM:
    """Anthropic-shaped fake. `responder(kwargs) -> str` decides each reply;
    defaults to echoing a benign JSON so JSON-parsing call sites don't crash."""

    def __init__(self, responder=None):
        self.calls = []
        self.messages = FakeMessages(self._wrap(responder))

    def _wrap(self, responder):
        def _inner(kwargs):
            self.calls.append(kwargs)
            if responder is not None:
                return responder(kwargs)
            return "{}"
        return _inner


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def fake_llm():
    return FakeLLM()
