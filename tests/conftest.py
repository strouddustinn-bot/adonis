"""
tests/conftest.py
=================
Shared fakes for the Adonis test suite. No network, no real Redis, no real
Anthropic. Everything the units-under-test touch is stubbed here so the
suite runs fully offline (and fast) in CI.
"""

from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

import pytest


@dataclass
class _Block:
    text: str


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


class FakeRedis:
    """In-memory async stand-in for the subset of redis.asyncio used by
    Adonis: string get/set/setex/delete/exists/incr, list lpush/ltrim/
    lrange/llen, keys, ping. Values are stored as bytes to mirror the real
    client (decode_responses=False)."""

    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}
        self.lists: defaultdict[str, list[bytes]] = defaultdict(list)
        self.counters: defaultdict[str, int] = defaultdict(int)
        self.hashes: defaultdict[str, dict[str, bytes]] = defaultdict(dict)
        self.published: list[tuple[str, object]] = []  # (channel, message) tuples

    @staticmethod
    def _b(v: object) -> bytes:
        if isinstance(v, bytes):
            return v
        return str(v).encode()

    async def ping(self) -> bool:
        return True

    async def set(self, k: str, v: object) -> bool:
        self.kv[k] = self._b(v)
        return True

    async def setex(self, k: str, _ttl: int, v: object) -> bool:
        self.kv[k] = self._b(v)
        return True

    async def get(self, k: str) -> bytes | None:
        return self.kv.get(k)

    async def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            if k in self.kv:
                del self.kv[k]
                n += 1
            if k in self.lists:
                del self.lists[k]
                n += 1
        return n

    async def exists(self, k: str) -> int:
        return 1 if (k in self.kv or k in self.lists or k in self.counters) else 0

    async def incr(self, k: str) -> int:
        self.counters[k] += 1
        return self.counters[k]

    async def lpush(self, k: str, *vals: object) -> int:
        for v in vals:
            self.lists[k].insert(0, self._b(v))
        return len(self.lists[k])

    async def ltrim(self, k: str, start: int, end: int) -> bool:
        if end == -1:
            self.lists[k] = self.lists[k][start:]
        else:
            self.lists[k] = self.lists[k][start : end + 1]
        return True

    async def lrange(self, k: str, start: int, end: int) -> list[bytes]:
        if end == -1:
            return list(self.lists[k][start:])
        return list(self.lists[k][start : end + 1])

    async def llen(self, k: str) -> int:
        return len(self.lists[k])

    async def keys(self, pattern: str = "*") -> list[bytes]:
        import fnmatch

        all_keys: list[str] = list(self.kv) + list(self.lists) + list(self.counters)
        return [self._b(k) for k in all_keys if fnmatch.fnmatch(k, pattern)]

    # ── hashes ──────────────────────────────────────────────────────────────
    async def hset(self, key: str, field: str, value: object) -> int:
        new = field not in self.hashes[key]
        self.hashes[key][field] = self._b(value)
        return 1 if new else 0

    async def hget(self, key: str, field: str) -> bytes | None:
        return self.hashes[key].get(field)

    async def hgetall(self, key: str) -> dict[str, bytes]:
        return dict(self.hashes[key])

    async def hdel(self, key: str, *fields: str) -> int:
        n = 0
        for f in fields:
            if f in self.hashes[key]:
                del self.hashes[key][f]
                n += 1
        return n

    async def hlen(self, key: str) -> int:
        return len(self.hashes[key])

    async def publish(self, channel: str, message: object) -> int:
        self.published.append((channel, message))
        return 0

    async def aclose(self) -> bool:
        return True


class FakeMessage:
    content: list[_Block]
    usage: _Usage

    def __init__(self, text: str) -> None:
        self.content = [_Block(text=text)]
        self.usage = _Usage(input_tokens=10, output_tokens=20)


class _FakeStream:
    """Async context manager mimicking anthropic's streaming response."""

    _text: str

    def __init__(self, text: str) -> None:
        self._text = text

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    @property
    def text_stream(self) -> AsyncGenerator[str, None]:
        # Chunk into a few pieces to exercise the delta path.
        async def _it() -> AsyncGenerator[str, None]:
            words = self._text.split(" ")
            for i, w in enumerate(words):
                yield (w if i == 0 else " " + w)

        return _it()

    async def get_final_message(self) -> FakeMessage:
        return FakeMessage(self._text)


LLMResponder = Callable[[dict[str, object]], str]


class FakeMessages:
    _responder: LLMResponder

    def __init__(self, responder: LLMResponder) -> None:
        self._responder = responder

    async def create(self, **kwargs: object) -> FakeMessage:
        text = self._responder(kwargs)
        return FakeMessage(text)

    def stream(self, **kwargs: object) -> _FakeStream:
        return _FakeStream(self._responder(kwargs))


class FakeLLM:
    """Anthropic-shaped fake. `responder(kwargs) -> str` decides each reply;
    defaults to echoing a benign JSON so JSON-parsing call sites don't crash."""

    calls: list[dict[str, object]]
    messages: FakeMessages

    def __init__(self, responder: LLMResponder | None = None) -> None:
        self.calls = []
        self.messages = FakeMessages(self._wrap(responder))

    def _wrap(self, responder: LLMResponder | None) -> LLMResponder:
        def _inner(kwargs: dict[str, object]) -> str:
            self.calls.append(kwargs)
            if responder is not None:
                return responder(kwargs)
            return "{}"

        return _inner


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
