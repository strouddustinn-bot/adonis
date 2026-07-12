"""Offline contract tests for the Obsidian Local REST API bridge."""

from collections.abc import Callable

import httpx
import pytest

from memory.obsidian_bridge import ObsidianBridge

RequestHandler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def requests() -> list[httpx.Request]:
    return []


def _bridge(
    handler: RequestHandler,
) -> tuple[ObsidianBridge, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return (
        ObsidianBridge(
            base_url="http://obsidian.test/",
            token="test-token",
            client=client,
        ),
        client,
    )


async def test_read_note_escapes_path_and_uses_bearer_auth(
    requests: list[httpx.Request],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="# Note")

    bridge, client = _bridge(handler)
    try:
        assert await bridge.read_note("MEMORY/My note #1.md") == "# Note"
    finally:
        await client.aclose()

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    expected_url = "http://obsidian.test/vault/MEMORY/My%20note%20%231.md"
    assert str(request.url) == expected_url
    assert request.headers["Authorization"] == "Bearer test-token"
    assert "Content-Type" not in request.headers


async def test_read_note_returns_none_for_missing_or_invalid_paths(
    requests: list[httpx.Request],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(404)

    bridge, client = _bridge(handler)
    try:
        assert await bridge.read_note("MEMORY/missing.md") is None
        assert await bridge.read_note("../outside.md") is None
    finally:
        await client.aclose()

    assert len(requests) == 1


async def test_write_note_accepts_any_successful_response(
    requests: list[httpx.Request],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201)

    bridge, client = _bridge(handler)
    try:
        assert await bridge.write_note("MEMORY/note.md", "hello ✓") is True
    finally:
        await client.aclose()

    request = requests[0]
    assert request.method == "PUT"
    assert request.content == "hello ✓".encode()
    assert request.headers["Content-Type"] == "text/markdown; charset=utf-8"


async def test_append_note_uses_atomic_post_without_reading_first(
    requests: list[httpx.Request],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    bridge, client = _bridge(handler)
    try:
        assert await bridge.append_note("MEMORY/log.md", "new entry\n") is True
    finally:
        await client.aclose()

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.content == b"new entry\n"


async def test_search_returns_json_matches(
    requests: list[httpx.Request],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[{"filename": "MEMORY/note.md", "score": 1.0}],
        )

    bridge, client = _bridge(handler)
    try:
        assert await bridge.search("adonis") == [
            {"filename": "MEMORY/note.md", "score": 1.0}
        ]
    finally:
        await client.aclose()

    request = requests[0]
    assert request.method == "POST"
    assert request.headers["Accept"] == "application/json"
    assert request.url.params == httpx.QueryParams(
        {"query": "adonis", "contextLength": "100"}
    )


async def test_search_validates_response_shape_and_handles_transport_errors(
    requests: list[httpx.Request],
) -> None:
    def malformed_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"unexpected": "object"})

    bridge, client = _bridge(malformed_handler)
    try:
        assert await bridge.search("adonis") == []
    finally:
        await client.aclose()

    def offline_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    bridge, client = _bridge(offline_handler)
    try:
        assert await bridge.search("adonis") == []
    finally:
        await client.aclose()

    assert requests[0].method == "POST"
    expected_params = httpx.QueryParams(
        {"query": "adonis", "contextLength": "100"}
    )
    assert requests[0].url.params == expected_params
