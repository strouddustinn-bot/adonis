"""
memory/obsidian_bridge.py
==========================
HTTP bridge to an Obsidian vault via the obsidian-local-rest-api plugin.
Install: https://github.com/coddingtonbear/obsidian-local-rest-api

Configure in .env:
  OBSIDIAN_API=http://localhost:27123
  OBSIDIAN_TOKEN=your_token_here
"""
from __future__ import annotations

import logging
import os
from typing import cast
from urllib.parse import quote

import httpx

log = logging.getLogger("obsidian")

DEFAULT_BASE_URL = "http://localhost:27123"
DEFAULT_TIMEOUT_S = 5.0


class ObsidianBridge:
    """Small async client for the obsidian-local-rest-api vault endpoints.

    The bridge owns its HTTP client unless one is injected, so production code
    can reuse connections while tests can provide an ``httpx.MockTransport``.
    Call :meth:`aclose` during application shutdown when the bridge owns the
    client.
    """

    base: str
    token: str
    _timeout_s: float
    _client: httpx.AsyncClient
    _owns_client: bool

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        configured_base = (
            base_url
            if base_url is not None
            else os.getenv("OBSIDIAN_API", DEFAULT_BASE_URL)
        )
        self.base = configured_base.rstrip("/")
        self.token = (
            token if token is not None else os.getenv("OBSIDIAN_TOKEN", "")
        )
        self._timeout_s = timeout_s
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def __aenter__(self) -> "ObsidianBridge":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the internal client; injected clients remain caller-owned."""
        if self._owns_client:
            await self._client.aclose()

    async def read_note(self, path: str) -> str | None:
        """Return a note's Markdown, or ``None`` when it cannot be read."""
        url = self._note_url_or_none(path, "read_note")
        if url is None:
            return None

        response = await self._request(
            "read_note", "GET", url, headers=self._headers()
        )
        if response is None:
            return None
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            self._log_status("read_note", path, response)
            return None
        return response.text

    async def write_note(self, path: str, content: str) -> bool:
        """Create or replace a note, returning whether Obsidian accepted it."""
        url = self._note_url_or_none(path, "write_note")
        if url is None:
            return False

        response = await self._request(
            "write_note",
            "PUT",
            url,
            content=content,
            headers=self._headers(content_type="text/markdown; charset=utf-8"),
        )
        if response is None:
            return False
        if not response.is_success:
            self._log_status("write_note", path, response)
            return False
        return True

    async def search(self, query: str) -> list[dict[str, object]]:
        """Return simple-search matches, or an empty list when search fails."""
        response = await self._request(
            "search",
            "POST",
            f"{self.base}/search/simple/",
            params={"query": query, "contextLength": 100},
            headers=self._headers(accept="application/json"),
        )
        if response is None:
            return []
        if response.status_code != 200:
            self._log_status("search", query, response)
            return []

        try:
            payload = cast(object, response.json())
        except ValueError:
            log.warning("Obsidian search returned invalid JSON")
            return []

        if not isinstance(payload, list):
            log.warning(
                "Obsidian search returned an unexpected response shape"
            )
            return []
        items = cast(list[object], payload)
        if not all(isinstance(item, dict) for item in items):
            log.warning(
                "Obsidian search returned an unexpected response shape"
            )
            return []
        return cast(list[dict[str, object]], items)

    async def append_note(self, path: str, content: str) -> bool:
        """Append Markdown through Obsidian's server-side endpoint."""
        url = self._note_url_or_none(path, "append_note")
        if url is None:
            return False

        response = await self._request(
            "append_note",
            "POST",
            url,
            content=content,
            headers=self._headers(content_type="text/markdown; charset=utf-8"),
        )
        if response is None:
            return False
        if not response.is_success:
            self._log_status("append_note", path, response)
            return False
        return True

    def _note_url_or_none(self, path: str, operation: str) -> str | None:
        try:
            return self._note_url(path)
        except ValueError as exc:
            log.warning(
                "Obsidian %s rejected path %r: %s", operation, path, exc
            )
            return None

    def _note_url(self, path: str) -> str:
        """Build an escaped vault URL while preserving folder separators."""
        normalized = path.replace("\\", "/")
        parts = normalized.split("/")
        if (
            not normalized
            or normalized.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError(
                "invalid vault path: relative path without traversal required"
            )
        return f"{self.base}/vault/{quote(normalized, safe='/')}"

    def _headers(
        self,
        *,
        content_type: str | None = None,
        accept: str | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if content_type:
            headers["Content-Type"] = content_type
        if accept:
            headers["Accept"] = accept
        return headers

    async def _request(
        self,
        operation: str,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        content: str | None = None,
        params: dict[str, str | int] | None = None,
    ) -> httpx.Response | None:
        try:
            return await self._client.request(
                method,
                url,
                content=content,
                headers=headers,
                params=params,
                timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            log.warning("Obsidian %s request failed: %s", operation, exc)
            return None

    @staticmethod
    def _log_status(
        operation: str, target: str, response: httpx.Response
    ) -> None:
        log.warning(
            "Obsidian %s failed for %r: HTTP %d",
            operation,
            target,
            response.status_code,
        )
