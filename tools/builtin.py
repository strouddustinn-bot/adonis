"""
tools/builtin.py
=================
Built-in tools always available to every agent.

  http_fetch   — GET an HTTP(S) URL, return status + text (size-capped)
  web_search   — DuckDuckGo HTML scrape, no API key required
  arxiv_search — query the arXiv public API
  vault_read   — read a note from the Obsidian vault
  vault_write  — append to a note in the Obsidian vault
  now          — current UTC ISO timestamp

Each is registered into the module singleton in tools.registry.REGISTRY.
"""
import html
import logging
import os
import re
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import httpx

from tools.registry import REGISTRY, Tool

log = logging.getLogger("tools.builtin")

MAX_BODY = 30_000  # bytes; keeps tool output bounded for context engine


def _safe_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


async def _http_fetch(args: dict) -> dict:
    url = args.get("url", "")
    if not _safe_url(url):
        return {"error": "invalid url"}
    headers = {"User-Agent": "AdonisAgent/1.0 (+research)"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as c:
        r = await c.get(url, headers=headers)
        body = r.text[:MAX_BODY]
        return {"status": r.status_code, "url": str(r.url), "body": body, "truncated": len(r.text) > MAX_BODY}


_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.S,
)


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


async def _web_search(args: dict) -> dict:
    query = args.get("query", "").strip()
    n     = max(1, min(int(args.get("n", 5)), 10))
    if not query:
        return {"error": "empty query"}
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0,
                                 headers={"User-Agent": "Mozilla/5.0 AdonisAgent"}) as c:
        r = await c.get(url)
    hits = []
    for m in _DDG_RESULT_RE.finditer(r.text):
        href, title, snippet = m.groups()
        hits.append({"url": href, "title": _strip_tags(title), "snippet": _strip_tags(snippet)})
        if len(hits) >= n:
            break
    return {"query": query, "hits": hits}


_ARXIV_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.S)
_ARXIV_FIELD_RE = re.compile(r"<(title|summary|id|published)>(.*?)</\1>", re.S)


async def _arxiv_search(args: dict) -> dict:
    query = args.get("query", "").strip()
    n     = max(1, min(int(args.get("n", 5)), 10))
    if not query:
        return {"error": "empty query"}
    url = f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}&max_results={n}"
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get(url)
    papers = []
    for entry in _ARXIV_ENTRY_RE.findall(r.text):
        fields = {k: v.strip() for k, v in _ARXIV_FIELD_RE.findall(entry)}
        papers.append({
            "title":     fields.get("title", "").replace("\n", " ").strip(),
            "url":       fields.get("id", "").strip(),
            "published": fields.get("published", "")[:10],
            "summary":   fields.get("summary", "").replace("\n", " ").strip()[:600],
        })
    return {"query": query, "papers": papers}


def _make_vault_tools(obsidian):
    async def _vault_read(args: dict) -> dict:
        if not obsidian: return {"error": "obsidian disabled"}
        text = await obsidian.read_note(args.get("path", ""))
        return {"path": args.get("path", ""), "text": text or "", "found": text is not None}

    async def _vault_append(args: dict) -> dict:
        if not obsidian: return {"error": "obsidian disabled"}
        ok = await obsidian.append_note(args.get("path", ""), args.get("text", ""))
        return {"path": args.get("path", ""), "ok": ok}

    return _vault_read, _vault_append


async def _now(_args: dict) -> dict:
    return {"utc": datetime.now(timezone.utc).isoformat()}


def register_builtins(*, obsidian=None) -> None:
    REGISTRY.register(Tool(
        name="http_fetch",
        description="HTTP GET a URL. Returns status, final url, body text (truncated to ~30KB).",
        schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        call=_http_fetch,
    ), overwrite=True)
    REGISTRY.register(Tool(
        name="web_search",
        description="Search the open web. Returns titles, URLs, and snippets for the top N hits.",
        schema={"type": "object",
                "properties": {"query": {"type": "string"}, "n": {"type": "integer", "default": 5}},
                "required": ["query"]},
        call=_web_search,
    ), overwrite=True)
    REGISTRY.register(Tool(
        name="arxiv_search",
        description="Search arXiv for academic papers. Returns title, URL, date, summary.",
        schema={"type": "object",
                "properties": {"query": {"type": "string"}, "n": {"type": "integer", "default": 5}},
                "required": ["query"]},
        call=_arxiv_search,
    ), overwrite=True)
    REGISTRY.register(Tool(
        name="now",
        description="Current UTC time as ISO-8601 string.",
        schema={"type": "object", "properties": {}},
        call=_now,
    ), overwrite=True)
    if obsidian:
        vread, vapp = _make_vault_tools(obsidian)
        REGISTRY.register(Tool(
            name="vault_read",
            description="Read a note from the Adonis Obsidian vault by path.",
            schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            call=vread,
        ), overwrite=True)
        REGISTRY.register(Tool(
            name="vault_append",
            description="Append text to a note in the Adonis Obsidian vault.",
            schema={"type": "object",
                    "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["path", "text"]},
            call=vapp,
        ), overwrite=True)
