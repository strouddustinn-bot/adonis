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
import ipaddress
import logging
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.parse import quote_plus, urlparse

import httpx

from tools.registry import REGISTRY, Tool

log = logging.getLogger("tools.builtin")

MAX_BODY = 30_000  # bytes; keeps tool output bounded for context engine

# Explicit vault path constraint. http_fetch is locked away from RFC1918,
# loopback, link-local (cloud metadata sits on 169.254.169.254) and IPv6
# unique-local. Skip these checks only if HTTP_FETCH_TRUST_INTERNAL=1 in
# the environment (don't set that lightly).
VAULT_ALLOWED_PREFIXES = ("MEMORY/", "SELF/", "ADONIS/", "WINS/", "LOSSES/")
TRUST_INTERNAL_HTTP    = os.getenv("HTTP_FETCH_TRUST_INTERNAL", "0") == "1"


def _safe_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _resolve_public(host: str) -> bool:
    """True iff `host` resolves only to public IPs. Blocks SSRF-style attempts
    to hit cloud-metadata, container sidecars, or other internal services."""
    if TRUST_INTERNAL_HTTP: return True
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def _safe_vault_path(p: str) -> str | None:
    """Normalise a vault path and reject traversal / disallowed prefixes."""
    if not p: return None
    if ".." in p.split("/"): return None
    if p.startswith("/"):    return None
    norm = str(PurePosixPath(p))
    if norm == "." or norm.startswith("/") or ".." in norm.split("/"):
        return None
    if not any(norm.startswith(pref) for pref in VAULT_ALLOWED_PREFIXES):
        return None
    return norm


async def _http_fetch(args: dict) -> dict:
    url = args.get("url", "")
    if not _safe_url(url):
        return {"error": "invalid url"}
    host = urlparse(url).hostname or ""
    if not _resolve_public(host):
        return {"error": f"refused: {host} resolves to a private/internal address"}
    headers = {"User-Agent": "AdonisAgent/1.0 (+research)"}
    async with httpx.AsyncClient(follow_redirects=False, timeout=15.0) as c:
        r = await c.get(url, headers=headers)
        # If the server redirects, re-validate the target host before following.
        if 300 <= r.status_code < 400:
            loc = r.headers.get("location", "")
            if loc and _safe_url(loc):
                next_host = urlparse(loc).hostname or ""
                if not _resolve_public(next_host):
                    return {"error": f"refused redirect: {next_host} is internal"}
                r = await c.get(loc, headers=headers)
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
        norm = _safe_vault_path(args.get("path", ""))
        if not norm:
            return {"error": "path refused: must be a relative path under "
                             + ", ".join(VAULT_ALLOWED_PREFIXES) + " with no '..'"}
        text = await obsidian.read_note(norm)
        return {"path": norm, "text": text or "", "found": text is not None}

    async def _vault_append(args: dict) -> dict:
        if not obsidian: return {"error": "obsidian disabled"}
        norm = _safe_vault_path(args.get("path", ""))
        if not norm:
            return {"error": "path refused: must be a relative path under "
                             + ", ".join(VAULT_ALLOWED_PREFIXES) + " with no '..'"}
        ok = await obsidian.append_note(norm, args.get("text", ""))
        return {"path": norm, "ok": ok}

    return _vault_read, _vault_append


async def _now(_args: dict) -> dict:
    return {"utc": datetime.now(timezone.utc).isoformat()}


def register_builtins(*, obsidian=None) -> None:
    REGISTRY.register(Tool(
        name="http_fetch",
        description="HTTP GET a URL. Returns status, final url, body text (truncated to ~30KB). Private and link-local addresses are refused.",
        schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        call=_http_fetch,
        required_capabilities=frozenset({"net:http_get"}),
    ), overwrite=True)
    REGISTRY.register(Tool(
        name="web_search",
        description="Search the open web. Returns titles, URLs, and snippets for the top N hits.",
        schema={"type": "object",
                "properties": {"query": {"type": "string"}, "n": {"type": "integer", "default": 5}},
                "required": ["query"]},
        call=_web_search,
        required_capabilities=frozenset({"net:web_search"}),
    ), overwrite=True)
    REGISTRY.register(Tool(
        name="arxiv_search",
        description="Search arXiv for academic papers. Returns title, URL, date, summary.",
        schema={"type": "object",
                "properties": {"query": {"type": "string"}, "n": {"type": "integer", "default": 5}},
                "required": ["query"]},
        call=_arxiv_search,
        required_capabilities=frozenset({"net:arxiv"}),
    ), overwrite=True)
    REGISTRY.register(Tool(
        name="now",
        description="Current UTC time as ISO-8601 string.",
        schema={"type": "object", "properties": {}},
        call=_now,
        required_capabilities=frozenset({"time:read"}),
    ), overwrite=True)
    if obsidian:
        vread, vapp = _make_vault_tools(obsidian)
        REGISTRY.register(Tool(
            name="vault_read",
            description="Read a note from the Adonis Obsidian vault by path (restricted to MEMORY/, SELF/, ADONIS/, WINS/, LOSSES/).",
            schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            call=vread,
            required_capabilities=frozenset({"vault:read"}),
        ), overwrite=True)
        REGISTRY.register(Tool(
            name="vault_append",
            description="Append text to a note in the Adonis Obsidian vault (restricted to MEMORY/, SELF/, ADONIS/, WINS/, LOSSES/).",
            schema={"type": "object",
                    "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["path", "text"]},
            call=vapp,
            required_capabilities=frozenset({"vault:write"}),
        ), overwrite=True)
