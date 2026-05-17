"""
openclaw/agents/vector.py
==========================
VECTOR — Lead-gen, SEO, and web-intelligence specialist.

Task shapes:
  {type:"find_leads",  content:"<sector/keyword>", n?:int}
  {type:"site_audit",  url:"https://...", focus?:"seo|copy|positioning"}
  {type:"keyword_scan",content:"<topic>"}
Atlas-dispatched task (no `type`) is treated as find_leads on `content`.
"""
import asyncio
import json
import logging
import re

from openclaw.base_agent import BaseAgent

log = logging.getLogger("vector")

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_META_DESC_RE = re.compile(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)', re.I)
_H_RE = re.compile(r"<h([1-3])[^>]*>(.*?)</h\1>", re.S | re.I)


def _strip(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _extract_page_summary(html: str) -> dict:
    title = _strip((_TITLE_RE.search(html) or [None, ""])[1] if _TITLE_RE.search(html) else "")
    md = _META_DESC_RE.search(html)
    desc = md.group(1) if md else ""
    headings = [_strip(t) for _, t in _H_RE.findall(html)][:10]
    return {"title": title[:200], "description": desc[:300], "headings": headings}


class VectorAgent(BaseAgent):
    NAME    = "vector"
    DOMAINS = ["leads", "seo", "web", "research", "traffic", "marketing", "search"]
    # Hits public web/SEO sources; writes nothing.
    CAPABILITIES = frozenset({"net:http_get", "net:web_search", "time:read"})

    async def handle(self, task: dict, session_id: str) -> dict:
        task_type = task.get("type")
        if task_type == "site_audit":
            return await self._site_audit(task, session_id)
        if task_type == "keyword_scan":
            return await self._keyword_scan(task, session_id)
        return await self._find_leads(task, session_id)

    async def _find_leads(self, task: dict, session_id: str) -> dict:
        query = (task.get("content") or task.get("task") or task.get("goal") or "").strip()
        if not query:
            return {"status": "error", "reason": "no query", "agent": self.NAME}
        n = max(3, min(int(task.get("n", 6)), 10))

        approved, _ = await self.evaluate_action(
            "vector:find_leads", f"Lead-gen scan for: {query[:200]}",
            payload={"query": query[:200]}, session_id=session_id,
        )
        if not approved:
            return {"status": "blocked", "agent": self.NAME}

        res = await self.use_tool("web_search", {"query": query, "n": n}, session_id=session_id)
        if not res.get("ok"):
            return {"status": "error", "agent": self.NAME, "reason": res.get("error")}
        hits = res["result"].get("hits", [])

        synth_prompt = (
            f"Query: {query}\n\nSearch results:\n"
            + "\n".join(f"- {h['title']} | {h['url']} | {h['snippet']}" for h in hits)
            + "\n\nReturn ONLY JSON: "
              "{\"leads\":[{\"name\":\"...\",\"url\":\"...\",\"fit\":\"...\",\"why\":\"...\"}],\"summary\":\"...\"}"
        )
        try:
            raw = await self.llm_call(
                system="Lead-gen analyst. Output strict JSON. Only include real-looking organizations.",
                user=synth_prompt, max_tokens=900,
            )
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(raw)
        except Exception as e:
            log.warning("[VECTOR] Lead synthesis failed: %s", e)
            parsed = {"leads": [], "summary": "", "raw_hits": hits}

        return {"status": "ok", "agent": self.NAME, "query": query, **parsed}

    async def _site_audit(self, task: dict, session_id: str) -> dict:
        url = task.get("url", "").strip()
        focus = task.get("focus", "seo")
        if not url:
            return {"status": "error", "reason": "no url", "agent": self.NAME}

        approved, _ = await self.evaluate_action(
            "vector:site_audit", f"Audit {url} for {focus}",
            payload={"url": url[:300], "focus": focus}, session_id=session_id,
        )
        if not approved:
            return {"status": "blocked", "agent": self.NAME}

        fetch = await self.use_tool("http_fetch", {"url": url}, session_id=session_id)
        if not fetch.get("ok"):
            return {"status": "error", "agent": self.NAME, "reason": fetch.get("error")}

        page = _extract_page_summary(fetch["result"].get("body", ""))
        audit = await self.llm_call(
            system=f"Senior {focus} auditor. Be specific and concrete. No filler.",
            user=f"URL: {url}\nTitle: {page['title']}\nMeta: {page['description']}\n"
                 f"Headings: {page['headings']}\n\nGive 5 concrete improvements.",
            max_tokens=600,
        )
        return {"status": "ok", "agent": self.NAME, "url": url, "page": page, "audit": audit}

    async def _keyword_scan(self, task: dict, session_id: str) -> dict:
        topic = (task.get("content") or task.get("task") or "").strip()
        if not topic:
            return {"status": "error", "reason": "no topic", "agent": self.NAME}

        approved, _ = await self.evaluate_action(
            "vector:keyword_scan", f"Keyword scan: {topic[:200]}",
            payload={"topic": topic[:200]}, session_id=session_id,
        )
        if not approved:
            return {"status": "blocked", "agent": self.NAME}

        res = await self.use_tool("web_search", {"query": topic, "n": 8}, session_id=session_id)
        if not res.get("ok"):
            return {"status": "error", "agent": self.NAME, "reason": res.get("error")}
        snippets = " ".join(h.get("snippet", "") for h in res["result"].get("hits", []))

        synth = await self.llm_call(
            system="SEO keyword analyst. Output JSON with high-intent keyword opportunities.",
            user=f"Topic: {topic}\n\nMarket signal (search snippets):\n{snippets[:3000]}\n\n"
                 "Return JSON: {\"keywords\":[{\"phrase\":\"...\",\"intent\":\"info|nav|commercial|transactional\",\"why\":\"...\"}]}",
            max_tokens=700,
        )
        try:
            parsed = json.loads(synth.strip().lstrip("```json").lstrip("```").rstrip("```").strip())
        except Exception:
            parsed = {"keywords": [], "raw": synth[:1000]}
        return {"status": "ok", "agent": self.NAME, "topic": topic, **parsed}
