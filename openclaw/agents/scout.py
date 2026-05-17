"""
openclaw/agents/scout.py
=========================
SCOUT — Research & Discovery specialist.

Pulls evidence from the open web (and arXiv for technical queries),
distills it into a structured claims+sources report, and optionally
stores the distilled findings in the L3 semantic vault so future tasks
can retrieve them.

Task shape:
  {type:"research", content:"...", arxiv?:bool, n?:int}
Atlas-dispatched task (no `type`) is treated as research on `content`/`task`.
"""
import asyncio
import json
import logging

from openclaw.base_agent import BaseAgent
from openclaw.contracts import Contract, ContractIn, ContractOut

log = logging.getLogger("scout")

TECH_HINTS = {"paper","arxiv","model","algorithm","benchmark","architecture","quantum","transformer","theory"}


class ScoutResearchIn(ContractIn):
    content: str
    arxiv:   bool = False
    n:       int  = 5


class ScoutResearchOut(ContractOut):
    query:    str
    summary:  str
    claims:   list = []
    n_hits:   int  = 0
    n_papers: int  = 0


class ScoutAgent(BaseAgent):
    NAME    = "scout"
    DOMAINS = ["research", "analysis", "arxiv", "investigate", "study", "sources", "news"]
    # Reads the open web, persists findings into MEMORY/.
    CAPABILITIES = frozenset({
        "net:http_get", "net:web_search", "net:arxiv",
        "vault:read:MEMORY/*", "vault:write:MEMORY/*",
        "time:read",
    })
    CONTRACTS = [
        Contract("scout.research", "scout", "research",
                 "Research a topic on the open web and (optionally) arXiv; return claims + sources.",
                 ScoutResearchIn, ScoutResearchOut, timeout_s=90),
    ]

    async def handle(self, task: dict, session_id: str) -> dict:
        query = (task.get("content") or task.get("task") or task.get("goal") or "").strip()
        if not query:
            return {"status": "error", "reason": "no query", "agent": self.NAME}

        n = max(3, min(int(task.get("n", 5)), 10))
        use_arxiv = bool(task.get("arxiv")) or any(h in query.lower() for h in TECH_HINTS)

        approved, _ = await self.evaluate_action(
            action_type="scout:research",
            description=f"Research the open web for: {query[:200]}",
            payload={"query": query[:300], "arxiv": use_arxiv},
            session_id=session_id,
        )
        if not approved:
            return {"status": "blocked", "agent": self.NAME, "reason": "fuse blocked research"}

        calls = [self.use_tool("web_search", {"query": query, "n": n}, session_id=session_id)]
        if use_arxiv:
            calls.append(self.use_tool("arxiv_search", {"query": query, "n": min(n, 5)}, session_id=session_id))

        results = await asyncio.gather(*calls)
        web = results[0]
        arxiv = results[1] if use_arxiv else {"ok": True, "result": {"papers": []}}

        if not web.get("ok"):
            return {"status": "error", "agent": self.NAME, "reason": f"web_search failed: {web.get('error')}"}

        hits   = web["result"].get("hits", [])
        papers = arxiv.get("result", {}).get("papers", []) if arxiv.get("ok") else []

        sources_blob = "\n".join(
            f"- [{h['title']}]({h['url']}) — {h['snippet']}" for h in hits
        )
        if papers:
            sources_blob += "\n\nArxiv:\n" + "\n".join(
                f"- {p['title']} ({p['published']}) — {p['url']}\n  {p['summary'][:300]}" for p in papers
            )

        distill_prompt = (
            f"Question: {query}\n\nEvidence:\n{sources_blob}\n\n"
            "Return ONLY JSON: "
            "{\"summary\":\"...\",\"claims\":[{\"claim\":\"...\",\"source_url\":\"...\",\"confidence\":0..1},...]}"
        )
        try:
            raw = await self.llm_call(
                system="Research synthesis engine. Output strict JSON. Cite source_url for every claim.",
                user=distill_prompt,
                max_tokens=800,
            )
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(raw)
        except Exception as e:
            log.warning("[SCOUT] Distill failed, returning raw: %s", e)
            parsed = {"summary": "", "claims": [], "raw_sources": sources_blob[:2000]}

        # Quietly distil into L3 for future retrieval.
        ctx = self.governor.context
        try:
            await ctx.distil_to_l3(session_id, json.dumps(parsed)[:4000],
                                   metadata={"agent": self.NAME, "query": query[:200]})
        except Exception as e:
            log.debug("[SCOUT] L3 distil skipped: %s", e)

        return {
            "status":   "ok",
            "agent":    self.NAME,
            "query":    query,
            "summary":  parsed.get("summary", ""),
            "claims":   parsed.get("claims", []),
            "n_hits":   len(hits),
            "n_papers": len(papers),
        }
