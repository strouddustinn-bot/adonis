"""
openclaw/agents/data.py
========================
DATA — structured-data analysis specialist.

Scout researches the open web; Forge writes; nobody owned *tabular* analysis.
DATA fills that gap: it parses CSV or JSON rows (stdlib only — no pandas) and
computes deterministic profiles and aggregations, then optionally layers an
LLM interpretation on top.

Task shapes:
  {type:"profile",    data:"<csv|json>", format?:"auto|csv|json"}
  {type:"data_query", data:"<csv|json>", question:"...", format?:"auto"}
Atlas-dispatched task (no `type`) is treated as a profile of `content`.
"""
import csv
import io
import json
import logging
import statistics
from collections import Counter
from typing import Optional

from openclaw.base_agent import BaseAgent
from openclaw.contracts import Contract, ContractIn, ContractOut

log = logging.getLogger("data")


class DataProfileIn(ContractIn):
    data:   str
    format: Optional[str] = "auto"   # auto | csv | json


class DataProfileOut(ContractOut):
    rows:    int  = 0
    columns: list = []
    stats:   dict = {}


class DataQueryIn(ContractIn):
    data:     str
    question: str
    format:   Optional[str] = "auto"


class DataQueryOut(ContractOut):
    answer:   Optional[str]  = None
    computed: Optional[dict] = None


class DataAgent(BaseAgent):
    NAME    = "data"
    DOMAINS = ["data", "csv", "table", "dataframe", "sql", "aggregate",
               "stats", "column", "dataset", "query", "rows"]
    # Reads structured input passed in the task; only needs vault for reference.
    CAPABILITIES = frozenset({"vault:read:MEMORY/*", "time:read"})
    CONTRACTS = [
        Contract("data.profile", "data", "profile",
                 "Parse CSV/JSON rows and return schema + per-column statistics.",
                 DataProfileIn, DataProfileOut, timeout_s=30),
        Contract("data.query", "data", "data_query",
                 "Answer a question about tabular data, computing aggregates where possible.",
                 DataQueryIn, DataQueryOut, timeout_s=45),
    ]

    async def handle(self, task: dict, session_id: str) -> dict:
        task_type = task.get("type", "profile")
        if task_type == "data_query":
            return await self._query(task, session_id)
        return await self._profile(task, session_id)

    # ── parsing + stats (pure, deterministic) ────────────────────────────────
    def _parse(self, data: str, fmt: str = "auto") -> list[dict]:
        data = (data or "").strip()
        if not data:
            return []
        if fmt == "json" or (fmt == "auto" and data[:1] in "[{"):
            obj = json.loads(data)
            if isinstance(obj, dict):
                obj = [obj]
            return [dict(r) for r in obj]
        reader = csv.DictReader(io.StringIO(data))
        return [dict(r) for r in reader]

    @staticmethod
    def _numeric(values) -> list[float]:
        out = []
        for v in values:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
        return out

    def _profile_rows(self, rows: list[dict]) -> tuple[list, dict]:
        cols: list = []
        for r in rows:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        stats: dict = {}
        for c in cols:
            vals = [r.get(c) for r in rows]
            non_null = [v for v in vals if v not in (None, "")]
            nums = self._numeric(vals)
            col_stat: dict = {"count": len(non_null),
                              "distinct": len(set(map(str, non_null)))}
            # Treat a column as numeric only if most non-null values parse.
            if nums and len(nums) >= max(1, len(non_null) // 2):
                col_stat.update({
                    "type": "numeric",
                    "min":  min(nums),
                    "max":  max(nums),
                    "mean": round(statistics.fmean(nums), 4),
                    "sum":  round(sum(nums), 4),
                })
            else:
                col_stat["type"] = "categorical"
                if non_null:
                    top, n = Counter(map(str, non_null)).most_common(1)[0]
                    col_stat["top"] = top
                    col_stat["top_count"] = n
            stats[c] = col_stat
        return cols, stats

    def _try_aggregate(self, question: str, stats: dict) -> dict:
        """Best-effort deterministic aggregate when the question names a column
        and an operation. Returns {} when nothing clean matches."""
        q = question.lower()
        target = next((c for c in stats if c.lower() in q), None)
        if not target:
            return {}
        col = stats[target]
        if col.get("type") != "numeric":
            return {"column": target,
                    **{k: col[k] for k in ("count", "distinct", "top") if k in col}}
        if any(w in q for w in ("average", "mean", "avg")):
            op = "mean"
        elif any(w in q for w in ("sum", "total")):
            op = "sum"
        elif any(w in q for w in ("max", "highest", "largest", "most")):
            op = "max"
        elif any(w in q for w in ("min", "lowest", "smallest", "least")):
            op = "min"
        else:
            return {"column": target, "stats": col}
        return {"column": target, "op": op, "value": col.get(op)}

    # ── handlers ─────────────────────────────────────────────────────────────
    async def _profile(self, task: dict, session_id: str) -> dict:
        raw = task.get("data") or task.get("content") or ""
        try:
            rows = self._parse(raw, task.get("format", "auto"))
        except Exception as e:
            return {"status": "error", "reason": f"parse failed: {e}", "agent": self.NAME}
        cols, stats = self._profile_rows(rows)
        return {"status": "ok", "agent": self.NAME,
                "rows": len(rows), "columns": cols, "stats": stats}

    async def _query(self, task: dict, session_id: str) -> dict:
        question = (task.get("question") or "").strip()
        raw = task.get("data") or task.get("content") or ""
        if not question:
            return {"status": "error", "reason": "no question", "agent": self.NAME}
        try:
            rows = self._parse(raw, task.get("format", "auto"))
        except Exception as e:
            return {"status": "error", "reason": f"parse failed: {e}", "agent": self.NAME}

        _, stats = self._profile_rows(rows)
        computed = self._try_aggregate(question, stats)

        approved, _ = await self.evaluate_action(
            action_type="data:query",
            description=f"Analyse {len(rows)} rows: {question[:160]}",
            payload={"rows": len(rows)}, session_id=session_id,
        )
        if not approved:
            return {"status": "blocked", "agent": self.NAME}

        try:
            answer = await self.llm_call(
                system=("Data analyst. Answer using ONLY the provided profile and "
                        "any computed aggregate. Be precise and quantitative."),
                user=(f"Question: {question}\n\n"
                      f"Profile: {json.dumps({'rows': len(rows), 'stats': stats})[:2500]}\n\n"
                      f"Computed: {json.dumps(computed)[:800]}"),
                max_tokens=400,
            )
        except Exception as e:
            log.warning("[DATA] interpretation failed: %s", e)
            answer = None

        return {"status": "ok", "agent": self.NAME,
                "answer": answer, "computed": computed}
