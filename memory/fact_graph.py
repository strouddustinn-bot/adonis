"""
memory/fact_graph.py
=====================
Conflict-Resolved Vector Graph — structured agentic memory.

Three layers behind a single async API:

  L1 short-term   in-memory LRU of the last N facts per entity
  L2 working      SQLite, JSON-typed metadata, indexed on (entity, attribute)
  L3 long-term    LanceDB embeddings for semantic recall (optional;
                  graph still works without it)

Conflict resolution: when an incoming fact has the same (entity, attribute)
as an existing active fact but a different value, the new and old are
scored on `confidence + recency_bonus`. The higher score wins; the loser
is marked `superseded`. If the winner's confidence is below the flag
threshold (default 0.55), the new fact is also marked `conflict` so a
human reviewer can resolve it later via `resolve_conflict()`.

Public API (all async-safe, sqlite serialised through a single connection):

    fg = FactGraph(db_path="/vault/facts.db", vector_path="/vault/facts.lance")
    await fg.startup()
    await fg.add(entity, attribute, value, source_agent=..., session_id=...)
    await fg.query(entity=..., attribute=..., text=..., n=10)
    await fg.resolve_conflict(fact_id, keep="new"|"old")
    await fg.recent(n=20)
    await fg.conflicts(n=20)
    await fg.stats()
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("fact_graph")

# Heuristic tuning knobs — tweak in code, not env, since these define the
# behaviour of conflict scoring and shouldn't drift across instances.
RECENCY_BONUS_PER_DAY = 0.01   # fresher facts get a slight edge per day
FLAG_THRESHOLD        = 0.55   # winning conflicts below this still flag for review
REINFORCE_DELTA       = 0.05   # repeated identical observations bump confidence
MAX_CONFIDENCE        = 0.99   # never quite certain

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  entity          TEXT NOT NULL,
  attribute       TEXT NOT NULL,
  value           TEXT NOT NULL,
  value_kind      TEXT DEFAULT 'string',
  metadata        TEXT,                              -- JSON blob
  confidence      REAL DEFAULT 0.7,
  status          TEXT DEFAULT 'active',             -- active|superseded|conflict
  source_agent    TEXT,
  session_id      TEXT,
  trace_id        TEXT,
  observed_at     TEXT NOT NULL,
  superseded_by   INTEGER,
  FOREIGN KEY (superseded_by) REFERENCES facts(id)
);
CREATE INDEX IF NOT EXISTS idx_facts_entity_attr ON facts(entity, attribute, status);
CREATE INDEX IF NOT EXISTS idx_facts_status      ON facts(status, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_entity      ON facts(entity, status);
"""


@dataclass
class Fact:
    id:            int
    entity:        str
    attribute:     str
    value:         str
    confidence:    float
    status:        str
    source_agent:  str
    session_id:    str
    trace_id:      str
    observed_at:   str
    metadata:      dict = field(default_factory=dict)
    superseded_by: Optional[int] = None


def _row_to_fact(r: sqlite3.Row) -> Fact:
    meta = {}
    try:
        if r["metadata"]: meta = json.loads(r["metadata"])
    except Exception:
        pass
    return Fact(
        id=r["id"], entity=r["entity"], attribute=r["attribute"], value=r["value"],
        confidence=r["confidence"], status=r["status"],
        source_agent=r["source_agent"] or "", session_id=r["session_id"] or "",
        trace_id=r["trace_id"] or "", observed_at=r["observed_at"],
        metadata=meta, superseded_by=r["superseded_by"],
    )


class _LRU:
    """Simple bounded LRU for hot fact lookups by (entity, attribute)."""
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._d: "OrderedDict[tuple[str,str], Fact]" = OrderedDict()

    def get(self, key) -> Optional[Fact]:
        if key not in self._d: return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value: Fact) -> None:
        if key in self._d: self._d.move_to_end(key)
        self._d[key] = value
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def invalidate(self, key) -> None:
        self._d.pop(key, None)


class _VectorIndex:
    """LanceDB-backed semantic search over (entity + attribute + value) text.
    Silently disables itself if lancedb or its embedding deps aren't available."""
    def __init__(self, path: str):
        self.path = path
        self.table = None
        self._ok = False
        try:
            import lancedb
            from lancedb.embeddings import get_registry
            registry = get_registry()
            self._embedder = registry.get("sentence-transformers").create(name="all-MiniLM-L6-v2")
            self._db = lancedb.connect(path)
            try:
                self.table = self._db.open_table("facts")
            except Exception:
                self.table = None  # created lazily on first add()
            self._ok = True
            log.info("[FG] LanceDB vector index ready at %s", path)
        except Exception as e:
            log.warning("[FG] vector index disabled: %s", e)

    def available(self) -> bool: return self._ok

    def _ensure_table(self):
        if self.table is not None or not self._ok: return
        from lancedb.pydantic import LanceModel, Vector
        emb = self._embedder
        # Build schema by extracting embedding dim once
        class _Row(LanceModel):
            fact_id: int
            entity:  str
            text:    str = emb.SourceField()
            vector:  Vector(emb.ndims()) = emb.VectorField()
        self.table = self._db.create_table("facts", schema=_Row, exist_ok=True)

    def add(self, fact_id: int, entity: str, attribute: str, value: str) -> None:
        if not self._ok: return
        try:
            self._ensure_table()
            self.table.add([{
                "fact_id": fact_id,
                "entity":  entity,
                "text":    f"{entity} {attribute} {value}",
            }])
        except Exception as e:
            log.debug("[FG] vector add failed: %s", e)

    def search(self, query: str, n: int = 10) -> list[int]:
        if not self._ok or self.table is None: return []
        try:
            rows = self.table.search(query).limit(n).to_list()
            return [int(r["fact_id"]) for r in rows]
        except Exception as e:
            log.debug("[FG] vector search failed: %s", e)
            return []


class FactGraph:
    """Top-level fact graph. All public methods are async to fit the rest of
    the runtime, but the underlying SQLite work is fast and runs in the event
    loop directly under a single lock — fine for the throughput we expect."""

    def __init__(self, db_path: str, vector_path: Optional[str] = None, lru_capacity: int = 256):
        self.db_path     = db_path
        self.vector_path = vector_path
        self._lock       = asyncio.Lock()
        self._lru        = _LRU(lru_capacity)
        self._conn:   Optional[sqlite3.Connection] = None
        self._vec:    Optional[_VectorIndex]       = None

    async def startup(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        for stmt in SCHEMA.strip().split(";"):
            if stmt.strip(): conn.execute(stmt)
        self._conn = conn
        log.info("[FG] SQLite ready at %s", self.db_path)
        if self.vector_path:
            self._vec = _VectorIndex(self.vector_path)

    async def shutdown(self) -> None:
        if self._conn:
            try: self._conn.close()
            except Exception: pass
            self._conn = None

    # ── public API ───────────────────────────────────────────────────────

    async def add(
        self,
        entity:        str,
        attribute:     str,
        value:         Any,
        *,
        confidence:    float = 0.7,
        source_agent:  str = "?",
        session_id:    str = "",
        trace_id:      str = "",
        metadata:      Optional[dict] = None,
    ) -> dict:
        """Insert or reinforce a fact, resolving conflicts in place.

        Returns a dict describing what happened:
          {"status": "new"|"reinforced"|"superseded"|"conflict_flagged",
           "id": int, "prior_id": optional int, "score_new": float, "score_old": float}
        """
        entity    = entity.strip().lower()
        attribute = attribute.strip().lower()
        value_str = _coerce_value(value)
        if not (entity and attribute and value_str):
            return {"status": "rejected", "reason": "empty entity/attribute/value"}

        meta_blob = json.dumps(metadata or {}, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()

        async with self._lock:
            existing = self._fetch_active(entity, attribute)

            if existing is None:
                new_id = self._insert(entity, attribute, value_str, confidence,
                                      "active", source_agent, session_id, trace_id, meta_blob, now)
                self._lru.put((entity, attribute), self._fetch_by_id(new_id))
                if self._vec: self._vec.add(new_id, entity, attribute, value_str)
                return {"status": "new", "id": new_id}

            if existing.value == value_str:
                new_conf = min(MAX_CONFIDENCE, existing.confidence + REINFORCE_DELTA)
                self._conn.execute(
                    "UPDATE facts SET confidence = ?, observed_at = ? WHERE id = ?",
                    (new_conf, now, existing.id),
                )
                self._lru.invalidate((entity, attribute))
                return {"status": "reinforced", "id": existing.id, "confidence": new_conf}

            # Conflict — score both and pick a winner.
            score_old = existing.confidence + _recency_bonus(existing.observed_at)
            score_new = confidence + _recency_bonus(now)
            new_id = self._insert(entity, attribute, value_str, confidence,
                                  "active", source_agent, session_id, trace_id, meta_blob, now)
            if score_new >= score_old:
                self._conn.execute(
                    "UPDATE facts SET status = 'superseded', superseded_by = ? WHERE id = ?",
                    (new_id, existing.id),
                )
                resolved_status = "superseded"
                if confidence < FLAG_THRESHOLD:
                    self._conn.execute("UPDATE facts SET status = 'conflict' WHERE id = ?", (new_id,))
                    resolved_status = "conflict_flagged"
            else:
                # incoming loses; record but flag for review
                self._conn.execute("UPDATE facts SET status = 'conflict' WHERE id = ?", (new_id,))
                resolved_status = "conflict_flagged"

            self._lru.invalidate((entity, attribute))
            if self._vec: self._vec.add(new_id, entity, attribute, value_str)
            return {
                "status":    resolved_status,
                "id":        new_id,
                "prior_id":  existing.id,
                "score_new": round(score_new, 3),
                "score_old": round(score_old, 3),
            }

    async def query(
        self,
        *,
        entity:    Optional[str] = None,
        attribute: Optional[str] = None,
        text:      Optional[str] = None,
        status:    str = "active",
        n:         int = 20,
    ) -> list[dict]:
        """Look facts up by any combination of:
            - entity (+ optional attribute) — exact match
            - text — semantic search via LanceDB if available, else substring
        """
        entity = entity.strip().lower() if entity else None
        attribute = attribute.strip().lower() if attribute else None
        async with self._lock:
            ids: list[int] = []
            if text and self._vec and self._vec.available():
                ids = self._vec.search(text, n=n)
            rows: list[Fact] = []

            if entity and attribute:
                hit = self._lru.get((entity, attribute))
                if hit and hit.status == status:
                    return [_to_dict(hit)]
                rows = self._fetch("entity = ? AND attribute = ? AND status = ?",
                                   (entity, attribute, status), n)
            elif entity:
                rows = self._fetch("entity = ? AND status = ?", (entity, status), n)
            elif ids:
                placeholders = ",".join("?" * len(ids))
                rows = self._fetch(f"id IN ({placeholders}) AND status = ?", (*ids, status), n)
            elif text:
                like = f"%{text.lower()}%"
                rows = self._fetch(
                    "(LOWER(entity) LIKE ? OR LOWER(attribute) LIKE ? OR LOWER(value) LIKE ?) AND status = ?",
                    (like, like, like, status), n,
                )
            else:
                rows = self._fetch("status = ?", (status,), n, order="observed_at DESC")

        return [_to_dict(r) for r in rows]

    async def resolve_conflict(self, fact_id: int, *, keep: str = "new") -> dict:
        """Manually settle a flagged conflict.

        `keep`:  "new" - keep this fact active, mark the conflicting partner superseded.
                 "old" - drop this fact (status=superseded), reactivate the prior.
                 "both"- demote both to 'superseded' (let the user re-state).
        """
        async with self._lock:
            target = self._fetch_by_id(fact_id)
            if not target:
                return {"status": "not_found", "id": fact_id}
            partner = self._fetch_partner(target)
            if keep == "new":
                self._conn.execute("UPDATE facts SET status='active' WHERE id=?", (target.id,))
                if partner:
                    self._conn.execute(
                        "UPDATE facts SET status='superseded', superseded_by=? WHERE id=?",
                        (target.id, partner.id),
                    )
            elif keep == "old":
                self._conn.execute("UPDATE facts SET status='superseded' WHERE id=?", (target.id,))
                if partner:
                    self._conn.execute(
                        "UPDATE facts SET status='active', superseded_by=NULL WHERE id=?",
                        (partner.id,),
                    )
            elif keep == "both":
                self._conn.execute("UPDATE facts SET status='superseded' WHERE id=?", (target.id,))
                if partner:
                    self._conn.execute("UPDATE facts SET status='superseded' WHERE id=?", (partner.id,))
            else:
                return {"status": "bad_arg", "keep": keep}
            self._lru.invalidate((target.entity, target.attribute))
        return {"status": "resolved", "id": fact_id, "kept": keep}

    async def recent(self, n: int = 20) -> list[dict]:
        async with self._lock:
            rows = self._fetch("status = 'active'", (), n, order="observed_at DESC")
        return [_to_dict(r) for r in rows]

    async def conflicts(self, n: int = 50) -> list[dict]:
        async with self._lock:
            rows = self._fetch("status = 'conflict'", (), n, order="observed_at DESC")
        return [_to_dict(r) for r in rows]

    async def stats(self) -> dict:
        async with self._lock:
            cur = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM facts GROUP BY status"
            )
            counts = {r["status"]: r["n"] for r in cur.fetchall()}
            total_entities = self._conn.execute(
                "SELECT COUNT(DISTINCT entity) AS n FROM facts"
            ).fetchone()["n"]
        return {
            "counts": counts,
            "total_facts": sum(counts.values()),
            "total_entities": total_entities,
            "vector_index": "lancedb" if (self._vec and self._vec.available()) else "disabled",
            "lru_size": len(self._lru._d),
        }

    # ── internals ────────────────────────────────────────────────────────

    def _insert(self, entity, attribute, value, confidence, status,
                source_agent, session_id, trace_id, metadata_blob, now) -> int:
        cur = self._conn.execute(
            "INSERT INTO facts(entity, attribute, value, metadata, confidence, status, "
            "source_agent, session_id, trace_id, observed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entity, attribute, value, metadata_blob, confidence, status,
             source_agent, session_id, trace_id, now),
        )
        return int(cur.lastrowid)

    def _fetch_active(self, entity: str, attribute: str) -> Optional[Fact]:
        cached = self._lru.get((entity, attribute))
        if cached and cached.status == "active":
            return cached
        cur = self._conn.execute(
            "SELECT * FROM facts WHERE entity=? AND attribute=? AND status='active' "
            "ORDER BY observed_at DESC LIMIT 1",
            (entity, attribute),
        )
        r = cur.fetchone()
        if r is None: return None
        f = _row_to_fact(r); self._lru.put((entity, attribute), f); return f

    def _fetch_by_id(self, fact_id: int) -> Optional[Fact]:
        r = self._conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return _row_to_fact(r) if r else None

    def _fetch_partner(self, fact: Fact) -> Optional[Fact]:
        if fact.superseded_by:
            return self._fetch_by_id(fact.superseded_by)
        r = self._conn.execute(
            "SELECT * FROM facts WHERE entity=? AND attribute=? AND id != ? "
            "AND status IN ('active','superseded') ORDER BY observed_at DESC LIMIT 1",
            (fact.entity, fact.attribute, fact.id),
        ).fetchone()
        return _row_to_fact(r) if r else None

    def _fetch(self, where: str, params: tuple, n: int, order: str = "observed_at DESC") -> list[Fact]:
        sql = f"SELECT * FROM facts WHERE {where} ORDER BY {order} LIMIT ?"
        cur = self._conn.execute(sql, (*params, n))
        return [_row_to_fact(r) for r in cur.fetchall()]


def _coerce_value(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, (dict, list)): return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def _recency_bonus(iso_ts: str) -> float:
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        days = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0)
        # Newer facts get a fractional bonus; older facts contribute 0.
        return max(0.0, RECENCY_BONUS_PER_DAY * (30.0 - days))
    except Exception:
        return 0.0


def _to_dict(f: Fact) -> dict:
    d = asdict(f)
    return d
