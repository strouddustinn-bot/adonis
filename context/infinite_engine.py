"""
context/infinite_engine.py
===========================
Infinite Context Engine — 4-tier hierarchical memory with QC compression.

Tier waterfall:
  L0 — Active window  (Redis, session TTL,  verbatim, ~10K tokens)
  L1 — Episode buffer (Redis, 7-day TTL,    QC Standard 50%, ~5K tokens)
  L2 — Session archive(Redis, 30-day TTL,   QC Ultra 90%,    ~2K tokens)
  L3 — Semantic vault (ChromaDB+Obsidian,   permanent,       ~0 until retrieved)

Overflow cascades down. Retrieval always searches up from L3 → L0.
Effective context: near-unlimited at minimal token cost.
"""
import os, json, logging, time
from dataclasses import dataclass, field
from typing import Optional
import asyncio
from compression.quantum_compress import QuantumCompressor, CompressionLevel

log = logging.getLogger("infinite_ctx")

L0_MAX_TURNS   = 50
L1_MAX_EPISODES= 20
L0_TTL         = 3600          # 1h session
L1_TTL         = 604800        # 7 days
L2_TTL         = 2592000       # 30 days
L3_TOP_K       = 5             # semantic chunks to retrieve per query

@dataclass
class ContextTurn:
    role:      str
    content:   str
    timestamp: float = field(default_factory=time.time)
    tokens:    int   = 0

@dataclass
class Episode:
    turns:    list[ContextTurn]
    summary:  str = ""
    session:  str = ""
    ts_start: float = 0.0
    ts_end:   float = 0.0

class InfiniteContextEngine:
    """
    Usage:
        engine = InfiniteContextEngine(qc, redis, chroma_client, obsidian)
        messages = await engine.build_context(session_id, new_user_message, token_budget=12000)
    """
    def __init__(self, qc: QuantumCompressor, redis_client, chroma_client=None, obsidian_bridge=None):
        self.qc     = qc
        self.redis  = redis_client
        self.chroma = chroma_client
        self.obs    = obsidian_bridge

    # ── Public API ───────────────────────────────────────────────────────────

    async def append_turn(self, session_id: str, role: str, content: str):
        """Add a new turn to L0. Trigger cascade if L0 overflows."""
        key = f"ctx:l0:{session_id}"
        turn = {"role": role, "content": content, "ts": time.time()}
        await self.redis.rpush(key, json.dumps(turn))
        await self.redis.expire(key, L0_TTL)

        turn_count = await self.redis.llen(key)
        if turn_count > L0_MAX_TURNS * 2:
            await self._cascade_l0_to_l1(session_id)

    async def build_context(
        self,
        session_id: str,
        new_message: str,
        token_budget: int = 12000,
        task_hint: str = ""
    ) -> list[dict]:
        """
        Assemble the optimal message list for an LLM call within token_budget.
        Returns OpenAI-style list: [{"role":"user","content":"..."}, ...]
        """
        messages = []
        remaining = token_budget

        # L3: Semantic retrieval (most relevant long-term knowledge first)
        l3_chunks = await self._retrieve_l3(new_message, task_hint)
        if l3_chunks:
            l3_text = "\n".join(l3_chunks)
            l3_tokens = self._estimate_tokens(l3_text)
            if l3_tokens <= remaining // 4:  # L3 gets at most 25% budget
                messages.append({"role": "user", "content": f"[Relevant context from memory]\n{l3_text}"})
                messages.append({"role": "assistant", "content": "Acknowledged. Proceeding with task."})
                remaining -= l3_tokens

        # L2: Compressed session archive
        l2 = await self._load_l2(session_id)
        if l2:
            l2_tokens = self._estimate_tokens(l2)
            alloc = min(l2_tokens, remaining // 4)
            messages.append({"role": "user", "content": f"[Session archive summary]\n{l2[:alloc*4]}"})
            messages.append({"role": "assistant", "content": "Got it."})
            remaining -= min(l2_tokens, alloc)

        # L1: Recent episode buffer
        l1 = await self._load_l1(session_id)
        if l1:
            l1_tokens = self._estimate_tokens(l1)
            alloc = min(l1_tokens, remaining // 3)
            messages.append({"role": "user", "content": f"[Recent episodes]\n{l1[:alloc*4]}"})
            messages.append({"role": "assistant", "content": "Got it."})
            remaining -= min(l1_tokens, alloc)

        # L0: Verbatim recent turns (fill remaining budget)
        l0_turns = await self._load_l0(session_id)
        for turn in l0_turns:
            t = self._estimate_tokens(turn["content"])
            if t > remaining: break
            messages.append({"role": turn["role"], "content": turn["content"]})
            remaining -= t

        # Append the new message
        messages.append({"role": "user", "content": new_message})
        return messages

    async def distil_to_l3(self, session_id: str, text: str, metadata: dict = None):
        """Permanently store knowledge in semantic vault (ChromaDB + Obsidian)."""
        if self.chroma:
            payload = await self.qc.compress(text, CompressionLevel.STANDARD)
            self.chroma.add(
                documents=[payload.expand(200)],
                metadatas=[{**(metadata or {}), "session": session_id, "hash": payload.raw_hash}],
                ids=[f"{session_id}:{payload.raw_hash}"]
            )
        if self.obs:
            date = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
            try:
                existing = await self.obs.read_note(f"MEMORY/semantic/{date}.md") or ""
                await self.obs.write_note(f"MEMORY/semantic/{date}.md", existing + f"\n---\n{text[:500]}\n")
            except: pass

    # ── Internal tier ops ────────────────────────────────────────────────────

    async def _load_l0(self, session_id: str) -> list[dict]:
        raw = await self.redis.lrange(f"ctx:l0:{session_id}", -L0_MAX_TURNS, -1)
        return [json.loads(r) for r in raw]

    async def _load_l1(self, session_id: str) -> str:
        raw = await self.redis.get(f"ctx:l1:{session_id}")
        return raw.decode() if raw else ""

    async def _load_l2(self, session_id: str) -> str:
        raw = await self.redis.get(f"ctx:l2:{session_id}")
        return raw.decode() if raw else ""

    async def _cascade_l0_to_l1(self, session_id: str):
        """Compress oldest L0 turns into L1 episode buffer."""
        key_l0 = f"ctx:l0:{session_id}"
        key_l1 = f"ctx:l1:{session_id}"
        raw_turns = await self.redis.lrange(key_l0, 0, L0_MAX_TURNS - 1)
        if not raw_turns: return

        text = " ".join(json.loads(t)["content"] for t in raw_turns)
        payload = await self.qc.compress(text, CompressionLevel.STANDARD)
        compressed = payload.expand(300)

        existing = await self.redis.get(key_l1) or b""
        merged = existing.decode() + f" | {compressed}"
        episode_count = merged.count(" | ")
        if episode_count > L1_MAX_EPISODES:
            await self._cascade_l1_to_l2(session_id, merged)
            merged = compressed

        await self.redis.setex(key_l1, L1_TTL, merged)
        await self.redis.ltrim(key_l0, L0_MAX_TURNS, -1)
        log.debug(f"[CTX] L0→L1 cascade for {session_id}")

    async def _cascade_l1_to_l2(self, session_id: str, l1_text: str):
        """Ultra-compress L1 into L2 session archive."""
        key_l2 = f"ctx:l2:{session_id}"
        payload = await self.qc.compress(l1_text, CompressionLevel.ULTRA)
        existing = await self.redis.get(key_l2) or b""
        merged = (existing.decode() + " " + payload.summary).strip()
        await self.redis.setex(key_l2, L2_TTL, merged)
        log.debug(f"[CTX] L1→L2 cascade for {session_id}")

    async def _retrieve_l3(self, query: str, hint: str = "") -> list[str]:
        """Semantic search in ChromaDB."""
        if not self.chroma: return []
        try:
            results = self.chroma.query(
                query_texts=[query + " " + hint],
                n_results=L3_TOP_K
            )
            return results.get("documents", [[]])[0]
        except Exception as e:
            log.warning(f"[CTX] L3 retrieval failed: {e}")
            return []

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)
