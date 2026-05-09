"""
compression/quantum_compress.py
================================
Quantum Compression Engine — Qwen3-inspired dense token packing.

Multi-stage pipeline:
  1. Semantic distillation  — LLM extracts subject/predicate/object atoms
  2. Anchor encoding        — strip stop-words, filler, redundant phrasing
  3. Relationship graph     — causal links stored as edge list
  4. Tiered expansion       — one payload, expand to any token budget at retrieval

One stored atom expands to 20 tokens (ultra) or 200 tokens (full) on demand.
All results Redis-cached with configurable TTL to avoid re-processing.
"""
import os, json, hashlib, logging, re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import asyncio

log = logging.getLogger("quantum_compress")

STOP_WORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "to","of","in","for","on","with","at","by","from","as","into","through",
    "that","this","these","those","it","its","they","their","there","then",
}

class CompressionLevel(Enum):
    MINIMAL    = "minimal"     # ~20% reduction  — preserves full atoms
    STANDARD   = "standard"    # ~50% reduction  — atoms + key relationships
    AGGRESSIVE = "aggressive"  # ~75% reduction  — summary atoms only
    ULTRA      = "ultra"       # ~90% reduction  — single-sentence essence

@dataclass
class CompressedPayload:
    level:       CompressionLevel
    atoms:       list[str]        # semantic triples
    summary:     str              # ultra-compressed single sentence
    edges:       list[tuple]      # (subject, relation, object) causal graph
    raw_hash:    str              # SHA256 of original, used as cache key
    token_est:   int              # estimated tokens in this payload
    expansion_budget: dict = field(default_factory=dict)  # budget_tokens -> text

    def expand(self, token_budget: int = 500) -> str:
        """Return the right fidelity for the given token budget."""
        if token_budget <= 30:
            return self.summary
        if token_budget <= 100:
            return " | ".join(self.atoms[:5]) if self.atoms else self.summary
        if token_budget <= 300:
            return " | ".join(self.atoms)
        # Full: reconstruct prose from atoms + edges
        parts = [" | ".join(self.atoms)]
        if self.edges:
            edge_text = "; ".join(f"{s} -{r}-> {o}" for s,r,o in self.edges[:10])
            parts.append(f"[relations: {edge_text}]")
        return " ".join(parts)


class QuantumCompressor:
    """
    Usage:
        qc = QuantumCompressor(anthropic_client, redis_client)
        payload = await qc.compress(text, level=CompressionLevel.STANDARD)
        context_snippet = payload.expand(token_budget=150)
    """
    CACHE_TTL = 60 * 60 * 24 * 7  # 7 days

    def __init__(self, anthropic_client, redis_client):
        self.llm   = anthropic_client
        self.redis = redis_client

    async def compress(self, text: str, level: CompressionLevel = CompressionLevel.STANDARD) -> CompressedPayload:
        raw_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_key = f"qc:{level.value}:{raw_hash}"

        cached = await self.redis.get(cache_key)
        if cached:
            log.debug(f"[QC] Cache hit {cache_key}")
            data = json.loads(cached)
            return CompressedPayload(**{**data, "level": CompressionLevel(data["level"])})

        log.debug(f"[QC] Compressing {len(text)} chars at level {level.value}")

        # Stage 1: Semantic distillation via LLM
        atoms, edges = await self._distill(text, level)

        # Stage 2: Anchor encoding (local, no LLM)
        anchored = self._anchor_encode(text)

        # Stage 3: Ultra summary
        summary = await self._summarise(text) if level in (CompressionLevel.AGGRESSIVE, CompressionLevel.ULTRA) else atoms[0] if atoms else anchored[:80]

        # Stage 4: Estimate tokens
        payload_text = " | ".join(atoms)
        token_est = max(1, len(payload_text.split()))

        payload = CompressedPayload(
            level=level, atoms=atoms, summary=summary,
            edges=edges, raw_hash=raw_hash, token_est=token_est
        )

        await self.redis.setex(cache_key, self.CACHE_TTL, json.dumps({
            "level": level.value, "atoms": atoms, "summary": summary,
            "edges": edges, "raw_hash": raw_hash, "token_est": token_est,
            "expansion_budget": {}
        }))
        return payload

    async def compress_batch(self, texts: list[str], level=CompressionLevel.STANDARD) -> list[CompressedPayload]:
        return await asyncio.gather(*[self.compress(t, level) for t in texts])

    def _anchor_encode(self, text: str) -> str:
        """Strip stop-words and punctuation. ~30% of original."""
        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        anchors = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        return " ".join(anchors)

    async def _distill(self, text: str, level: CompressionLevel) -> tuple[list[str], list[tuple]]:
        depth = {"minimal":"detailed","standard":"concise","aggressive":"brief","ultra":"minimal"}[level.value]
        prompt = f"""Extract semantic atoms from this text as {depth} subject-predicate-object triples.
Also extract up to 5 causal relationships as (subject, relation, object) tuples.
Return ONLY JSON: {{"atoms":["subj -> pred -> obj",...], "edges":[["s","r","o"],...]}}

Text: {text[:2000]}"""
        try:
            r = await self.llm.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=600,
                messages=[{"role":"user","content":prompt}])
            d = json.loads(r.content[0].text.strip())
            atoms = [str(a) for a in d.get("atoms",[])][:20]
            edges = [tuple(e) for e in d.get("edges",[]) if len(e)==3][:10]
            return atoms, edges
        except Exception as e:
            log.warning(f"[QC] Distill failed: {e}")
            # Fallback: naive sentence splitting
            sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
            return sentences[:8], []

    async def _summarise(self, text: str) -> str:
        try:
            r = await self.llm.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=60,
                messages=[{"role":"user","content":f"Summarise in one dense sentence (max 20 words): {text[:1000]}"}])
            return r.content[0].text.strip()
        except:
            words = text.split()
            return " ".join(words[:20]) + ("..." if len(words)>20 else "")
