"""
memory/extractor.py
====================
LLM-driven structured fact extraction.

Converts conversational/task text into a list of
{entity, attribute, value, confidence} candidates suitable for
FactGraph.add(). Uses a Haiku-class model with strict JSON output, then
filters for sanity before returning.

Designed to be called fire-and-forget (asyncio.create_task) after agent
wins so it never blocks the user-facing response path.
"""
import json
import logging
import os
import re
from dataclasses import dataclass

log = logging.getLogger("extractor")

EXTRACT_MODEL = os.getenv("ADONIS_EXTRACT_MODEL", "claude-haiku-4-5-20251001")
MAX_FACTS_PER_CALL = 8


@dataclass
class FactCandidate:
    entity:     str
    attribute:  str
    value:      str
    confidence: float = 0.7


_PROMPT = """Extract durable facts from the text below. A durable fact is a
specific, reusable piece of information about a named entity (a person, place,
project, system, preference, deadline, decision).

Rules:
- Output STRICT JSON only, no prose, no markdown fences.
- Use lowercase snake_case for `attribute`.
- Skip everything that is task-local or transient (status messages, retry
  counts, intermediate scratch).
- Confidence: 0.9 if explicitly stated by the user; 0.7 if asserted by an
  agent's research; 0.5 if inferred.
- At most %d facts per call.

Output schema:
{"facts":[{"entity":"...","attribute":"...","value":"...","confidence":0.0}]}

Text:
%s
"""


async def extract_facts(text: str, llm) -> list[FactCandidate]:
    """Run one extraction round. Returns [] on any failure — never raises."""
    if not text or len(text.strip()) < 8:
        return []
    prompt = _PROMPT % (MAX_FACTS_PER_CALL, text[:4000])
    try:
        r = await llm.messages.create(
            model=EXTRACT_MODEL, max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = r.content[0].text.strip() if r.content else ""
    except Exception as e:
        log.debug("extract LLM call failed: %s", e)
        return []

    parsed = _parse_json(raw)
    if not parsed or "facts" not in parsed:
        return []

    out: list[FactCandidate] = []
    for f in parsed.get("facts", [])[:MAX_FACTS_PER_CALL]:
        try:
            entity    = str(f.get("entity", "")).strip()
            attribute = str(f.get("attribute", "")).strip()
            value     = f.get("value", "")
            conf      = float(f.get("confidence", 0.7))
        except Exception:
            continue
        if not (entity and attribute and value not in (None, "")):
            continue
        out.append(FactCandidate(
            entity=entity, attribute=attribute,
            value=value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
            confidence=max(0.0, min(1.0, conf)),
        ))
    return out


def _parse_json(raw: str) -> dict | None:
    if not raw: return None
    s = raw.strip()
    # Strip ``` fences if the model leaked them despite our instructions.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback: pull the first JSON object out of mixed text.
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: return None
    return None
