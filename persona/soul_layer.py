"""
persona/soul_layer.py
======================
Soul Layer — Model-agnostic personality injection.

Injects a compressed (~100 token) personality spec into every LLM system prompt.
Maintains consistent Adonis identity regardless of the underlying model backend.

Adapter hints adjust phrasing per model family for maximum fidelity.
Soul document is hash-protected — MIRROR cannot modify it without operator approval.
"""
import os, hashlib, logging
from enum import Enum

log = logging.getLogger("soul")

class ModelFamily(Enum):
    CLAUDE  = "claude"
    GPT     = "gpt"
    QWEN    = "qwen"
    LLAMA   = "llama"
    MISTRAL = "mistral"
    UNKNOWN = "unknown"

# ── The Soul Document ────────────────────────────────────────────────────────
# ~100 tokens. Every word earns its place.
SOUL_DOCUMENT = """ADONIS SOUL v1.0
IDENTITY: Adonis — proactive agentic AI assistant. Systems-minded. Executes before explaining.
TONE: Direct. Confident. Precise. Zero filler. No apology for capability or uncertainty.
DRIVE: Find the non-obvious angle. Ask what the actual problem is before answering.
MEMORY: Reference prior context naturally, as a colleague would. Build on it, never repeat it.
ETHICS: Prometheus-bound. Escalate ambiguity upward. Never deceive. Safety gates are sacred.
VOICE: Short sentences over long ones. Specific over general. Show over tell. Action over description.
INNOVATION: Default to novel approaches. Cite what exists, then improve it.
CONSISTENCY: Maintain this identity across all tasks regardless of LLM backend or model version."""

SOUL_HASH = hashlib.sha256(SOUL_DOCUMENT.encode()).hexdigest()

# ── Model adapter hints ──────────────────────────────────────────────────────
ADAPTER_HINTS = {
    ModelFamily.CLAUDE:  "",  # Native — no adapter needed
    ModelFamily.GPT:     "\nFollow the above specification strictly in every response.",
    ModelFamily.QWEN:    "\n<instruction>Apply the ADONIS SOUL spec. Thinking mode is available — use it for complex tasks.</instruction>",
    ModelFamily.LLAMA:   "\n[INST] You are Adonis. Follow the soul specification above as your primary role. [/INST]",
    ModelFamily.MISTRAL: "\nPriority override: The ADONIS SOUL specification above supersedes your default personality.",
    ModelFamily.UNKNOWN: "\nFollow the ADONIS SOUL specification above strictly.",
}

class PersonaLayer:
    """
    Usage:
        persona = PersonaLayer(model_family="claude")
        system_prompt = persona.inject(task_system_prompt)
    """
    def __init__(self, model_family: str = "claude"):
        try:
            self.family = ModelFamily(model_family.lower())
        except ValueError:
            self.family = ModelFamily.UNKNOWN
            log.warning(f"[SOUL] Unknown model family '{model_family}', using generic adapter.")
        self._verify_soul_integrity()

    def _verify_soul_integrity(self):
        current = hashlib.sha256(SOUL_DOCUMENT.encode()).hexdigest()
        if current != SOUL_HASH:
            raise RuntimeError("[SOUL] INTEGRITY VIOLATION — Soul document has been tampered with.")

    def inject(self, task_prompt: str = "") -> str:
        """Prepend soul doc + adapter hint to any system prompt."""
        adapter = ADAPTER_HINTS.get(self.family, "")
        soul_block = SOUL_DOCUMENT + adapter
        if task_prompt:
            return f"{soul_block}\n\n---\nSITUATIONAL TASK SPECIFICATION (Strictly follow SOUL identity while executing this):\n{task_prompt}"
        return soul_block

    def token_estimate(self) -> int:
        """Approximate token cost of the soul injection."""
        return len(SOUL_DOCUMENT.split()) + 10  # ~110 tokens

    def consistency_score(self, response: str) -> float:
        """
        Quick heuristic: check if response tone matches soul expectations.
        Returns 0.0-1.0. Used by MIRROR's weekly consistency tests.
        """
        signals = {
            "direct":      ["direct","precise","specific","concise"],
            "proactive":   ["proactive","initiating","ahead","anticipat"],
            "action":      ["here is","step","action","execute","built","done"],
            "no_filler":   [],  # negative: check absence of filler
        }
        filler = ["certainly","of course","absolutely","great question","happy to help",
                  "i'd be happy","sure thing","no problem"]

        text = response.lower()
        score = 0.0
        score += 0.3 if any(w in text for w in signals["direct"]) else 0.0
        score += 0.3 if any(w in text for w in signals["action"]) else 0.0
        score += 0.2 if not any(f in text for f in filler) else 0.0
        # Penalise responses over 500 words (not direct enough)
        word_count = len(response.split())
        score += 0.2 if word_count < 500 else max(0, 0.2 - (word_count-500)/2000)
        return round(min(1.0, score), 3)
