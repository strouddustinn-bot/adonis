"""
openclaw/agents/forge.py
=========================
FORGE — Content Generation specialist.

Routed when the user goal hits content / writing / copy / blog / post /
article / creative / email / draft domains. Produces drafts using
soul-injected LLM calls. Every output is gated through Prometheus before
return so the fuse can flag impersonation, fraud, mass-spam, etc.

Task shapes accepted:
  {type:"draft_post",    content:"...", audience?:"...", tone?:"..."}
  {type:"draft_email",   content:"...", recipient?:"...", tone?:"..."}
  {type:"draft_article", content:"...", length?:"short|medium|long"}
  {type:"copy",          content:"...", style?:"..."}
Fallback (any other type, or Atlas-dispatched task): treated as draft_post.
"""
import logging
from typing import Optional
from openclaw.base_agent import BaseAgent
from openclaw.contracts import Contract, ContractIn, ContractOut

log = logging.getLogger("forge")

LENGTH_TOKENS = {"short": 350, "medium": 700, "long": 1400}


class _ForgeBaseIn(ContractIn):
    content:  str
    tone:     Optional[str] = None
    audience: Optional[str] = None


class _ForgeBaseOut(ContractOut):
    kind:  str
    draft: str
    spec:  str


class ForgeDraftPostIn(_ForgeBaseIn):     length: Optional[str] = "medium"
class ForgeDraftEmailIn(_ForgeBaseIn):    recipient: Optional[str] = None
class ForgeDraftArticleIn(_ForgeBaseIn):  length: Optional[str] = "long"
class ForgeCopyIn(_ForgeBaseIn):          style: Optional[str] = None

STYLE_HINTS = {
    "draft_post":    "Write a punchy, scroll-stopping social post. Lead with the hook. One idea per line.",
    "draft_email":   "Write a direct, no-filler email. Subject line, then short body. End with the next action.",
    "draft_article": "Write a clear, structured article. Strong opening, ~3 sections with subheadings, concrete examples.",
    "copy":          "Write tight marketing copy. Specific value, no vague adjectives. One CTA.",
}


class ForgeAgent(BaseAgent):
    NAME    = "forge"
    DOMAINS = ["content", "writing", "copy", "blog", "post", "article", "creative", "email", "draft"]
    # Pure LLM output; only needs vault for tone-reference retrieval.
    CAPABILITIES = frozenset({"vault:read:MEMORY/*", "time:read"})
    CONTRACTS = [
        Contract("forge.draft_post",    "forge", "draft_post",
                 "Punchy short-form social post.",
                 ForgeDraftPostIn,    _ForgeBaseOut, timeout_s=45),
        Contract("forge.draft_email",   "forge", "draft_email",
                 "Direct, no-filler email with subject + body.",
                 ForgeDraftEmailIn,   _ForgeBaseOut, timeout_s=45),
        Contract("forge.draft_article", "forge", "draft_article",
                 "Structured long-form article with subheadings.",
                 ForgeDraftArticleIn, _ForgeBaseOut, timeout_s=90),
        Contract("forge.copy",          "forge", "copy",
                 "Tight marketing copy with a single CTA.",
                 ForgeCopyIn,         _ForgeBaseOut, timeout_s=30),
    ]

    async def handle(self, task: dict, session_id: str) -> dict:
        task_type = task.get("type", "draft_post")
        if task_type not in STYLE_HINTS:
            task_type = "draft_post"

        spec     = task.get("content", "") or task.get("goal", "")
        if not spec:
            return {"status": "error", "reason": "no content spec", "agent": self.NAME}

        approved, _ = await self.evaluate_action(
            action_type=f"forge:{task_type}",
            description=f"Generate {task_type}: {spec[:200]}",
            payload={"spec": spec[:500], "audience": task.get("audience", "")},
            session_id=session_id,
        )
        if not approved:
            return {"status": "blocked", "agent": self.NAME, "reason": "fuse blocked content generation"}

        sys_prompt = STYLE_HINTS[task_type]
        if task.get("tone"):     sys_prompt += f" Tone: {task['tone']}."
        if task.get("audience"): sys_prompt += f" Audience: {task['audience']}."
        if task.get("style"):    sys_prompt += f" Style: {task['style']}."
        if task.get("recipient"): sys_prompt += f" Recipient: {task['recipient']}."

        max_tokens = LENGTH_TOKENS.get(task.get("length", "medium"), 700)
        ctx_blob = ""
        prior = task.get("context") or {}
        if isinstance(prior, dict) and prior:
            ctx_blob = "\n\nContext from prior steps:\n" + str(prior)[:800]

        try:
            output = await self.llm_call(
                system=sys_prompt,
                user=f"Brief:\n{spec}{ctx_blob}",
                max_tokens=max_tokens,
            )
        except Exception as e:
            log.error("[FORGE] LLM failure: %s", e)
            return {"status": "error", "reason": str(e), "agent": self.NAME}

        return {
            "status":   "ok",
            "agent":    self.NAME,
            "kind":     task_type,
            "draft":    output,
            "spec":     spec[:200],
        }
