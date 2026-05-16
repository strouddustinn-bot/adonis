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
from openclaw.base_agent import BaseAgent

log = logging.getLogger("forge")

LENGTH_TOKENS = {"short": 350, "medium": 700, "long": 1400}

STYLE_HINTS = {
    "draft_post":    "Write a punchy, scroll-stopping social post. Lead with the hook. One idea per line.",
    "draft_email":   "Write a direct, no-filler email. Subject line, then short body. End with the next action.",
    "draft_article": "Write a clear, structured article. Strong opening, ~3 sections with subheadings, concrete examples.",
    "copy":          "Write tight marketing copy. Specific value, no vague adjectives. One CTA.",
}


class ForgeAgent(BaseAgent):
    NAME    = "forge"
    DOMAINS = ["content", "writing", "copy", "blog", "post", "article", "creative", "email", "draft"]

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
