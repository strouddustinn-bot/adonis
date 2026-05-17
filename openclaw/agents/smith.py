"""
openclaw/agents/smith.py
=========================
SMITH — Self-Debugging Agent

Watches for agent failures, classifies them, generates patches,
routes fixes through Prometheus, applies them, and logs patterns to MIRROR.

Failure classes:
  SYNTAX        — Python/JSON parse errors
  LOGIC         — wrong output, hallucination
  TOOL_FAILURE  — API/network/service errors
  TIMEOUT       — exceeded time budget
  PROMETHEUS    — blocked by ethical fuse (not patchable by Smith)
"""
import os, json, logging, asyncio
from openclaw.base_agent import BaseAgent

log = logging.getLogger("smith")

class FailureClass:
    SYNTAX     = "syntax"
    LOGIC      = "logic"
    TOOL       = "tool_failure"
    TIMEOUT    = "timeout"
    PROMETHEUS = "prometheus_block"
    UNKNOWN    = "unknown"

class SmithAgent(BaseAgent):
    NAME    = "smith"
    DOMAINS = ["code","debug","engineering","fix","api","integration","error","bug","script","patch"]
    # Self-debug: reads vault patch queue; doesn't go out to the network on its own.
    CAPABILITIES = frozenset({"vault:read:SELF/*", "time:read"})

    async def handle(self, task: dict, session_id: str) -> dict:
        task_type = task.get("type","debug")

        if task_type == "debug_failure":
            return await self._debug_failure(task, session_id)
        elif task_type == "generate_code":
            return await self._generate_code(task, session_id)
        elif task_type == "review_code":
            return await self._review_code(task, session_id)
        else:
            return await self._generate_code(task, session_id)

    # ── Debug failure ────────────────────────────────────────────────────────

    async def _debug_failure(self, task: dict, session_id: str) -> dict:
        error       = task.get("error","")
        context_    = task.get("context","")
        agent_name  = task.get("failed_agent","unknown")
        attempt     = task.get("attempt", 1)

        if attempt > 3:
            return {"status":"failed","reason":f"Max retry attempts reached for {agent_name}","agent":self.NAME}

        # Classify failure
        failure_class = self._classify(error)
        log.info(f"[SMITH] Debugging {agent_name}: {failure_class} (attempt {attempt})")

        if failure_class == FailureClass.PROMETHEUS:
            return {"status":"blocked","reason":"Prometheus block — Smith cannot patch ethical decisions.","agent":self.NAME}

        # Generate fix
        fix = await self._generate_fix(error, context_, failure_class, agent_name)
        if not fix:
            return {"status":"failed","reason":"Could not generate fix","agent":self.NAME}

        # Gate fix through Prometheus
        approved, _ = await self.evaluate_action(
            action_type="apply_patch",
            description=f"Apply code patch to agent {agent_name}: {fix[:200]}",
            payload={"agent":agent_name,"patch_preview":fix[:200]},
            session_id=session_id
        )
        if not approved:
            return {"status":"blocked","reason":"Patch blocked by Prometheus","agent":self.NAME}

        # Log pattern for MIRROR
        await self.redis.lpush("smith:patterns", json.dumps({
            "agent":agent_name,"class":failure_class,"error_snippet":error[:100],"fix_snippet":fix[:100]
        }))
        await self.redis.ltrim("smith:patterns", 0, 499)

        return {
            "status":         "patched",
            "failed_agent":   agent_name,
            "failure_class":  failure_class,
            "fix":            fix,
            "next_action":    f"Retry {agent_name} with patch applied",
            "agent":          self.NAME,
        }

    def _classify(self, error: str) -> str:
        e = error.lower()
        if any(k in e for k in ["syntaxerror","parse error","json decode","invalid syntax"]): return FailureClass.SYNTAX
        if any(k in e for k in ["timeout","timed out","deadline exceeded"]):                  return FailureClass.TIMEOUT
        if any(k in e for k in ["connection","network","http","status code","api error"]):     return FailureClass.TOOL
        if any(k in e for k in ["prometheus","fuse","blocked","locked"]):                     return FailureClass.PROMETHEUS
        if any(k in e for k in ["assertionerror","keyerror","typeerror","valueerror"]):       return FailureClass.LOGIC
        return FailureClass.UNKNOWN

    async def _generate_fix(self, error: str, context: str, failure_class: str, agent_name: str) -> str:
        prompt = f"""You are Smith, the self-debugging agent for Adonis AI.
A {failure_class} error occurred in agent '{agent_name}'.

Error: {error[:500]}
Context: {context[:500]}

Generate a concrete, specific fix. For code errors: show the corrected code snippet.
For API errors: show the correct retry strategy. For logic errors: show the corrected logic.
Be brief and precise. No explanation — just the fix."""
        try:
            return await self.llm_call(
                system="You are Smith, Adonis self-debugging agent. Output fixes only, no commentary.",
                user=prompt, max_tokens=400)
        except Exception as e:
            log.error(f"[SMITH] Fix generation failed: {e}"); return ""

    # ── Code generation ──────────────────────────────────────────────────────

    async def _generate_code(self, task: dict, session_id: str) -> dict:
        spec     = task.get("spec","")
        language = task.get("language","python")
        approved, _ = await self.evaluate_action("generate_code",f"Generate {language} code: {spec[:100]}",session_id=session_id)
        if not approved:
            return {"status":"blocked","agent":self.NAME}
        code = await self.llm_call(
            system=f"Expert {language} engineer. Write clean, production-ready code only. No explanations.",
            user=spec, max_tokens=1500)
        return {"status":"ok","code":code,"language":language,"agent":self.NAME}

    async def _review_code(self, task: dict, session_id: str) -> dict:
        code = task.get("code","")
        review = await self.llm_call(
            system="Senior code reviewer. Identify bugs, security issues, and performance problems. Be specific.",
            user=f"Review this code:\n\n{code}", max_tokens=800)
        return {"status":"ok","review":review,"agent":self.NAME}
