
"""
routing/planner.py
======================
Hierarchical Planner with Subgoal Verification Checkpoints.
Prevents error compounding in long-running tasks by enforcing intermediate validation.
"""
import asyncio, json, logging, time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

log = logging.getLogger("planner")

class VerificationStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    AMBIGUOUS = "AMBIGUOUS"

@dataclass
class Subgoal:
    id: int
    description: str
    expected_outcome: str
    status: str = "pending" # pending, in_progress, completed, failed
    result: Any = None
    verification_score: float = 0.0

@dataclass
class Plan:
    goal: str
    subgoals: List[Subgoal]
    metadata: Dict[str, Any] = field(default_factory=dict)

class HierarchicalPlanner:
    def __init__(self, llm_client, verifier_fn=None):
        self.llm = llm_client
        # If no custom verifier provided, we use a internal LLM-based judge
        self.verifier = verifier_fn or self._default_llm_verifier

    async def decompose(self, goal: str) -> Plan:
        \"\"\"Breaks a high-level goal into a sequence of verifiable subgoals.\"\"\"
        prompt = (
            f"Decompose the following goal into a sequence of strictly verifiable subgoals.\n"
            f"Goal: {goal}\n\n"
            f"For each subgoal, provide:\n"
            f"1. a clear description of the action.\n"
            f"2. a precise 'expected_outcome' that can be validated (e.g., 'A CSV file exists' or 'Search result contains X').\n\n"
            f"Output as JSON list: [{{'description': '...', 'expected_outcome': '...'}}]"
        )
        
        try:
            resp = await self.llm.messages.create(
                model="claude-sonnet-4-6", 
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            data = json.loads(resp.content[0].text.strip())
            subgoals = [Subgoal(id=i, **item) for i, item in enumerate(data)]
            return Plan(goal=goal, subgoals=subgoals)
        except Exception as e:
            log.error(f"[PLANNER] Decomposition failed: {e}")
            # Fallback: treating the goal as a single subgoal
            return Plan(goal=goal, subgoals=[Subgoal(id=0, description=goal, expected_outcome="Goal achieved")])

    async def _default_llm_verifier(self, subgoal: Subgoal, result: Any) -> tuple[VerificationStatus, float]:
        \"\"\"Lightweight LLM judge for verification.\"\"\"
        prompt = (
            f"Subgoal: {subgoal.description}\n"
            f"Expected Outcome: {subgoal.expected_outcome}\n"
            f"Actual Result: {result}\n\n"
            f"Does the result satisfy the expected outcome? Answer ONLY 'YES', 'NO', or 'UNCLEAR'.\n"
            f"Provide a confidence score 0.0-1.0."
        )
        try:
            resp = await self.llm.messages.create(
                model="claude-haiku-4-5-20251001", 
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.content[0].text.upper()
            if "YES" in text: return VerificationStatus.PASSED, 1.0
            if "NO" in text: return VerificationStatus.FAILED, 1.0
            return VerificationStatus.AMBIGUOUS, 0.5
        except:
            return VerificationStatus.AMBIGUOUS, 0.0

    async def execute_with_checkpoints(self, plan: Plan, executor_fn: Callable):
        \"\"\"
        Executes the plan and verifies every step.
        If a checkpoint fails, it triggers a recovery logic.
        \"\"\"
        for subgoal in plan.subgoals:
            subgoal.status = "in_progress"
            log.info(f"[PLANNER] Executing Subgoal {subgoal.id}: {subgoal.description}")
            
            # 1. Execution
            try:
                result = await executor_fn(subgoal)
                subgoal.result = result
            except Exception as e:
                log.error(f"[PLANNER] Execution error at Subgoal {subgoal.id}: {e}")
                result = f"Error: {e}"

            # 2. Verification
            status, score = await self.verifier(subgoal, result)
            subgoal.verification_score = score
            
            if status == VerificationStatus.PASSED:
                subgoal.status = "completed"
                log.info(f"[PLANNER] Checkpoint {subgoal.id} PASSED.")
            else:
                log.warn(f"[PLANNER] Checkpoint {subgoal.id} {status.value}. Triggering recovery...")
                recovery_result = await self._trigger_recovery(subgoal, result)
                if recovery_result:
                    subgoal.result = recovery_result
                    # Re-verify after recovery
                    status, score = await self.verifier(subgoal, recovery_result)
                    if status == VerificationStatus.PASSED:
                        subgoal.status = "completed"
                        log.info(f"[PLANNER] Checkpoint {subgoal.id} recovered and PASSED.")
                    else:
                        subgoal.status = "failed"
                        raise RuntimeError(f"critical failure at subgoal {subgoal.id}: recovery insufficient")
                else:
                    subgoal.status = "failed"
                    raise RuntimeError(f"Critical failure at subgoal {subgoal.id}: recovery impossible")

        return [sg.result for sg in plan.subgoals]

    async def _trigger_recovery(self, subgoal: Subgoal, failed_result: Any):
        \"\"\"
        Recovery Logic:
        1. Rephrase: Try the same subgoal with a refined prompt.
        2. Escalate: Ask the Orchestrator (Atlas) for a new approach.
        \"\"\"
        log.info(f"[RECOVERY] Attempting to rescue Subgoal {subgoal.id}")
        # For this implementation, we perform a 'Rephrase' using the LLM to suggest a better way to execute the subgoal
        prompt = (
            f"Subgoal: {subgoal.description}\n"
            f"Expected Outcome: {subgoal.expected_outcome}\n"
            f"Failed Result: {failed_result}\n\n"
            f"Why did this fail? Provide a refined, more specific execution instruction to achieve the expected outcome."
        )
        try:
            resp = await self.llm.messages.create(
                model="claude-sonnet-4-6", 
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            # In a real system, we would pass this revised instruction back to the executor_fn
            return f"RECOVERY_ACTION: {resp.content[0].text.strip()}"
        except:
            return None
