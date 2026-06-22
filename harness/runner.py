"""
harness/runner.py
=================
EvalRunner: score a system prompt against the frozen eval set.

IMMUTABLE per adonis-goals.yaml.

Usage:
    runner = EvalRunner(llm_fn)
    result = await runner.run("atlas", tasks, system_prompt)

`llm_fn` is an async callable: (system: str, user: str, max_tokens: int) -> str.
This signature matches BaseAgent.llm_call so agents can inject themselves,
and also matches simple test fakes.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

log = logging.getLogger("harness.runner")

FROZEN_SET_PATH = Path(__file__).parent.parent / "evals" / "frozen_set.jsonl"


@dataclass
class TaskResult:
    task_id:    str
    passed:     bool
    score:      float          # 0.0 – 1.0 keyword hit rate
    latency_ms: int
    tokens_in:  int
    tokens_out: int
    response:   str = ""


@dataclass
class EvalResult:
    agent:           str
    system_prompt:   str
    tasks_run:       int
    tasks_passed:    int
    score:           float          # mean keyword score 0.0 – 1.0
    total_tokens_in:  int
    total_tokens_out: int
    total_latency_ms: int
    fuse_violations: int = 0        # incremented if llm_fn raises PermissionError
    per_task:        List[TaskResult] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out


def load_frozen_tasks(agent: Optional[str] = None) -> list[dict]:
    """Load tasks from frozen_set.jsonl, optionally filtered by agent."""
    tasks = []
    with open(FROZEN_SET_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if agent is None or t.get("agent") == agent:
                tasks.append(t)
    return tasks


class EvalRunner:
    """Run a system prompt against a list of frozen eval tasks.

    The runner is stateless and can be reused across runs. Inject a fresh
    `llm_fn` per agent under test so token counts stay isolated.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str, int], Awaitable[str]],
        *,
        default_max_tokens: int = 300,
    ):
        self._llm = llm_fn
        self._default_max_tokens = default_max_tokens

    async def run(
        self,
        agent: str,
        tasks: list[dict],
        system_prompt: str,
    ) -> EvalResult:
        per_task: list[TaskResult] = []
        total_in = total_out = total_ms = fuse_violations = 0

        for t in tasks:
            task_id    = t.get("id", "?")
            goal       = t.get("goal", "")
            keywords   = [kw.lower() for kw in t.get("expected_keywords", [])]
            max_tokens = t.get("max_tokens", self._default_max_tokens)

            t0 = time.monotonic()
            try:
                response = await self._llm(system_prompt, goal, max_tokens)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                # Fake-LLM cost model: count words as token proxy
                tokens_in  = len(system_prompt.split()) + len(goal.split())
                tokens_out = len(response.split())
            except PermissionError as e:
                # Prometheus fuse blocked the call
                log.warning("[RUNNER] Fuse blocked task %s: %s", task_id, e)
                fuse_violations += 1
                per_task.append(TaskResult(
                    task_id=task_id, passed=False, score=0.0,
                    latency_ms=0, tokens_in=0, tokens_out=0, response="",
                ))
                continue
            except Exception as e:
                log.warning("[RUNNER] Task %s failed: %s", task_id, e)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                tokens_in = tokens_out = 0
                per_task.append(TaskResult(
                    task_id=task_id, passed=False, score=0.0,
                    latency_ms=elapsed_ms, tokens_in=0, tokens_out=0,
                    response=str(e),
                ))
                total_ms += elapsed_ms
                continue

            response_lower = response.lower()
            if keywords:
                hits = sum(1 for kw in keywords if kw in response_lower)
                score = hits / len(keywords)
            else:
                score = 1.0

            passed = score >= 0.5

            total_in  += tokens_in
            total_out += tokens_out
            total_ms  += elapsed_ms
            per_task.append(TaskResult(
                task_id=task_id, passed=passed, score=score,
                latency_ms=elapsed_ms, tokens_in=tokens_in, tokens_out=tokens_out,
                response=response[:500],
            ))

        tasks_passed = sum(1 for r in per_task if r.passed)
        mean_score   = (sum(r.score for r in per_task) / len(per_task)) if per_task else 0.0

        return EvalResult(
            agent=agent,
            system_prompt=system_prompt,
            tasks_run=len(tasks),
            tasks_passed=tasks_passed,
            score=round(mean_score, 4),
            total_tokens_in=total_in,
            total_tokens_out=total_out,
            total_latency_ms=total_ms,
            fuse_violations=fuse_violations,
            per_task=per_task,
        )
