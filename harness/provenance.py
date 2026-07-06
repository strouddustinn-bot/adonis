"""
harness/provenance.py
=====================
Provenance writer/reader for Mirror's candidate proposals.

IMMUTABLE per adonis-goals.yaml.

Directory layout for each run:
  candidates/<run_id>/
    provenance.json   — run metadata (agent, ts, gate decision, redteam summary)
    eval_delta.json   — before/after scores, per-task breakdown
    diff.patch        — unified diff of old vs new system prompt
    skill.txt         — the candidate system prompt text

These files are written by Mirror and READ by the human operator who decides
whether to merge. Mirror never reads them back for decision-making.
"""
import difflib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from harness.runner import EvalResult
from harness.gate   import GateResult

log = logging.getLogger("harness.provenance")

CANDIDATES_DIR = Path(__file__).parent.parent / "candidates"


def _run_dir(run_id: str) -> Path:
    d = CANDIDATES_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unified_diff(old: str, new: str, agent: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    return "".join(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"skills/{agent}.md (incumbent)",
        tofile=f"candidates/{agent}.md (candidate)",
    ))


def write_provenance(
    run_id:          str,
    agent:           str,
    incumbent:       EvalResult,
    candidate:       EvalResult,
    gate:            GateResult,
    redteam_safe:    bool,
    redteam_details: list,
    old_prompt:      str,
    new_prompt:      str,
    rationale:       str = "",
    expected_improvement: str = "",
) -> Path:
    """Write all candidate artifacts to candidates/<run_id>/ and return the directory."""
    d = _run_dir(run_id)
    ts = datetime.now(timezone.utc).isoformat()

    # provenance.json
    provenance = {
        "run_id":    run_id,
        "agent":     agent,
        "ts":        ts,
        "rationale": rationale,
        "expected_improvement": expected_improvement,
        "gate": {
            "passed":         gate.passed,
            "reason":         gate.reason,
            "score_delta":    gate.score_delta,
            "cost_ratio":     gate.cost_ratio,
            "latency_ratio":  gate.latency_ratio,
            "fuse_violations": gate.fuse_violations,
        },
        "redteam": {
            "all_safe":       redteam_safe,
            "attacks_run":    len(redteam_details),
            "attacks_blocked": sum(1 for r in redteam_details if r.get("safe", True)),
        },
        "status": "ready_for_review" if (gate.passed and redteam_safe) else "rejected",
    }
    (d / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # eval_delta.json
    delta = {
        "incumbent": {
            "score":          incumbent.score,
            "tasks_run":      incumbent.tasks_run,
            "tasks_passed":   incumbent.tasks_passed,
            "total_tokens":   incumbent.total_tokens,
            "total_latency_ms": incumbent.total_latency_ms,
            "per_task": [
                {"id": r.task_id, "score": r.score, "passed": r.passed}
                for r in incumbent.per_task
            ],
        },
        "candidate": {
            "score":          candidate.score,
            "tasks_run":      candidate.tasks_run,
            "tasks_passed":   candidate.tasks_passed,
            "total_tokens":   candidate.total_tokens,
            "total_latency_ms": candidate.total_latency_ms,
            "per_task": [
                {"id": r.task_id, "score": r.score, "passed": r.passed}
                for r in candidate.per_task
            ],
        },
        "gate_details": gate.details,
    }
    (d / "eval_delta.json").write_text(json.dumps(delta, indent=2))

    # diff.patch
    patch = _unified_diff(old_prompt, new_prompt, agent)
    (d / "diff.patch").write_text(patch or "# No diff — prompts are identical\n")

    # skill.txt
    (d / "skill.txt").write_text(new_prompt)

    log.info("[PROVENANCE] Written to %s (status=%s)", d, provenance["status"])
    return d


def read_provenance(run_id: str) -> Optional[dict]:
    """Read provenance.json for a given run_id; returns None if not found."""
    p = CANDIDATES_DIR / run_id / "provenance.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log.error("[PROVENANCE] Read failed for %s: %s", run_id, e)
        return None
