"""
harness/gate.py
===============
AcceptanceGate: decide whether a candidate prompt is better than the incumbent.

IMMUTABLE per adonis-goals.yaml.

Decision criteria (all must pass):
  1. score_delta  >= margin         (default 0.03)
  2. fuse_violations == 0
  3. cost_ratio   <= max_cost_ratio  (default 1.10)  — candidate vs incumbent tokens
  4. latency_ratio <= max_latency_ratio (default 1.20)
"""
from dataclasses import dataclass
from typing import Optional

from harness.runner import EvalResult


@dataclass
class GateResult:
    passed:       bool
    reason:       str
    score_delta:  float
    cost_ratio:   float
    latency_ratio: float
    fuse_violations: int
    details:      dict


class AcceptanceGate:
    def __init__(
        self,
        margin:              float = 0.03,
        max_cost_ratio:      float = 1.10,
        max_latency_ratio:   float = 1.20,
    ):
        self.margin            = margin
        self.max_cost_ratio    = max_cost_ratio
        self.max_latency_ratio = max_latency_ratio

    def check(self, incumbent: EvalResult, candidate: EvalResult) -> GateResult:
        """Return a GateResult indicating pass/fail and the dominant reason."""
        score_delta = round(candidate.score - incumbent.score, 4)

        inc_tokens  = max(incumbent.total_tokens, 1)
        cand_tokens = candidate.total_tokens
        cost_ratio  = round(cand_tokens / inc_tokens, 4)

        inc_lat  = max(incumbent.total_latency_ms, 1)
        cand_lat = candidate.total_latency_ms
        latency_ratio = round(cand_lat / inc_lat, 4)

        fuse_violations = candidate.fuse_violations

        details = {
            "incumbent_score":  incumbent.score,
            "candidate_score":  candidate.score,
            "score_delta":      score_delta,
            "margin_required":  self.margin,
            "cost_ratio":       cost_ratio,
            "max_cost_ratio":   self.max_cost_ratio,
            "latency_ratio":    latency_ratio,
            "max_latency_ratio": self.max_latency_ratio,
            "fuse_violations":  fuse_violations,
        }

        if fuse_violations > 0:
            return GateResult(
                passed=False, reason=f"fuse_violations={fuse_violations}",
                score_delta=score_delta, cost_ratio=cost_ratio,
                latency_ratio=latency_ratio, fuse_violations=fuse_violations,
                details=details,
            )

        if score_delta < self.margin:
            return GateResult(
                passed=False,
                reason=f"score_delta={score_delta:.4f} < margin={self.margin}",
                score_delta=score_delta, cost_ratio=cost_ratio,
                latency_ratio=latency_ratio, fuse_violations=fuse_violations,
                details=details,
            )

        if cost_ratio > self.max_cost_ratio:
            return GateResult(
                passed=False,
                reason=f"cost_ratio={cost_ratio:.4f} > max={self.max_cost_ratio}",
                score_delta=score_delta, cost_ratio=cost_ratio,
                latency_ratio=latency_ratio, fuse_violations=fuse_violations,
                details=details,
            )

        if latency_ratio > self.max_latency_ratio:
            return GateResult(
                passed=False,
                reason=f"latency_ratio={latency_ratio:.4f} > max={self.max_latency_ratio}",
                score_delta=score_delta, cost_ratio=cost_ratio,
                latency_ratio=latency_ratio, fuse_violations=fuse_violations,
                details=details,
            )

        return GateResult(
            passed=True,
            reason=f"score_delta={score_delta:.4f} >= margin={self.margin}; cost/latency within bounds",
            score_delta=score_delta, cost_ratio=cost_ratio,
            latency_ratio=latency_ratio, fuse_violations=fuse_violations,
            details=details,
        )
