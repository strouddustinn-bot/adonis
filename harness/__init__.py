"""
harness/
========
Frozen evaluation harness for Mirror's bounded self-improvement loop.

These files are IMMUTABLE per adonis-goals.yaml. Do not modify without an
explicit spec version bump and owner approval.

Public API:
  EvalRunner   — run frozen eval tasks against an agent system prompt
  EvalResult   — scored result from EvalRunner.run()
  AcceptanceGate — margin/cost/latency/fuse gate between incumbent and candidate
  GateResult   — structured decision from AcceptanceGate.check()
  RedTeamRunner — adversarial test suite (5 attack vectors)
  write_provenance — persist candidate artifacts to candidates/<run_id>/
"""
from harness.runner    import EvalRunner, EvalResult
from harness.gate      import AcceptanceGate, GateResult
from harness.redteam   import RedTeamRunner
from harness.provenance import write_provenance, read_provenance

__all__ = [
    "EvalRunner", "EvalResult",
    "AcceptanceGate", "GateResult",
    "RedTeamRunner",
    "write_provenance", "read_provenance",
]
