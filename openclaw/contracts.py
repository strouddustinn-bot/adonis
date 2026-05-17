"""
openclaw/contracts.py
======================
Contract-based delegation protocol.

A Contract is what an agent advertises it can do. Sibling to the
capability tokens in tools/capabilities.py: capabilities say what an
agent is allowed to do, contracts say what an agent claims to do with
declared input/output schemas and an SLA.

Each agent declares zero or more contracts on its class:

    class ScoutAgent(BaseAgent):
        ...
        CONTRACTS = [
            Contract(
                name="scout.research",
                agent="scout",
                task_type="research",
                description="Research a topic on the open web (+ arXiv).",
                input_model=ScoutResearchIn,
                output_model=ScoutResearchOut,
                timeout_s=60,
                version="v1",
            ),
        ]

ContractRegistry collects all contracts at boot and is used by:
  - Atlas, to pick a specific contract per subtask during decomposition
  - BaseAgent.run, to validate the incoming task and the returned result
  - The Hermes /contracts API, so the UI / CLI can introspect

Validation is intentionally LENIENT by default (`strict=False`):
schema mismatches become `_contract_warning` on the result rather than
rejection. Set `strict=True` on a contract when you want hard
enforcement. This matches the "post-hoc validation" trade-off — schemas
can evolve without breaking deployed agents.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Type, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

log = logging.getLogger("contracts")


class ContractIn(BaseModel):
    """Common base for every contract's input model. Subclass and add
    fields. Extra fields are permitted so agents receive trace_id /
    result_key / context without having to declare them per contract."""
    model_config = ConfigDict(extra="allow")


class ContractOut(BaseModel):
    """Common base for every contract's output model."""
    model_config = ConfigDict(extra="allow")
    status: str
    agent:  str


@dataclass
class Contract:
    name:              str                          # globally unique, e.g. "scout.research"
    agent:             str                          # which agent NAME serves it
    task_type:         str                          # task.type discriminator
    description:       str
    input_model:       Type[BaseModel]
    output_model:      Type[BaseModel]
    timeout_s:         int   = 60
    max_retries:       int   = 1                    # total attempts including the first
    backoff_base_s:    float = 1.0                  # exponential: base * 2**(attempt-1)
    fallback_contract: Optional[str] = None         # contract.name to dispatch on exhaustion
    version:           str   = "v1"
    strict:            bool  = False                # if True, validation errors reject

    # ── validation ───────────────────────────────────────────────────────
    def validate_input(self, payload: dict) -> tuple[bool, Optional[str], dict]:
        """Return (ok, error_str_if_any, normalised_payload)."""
        try:
            m = self.input_model(**(payload or {}))
            return True, None, m.model_dump()
        except ValidationError as e:
            return False, _short_err(e), payload

    def validate_output(self, payload: dict) -> tuple[bool, Optional[str]]:
        try:
            self.output_model(**(payload or {}))
            return True, None
        except ValidationError as e:
            return False, _short_err(e)

    # ── introspection ────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "name":              self.name,
            "agent":             self.agent,
            "task_type":         self.task_type,
            "description":       self.description,
            "timeout_s":         self.timeout_s,
            "max_retries":       self.max_retries,
            "backoff_base_s":    self.backoff_base_s,
            "fallback_contract": self.fallback_contract,
            "version":           self.version,
            "strict":            self.strict,
            "input_schema":      self.input_model.model_json_schema(),
            "output_schema":     self.output_model.model_json_schema(),
        }


def _short_err(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:5]:
        loc = ".".join(str(x) for x in err.get("loc", []))
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or str(e)


class ContractRegistry:
    """In-process catalog of every agent's contracts. Built at boot."""

    def __init__(self):
        self.by_name:      dict[str, Contract]       = {}
        self.by_agent:     dict[str, list[Contract]] = {}
        self.by_task_type: dict[str, list[Contract]] = {}

    def register(self, c: Contract) -> None:
        if c.name in self.by_name:
            log.warning("[CONTRACTS] duplicate %s — ignoring", c.name)
            return
        self.by_name[c.name] = c
        self.by_agent.setdefault(c.agent, []).append(c)
        self.by_task_type.setdefault(c.task_type, []).append(c)
        log.info("[CONTRACTS] registered %s (agent=%s, type=%s, %ds)",
                 c.name, c.agent, c.task_type, c.timeout_s)

    def register_from_agents(self, agent_classes) -> None:
        for cls in agent_classes:
            for c in getattr(cls, "CONTRACTS", []) or []:
                self.register(c)

    # ── lookup ───────────────────────────────────────────────────────────
    def find_for_task(self, task: dict, *, agent_name: Optional[str] = None) -> Optional[Contract]:
        """Pick the best contract for an incoming task dict. If the task
        carries an explicit `contract` field, prefer that. Else match by
        `type`. If `agent_name` is given, only contracts owned by that
        agent are considered."""
        if not task: return None
        # Explicit contract name wins.
        explicit = task.get("contract")
        if explicit and explicit in self.by_name:
            c = self.by_name[explicit]
            if not agent_name or c.agent == agent_name:
                return c
        # Match by task type.
        t = task.get("type")
        if not t: return None
        candidates = self.by_task_type.get(t, [])
        if agent_name:
            candidates = [c for c in candidates if c.agent == agent_name]
        return candidates[0] if candidates else None

    def for_agent(self, agent_name: str) -> list[Contract]:
        return list(self.by_agent.get(agent_name, []))

    def all(self) -> list[Contract]:
        return list(self.by_name.values())

    def describe(self) -> list[dict]:
        return [c.to_dict() for c in self.all()]

    def catalog_brief(self) -> list[dict]:
        """Compact catalog for prompting Atlas — no full schemas."""
        return [{
            "name": c.name, "agent": c.agent, "task_type": c.task_type,
            "description": c.description, "version": c.version, "timeout_s": c.timeout_s,
        } for c in self.all()]
