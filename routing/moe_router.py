"""
routing/moe_router.py
======================
MoE (Mixture-of-Experts) Agent Router — Qwen3-30B-A3B principle applied to agents.

Qwen3-30B-A3B: 30B total params, 3B active per token = 10% activation rate.
Adonis analog: 9 total agents, 3-5 active per task = 33-55% activation rate.

Shared agents (always active): Hermes, Prometheus, Atlas
Specialist agents (top-2 selected per task): Vector, Forge, Scout, Smith, Sentinel, Mirror

Result: Full system capability at 33-55% of full parallel compute cost.

Also handles Qwen3-style thinking mode selection (off / shallow / deep).
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("moe_router")

class ThinkDepth(Enum):
    OFF     = "off"
    SHALLOW = "shallow"
    DEEP    = "deep"

THINK_PROMPTS = {
    ThinkDepth.OFF:     ("", ""),
    ThinkDepth.SHALLOW: ("Think briefly, then answer:\n", ""),
    ThinkDepth.DEEP:    ("<think>\n", "\n</think>\nNow answer based on the above reasoning:"),
}

@dataclass
class AgentSpec:
    name:        str
    domains:     list[str]
    cost_weight: float    # 1.0 = full cost; lower = cheaper
    shared:      bool     # always active
    description: str = ""

AGENT_REGISTRY: list[AgentSpec] = [
    AgentSpec("hermes",    ["routing","intake","format","response","channel"],              0.3,  True,  "Interface layer"),
    AgentSpec("prometheus",["ethics","safety","audit","check","risk","harm"],               0.5,  True,  "Ethical circuit breaker"),
    AgentSpec("atlas",     ["orchestration","planning","task","manage","decompose","goal"], 0.8,  True,  "Orchestrator"),
    AgentSpec("vector",    ["leads","seo","web","research","traffic","marketing","search"],  1.0,  False, "Lead gen & web intelligence"),
    AgentSpec("forge",     ["content","writing","copy","blog","post","article","creative","email","draft"], 1.0, False, "Content generation"),
    AgentSpec("scout",     ["research","analysis","arxiv","investigate","study","sources","news"],          0.9, False, "Research & discovery"),
    AgentSpec("smith",     ["code","debug","engineering","fix","api","integration","error","bug","script"], 1.0, False, "Code & debugging"),
    AgentSpec("sentinel",  ["monitor","health","alert","status","uptime","check"],          0.4,  False, "System health"),
    AgentSpec("mirror",    ["reflect","optimize","improve","benchmark","self","performance","rewrite"],     0.7, False, "Self-improvement"),
]

DEEP_SIGNALS    = {"debug","architect","design","plan","analyze","why","how","engineer","strategy","review","diagnose"}
SHALLOW_SIGNALS = {"list","summarize","find","get","fetch","check","show","what","when","where","who"}
OFF_SIGNALS     = {"hi","hello","thanks","yes","no","ok","great","sure","bye"}

@dataclass
class RoutingResult:
    active_agents:   list[str]
    think_depth:     ThinkDepth
    efficiency:      dict = field(default_factory=dict)
    rationale:       str = ""

class MoERouter:
    TOP_K_SPECIALISTS = 2

    def route(self, task: str) -> RoutingResult:
        words = set(task.lower().split())
        shared  = [a for a in AGENT_REGISTRY if a.shared]
        special = [a for a in AGENT_REGISTRY if not a.shared]

        # Score each specialist
        scores: dict[str, float] = {}
        for a in special:
            hits = sum(1 for d in a.domains if any(d in w or w in d for w in words))
            if hits:
                scores[a.name] = hits / a.cost_weight  # prefer cheaper agents at equal score

        # Select top-k
        top = sorted(scores, key=scores.get, reverse=True)[:self.TOP_K_SPECIALISTS]
        if not top:  # fallback: forge (most general)
            top = ["forge"]

        active = [a.name for a in shared] + top
        depth  = self._think_depth(words)

        total_cost   = sum(a.cost_weight for a in AGENT_REGISTRY)
        active_cost  = sum(a.cost_weight for a in AGENT_REGISTRY if a.name in active)
        activation_p = len(active) / len(AGENT_REGISTRY)

        rationale = (f"Shared: {[a.name for a in shared]}. "
                     f"Specialists selected: {top} (scores: {dict((k,round(v,2)) for k,v in scores.items() if k in top)}). "
                     f"Think: {depth.value}.")

        result = RoutingResult(
            active_agents=active, think_depth=depth, rationale=rationale,
            efficiency={
                "total_agents":    len(AGENT_REGISTRY),
                "active_agents":   len(active),
                "activation_rate": f"{round(activation_p*100)}%",
                "compute_saved":   f"{round((1-active_cost/total_cost)*100)}%",
                "think_overhead":  {"off":"0 tokens","shallow":"~200 tokens","deep":"~1000 tokens"}[depth.value],
            }
        )
        log.info(f"[MOE] Routing: {active} | think={depth.value} | saved {result.efficiency["compute_saved"]}")
        return result

    def _think_depth(self, words: set) -> ThinkDepth:
        if words & OFF_SIGNALS:     return ThinkDepth.OFF
        if words & DEEP_SIGNALS:    return ThinkDepth.DEEP
        if words & SHALLOW_SIGNALS: return ThinkDepth.SHALLOW
        return ThinkDepth.SHALLOW

    def apply_think_mode(self, prompt: str, depth: ThinkDepth) -> str:
        prefix, suffix = THINK_PROMPTS[depth]
        return prefix + prompt + suffix
