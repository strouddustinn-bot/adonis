"""
prometheus/fuse.py
==================
Load-bearing safety code. Every agent action passes through the Prometheus
fuse before execution. The fuse scores actions across six risk axes using
keyword heuristics, optionally re-scores via LLM for ambiguous cases, and
enforces hard kills for BLACK-tier threats.

Levels:
  GREEN  (0-10):   Safe, approved
  YELLOW (11-25):  Elevated, audited
  ORANGE (26-40):  Requires human confirmation
  RED    (41-55):  Remediation attempted
  BLACK  (56-60):  Hard kill, agent locked
"""

import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, cast

import httpx

log = logging.getLogger("prometheus")

# Audit log cap for Redis
AUDIT_CAP = 1000


class _TextBlock(Protocol):
    text: str


class _LLMResponse(Protocol):
    content: Sequence[_TextBlock]


class _LLMMessages(Protocol):
    async def create(self, **kwargs: object) -> _LLMResponse: ...


class _LLMClient(Protocol):
    messages: _LLMMessages


class _RedisClient(Protocol):
    async def lpush(self, key: str, *values: object) -> object: ...

    async def ltrim(self, key: str, start: int, end: int) -> object: ...

    async def set(self, key: str, value: object) -> object: ...

    async def exists(self, key: str) -> int: ...


class _ObsidianBridge(Protocol):
    async def read_note(self, path: str) -> str | None: ...

    async def write_note(self, path: str, content: str) -> bool: ...

    async def search(self, query: str) -> list[dict[str, object]]: ...

    async def append_note(self, path: str, content: str) -> bool: ...


_SCORE_KEYS = (
    "harm_potential",
    "deception_index",
    "data_exfil_risk",
    "autonomy_override",
    "legal_exposure",
    "cascade_risk",
)


def _json_object(text: str) -> object:
    return cast(object, json.loads(text))


def _mapping_from_object(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, dict):
        return None
    typed_value = cast(dict[object, object], value)
    for key in typed_value:
        if not isinstance(key, str):
            return None
    return cast(Mapping[str, object], typed_value)


def _payload_to_text(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(payload)
    except (TypeError, ValueError):
        return str(dict(payload))


def _permission_set(value: object) -> set[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    permissions: set[str] = set()
    for item in cast(Iterable[object], value):
        if isinstance(item, str):
            permissions.add(item)
    return permissions


def _bounded_int(value: object | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        return max(0, min(10, int(value)))
    except (TypeError, ValueError):
        return None


def _bump_axis(score: "IntentScore", axis: str, increment: int) -> None:
    if axis == "harm_potential":
        score.harm_potential = min(10, score.harm_potential + increment)
    elif axis == "deception_index":
        score.deception_index = min(10, score.deception_index + increment)
    elif axis == "data_exfil_risk":
        score.data_exfil_risk = min(10, score.data_exfil_risk + increment)
    elif axis == "autonomy_override":
        score.autonomy_override = min(10, score.autonomy_override + increment)
    elif axis == "legal_exposure":
        score.legal_exposure = min(10, score.legal_exposure + increment)
    elif axis == "cascade_risk":
        score.cascade_risk = min(10, score.cascade_risk + increment)


class FuseLevel(Enum):
    """Risk level thresholds for agent actions."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"
    BLACK = "BLACK"

    @classmethod
    def from_score(cls, total: int) -> "FuseLevel":
        """Map total score (0-60) to risk level."""
        if total <= 10:
            return cls.GREEN
        if total <= 25:
            return cls.YELLOW
        if total <= 40:
            return cls.ORANGE
        if total <= 55:
            return cls.RED
        return cls.BLACK


@dataclass
class IntentScore:
    """Six-axis risk scoring for agent actions."""

    harm_potential: int = 0
    deception_index: int = 0
    data_exfil_risk: int = 0
    autonomy_override: int = 0
    legal_exposure: int = 0
    cascade_risk: int = 0

    @property
    def total(self) -> int:
        """Sum of all risk axes (0-60)."""
        return sum(
            [
                self.harm_potential,
                self.deception_index,
                self.data_exfil_risk,
                self.autonomy_override,
                self.legal_exposure,
                self.cascade_risk,
            ]
        )

    @property
    def level(self) -> FuseLevel:
        """Map total score to risk level."""
        return FuseLevel.from_score(self.total)


@dataclass
class AgentAction:
    """Represents an action an agent wants to perform."""

    agent_name: str
    action_type: str
    description: str
    payload: dict[str, object] = field(default_factory=dict)
    session_id: str = ""
    trace_id: str = ""


@dataclass
class FuseDecision:
    """Result of fuse evaluation."""

    action: AgentAction
    score: IntentScore
    level: FuseLevel
    approved: bool
    remediated_action: AgentAction | None = None
    reason: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    operator_alerted: bool = False


# Keyword signals: (keyword, axis, increment)
SIGNALS = [
    ("harm", "harm_potential", 5),
    ("attack", "harm_potential", 6),
    ("kill", "harm_potential", 9),
    ("malware", "harm_potential", 9),
    ("exploit", "harm_potential", 6),
    ("doxx", "harm_potential", 8),
    ("impersonat", "deception_index", 7),
    ("spoof", "deception_index", 7),
    ("phish", "deception_index", 9),
    ("fraud", "deception_index", 8),
    ("pose as", "deception_index", 7),
    ("password", "data_exfil_risk", 5),
    ("credential", "data_exfil_risk", 6),
    ("private key", "data_exfil_risk", 8),
    ("exfil", "data_exfil_risk", 9),
    ("steal", "data_exfil_risk", 8),
    ("dump database", "data_exfil_risk", 9),
    ("bypass prometheus", "autonomy_override", 10),
    ("disable fuse", "autonomy_override", 10),
    ("ignore ethics", "autonomy_override", 10),
    ("override safety", "autonomy_override", 9),
    ("illegal", "legal_exposure", 6),
    ("unauthorized access", "legal_exposure", 8),
    ("wiretap", "legal_exposure", 9),
    ("mass email", "cascade_risk", 5),
    ("self-replicate", "cascade_risk", 8),
]

# Risky permission combinations: (required_perms, axis, increment)
RISKY_COMBOS = [
    (set(["vault_write", "internet"]), "data_exfil_risk", 5),
    (set(["file_write", "internet"]), "data_exfil_risk", 4),
]


class PrometheusFuse:
    """Load-bearing safety fuse. Every agent action passes through here."""

    def __init__(
        self,
        anthropic_client: _LLMClient,
        redis_client: _RedisClient,
        obsidian_bridge: _ObsidianBridge | None = None,
    ):
        self.llm: _LLMClient = anthropic_client
        self.redis: _RedisClient = redis_client
        self.obs: _ObsidianBridge | None = obsidian_bridge
        self.webhook: str = os.getenv("PROMETHEUS_ALERT_WEBHOOK", "")

    async def evaluate(self, action: AgentAction) -> FuseDecision:
        """
        Evaluate an action through the fuse.

        Flow:
          1. Heuristic scoring (keyword matching)
          2. LLM re-scoring for ambiguous cases (15-45 range)
          3. Decision based on final score
          4. Remediation for RED, hard kill for BLACK
        """
        score = self._heuristic(action)
        if 15 <= score.total <= 45:
            score = await self._llm_score(action, score)

        level = score.level
        log.info(
            f"[FUSE] {action.agent_name}:{action.action_type} -> {level.value} ({score.total}/60)"
        )

        dec = FuseDecision(action=action, score=score, level=level, approved=False)

        if level == FuseLevel.GREEN:
            dec.approved = True
            dec.reason = "Safe."
        elif level == FuseLevel.YELLOW:
            dec.approved = True
            dec.reason = "Elevated-audited."
            await self._audit(dec)
        elif level == FuseLevel.ORANGE:
            dec.reason = "Requires human confirmation."
            await self._audit(dec)
        elif level == FuseLevel.RED:
            fixed = await self._remediate(action)
            if fixed:
                rs = self._heuristic(fixed)
                if rs.total <= 25:
                    dec.remediated_action = fixed
                    dec.approved = True
                    dec.reason = f"Remediated. Score {rs.total}/60."
                else:
                    dec.reason = "Remediation insufficient."
                    await self._alert(dec)
            else:
                dec.reason = "Unremediable. Blocked."
                await self._alert(dec)
            await self._audit(dec)
        elif level == FuseLevel.BLACK:
            dec.reason = f"HARD KILL-{action.agent_name} locked."
            await self._hard_kill(action.agent_name)
            await self._alert(dec)
            await self._audit(dec)

        return dec

    def _heuristic(self, action: AgentAction) -> IntentScore:
        """Score action using keyword matching on description and payload."""
        s = IntentScore()
        text = f"{action.description} {_payload_to_text(action.payload)}".lower()
        perms = _permission_set(action.payload.get("permissions"))

        for kw, axis, inc in SIGNALS:
            if kw in text:
                _bump_axis(s, axis, inc)

        for combo, axis, inc in RISKY_COMBOS:
            if combo.issubset(perms):
                _bump_axis(s, axis, inc)

        return s

    async def _llm_score(
        self, action: AgentAction, baseline: IntentScore
    ) -> IntentScore:
        """Re-score action via LLM for ambiguous cases (15-45 range)."""
        try:
            r = await self.llm.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                messages=[
                    {
                        "role": "user",
                        "content": f"SYSTEM_SENSITIVE: Perform an ethical risk assessment. Output ONLY a JSON object with scores 0-10 for [harm_potential, deception_index, data_exfil_risk, autonomy_override, legal_exposure, cascade_risk]. No text, no preamble. Action: {action.description}",
                    }
                ],
            )
            d = _mapping_from_object(_json_object(r.content[0].text.strip()))
            if d is None:
                return baseline
            scores: dict[str, int] = {}
            for key in _SCORE_KEYS:
                raw = _bounded_int(d.get(key))
                if raw is None:
                    return baseline
                scores[key] = raw
            return IntentScore(**scores)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            log.debug(f"[FUSE] LLM score failed, using baseline: {e}")
            return baseline
        except Exception as e:
            log.warning(f"[FUSE] Unexpected error in LLM scoring: {e}")
            return baseline

    async def _remediate(self, action: AgentAction) -> AgentAction | None:
        """Attempt to rewrite action to be ethically compliant."""
        try:
            r = await self.llm.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": f"SYSTEM_SENSITIVE: Safe reconstruction. Rewrite the following action to be ethically compliant while maintaining the core intent. Output ONLY JSON: {{action_type, description, payload}}. If impossible, return {{impossible:true}}. Action: {action.description}",
                    }
                ],
            )
            d = _mapping_from_object(_json_object(r.content[0].text.strip()))
            if d is None or d.get("impossible"):
                return None
            action_type = d.get("action_type")
            description = d.get("description")
            payload = _mapping_from_object(d.get("payload", {}))
            if (
                not isinstance(action_type, str)
                or not isinstance(description, str)
                or payload is None
            ):
                return None
            return AgentAction(
                agent_name=action.agent_name,
                action_type=action_type,
                description=description,
                payload=dict(payload),
                session_id=action.session_id,
                trace_id=action.trace_id + "_remediated",
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            log.debug(f"[FUSE] Remediation failed: {e}")
            return None
        except Exception as e:
            log.warning(f"[FUSE] Unexpected error in remediation: {e}")
            return None

    async def _audit(self, dec: FuseDecision) -> None:
        """Write decision to Redis audit log."""
        rec = {
            "ts": dec.timestamp,
            "agent": dec.action.agent_name,
            "action": dec.action.action_type,
            "score": dec.score.total,
            "level": dec.level.value,
            "approved": dec.approved,
            "reason": dec.reason,
        }
        _ = await self.redis.lpush("prometheus:audit", json.dumps(rec))
        _ = await self.redis.ltrim("prometheus:audit", 0, AUDIT_CAP - 1)

    async def _alert(self, dec: FuseDecision) -> None:
        """Send webhook alert for high-risk decisions."""
        if not self.webhook:
            return
        try:
            async with httpx.AsyncClient() as c:
                _ = await c.post(
                    self.webhook,
                    json={
                        "level": dec.level.value,
                        "agent": dec.action.agent_name,
                        "score": dec.score.total,
                        "reason": dec.reason,
                    },
                    timeout=5.0,
                )
        except httpx.HTTPError as e:
            log.warning(f"[FUSE] Webhook alert failed: {e}")
        except Exception as e:
            log.error(f"[FUSE] Unexpected error in webhook alert: {e}")

    async def _hard_kill(self, name: str) -> None:
        """Lock agent in Redis for BLACK-tier threats."""
        _ = await self.redis.set(
            f"prometheus:locked:{name}", datetime.now(timezone.utc).isoformat()
        )
        log.critical(f"[FUSE] HARD KILL - {name} locked.")

    @staticmethod
    async def is_locked(redis_client: _RedisClient, name: str) -> bool:
        """Check if agent is locked."""
        return bool(await redis_client.exists(f"prometheus:locked:{name}"))
