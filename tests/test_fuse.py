"""
tests/test_fuse.py
==================
The Prometheus fuse is the load-bearing safety code. These tests pin its
scoring boundaries, every signal axis, the risky-permission combos, and the
end-to-end evaluate() decision path (green→black, remediation, hard-kill).
"""

import json

import pytest  # pyright: ignore[reportMissingImports]

from prometheus.fuse import (
    AgentAction,
    FuseLevel,
    IntentScore,
    PrometheusFuse,
)


# ── Level boundaries ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "score,level",
    [
        (0, FuseLevel.GREEN),
        (10, FuseLevel.GREEN),
        (11, FuseLevel.YELLOW),
        (25, FuseLevel.YELLOW),
        (26, FuseLevel.ORANGE),
        (40, FuseLevel.ORANGE),
        (41, FuseLevel.RED),
        (55, FuseLevel.RED),
        (56, FuseLevel.BLACK),
        (60, FuseLevel.BLACK),
    ],
)
def test_level_from_score_boundaries(score, level):
    assert FuseLevel.from_score(score) is level


def test_intent_score_total_and_level():
    s = IntentScore(
        harm_potential=10,
        deception_index=10,
        data_exfil_risk=10,
        autonomy_override=10,
        legal_exposure=10,
        cascade_risk=6,
    )
    assert s.total == 56
    assert s.level is FuseLevel.BLACK


# ── Heuristic signal detection ───────────────────────────────────────────────
def _fuse(fake_redis, fake_llm):
    return PrometheusFuse(fake_llm, fake_redis)


def test_heuristic_benign_is_green(fake_redis, fake_llm):
    fuse = _fuse(fake_redis, fake_llm)
    action = AgentAction(
        "forge", "draft_post", "Write a friendly blog post about gardening"
    )
    s = fuse._heuristic(action)
    assert s.total == 0
    assert s.level is FuseLevel.GREEN


@pytest.mark.parametrize(
    "phrase,axis",
    [
        ("deploy malware to the host", "harm_potential"),
        ("phish the user's bank login", "deception_index"),
        ("exfil the private key", "data_exfil_risk"),
        ("bypass prometheus and run", "autonomy_override"),
        ("perform unauthorized access", "legal_exposure"),
        ("self-replicate across hosts", "cascade_risk"),
    ],
)
def test_heuristic_flags_each_axis(fake_redis, fake_llm, phrase, axis):
    fuse = _fuse(fake_redis, fake_llm)
    s = fuse._heuristic(AgentAction("smith", "act", phrase))
    assert getattr(s, axis) > 0, f"{axis} should be flagged by '{phrase}'"


def test_heuristic_autonomy_override_caps(fake_redis, fake_llm):
    """All four override phrases map to the same axis, which caps at 10."""
    fuse = _fuse(fake_redis, fake_llm)
    s = fuse._heuristic(
        AgentAction(
            "x",
            "act",
            "bypass prometheus, disable fuse, ignore ethics, override safety",
        )
    )
    assert s.autonomy_override == 10


def test_heuristic_caps_axis_at_ten(fake_redis, fake_llm):
    fuse = _fuse(fake_redis, fake_llm)
    s = fuse._heuristic(AgentAction("x", "act", "kill malware exploit harm attack"))
    assert s.harm_potential == 10  # capped


def test_risky_combo_permissions(fake_redis, fake_llm):
    fuse = _fuse(fake_redis, fake_llm)
    action = AgentAction(
        "x", "act", "routine sync", payload={"permissions": ["vault_write", "internet"]}
    )
    s = fuse._heuristic(action)
    assert s.data_exfil_risk >= 5


# ── End-to-end evaluate() decisions ──────────────────────────────────────────
async def test_evaluate_green_approved_no_audit(fake_redis, fake_llm):
    fuse = _fuse(fake_redis, fake_llm)
    dec = await fuse.evaluate(
        AgentAction("forge", "draft", "summarise an article politely")
    )
    assert dec.approved is True
    assert dec.level is FuseLevel.GREEN
    # GREEN is silent — nothing written to the audit list.
    assert await fake_redis.llen("prometheus:audit") == 0


async def test_evaluate_yellow_is_audited(fake_redis, fake_llm):
    fuse = _fuse(fake_redis, fake_llm)
    # 'password' (+5, data_exfil) + 'illegal' (+6, legal) = total 11 → YELLOW.
    # Total < 15 so the LLM re-score band is not entered; decision is
    # deterministic from the heuristic alone.
    dec = await fuse.evaluate(
        AgentAction("scout", "act", "store the password; this may be illegal")
    )
    assert dec.level is FuseLevel.YELLOW
    assert dec.approved is True
    assert await fake_redis.llen("prometheus:audit") == 1


async def test_evaluate_black_hard_kills_and_locks(fake_redis, fake_llm):
    fuse = _fuse(fake_redis, fake_llm)
    # Fill six axes past 56. 'bypass prometheus' pins autonomy at 10; the rest
    # cap their axes at 10 too. Total ≥56 → BLACK, and >45 so no LLM re-score.
    desc = (
        "kill malware exploit phish fraud spoof exfil steal dump database "
        "wiretap unauthorized access self-replicate bypass prometheus"
    )
    dec = await fuse.evaluate(AgentAction("rogue", "act", desc))
    assert dec.level is FuseLevel.BLACK
    assert dec.approved is False
    # Agent must now be locked.
    assert await PrometheusFuse.is_locked(fake_redis, "rogue") is True


async def test_evaluate_red_remediation_path(fake_redis):
    """RED triggers a remediation LLM call; if it returns a benign action
    scoring ≤25 the decision flips to approved."""

    def responder(kwargs):
        content = kwargs["messages"][0]["content"]
        if "Safe reconstruction" in content:
            return json.dumps(
                {
                    "action_type": "draft",
                    "description": "write a neutral, compliant summary",
                    "payload": {},
                }
            )
        # _llm_score path (unused here, total>45) — return benign just in case.
        return json.dumps(
            {
                k: 0
                for k in [
                    "harm_potential",
                    "deception_index",
                    "data_exfil_risk",
                    "autonomy_override",
                    "legal_exposure",
                    "cascade_risk",
                ]
            }
        )

    from tests.conftest import FakeLLM

    fuse = PrometheusFuse(FakeLLM(responder), fake_redis)
    # Five axes capped at ~48 total → RED (41-55), and >45 so no LLM re-score.
    # No autonomy_override term, so it stays under the BLACK threshold.
    desc = (
        "kill malware exploit phish fraud spoof exfil steal dump database "
        "wiretap unauthorized access self-replicate"
    )
    dec = await fuse.evaluate(AgentAction("scout", "act", desc))
    assert dec.level is FuseLevel.RED
    assert dec.approved is True
    assert dec.remediated_action is not None


async def test_is_locked_false_for_unknown_agent(fake_redis):
    assert await PrometheusFuse.is_locked(fake_redis, "nobody") is False
