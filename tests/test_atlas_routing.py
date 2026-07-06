"""
tests/test_atlas_routing.py
===========================
Atlas must route each decomposed subgoal to the best-fit specialist contract
(not force everything through forge.draft_post), and it must construct with the
same 4-arg signature supervisord uses for every agent.
"""
import pytest

from openclaw.agents.atlas import AtlasAgent
from openclaw.agents.forge import ForgeAgent
from openclaw.agents.mirror import MirrorAgent
from openclaw.agents.scout import ScoutAgent
from openclaw.agents.sentinel import SentinelAgent
from openclaw.agents.smith import SmithAgent
from openclaw.agents.vector import VectorAgent
from openclaw.contracts import ContractRegistry


class _Gov:
    """Minimal governor stub carrying just a contract registry."""
    def __init__(self):
        self.contract_registry = ContractRegistry()
        self.contract_registry.register_from_agents([
            AtlasAgent, ForgeAgent, MirrorAgent, ScoutAgent,
            SentinelAgent, SmithAgent, VectorAgent,
        ])


@pytest.fixture
def atlas(fake_redis, fake_llm):
    return AtlasAgent(fake_llm, fake_redis, fuse=None, governor=_Gov())


def test_all_agents_construct_with_uniform_signature(fake_redis, fake_llm):
    gov = _Gov()
    for cls in (AtlasAgent, ForgeAgent, MirrorAgent, ScoutAgent,
                SentinelAgent, SmithAgent, VectorAgent):
        agent = cls(fake_llm, fake_redis, fuse=None, governor=gov)
        assert agent.NAME
        assert agent.channel == f"adonis:agent:{agent.NAME}"


def test_atlas_has_planner(atlas):
    assert atlas.planner is not None


@pytest.mark.parametrize("description,expected_agent", [
    ("debug the python api error in the script", "smith"),
    ("research recent arxiv papers on transformers", "scout"),
    ("write a blog post about productivity", "forge"),
    ("find sales leads in the saas sector", "vector"),
    ("check system health and uptime status", "sentinel"),
])
def test_select_contract_routes_by_domain(atlas, description, expected_agent):
    name = atlas._select_contract(description)
    assert name.startswith(expected_agent + "."), f"{description!r} -> {name}"


def test_select_contract_falls_back_to_forge(atlas):
    assert atlas._select_contract("zzz qqq xyzzy") == "forge.draft_post"


def test_select_contract_never_recurses_into_atlas(atlas):
    # 'orchestrate plan manage goal' are Atlas's own domains; must not pick atlas.*
    name = atlas._select_contract("orchestrate and plan and manage the goal")
    assert not name.startswith("atlas.")
