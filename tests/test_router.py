"""
tests/test_router.py
====================
MoE router: shared agents are always active, specialists are top-K by domain
hit density / cost, and think-depth is keyword-driven.
"""
from routing.moe_router import MoERouter, ThinkDepth


def test_shared_agents_always_active():
    r = MoERouter().route("write a blog post about dogs")
    for shared in ("hermes", "prometheus", "atlas"):
        assert shared in r.active_agents


def test_content_goal_routes_to_forge():
    r = MoERouter().route("draft a blog post and email copy")
    assert "forge" in r.active_agents


def test_code_goal_routes_to_smith():
    r = MoERouter().route("debug this api error in my script")
    assert "smith" in r.active_agents


def test_topk_caps_specialist_count():
    r = MoERouter().route("research code content health monitor leads seo")
    specialists = [a for a in r.active_agents
                   if a not in ("hermes", "prometheus", "atlas")]
    assert len(specialists) <= MoERouter.TOP_K_SPECIALISTS


def test_data_goal_routes_to_data():
    r = MoERouter().route("analyze the csv dataset columns and aggregate stats")
    assert "data" in r.active_agents


def test_schedule_goal_routes_to_scheduler():
    r = MoERouter().route("schedule a daily recurring report")
    assert "scheduler" in r.active_agents


def test_stopwords_do_not_pollute_routing():
    # 'a' must not match data's a-containing domains and hijack a content goal.
    r = MoERouter().route("draft a blog post and email copy")
    assert "forge" in r.active_agents
    assert "data" not in r.active_agents


def test_fallback_to_forge_when_no_domain_hit():
    r = MoERouter().route("zzzz qqqq")
    assert "forge" in r.active_agents


def test_think_depth_off_for_smalltalk():
    assert MoERouter().route("hi thanks").think_depth is ThinkDepth.OFF


def test_think_depth_deep_for_analysis():
    assert MoERouter().route("design and architect a strategy").think_depth is ThinkDepth.DEEP


def test_think_depth_shallow_default():
    assert MoERouter().route("list the files").think_depth is ThinkDepth.SHALLOW


def test_efficiency_report_present():
    r = MoERouter().route("research arxiv papers")
    assert "activation_rate" in r.efficiency
    assert "compute_saved" in r.efficiency


def test_apply_think_mode_wraps_prompt():
    router = MoERouter()
    wrapped = router.apply_think_mode("solve x", ThinkDepth.DEEP)
    assert "solve x" in wrapped
    assert wrapped != "solve x"  # deep mode adds <think> scaffolding
