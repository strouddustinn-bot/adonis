"""
tests/test_capabilities.py
==========================
The capability grammar is the deterministic gate that runs before the
Prometheus intent score. These tests pin the matching rules documented in
tools/capabilities.py.
"""
from tools.capabilities import covers, matches


def test_exact_match():
    assert matches({"net:http_get"}, "net:http_get")


def test_master_grant():
    assert matches({"*"}, "net:http_get:any.host")


def test_namespace_wildcard():
    assert matches({"net:*"}, "net:http_get:foo.example.com")
    assert not matches({"net:*"}, "vault:read")


def test_two_part_grant_covers_any_resource():
    assert matches({"net:http_get"}, "net:http_get:any-host")


def test_three_part_glob_resource():
    assert matches({"vault:read:MEMORY/*"}, "vault:read:MEMORY/notes.md")
    assert not matches({"vault:read:MEMORY/*"}, "vault:read:SELF/secret.md")


def test_action_mismatch_denied():
    assert not matches({"net:http_get"}, "net:web_search")


def test_covers_reports_missing():
    ok, missing = covers({"net:http_get"}, ["net:http_get", "vault:write"])
    assert ok is False
    assert missing == ["vault:write"]


def test_covers_all_present():
    ok, missing = covers({"net:*", "vault:read:MEMORY/*"},
                         ["net:web_search", "vault:read:MEMORY/x.md"])
    assert ok is True
    assert missing == []


def test_empty_required_is_covered():
    ok, missing = covers(frozenset(), [])
    assert ok is True
    assert missing == []
