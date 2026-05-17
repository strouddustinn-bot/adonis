"""
tools/capabilities.py
======================
Capability-token policy layer for tool calls.

Capabilities are namespaced, optionally with a resource argument. Format:

    <namespace>[:<action>[:<resource>]]

Examples:
    "net:*"                                 entire net namespace
    "net:http_get"                          any HTTP GET
    "net:http_get:api.anthropic.com"        only that host
    "vault:read:MEMORY/*"                   read under MEMORY/ (fnmatch glob)
    "mcp:github"                            any tool from the github MCP server
    "*"                                     master grant (rare)

Matching rules:
    - Exact match always wins.
    - "*" grants everything.
    - A 1- or 2-part grant covers any required cap whose first parts match
      (so "net:*" grants "net:http_get:foo.example.com"; "net:http_get"
      grants "net:http_get:any-host").
    - A 3-part grant matches a 3-part required iff the resource glob
      (fnmatch) matches and the namespace + action match exactly.

This is a STRUCTURAL check, evaluated before the Prometheus intent score.
It does not sandbox process memory or syscalls — for that you'd need a
separate isolation layer (bubblewrap, gVisor, WASM). It does mean a
compromised tool implementation cannot exfiltrate through a capability
the caller's agent wasn't granted, and every check is logged.
"""
from __future__ import annotations
import fnmatch
from typing import Iterable

# Namespaces in use today:
#   net:*       outbound network
#       net:http_get        any GET request (http_fetch)
#       net:web_search      DuckDuckGo HTML scrape (web_search)
#       net:arxiv           arXiv API (arxiv_search)
#       net:anthropic       Anthropic API (implicit, via llm_call)
#   vault:*     Obsidian / local vault writes & reads
#       vault:read, vault:write
#   time:read   wall clock
#   mcp:<name>  call into a specific external MCP server's tools
ALL_CAPABILITIES: frozenset[str] = frozenset([
    "net:http_get", "net:web_search", "net:arxiv", "net:anthropic",
    "vault:read", "vault:write",
    "time:read",
])


def _parts(cap: str) -> list[str]:
    """Split a capability into up to 3 parts: namespace, action, resource."""
    return cap.split(":", 2)


def matches(granted: Iterable[str], required: str) -> bool:
    """True iff `granted` covers the single `required` capability."""
    g = set(granted)
    if "*" in g: return True
    if required in g: return True

    r = _parts(required)
    r_ns     = r[0] if len(r) > 0 else ""
    r_action = r[1] if len(r) > 1 else ""
    r_res    = r[2] if len(r) > 2 else ""

    # Namespace-only wildcard: "net:*" grants anything under net.
    if r_ns and f"{r_ns}:*" in g:
        return True

    for grant in g:
        gp = _parts(grant)
        if not gp or gp[0] != r_ns: continue

        # 1-part grant ("net") only matches if required is also 1-part — unusual.
        if len(gp) == 1:
            if len(r) == 1: return True
            continue

        # 2-part grant ("net:http_get") matches if action matches, regardless of resource.
        if len(gp) == 2:
            if gp[1] == r_action: return True
            if gp[1] == "*":      return True
            continue

        # 3-part grant ("net:http_get:host"): action + glob over resource.
        if gp[1] != r_action and gp[1] != "*":
            continue
        if not r_res:
            # Required is 2-part, grant is 3-part — too specific, skip.
            continue
        if fnmatch.fnmatchcase(r_res, gp[2]):
            return True

    return False


def covers(granted: Iterable[str], required: Iterable[str]) -> tuple[bool, list[str]]:
    """Return (ok, missing) — ok=True iff every required cap is granted."""
    granted_set = set(granted)
    required_list = list(required or [])
    missing = [r for r in required_list if not matches(granted_set, r)]
    return (not missing), missing


# Default capability matrix per agent. Override at the agent class level
# via the CAPABILITIES attribute. Atlas can dispatch to anything via
# mcp:* and uses tools only through its specialists, so we keep it tight.
DEFAULT_AGENT_CAPABILITIES: dict[str, frozenset[str]] = {
    "atlas":      frozenset({"time:read"}),
    "forge":      frozenset({"vault:read"}),
    "scout":      frozenset({"net:http_get", "net:web_search", "net:arxiv",
                             "vault:read", "vault:write"}),
    "vector":     frozenset({"net:http_get", "net:web_search"}),
    "sentinel":   frozenset({"time:read"}),
    "smith":      frozenset({"vault:read"}),
    "mirror":     frozenset({"vault:read", "vault:write"}),
}
