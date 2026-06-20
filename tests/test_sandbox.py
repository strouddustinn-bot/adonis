"""
tests/test_sandbox.py
=====================
MCP plugin containment: secret-free environment, command allowlisting,
optional wrapping/limits. These are the controls between a third-party MCP
subprocess and the rest of the host.
"""
import pytest

from tools import sandbox


def test_env_excludes_parent_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("OBSIDIAN_TOKEN", "tok")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = sandbox.build_subprocess_env({"PLUGIN_VAR": "1"})
    assert "ANTHROPIC_API_KEY" not in env
    assert "OBSIDIAN_TOKEN" not in env
    assert env.get("PATH") == "/usr/bin"
    assert env["PLUGIN_VAR"] == "1"


def test_env_passthrough_allowlist(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("MY_SAFE_VAR", "yes")
    monkeypatch.setenv("MCP_ENV_PASSTHROUGH", "MY_SAFE_VAR")
    env = sandbox.build_subprocess_env(None)
    assert env["MY_SAFE_VAR"] == "yes"
    assert "ANTHROPIC_API_KEY" not in env


def test_server_env_overrides_base(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    env = sandbox.build_subprocess_env({"PATH": "/custom/bin"})
    assert env["PATH"] == "/custom/bin"


def test_allowlist_unset_allows_all(monkeypatch):
    monkeypatch.delenv("MCP_ALLOWED_COMMANDS", raising=False)
    sandbox.enforce_command_allowlist(["anything", "--flag"])  # no raise


def test_allowlist_blocks_unlisted(monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_COMMANDS", "mcp-server-github,mcp-server-fs")
    with pytest.raises(PermissionError):
        sandbox.enforce_command_allowlist(["/usr/bin/curl"])


def test_allowlist_matches_basename(monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_COMMANDS", "mcp-server-fs")
    sandbox.enforce_command_allowlist(["/opt/bin/mcp-server-fs", "--root", "/vault"])


def test_wrap_command_off_is_passthrough(monkeypatch):
    monkeypatch.delenv("MCP_SANDBOX", raising=False)
    cmd = ["mcp-server-fs", "--root", "/vault"]
    assert sandbox.wrap_command(cmd) == cmd


def test_wrap_command_strict_missing_bwrap_raises(monkeypatch):
    monkeypatch.setenv("MCP_SANDBOX", "bwrap")
    monkeypatch.setenv("MCP_SANDBOX_STRICT", "1")
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError):
        sandbox.wrap_command(["x"])


def test_wrap_command_bwrap_wraps_when_available(monkeypatch):
    monkeypatch.setenv("MCP_SANDBOX", "bwrap")
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/bwrap")
    out = sandbox.wrap_command(["mcp-server-fs"])
    assert out[0] == "/usr/bin/bwrap"
    assert "--die-with-parent" in out
    assert out[-1] == "mcp-server-fs"


def test_resource_limiter_none_when_unset(monkeypatch):
    for k in ("MCP_RLIMIT_CPU", "MCP_RLIMIT_AS_MB", "MCP_RLIMIT_NOFILE"):
        monkeypatch.delenv(k, raising=False)
    assert sandbox.resource_limiter() is None


def test_resource_limiter_callable_when_set(monkeypatch):
    monkeypatch.setenv("MCP_RLIMIT_CPU", "30")
    fn = sandbox.resource_limiter()
    assert callable(fn)
