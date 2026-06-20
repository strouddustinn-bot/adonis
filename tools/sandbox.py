"""
tools/sandbox.py
================
Process-isolation helpers for external MCP server subprocesses.

An MCP server is a child process Adonis spawns from the MCP_SERVERS JSON.
Without containment it inherits Adonis's *entire* environment — including
ANTHROPIC_API_KEY and any other secret in os.environ — plus unrestricted
filesystem and resource access. This module narrows that blast radius. Every
control degrades gracefully and is off (or permissive) by default so existing
deployments keep working; tighten via env vars:

  build_subprocess_env   — pass ONLY an allowlisted base env (PATH/HOME/…)
                           plus the server's explicit `env`. Secrets in
                           os.environ never reach a plugin unless the plugin's
                           own `env` names them. (Always on — the key fix.)
  enforce_command_allowlist — MCP_ALLOWED_COMMANDS gates which executables may
                           be launched (basename or full path). Empty = allow
                           all (back-compat).
  wrap_command           — MCP_SANDBOX=bwrap wraps the command in bubblewrap
                           (read-only root, private /tmp, unshared namespaces).
                           MCP_SANDBOX_NONET=1 also unshares the network.
  resource_limiter       — optional POSIX rlimits (CPU s / address space MB /
                           open files) applied in the child via preexec_fn.

Set MCP_SANDBOX_STRICT=1 to make a missing backend a hard error instead of a
logged warning.
"""
import logging
import os
import shutil

log = logging.getLogger("tools.sandbox")

# Only these keys cross into a plugin's environment by default. Notably absent:
# ANTHROPIC_API_KEY, OBSIDIAN_TOKEN, TELEGRAM_BOT_TOKEN, PROMETHEUS_* …
_BASE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ", "TERM")


def _strict() -> bool:
    return os.getenv("MCP_SANDBOX_STRICT", "0") == "1"


def build_subprocess_env(server_env: dict | None) -> dict:
    """Construct a minimal environment for an MCP subprocess: an allowlisted
    base, plus any keys named in MCP_ENV_PASSTHROUGH, plus the server's own
    explicit `env` (which always wins). The parent's secrets are excluded."""
    env: dict = {k: os.environ[k] for k in _BASE_ENV_KEYS if k in os.environ}
    for k in (x.strip() for x in os.getenv("MCP_ENV_PASSTHROUGH", "").split(",")):
        if k and k in os.environ:
            env[k] = os.environ[k]
    env.update(server_env or {})
    return env


def enforce_command_allowlist(command: list[str]) -> None:
    """Raise PermissionError if MCP_ALLOWED_COMMANDS is set and `command`'s
    executable is not on it (matched by basename or full path). No allowlist
    configured → allow everything (back-compat)."""
    allow = [x.strip() for x in os.getenv("MCP_ALLOWED_COMMANDS", "").split(",") if x.strip()]
    if not allow:
        return
    if not command:
        raise PermissionError("empty MCP command")
    exe = command[0]
    if exe not in allow and os.path.basename(exe) not in allow:
        raise PermissionError(
            f"MCP command '{exe}' is not in MCP_ALLOWED_COMMANDS ({allow})")


def wrap_command(command: list[str]) -> list[str]:
    """Optionally wrap `command` for filesystem/namespace isolation. Controlled
    by MCP_SANDBOX: unset/none → unchanged; bwrap → bubblewrap profile."""
    mode = os.getenv("MCP_SANDBOX", "").strip().lower()
    if mode in ("", "none", "off"):
        return command
    if mode == "bwrap":
        bwrap = shutil.which("bwrap")
        if not bwrap:
            msg = "MCP_SANDBOX=bwrap requested but bubblewrap (bwrap) not found"
            if _strict():
                raise RuntimeError(msg)
            log.warning("%s; running unwrapped.", msg)
            return command
        wrapped = [
            bwrap,
            "--die-with-parent",
            "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--ro-bind", "/", "/",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
        ]
        if os.getenv("MCP_SANDBOX_NONET", "0") == "1":
            wrapped.append("--unshare-net")
        return wrapped + command
    msg = f"unknown MCP_SANDBOX mode '{mode}'"
    if _strict():
        raise RuntimeError(msg)
    log.warning("%s; running unwrapped.", msg)
    return command


def resource_limiter():
    """Return a preexec_fn applying POSIX rlimits, or None if none configured
    or the platform lacks `resource` (e.g. Windows)."""
    cpu    = os.getenv("MCP_RLIMIT_CPU", "")
    mem_mb = os.getenv("MCP_RLIMIT_AS_MB", "")
    nofile = os.getenv("MCP_RLIMIT_NOFILE", "")
    if not any([cpu, mem_mb, nofile]):
        return None
    try:
        import resource
    except ImportError:
        log.warning("rlimits requested but `resource` unavailable on this platform.")
        return None

    def _apply():
        if cpu.isdigit():
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu), int(cpu)))
        if mem_mb.isdigit():
            b = int(mem_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
        if nofile.isdigit():
            resource.setrlimit(resource.RLIMIT_NOFILE, (int(nofile), int(nofile)))

    return _apply
