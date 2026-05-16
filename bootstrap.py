#!/usr/bin/env python3
"""
bootstrap.py — Adonis interactive bootstrap + startup wizard.

Walks a fresh install from "git clone" to "Adonis answered me" in one
script. Idempotent: re-running detects what's already done and skips it.

Usage:
  python3 bootstrap.py            # full interactive wizard
  python3 bootstrap.py --start    # skip config, just (re)start the stack
  python3 bootstrap.py --status   # just probe health and exit
  python3 bootstrap.py --reset    # stop containers (keeps volumes)
  python3 bootstrap.py --nuke     # stop + remove volumes (DESTRUCTIVE)
"""
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from getpass import getpass
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
ENV      = ROOT / ".env"
EXAMPLE  = ROOT / ".env.example"
FUSE     = ROOT / "prometheus" / "fuse.py"
KEYS_DIR = Path.home() / "API Keys chmod 600"
DEFAULT_PORT = 8088

# Set in main() — True only when we have a real TTY and the user didn't pass -y.
# When False, ask_yn/ask quietly return their default and ask_secret aborts.
INTERACTIVE = True


# ── tiny output helpers ─────────────────────────────────────────────────────
def _c(s, code): return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s
def step(n, s):  print(_c(f"\n[{n}] {s}", "1;36"))
def ok(s):       print(_c(f"  [ok]   {s}", "32"))
def warn(s):     print(_c(f"  [warn] {s}", "33"))
def fail(s):     print(_c(f"  [err]  {s}", "31"))
def hint(s):     print(_c(f"         {s}", "90"))
def ask(s, default=None):
    if not INTERACTIVE:
        hint(f"(non-interactive) {s} -> '{default or ''}'")
        return default or ""
    suffix = f" [{default}]" if default else ""
    try:
        a = input(_c(f"  ?  {s}{suffix}: ", "35")).strip()
    except EOFError:
        warn(f"stdin closed while asking '{s}' — using default '{default or ''}'")
        return default or ""
    return a or (default or "")

def ask_secret(s):
    if not INTERACTIVE:
        fail(f"non-interactive run but '{s}' requires a secret value.")
        hint("Re-run from a TTY, or pre-populate .env before invoking the wizard.")
        sys.exit(1)
    try:
        return getpass(_c(f"  ?  {s}: ", "35")).strip()
    except EOFError:
        fail(f"stdin closed while asking for '{s}'.")
        sys.exit(1)

def ask_yn(s, default=True):
    if not INTERACTIVE:
        hint(f"(non-interactive) {s} -> {'yes' if default else 'no'}")
        return default
    d = "Y/n" if default else "y/N"
    try:
        a = input(_c(f"  ?  {s} [{d}]: ", "35")).strip().lower()
    except EOFError:
        warn(f"stdin closed while asking '{s}' — using default ({'yes' if default else 'no'})")
        return default
    if not a: return default
    return a in ("y", "yes")


# ── preflight ───────────────────────────────────────────────────────────────
def preflight() -> tuple[list[str], list[str]]:
    """Return (compose_cmd_prefix, warnings). Aborts on hard failure."""
    if sys.version_info < (3, 10):
        fail(f"Python 3.10+ required (have {sys.version.split()[0]}).")
        sys.exit(1)
    ok(f"Python {sys.version.split()[0]}")

    if not _which("docker"):
        fail("'docker' not on PATH. Install Docker Engine first.")
        sys.exit(1)

    rc, out, _ = _run(["docker", "info"], check=False)
    if rc != 0:
        fail("Docker daemon not reachable. Start it: 'sudo systemctl start docker' (Fedora/RHEL).")
        hint("Or add yourself to the docker group: 'sudo usermod -aG docker $USER' then re-login.")
        sys.exit(1)
    ok("Docker daemon reachable")

    rc, _, _ = _run(["docker", "compose", "version"], check=False)
    if rc == 0:
        compose = ["docker", "compose"]
    else:
        if not _which("docker-compose"):
            fail("Neither 'docker compose' nor 'docker-compose' is available.")
            sys.exit(1)
        compose = ["docker-compose"]
    ok(f"Compose: '{' '.join(compose)}'")

    warnings = []
    port = int(_read_env_value("HERMES_PORT") or DEFAULT_PORT)
    if not _port_free(port):
        warnings.append(f"Port {port} is in use. Change HERMES_PORT in .env, or free the port.")
        warn(warnings[-1])
    else:
        ok(f"Port {port} is free")

    return compose, warnings


# ── .env handling ───────────────────────────────────────────────────────────
def _read_env_value(key: str) -> str | None:
    if not ENV.exists(): return None
    for line in ENV.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


def _set_env_values(updates: dict[str, str]) -> None:
    """Patch .env in place: replace existing keys, append new ones, keep comments."""
    if not ENV.exists():
        ENV.write_text(EXAMPLE.read_text() if EXAMPLE.exists() else "")
    lines = ENV.read_text().splitlines()
    seen = set()
    for i, line in enumerate(lines):
        if not line or line.startswith("#") or "=" not in line: continue
        k = line.split("=", 1)[0]
        if k in updates:
            lines[i] = f"{k}={updates[k]}"
            seen.add(k)
    for k, v in updates.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    ENV.write_text("\n".join(lines) + "\n")


def _find_anthropic_key_in_keys_dir() -> str | None:
    if not KEYS_DIR.is_dir(): return None
    cand = list(KEYS_DIR.glob("*ANTHROPIC*")) + list(KEYS_DIR.glob("*adonis*"))
    seen = set()
    for f in cand:
        if not f.is_file() or f in seen: continue
        seen.add(f)
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if "ANTHROPIC" in k.upper() and v.strip().startswith("sk-ant-"):
                    return v.strip().strip('"').strip("'")
        m = re.search(r"sk-ant-[A-Za-z0-9_\-]+", text)
        if m:
            return m.group(0)
    return None


def configure_env() -> None:
    if not ENV.exists():
        if not EXAMPLE.exists():
            fail(".env.example missing — can't seed .env.")
            sys.exit(1)
        ENV.write_text(EXAMPLE.read_text())
        ok("Created .env from .env.example")

    updates: dict[str, str] = {}

    # ── ANTHROPIC_API_KEY ─────────────────────────────────────────────────
    current = _read_env_value("ANTHROPIC_API_KEY") or ""
    needs_key = (not current
                 or "placeholder" in current
                 or not current.startswith("sk-ant-"))
    if needs_key:
        warn("ANTHROPIC_API_KEY is missing or a placeholder.")
        key = _find_anthropic_key_in_keys_dir()
        if key:
            ok(f"Found a real key in '{KEYS_DIR}'.")
            if ask_yn("Use that key?", default=True):
                updates["ANTHROPIC_API_KEY"] = key
        if "ANTHROPIC_API_KEY" not in updates:
            entered = ask_secret("Paste your Anthropic API key (input hidden)")
            if not entered.startswith("sk-ant-"):
                fail("That doesn't look like an Anthropic key (must start with 'sk-ant-').")
                sys.exit(1)
            updates["ANTHROPIC_API_KEY"] = entered
    else:
        ok("ANTHROPIC_API_KEY already set.")

    # ── ADONIS_MODEL ──────────────────────────────────────────────────────
    model = _read_env_value("ADONIS_MODEL") or "claude-sonnet-4-6"
    if not _read_env_value("ADONIS_MODEL"):
        updates["ADONIS_MODEL"] = model
    ok(f"Model: {model}")

    # ── PROMETHEUS_HASH ───────────────────────────────────────────────────
    if not FUSE.exists():
        fail("prometheus/fuse.py is missing — cannot compute boot hash.")
        sys.exit(1)
    real = hashlib.sha256(FUSE.read_bytes()).hexdigest()
    current_hash = _read_env_value("PROMETHEUS_HASH") or "UNSET"
    if current_hash == "UNSET":
        warn("PROMETHEUS_HASH is UNSET — boot will warn but continue.")
        if ask_yn(f"Pin it to the current fuse.py hash ({real[:12]}...)?", default=True):
            updates["PROMETHEUS_HASH"] = real
            ok("Hash pinned.")
    elif current_hash != real:
        warn("PROMETHEUS_HASH does NOT match the current fuse.py — boot will FAIL.")
        hint(f"  expected: {current_hash}")
        hint(f"  actual:   {real}")
        if ask_yn("Repin to the current hash? (only if you intentionally edited fuse.py)", default=False):
            updates["PROMETHEUS_HASH"] = real
            ok("Hash repinned.")
        else:
            fail("Aborting — please restore prometheus/fuse.py or repin manually.")
            sys.exit(1)
    else:
        ok(f"PROMETHEUS_HASH matches fuse.py ({real[:12]}...)")

    # ── OBSIDIAN (optional) ───────────────────────────────────────────────
    have_obs = _read_env_value("OBSIDIAN_API")
    if not have_obs:
        if ask_yn("Enable Obsidian long-term memory? (needs the local-rest-api plugin in your vault)", default=False):
            api = ask("OBSIDIAN_API URL", default="http://host.docker.internal:27123")
            tok = ask_secret("OBSIDIAN_TOKEN (input hidden, blank to skip)")
            if api:
                updates["OBSIDIAN_API"] = api
                if tok: updates["OBSIDIAN_TOKEN"] = tok

    # ── MCP_SERVERS (optional) ────────────────────────────────────────────
    if not _read_env_value("MCP_SERVERS"):
        if ask_yn("Configure external MCP servers now? (you can do this later)", default=False):
            example = '[{"name":"fs","command":["mcp-server-filesystem","--root","/vault"]}]'
            entered = ask(f"Paste MCP_SERVERS JSON (or blank to skip).\n      Example: {example}\n     ", default="")
            if entered:
                try:
                    json.loads(entered)
                    updates["MCP_SERVERS"] = entered
                except Exception:
                    warn("Not valid JSON — skipped.")

    if updates:
        _set_env_values(updates)
        ok(f"Wrote {len(updates)} value(s) to .env")
    else:
        ok(".env is already complete.")


# ── docker lifecycle ────────────────────────────────────────────────────────
def compose_up(compose: list[str], rebuild: bool = True) -> None:
    cmd = compose + ["up", "-d"] + (["--build"] if rebuild else [])
    rc, _, _ = _run(cmd, stream=True)
    if rc != 0:
        fail("docker compose up failed. Inspect the build output above.")
        sys.exit(1)
    ok("Containers up.")


def compose_down(compose: list[str], wipe: bool = False) -> None:
    cmd = compose + ["down"] + (["-v"] if wipe else [])
    _run(cmd, stream=True)


# ── boot wait + sanity ping ─────────────────────────────────────────────────
def wait_for_boot(port: int, timeout_s: int = 120) -> dict:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                body = r.read().decode()
                return json.loads(body)
        except urllib.error.URLError as e:
            last_err = str(e)
        except Exception as e:
            last_err = str(e)
        time.sleep(1)
    fail(f"Hermes API did not respond on :{port} within {timeout_s}s. Last error: {last_err}")
    hint("Run: docker compose logs --tail=200 adonis")
    sys.exit(1)


def sanity_ping(port: int) -> bool:
    url = f"http://localhost:{port}/ask"
    body = json.dumps({"message": "Say hello in 5 words.", "max_tokens": 50}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        fail(f"Sanity ping HTTP {e.code}: {e.read().decode()[:300]}")
        return False
    except Exception as e:
        fail(f"Sanity ping failed: {e}")
        return False
    answer = (payload.get("answer") or "").strip()
    if not answer:
        fail("Sanity ping returned empty answer. Check the API key.")
        return False
    ok(f"Adonis answered: {answer[:120]}")
    return True


# ── tiny utils ──────────────────────────────────────────────────────────────
def _which(prog: str) -> str | None:
    rc, out, _ = _run(["which", prog], check=False)
    return out.strip() if rc == 0 and out.strip() else None


def _run(cmd: list[str], check: bool = False, stream: bool = False) -> tuple[int, str, str]:
    if stream:
        rc = subprocess.call(cmd, cwd=str(ROOT))
        return rc, "", ""
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if check and p.returncode != 0:
        fail(f"command failed: {' '.join(cmd)}\n{p.stderr}")
        sys.exit(p.returncode)
    return p.returncode, p.stdout, p.stderr


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _print_health(report: dict) -> None:
    print()
    print(_c("  Service status:", "1"))
    for k in ("redis", "chroma", "obsidian"):
        v = report.get(k, "?")
        marker = "ok" if (v == "ok" or (isinstance(v, dict) and v.get("ok"))) else "off"
        line = f"    {k:9s} {v}"
        print(_c(line, "32" if marker == "ok" else "33"))
    fae = report.get("fuse_audit_entries", -1)
    print(_c(f"    fuse_audit_entries  {fae}", "90"))


def _print_quickstart(port: int) -> None:
    print()
    print(_c("Adonis is alive. Try:", "1;32"))
    print(_c(f"  python3 -m hermes.cli ask    \"introduce yourself in one sentence\"", "0"))
    print(_c(f"  python3 -m hermes.cli task   \"research agentic AI safety, then draft a 200-word post\"", "0"))
    print(_c(f"  python3 -m hermes.cli health", "0"))
    print(_c(f"  python3 -m hermes.cli ges", "0"))
    print()
    print(_c("Or talk to it via curl:", "1"))
    print(_c(f"  curl -s localhost:{port}/health | python3 -m json.tool", "0"))
    print()
    print(_c("Logs:    docker compose logs -f adonis", "90"))
    print(_c("Stop:    python3 bootstrap.py --reset", "90"))
    print(_c("Wipe:    python3 bootstrap.py --nuke   (DESTROYS Redis + Chroma data)", "90"))


# ── modes ───────────────────────────────────────────────────────────────────
def mode_full() -> None:
    print(_c("\nAdonis bootstrap wizard\n=======================", "1;35"))
    step(1, "Preflight checks")
    compose, _ = preflight()

    step(2, "Configure .env")
    configure_env()

    step(3, "Build + start the stack")
    compose_up(compose, rebuild=True)

    port = int(_read_env_value("HERMES_PORT") or DEFAULT_PORT)
    step(4, f"Wait for Hermes API on :{port}")
    report = wait_for_boot(port)
    ok("Health endpoint is up.")
    _print_health(report)

    step(5, "Sanity ping (one tiny LLM call)")
    sanity_ping(port)

    _print_quickstart(port)


def mode_start() -> None:
    compose, _ = preflight()
    compose_up(compose, rebuild=False)
    port = int(_read_env_value("HERMES_PORT") or DEFAULT_PORT)
    report = wait_for_boot(port)
    _print_health(report)
    _print_quickstart(port)


def mode_status() -> None:
    port = int(_read_env_value("HERMES_PORT") or DEFAULT_PORT)
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3) as r:
            print(r.read().decode())
    except Exception as e:
        fail(f"Hermes not reachable on :{port} ({e})")
        sys.exit(1)


def mode_reset(wipe: bool) -> None:
    compose, _ = preflight()
    if wipe:
        warn("This will DELETE Redis + Chroma volumes (all cached context + L3 vectors).")
        if not ask_yn("Are you sure?", default=False):
            print("Aborted."); return
    compose_down(compose, wipe=wipe)
    ok("Stopped.")


def main() -> int:
    p = argparse.ArgumentParser(description="Adonis bootstrap + startup wizard")
    p.add_argument("-y", "--yes", action="store_true",
                   help="non-interactive: accept defaults for every optional prompt")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--start",  action="store_true", help="skip config, just start the stack")
    g.add_argument("--status", action="store_true", help="probe /health and exit")
    g.add_argument("--reset",  action="store_true", help="stop containers (keep volumes)")
    g.add_argument("--nuke",   action="store_true", help="stop + delete volumes (DESTRUCTIVE)")
    args = p.parse_args()

    global INTERACTIVE
    INTERACTIVE = sys.stdin.isatty() and not args.yes
    if not INTERACTIVE:
        warn("running non-interactively — all optional prompts will accept their defaults.")

    try:
        if args.status: mode_status()
        elif args.start: mode_start()
        elif args.reset: mode_reset(wipe=False)
        elif args.nuke:  mode_reset(wipe=True)
        else:            mode_full()
    except KeyboardInterrupt:
        print()
        fail("Interrupted.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
