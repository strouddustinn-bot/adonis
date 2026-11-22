# Adonis — repository instructions

## Purpose
Python agent runtime with efficiency layers, an ethics circuit, learning loops, and an HTTP interface. The project README describes Redis, ChromaDB, an Obsidian vault, Docker/Compose, and bootstrap tooling; verify the active implementation before extending it.

## Operating constraints
- Preserve action authorization and safety checks independently of LLM-generated suggestions.
- Do not weaken the ethics circuit, bypass approval gates, or let generated content execute unverified commands.
- Avoid leaking environment variables, model credentials, vault contents, or user memory into logs, commits, or prompts.
- Treat self-modifying prompts or tools as proposals that need tests and explicit authorization where they affect persistent code or external systems.
- Never run destructive bootstrap/reset modes or deploy production resources as routine validation.

## Navigation and verification
Start with `README.md`, `bootstrap.py`, `install.sh`, and the `hermes/` module. The README documents `python3 bootstrap.py --status` for checking the configured stack and `python3 -m hermes.cli health` for an agent health check. These are environment-dependent; report when dependencies or credentials are unavailable. Prefer targeted offline checks and tests for the changed component.

## Development expectations
Verify actual dependencies and interfaces before choosing commands. Keep external service effects explicit, reversible, and approved. Document what was tested; do not claim the agent's safety or autonomy is proved by a demonstration.
