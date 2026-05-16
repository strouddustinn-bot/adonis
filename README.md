# Adonis

A self-improving, ethically-governed AI agent runtime in Python.

Adonis is a single-process async system that routes every LLM call through five efficiency layers, gates every action through a tamper-protected ethics circuit, learns from its own wins and losses, and proposes prompt rewrites for its underperforming agents on a recurring cycle. It runs locally on Redis + ChromaDB + an Obsidian vault, behind a small HTTP API.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/strouddustinn-bot/adonis/main/install.sh | bash
```

The installer clones into `~/adonis` (override with `ADONIS_DIR=...`), then launches an interactive bootstrap wizard that:

1. Verifies prerequisites (Python ≥ 3.10, Docker, Compose).
2. Helps you populate `.env` — auto-detects an Anthropic key in `~/API Keys chmod 600/` if present, pins the Prometheus fuse hash, asks before enabling Obsidian + MCP.
3. Runs `docker compose up -d --build`.
4. Waits for the Hermes API to come up, prints dependency status, and runs a sanity ping so you know the API key works end-to-end.

After install, talk to it:

```bash
python3 -m hermes.cli ask  "introduce yourself in one sentence"
python3 -m hermes.cli task "research agentic AI safety, then draft a 200-word post"
python3 -m hermes.cli health
python3 -m hermes.cli ges
```

Or via HTTP on `localhost:8088`:

```bash
curl -s -X POST localhost:8088/ask \
  -H 'Content-Type: application/json' \
  -d '{"message":"introduce yourself"}'
```

## What's in the box

```
hermes/        HTTP entrypoint (FastAPI) + CLI client
glasswing/     per-call efficiency pipeline (router → context → soul → cache → GES)
context/       four-tier hierarchical memory (Redis x3 + Chroma + Obsidian)
compression/   semantic-atom compression (Qwen-style dense token packing)
persona/       hash-protected soul layer (model-family adapters)
routing/       MoE agent router (Qwen3-30B-A3B-style top-K activation)
prometheus/    ethical fuse + boot-integrity gate (do not edit casually)
openclaw/      agents: atlas, mirror, smith, forge, scout, vector, sentinel
tools/         tool registry + builtin tools + minimal MCP stdio client
memory/        Obsidian bridge (long-term vault)
supervisord.py async runtime supervisor
bootstrap.py   interactive setup wizard
```

## Architecture in one paragraph

Every user message hits Hermes (`/ask`), which calls Glasswing's `prepare()`. Glasswing checks the response cache (≈200ms, often no LLM call), routes the task through the MoE router to select the minimum active agents and think-depth, builds a context window from the four-tier memory (Redis hot / Redis episode / Redis archive / Chroma+Obsidian semantic), injects the hash-locked Adonis soul, then hands the call to Claude. After each response, Glasswing records a Glasswing Efficiency Score (GES). For multi-agent goals, Hermes publishes to Atlas, which decomposes into subtasks and dispatches specialists over Redis pub/sub. Every external action — agent code patches, content publication, tool calls — is gated through the Prometheus fuse, which scores intent across six axes (harm, deception, exfiltration, autonomy override, legal exposure, cascade risk) and either approves, audits, remediates, blocks, or hard-kills the agent depending on the resulting level. Wins go to a Redis log + occasionally distilled into the semantic vault; losses go to Smith for patch generation; GES drives Mirror's daily self-improvement cycle.

## Safety model

The fuse (`prometheus/fuse.py`) is the load-bearing safety code. The boot gate (`boot_check.py`) hashes it at startup and refuses to launch if it has been tampered with. The Docker container mounts `./prometheus` read-only, so the file cannot be modified from inside the running agent.

Levels:

| Score 0–60 | Level  | Action |
|-----------:|:-------|:-------|
|   0–10     | GREEN  | approved silently |
|  11–25     | YELLOW | approved + audited |
|  26–40     | ORANGE | requires human confirmation |
|  41–55     | RED    | auto-remediation attempted, audited |
|  56+       | BLACK  | hard kill — agent locked, operator release required |

If you edit `prometheus/fuse.py`, the SHA-256 in `.env`'s `PROMETHEUS_HASH` must be updated. The bootstrap wizard offers to repin for you.

## Memory

| Tier | Backend | TTL | Fidelity |
|-----:|:--------|:----|:---------|
| L0   | Redis   | 1 h | verbatim (~10K tokens) |
| L1   | Redis   | 7 d | standard compression (~5K tokens, ~50% reduction) |
| L2   | Redis   | 30 d | ultra compression (~2K tokens, ~90% reduction) |
| L3   | Chroma + Obsidian | permanent | semantic atoms, retrieved on demand |

Compression is LLM-driven: each block is distilled to subject-predicate-object atoms plus a small causal edge graph, then re-expanded to any token budget at retrieval time. Cascades L0→L1→L2 happen automatically on overflow. L3 retrieval is top-K semantic search at query time.

## Tools

Adonis ships with a small built-in tool catalog (`http_fetch`, `web_search` via DuckDuckGo, `arxiv_search`, `vault_read`, `vault_append`, `now`). Any external MCP server can be added via the `MCP_SERVERS` JSON in `.env`. Every tool call is gated through Prometheus.

## Self-improvement

- **Mirror** runs a recurring cycle: collect GES → identify worst-performing agent → propose a prompt rewrite → benchmark old vs new → log to the Obsidian vault for human review. (By default, Mirror does *not* edit source files — proposals go to `SELF/improvement_queue.md` in the vault. Flipping this to actually rewrite agent code is one fuse-gated edit away.)
- **Smith** classifies every failure (syntax / logic / tool / timeout / prometheus-block), generates a fix, gates it through the fuse, and records the pattern.
- **Wins and losses** are logged per agent. Every 10th win is distilled into the L3 semantic vault, so future similar tasks can retrieve prior winning approaches at routing time.

## Run modes

```bash
python3 bootstrap.py            # full interactive wizard
python3 bootstrap.py --start    # skip config, just (re)start the stack
python3 bootstrap.py --status   # probe /health and exit
python3 bootstrap.py --reset    # stop containers (keep volumes)
python3 bootstrap.py --nuke     # stop + delete Redis/Chroma volumes
```

## Configuration

All settings live in `.env`. The bootstrap wizard fills these in, but for reference:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (required) |
| `ADONIS_MODEL`      | Default model id (default `claude-sonnet-4-6`) |
| `PROMETHEUS_HASH`   | SHA-256 of `prometheus/fuse.py` — tamper gate |
| `PROMETHEUS_ALERT_WEBHOOK` | Optional ops webhook for RED/BLACK decisions |
| `HERMES_PORT`       | HTTP port for the Hermes API (default 8088) |
| `OBSIDIAN_API`      | Obsidian local-rest-api URL (blank = disabled) |
| `OBSIDIAN_TOKEN`    | Obsidian local-rest-api bearer token |
| `MCP_SERVERS`       | JSON list of external MCP servers to attach |
| `LOG_LEVEL`         | `DEBUG` / `INFO` / `WARNING` (default `INFO`) |

## Contributing

PRs welcome. If your change touches `prometheus/fuse.py`, expect extra scrutiny — that file is the only thing standing between the agents and the rest of the world.

## License

MIT — see [LICENSE](LICENSE).
