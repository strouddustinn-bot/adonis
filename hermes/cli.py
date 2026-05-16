"""
hermes/cli.py
==============
Tiny client for talking to the Hermes API.

Examples:
  python -m hermes.cli ask "Summarise the latest commit"
  python -m hermes.cli task "Research, then draft a 200-word post about ..."
  python -m hermes.cli health
  python -m hermes.cli ges
"""
import argparse
import json
import os
import sys

import httpx


def _base() -> str:
    return os.getenv("ADONIS_API", "http://localhost:8088").rstrip("/")


def main() -> int:
    p = argparse.ArgumentParser(prog="adonis")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("ask",   help="Single-turn ask")
    pa.add_argument("message")
    pa.add_argument("--session", default=None)

    pt = sub.add_parser("task",  help="Multi-agent goal (routed via Atlas)")
    pt.add_argument("goal")
    pt.add_argument("--session", default=None)
    pt.add_argument("--timeout", type=int, default=60)

    sub.add_parser("health", help="Runtime liveness + dependency status")
    sub.add_parser("ges",    help="Glasswing Efficiency Score report")
    pau = sub.add_parser("audit", help="Recent Prometheus fuse decisions")
    pau.add_argument("-n", type=int, default=20)

    args = p.parse_args()
    base = _base()

    try:
        with httpx.Client(timeout=120.0) as c:
            if args.cmd == "ask":
                body = {"message": args.message}
                if args.session: body["session_id"] = args.session
                r = c.post(f"{base}/ask", json=body)
            elif args.cmd == "task":
                body = {"goal": args.goal, "timeout_s": args.timeout}
                if args.session: body["session_id"] = args.session
                r = c.post(f"{base}/task", json=body)
            elif args.cmd == "health":
                r = c.get(f"{base}/health")
            elif args.cmd == "ges":
                r = c.get(f"{base}/ges")
            elif args.cmd == "audit":
                r = c.get(f"{base}/audit", params={"n": args.n})
            else:
                p.print_help(); return 2
            r.raise_for_status()
            print(json.dumps(r.json(), indent=2))
            return 0
    except httpx.HTTPStatusError as e:
        sys.stderr.write(f"HTTP {e.response.status_code}: {e.response.text}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
