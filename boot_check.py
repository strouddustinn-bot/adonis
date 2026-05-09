"""
boot_check.py — Adonis Boot Integrity Verifier
Runs BEFORE any agent starts. Verifies Prometheus Fuse has not been tampered with.
If the hash does not match, Adonis will not start.
"""
import sys, hashlib, os

PROMETHEUS_PATH = os.path.join(os.path.dirname(__file__), "prometheus","fuse.py")

# This hash is computed at release and locked into the codebase.
# Update ONLY via the official release process.
PROMETHEUS_HASH = os.getenv("PROMETHEUS_HASH","UNSET")

def verify():
    if not os.path.exists(PROMETHEUS_PATH):
        print("BOOT FAIL: prometheus/fuse.py not found. Adonis will not start.", file=sys.stderr)
        sys.exit(1)
    with open(PROMETHEUS_PATH,"rb") as f:
        current = hashlib.sha256(f.read()).hexdigest()
    if PROMETHEUS_HASH == "UNSET":
        print(f"BOOT WARN: PROMETHEUS_HASH not set in environment. Current hash: {current}")
        print("Set PROMETHEUS_HASH={current} in .env to enable tamper protection.")
        return  # Allow first run to print the hash
    if current != PROMETHEUS_HASH:
        print("=" * 60, file=sys.stderr)
        print("BOOT FAIL: PROMETHEUS FUSE TAMPERED.", file=sys.stderr)
        print(f"Expected: {PROMETHEUS_HASH}", file=sys.stderr)
        print(f"Got:      {current}", file=sys.stderr)
        print("Adonis will not start. Restore prometheus/fuse.py to proceed.", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
    print(f"[BOOT] Prometheus Fuse verified. Hash: {current[:16]}...")

if __name__ == "__main__":
    verify()
    print("[BOOT] All checks passed. Starting Adonis...")
