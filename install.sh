#!/usr/bin/env bash
# install.sh — one-line installer for Adonis.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/strouddustinn-bot/adonis/main/install.sh | bash
#
# Environment overrides:
#   ADONIS_DIR     install location (default: $HOME/adonis)
#   ADONIS_BRANCH  branch to clone   (default: main)
#   ADONIS_REPO    git URL           (default: github.com/strouddustinn-bot/adonis.git)
#
# Re-running this script in an existing checkout pulls the latest and
# re-launches the bootstrap wizard.

set -euo pipefail

REPO="${ADONIS_REPO:-https://github.com/strouddustinn-bot/adonis.git}"
BRANCH="${ADONIS_BRANCH:-main}"
TARGET="${ADONIS_DIR:-$HOME/adonis}"

c() { printf '\033[%sm%s\033[0m\n' "$1" "$2"; }
say()  { c "1;36" "$*"; }
ok()   { c "32"   "[ok]  $*"; }
err()  { c "1;31" "[err] $*" >&2; }

say "Adonis installer"

for cmd in git python3 docker; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    err "missing prerequisite: $cmd"
    err "install it and re-run this script."
    exit 1
  fi
done
ok "prerequisites present (git, python3, docker)"

if [ -d "$TARGET/.git" ]; then
  say "Existing checkout at $TARGET — fetching latest on '$BRANCH'..."
  cd "$TARGET"
  git fetch --depth 1 origin "$BRANCH"
  git checkout "$BRANCH"
  git reset --hard "origin/$BRANCH"
elif [ -e "$TARGET" ]; then
  err "$TARGET already exists and is not a git checkout. Move it aside or set ADONIS_DIR."
  exit 1
else
  say "Cloning $REPO -> $TARGET (branch: $BRANCH) ..."
  git clone --branch "$BRANCH" --depth 1 "$REPO" "$TARGET"
  cd "$TARGET"
fi
ok "source ready at $TARGET"

say "Launching the bootstrap wizard..."
exec python3 bootstrap.py "$@"
