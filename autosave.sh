#!/usr/bin/env bash
#
# Casefile — auto-save watcher.
# Leave this running in a terminal. Every time you save/create/delete a file in
# the project (a .py file, a report, anything not ignored), it waits a few quiet
# seconds and then commits + pushes to GitHub automatically.
#
# Usage:
#   bash autosave.sh                 # watches the current folder
#   bash autosave.sh /path/to/proj   # watches a specific folder
#   DEBOUNCE=15 bash autosave.sh      # wait 15s of quiet before committing
#
# Requires: inotify-tools   ->   sudo apt install inotify-tools
# (If you can't install it, use the Python fallback: python3 autosave_watchdog.py)
#
set -uo pipefail

REPO_DIR="${1:-$PWD}"
BRANCH="${BRANCH:-main}"
DEBOUNCE="${DEBOUNCE:-8}"     # seconds of no-activity before we commit
EXCLUDE='(/\.git/|__pycache__|\.venv|/venv/|/env/|chroma_db/|\.idea/|\.vscode/|~\$)'

cd "$REPO_DIR" || { echo "Can't cd into $REPO_DIR"; exit 1; }

command -v inotifywait >/dev/null 2>&1 || {
  echo "!! inotifywait not found. Install it:  sudo apt install inotify-tools"
  echo "   (or use the Python version:  python3 autosave_watchdog.py)"
  exit 1
}
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "!! $REPO_DIR is not a git repo yet. Run:  bash setup_github.sh"
  exit 1
}

commit_and_push() {
  # nothing changed? do nothing.
  [[ -z "$(git status --porcelain)" ]] && return 0
  git add -A
  git commit -q -m "autosave: $(date '+%Y-%m-%d %H:%M:%S')" || return 0
  if git push -q origin "$BRANCH" 2>/dev/null; then
    echo "[autosave] pushed  $(date '+%H:%M:%S')"
  else
    echo "[autosave] committed locally, push failed (offline? auth?) — will retry on next change"
  fi
}

echo "[autosave] watching '$REPO_DIR' — every save auto-pushes to '$BRANCH'. Ctrl-C to stop."
commit_and_push    # sweep up anything already pending right now

while true; do
  # Block until the first change happens...
  inotifywait -q -r -e close_write,create,move,delete \
      --exclude "$EXCLUDE" "$REPO_DIR" >/dev/null 2>&1 || true
  # ...then keep resetting a short timer while more changes keep arriving
  # (so a burst of saves = one commit, not fifty).
  while inotifywait -q -r -t "$DEBOUNCE" -e close_write,create,move,delete \
      --exclude "$EXCLUDE" "$REPO_DIR" >/dev/null 2>&1; do :; done
  commit_and_push
done
