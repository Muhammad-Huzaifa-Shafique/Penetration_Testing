#!/usr/bin/env python3
"""
Casefile — auto-save watcher (Python fallback for autosave.sh).

Use this if you can't install inotify-tools. Same idea: watch the project
folder and, a few quiet seconds after any save, commit + push to GitHub.

Setup:
    pip install watchdog
    python3 autosave_watchdog.py                # watches current folder
    python3 autosave_watchdog.py /path/to/proj  # watches a specific folder

Stop with Ctrl-C.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    sys.exit("watchdog not installed. Run:  pip install watchdog")

REPO_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
BRANCH = "main"
DEBOUNCE = 8.0  # seconds of quiet before committing

IGNORE_PARTS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "chroma_db", ".idea", ".vscode",
}


def is_ignored(path: str) -> bool:
    parts = set(Path(path).parts)
    if parts & IGNORE_PARTS:
        return True
    name = Path(path).name
    return name.startswith("~$") or name.endswith((".pyc", ".swp"))


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_DIR,
        capture_output=True, text=True,
    )


def commit_and_push() -> None:
    status = git("status", "--porcelain").stdout.strip()
    if not status:
        return
    git("add", "-A")
    msg = f"autosave: {datetime.now():%Y-%m-%d %H:%M:%S}"
    if git("commit", "-q", "-m", msg).returncode != 0:
        return
    push = git("push", "-q", "origin", BRANCH)
    ts = datetime.now().strftime("%H:%M:%S")
    if push.returncode == 0:
        print(f"[autosave] pushed  {ts}")
    else:
        print(f"[autosave] committed locally, push failed ({push.stderr.strip()[:80]}) "
              f"— will retry on next change")


class Debouncer(FileSystemEventHandler):
    def __init__(self) -> None:
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_any_event(self, event) -> None:
        if event.is_directory or is_ignored(event.src_path):
            return
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE, commit_and_push)
            self._timer.daemon = True
            self._timer.start()


def main() -> None:
    if not (REPO_DIR / ".git").exists():
        sys.exit(f"{REPO_DIR} is not a git repo yet. Run:  bash setup_github.sh")

    print(f"[autosave] watching '{REPO_DIR}' — every save auto-pushes to '{BRANCH}'. Ctrl-C to stop.")
    commit_and_push()  # sweep up anything already pending

    observer = Observer()
    observer.schedule(Debouncer(), str(REPO_DIR), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[autosave] stopped.")
    observer.join()


if __name__ == "__main__":
    main()
