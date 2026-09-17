#!/usr/bin/env bash
#
# Casefile — one-time GitHub setup.
# Run this ONCE from the ROOT of your project folder (the folder that holds
# your weekly code, corpus/, venv/, etc.). It will:
#   1. make the folder a git repo (if it isn't already)
#   2. drop in the .gitignore (so venv/, chroma_db/, the huge corpus are skipped)
#   3. commit everything you have so far ("all the past")
#   4. create a repo on your GitHub account and push to it
#
# Usage:
#   bash setup_github.sh                 # repo name defaults to "casefile", private
#   bash setup_github.sh my-repo-name    # custom repo name
#   VISIBILITY=public bash setup_github.sh
#
set -euo pipefail

REPO_NAME="${1:-casefile}"
VISIBILITY="${VISIBILITY:-private}"     # private | public
BRANCH="main"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo " Casefile GitHub setup"
echo " repo name : $REPO_NAME  ($VISIBILITY)"
echo " folder    : $PWD"
echo "=================================================="

# --- 1. prerequisites -------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
  echo "!! git is not installed.  Install it with:  sudo apt update && sudo apt install git"
  exit 1
fi

# --- 2. copy .gitignore in beside your code (if not already there) ----------
if [[ ! -f .gitignore && -f "$SCRIPT_DIR/.gitignore" ]]; then
  cp "$SCRIPT_DIR/.gitignore" .gitignore
  echo "[setup] added .gitignore"
fi

# --- 3. init the repo -------------------------------------------------------
if [[ ! -d .git ]]; then
  git init -b "$BRANCH" >/dev/null
  echo "[setup] initialised empty git repo on branch '$BRANCH'"
else
  echo "[setup] already a git repo — continuing"
  git symbolic-ref -q HEAD "refs/heads/$BRANCH" 2>/dev/null || git branch -M "$BRANCH" 2>/dev/null || true
fi

# --- 4. identity (only sets it if you haven't already) ----------------------
git config user.name  >/dev/null 2>&1 || git config user.name  "Muhammad Huzaifa Shafique"
git config user.email >/dev/null 2>&1 || git config user.email "huzaifacheema123@gmail.com"

# --- 5. first commit of everything you have --------------------------------
git add -A
if git diff --cached --quiet; then
  echo "[setup] nothing new to commit"
else
  git commit -q -m "Initial commit: Casefile project (all work to date)"
  echo "[setup] committed your current project"
fi

# --- 6. create the GitHub repo and push ------------------------------------
if command -v gh >/dev/null 2>&1; then
  # GitHub CLI is the easy path — it handles auth for you.
  if ! gh auth status >/dev/null 2>&1; then
    echo "[setup] you're not logged in to GitHub CLI. Launching login..."
    echo "        (choose GitHub.com  ->  HTTPS  ->  login with a browser)"
    gh auth login
  fi

  if gh repo view "$REPO_NAME" >/dev/null 2>&1; then
    echo "[setup] repo '$REPO_NAME' already exists on your account — linking to it"
    git remote get-url origin >/dev/null 2>&1 || \
      git remote add origin "$(gh repo view "$REPO_NAME" --json url -q .url).git"
    git push -u origin "$BRANCH"
  else
    echo "[setup] creating $VISIBILITY repo '$REPO_NAME' on your account and pushing..."
    gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
  fi
  echo
  echo "[setup] DONE ✅  Your project is on GitHub:"
  gh repo view "$REPO_NAME" --json url -q .url
else
  # No gh — fall back to manual instructions.
  cat <<EOF

[setup] GitHub CLI ('gh') isn't installed. Two choices:

  A) Install it (recommended, one time):
       sudo apt update && sudo apt install gh
       gh auth login
       # then re-run:  bash setup_github.sh $REPO_NAME

  B) Do it by hand:
       1. Create an EMPTY repo named "$REPO_NAME" at https://github.com/new
          (do NOT add a README/.gitignore — this folder already has them)
       2. Link and push:
            git remote add origin https://github.com/<your-username>/$REPO_NAME.git
            git push -u origin $BRANCH
          When it asks for a password, paste a GitHub Personal Access Token
          (github.com -> Settings -> Developer settings -> Fine-grained tokens,
           give it "Contents: Read and write" on this repo).

EOF
fi

echo
echo "Next: start the auto-saver so every future save pushes itself:"
echo "    bash autosave.sh"
