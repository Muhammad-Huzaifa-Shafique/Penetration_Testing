#!/usr/bin/env bash
#
# collect_corpus.sh — Casefile capstone, Week 2
# Builds the raw knowledge corpus for the local RAG pentest companion.

set -uo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
CORPUS="$PROJECT_ROOT/corpus"
HT_DIR="$CORPUS/hacktricks"
MAN_DIR="$CORPUS/manpages"
MANIFEST="$CORPUS/MANIFEST.txt"

mkdir -p "$MAN_DIR"
echo "==> Corpus target: $CORPUS"

clean() {
  if command -v col >/dev/null 2>&1; then
    col -bx
  else
    sed -e 's/\x1b\[[0-9;]*m//g' -e 's/.\x08//g' -e 's/.\x08//g'
  fi
}

HT_URL="https://github.com/HackTricks-wiki/hacktricks.git"
if [ -d "$HT_DIR/.git" ]; then
  echo "==> HackTricks already cloned — pulling latest"
  git -C "$HT_DIR" pull --ff-only || echo "    (pull skipped; keeping existing copy)"
else
  echo "==> Cloning HackTricks (shallow, this is the big step)"
  git clone --depth 1 "$HT_URL" "$HT_DIR"
fi

TOOLS=(
  nmap masscan arp-scan netdiscover dig host whois nc ncat enum4linux
  gobuster ffuf feroxbuster nikto sqlmap wpscan whatweb dirb curl wget
  smbclient smbmap rpcclient snmpwalk ldapsearch netexec crackmapexec
  hydra medusa john hashcat crunch cewl hashid
  searchsploit evil-winrm socat
  sudo find chmod getcap ss
)

echo "==> Collecting man pages for ${#TOOLS[@]} tools"
collected=(); helponly=(); missing=()

for t in "${TOOLS[@]}"; do
  if man -w "$t" >/dev/null 2>&1; then
    MANWIDTH=80 MANPAGER=cat man "$t" 2>/dev/null | clean > "$MAN_DIR/$t.man.txt"
    if [ -s "$MAN_DIR/$t.man.txt" ]; then collected+=("$t")
    else rm -f "$MAN_DIR/$t.man.txt"; missing+=("$t"); fi
  elif command -v "$t" >/dev/null 2>&1; then
    { "$t" --help 2>&1 || true; } | clean > "$MAN_DIR/$t.help.txt"
    if [ -s "$MAN_DIR/$t.help.txt" ]; then helponly+=("$t")
    else rm -f "$MAN_DIR/$t.help.txt"; missing+=("$t"); fi
  else
    missing+=("$t")
  fi
done

{
  echo "Casefile corpus manifest — generated $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo
  echo "HackTricks:"
  if [ -d "$HT_DIR" ]; then
    echo "  markdown files: $(find "$HT_DIR" -name '*.md' | wc -l)"
    echo "  disk size:      $(du -sh "$HT_DIR" 2>/dev/null | cut -f1)"
  fi
  echo
  echo "Man pages collected (${#collected[@]}): ${collected[*]:-none}"
  echo "Help-only, no man (${#helponly[@]}): ${helponly[*]:-none}"
  echo "Missing / not installed (${#missing[@]}): ${missing[*]:-none}"
} | tee "$MANIFEST"

echo
echo "==> Done. Corpus inventory:"
echo "    HackTricks .md files : $(find "$HT_DIR" -name '*.md' 2>/dev/null | wc -l)"
echo "    man/help text files  : $(find "$MAN_DIR" -type f | wc -l)"
echo "    manifest             : $MANIFEST"
