#!/usr/bin/env bash
# Casefile Week 5 screen-recording runner. Run it from the project folder: bash demo_week5.sh
# Press Enter to move on. Everything on screen is live output or your own files,
# except the results table on screen 4, which is typed in (counted by eye).
cd "$(dirname "$0")" || exit 1
source ~/.venv/bin/activate || { echo "Could not activate ~/.venv"; exit 1; }

CARD=/tmp/nomic_card.md
LIST=/tmp/casefile_list.txt
Q_SUID="How do I escalate privileges on Linux when I find a SUID binary?"
Q_KERB="Kerberoasting SPN service ticket crack offline"

title() { clear; printf '\n   %s\n   %s\n\n' "$1" "${1//?/-}"; }
next() { read -r -p $'\n   [Enter = next] ' _; }
show() { fold -s -w 96 | sed 's/^/   /'; }

echo "Warming up: loading the models once and saving the model card..."
python retrieve.py --list > "$LIST" || exit 1
python retrieve.py -e nomic-embed-text -e all-minilm -k 1 "warm up" > /dev/null || exit 1
rm -f "$CARD"
curl -fsSL --max-time 20 -o "$CARD" \
  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/raw/main/README.md ||
  echo "(Could not download the model card: show it in the browser during screen 2.)"
read -r -p $'\nReady. Start your screen recorder, then press Enter. ' _

title "1/5  What happened: one question, two embedding models"
python retrieve.py -e nomic-embed-text -e all-minilm "$Q_SUID"
next

title "2/5  What went wrong: my Week 4 build sent plain text"
grep -n -A14 "def embed_texts" build_index.py
printf '\n   mentions of search_document / search_query in build_index.py: '
grep -c -E "search_document|search_query" build_index.py
next
printf '\n   nomic-embed-text model card (Hugging Face):\n\n'
if grep -qi "task instruction prefix" "$CARD" 2>/dev/null; then
  grep -i -m1 "task instruction prefix" "$CARD" | show
  echo
  grep -o -i "you embed your documents as[^.]*\." "$CARD" | show
else
  echo "   (not downloaded: switch to the browser)"
fi
next

title "3/5  Why it seemed reasonable: plain nomic still found the right pages for jargon"
python retrieve.py -e nomic-embed-text -e all-minilm "$Q_KERB"
next

title "4/5  What I learned: a third index, only the prefix changed"
grep -E "^Collection|vectors|dimensions" "$LIST"
echo
grep -n "DOC_PREFIX" build_nomic_prefixed.py
next
cat <<'TABLE'

   Counted by eye from week5_compare_3way.txt (top 5 results of each index)

   question                                nomic    +prefix   all-minilm
   --------------------------------------  -------  --------  ----------
   SUID privesc, plain English: on topic   1 of 5   2 of 5    5 of 5
   nmap: where the -p- chunk lands         #5       #3        #1
   Kerberoast, no keyword: on topic        1 of 5   2 of 5    1 of 5
   Kerberoast, page's words: on topic      4 of 5   4 of 5    1 of 5
   ssh -L: where the -L chunk lands        #4       #2        #2
   SMB shares, plain English: on topic     ~3 of 5  2-3 of 5  5 of 5
   Sourdough (off-topic): best score       0.50     0.58      0.29
TABLE
next

title "5/5  What's next"
cat <<'NEXT'
   Week 6: question -> retrieve() -> prompt with the chunks -> llama3.1:8b -> English answer
   Then: make the prefixed index the default, and Week 7 (sources under every answer)
NEXT
