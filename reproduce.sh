#!/usr/bin/env bash
# Reproduce the published findings.
#
#   ./reproduce.sh analysis     re-derive every number from the committed CSVs (no API, seconds)
#   ./reproduce.sh audit        validity-check every collected cell
#   ./reproduce.sh smoke        collect a small fresh sample (needs an API key, ~1 min)
#   ./reproduce.sh report       rebuild the HTML report from the archived versions
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-analysis}" in
  analysis)
    echo "Re-deriving published numbers from data/ ..."
    python3 src/analysis.py
    ;;
  audit)
    echo "Validity checks. A FAIL means that cell is not a usable measurement."
    python3 src/audit.py audit
    ;;
  smoke)
    : "${OPENROUTER_API_KEY:?set it, or put the key in ~/.openrouter_key and export it}"
    echo "Collecting a small fresh sample on the real-world fact task ..."
    python3 src/harness.py --task facts --model qwen/qwen3.6-27b \
      --conditions immediate,cot --n 20 --ks 2,3,4 --out smoke_check
    echo "Compare against the published curve: 1.000 / 0.383 / 0.017 at n=120."
    ;;
  report)
    python3 report/build_report.py
    echo "Open report/report.html"
    ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
