#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
# 2x2: few-shot count x filler, at k=2 where the filler effect is largest.
# n=150 gives ~80% power at alpha=0.03 for the +0.15 effect seen at k=2.
M=google/gemma-3-27b-it
for fs in 1 3; do
  for cond in immediate filler; do
    echo "### fewshot=$fs cond=$cond"
    python3 pilot2.py --task serial --model "$M" --conditions $cond --n 150 --ks 2 \
      --fewshot $fs --filler-n 100 --out "FS_${fs}_${cond}" 2>&1 | grep -E "^ *(immediate|filler)|audit|FAIL"
  done
done
echo "### ALL DONE"
