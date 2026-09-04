#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
# Few-shot count as a design factor, not a constant.
# Seeds are now crc32-derived, so the SAME problems appear at 1-shot and 3-shot
# and in both conditions -- every comparison here is genuinely paired.
for m in google/gemma-3-27b-it qwen/qwen3.6-27b; do
  for fs in 1 3; do
    echo "### $m fewshot=$fs"
    python3 pilot2.py --task serial --model "$m" --conditions immediate,filler \
      --n 300 --ks 2,3,4,6 --fewshot $fs --filler-n 100 \
      --out "G_fs${fs}_$(basename $m)" 2>&1 | grep -E "^ *(immediate|filler)|audit|FAIL"
  done
done
echo "### ALL DONE"
