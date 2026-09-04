#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
# 32-node table held constant across k, so table size is not a confound
for m in anthropic/claude-opus-4.5 google/gemini-2.5-flash deepseek/deepseek-chat; do
  for t in parallel parallel_count serial; do
    echo "### $m | $t | nodes=32"
    python3 ../src/harness.py --task $t --conditions immediate --n 40 --nodes 32 \
      --ks 2,4,8,12,16,20,24 --model "$m" --out "W_${t}_$(basename $m)" 2>&1 | tail -10
  done
done
echo "### ALL DONE"
