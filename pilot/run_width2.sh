#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.runpod_key >/dev/null 2>&1; cat ~/.openrouter_key)
# 80-node table held constant across k so table size is not a confound; k up to 64
for m in qwen/qwen3.6-27b qwen/qwen3.6-35b-a3b google/gemma-3-27b-it; do
  for t in parallel parallel_count; do
    echo "### $m | $t | nodes=80"
    python3 pilot2.py --task $t --conditions immediate --n 30 --nodes 80 \
      --ks 8,16,24,32,48,64 --model "$m" --out "W2_${t}_$(basename $m)" 2>&1 | tail -9
  done
done
echo "### ALL DONE"
