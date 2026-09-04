#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
# Real-world multi-hop facts, with the filler intervention. Both conditions in one
# invocation so trials are paired on the same generated questions.
for m in qwen/qwen3.6-27b google/gemma-3-27b-it; do
  echo "### $m"
  python3 pilot2.py --task facts --model "$m" --conditions immediate,filler,cot \
    --n 150 --ks 1,2,3,4,5 --filler-n 100 --out "FACTS_$(basename $m)" 2>&1 | tail -22
done
echo "### ALL DONE"
