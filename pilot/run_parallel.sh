#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
M=qwen/qwen3.6-27b
echo "### parallel immediate k=2..16"
python3 pilot2.py --task facts_parallel --model "$M" --conditions immediate --n 120 \
  --ks 2,3,4,6,8,12,16 --out F2_parallel 2>&1 | grep -E "run_id|^ *immediate|FAIL"
echo "### parallel cot (validity)"
python3 pilot2.py --task facts_parallel --model "$M" --conditions cot --n 40 \
  --ks 2,4,8,16 --out F2_parallel_cot 2>&1 | grep -E "run_id|^ *cot|FAIL"
echo "### DONE"
