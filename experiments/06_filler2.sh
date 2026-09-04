#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
# Filler budget matched to how many tokens the model actually spends on CoT for the
# same k (median from cotlen.csv), plus the old 100-token level for comparison.
# k -> median CoT tokens
declare -A COT=( [2]=1344 [3]=1650 [4]=1894 [6]=2422 )
M=qwen/qwen3.6-27b
for k in 2 3 4 6; do
  c=${COT[$k]}
  for f in 100 $c $((c*2)); do
    echo "### k=$k filler=$f"
    python3 ../src/harness.py --task arith --model "$M" --conditions filler --n 30 \
      --ks $k --filler-n $f --out "F_k${k}_f${f}" 2>&1 | grep -E "^ *filler|audit|FAIL"
  done
  echo "### k=$k immediate baseline"
  python3 ../src/harness.py --task arith --model "$M" --conditions immediate --n 30 \
    --ks $k --out "F_k${k}_f0" 2>&1 | grep -E "^ *immediate|audit|FAIL"
done
echo "### ALL DONE"
