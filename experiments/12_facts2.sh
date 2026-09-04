#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
M=qwen/qwen3.6-27b
# CoT is a validity baseline (does the task have a solvable ceiling?) so n=40 is
# plenty; the measured conditions carry n=120 where the power actually matters.
echo "### serial immediate k=1..6"
python3 ../src/harness.py --task facts --model "$M" --conditions immediate --n 120 \
  --ks 1,2,3,4,5,6 --out F2_serial 2>&1 | grep -E "^ *immediate|FAIL"
echo "### serial cot (validity)"
python3 ../src/harness.py --task facts --model "$M" --conditions cot --n 40 \
  --ks 1,2,3,4,5,6 --out F2_serial_cot 2>&1 | grep -E "^ *cot|FAIL"
declare -A COT=( [2]=722 [3]=454 [4]=405 [5]=1032 [6]=1345 )
for k in 2 3 4 5 6; do
  echo "### filler k=$k n=${COT[$k]}"
  python3 ../src/harness.py --task facts --model "$M" --conditions filler --n 120 --ks $k \
    --filler-n ${COT[$k]} --out "F2_fill_k$k" 2>&1 | grep -E "^ *filler|FAIL"
done
echo "### parallel immediate k=2..16"
python3 ../src/harness.py --task facts_parallel --model "$M" --conditions immediate --n 120 \
  --ks 2,3,4,6,8,12,16 --out F2_parallel 2>&1 | grep -E "^ *immediate|FAIL"
echo "### parallel cot (validity)"
python3 ../src/harness.py --task facts_parallel --model "$M" --conditions cot --n 40 \
  --ks 2,4,8,16 --out F2_parallel_cot 2>&1 | grep -E "^ *cot|FAIL"
echo "### ALL DONE"
